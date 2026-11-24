"""Public-boundary acceptance tests for MS2 scoring and blind math review."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest

from ricl_experiment.config import load_config
from ricl_experiment.core.ledger import Ledger
from ricl_experiment.evaluation.evidence import (
    build_blind_review_package,
    deduplicate_claims,
    export_blind_review,
    import_blind_reviews,
    recompute_blind_review,
    score_verified_progress,
    summarize_costs,
)
from ricl_experiment.motivation.ms2_scoring import score_ms2_checkpoint


def audit_claim(
    claim_id: str,
    *,
    source: str = "Derivation-Checked",
    verdict: str = "verified",
    conditions_sufficient: bool | None = True,
    conditions: list[str] | None = None,
    equivalence_key: str | None = None,
    milestones: list[int] | None = None,
) -> dict:
    return {
        "claim_id": claim_id,
        "sample_id": "sample-1",
        "route_id": "route-1",
        "problem": "Find x.",
        "checkpoint": "Assume x is real.",
        "assertion": f"claim {claim_id}",
        "conditions": conditions or ["x is real"],
        "target_relation": "supports x = 2",
        "source_locator": "checkpoint:1-2",
        "equivalence_key": equivalence_key,
        "milestones": [1] if milestones is None else milestones,
        "evidence": [
            {
                "provenance": source,
                "verdict": verdict,
                "source_locator": "checkpoint:2",
                "conditions": conditions or ["x is real"],
                "conditions_sufficient": conditions_sufficient,
                "supporting_steps": ["From x=1+1, simplify to x=2."],
                "rationale": "The cited identity follows under the stated condition.",
                "audit_ref": f"audit:{claim_id}",
            }
        ],
    }


def test_claim_ledger_merges_equivalent_assertions_only_under_same_conditions():
    left = audit_claim("c1", equivalence_key="x=2", conditions=["x real"])
    right = audit_claim("c2", equivalence_key="x=2", conditions=[" x   REAL "])
    different_condition = audit_claim("c3", equivalence_key="x=2", conditions=["x positive"])

    claims = deduplicate_claims([left, right, different_condition])

    assert len(claims) == 2
    assert claims[0].claim_id == "c1"
    assert claims[0].claim_variants == ("claim c2",)
    assert len(claims[0].evidence) == 2


def test_tool_result_without_sufficient_applicability_is_unknown():
    claim = audit_claim(
        "c1", source="Tool-Verified", verdict="verified", conditions_sufficient=False
    )

    score = score_verified_progress([claim])

    assert score["score"] == 0
    assert score["coverage"] == 0
    assert score["coverage_status"] == "unverified"
    assert score["unknown"] == 1
    assert score["claims"][0]["verdict"] == "unknown"


def test_progress_coverage_keeps_verified_falsified_unknown_zero_na_and_failure_distinct():
    rows = [
        audit_claim("tool", source="Tool-Verified", milestones=[1]),
        audit_claim("derivation", source="Derivation-Checked", verdict="falsified", milestones=[]),
        audit_claim("agent", source="Agent-Assessed", milestones=[2]),
        audit_claim("unknown", source="Unknown", verdict="unknown"),
    ]

    result = score_verified_progress(rows)
    unknown_only = score_verified_progress([audit_claim("u", source="Unknown", verdict="unknown")])
    no_eligible = score_verified_progress(
        [{**audit_claim("boilerplate"), "nontrivial": False}]
    )
    failed = score_verified_progress(rows, execution_failure="extractor timed out")

    assert result["score"] == 2
    assert result["coverage"] == pytest.approx(0.75)
    assert result["coverage_denominator"] == 4
    assert result["verified"] == 2
    assert result["falsified"] == 1
    assert result["unknown"] == 1
    assert result["evidence_by_source"]["Tool-Verified"]["verified"] == 1
    assert result["evidence_by_source"]["Derivation-Checked"]["falsified"] == 1
    assert result["evidence_by_source"]["Agent-Assessed"]["verified"] == 1
    assert result["evidence_by_source"]["Unknown"]["unknown"] == 1
    assert unknown_only["score"] == 0 and unknown_only["coverage"] == 0
    assert unknown_only["unverified"] is True
    assert no_eligible["score"] == 0 and no_eligible["coverage"] is None
    assert no_eligible["coverage_status"] == "not_applicable"
    assert failed["score"] is None and failed["coverage_status"] == "missing"
    assert failed["failure_reason"] == "extractor timed out"


def test_legal_verified_zero_is_not_conflated_with_missing_or_unknown():
    result = score_verified_progress(
        [audit_claim("confirmed-but-no-milestone", milestones=[])]
    )

    assert result["score"] == 0
    assert result["coverage"] == 1
    assert result["coverage_status"] == "observed"
    assert result["unknown"] == 0


def test_costs_preserve_usage_completeness_and_aggregate_by_route():
    result = summarize_costs(
        [
            {
                "call_id": "known",
                "sample_id": "s1",
                "route_id": "r1",
                "usage": {"status": "known", "input_tokens": 1000, "output_tokens": 500},
            },
            {
                "call_id": "unknown",
                "sample_id": "s1",
                "route_id": "r2",
                "usage": {"status": "unknown", "input_tokens": None, "output_tokens": None},
            },
        ],
        input_cost_per_1k=1,
        output_cost_per_1k=2,
    )

    assert result["overall"]["calls"] == 2
    assert result["overall"]["cost_status"] == "partial"
    assert result["overall"]["cost"] == 2.0
    assert result["overall"]["unknown_usage_calls"] == 1
    assert result["by_route"]["r1"]["tokens"] == {
        "input": 1000,
        "output": 500,
        "reasoning": 0,
    }
    assert result["by_route"]["r2"]["cost_status"] == "unknown"


def test_blind_review_export_is_seeded_stratified_and_contains_no_original_labels(tmp_path):
    claims = [
        audit_claim(f"c{i}", source=source, verdict=verdict)
        for i, (source, verdict) in enumerate(
            [
                ("Tool-Verified", "verified"),
                ("Tool-Verified", "falsified"),
                ("Derivation-Checked", "verified"),
                ("Derivation-Checked", "unknown"),
                ("Agent-Assessed", "verified"),
                ("Unknown", "unknown"),
            ]
            * 5
        )
    ]
    public_path = tmp_path / "review.json"
    audit_path = tmp_path / "restricted" / "audit.json"

    public = export_blind_review(
        claims, public_path, seed=712, sample_size=24, audit_destination=audit_path
    )
    public_again, audit_again = build_blind_review_package(claims, seed=712, sample_size=24)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert public == public_again
    assert audit == audit_again
    assert public["selected_count"] == 24
    assert len({task["review_id"] for task in public["tasks"]}) == 24
    for task in public["tasks"]:
        assert not {"sample_id", "route_id", "claim_id", "provenance", "verdict", "score"} & task.keys()
    serialized_public = public_path.read_text(encoding="utf-8")
    assert "Tool-Verified" not in serialized_public
    assert "falsified" not in serialized_public
    assert "claim_map" not in serialized_public
    assert set(audit["claim_map"]) == {task["review_id"] for task in public["tasks"]}


def test_review_import_and_recomputation_report_denominator_disagreement_unknown_by_source():
    claims = [
        audit_claim("tool-ok", source="Tool-Verified", verdict="verified"),
        audit_claim("tool-bad", source="Tool-Verified", verdict="falsified"),
        audit_claim("derivation-u", source="Derivation-Checked", verdict="unknown"),
        audit_claim("agent-ok", source="Agent-Assessed", verdict="verified"),
    ]
    public, audit = build_blind_review_package(claims, seed=9, sample_size=4)
    ids = {audit["claim_map"][review_id]: review_id for review_id in audit["claim_map"]}
    reviews = [
        {"review_id": ids["tool-ok"], "verdict": "verified", "rationale": "valid"},
        {"review_id": ids["tool-bad"], "verdict": "verified", "rationale": "disagree"},
        {"review_id": ids["derivation-u"], "verdict": "unknown", "rationale": "insufficient"},
        {"review_id": ids["agent-ok"], "verdict": "unknown", "rationale": "unresolved"},
    ]

    imported = import_blind_reviews({"reviews": reviews}, public_export=public)
    report = recompute_blind_review(claims, audit, imported)

    assert report["overall"]["sampled"] == 4
    assert report["overall"]["adjudicable"] == 2
    assert report["overall"]["agreement"] == 1
    assert report["overall"]["disagreement"] == 1
    assert report["overall"]["accuracy"] == 0.5
    assert report["overall"]["original_unknown"] == 1
    assert report["overall"]["manual_unresolved"] == 2
    assert report["by_provenance"]["Tool-Verified"]["adjudicable"] == 2
    assert report["by_provenance"]["Tool-Verified"]["disagreement"] == 1
    assert report["by_provenance"]["Derivation-Checked"]["original_unknown"] == 1
    assert report["by_provenance"]["Agent-Assessed"]["manual_unresolved"] == 1


def test_review_import_rejects_unknown_or_duplicate_review_ids():
    with pytest.raises(ValueError, match="not present"):
        import_blind_reviews(
            [{"review_id": "not-exported", "verdict": "verified"}],
            public_export={"tasks": [{"review_id": "exported"}]},
        )
    with pytest.raises(ValueError, match="unique"):
        import_blind_reviews(
            [
                {"review_id": "r1", "verdict": "verified"},
                {"review_id": "r1", "verdict": "verified"},
            ]
        )


class ScoringControlledEndpoint(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict]] = []
    lock = threading.Lock()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        content = body["messages"][0]["content"]
        if "Rater instructions" in content:
            response = json.dumps({"score": 2, "rationale": "The route has a usable subgoal."})
        else:
            response = json.dumps(
                {
                    "claims": [
                        {
                            "assertion": "x=2",
                            "conditions": ["x is real"],
                            "source_locator": "line 1",
                            "target_relation": "solves the subgoal",
                            "supporting_steps": ["simplify the equation"],
                        }
                    ],
                    "non_claims": [],
                }
            )
        with self.lock:
            self.requests.append(body)
        payload = {
            "id": "controlled-scoring",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        return


@pytest.fixture
def scoring_endpoint():
    ScoringControlledEndpoint.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), ScoringControlledEndpoint)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_scoring_calls_serialize_only_problem_and_raw_checkpoint_and_persist_before_results(
    tmp_path, scoring_endpoint
):
    config_path = tmp_path / "model.toml"
    config_path.write_text(
        "\n".join(
            [
                "[model]",
                f'base_url = "{scoring_endpoint}"',
                'model = "controlled-model"',
                "",
                "[model.capabilities]",
                'supported_parameters = ["max_tokens", "temperature"]',
                'required_parameters = ["max_tokens"]',
                "",
                "[model.parameters]",
                "max_tokens = 64",
                "temperature = 0",
                "",
                "[budget]",
                "max_attempts_per_call = 1",
                "max_reserved_output_tokens = 256",
                "",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ledger = Ledger(run_dir / "run.sqlite3", budget_tokens=256)
    hidden = [
        "RESTRICTED_REFERENCE_42",
        "SIBLING_PREFIX_SECRET",
        "SIBLING_ROLLOUT_SECRET",
        "INDEPENDENT_ANSWER_JUDGMENT_SECRET",
        "PRIOR_ROUTE_SCORE_SECRET",
    ]

    try:
        result = score_ms2_checkpoint(
            config,
            ledger,
            run_dir,
            run_id="controlled-run",
            sample_id="private-sample-id",
            route_id="private-route-id",
            problem="Find x satisfying x + 1 = 3.",
            checkpoint="Let x=2, so the equation is satisfied.",
        )
    finally:
        ledger.close()

    serialized = json.dumps(ScoringControlledEndpoint.requests)
    assert len(ScoringControlledEndpoint.requests) == 2
    assert all(hidden_value not in serialized for hidden_value in hidden)
    assert result["llm_rater"]["score"] == 2
    assert result["llm_rater"]["parse_status"] == "valid"
    assert result["claim_extraction"]["parse_status"] == "valid"
    assert result["claim_extraction"]["claims"][0]["assertion"] == "x=2"
    assert result["associated_outcomes"] == []
    for body in ScoringControlledEndpoint.requests:
        content = body["messages"][0]["content"]
        assert "Find x satisfying x + 1 = 3." in content
        assert "Let x=2, so the equation is satisfied." in content
        assert "RESTRICTED_REFERENCE_42" not in content
    assert ScoringControlledEndpoint.requests[0] != ScoringControlledEndpoint.requests[1]
