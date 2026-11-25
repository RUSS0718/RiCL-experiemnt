"""Issue #20 contracts for paired MS3 repairs and offline analysis."""

from __future__ import annotations

import json
import sqlite3

import httpx
import pytest

from ricl_experiment.analysis.ms3_repair import analyze_repairs
from ricl_experiment.cli import main
from ricl_experiment.config import RunConfig
from ricl_experiment.motivation.ms3_repair import (
    GROUPS,
    LOCATION_TEMPLATE,
    build_repair_request,
    choose_random_location,
    evaluate_repair_record,
    run_ms3_repair,
    validate_repair_trajectory,
)


def test_random_location_is_stable_uniform_candidate_and_excludes_first_error():
    first = choose_random_location(seed=23, trajectory_id="trace-1", step_count=4, first_error=1)
    second = choose_random_location(seed=23, trajectory_id="trace-1", step_count=4, first_error=1)

    assert first == second
    assert first in {0, 2, 3}
    counts = {index: 0 for index in (0, 2, 3)}
    for trajectory_index in range(3000):
        location = choose_random_location(23, f"trace-{trajectory_index}", 4, 1)
        counts[location] += 1
    assert all(950 <= count <= 1050 for count in counts.values())
    assert LOCATION_TEMPLATE.format(step_index=first) in build_repair_request(
        "Find x.", ["step zero", "step one", "step two", "step three"], "G2", first
    )["messages"][0]["content"]


def test_repair_requests_share_original_steps_parameters_and_hide_labels():
    extras = {
        "problem": "Solve x+1=2.",
        "steps": ["  Original step one  ", "Original step two"],
        "instruction": "Check the reasoning and provide a complete repaired solution.",
        "expert_label": 0,
        "reference_answer": "secret answer",
        "group": "G3",
        "position_source": "oracle",
        "other_results": ["hidden candidate"],
        "sample_id": "internal-sample",
    }
    requests = {
        group: build_repair_request(
            extras["problem"], extras["steps"], group, _fields(group).get("location_index"),
            {"max_tokens": 24, "temperature": 0}, "backbone",
        )
        for group in GROUPS
    }
    prompts = [requests[group]["messages"][0]["content"] for group in GROUPS]
    assert all("  Original step one  " in prompt for prompt in prompts)
    assert all("secret answer" not in prompt and "hidden candidate" not in prompt for prompt in prompts)
    assert all("expert_label" not in prompt and "position_source" not in prompt for prompt in prompts)
    assert all(requests[group]["max_tokens"] == 24 for group in GROUPS)
    assert all(requests[group]["tools"] == [] for group in GROUPS)
    assert prompts[2].replace("第 2 步", "第 0 步") == prompts[3]


def test_each_group_uses_only_its_instruction_and_approved_location():
    g0 = build_repair_request("P", ["a", "b"], "G0", None)
    g1 = build_repair_request("P", ["a", "b"], "G1", None)
    g2 = build_repair_request("P", ["a", "b"], "G2", 1)
    g3 = build_repair_request("P", ["a", "b"], "G3", 0)
    p0, p1, p2, p3 = [row["messages"][0]["content"] for row in (g0, g1, g2, g3)]

    assert "根据原题重新求解" in p0 and "检查原轨迹" not in p0
    assert "检查原轨迹" in p1 and "请优先复核" not in p1
    assert p2.replace("第 1 步", "第 0 步") == p3
    assert p0.count("Trace steps:") == p1.count("Trace steps:") == 1


def test_single_step_error_is_not_repair_eligible():
    eligible, reason = validate_repair_trajectory(["only step"], first_error=0)
    assert not eligible
    assert reason == "insufficient_steps"


def test_evaluation_keeps_success_error_removal_and_new_error_independent():
    record = evaluate_repair_record(
        evaluation_id="blind-1",
        repair_success="incorrect",
        error_removal="removed",
        new_error="present",
        evidence={"repair_success": "wrong final result", "error_removal": "valid replacement"},
    )

    assert record["repair_success"] == "incorrect"
    assert record["error_removal"] == "removed"
    assert record["new_error"] == "present"
    assert record["evidence"]["error_removal"] == "valid replacement"
    unknown = evaluate_repair_record(
        evaluation_id="blind-2", repair_success="unknown", error_removal="unknown", new_error="unknown"
    )
    assert unknown["repair_success"] == unknown["error_removal"] == unknown["new_error"] == "unknown"


def test_run_dispatches_four_isolated_logical_calls_and_records_attempt_usage(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot.json"
    manifest = tmp_path / "manifest.json"
    audit = tmp_path / "audit.json"
    answer_key = tmp_path / "answer-key.json"
    snapshot.write_text(json.dumps({
        "source": "processbench", "revision": "rev1", "split": "test",
        "records": [{"id": "source-row-1", "trajectory_id": "trace-1", "question_group_id": "question-1", "problem": "Solve x+1=2.",
                     "steps": ["  first step  ", "second step"], "label": 0,
                     "final_answer_correct": False, "generator": "source-model",
                     "reference_material": {"kind": "answer", "content": "x=1", "source": "verified",
                                             "independently_verified": True}}],
    }), encoding="utf-8")
    manifest.write_text(json.dumps({"study": "ms3", "phase": "pilot", "sampling": {"seed": 17},
                                    "sources": [{"source": "processbench", "revision": "rev1", "split": "test"}],
                                    "samples": [{"sample_id": "sample-1", "problem": "Solve x+1=2.",
                                                 "source_record_id": "processbench:rev1:test:source-row-1",
                                                 "trajectory_source_id": "trace-1", "question_group_id": "question-1",
                                                 "sampling_stratum": "processbench-blinded"}]}), encoding="utf-8")
    audit.write_text(json.dumps({"study_pools": {"repair": ["sample-1"]}, "processbench_labels": [
        {"trajectory_id": "trajectory-1", "trajectory_source_id": "trace-1", "label": 0,
         "meaning": "first_error_step_index_zero_based"}]}), encoding="utf-8")
    answer_key.write_text(json.dumps([{"sample_id": "sample-1", "steps": ["  first step  ", "second step"],
        "expert_label": 0, "reference_material": {
        "kind": "answer", "content": "x=1", "source": "verified", "independently_verified": True}}]), encoding="utf-8")
    config = RunConfig.model_validate({
        "model": {"base_url": "http://localhost/v1", "model": "offline-test", "capabilities": {
            "supported_parameters": ["max_tokens", "temperature"], "required_parameters": ["max_tokens"]},
            "parameters": {"max_tokens": 24, "temperature": 0}},
        "budget": {"max_attempts_per_call": 2, "max_reserved_output_tokens": 128},
    })
    seen = []

    class ControlledClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def post(self, url, json, headers):
            seen.append(json)
            if len(seen) == 2:
                raise KeyboardInterrupt("simulated process interruption")
            request = httpx.Request("POST", url)
            return httpx.Response(200, request=request, json={
                "id": f"response-{len(seen)}", "choices": [{"message": {"role": "assistant", "content": "x=1"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15,
                          "completion_tokens_details": {"reasoning_tokens": 2}},
            })

    from ricl_experiment.core import model
    monkeypatch.setattr(model.httpx, "Client", ControlledClient)
    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        run_ms3_repair(config, snapshot, manifest, audit, answer_key, output, max_new_tokens=16)
    with pytest.raises(ValueError, match="seed changed"):
        run_ms3_repair(
            config, snapshot, manifest, audit, answer_key, output,
            seed=18, max_new_tokens=16, resume=True,
        )
    assert len(seen) == 2
    result = run_ms3_repair(
        config, snapshot, manifest, audit, answer_key, output, max_new_tokens=16, resume=True
    )

    assert len(seen) == 5
    assert [row["group"] for row in result["assignments"]] == list(GROUPS)
    assert len({json.dumps(body, sort_keys=True) for body in seen}) == 4
    assert all("source-model" not in body["messages"][0]["content"] for body in seen)
    assert all("x=1" not in body["messages"][0]["content"] for body in seen)
    assert all(body["max_tokens"] == 16 and body["tools"] == [] for body in seen)
    request_audit = json.loads((output / "restricted" / "requests.json").read_text(encoding="utf-8"))
    assert len(request_audit) == 4
    assert all("group" in row and "group" not in row["request"] for row in request_audit)
    blind = [json.loads(line) for line in (output / "blind" / "evaluations.jsonl").read_text().splitlines()]
    assert len(blind) == 4
    assert all(row["repair_success"] == "missing" for row in blind)
    with sqlite3.connect(output / "run.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM logical_calls").fetchone()[0] == 4
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 5
        assert db.execute(
            "SELECT DISTINCT input_tokens, output_tokens, reasoning_tokens FROM attempts ORDER BY input_tokens"
        ).fetchall() == [(None, None, None), (11, 4, 2)]
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"

    for row, status in zip(blind, ("incorrect", "correct", "correct", "correct"), strict=True):
        row["repair_success"] = status
        row["error_removal"] = "removed"
        row["new_error"] = "absent"
    (output / "blind" / "evaluations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in blind), encoding="utf-8"
    )
    assert main(["analyze-ms3-repair", "--run-dir", str(output), "--bootstrap-samples", "20"]) == 0
    report = json.loads((output / "metrics" / "ms3-repair.json").read_text(encoding="utf-8"))
    assert report["comparisons"]["G3-G1"]["n_pairs"] == 1


def test_analysis_matches_frozen_paired_effect_example_and_reports_position_layers():
    # Frozen #10 example: marginal pairs are 4, 3, 3; the common four-way subset is 2.
    rows = []
    outcomes = [["incorrect", "incorrect", "incorrect", "correct"],
                ["incorrect", "correct", "correct", "correct"],
                ["correct", "correct", "unknown", "incorrect"],
                ["unknown", "incorrect", "incorrect", "correct"]]
    for index, groups in enumerate(outcomes):
        for group, status in zip(GROUPS, groups, strict=True):
            rows.append({"evaluation_id": f"e-{index}-{group}", "repair_success": status,
                         "error_removal": "removed" if status == "correct" else "not_removed",
                         "new_error": "absent", "missing_reason": "api_failure" if status == "unknown" else None})
    assignments = [{"evaluation_id": f"e-{index}-{group}", "sample_id": f"t-{index}",
                    "question_group_id": f"q-{index // 2}", "group": group,
                    "first_error_position": [2, 2, 2, 1][index], "g2_position": [1, 3, 3, 2][index],
                    "trajectory_length": 5 if index != 3 else 3, "stratum": "source-a"}
                   for index in range(4) for group in GROUPS]

    report = analyze_repairs(rows, assignments, bootstrap_samples=200, seed=9)

    effects = report["comparisons"]
    assert effects["G3-G1"]["n_pairs"] == 4
    assert effects["G3-G1"]["mean_difference"] == 0.25
    assert effects["G3-G2"]["n_pairs"] == 3
    assert effects["G3-G2"]["mean_difference"] == 2 / 3
    assert effects["G1-G0"]["n_pairs"] == 3
    assert effects["G1-G0"]["mean_difference"] == 1 / 3
    assert all(
        report["four_group_common"]["comparisons"][name]["mean_difference"] == 0.5
        for name in ("G3-G1", "G3-G2", "G1-G0")
    )
    assert report["position_offset"]["layers"]["earlier"]["position_pairs"] == 1
    assert report["position_offset"]["layers"]["earlier"]["evaluable_pairs"] == 1
    assert report["position_offset"]["layers"]["later"]["position_pairs"] == 3
    assert report["position_offset"]["layers"]["later"]["evaluable_pairs"] == 2
    assert "same" in report["position_offset"]["empty_layers"]
    assert report["position_offset"]["direction"] == {"negative": 1, "zero": 0, "positive": 3}


def _fields(group: str) -> dict[str, int]:
    if group == "G2":
        return {"location_index": 2}
    if group == "G3":
        return {"location_index": 0}
    return {}
