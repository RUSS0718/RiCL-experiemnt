# MS1 dependency and paired re-derivation artifacts

Issue #25 adds an offline analysis slice over persisted MS2 trace-shaped JSON. It does not fetch data, call a model, or claim that a human or API review occurred. The repository tests use controlled fixtures only.

## Sampling input

`ms1-plan` consumes one or more `traces/ms2.json`-shaped objects plus a separate independent-outcome sidecar. The sidecar is required so the MS2 runner's answer-key heuristic is not silently reused as the independent final judgment:

```json
[
  {
    "path_id": "run:sample:candidate:rollout:0",
    "status": "incorrect",
    "reference": "independent-ledger:case-1"
  }
]
```

A path is one completed Checkpoint candidate plus one completed independent rollout. Paths with missing text or segment boundaries, missing independent outcomes or review references, `unknown`, `no_evaluable_output`, runtime failure, missing question-group IDs, or early-correct Checkpoints are excluded and reported. Sampling is fixed-seed, round-robin across `source|status` strata, and reserves at most one path per `question_group_id` across both failure and correct-reference pools. The plan retains candidate counts, group counts, selected counts, deficits, excluded records, and group conflicts.

```powershell
uv run python -m ricl_experiment ms1-plan `
  --trace artifacts/motivation/<run>/traces/ms2.json `
  --outcomes restricted/ms1-independent-outcomes.json `
  --seed 20261125 `
  --output artifacts/ms1/plan.json
```

The default targets are 20 `incorrect` paths and 10 `correct` reference paths. Controlled tests override them with smaller quotas.

## Blind annotation and reconciliation

`ms1-blind` removes outcome and answer-key fields before annotation. Each selected path contains the problem, preserved Checkpoint/rollout text and segment bounds, but not `sampling_outcome`, `reference_answer`, `expert_label`, or the runner's `evaluation` field.

Each path must import exactly two independent records. Nodes contain `step_id`, half-open text span, claim, applicability, `dependency_ids`, optional per-edge `dependency_status`, `validity_status` (`Valid`, `Invalid`, `Unsupported`, or `Unknown`), verification evidence, optional token count, an equivalence-group ID, and optional `is_nontrivial` (default `true`). A disagreement is sent through the optional third-reviewer record. Without that record, the imported consensus remains `Unknown`/`pending_third_review`; it is not silently resolved.

```powershell
uv run python -m ricl_experiment ms1-blind `
  --plan artifacts/ms1/plan.json `
  --output artifacts/ms1/blind.json

uv run python -m ricl_experiment ms1-import-annotations `
  --plan artifacts/ms1/plan.json `
  --independent restricted/annotations-reviewer-a.json `
  --independent restricted/annotations-reviewer-b.json `
  --third-review restricted/annotations-adjudicator.json `
  --output artifacts/ms1/annotations.json
```

The import is validation/reconciliation only. It does not synthesize mathematical evidence or replace an absent review.

## Metrics

`ms1-analyze` reads the reconciled annotations and an optional offline re-derivation output file. It emits per-path metrics, aggregate numerators/denominators, and missingness:

- **Error Propagation Ratio**: confirmed reachable nodes after the confirmed first error divided by all nodes after that first error. Unknown dependency edges remain in the denominator and are counted separately.
- **Dependency Depth**: shortest confirmed dependency-edge distance from the first error to each confirmed affected later node.
- **Reusable Claim Ratio**: unique non-trivial equivalence groups with at least one sufficiently evidenced `Valid` derivation that does not depend on a confirmed invalid node, divided by all unique non-trivial equivalence groups. `Unsupported` and `Unknown` groups remain in the denominator.
- **Re-derived Claim Tokens**: union length of supplied half-open token spans in the paired re-derivation output, divided by visible output tokens. If exact token spans are absent, character spans are mapped with a deterministic regex tokenizer and `token_count_estimated` is `true`. Hidden/reasoning usage is reported separately and never added to visible-output denominators.

A re-derivation record is an offline import, for example:

```json
{
  "path_id": "run:sample:candidate:rollout:0",
  "status": "completed",
  "token_spans": [[4, 8], [6, 10]],
  "visible_output_tokens": 20,
  "hidden_usage_tokens": 7
}
```

The overlapping spans above count as six visible repeated tokens, not eight. Missing re-derivation output is retained as a missing path rather than imputed.

```powershell
uv run python -m ricl_experiment ms1-analyze `
  --annotations artifacts/ms1/annotations.json `
  --rederivations restricted/ms1-rederivations.jsonl `
  --output artifacts/ms1/metrics.json
```

## Case export

`ms1-export-cases` writes exactly two or three selected paths from the analysis artifact. It is a presentation/export helper only; it does not label controlled fixtures as real cases.

```powershell
uv run python -m ricl_experiment ms1-export-cases `
  --analysis artifacts/ms1/metrics.json `
  --path-id run:sample-a:candidate:rollout:0 `
  --path-id run:sample-b:candidate:rollout:1 `
  --output artifacts/ms1/cases.json
```