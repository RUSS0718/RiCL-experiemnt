# Architecture and workflow

This document preserves the agreed bootstrap choices and target contracts. Issue #14 implements the first vertical slice: a configurable, single-sample `run`/`resume` CLI with HTTPX dispatch and SQLite auditing. Issue #15 adds local fixed-snapshot preparation and independent answer/ProcessBench audit artifacts. Issue #16 adds the MS2 control path with three candidate slots, immutable visible Checkpoints, two role-isolated continuations per non-empty Checkpoint, and offline answer-judgment references. Issue #19 adds paired MS3 diagnosis and fixed-denominator offline analysis; Issue #20 adds the separate four-group repair and paired-analysis path. Evidence scoring and primary MS2 analysis remain later slices.

## Selected stack

| Area | Choice | Purpose |
| --- | --- | --- |
| Language/runtime | Python 3.12 | Experiment orchestration, model calls, mathematical checks |
| Package/environment | uv, `pyproject.toml`, committed `uv.lock` | Reproducible setup and dependency resolution |
| Model transport | HTTPX, OpenAI-compatible HTTP API | Transparent requests, raw response capture, usage and failure records |
| Agent orchestration | Native Python | Explicit, inspectable study control flow |
| CLI | `argparse` | Prepare, run, resume, analyze, and plot |
| Contracts | Pydantic v2 | Configurations and validated data/model contracts |
| Configuration | TOML, read with `tomllib` | Versioned experiment and model settings |
| Data | Hugging Face Datasets | Load pinned dataset revisions |
| Mathematical checks | SymPy | Symbolic checks only where applicable |
| Run ledger | SQLite | Call identity, attempts, responses, reservations, and recovery state |
| Interchange | JSON / JSONL | Versioned manifests and derived trace exports |
| Analysis | NumPy, pandas, SciPy | Metrics and statistical summaries |
| Figures | Matplotlib | Reproducible PDF, SVG, and PNG plots |
| Tests and lint | pytest, Ruff | Contract tests and static checks |
| Work tracking | Git and GitHub Issues | Link protocol, work, and source revisions |

Do not add LangChain, LangGraph, AutoGen, CrewAI, a web framework/frontend, Docker/Kubernetes, Redis/Celery, local-model training/inference stacks, or experiment-tracking services without a new requirement. The first provider integration should use the direct HTTPX transport rather than a general provider router.

## Target layout

```text
AGENTS.md
README.md
pyproject.toml
uv.lock
.python-version
configs/
  models/default.toml
  motivation/ms1.toml
  motivation/ms2.toml
  motivation/ms3.toml
src/ricl_experiment/
  __main__.py  cli.py  config.py
  core/        model.py  ledger.py  runner.py  schemas.py
  datasets/    math500.py  omni_math.py  processbench.py  manifest.py
  evaluation/  answer.py  evidence.py  review.py
  motivation/  ms1_dependency.py  ms2_progress.py  ms3_repair.py
  analysis/    metrics.py  statistics.py  figures.py
docs/           protocol and decisions
prompts/        study prompts
 tests/         fixtures and contract tests
data/           local fixed snapshots; ignored by Git
artifacts/      local run output; ignored by Git
```

Create modules when their vertical slice is implemented; do not add empty placeholder files or directories.

## Module responsibilities

- `core/model.py`: request/response transport, effective parameters, usage, and raw response references.
- `core/ledger.py`: SQLite authority for logical-call identity, individual send attempts, response persistence, budget reservations, and resume state.
- `core/runner.py`: study execution, bounded dispatch, and recovery.
- `datasets/`: pinned source loading, eligibility/stratified sampling, identity, overlap, and versioned manifests.
- `evaluation/answer.py`: answer extraction, normalization, and independent equivalence decisions. Ambiguous/open-proof cases go to blinded review.
- `evaluation/evidence.py` and `review.py`: evidence records, blinded review tasks, and adjudication imports.
- `motivation/`: MS1, MS2, and MS3 study-specific orchestration. Issue #16's `ms2.py` owns candidate/Checkpoint/rollout control and the trace export; it does not calculate later evidence or ranking metrics.
- `analysis/`: deterministic offline metric/statistical calculation and plotting.

**Offline boundary:** `analysis/` must run without API credentials or network access. Analysis and figures are recomputed from persisted run artifacts; they must never call a model.

## Run artifact contract

Each run has an explicit, stable `run_id` under `artifacts/motivation/<run_id>/`:

```text
config.json
manifest.json
metadata.json
run.sqlite3                 # authoritative call/response/usage ledger
responses/                  # raw response bytes, referenced by the ledger
traces/                     # derived candidates, rollouts, evaluations (JSONL)
metrics/                    # derived per-sample CSV and summary/bootstrap JSON
figures/                    # derived PDF/PNG/SVG
report.md                   # derived report
```

SQLite is the source of truth for dispatch and persisted results. JSONL, metrics, figures, and reports are reproducible exports, not separately maintained state. Raw/restricted traces stay local; publish only after license, redaction, and label-leakage checks.

## CLI status

Implemented in issue #14:

```powershell
uv run python -m ricl_experiment run --study ms2 --phase pilot --config configs/models/default.toml --manifest configs/examples/single-sample.json
uv run python -m ricl_experiment resume --run-id <run_id> --config configs/models/default.toml --manifest configs/examples/single-sample.json
```

`prepare`, `exclude-groups`, and `overlap` create blinded manifests and offline audit exports from local pinned snapshots; see [dataset-manifests.md](dataset-manifests.md). The legacy `run` and `resume` retain their single-sample contract. `ms2 run`/`ms2 resume` consume a public MS2 manifest plus a separate restricted answer key, use three candidate slots per sample, and dispatch two independent continuations for each non-empty response. Continuation requests contain only the problem, original Checkpoint text, and frozen neutral instruction; answer keys, labels, scores, and sibling rollouts remain offline. `traces/ms2.json` preserves prefix and continuation text, segment bounds, source/attempt references, stop/truncation state, failures, usage, and evaluation references. `ms2-analyze` rebuilds MS2 metrics and coverage figures offline from `traces/ms2.json` plus adjacent scoring artifacts. `ms3-diagnose` runs the three isolated diagnosis methods and `ms3-analyze` computes fixed-denominator metrics offline. `ms3-repair` and `analyze-ms3-repair` provide the separate four-group repair path. `ms3-pilot-check` verifies the frozen 10-trajectory MS3 Pilot shape, method/repair coverage, request leakage, SQLite ledger identity, restart persistence, and offline metric recomputation; it also writes sample-size/cost planning and explicitly separates local engineering evidence from real API and human-sign-off records. See [ms3-pilot-acceptance.md](ms3-pilot-acceptance.md). Static model capabilities are checked before dispatch. Resume validates the original `run_id` and rejects changed config, manifest, answer key, or prompt fingerprints. Each HTTP send is preceded by a committed attempt row and output-token reservation; unknown usage remains reserved. Responses and unique logical-call analysis ownership are committed in SQLite, and only one scheduler process may hold a run at a time. MS1 offline tools plan a frozen sample, create blind annotation packets, reconcile two independent annotations and optional third reviews, and recompute dependency/re-derivation metrics and cases. They do not perform the requested Agent annotations or model re-derivations. Real Pilot/confirmatory results and final research reporting still require their own signed run and review artifacts.

For the isolated MS2 adjustment-validation phase, the runner requests a Checkpoint before the final answer, standardizes continuation answers, and enables JSON-object mode for evidence extraction and review; the frozen phase and prompt versions are recorded in the trace and run fingerprint. See [MS2 adjustment validation](ms2-adjustment-validation.md).

Model requests must be built from role-specific allowlists. Standard answers, expert labels, independent evaluations, and future outcomes remain unavailable to generation/rating roles except for protocol-defined oracle conditions. Test the actual serialized request, not only an internal role declaration.
