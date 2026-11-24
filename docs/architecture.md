# Architecture and workflow

This document preserves the agreed bootstrap choices and target contracts. Issue #14 implements the first vertical slice: a configurable, single-sample `run`/`resume` CLI with HTTPX dispatch and SQLite auditing. Issue #15 adds local fixed-snapshot preparation and independent answer/ProcessBench audit artifacts. Prepared manifests are not yet consumed by multi-sample study orchestration; analysis and plotting remain future slices.

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
- `motivation/`: MS1, MS2, and MS3 study-specific orchestration.
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

`prepare`, `exclude-groups`, and `overlap` create blinded manifests and offline audit exports from local pinned snapshots; see [dataset-manifests.md](dataset-manifests.md). `run` and `resume` share validated TOML/JSON configuration. The initial runner accepts exactly one manifest sample. `analyze` and `plot`, plus study-specific multi-sample orchestration, remain future issue slices. Static model capabilities are checked before dispatch. Resume requires the original `run_id` and rejects changed config, manifest, or prompt fingerprints. Each HTTP send has a committed attempt row and output-token reservation first; unknown usage remains reserved. Responses and the unique logical-call analysis owner are committed in SQLite, and only one scheduler process may hold a run at a time.

Model requests must be built from role-specific allowlists. Standard answers, expert labels, independent evaluations, and future outcomes remain unavailable to generation/rating roles except for protocol-defined oracle conditions. Test the actual serialized request, not only an internal role declaration.
