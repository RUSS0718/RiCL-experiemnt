# Agent entry points

- Project setup and current status: [README.md](README.md).
- Research architecture and artifact boundaries: [docs/README.md](docs/README.md) → [architecture.md](docs/architecture.md).
- Runtime code: `src/ricl_experiment/`; behavior tests: `tests/`.
- Protocol and acceptance criteria: GitHub Issues #13–#26; record durable additions in `docs/`.
- Keep local datasets, run artifacts, and credentials out of Git; generated data belongs under ignored `data/` or `artifacts/`.

## Research execution rules

- Prioritize real empirical evidence over additional infrastructure and mock-based validation.
- Never treat passing tests, offline fixtures, or artifact validators as evidence of model effectiveness.
- Keep pilot, protocol-development, and confirmatory samples isolated by question group.
- Do not change frozen prompts, sampling rules, budgets, or primary metrics after observing confirmatory outcomes.
- Distinguish tool-verified mathematical evidence from LLM-assessed evidence.
- Never fabricate results, reviewer judgments, model usage, or sign-off records.
- All analysis must remain reproducible offline without API credentials.
- Real API execution requires an approved model, sample scope, and bounded budget.
- Avoid unrelated refactoring before MS2/MS3 Pilot acceptance.

## Research Lead Review Interface

When requesting human approval, provide a concise, decision-oriented summary instead of dumping raw artifacts.

The summary must contain:

- Current finding and research status.
- Blocking issues and their root causes.
- Recommended decision and alternatives.
- Exact next experiment scope.
- Expected API calls, token usage, and maximum cost.
- Acceptance criteria.
- Explicit authorization boundaries.

Keep machine-readable artifacts, hashes, ledgers, provenance, and detailed audit reports available for traceability, but do not require the research lead to inspect them routinely.

Do not request repeated approval for unchanged decisions or ordinary implementation details.

Human approval should focus on scientific validity, protocol changes, spending authority, and experiment progression.
