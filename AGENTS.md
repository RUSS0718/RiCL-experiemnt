# Agent entry points

- **Cross-repo method alignment (Source of Truth):** `D:\project\RiCL-home\docs\METHOD_ALIGNMENT_v0.2.md` (ACL 2027 Method v0.2: On-Demand Lemma DAG + Hybrid Verifier + `{Continue, Pruned-Repair, Salvage-Switch}`; original dual-arm CCL is reserved for offline/controlled attribution analysis, not online inference).
- Project setup and current status: [README.md](README.md).
- Research architecture and artifact boundaries: [docs/README.md](docs/README.md) → [architecture.md](docs/architecture.md).
- Runtime code: `src/ricl_experiment/`; behavior tests: `tests/`.
- Protocol and acceptance criteria: GitHub Issues #13–#26 (v1.0 Motivation archive) and Track 1 v0.2 issues; record durable additions in `docs/`.
- Keep local datasets, run artifacts, and credentials out of Git; generated data belongs under ignored `data/` or `artifacts/`.

## Two-Track Governance (v0.2)

- **Track 1 — Exploration / Dev (ACTIVE):**
  - Goal: fast empirical signal on 30–50 real hard math problems (`AIME / OlympiadBench / MATH Level 5`, target model pass@1 ~30%–65%) using `BASE_URL=https://cn2.ezaigc.cn` and `max_tokens >= 8192`.
  - **Inspect freely:** unsealing group effects and error cases on the dev set is required for iteration; do NOT lock results behind an 80% gate during Track 1.
  - **Tolerance for proxy quirks:** minor usage subtotal drift (e.g., `reasoning_tokens` exceeding `completion_tokens` by 1–4 tokens on length truncation) must log a `warning` and continue — do NOT halt the entire run.
  - **Lightweight guardrails:** enforce a clear per-run max-request cap and local estimated cost cap; do NOT build extra audit frameworks, thousand-key exclusion locks, or unrelated unit tests before seeing real `Accuracy vs. Total Tokens` curves.
- **Track 2 — Confirmatory / Paper Eval (after Track 1 shows clear gain):**
  - Freeze prompts, sampling rules, budgets, and primary metrics on isolated test splits; restore full ledgers, strict missingness accounting, and statistical tests.

## Research execution rules

- Prioritize real empirical evidence over additional infrastructure and mock-based validation.
- Never treat passing tests, offline fixtures, or artifact validators as evidence of model effectiveness.
- Keep Track 1 dev samples and Track 2 confirmatory test samples isolated by question group.
- In Track 2, do not change frozen prompts, sampling rules, budgets, or primary metrics after observing confirmatory outcomes.
- Distinguish tool-verified (`Executable-Verified`) mathematical evidence from LLM-assessed (`Step-Reviewed`) evidence.
- Never fabricate results, reviewer judgments, model usage, or sign-off records.
- All analysis must remain reproducible offline without API credentials.
- Real API execution requires an approved model, sample scope, and bounded budget.

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
