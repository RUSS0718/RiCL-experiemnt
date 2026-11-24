# MS3 Pilot acceptance gate

Issue #22 adds `ms3-pilot-check`, an offline gate over already persisted MS3 artifacts. It is an engineering acceptance tool, not a real-provider runner and not a substitute for human sign-off.

## Inputs and command

Run the diagnosis and repair paths first, using the existing frozen manifest and restricted audit artifacts. The gate then reads only local JSON/JSONL/SQLite files:

```powershell
uv run python -m ricl_experiment ms3-pilot-check `
  --manifest <pilot>/manifest.json `
  --audit <pilot>/restricted/audit.json `
  --answer-key <pilot>/restricted/answer-key.json `
  --diagnosis-run <diagnosis-run> `
  --diagnosis-analysis <diagnosis-run>/metrics/ms3.json `
  --repair-run <repair-run> `
  --repair-analysis <repair-run>/metrics/ms3-repair.json `
  --output <acceptance>/status.json `
  --plan-output <acceptance>/sample-size-cost.json `
  --pricing <acceptance>/pricing.json
```

`pricing.json` is optional and must be a JSON object with non-negative `input_per_1k_tokens` and `output_per_1k_tokens`. Without it, the gate still reports observed token totals but marks cost planning as `pricing_not_provided`.

## Frozen engineering checks

The status artifact reports each check separately and returns exit code 0 only when all local checks pass:

- **Sample shape:** exactly 10 unique trajectory/question-group identities, with 8 error labels and 2 `-1` no-error labels. The diagnosis pool must contain all 10 and the repair pool exactly the 8 error trajectories.
- **Diagnosis methods:** exactly `direct_judge`, `step_by_step_critic`, and `reflection`; each must contain one result per trajectory and at least 9 legal results (`completed` or `abstained`). Parse, schema, API, and runtime failures remain visible in status counts.
- **Repair coverage:** the error pool must have one assignment for each of `G0`, `G1`, `G2`, and `G3`. At least 6 error trajectories must have independently evaluable `repair_success` for all four groups. Missing, unknown, and failed evaluations remain in the underlying artifacts.
- **Leakage:** serialized diagnosis and repair requests are scanned for restricted fields and restricted answer/reference text. The check examines actual request artifacts, not only role declarations.
- **Ledger:** logical call IDs must match request artifacts; analysis ownership must be unique; attempts must point to persisted response files.
- **Restart:** both SQLite ledgers are reopened and compared for stable persisted state. A provider restart/resume execution is never inferred from this check. Pass `--restart-record` only when a separate local or real execution record exists; a missing record is reported as `resume_execution: not_run`.
- **Offline recompute:** diagnosis metrics are rebuilt with `analyze_diagnoses`; repair metrics are rebuilt with `analyze_repairs`; canonical JSON must match the supplied reports.

## Planning artifact

`sample-size-cost.json` contains, for `G3-G1`, `G3-G2`, and `G1-G0`:

- observed candidate/evaluable pair counts and evaluable rates;
- the pilot paired-difference standard deviation, or a documented binary fallback when the pilot is too small or degenerate;
- baseline and conservative required evaluable/candidate counts for a configurable 95% normal-approximation half-width;
- recorded input/output/reasoning tokens, unknown-usage attempts, and an estimate only when explicit prices are supplied.

The estimate is labelled as a calculation over recorded local artifacts, not a provider invoice.

## External evidence boundary

The generated status always includes `external_evidence` and `confirmatory_readiness`. If no records are passed, or a requested record path is missing, the report says:

- `real_api.status = not_provided`;
- `human_signoff.status = not_provided`;
- `confirmatory_readiness = not_ready_without_external_evidence`.

A missing record is reported as `missing_record`; a supplied record is only marked as present and does not become approval. The gate does not read environment credentials, call a model endpoint, fabricate provider usage, or infer human approval from passing local checks. Existing MS3 transport tests use controlled local endpoints; this gate itself remains offline.
