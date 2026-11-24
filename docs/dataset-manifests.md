# Fixed-snapshot sample manifests

Issue #15 provides deterministic preparation from local JSON snapshots. It does not download data or call a model provider. Before creating a snapshot, retrieve data from the exact pinned dataset revision in `src/ricl_experiment/datasets/manifest.py`; the `revision` field is checked, while the SHA-256 snapshot fingerprint records the exact local JSON bytes. The fingerprint does not independently prove dataset provenance.

## Snapshot and plan

A snapshot file contains one source, pinned revision, split, and a `records` array. Keep the source's fields intact; add stable `question_group_id` values from reviewed upstream mappings when available. ProcessBench error records need independently verified scoring material before they qualify for repair.

```json
{
  "source": "math500",
  "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
  "split": "test",
  "records": [{
    "unique_id": "row-1",
    "problem": "Solve 2 + 2.",
    "answer": "4",
    "subject": "Algebra",
    "level": 4
  }]
}
```

Plans specify `study` (`ms2` or `ms3`), `phase` (`pilot`, `adjustment_validation`, or `confirmatory`), an integer `seed`, and a positive quota per stratum. Generate stratum keys with the public `format_stratum([...])` API; MATH-500 uses source/subject/level/image status, Omni-MATH uses source/domain/difficulty/source, and ProcessBench uses source/subset/generator/error state. Add `excluded_question_group_ids` to keep previously observed question groups out of a later freeze.

```json
{
  "study": "ms2",
  "phase": "pilot",
  "seed": 17,
  "quotas": {"[\"math500\",\"Algebra\",4,\"no_image\"]": 1}
}
```

Run from the repository root:

```powershell
uv run python -m ricl_experiment prepare `
  --snapshot data/snapshots/math500-test.json `
  --plan configs/motivation/ms2-pilot.json `
  --output artifacts/manifests/ms2-pilot
```

The output path must not already exist. `manifest.json` is the blinded sample list. `restricted/audit.json` contains qualification and identity review records, exclusions, strata counts, ProcessBench labels, and pool membership. `restricted/answer-key.json` contains reference answers/solutions or ProcessBench scoring material. Keep the `restricted/` directory access-controlled; it is intentionally separate from the manifest supplied to model roles. Record each output manifest fingerprint with the study materials.

The identity rule keeps exact normalized problem matches together when no upstream group mapping is present. A conservative lexical near-match heuristic (SequenceMatcher ratio >= 0.97 after case-folding, shortlisted by shared token trigrams) creates review candidates only; it never merges groups. Candidate groups are held out until reviewed, and the audit records both source record IDs, candidate IDs, and the review reason. This heuristic is not exhaustive, so review cross-source overlap before freezing confirmatory samples. Explicit `identity_status`/`identity_candidate_ids` annotations in the snapshot remain supported.

## Isolation and overlap helpers

For a prior MS2 confirmatory manifest, export its unique original-question group IDs and feed them into a later plan's `excluded_question_group_ids`:

```powershell
uv run python -m ricl_experiment exclude-groups `
  --manifest artifacts/manifests/ms2-confirmatory/manifest.json `
  --output artifacts/manifests/ms2-excluded-groups.json
```

For same-phase MS2 and MS3 manifests, record their overlap:

```powershell
uv run python -m ricl_experiment overlap `
  --left artifacts/manifests/ms2-pilot/manifest.json `
  --right artifacts/manifests/ms3-pilot/manifest.json `
  --output artifacts/manifests/pilot-overlap.json
```

`assert_phase_exclusive` checks Pilot/adjustment-validation versus confirmatory rows by `question_group_id`; same-stage cross-Study overlap is reported, not rejected. All identity checks depend on the quality of upstream group mappings and manual reviews.

## ProcessBench and answer evaluation

A ProcessBench trace must have at least two non-empty steps, a valid trajectory ID, a Boolean `final_answer_correct`, and a legal label. Label `0` means the first erroneous step is at zero-based index 0; `n >= 0` means index `n`; `-1` means no erroneous step. Error traces without independently verified reference material are excluded from both the shared diagnosis/repair pool. Correct traces can enter diagnosis only. These labels and pools are written only to the restricted audit sidecar, not the public manifest.

Closed-answer rules use `closed-answer-v1` and accept boxed, marked, or supported numeric/symbolic final answers. Normalization and the safe expression equivalence subset are deterministic. Conflicting explicit answers, non-extractable or out-of-domain answers, missing references, and open proofs go to `blind_review`. Adjudication can record `correct`, `incorrect`, or `unknown`; absent output is a separate `no_evaluable_output` category. The result writer emits one file per category and refuses to overwrite an existing output directory.

Model request bodies are built from a role-specific field allowlist. The current `run` command still accepts exactly one prompt sample; `prepare` creates multi-sample manifests but does not wire them into multi-sample study execution. MS1/MS2/MS3 orchestration and evidence/reviewer workflows remain later work.
