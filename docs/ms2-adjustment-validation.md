# MS2 adjustment validation

The `adjustment_validation` phase is isolated from the original Pilot: its frozen manifest excludes all Pilot question groups, and its run uses a separate run ID and artifacts directory.

Its versioned request contract asks each initial candidate to stop before a final-answer line, asks both independent continuations to end with exactly one `Final answer: <answer>` line, and enables OpenAI-compatible JSON-object response mode for evidence extraction and evidence review. The blinded rater request is unchanged. Answer keys, rollout siblings, and scoring outcomes remain excluded from model-visible payloads.

The run metadata and trace retain template versions and the exact instructions; request bodies and attempt fingerprints remain authoritative in SQLite. Resume verifies the frozen phase templates and inputs. Analyze the trace and scoring artifacts offline; this protocol change does not itself constitute Pilot acceptance or human sign-off.
