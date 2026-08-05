# Experiment Evidence Contract

Before a run, record the question, one primary intervention, hypothesis,
baseline, success/failure rule, resolved config, clean commit, seed, data and
tokenizer identity, training budget, batch semantics, precision, hardware, and
kernel/fallback selection.

Correctness gates include causal/window boundaries, packed-document isolation,
state reset, last-write updates, parallel/chunk/recurrent equivalence where
applicable, finite forward/backward, and checkpoint/decode semantics.

Performance evidence separates compile/warmup from steady state and records
shapes, dtype, device topology, throughput statistic, peak allocated memory,
prefill/decode behavior, KV or recurrent state bytes, and multiple lengths.

Required local artifacts are resolved config, Git/environment/protected-file
manifest, raw log, structured metrics, checkpoint selector, final summary, and
exact failure record. A dirty or uncommitted tree is diagnostic-only evidence.

Candidate decisions use a tolerance-aware Pareto frontier. A candidate is a win
only if it improves at least one objective beyond calibrated noise without a
meaningful regression. Near ties require retesting; crashes and missing metrics
are invalid. Promotion and confirmation report seed-level values and aggregate
uncertainty, never only the best seed.
