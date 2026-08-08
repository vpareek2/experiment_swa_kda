# Autoresearch Program

Your job is to discover defensible SWA/KDA architecture improvements, not to
maximize a single visible number.

## Mode selection

The default program below governs architecture-quality research. When a
supervisor explicitly assigns the completed protected KDA training-speed loop,
read and follow `program_kda_training_speed.md`. When it assigns the staged KDA
CUDA-ownership loop, read and follow `program_kda_cuda_ownership.md` instead.
Those modes have systems-only objectives and do not authorize quality training,
general evaluation, or direct candidate edits by the supervisor.

1. Read `AGENTS.md`, `runbook/index.md`, and the active architecture/evaluation
   streams. Verify `research doctor` and the current frontier artifacts.
2. State one hypothesis and predicted observables. Change only one primary
   architecture axis and only candidate-allowed paths.
3. Run focused correctness tests. A masking, state, boundary, NaN, OOM, crash,
   or artifact failure invalidates the candidate.
4. Commit the candidate so the run has immutable provenance. Run the discovery
   config; do not alter the protected config, probe, evaluator, data, tokenizer,
   decision logic, or supervisor.
5. Confirm the registered probe calibration matches the current protected-code
   and protocol hashes; an absent or stale calibration makes the run invalid.
6. Inspect BPB, memory AUC, update accuracy, worst slice, throughput, peak
   memory, and state bytes. Keep a change only when its Pareto classification
   and raw evidence justify it. `retest` is unresolved, not success.
7. Record the hypothesis, commit, config, artifact path, complete metrics,
   failure modes, and next decision in the runbook. Never cherry-pick only a
   favorable seed or hide failed runs.
8. Request promotion only after discovery correctness and frontier gates pass.
   Promotion uses the fixed-token config and three declared seeds. Private
   confirmation uses five supervisor-derived seeds in a separate restricted
   environment and may not be inspected or tuned against by candidate agents.

Do not change evaluation code to accommodate a candidate. If a legitimate new
mixer needs a protocol extension, stop and ask a human to update and re-freeze
the protected suite before comparing results.
