# Exact KDA Research Operating Contract

This repository optimizes end-to-end Kimi Delta Attention training throughput
while preserving the full KDA computation and numerical behavior. The published
lane is a six-layer nanochat model on NVIDIA GB10/SM121.

## Session protocol

- Read this file, `AGENTS.md.local` if present, `program.md`, and
  `runbook/index.md`.
- State `Mode: No-Edits` for planning/review or `Mode: Execution` for changes.
- In Execution mode, inspect `git status --short` before editing and preserve
  unrelated user changes.
- Use `uv` for Python dependencies and execution.
- Do not launch training or a costly GPU campaign unless the user requests it.

## Non-negotiable integrity rules

- Optimize the exact KDA training path. Do not freeze trainable parameters,
  detach values that require gradients, install surrogate/value-only backwards,
  alter the KDA equations, or silently fall back to another backend.
- `tests/kda_oracle.py`, correctness tests, benchmark parsing, and release
  evidence are protected from candidate-driven changes.
- FLA is a performance comparator, never the numerical oracle.
- Report crashes, OOMs, NaNs, timeouts, parser failures, and missing metrics as
  invalid runs—not slow runs and never improvements.
- Do not claim performance or correctness without saved matched-run evidence
  tied to a clean commit.
- Never optimize against private confirmation seeds or unpublished holdouts.

## Matched comparison contract

Unless it is the declared intervention, hold constant model shape, KDA layer
count, sequence length, batch and accumulation, precision, execution mode,
seed/data order, optimizer/schedule, compile treatment, hardware, backend flags,
warm-up, scored steps, and aggregation. Record requested and resolved backends,
fallback state, peak memory, all run medians, failures, and source provenance.

The release benchmark is frozen in `docs/BENCHMARK.md`. Its headline must always
distinguish the strongest observed run median (45,058 tok/s) from the confirmed
three-run median (44,942 tok/s).

## Candidate workflow

1. State one mechanism and a quantitative expected effect.
2. Work on an isolated branch/worktree; retain the current exact baseline.
3. Run the smallest independent-oracle and random-upstream gradient gate that
   can falsify the candidate.
4. Commit the candidate before conclusion-bearing performance work.
5. Run Level 1 production-shape kernel/layer measurement before any trainer.
6. Advance only a material exact result to matched end-to-end trainers.
7. Compare losses and final model tensors when the intervention touches
   backward, accumulation, caching, capture, or optimizer boundaries.
8. Record accepted, rejected, neutral, and invalid outcomes in the attempt
   ledger and KDA runbook.

Candidate source changes should normally stay within `nanochat/mixers/kda.py`,
`nanochat/mixers/cuda_kda/`, and narrowly required KDA integration in
`nanochat/gpt.py`. Changes to the benchmark, oracle, trainer, results, or release
contract require explicit human scope expansion.

## Useful commands

```bash
uv run --no-sync python -m pytest -q
uv run --no-sync python -m pytest -q \
  tests/test_kda_operator.py tests/test_kda_layer.py \
  tests/test_kda_cuda.py tests/test_kda_integration.py
```

Read `program.md` before autonomous experimentation. Update
`runbook/streams/kda_cuda_ownership.md` after any material state, evidence,
constraint, failure, or next-action change. Never commit datasets, checkpoints,
secrets, private paths, bulky profiler traces, or build caches.
