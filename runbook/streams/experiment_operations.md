# Experiment Operations

## Objective

Maintain one reproducible PyTorch/uv path for setup, training, evaluation,
artifact capture, and comparison on a single DGX Spark.

## 2026-08-05 [codex] promote nanochat and add research runner

**Context**

- Promoted the cloned nanochat tree to repository root and initialized root Git.
- Preserved the upstream revision in `UPSTREAM.md`; `ref/` remains ignored.
- Added declarative discovery/promotion configs, structured trainer/evaluator
  results, artifact manifests, and supervisor commands.

**Commands**

```bash
cd <repo>
uv run --no-sync python -m pytest -q tests/test_research_harness.py
uv run --no-sync python -m pytest -q
uv run --no-sync research doctor --config configs/research/discovery.toml
```

**Artifacts**

- `UPSTREAM.md`
- `configs/research/{discovery,promotion}.toml`
- `nanochat/research/`
- `containers/research.Dockerfile`
- `tests/test_research_harness.py`

**Result**

- Research-harness tests: 16 passed with the known GB10 capability warning.
- Full suite: 64 passed, 10 skipped. Environment checks passed; research
  readiness is false only because the new repository has no clean commit yet.
- A pure-SWA fixed-time smoke stopped on budget at 0.0507 measured seconds,
  trained 19,456 tokens, recorded windows `[[64, 0], [64, 0]]`, and emitted a
  structured result. This was a plumbing check, not comparative evidence.
- Discovery is 300 seconds of measured steady-state training; promotion is
  100,663,296 tokens. No costly campaign was launched during implementation.

**Next**

- Create the initial clean commit before any conclusion-bearing run.
- Calibrate decision tolerances with the declared five baseline seeds.
- Build and exercise the restricted container before private confirmation.

## 2026-08-05 [codex] select a GB10-compatible Triton assembler

**Context**

- The first full-attention discovery run reached initial validation and then
  crashed during compiled backward. Triton 3.5.1 targeted `sm_121a`, but its
  bundled CUDA 12.8 `ptxas` did not recognize that target.
- The installed system CUDA 13.1 assembler advertises `sm_121a` support.

**Commands**

```bash
cd <repo>
TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas uv run --no-sync python <compiled-backward-smoke>
uv run --no-sync python -m pytest -q
```

**Artifacts**

- `runs/20260805T233503Z-baseline-full-f16035bd-s42/train.log`
- `nanochat/research/artifacts.py`
- `nanochat/research/runner.py`

**Result**

- Root cause: `ptxas fatal: Value 'sm_121a' is not defined for option 'gpu-name'`.
- A compiled BF16 RMSNorm/matmul backward completed with finite gradients using
  `/usr/local/cuda/bin/ptxas` from CUDA 13.1.
- The runner now selects and records that assembler on SM 12.1 systems.

**Next**

- Commit the compatibility fix, then rerun the full-attention discovery baseline.
