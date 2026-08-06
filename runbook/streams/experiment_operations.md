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

## 2026-08-06 [codex] establish full-attention probe-v2 baseline

**Context**

- Repeated the five-minute full-attention discovery baseline after registering
  associative-recall probe v2 at clean commit `f754fbc`.

**Commands**

```bash
uv run --no-sync research run \
  --config configs/research/discovery.toml \
  --candidate configs/candidates/baseline_full.toml
```

**Artifacts**

- `runs/20260806T020617Z-baseline-full-f754fbc3-s42/summary.json`

**Result**

- Complete and classified as the first valid v2 frontier candidate.
- Validation BPB 1.150807, median throughput 155,077 tok/s, peak allocated
  memory 1,990.66 MiB, and inference-state estimate 9,437,184 bytes.
- Probe easy accuracy 1.0, memory AUC 0.988607, update accuracy 0.804688, and
  256/512/1024/2048 load-curve accuracy 0.9967/0.9948/0.9902/0.9648.
- Worst slice was 0.492188 on the eight-overwrite stress cell, not the
  long-distance load cells.

**Next**

- Run the matched pure-SWA discovery baseline from the same commit and compare
  BPB, throughput, state bytes, and the calibrated boundary/long-range slices.

## 2026-08-06 [codex] establish pure-SWA probe-v2 baseline

**Context**

- Ran the matched pure-SWA discovery candidate after recording the full-attention
  baseline. Protected code/config and the registered v2 calibration were unchanged.

**Commands**

```bash
uv run --no-sync research run \
  --config configs/research/discovery.toml \
  --candidate configs/candidates/baseline_swa.toml
```

**Artifacts**

- `runs/20260806T023214Z-baseline-swa-a7062d9e-s42/summary.json`

**Result**

- Complete; classified `retest` because state bytes improved while BPB, memory
  AUC, update accuracy, and measured throughput regressed.
- Versus full attention: BPB 1.209343 versus 1.150807; memory AUC 0.569336
  versus 0.988607; update accuracy 0.615234 versus 0.804688; throughput 128,106
  versus 155,077 tok/s; state estimate 2,359,296 versus 9,437,184 bytes (-75%).
- Peak allocated training memory was effectively unchanged near 1,991 MiB.
  The current PyTorch SDPA fallback does not realize SWA kernel-efficiency gains.
- Boundary accuracy was 1.0000/0.9844/0.9766 at distance 255/256/257, 0.8672
  at 512, and 0.03125 at 1,024. The gradual reach reflects propagation through
  stacked local layers; the calibrated long-range failure is unambiguous.

**Next**

- Preserve both baselines as the v2 reference frontier/evidence set.
- Integrate a correctness-first KDA mixer behind the protected probe shell.
  Treat an optimized SWA kernel as a separate performance intervention so it
  is not confounded with KDA quality experiments.
