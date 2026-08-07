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

## 2026-08-06 [agent] bounded 4k full-attention feasibility smoke

**Context**

- The current 1,024-token training lane is too short for full RULER coverage.
  Before designing a matched longer-context lane, the requested question was
  whether the existing GB10 setup can execute the current model at 4,096 tokens.
- This was one optimizer update and one checkpoint-load forward check, not a
  discovery or quality run.

**Commands**

```bash
NANOCHAT_DTYPE=bfloat16 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
  uv run --no-sync python -m scripts.base_train \
  --seed 42 --depth 6 --head-dim 128 --window-pattern L \
  --max-seq-len 4096 --device-batch-size 2 --total-batch-size 32768 \
  --num-iterations 1 --eval-every -1 --core-metric-every -1 \
  --sample-every -1 --save-every -1 --model-tag smoke-full-4k --run dummy
```

**Artifacts**

- Ignored smoke log: `runs/4k-full-attention-smoke.log`
- Nanochat-cache checkpoint tag: `smoke-full-4k`, step 1

**Result**

- Complete: one BF16 full-attention optimizer update at 4,096 tokens with
  finite loss 10.396439 and peak allocated memory 1,672.17 MiB. The checkpoint
  reloaded and produced finite logits of shape `[1, 4096, 32768]`.
- The microbatch was `2 × 4096 = 8,192` tokens with four accumulation steps,
  preserving the 32,768-token global batch. The instantiated model had
  73,531,538 parameters.
- The reported first-step 3,750 tok/s and 8.736 s step time include compilation
  and warmup; they are not throughput evidence. PyTorch SDPA fallback was used
  because Flash Attention 3 is unavailable.

**Next**

- Add a frozen 4k lane only after the RULER preparation path rejects impossible
  prompt/generation budgets before launching work. Run equivalent bounded SWA
  and KDA correctness/memory checks before any matched longer-context campaign.

## 2026-08-06 [agent] bounded 4k SWA/KDA execution checks

**Context**

- After the full-attention 4k smoke, ran the same one-update budget for pure
  SWA and KDA before allowing any longer-context comparison.

**Commands**

```bash
# Shared: BF16, sequence length 4096, device batch 2, global batch 32768,
# one optimizer update, no periodic evaluation.
uv run --no-sync python -m scripts.base_train --window-pattern S --no-force-final-full <shared arguments>
uv run --no-sync python -m scripts.base_train --window-pattern K --no-force-final-full --kda-backend fla_triton <shared arguments>
```

**Artifacts**

- Ignored smoke logs: `runs/4k-swa-smoke.log`, `runs/4k-kda-smoke.log`
- Nanochat-cache checkpoint tag: `smoke-swa-4k`, step 1

**Result**

- Pure SWA completed one update with finite loss 10.396439 and peak allocated
  memory 1,704.42 MiB. The first step took 7.267 s / 4,509 tok/s including
  compilation and warmup, so it is not comparative throughput evidence. SDPA
  warned that it has no sliding-window implementation.
- KDA did not reach its first optimizer update. It remained at compilation/
  setup for 900 seconds, was terminated, and is an invalid run. No checkpoint,
  quality, memory, or performance value may be inferred from it.

**Next**

- Diagnose the 4k KDA compile/setup timeout with a tightly bounded backend
  smoke before any 4k KDA training attempt. Keep the full/SWA work as
  correctness-only evidence.

## 2026-08-06 [agent] add bounded systems-benchmark foundation

**Context**

- The 4k KDA path timed out during compilation/setup, while the existing
  training summary conflates compile and first-step timing. A protected
  systems lane is required before automated performance iteration can be
  trusted.

**Commands**

```bash
uv run --no-sync python -m pytest -q
uv run --no-sync research systems --help
uv run --no-sync research doctor --config configs/research/systems_4k.toml
```

**Artifacts**

- `configs/research/systems_4k.toml`
- `nanochat/research/{config,systems,cli}.py`
- `tests/test_general_eval.py`

**Result**

- Added a frozen 4k systems config and `research systems` command. It requires
  a clean commit, runs the cold first-update subprocess under a 120-second
  timeout, then records a separately warmed training sample after fixed
  warmup/timed counts. Logs, resolved config, environment, compile result, and
  summary are saved as artifacts.
- A compile timeout is returned as `compile_timeout`, not converted into a
  throughput number. The command explicitly marks prefill/decode `not_run`;
  those measurements and profiler traces remain the next implementation step,
  so this initial foundation must not be described as a complete systems suite.
- Config validation covers timing budgets and shapes; 30 focused tests passed.
  Doctor validates all prepared evaluation inputs but is not research-ready in
  the current dirty worktree.

**Next**

- Commit the systems foundation, establish clean full/SWA baselines, and use
  the 120-second cap to reproduce the KDA compile failure. Add isolated
  prefill/decode and profiler artifacts before treating systems results as a
  complete candidate-selection signal.

## 2026-08-07 [agent] establish bounded 4k full-attention systems baseline

**Context**

- Ran the newly committed systems command only for the declared full-attention
  reference. This is a systems baseline, not a quality or architecture result.

**Commands**

```bash
uv run --no-sync research systems --config configs/research/systems_4k.toml
```

**Artifacts**

- Ignored artifact directory:
  `runs/systems-20260807T062318Z-systems-4k-40c427aa-s42/`

**Result**

- Cold first-update process completed in 6.606 s using the system CUDA 13.1
  assembler. The initial systems command first failed because it selected
  Triton's bundled assembler; the systems runner was corrected, committed, and
  rerun before accepting this baseline.
- After three warmup updates, ten matched full-attention training updates had
  median 0.226376 s step time and 144,750 tok/s at sequence length 4,096,
  device batch 2, and global batch 32,768. These are warmed training-only
  measurements, separate from compile time.
- Prefill and decode remain explicitly `not_run`; this artifact is not a
  complete systems scorecard. No KDA comparison was launched.

**Next**

- Add isolated prefill/decode measurement and profiler artifacts. Then rerun
  the KDA cold phase under the 120-second cap to classify its failure without
  another unbounded wait.

## 2026-08-07 [agent] classify the 4k KDA systems failure

**Context**

- The bounded systems runner was used to distinguish a compiler-only issue from
  the actual FLA KDA execution path. A matched eager full-attention control was
  run only for diagnosis.

**Commands**

```bash
uv run --no-sync research systems --config configs/research/systems_4k.toml \
  --candidate configs/candidates/kda_only.toml
TORCH_COMPILE_DISABLE=1 <one- and three-update KDA diagnostics at 4096>
TORCH_COMPILE_DISABLE=1 <two-update full-attention control at 4096>
```

**Artifacts**

- Ignored systems artifact:
  `runs/systems-20260807T180253Z-kda-only-b8a4492a-s42/`
- Ignored diagnostic logs: `runs/4k-kda-eager-{diagnostic,warm-diagnostic}.log`,
  `runs/4k-full-eager-control.log`

**Result**

- The compiled KDA cold phase hit the declared 120-second cap without reaching
  a training step. Its log ends after FLA Blackwell allocator registration; the
  bounded runner terminated it and no worker remained active.
- Disabling Torch compilation produced finite 4k KDA training with resolved
  `fla_triton` backend, no fallback, and three step times of 37.905, 38.656,
  and 39.176 seconds (864/847/836 tok/s). The diagnostic process was terminated
  during teardown at its 120-second wall cap, so it is not a valid systems-run
  artifact, but the completed structured steps establish that the problem is
  not compile-only.
- The matched eager full-attention control completed at 1.212 then 0.702 s
  (27,041 then 46,663 tok/s), with comparable peak allocation near 5.6 GiB.
  Thus the present eager 4k KDA integration is roughly 56x slower on the second
  step despite having fewer parameters. This is diagnostic evidence of a real
  execution-path bottleneck, not an architecture-quality conclusion.

**Next**

- Add a bounded profiler capture for one eager KDA step and attribute time among
  local projections/convolutions, FLA chunk calls, norms, and optimizer work.
  Do not launch another compiled KDA training run until that trace identifies a
  concrete primary intervention.

## 2026-08-07 [agent] reject an unbounded full-step KDA trace

**Context**

- Attempted the requested eager 4k KDA profile after the bounded systems
  diagnosis. The first approach captured a full CPU/CUDA Chrome trace around a
  warm step, which is not suitable at this shape.

**Commands**

```bash
TORCH_COMPILE_DISABLE=1 <one-warmup, one-profiled 4k KDA optimizer step>
```

**Artifacts**

- No retained profile artifact. The partial ignored trace was deleted.

**Result**

- The profile process exceeded its 240-second cap and was terminated. It had
  emitted a 3.5 GiB partial Chrome trace before reporting an operator table;
  the trace, script, and log were deleted immediately and no profiler process
  remained. This is an invalid profiling attempt, not performance evidence.

**Next**

- Do not export a full Chrome trace at this shape. Add low-overhead CUDA-event
  component timing around KDA projections/convolutions, FLA chunk work, output
  gate/norm, and optimizer; only then use a single restricted profiler table or
  a targeted kernel trace for the identified dominant component.

## 2026-08-07 [agent] vectorize the diagnosed KDA short-convolution bottleneck

**Context**

- The restricted no-export CUDA profiler was run after the rejected full Chrome
  trace. It completed in 94 seconds and retained only a 16 KiB key-averages
  table plus a 74-byte summary; no Chrome trace exists. The table showed over
  1.09 million CUDA launches, dominated by tiny generic elementwise kernels.
- The cause was local and concrete: `ShortConvolution.forward` used a Python
  loop over every token. At 4k, six KDA layers and q/k/v convolutions execute
  73,728 iterations before backward/optimizer work, rather than issuing a
  grouped depthwise convolution over the sequence. FLA KDA kernels appeared in
  the table but were not the dominant captured kernel time.

**Commands**

```bash
uv run --no-sync python -m pytest -q tests/test_kda_layer.py tests/test_kda_operator.py
uv run --no-sync python -m pytest -q tests/test_kda_cuda.py
TORCH_COMPILE_DISABLE=1 <three-update matched 4k KDA eager diagnostic>
uv run --no-sync research systems --config configs/research/systems_4k.toml \
  --candidate configs/candidates/kda_only.toml
```

**Artifacts**

- Ignored restricted profiler: `runs/4k-kda-eager-profile/cuda-key-averages.txt`
  and `summary.json`; it has no `trace.json`.
- Ignored eager diagnostic: `runs/4k-kda-eager-vectorized-conv.log`.
- Ignored compiled systems result:
  `runs/systems-20260807T201839Z-kda-only-0a4c60c8-s42/`.
- Candidate commit: `0a4c60c` (`Vectorize KDA short convolution`).

**Result**

- Replaced only the full-sequence short-convolution token loop with equivalent
  grouped `F.conv1d`; its cache convention and single-token path remain the
  same. Focused CPU tests passed (22) and CUDA tests passed (15), including
  production-dimension compiled forward/backward coverage.
- With compilation disabled and the same 4k/batch/seed/model settings as the
  prior eager diagnosis, the post-change steady updates were 0.79523 and
  0.79542 s (41,205/41,195 tok/s), versus 38.656 and 39.176 s
  (847/836 tok/s) before. The loss sequence was the same to shown precision,
  FLA resolved to `fla_triton` without fallback, and peak allocation fell from
  5,980 to 5,664 MiB. This is a roughly 49x eager diagnostic speedup from the
  single vectorization intervention.
- The official compiled systems phase is still invalid, but it no longer
  timed out in KDA setup. It crashed after 96.28 seconds in protected
  `adamw_step_fused`: Dynamo reached its fullgraph recompile limit because its
  parameter loop encountered ranks 3 then 1. No optimizer step completed. This
  is an optimizer-compilation integration blocker, not evidence that the new
  KDA mixer is slow or incorrect.

**Next**

- Preserve the vectorized KDA candidate. Before a compiled systems claim,
  address the protected optimizer's rank-polymorphic fullgraph compilation in
  a separately reviewed systems change (or deliberately benchmark model
  compilation with an eager optimizer); do not hide it by a candidate fallback.
