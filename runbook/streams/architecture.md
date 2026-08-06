# SWA + Linear Attention Architecture

## Objective

Test whether local exact attention plus recurrent KDA memory can eliminate
periodic global attention at a better quality/efficiency frontier.

## Current decisions

- PyTorch/nanochat is the implementation base; JAX is no longer a constraint.
- Baselines precede KDA: full attention, pure SWA, then KDA-only and hybrids.
- `S` has an explicit window. `force_final_full` is separately recorded so a
  pure-SWA baseline cannot silently contain a global final layer.
- First hybrid layout is layerwise composition. Within-layer mixtures wait until
  layerwise baselines are understood.
- The canonical memory probe measures load, last-write-wins updates,
  interference, and distances around the SWA boundary.

## 2026-08-05 [codex] make SWA topology explicit

**Context**

- Removed an experimental confound in inherited window semantics while keeping
  checkpoint-compatible defaults.

**Commands**

```bash
cd <repo>
uv run --no-sync python -m pytest -q tests/test_research_harness.py
```

**Artifacts**

- `nanochat/gpt.py`
- `scripts/base_train.py`
- `nanochat/research/memory.py`
- `nanochat/research/probe.py`

**Result**

- Explicit 128-token pure-SWA final-layer behavior and opt-in global-final
  behavior are covered by tests.
- KDA is not integrated yet; no KDA or hybrid performance claim exists.

**Next**

- Pin FlashKDA equations, tensor layouts, resets, and recurrent/chunk parity.
- Add a correctness-first PyTorch mixer behind the same protected probe shell.

## 2026-08-05 [codex] audit K3 KDA references and FLA compatibility

**Context**

- Audited the imported Flash Linear Attention, FlashKDA, Kimi Linear, and Kimi
  K3 reference material before choosing an implementation source.
- The question was whether current FLA already includes K3's lower-bounded
  decay and full-rank output-gate changes.

**Commands**

```bash
rg -n "safe_gate|lower_bound|use_full_rank_gate|attn_res" ref/
git -C ref/flash-linear-attention log -- fla/layers/kda.py fla/ops/kda/gate.py
```

**Artifacts**

- `ref/k3_change.md`
- `ref/modeling_kimi_{linear,k3}.py`
- `ref/flash-linear-attention/fla/{layers,ops}/kda*`
- `ref/FlashKDA/`

**Result**

- The core recurrence is unchanged in K3. K3 adds log-decay
  `g = -5 * sigmoid(exp(A_log) * (z + dt_bias))` and a direct full-rank output
  gate before head-wise gated RMSNorm/output projection.
- Current FLA operators implement and test the K3 lower-bounded gate, including
  forward/backward, fused gate activation, QK L2 normalization, beta sigmoid,
  chunk/recurrent state, GVA, and ragged layouts.
- FLA's high-level `KimiDeltaAttention` exposes `safe_gate/lower_bound` but
  still hardcodes a two-layer bottleneck output gate. It is therefore only
  partially K3-compatible. Its KDA model does independently support Attention
  Residuals and configurable hybrid full-attention layers.
- The K3 text implementation lives in `modeling_kimi_linear.py` and selects
  both changes through `gate_lower_bound` and `use_full_rank_gate` config keys.
  `modeling_kimi_k3.py` delegates language modeling to that implementation.
- FlashKDA is an inference-only CUTLASS forward backend: BF16, K=V=128, no GVA,
  safe gate required. Its declared build architectures stop at `120a`; GB10
  `121a` must be compiled and smoke-tested rather than assumed supported.

**Next**

- Implement a small K3-compatible layer locally: use FLA's recurrence/gate as
  the operator contract, but use the direct full-rank output gate from the K3
  reference. Keep Attention Residuals as a separate later intervention.
- First add a dependency-free PyTorch recurrent oracle, then validate an FLA
  Triton training backend against it on GB10 before running the memory probe.

## 2026-08-05 [codex] establish red KDA parity suite

**Context**

- Added the tests before production KDA code so implementation work has an
  independent, observable correctness target.
- The scope is the K3 mixer and nanochat integration. Attention Residuals, GVA,
  variable-length packing, and the optional FlashKDA CUTLASS backend remain
  separate interventions.

**Commands**

```bash
uv run --no-sync python -m pytest --collect-only -q
uv run --no-sync python -m pytest -q --ignore=tests/test_kda_operator.py --ignore=tests/test_kda_layer.py --ignore=tests/test_kda_cuda.py --ignore=tests/test_kda_integration.py
uv run --no-sync python -m pytest -q -m "not cuda" tests/test_kda_operator.py tests/test_kda_layer.py tests/test_kda_integration.py
uv run --no-sync python -m pytest -q -m cuda tests/test_kda_cuda.py
```

**Artifacts**

- `tests/kda_oracle.py`
- `tests/test_kda_{operator,layer,cuda,integration}.py`
- `pyproject.toml`

**Result**

- The dependency-free oracle passes its six gate, recurrence, causality,
  update-order, batch-isolation, and short-convolution checks.
- The unchanged suite passes: 69 passed, 10 skipped.
- Test collection succeeds. The production-facing tests are intentionally red:
  the CPU/integration lane fails only on missing KDA APIs or unsupported `K`
  behavior, and all ten Spark CUDA cases select and fail on the missing module
  rather than skipping.
- The suite fixes the public state at FP32 `[B,H,V,K]` and tests unequal K/V
  dimensions, fused K3 transforms, gradients, chunk boundaries, decode parity,
  mixer dispatch, cache allocation, state accounting, and probe wiring.

**Next**

- Implement `nanochat.kda` reference mode until operator and layer CPU tests
  turn green, then integrate KDA patterns/cache/research accounting.
- Only after CPU parity passes, add the optimized training and recurrent CUDA
  backends and clear the marked Spark lane.

## 2026-08-05 [codex] implement correctness-first K3 mixer

**Context**

- Implemented the CPU/reference phase against the previously committed-style
  independent tests. No tolerances, skips, or oracle expectations were relaxed.
- The implementation is device-agnostic PyTorch but remains an intentionally
  sequential token loop; it is a correctness fallback, not a performance claim.

**Commands**

```bash
uv run --no-sync python -m pytest -q tests/test_kda_operator.py tests/test_kda_layer.py
uv run --no-sync python -m pytest -q tests/test_kda_integration.py
uv run --no-sync python -m pytest -q -m "not cuda"
TORCH_COMPILE_DISABLE=1 uv run --no-sync python <tiny KDA optimizer smoke>
uv run --no-sync python <compiled KDA forward/backward smoke>
uv run --no-sync python -m pytest -q tests/test_kda_cuda.py::test_chunk_bf16_matches_oracle_across_boundaries --maxfail=1
```

**Artifacts**

- `nanochat/mixers/kda.py`
- `nanochat/gpt.py`
- `nanochat/engine.py`
- `nanochat/research/{config,probe,runner}.py`

**Result**

- Added the K3 lower-bounded gate, Q/K normalization, beta sigmoid, channelwise
  delta recurrence, FP32 V-first state, causal kernel-4 Q/K/V convolutions,
  gated RMSNorm, and direct full-rank output gate without importing `ref/`.
- Added exact K/S/L dispatch, KDA-aware initialization and optimizer grouping,
  fixed-size recurrent cache cloning/reset, central cache-position advancement,
  research state accounting, candidate validation, and canonical-probe support.
- CPU/reference gate is green: 106 passed, 10 existing hardware skips, and 10
  CUDA tests deselected. Operator/layer parity is 22/22; integration is 15/15.
- Tiny KDA optimizer update passed eagerly with finite loss. The first compiled
  CPU optimizer attempt did not reach KDA math because the no-sync environment
  lacks `setuptools`; disabling compilation verified the update itself.
- A compiled CUDA execution of the reference layer completed forward/backward
  with finite output and input gradients. Full-attention and SWA/KDA cached
  decode were also manually checked against one-shot logits.
- The marked optimized CUDA test now fails explicitly with
  `NotImplementedError: optimized chunk KDA backend is not implemented`; it no
  longer fails because the KDA module is absent and does not silently fall back.

**Next**

- Integrate the FLA-derived chunk training backend and fused recurrent decode
  backend, preserving the no-fallback CUDA parity gate.
- After all marked CUDA tests pass on GB10, run the canonical memory probe before
  any discovery comparison or architectural conclusion.

## 2026-08-05 [codex] integrate FLA Triton KDA on SM121

**Context**

- Added the first production CUDA backend for the tested K3-compatible mixer.
  The goal of this step was strict parity and fail-closed execution on the DGX
  Spark, not a FlashKDA port or an architecture-quality claim.
- GB10 reports compute capability 12.1 while the installed PyTorch wheel only
  advertises compiled architectures through 12.0. Triton therefore uses the
  CUDA 13.1 system `ptxas`, which explicitly supports `sm_121a`.

**Commands**

```bash
uv lock
uv sync --extra gpu --group dev
uv run --no-sync python -m pytest -q -m "not cuda"
uv run --no-sync python -m pytest -q
uv run --no-sync research doctor --config configs/research/discovery.toml
compute-sanitizer --tool memcheck uv run --no-sync python <KDA chunk/backward and recurrent smoke>
uv run --no-sync python <KDA kernel diagnostic benchmark>
```

**Artifacts**

- `nanochat/cuda_compat.py`
- `nanochat/mixers/kda.py`
- `nanochat/{gpt,engine,checkpoint_manager}.py`
- `nanochat/research/{artifacts,config,probe,runner}.py`
- `scripts/base_train.py`
- `configs/research/{discovery,promotion}.toml`
- `tests/test_kda_cuda.py`
- `tests/test_research_harness.py`
- `pyproject.toml` and `uv.lock`

**Result**

- Pinned `fla-core==0.5.2` and integrated its Triton chunk operator for training
  and multi-token prefill plus fused recurrent KDA for single-token cached
  decode. FLA is loaded lazily after selecting the CUDA 13.1 assembler.
- The production path uses K3 lower-bounded safe gates, fused Q/K L2
  normalization and beta sigmoid, FP32 V-first recurrent state, and the local
  layer's direct full-rank output gate. FlashKDA and TileLang are deliberately
  disabled so the selected implementation is reproducible.
- Optimized execution is fail-closed: CUDA KDA research runs do not silently
  enter the sequential reference implementation. Unsupported dtype/shape and
  differentiable recurrent requests fail explicitly, while CPU remains a
  correctness/reference lane.
- Research configs, subprocesses, manifests, checkpoints, probe topology, and
  training summaries now record the requested/resolved backend, FLA version,
  assembler, and fallback policy. `research doctor` validates all hardware and
  dependency gates; `research_ready` remains false until the dirty tree is
  committed.
- Full repository validation passes: 121 passed and 10 existing skips. The 15
  CUDA cases cover boundary lengths, D=128, forward/backward parity, state
  immutability, unsupported-path rejection, no-reference fail-closed behavior,
  model prefill/decode parity, and a compiled full-model backward pass.
- Compute Sanitizer reported zero errors for chunk forward/backward and recurrent
  decode. The compiled D=128 hybrid GPT smoke produced finite outputs and
  gradients.
- Diagnostic operator measurements on this GB10 were 4.09 ms/step for
  `[8,1024,3,128]` chunk forward/backward, 5.99 ms/step for
  `[16,1024,2,128]`, and 0.0427 ms/token for `[1,1,3,128]` recurrent decode.
  These are local warmed-kernel diagnostics, not end-to-end comparisons.

**Next**

- Commit this validated backend so `research doctor` becomes research-ready.
- Run the repaired canonical memory-probe baseline and KDA candidate under the
  same protected shell, then inspect per-distance and interference metrics.
- Only after that gate, run a small matched-token discovery comparison. Treat a
  FlashKDA `sm_121a` port as a later inference optimization if recurrent decode
  profiling shows the Triton backend is material.
