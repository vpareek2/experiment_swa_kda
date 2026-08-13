# Project CUDA KDA backend

This directory contains the retained, complete project-owned CUDA backend for
KDA. It targets the release lane's SM121 production shape while preserving
fallbacks for correctness coverage. Runtime routing and the public KDA API live
in `nanochat/mixers/kda.py`.

## Source map

| Source | Responsibility |
|---|---|
| `chunk.cu` | Torch operator registration, validation, dispatch, compatibility paths |
| `chunk_wy_forward.cu` | optimized chunk/WY forward recurrence |
| `chunk_wy_backward.cu` | optimized exact chunk/WY backward |
| `chunk_wy_common.cuh` | shared recurrence/layout utilities |
| `recurrent_decode.cu` | single-token recurrent inference |
| `causal_convolution_forward.cu` | depthwise causal convolution + SiLU forward |
| `causal_convolution_backward.cu` | exact convolution/input/weight backward |
| `rmsnorm_gate.cu` | fused KDA RMSNorm/gate operator used by routing |
| `__init__.py` | content-addressed build and audited provenance declaration |

The five ownership-gated components—chunk forward/backward, recurrent decode,
and causal convolution forward/backward—are all implemented by project native
CUDA. The runtime does not import FLA for `project_cuda` execution.

## Tensor and state conventions

KDA activation tensors are contiguous BF16. `q`, `k`, and raw gate use
`[B,T,H,K]`; `v` and output use `[B,T,H,V]`; beta logits use `[B,T,H]`.
`A_log` and `dt_bias` are FP32. The non-mutating recurrent state is FP32 and
V-first: `[B,H,V,K]`.

Convolution input/output is BF16 `[B,T,C]`, weights are `[C,W]`, and cached
state is `[B,C,W]`. The operation is depthwise causal cross-correlation followed
by SiLU. Backward returns gradients for inputs, parameters, and input state.

The authoritative mathematical reference is `tests/kda_oracle.py`; kernel ABI
and production gates are frozen in
`configs/research/kda_cuda_ownership.toml`.

The most training-sensitive frozen operator is:

```text
nanochat_kda::chunk_backward(
  Tensor q, Tensor k, Tensor v, Tensor raw_gate, Tensor beta_logits,
  Tensor A_log, Tensor dt_bias, Tensor? initial_state,
  Tensor output, Tensor? final_state, Tensor grad_output,
  Tensor? grad_final_state, float lower_bound, float scale
) -> (Tensor dq, Tensor dk, Tensor dv, Tensor draw_gate,
      Tensor dbeta_logits, Tensor dA_log, Tensor ddt_bias,
      Tensor? dinitial_state)
```

The other registered operators are `chunk_forward`, `recurrent_decode`,
`causal_convolution_forward`, and `causal_convolution_backward` in the
`nanochat_kda` namespace. Their schemas are enforced by the native registration
and supervisor audit.

## Selective PTX

The production forward and backward recurrence paths use documented
`mma.sync.m16n8k16`/`ldmatrix` instructions behind architecture guards. Setting
`NANOCHAT_DISABLE_SELECTIVE_PTX=1` builds the exact standard-CUDA WMMA fallback.
The provenance record in `__init__.py` names the evidence artifacts and build
fallback for both selective-PTX regions.

## Verification

```bash
uv run --no-sync python -m pytest -q \
  tests/test_kda_operator.py tests/test_kda_layer.py \
  tests/test_kda_cuda.py tests/test_kda_integration.py
```

CUDA Compute Sanitizer gates are part of the campaign supervisor rather than
the default test command. See `docs/NUMERICAL_INTEGRITY.md` and the KDA runbook
for the recorded audit.
