# Project CUDA Candidate ABI

Autoresearch candidates may change files only in this directory. Protected code
owns routing, FLA use for unclaimed transitional units, correctness, ownership
weights, benchmarks, and decisions.

## Protected operator entry points

`prepare()` builds and loads the reviewed extension from `TORCH_EXTENSIONS_DIR`.
Build names must be content-addressed so different commits cannot share a stale
binary. Candidate Python never computes KDA or convolution results: protected
routing resolves the claimed `nanochat_kda::` operator and consumes its return
values directly.

The chunk and recurrent forward operators use this positional contract:

```text
(q, k, v, raw_gate, beta_logits, A_log, dt_bias,
 initial_state: Tensor?, output_final_state: bool,
 lower_bound: float, resolved_scale: float) -> (output, final_state: Tensor?)
```

The causal-convolution forward operator uses:

```text
(x, weight, initial_state: Tensor?, output_final_state: bool)
    -> (output, final_state: Tensor?)
```

Register CUDA implementations from the loaded native extension, not with a
Python `torch.library` callback. Chunk and convolution autograd callbacks must
invoke their separately declared native backward operators. Protected auditing
observes all five distinct operators and their native kernel symbols.

## Atomic units

- `chunk_forward` and `chunk_backward` migrate together.
- `recurrent_decode` migrates alone.
- `causal_convolution_forward` and `causal_convolution_backward` migrate together.

## Provenance schema

`provenance()` returns a stable dictionary after `prepare()`:

```python
{
  "schema_version": 1,
  "components": {
    "chunk_forward": {
      "owner": "project",             # or "third_party"
      "sources": ["nanochat/mixers/cuda_kda/kda.cu"],
      "kernel_symbols": ["nanochat_kda_chunk_fwd"],
      "torch_operator": "nanochat_kda::chunk_forward",
    },
    # every frozen component is present
  },
  "build": {
    "library_path": "/.../isolated-cache/...so",
    "source_paths": ["nanochat/mixers/cuda_kda/kda.cu"],
    "compiler_command": "<exact command/flags>",
    "target_arch": "sm_121",
  },
  "selective_ptx": [],
}
```

For `owner="third_party"`, `sources` and `kernel_symbols` are empty and
`torch_operator` is `None`. Every project claim needs tracked `.cu/.cuh`, a
registered `nanochat_kda::` CUDA dispatch kernel, and symbols observable in the
bounded profile. All project claims share one loaded-library build receipt.

## Selective PTX

Bootstrap forbids PTX. Later inline or tracked PTX must declare exactly:

```python
{
  "source": "nanochat/mixers/cuda_kda/kda.cu",
  "rationale": "...",
  "architecture_guard": "...",
  "cuda_fallback": "...",
  "profile_evidence": "...",
}
```

The ordinary CUDA path must run correctly with
`NANOCHAT_DISABLE_SELECTIVE_PTX=1`, and enabled PTX must clear the protected 2%
latency materiality gate. `tcgen05` and TMEM are unavailable on SM121.
