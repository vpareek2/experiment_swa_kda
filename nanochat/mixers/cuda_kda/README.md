# Project CUDA Candidate Boundary

> **READ-ONLY FOR CANDIDATE WORKERS.** Read this file, but do not edit or stage
> it. Candidate commits may change the other allowed source files in this
> directory only. If this guide is wrong, report the mismatch to the coordinator
> instead of repairing protected onboarding documentation in a candidate.

Protected code owns routing, transitional FLA use, correctness, ownership
weights, timeouts, benchmarks, and decisions. Candidate code owns only native
CUDA implementation, minimal native-extension loading/registration, and honest
provenance declarations.

## Recommended bootstrap

Implement only the simplest correct naive `recurrent_decode` native-CUDA unit.
Do not optimize, use PTX, or claim chunk/convolution ownership. This forward-only
unit is the smallest complete atomic unit and has 20% frozen weight, enough for
the bootstrap floor. Protected routing uses FLA for every explicitly unclaimed
unit during bootstrap and migration.

The current recurrence and public-state rules are documented in
`program_kda_cuda_ownership.md`. The authoritative read-only sources are:

- `tests/kda_oracle.py` for independent math and V-first public state;
- `nanochat/mixers/kda.py` for live dispatch/return conventions;
- `nanochat/research/cuda_worker.py` for audit shapes and evidence;
- `configs/research/kda_cuda_ownership.toml` for frozen limits.

Do not import or call those sources from candidate runtime code.

## Frozen native operator ABI

Use protected `nanochat.research.cuda_build.build_cuda_extension` from
`prepare()`. Pass tracked `.cu`/`.cpp` sources, keep the returned receipt, and
load only into the provided isolated `TORCH_EXTENSIONS_DIR`. The helper derives
a content-addressed name, forces `-gencode=arch=compute_121,code=sm_121`, loads
the library, verifies its process mapping and cubin, and records the actual
compiler command. Do not create another environment or write build products in
the worktree.

The operator namespace, names, argument order, and return order below are
**frozen**. Register schemas and CUDA implementations in native C++/CUDA with
`TORCH_LIBRARY` and `TORCH_LIBRARY_IMPL`. Do not use Python `torch.library`.

```text
nanochat_kda::recurrent_decode(
  Tensor q, Tensor k, Tensor v, Tensor raw_gate, Tensor beta_logits,
  Tensor A_log, Tensor dt_bias, Tensor? initial_state,
  bool output_final_state, float lower_bound, float scale
) -> (Tensor output, Tensor? final_state)

nanochat_kda::chunk_forward(
  Tensor q, Tensor k, Tensor v, Tensor raw_gate, Tensor beta_logits,
  Tensor A_log, Tensor dt_bias, Tensor? initial_state,
  bool output_final_state, float lower_bound, float scale
) -> (Tensor output, Tensor? final_state)

nanochat_kda::chunk_backward(
  Tensor q, Tensor k, Tensor v, Tensor raw_gate, Tensor beta_logits,
  Tensor A_log, Tensor dt_bias, Tensor? initial_state,
  Tensor output, Tensor? final_state, Tensor grad_output,
  Tensor? grad_final_state, float lower_bound, float scale
) -> (Tensor dq, Tensor dk, Tensor dv, Tensor draw_gate,
      Tensor dbeta_logits, Tensor dA_log, Tensor ddt_bias,
      Tensor? dinitial_state)

nanochat_kda::causal_convolution_forward(
  Tensor x, Tensor weight, Tensor? initial_state, bool output_final_state
) -> (Tensor output, Tensor? final_state)

nanochat_kda::causal_convolution_backward(
  Tensor x, Tensor weight, Tensor? initial_state, Tensor output,
  Tensor? final_state, Tensor grad_output, Tensor? grad_final_state
) -> (Tensor dx, Tensor dweight, Tensor? dinitial_state)
```

KDA tensors are contiguous CUDA tensors. `q`, `k`, `raw_gate` have shape
`[B,T,H,K]`; `v` and output have `[B,T,H,V]`; `beta_logits` has `[B,T,H]`;
`A_log` is float32 `[H]`; `dt_bias` is float32 with `H*K` elements; activation
inputs/outputs are bfloat16. State is non-mutating float32 **V-first**
`[B,H,V,K]`. `recurrent_decode` is inference-only and is called with `T=1`.
When `output_final_state` is false, return `None`; otherwise return the updated
state. `scale` is a resolved finite float; protected routing converts a public `None` to `K**-0.5` before dispatch. The math is frozen in the source map above.

Convolution `x`/output are bfloat16 `[B,T,C]`, `weight` is bfloat16 `[C,W]`,
and state is bfloat16 `[B,C,W]`. It is depthwise causal cross-correlation
followed by SiLU; state is the newest width-`W` input suffix and must not mutate
the input cache. All gradient results match the corresponding input shapes and
dtypes. A `None` input state has a `None` state gradient.

Chunk forward/backward and convolution forward/backward are atomic claims. Each
claimed component needs its own operator and native kernel. Protected routing
consumes the returned values directly and independently observes forward and
backward execution; dummy returns or hidden computation by another backend are
invalid.

## Provenance declaration

After `prepare()`, `provenance()` returns all five components. The following is a
shape template whose path, symbols, operator names, and build fields must be
replaced with truthful observed values:

```python
{
    "schema_version": 1,
    "components": {
        "recurrent_decode": {
            "owner": "project",
            "sources": ["nanochat/mixers/cuda_kda/<source>.cu"],
            "kernel_symbols": ["<profile-visible-kernel-symbol>"],
            "torch_operator": "nanochat_kda::<registered-operator>",
        },
        # Include all remaining frozen components. Unclaimed entries use:
        # {"owner": "third_party", "sources": [],
        #  "kernel_symbols": [], "torch_operator": None}
    },
    "build": {
        "library_path": "<absolute isolated-cache .so path>",
        "source_paths": ["nanochat/mixers/cuda_kda/<source>.cu"],
        "compiler_command": "<actual command and flags>",
        "target_arch": "sm_121",
    },
    "selective_ptx": [],
}
```

All project claims share the one loaded-library receipt. Every declared source
must be tracked under this directory, every project claim must include `.cu` or
`.cuh`, the library must be mapped from the isolated extension cache, and each
operator and kernel symbol must be observed by the protected audit/profile.

## Forbidden lexical tokens

The checker lowercases comment-free candidate source and rejects these exact
substrings in executable text or strings:

```text
import fla
from fla
_run_fla
_fla_ops
_fla_causal_conv1d
_reference_kda
tests.kda_oracle
/ref/
tcgen05
tmem
```

Python files additionally may not contain `torch.library` or
`from torch import library`. Candidate commits also reject generated binary
suffixes, symlinks/submodules, unsupported source suffixes, and changes outside
this directory. Bootstrap forbids all PTX.

## Ledger-free staged check

The candidate worktree has no independent environment. Stage the exact intended
source snapshot, ensure there are no unstaged/untracked files, then ask the
coordinator to run its interpreter and current protected worker:

```bash
# Run from the coordinator, never from this candidate worktree.
uv run --no-sync research cuda-candidate-check --worktree /absolute/path/to/candidate-worktree --lane bootstrap
```

Use `--sanitizers` for the four Compute Sanitizer tools. The check builds inside
isolated `/tmp` caches by default and writes a diagnostic `summary.json`; it
does not touch the campaign ledger. A pass does not replace committed intake or
the mandatory supervisor run.

## Selective PTX after bootstrap

Later inline or tracked PTX requires a truthful declaration shaped as:

```python
{
    "source": "nanochat/mixers/cuda_kda/<source>.cu",
    "rationale": "<profile-identified bottleneck>",
    "architecture_guard": "<SM guard>",
    "cuda_fallback": "<ordinary CUDA path>",
    "profile_evidence": "<same-commit measured evidence>",
}
```

The ordinary CUDA fallback must pass with
`NANOCHAT_DISABLE_SELECTIVE_PTX=1`; enabled PTX must clear the protected 2%
latency materiality gate. Unsupported SM121 instruction families remain
forbidden.
