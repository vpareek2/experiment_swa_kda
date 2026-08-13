# KDA Autoresearch

An evidence-first campaign to optimize **numerically exact Kimi Delta Attention
(KDA) training** on NVIDIA GB10. The retained implementation is a project-owned
CUDA backend integrated into a small nanochat training model; it does not freeze
parameters, substitute a value-only backward, or change the KDA equations.

## Result

On the pinned GB10 lane, exact project KDA reached **up to 45,058 tokens/s**.
Across three matched runs, its median was **44,942 tokens/s**, versus **43,937
tokens/s** for `fla-core==0.5.2` (+2.29%).

| Backend | Run medians (tok/s) | Three-run median | Peak allocation |
|---|---:|---:|---:|
| Project CUDA | 45,058 · 44,942 · 44,842 | **44,942** | 5,743 MiB |
| FLA Triton | 43,958 · 43,898 · 43,937 | **43,937** | 5,550 MiB |

The 45,058 figure is the strongest observed **run median**, not the aggregate.
All six runs used eager BF16, B2/T4096/H3/D128, six KDA layers, four gradient
accumulation steps, seed 42, and the same model/data/optimizer settings. See
[the benchmark contract](docs/BENCHMARK.md) and the tracked
[machine-readable result](results/gb10-exact-vs-fla.json).

## What “exact” means here

“Exact” means the optimization retains the full training computation and its
gradients. The project CUDA path passed independent-oracle forward/gradient
checks, accumulated-gradient and optimizer-equivalence checks, boundary cases,
and CUDA memcheck/racecheck/synccheck/initcheck gates. FLA is the performance
comparator, not the numerical oracle. See [numerical integrity](docs/NUMERICAL_INTEGRITY.md).

Two faster-looking experiments are intentionally *not* results: a local
value-only path reached 48k–50k tokens/s by freezing KDA parameters and replacing
their backward computation. That changes training, so it was rejected. Failed,
neutral, and invalid attempts remain in the campaign record instead of being
silently discarded.

## Repository guide

- `nanochat/mixers/kda.py`: KDA layer, routing, and public state semantics.
- `nanochat/mixers/cuda_kda/`: retained native CUDA implementation.
- `tests/kda_oracle.py`: independent mathematical oracle.
- `tests/test_kda_*.py`: correctness, integration, and backend tests.
- `configs/research/kda_cuda_ownership.toml`: frozen GB10 campaign contract.
- `scripts/kda_cuda_development.py`: staged candidate measurement harness.
- `results/`: curated release evidence and the normalized attempt table.
- `runbook/streams/kda_cuda_ownership.md`: append-only detailed lab record.

The repository remains a nanochat fork because the claim is training throughput,
not an isolated kernel number. The inherited trainer supplies real projection,
optimizer, accumulation, data-loading, and model costs around six KDA layers.

## Setup and verification

The published performance lane is NVIDIA GB10 (SM121), PyTorch 2.9.1 + CUDA
13.0, Python 3.10, and `fla-core==0.5.2`. It is a hardware-specialized research
artifact, not a portable KDA library.

```bash
uv sync --extra gpu --group dev
uv run --no-sync python -m pytest -q
```

Run the focused correctness suite:

```bash
uv run --no-sync python -m pytest -q \
  tests/test_kda_operator.py tests/test_kda_layer.py \
  tests/test_kda_cuda.py tests/test_kda_integration.py
```

The exact trainer invocation is documented in [docs/BENCHMARK.md](docs/BENCHMARK.md).
It downloads/uses nanochat training data and performs GPU work, so it is not run
automatically by the test suite.

## Campaign

The optimization loop recorded 366 ledger events covering kernel fusion,
launch reduction, layouts, scheduling, CUDA Graphs, persistent/cluster kernels,
and deliberately rejected surrogates. [docs/CAMPAIGN.md](docs/CAMPAIGN.md)
explains how to read the record; [results/attempts.csv](results/attempts.csv)
is the normalized public index.

This release preserves the original research history at the annotated tag
`archive/swa-kda-research-20260813`. The default branch is intentionally focused
on the exact-KDA systems campaign.

## Provenance and license

The training substrate began from
[`karpathy/nanochat`](https://github.com/karpathy/nanochat) at commit
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`. External references and pinned
revisions are listed in [UPSTREAM.md](UPSTREAM.md) and [NOTICE](NOTICE). The
repository is MIT licensed.
