# GB10 release benchmark

This is the frozen contract behind the release throughput claim. It measures a
real seven-step training loop around six KDA layers, not an isolated kernel.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GB10, compute capability 12.1 |
| PyTorch / CUDA | 2.9.1+cu130 / 13.0 |
| Project backend | `project_cuda` |
| Comparator | `fla-core==0.5.2`, `fla_triton` |
| Precision / execution | BF16 / eager |
| Model | depth 6, width 384, 3 heads, head dimension 128 |
| Mixer pattern | `KKKKKK`; final layer is not forced global |
| Workload | device batch 2, sequence 4096, global batch 32,768 tokens |
| Accumulation | 4 microsteps per optimizer update |
| Seed | 42 |

`FLA_FLASH_KDA=0`, `FLA_TILELANG=0`, and backend fallback is forbidden. Each
run has seven updates. Steps 0–1 are warm-up; the run score is the median
tokens/s over steps 2–6. The release score is the median of three run medians.

The recorded same-session order was:

```text
project-1, fla-1, fla-2, project-2, project-3, fla-3
```

That order interleaves implementations and reverses the order within the middle
pair to expose simple thermal/time drift. All three positional pairs favored the
project backend.

## Trainer invocation

After installing the GPU environment and preparing nanochat's training data,
run the following once per backend and give each run a unique `--run` and
`--model-tag`:

```bash
FLA_FLASH_KDA=0 FLA_TILELANG=0 NANOCHAT_DTYPE=bfloat16 \
uv run --no-sync python -m scripts.base_train \
  --seed 42 --depth 6 --head-dim 128 --window-pattern K \
  --kda-backend project_cuda --no-force-final-full \
  --max-seq-len 4096 --device-batch-size 2 --total-batch-size 32768 \
  --num-iterations 7 --eval-every -1 --core-metric-every -1 \
  --sample-every -1 --save-every -1
```

Change only `--kda-backend project_cuda` to `--kda-backend fla_triton` for the
comparator. Do not insert an extra `--` before the options; the current parser
rejects it. Use fixed, separate extension/Triton caches for warm build and
scored execution, and exclude one-time compilation from scoring.

## Recorded result

| Metric | Project CUDA | FLA Triton |
|---|---:|---:|
| Run medians, tok/s | 45,058; 44,942; 44,842 | 43,958; 43,898; 43,937 |
| Median of run medians | **44,942** | **43,937** |
| Median step time | 729.112 ms | 745.794 ms |
| Peak allocation | 5,743.093 MiB | 5,550.471 MiB |

Aggregate gain is +1,005 tok/s (+2.287%). Positional-pair gains are +1,100,
+1,044, and +905 tok/s. The strongest observed exact run median is 45,058
tok/s; it must not be described as the three-run aggregate.

The canonical machine-readable record is
[`results/gb10-exact-vs-fla.json`](../results/gb10-exact-vs-fla.json). It includes
all scored steps, backend resolution, memory, and log hashes. Raw logs and
profiler traces are intentionally excluded from Git because they are bulky;
their hashes and the append-only runbook preserve the evidence chain.

## Scope of the claim

This is a pinned single-system comparison. It does not claim portability,
multi-GPU scaling, quality superiority, lower memory, or a universal advantage
over FLA. No quality evaluation was run for this final throughput confirmation;
mathematical equivalence is supported by the separate correctness gates.
