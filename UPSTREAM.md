# Provenance

## Training substrate

This repository began from `karpathy/nanochat` at commit
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` and retains its MIT license. The
upstream source was imported on 2026-08-05 so model, trainer, correctness tests,
campaign harness, and evidence could share one Git history. Early compatibility
work selected PyTorch's CUDA 13.0 wheel for the ARM64 NVIDIA GB10 system.

## KDA references

Reference source was kept under ignored `ref/` checkouts for offline study and
comparison; it is not imported by the retained project CUDA runtime.

| Repository | Pinned revision | License | Use |
|---|---|---|---|
| `MoonshotAI/FlashKDA` | `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b` | MIT | Equations/reference implementation |
| `fla-org/flash-linear-attention` | `a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d` | MIT | Offline reference; packaged `fla-core==0.5.2` is the release comparator |
| `RishiShah99/lethe` | `e3ed0ccb5f8146f96dff8c507b915afdd696f96a` | MIT | Offline systems reference |

The machine-readable historical source record is
`runs/kda-cuda-development/reference-sources.json`. See `NOTICE` for copyright
attributions.
