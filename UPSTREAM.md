# Upstream provenance

This repository began from `karpathy/nanochat` at commit
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` and retains its MIT license.
The upstream source was promoted into this repository on 2026-08-05 so the
model, research harness, runbook, and evaluations share one history.

Local compatibility changes made before promotion changed the PyTorch wheel
source from CUDA 12.8 to CUDA 13.0 for the ARM64 NVIDIA GB10 development
system. Those changes are part of this repository's initial state.

External task generators or reference implementations added later must record
their repository, exact revision, license, and local modifications here or in
a dedicated file under `third_party/`.
