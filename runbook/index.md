# SWA + Linear Attention Runbook

Shared project memory for humans and research agents. The implementation is now
a PyTorch nanochat fork; SWA means sliding-window attention. The correctness-first
K3 KDA mixer and audited project-owned CUDA backend are integrated on GB10/SM121.
Attempt342 reached 43,840 tok/s against the fixed 43,680 tok/s FLA reference and
is now on `main`. Controlled hybrid discovery and promotion remain outstanding,
so no architecture-quality result is claimed yet.

## Active streams

- [Experiment operations](streams/experiment_operations.md)
- [Architecture](streams/architecture.md)
- [Evaluation and autoresearch](streams/evaluation.md)
- [KDA training-speed autoresearch](streams/kda_training_speed.md)
- [KDA CUDA-ownership autoresearch](streams/kda_cuda_ownership.md)

## References

- [Experiment evidence contract](references/experiment_evidence.md)
- [DGX Spark / GB10 architecture notes](references/dgx_spark_gb10_architecture.md)

## Archive

- [2026 archive](archive/2026/index.md)

Update a stream after any change to run status, evidence, artifacts,
constraints, or next actions. Entries require date/actor, Context, Commands,
Artifacts, Result, and Next. Never record secrets or private supervisor state.
