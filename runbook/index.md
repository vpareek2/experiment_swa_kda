# SWA + Linear Attention Runbook

Shared project memory for humans and research agents. The implementation is now
a PyTorch nanochat fork; SWA means sliding-window attention. The correctness-first
K3 KDA mixer and its FLA Triton backend are integrated on GB10/SM121. Controlled
memory-probe and discovery experiments remain outstanding, so no architecture
quality result is claimed yet. The CUDA-ownership performance stream currently
retains attempt255 at 41,922 tok/s against the fixed 43,680 tok/s FLA reference.

## Active streams

- [Experiment operations](streams/experiment_operations.md)
- [Architecture](streams/architecture.md)
- [Evaluation and autoresearch](streams/evaluation.md)
- [KDA training-speed autoresearch](streams/kda_training_speed.md)
- [KDA CUDA-ownership autoresearch](streams/kda_cuda_ownership.md)

## References

- [Experiment evidence contract](references/experiment_evidence.md)

## Archive

- [2026 archive](archive/2026/index.md)

Update a stream after any change to run status, evidence, artifacts,
constraints, or next actions. Entries require date/actor, Context, Commands,
Artifacts, Result, and Next. Never record secrets or private supervisor state.
