# SWA + Linear Attention Runbook

Shared project memory for humans and research agents. The implementation is now
a PyTorch nanochat fork; SWA means sliding-window attention. KDA remains the next
architecture milestone, not a completed result.

## Active streams

- [Experiment operations](streams/experiment_operations.md)
- [Architecture](streams/architecture.md)
- [Evaluation and autoresearch](streams/evaluation.md)

## References

- [Experiment evidence contract](references/experiment_evidence.md)

## Archive

- [2026 archive](archive/2026/index.md)

Update a stream after any change to run status, evidence, artifacts,
constraints, or next actions. Entries require date/actor, Context, Commands,
Artifacts, Result, and Next. Never record secrets or private supervisor state.
