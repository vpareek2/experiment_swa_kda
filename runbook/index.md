# Exact KDA Autoresearch Runbook

Durable lab memory for the GB10 exact-KDA training-throughput campaign. The
retained project-owned CUDA backend reaches up to 45,058 tok/s; its confirmed
three-run median is 44,942 tok/s versus 43,937 tok/s for pinned FLA.

## Active stream

- [KDA CUDA ownership and optimization](streams/kda_cuda_ownership.md)

## Supporting history

- [Earlier KDA training-speed loop](streams/kda_training_speed.md)
- [Experiment evidence contract](references/experiment_evidence.md)
- [DGX Spark / GB10 architecture notes](references/dgx_spark_gb10_architecture.md)
- [GB10-guided KDA optimization plan](references/gb10_kda_hardware_guided_plan.md)

SWA and general architecture-research history is preserved at the annotated
tag `archive/swa-kda-research-20260813`; it is not part of the active release
story.

Update the KDA stream with `## YYYY-MM-DD [actor] title`, followed by `Context`,
`Commands`, `Artifacts`, `Result`, and `Next`. Include exact failures and never
record secrets, private confirmation state, or machine-private paths.
