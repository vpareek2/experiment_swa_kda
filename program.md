# Exact KDA Autoresearch Program

The objective is the highest end-to-end KDA training throughput that preserves
the complete KDA forward, backward, parameter updates, and numerical behavior.
The fixed release workload and aggregation are defined in `docs/BENCHMARK.md`.

## Optimization loop

1. Read `AGENTS.md`, the latest KDA runbook entries, the normalized attempt
   table, and the retained implementation before proposing work.
2. Identify one profile-supported mechanism. State the expected saved time at
   layer and trainer scope and a falsifiable phase gate.
3. Create an isolated candidate from the retained exact baseline. Change one
   primary scheduling, layout, fusion, or dataflow axis.
4. Run focused oracle, boundary, random-upstream gradient, and accumulation
   checks. A crash, timeout, sanitizer finding, fallback, or missing metric is
   invalid. Any mathematical or optimizer divergence rejects the candidate.
5. Commit the exact candidate and run production-shape Level 1 measurements.
   Treat small effects within run-to-run noise as neutral.
6. Only after a material Level 1 result, run matched end-to-end trainers in an
   interleaved order. Preserve all runs and use the declared aggregate.
7. Retain a candidate only when exactness and end-to-end improvement both hold.
   Never retain a surrogate merely because its throughput is higher.
8. Append the complete outcome to the ledger and KDA runbook, including negative
   results and invalid invocations.

The detailed historical supervisor protocol remains in
`program_kda_cuda_ownership.md`. It explains ownership auditing, native ABI,
sanitizers, staged gates, and artifact capture. Current development starts from
the exact retained `main` implementation; it does not restart the old migration
or optimize against its fixed historical FLA target.
