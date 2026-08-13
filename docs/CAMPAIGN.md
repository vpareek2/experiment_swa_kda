# Reading the campaign

This repository is code evidence for an autoresearch campaign, including the
ideas that failed. The detailed runbook is intentionally long; the files under
`results/` provide a compact entry point.

## Evidence layers

1. **Correctness gate:** independent oracle, random upstream gradients,
   boundaries, routing, and sanitizers.
2. **Level 1:** production-shape operator/layer timing and profiler evidence.
3. **Level 2:** matched seven-step training blocks with a fixed aggregation.
4. **Retention:** only exact, material end-to-end improvements are merged into
   the baseline. A fast microbenchmark alone is not a win.

Every attempt ledger event has a stable `loop_id`, status, candidate/parent
commit when applicable, artifact pointer, and the evidence available at that
stage. Statuses are descriptive rather than a single pass/fail enum because the
campaign includes design rejections, invalid invocations, profile-only probes,
retests, accepted foundations, and later corrections.

## Important milestones

- The project first integrated a correctness-first KDA layer and independent
  oracle, then replaced runtime FLA components with project-owned CUDA.
- Attempt 266 was an accepted performance baseline.
- Attempt 338 carried the main performance improvement used by the final
  implementation.
- Attempt 342 completed the audit/profile-symbol work and exceeded the campaign's
  historical fixed target with a 43,840 tok/s three-run median in that lane.
- Later exact work tested projection fusion, cache reuse, CUDA Graphs, stream
  overlap, producer megakernels, and CTA clusters. Positive microbench effects
  were neutral or negative in the trainer and were rejected.
- A fresh release confirmation of the retained exact source produced a 44,942
  tok/s three-run median and one 45,058 tok/s run median against 43,937 tok/s FLA.

The changing absolute values across historical entries reflect different
campaign epochs, machine state, and matched baselines. Compare numbers only
inside their declared block; do not combine the best number from every epoch.

## Files

- `results/attempts.csv`: one normalized row per ledger event.
- `results/campaign-summary.json`: counts by status and evidence availability.
- `results/gb10-exact-vs-fla.json`: final matched release comparison.
- `runs/kda-cuda-development/attempt-index.jsonl`: source ledger.
- `runbook/streams/kda_cuda_ownership.md`: detailed human-readable record.

Regenerate the curated tables with:

```bash
uv run --no-sync python scripts/export_kda_results.py --check
```

Without `--check`, the command rewrites the two derived campaign files.
