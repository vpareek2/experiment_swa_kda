# Curated results

This directory contains small, reviewable release evidence. It deliberately
excludes checkpoints, compiler caches, raw profiler databases, and full logs.

- `gb10-exact-vs-fla.json` is the canonical final matched comparison.
- `attempts.csv` normalizes the append-only development ledger for analysis.
- `campaign-summary.json` reports ledger coverage and status counts.

`attempts.csv` and `campaign-summary.json` are deterministic derivatives of
`runs/kda-cuda-development/attempt-index.jsonl`. Check them with:

```bash
uv run --no-sync python scripts/export_kda_results.py --check
```

Artifact paths in the ledger refer to ignored local evidence directories. They
serve as stable provenance pointers; bulky contents are not promised in Git.
