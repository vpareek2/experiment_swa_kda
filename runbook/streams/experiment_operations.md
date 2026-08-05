# Experiment Operations

## Objective

Maintain one reproducible PyTorch/uv path for setup, training, evaluation,
artifact capture, and comparison on a single DGX Spark.

## 2026-08-05 [codex] promote nanochat and add research runner

**Context**

- Promoted the cloned nanochat tree to repository root and initialized root Git.
- Preserved the upstream revision in `UPSTREAM.md`; `ref/` remains ignored.
- Added declarative discovery/promotion configs, structured trainer/evaluator
  results, artifact manifests, and supervisor commands.

**Commands**

```bash
cd <repo>
uv run --no-sync python -m pytest -q tests/test_research_harness.py
uv run --no-sync python -m pytest -q
uv run --no-sync research doctor --config configs/research/discovery.toml
```

**Artifacts**

- `UPSTREAM.md`
- `configs/research/{discovery,promotion}.toml`
- `nanochat/research/`
- `containers/research.Dockerfile`
- `tests/test_research_harness.py`

**Result**

- Research-harness tests: 16 passed with the known GB10 capability warning.
- Full suite: 64 passed, 10 skipped. Environment checks passed; research
  readiness is false only because the new repository has no clean commit yet.
- A pure-SWA fixed-time smoke stopped on budget at 0.0507 measured seconds,
  trained 19,456 tokens, recorded windows `[[64, 0], [64, 0]]`, and emitted a
  structured result. This was a plumbing check, not comparative evidence.
- Discovery is 300 seconds of measured steady-state training; promotion is
  100,663,296 tokens. No costly campaign was launched during implementation.

**Next**

- Create the initial clean commit before any conclusion-bearing run.
- Calibrate decision tolerances with the declared five baseline seeds.
- Build and exercise the restricted container before private confirmation.
