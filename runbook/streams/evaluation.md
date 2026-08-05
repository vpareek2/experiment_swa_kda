# Evaluation and Autoresearch

## Objective

Provide a deterministic, auditable suite usable by human researchers and
autonomous candidate agents without letting candidates rewrite their exam.

## 2026-08-05 [codex] implement protected v1 harness

**Context**

- Added strict TOML configs, controlled associative-memory generation, a
  canonical probe model, machine-readable artifacts, tolerance-aware Pareto
  decisions, protected-file manifests, HMAC result signing, and a hardened
  network-disabled Docker command generator.

**Commands**

```bash
cd <repo>
uv run --no-sync python -m pytest -q tests/test_research_harness.py
uv run --no-sync research doctor --config configs/research/discovery.toml
```

**Artifacts**

- `nanochat/research/`
- `configs/research/`
- `configs/candidates/{baseline_full,baseline_swa}.toml`
- `program.md`
- `containers/research.Dockerfile`

**Result**

- Determinism, exact target distance, masked labels, update semantics, Pareto
  classification, atomic artifacts, tamper detection, signing, and one actual
  probe training step have automated coverage.
- Candidate TOML can change only the named architecture fields; budget, probe,
  decision, protection, data, and evaluation settings come from the frozen
  protocol. Seed campaigns emit mean, standard deviation, 95% intervals, and
  calibrated noise floors.
- CORE and BPB are inherited and now emit structured JSON. RULER, MAD, and
  Zoology task ports are not yet vendored; they remain promotion-suite work.
- The supervisor is tamper-evident and narrows candidate changes. It is not a
  proof against a process with host access; private confirmation must run in a
  separately controlled environment.
- The local protected manifest currently verifies with no changed files. The
  restricted Docker command was generated, but the image was not built or run.

**Next**

- Calibrate metric noise from five baseline seeds before accepting candidates.
- Add long-context RULER subsets and provenance-pinned external task adapters.
- Run internal promotion on three seeds and supervisor-held confirmation on five.
