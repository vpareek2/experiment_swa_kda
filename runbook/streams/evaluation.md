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

## 2026-08-05 [codex] replace invalid memory probe with calibrated v2

**Context**

- V1 spent about one million context tokens but supervised only 4,096 answer
  tokens. Full attention and SWA both stayed near chance, so its memory scores
  could not discriminate architectures.
- Initial v2 trials exposed seed-sensitive undertraining. The final instrument
  fixes initialization and training data/order, then uses three independent
  held-out evaluation streams. This removes an optimization lottery while
  retaining measurement-uncertainty checks.

**Commands**

```bash
cd <repo>
uv run --no-sync python -m pytest -q
# Direct dirty-worktree validation used the same run_probe_calibration routine.
# After committing protected changes, register it with:
uv run --no-sync research probe --config configs/research/discovery.toml --calibrate
```

**Artifacts**

- `nanochat/research/{config,memory,probe,runner,cli,decision}.py`
- `configs/research/{discovery,promotion}.toml`
- `tests/test_research_harness.py`

**Result**

- V2 trains on exactly 544,768 supervised answer tokens across four declared
  curriculum stages. It evaluates load, last-write updates, interference,
  255/256/257-token boundary behavior, and lengths through 2,048 with Wilson
  intervals. Oracle and fixed-batch overfit controls must pass.
- Direct three-seed calibration passed every gate. Full attention achieved
  easy accuracy 0.9922–0.9930, memory AUC 0.9756–0.9789, and update accuracy
  0.7324–0.7715 in about 68.4 seconds. Pure SWA achieved easy accuracy
  0.9957–0.9965 but memory AUC only 0.5567–0.5650 in about 74.3 seconds; its
  known long-range limitation was detected. No threshold was relaxed.
- The runner rejects absent, invalid, or stale calibrations using both protocol
  and protected-file hashes. V1 summaries cannot enter the v2 frontier or seed
  aggregates and are labeled `legacy/ineligible` in reports.
- Full suite: 69 passed, 10 skipped, with only the known GB10 capability warning.

**Next**

- Commit the protected v2 implementation, run the official calibration command
  from that clean commit, then rerun matched full-attention and pure-SWA baselines.
- Only after those baselines enter the v2 frontier should KDA integration begin.
