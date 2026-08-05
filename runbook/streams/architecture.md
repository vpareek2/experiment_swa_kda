# SWA + Linear Attention Architecture

## Objective

Test whether local exact attention plus recurrent KDA memory can eliminate
periodic global attention at a better quality/efficiency frontier.

## Current decisions

- PyTorch/nanochat is the implementation base; JAX is no longer a constraint.
- Baselines precede KDA: full attention, pure SWA, then KDA-only and hybrids.
- `S` has an explicit window. `force_final_full` is separately recorded so a
  pure-SWA baseline cannot silently contain a global final layer.
- First hybrid layout is layerwise composition. Within-layer mixtures wait until
  layerwise baselines are understood.
- The canonical memory probe measures load, last-write-wins updates,
  interference, and distances around the SWA boundary.

## 2026-08-05 [codex] make SWA topology explicit

**Context**

- Removed an experimental confound in inherited window semantics while keeping
  checkpoint-compatible defaults.

**Commands**

```bash
cd <repo>
uv run --no-sync python -m pytest -q tests/test_research_harness.py
```

**Artifacts**

- `nanochat/gpt.py`
- `scripts/base_train.py`
- `nanochat/research/memory.py`
- `nanochat/research/probe.py`

**Result**

- Explicit 128-token pure-SWA final-layer behavior and opt-in global-final
  behavior are covered by tests.
- KDA is not integrated yet; no KDA or hybrid performance claim exists.

**Next**

- Pin FlashKDA equations, tensor layouts, resets, and recurrent/chunk parity.
- Add a correctness-first PyTorch mixer behind the same protected probe shell.
