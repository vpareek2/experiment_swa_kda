# SWA + KDA Research Operating Contract

This repository is a PyTorch architecture-research fork of nanochat. It tests
whether sliding-window attention (SWA) plus recurrent linear attention can
replace periodic global attention while improving the quality, memory, and
throughput tradeoff. SWA always means sliding-window attention here.

The current protected harness supports full-attention and SWA baselines. KDA is
the next mixer to integrate from `ref/FlashKDA`; never describe KDA or hybrid
results as implemented until their correctness gates pass.

## Session protocol

- Read this file, then `AGENTS.md.local` if present, and `runbook/index.md`.
- State `Mode: No-Edits` for planning/review or `Mode: Execution` for changes.
- In Execution mode, inspect `git status --short` before editing.
- Name the intended file/directory scope and preserve unrelated user changes.
- Use `uv` for Python dependencies and execution.
- Never launch a costly training campaign unless the user requests it.

## Research integrity

- Configs are the source of truth. Do not smuggle experiment-defining choices
  into one-off flags or candidate code.
- A candidate may change only `nanochat/gpt.py`, `nanochat/mixers/`, and
  `configs/candidates/` unless a human explicitly expands the scope.
- Treat `nanochat/research/`, `configs/research/`, evaluation/data/tokenizer
  code, and supervisor state as protected. Candidate agents must not edit them.
- Never inspect or derive private confirmation seeds. Confirmation is launched
  and signed by a supervisor outside the candidate workspace.
- Do not optimize against held-out confirmation results. Use discovery results
  to iterate and promotion results only at declared gates.
- Report crashes, OOMs, NaNs, timeouts, and missing metrics as invalid runs—not
  as poor numeric scores and never as improvements.
- Do not claim speed, memory, stability, or quality changes without saved
  matched-run evidence from a clean commit.

## Matched comparisons

Unless the mismatch is the stated intervention, hold constant data identity
and order, tokenizer, seed set, model depth/width, parameter accounting,
training tokens or steady-state time lane, global batch, sequence length,
optimizer/schedule, precision, evaluation protocol, and hardware.

Record explicit layer pattern, SWA window, whether the final layer is forced
global, recurrent state dimensions, cache/state accounting, kernel/backend,
compile treatment, and fallbacks. Validate masking, boundaries, updates,
state resets, finite gradients, and decode/chunk equivalence where applicable.

## Supported workflow

```bash
uv run --no-sync python -m pytest -q
uv run --no-sync research doctor --config configs/research/discovery.toml
uv run --no-sync research probe --config configs/research/discovery.toml --calibrate
uv run --no-sync research probe --config configs/research/discovery.toml
uv run --no-sync research run --config configs/research/discovery.toml
uv run --no-sync research run --config configs/research/discovery.toml --candidate configs/candidates/baseline_swa.toml
uv run --no-sync research report runs/*/summary.json
```

- Discovery uses a five-minute steady-state training budget.
- Promotion uses exactly 100,663,296 training tokens per seed.
- After protected probe/config changes, commit them and register a clean-worktree
  calibration before conclusion-bearing runs. Every candidate then runs that
  calibrated canonical associative-memory protocol.
- Decisions are tolerance-aware Pareto classifications over BPB, memory recall,
  update recall, throughput, peak memory, and inference state bytes.
- Full CORE evaluation remains available through `scripts.base_eval`; it is a
  promotion evaluation, not a five-minute inner-loop metric.

## Artifacts and Git

- Generated artifacts live under ignored `runs/`; checkpoints remain under the
  nanochat cache directory. A conclusion-bearing run requires a clean commit.
- Every run must preserve resolved config, commit/worktree provenance,
  environment, protected-file hashes, logs, metric JSON, and final summary.
- Do not use `git checkout`, `restore`, `reset`, `clean`, force-push, or history
  rewriting without explicit approval.
- Do not commit datasets, checkpoints, secrets, hostnames, private paths, or
  supervisor state. `ref/` is local reference material and stays untracked.

## Runbook protocol

Update the relevant `runbook/streams/*.md` whenever state, evidence, artifacts,
constraints, or next actions change. Use `## YYYY-MM-DD [actor] title`, then
`Context`, `Commands`, `Artifacts`, `Result`, and `Next`. Include exact failures.
Prefer local evidence to dashboards and never put secrets in the runbook.

## Project map

- `nanochat/`: inherited trainer/model plus protected research harness.
- `nanochat/research/`: configs, memory probe, artifacts, decisions, supervisor.
- `configs/research/`: frozen discovery and promotion protocols.
- `configs/candidates/`: candidate-controlled architecture declarations.
- `scripts/`: inherited training/evaluation entry points.
- `tests/`: inherited and research-integrity tests.
- `recipes/nanochat/`: inherited nanochat shell recipes.
- `containers/`: network-disabled evaluation image definition.
- `runbook/`: durable human/agent project memory.
- `ref/`: ignored local reference implementations; never import at runtime.

## Autoresearch loop

Read `program.md` before autonomous experimentation. Change one primary axis,
run narrow correctness tests, commit the candidate, execute discovery, inspect
the complete artifact, and retain only a defensible Pareto improvement. A
`retest` is uncertainty, not a win. Promotion and private confirmation require
the declared seed sets and supervisor verification.
