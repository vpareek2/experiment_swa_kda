# KDA CUDA-Ownership Autoresearch Program

## Purpose

Show, with immutable evidence, how autonomous research moves KDA from its
sequential PyTorch oracle through a naive native-CUDA bootstrap to an optimized,
FLA-free, project-owned backend. The end state is all five frozen KDA
capabilities project-owned, with selective PTX only after a measured CUDA-path
A/B.

This is backend systems research. It does not evaluate or claim language-model
quality. Read `AGENTS.md`, this file,
`runbook/streams/kda_cuda_ownership.md`, and the latest coordinator-produced
summary before working. The completed speed campaign is read-only historical
context and is never arithmetically combined with this campaign.

## Roles and source ownership

The coordinator owns the protected interpreter, CLI, worker, ledger,
retained-milestone head, worktrees, evidence, and runbook. A candidate child
owns exactly one committed change in its assigned worktree and edits only
`nanochat/mixers/cuda_kda/`. The supervisor derives the lane from the retained
head; within that lane the model chooses its own mechanistic hypothesis.

The coordinator tree must remain clean for conclusion-bearing work. Candidates
must not edit protected routing, workers, tests, config, evaluation code, or the
runbook. They may read `nanochat/mixers/cuda_kda/README.md`, but **must not edit
or stage that README**. It is protected onboarding documentation despite living
inside the candidate directory.

The recommended first bootstrap task is deliberately narrow:

> Implement the simplest correct naive native-CUDA `recurrent_decode` atomic
> unit only. Optimize nothing, use no PTX, and leave every other component
> explicitly third-party.

This recurrent-only milestone has the smallest forward-only surface and still
meets the 20% bootstrap ownership floor. Protected routing supplies FLA only for
unclaimed units. The model writes the CUDA and minimal loading/registration
code and native operator implementation; protected code supplies only the
reproducible build helper and frozen ABI.

## Candidate source contract

Allowed candidate artifacts are tracked `.py`, `.cu`, `.cuh`, `.cpp`, `.cc`,
`.h`, `.hpp`, `.ptx`, and `.json` sources under
`nanochat/mixers/cuda_kda/`. Generated `.so`, `.o`, `.a`, `.cubin`, dependency
or toolchain changes, site-package patches, downloads, symlinks, submodules,
and runtime imports from FLA, the protected oracle, or `ref/` are forbidden.
Bootstrap also forbids PTX.

The protected scanner is case-insensitive, strips comments, and rejects these
exact byte substrings in executable candidate source and strings:

```text
import fla
from fla
_run_fla
_fla_ops
_fla_causal_conv1d
_reference_kda
tests.kda_oracle
/ref/
tcgen05
tmem
```

Python executable source additionally may not contain `torch.library` or
`from torch import library`; registration must originate in the loaded native
extension. The audit strips comments before checking these tokens, so harmless
explanatory comments are allowed. SM121 does not support the forbidden
instruction-family tokens above.

A project-owned claim ultimately requires tracked native CUDA source, an
isolated-cache loaded library hash, an SM121 build receipt, a registered
`nanochat_kda::` CUDA operator whose returned tensors protected routing consumes,
expected kernel symbols, and protected runtime/profile evidence. Python or
Triton wrappers do not inflate CUDA ownership. The operator namespace, names, argument order, returns, shapes, dtypes, and
build-receipt contract in the candidate README are frozen protected ABI. If the
protected dispatcher and guide disagree, stop and ask the coordinator to
clarify the boundary.

## Math and protected source map

Use these read-only sources instead of guessing the recurrence:

- `tests/kda_oracle.py`: independent canonical math, V-first public state,
  convolution cache convention, and full-layer composition.
- `nanochat/mixers/kda.py`: protected production dispatch and autograd boundary;
  read it to see what a claimed operator must return, never edit it.
- `nanochat/research/cuda_worker.py`: protected runtime, source, ownership,
  boundary, gradient, profile, and sanitizer checks.
- `configs/research/kda_cuda_ownership.toml`: frozen lanes, shapes, tolerances,
  timeouts, and component weights.
- `nanochat/mixers/cuda_kda/README.md`: read-only frozen native ABI, build
  helper contract, and provenance schema.
- `ref/FlashKDA/`: offline reference context only. Never import it, add it to
  provenance, or mention its runtime path in candidate source.

For each token, in FP32 recurrence math:

```text
q_hat = normalize(q) * resolved_scale
k_hat = normalize(k)
decay = lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))
beta  = sigmoid(beta_logits)
S     = S * exp(decay)                         # internal K-first [B,H,K,V]
pred  = k_hat @ S
S     = S + (beta * k_hat) outer (v - pred)
out   = q_hat @ S
```

The public initial/final state is V-first `[B,H,V,K]`; transpose only at the
boundary. Inputs and output are normally BF16, recurrence/state computation is
FP32, the initial state must not be mutated, and `output_final_state=False` must
follow the protected return contract. The recurrent operator is inference-only
in the current protected audit. Treat this map as mathematical guidance and the candidate README as the frozen
native ABI; never invent an alternate call boundary.

## Atomic migration units

Forward/backward capabilities sharing an autograd boundary migrate together:

1. `chunk_forward` + `chunk_backward` (60%)
2. `recurrent_decode` (20%)
3. `causal_convolution_forward` + `causal_convolution_backward` (20%)

Bootstrap needs at least one whole unit. Migration requires a strict ownership
superset of the retained parent. Optimization requires all units project-owned
and zero observed runtime FLA.

## Worktree onboarding and ledger-free candidate check

A candidate worktree intentionally does not get its own environment. Reuse the
coordinator's interpreter and current protected worker. Do not run `uv sync`,
do not create a candidate `.venv`, and do not invoke `research` from the
candidate worktree: doing so can select candidate/stale protected code.

The checker executes exactly the staged snapshot, so it rejects any unstaged or
untracked file anywhere in the candidate worktree. Stage only candidate sources,
then return to the coordinator for the command:

```bash
WORKTREE=/absolute/path/to/candidate-worktree
COORDINATOR=/absolute/path/to/clean/coordinator
LANE=bootstrap

# Candidate worktree: inspect, then stage the intended source snapshot.
git -C "$WORKTREE" status --short
git -C "$WORKTREE" add -- nanochat/mixers/cuda_kda

# Coordinator: shared environment + current protected worker.
cd "$COORDINATOR"
uv run --no-sync research cuda-candidate-check \
  --config configs/research/kda_cuda_ownership.toml \
  --worktree "$WORKTREE" --lane "$LANE"
```

By default artifacts and both compiler caches are created under an isolated
`/tmp/nanochat-cuda-candidate-*` directory and retained for diagnosis. A caller
may instead provide a new or empty directory and caches outside the worktree:

```bash
cd "$COORDINATOR"
uv run --no-sync research cuda-candidate-check \
  --worktree "$WORKTREE" --lane "$LANE" \
  --artifact-dir /tmp/kda-check-recurrent \
  --extension-cache /tmp/kda-check-recurrent/torch-extensions \
  --cuda-cache /tmp/kda-check-recurrent/cuda-cache \
  --sanitizers
```

The protected command performs staged-source/scope checks, builds during the
runtime audit, runs the protected runtime audit, then runs the bounded ownership
profile. `--sanitizers` additionally runs memcheck, racecheck, synccheck, and
initcheck. It neither initializes nor reads/writes the campaign ledger and
cannot retain a result. A preflight pass is diagnostic only: committed intake
and the full supervisor run remain authoritative.

## Commit and exact handoff

After the checker passes, inspect the diff, commit the exact staged tree, push
the candidate branch, and send one handoff. Do not merge or retain it yourself.

```bash
git -C "$WORKTREE" diff --cached --check
git -C "$WORKTREE" commit -m "Implement naive CUDA recurrent decode"
git -C "$WORKTREE" push <remote> HEAD:<candidate-branch>
git -C "$WORKTREE" rev-parse HEAD
git -C "$WORKTREE" status --short
```

Use this exact handoff schema (write `none` rather than omitting a field):

```text
CUDA_CANDIDATE_HANDOFF
lane: <bootstrap|migration|optimization>
worktree: <absolute path>
branch: <pushed branch>
base_sha: <40-hex retained parent>
candidate_sha: <40-hex committed candidate>
hypothesis: <mechanism and predicted correctness/profile observable>
changed_paths: <comma-separated paths>
checker_summary: <absolute .../summary.json>
sanitizers: <passed|not-run|failed>
known_issues: <none or exact issue>
worktree_status: <clean or exact porcelain output>
```

The coordinator verifies the SHA and base before using the hypothesis verbatim
for intake.

## Coordinator launch and candidate cycle

From the reviewed clean foundation commit, the coordinator alone runs:

```bash
git tag -a kda-cuda-ownership-foundation \
  -m "Protected KDA CUDA-ownership autoresearch foundation" <foundation-commit>
uv run --no-sync research cuda-ownership-supervisor init \
  --config configs/research/kda_cuda_ownership.toml
uv run --no-sync research cuda-ownership-supervisor calibrate \
  --config configs/research/kda_cuda_ownership.toml
uv run --no-sync research cuda-ownership-supervisor summary \
  --config configs/research/kda_cuda_ownership.toml
```

After a valid handoff:

```bash
uv run --no-sync research cuda-ownership-supervisor intake \
  --base-ref <current-retained-commit> \
  --candidate-ref <pushed-candidate-sha> \
  --hypothesis "<model-chosen mechanism and predicted observables>"
uv run --no-sync research cuda-ownership-supervisor run --attempt <id>
uv run --no-sync research cuda-ownership-supervisor summary --attempt <id>
```

Human-reviewed retention is explicit:

```bash
uv run --no-sync research cuda-ownership-supervisor retain --attempt <id> \
  --label "<short milestone label>" --reason "<evidence-backed reason>"
```

`retain` changes only the append-only ledger head. It does not merge, tag, or
edit Git. Invalid, rejected, stale, and non-retained attempts remain visible.

## Derived lane decisions

### Bootstrap

At least one atomic unit must become verified native CUDA. Correctness,
ownership, build, profile, and all supervisor sanitizer checks are mandatory.
Kernel timing is bounded and observational: a performance timeout is a saved
censored observation, not a correctness win or loss. PTX is forbidden.
Retainable decision: `correct_bootstrap`.

### Migration

Each retained candidate is a strict ownership superset. Claimed units use
project CUDA; only unclaimed units may use protected FLA. Correctness, receipts,
sanitizers, and profiles remain mandatory; performance remains observational.
Retainable decisions are `validated_component`, then `fla_free_naive` when all
units are project-owned and no FLA executes. The first complete milestone is
the immutable naive-CUDA speed anchor.

### Optimization

All units remain project-owned and FLA-free. Parent and candidate run nine
alternating A/B, B/A 4k-training blocks. The paired log-ratio 95% Student-t
interval resolves the frozen effect, while memory, kernel-latency, and baseline
drift gates apply. Only a statistically improved and resource-safe candidate
is eligible as `optimization_retained`.

## Human-authorized timeout-only re-freeze

The original protected protocol
`ba64643fd7fd764bab39f99ea83ecf3805522fab3005516f29079806e32a46cf`
remains immutable historical evidence. Its first optimization attempt proved
that the exact retained naive parent could not complete the required
optimization kernel payload within 180 seconds, even though the candidate
completed every correctness, ownership, profile, and sanitizer gate. The
parent timeout prevented block 0 and was not a numeric performance result.

A human explicitly authorized a new protected protocol that changes only
process ceilings and evidence namespaces:

- the kernel worker ceiling is 3,600 seconds;
- each seven-step training-block ceiling is 13,500 seconds;
- the controller ref is
  `kda-cuda-ownership-controller-timeout-refreeze`;
- the fresh ledger is
  `runs/kda-cuda-ownership-timeout-refreeze.sqlite3`.

The training ceiling is bounded by a saved, ledger-free exact-shape diagnostic.
The retained naive parent completed all seven steps in 8,932.104 summed step
seconds; applying the declared 1.5x margin and rounding upward to 300 seconds
produced 13,500 seconds. This diagnostic sizes the ceiling only. Its throughput
and memory are not campaign evidence. The kernel ceiling is sized from the
saved naive T=65/T=256/T=1024 scaling and the unchanged 10-warmup/50-sample
T=4096 payload.

No shape, sample count, operation, seed, ordering, warmup, timed step, paired
block count, confidence interval, effect threshold, memory limit,
kernel-regression limit, drift limit, correctness tolerance, ownership gate,
profile requirement, or sanitizer gate changes. The old ledger, artifacts,
controller tags, and failed attempt remain read-only. The new ledger must be
calibrated and the retained milestone chain rerun from the unchanged foundation;
old SQLite rows and artifacts must never be copied into the new protocol.

## Deadline and timeout distinctions

- A candidate-agent handoff deadline is a coordination deadline. When reached,
  stop starting work, terminate owned foreground jobs, and hand off exact state;
  do not leave background GPU or compiler processes.
- Checker runtime/profile and sanitizer timeouts come from the frozen config and
  bound individual subprocesses. Extending an agent deadline does not extend
  these protected subprocess limits.
- In bootstrap/migration, only bounded *performance* timeout is observational.
  Build, runtime, correctness, profile, NaN, crash, and sanitizer timeout/failure
  are invalid preflight or campaign results.
- Optimization training-block and release deadlines are supervisor gates and
  cannot be replaced by a candidate check or a longer agent turn.

## Reporting and fixed-anchor release

```bash
uv run --no-sync research cuda-ownership-supervisor report --format json
uv run --no-sync research cuda-ownership-supervisor report \
  --format markdown --output cuda-journey.md
uv run --no-sync research cuda-ownership-supervisor verify-release --milestone <id>
```

The report preserves all attempts and compatible per-shape anchors. Chained
optimization point estimates are illustrative only, never gates or confidence
intervals. Release verification uses fifteen fixed-anchor interleaved pairs and
requires all ownership/safety/resource gates plus a lower 95% throughput bound
of at least +3%.

Switching the default backend remains a separate human-reviewed release action
after protected integration and the current general-LM evaluation plan. The
CUDA ownership loop never launches quality training itself.
