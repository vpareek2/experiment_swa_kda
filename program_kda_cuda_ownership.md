# KDA CUDA-Ownership Autoresearch Program

## Purpose

Show, with immutable evidence, how autonomous research moves KDA from its
sequential PyTorch oracle through a naive native-CUDA bootstrap to an optimized,
FLA-free, project-owned backend. The end state is stronger than the minimum
ownership goal: all five frozen KDA capabilities are project-owned, while
selective PTX is accepted only after a measured CUDA-path A/B.

This is backend systems research. It does not evaluate or claim language-model
quality. Read `AGENTS.md`, this file, `runbook/streams/kda_cuda_ownership.md`,
and the latest `research cuda-ownership-supervisor summary` before working.
The completed speed campaign is read-only historical context and is never
arithmetically combined with this campaign.

## Roles and hypothesis freedom

The Prime parent owns the protected ledger, retained-milestone head, worktrees,
commands, evidence, and runbook. A child owns exactly one committed candidate
in its assigned worktree. The supervisor derives the maturity lane from the
retained head; within that lane, the model chooses its own mechanistic
hypothesis. There is no supervisor-authored menu or rotation of attempt
categories.

The first child receives one deliberate instruction: **write the simplest
correct naive project-owned CUDA implementation for at least one atomic unit;
optimize nothing and use no PTX.** The model writes this code, not the human.

## Frozen Git and source contract

After review, commit this protected foundation and create the immutable tag
`kda-cuda-ownership-foundation`. Never move it or any `kda-speed-*` tag. Every
candidate branches from the current human-retained milestone and is pushed
before intake. The coordinator remains clean. Never checkout/reset/clean/rebase,
force-push, automatically merge, or automatically move a tag.

Candidates edit only `nanochat/mixers/cuda_kda/`. Tracked `.cu/.cuh`, C++/header,
Python build glue, JSON claims, and selectively justified PTX are allowed.
Generated `.so/.o/.a/.cubin`, dependency/toolchain changes, site-package
patches, downloads, and runtime imports from FLA, the protected reference, or
`ref/` are forbidden.

A project-owned claim requires tracked native CUDA source, an isolated-cache
loaded library hash, an SM121 build receipt, a registered `nanochat_kda::` CUDA
operator whose return values are consumed directly by protected routing,
expected kernel symbols, and protected runtime dispatch evidence.
Python/Triton wrappers do not inflate native-CUDA ownership.

## Atomic migration units

Forward/backward capabilities that share one autograd boundary migrate together:

1. `chunk_forward` + `chunk_backward` (weight 60%)
2. `recurrent_decode` (weight 20%)
3. `causal_convolution_forward` + `causal_convolution_backward` (weight 20%)

Protected routing calls the candidate only for claimed units. During bootstrap
and migration it calls protected FLA only for explicitly unclaimed units. A
claimed unit resolving to FLA is invalid. The optimization lane requires every
unit project-owned and observes zero runtime FLA.

## One-time launch preparation

From the reviewed clean foundation commit:

```bash
git tag -a kda-cuda-ownership-foundation -m "Protected KDA CUDA-ownership autoresearch foundation" <foundation-commit>
research cuda-ownership-supervisor init --config configs/research/kda_cuda_ownership.toml
research cuda-ownership-supervisor calibrate --config configs/research/kda_cuda_ownership.toml
research cuda-ownership-supervisor summary --config configs/research/kda_cuda_ownership.toml
```

Calibration saves bounded, comparison-compatible sequential-PyTorch and retained
FLA operator anchors. It does not modify either implementation.

## Candidate cycle

```bash
research cuda-ownership-supervisor intake \
  --base-ref <current-retained-commit> \
  --candidate-ref <pushed-candidate-sha> \
  --hypothesis "<model-chosen mechanism and predicted observables>"
research cuda-ownership-supervisor run --attempt <id>
research cuda-ownership-supervisor summary --attempt <id>
```

Intake rejects a base other than the current retained head. After reviewing a
milestone-eligible result, the human/supervising parent appends it explicitly:

```bash
research cuda-ownership-supervisor retain --attempt <id> \
  --label "<short milestone label>" --reason "<evidence-backed reason>"
```

`retain` changes only the append-only ledger head. It does not merge, tag, or
edit Git. Invalid, rejected, and non-retained attempts remain visible forever.
A stale parallel attempt cannot advance the head.

## Derived lanes

### Bootstrap

- No retained CUDA candidate exists.
- At least one atomic unit must become verified native CUDA.
- Strong correctness and all four sanitizer tools are mandatory.
- Kernel timing is bounded and observational; a timeout is saved explicitly as
  a censored observation and does not invalidate otherwise correct code.
- Crashes, NaNs, malformed evidence, and correctness/safety failures are invalid.
- FLA is allowed only for unclaimed units.
- PTX is forbidden.
- Retainable decision: `correct_bootstrap`.

### Migration

- Each retained candidate must be a strict ownership superset of its parent.
- Claimed units use project CUDA; unclaimed units may still use protected FLA.
- Correctness, source/binary/operator receipts, sanitizers, and bounded profiles
  remain mandatory. Performance remains observational.
- Retainable decisions: `validated_component`, then `fla_free_naive` when all
  units are project-owned and no FLA executes.
- The first `fla_free_naive` milestone becomes the immutable naive-CUDA speed
  anchor for the educational waterfall.

### Optimization

- Opens only after a retained complete FLA-free naive backend.
- All units must remain project-owned; any FLA execution is invalid.
- Parent and candidate both run `project_cuda` in nine alternating A/B, B/A
  4k-training blocks, with two warmups and five timed steps per process.
- The paired log-ratio 95% Student-t interval resolves 0.75% effects; peak
  allocation, kernel-latency, and one-percent baseline-drift limits apply.
- Only a statistically improved, resource-safe candidate is milestone-eligible
  as `optimization_retained`.

## Correctness, safety, profiling, and PTX

Every lane validates independent-oracle outputs, V-first FP32 state, tile and
recurrence boundaries, unequal K/V dimensions, nonzero-state nonmutation,
causality, extreme gates, output/state gradients, complete-layer gradients, and
chunk/recurrent equivalence. Compute Sanitizer memcheck, racecheck, synccheck,
and initcheck are mandatory.

Microbenchmarks retain raw decode, chunk forward/backward, and causal-convolution
forward/backward samples. Profiles are bounded key averages, capped at 256 KiB,
and never export Chrome traces.

Bootstrap forbids PTX. Later PTX needs an exact declaration, SM guard, ordinary
CUDA fallback, source/build evidence, disabled-PTX correctness, and at least the
frozen 2% same-commit latency benefit. `tcgen05` and TMEM remain forbidden on
SM121.

## Educational anchors and report

```bash
research cuda-ownership-supervisor report --format json
research cuda-ownership-supervisor report --format markdown --output cuda-journey.md
```

The report retains all raw attempts and shows compatible per-shape speedups
against the sequential Python anchor, FLA operator anchor, first bootstrap,
first complete FLA-free naive backend, and direct parent. Ownership changes are
reported as explicit components and percentage points.

Chained retained optimization point estimates are illustrative only—never a
gate or confidence interval. Historical speed evidence is a hash-pinned,
non-comparison-compatible context section and is never multiplied into the
CUDA campaign.

## Fixed-anchor release verification

After retaining a complete milestone:

```bash
research cuda-ownership-supervisor verify-release --milestone <id>
```

This appends a separate immutable release run: fifteen interleaved pairs compare
that exact project-CUDA milestone directly against clean fixed anchor
`0b4b24773c2696c23338d7600101d7072b592aa9`. Promotion requires all ownership,
safety, resource gates and a lower 95% throughput-confidence bound of at least
+3%. It never overwrites discovery evidence.

Switching the default backend remains a separate human-reviewed release step
after protected integration and the current general-LM evaluation plan. The
inner loop never launches quality training by itself.
