# KDA Training-Speed Autoresearch

This stream is supervisor-owned durable context for the protected speed-only
loop. Candidate workers read it but never edit it.

## Task briefing

- **Objective:** steady-state eager 4k KDA training tokens/second only.
- **Frozen protocol:** `configs/research/kda_training_speed.toml`.
- **Supervisor interface:** `research speed-supervisor {init,intake,run,summary}`.
- **Candidate writable scope:** `nanochat/mixers/kda.py` only.
- **Correctness suite:** `tests/test_kda_layer.py`, `test_kda_operator.py`,
  `test_kda_cuda.py`, and `test_kda_integration.py` as declared by the config.
- **Score:** median of three timed updates after one warmup. Cold setup is
  diagnostic, not scored. A valid improvement requires at least 3% throughput
  gain and at most 3% baseline drift in baseline-pre/candidate/baseline-post.
- **Not evaluated:** language quality, BPB, CORE, RULER, LongBench, memory
  probes, prefill/decode speed, or parameter/quality tradeoffs.

## Verified local environment facts

Obtain the current machine facts from `research doctor --config
configs/research/kda_training_speed.toml`; do not infer them from marketing
names. At supervisor foundation time, doctor reported:

- NVIDIA GB10, compute capability 12.1, ARM64 host;
- PyTorch 2.9.1+cu130, CUDA runtime 13.0, Triton 3.5.1;
- selected PTXAS CUDA 13.1; FLA core 0.5.2; BF16 available;
- the installed PyTorch emits a support-range warning for capability 12.1.

The current speed lane deliberately sets `TORCH_COMPILE_DISABLE=1` for all
baseline and candidate subprocesses. This is a declared comparable eager lane,
not an automatic fallback. Full compiled KDA previously failed in the protected
fused AdamW path after Dynamo rank-polymorphic recompiles; that compiler issue
is out of candidate scope.

## Existing evidence

- A restricted no-export CUDA key-averages profile of eager 4k KDA completed;
  no Chrome trace is retained. It showed over one million CUDA launches,
  dominated by tiny generic elementwise kernels. The current short-convolution
  implementation contains a per-token Python loop, making it a concrete
  candidate hypothesis source.
- FLA Triton KDA kernels were present in the profile. The evidence does not
  authorize importing local reference code or changing recurrence semantics.
- A direct vectorized-convolution experiment was intentionally reverted because
  candidates must be produced by the autoresearch process, not by the
  supervisor. Treat its result only as historical diagnostic context, not a
  retained implementation or baseline.

## Local reference source map

`ref/` is ignored, local-only, and may be read but never imported at runtime.
It is not copied automatically into detached benchmark worktrees. The
supervisor may expose its local location to a candidate worker out of band.

| Purpose | Expected local material | Canonical source |
| --- | --- | --- |
| KDA/FLA implementation comparison | `ref/FlashKDA/` | Flash Linear Attention / KDA upstream source, pinned separately |
| CUDA execution model | locally cached guide/PDF | NVIDIA CUDA C++ Programming Guide |
| Optimization practices | locally cached guide/PDF | NVIDIA CUDA Best Practices Guide |
| Triton kernel patterns | locally cached docs/examples | Triton documentation |
| GB10 platform facts | locally cached vendor documentation | NVIDIA DGX Spark documentation |
| GPU-kernel agent guidance | `ref/nvidia-skills/` at `a9ea6f987f3a2e5efbd6f94ad394dc0b64265719` | [NVIDIA/skills](https://github.com/NVIDIA/skills), Apache-2.0 + CC-BY-4.0 |

The pinned NVIDIA catalog is 64 MiB and read-only. For this loop, read only
when relevant: `tilegym-improve-cutile-kernel-perf` for profile-led
experimentation, `tilegym-cutile-autotuning` for search/compile tradeoffs,
`tilegym-converting-cutile-to-triton` for kernel/layout concepts, and
`nemo-mbridge-perf-cuda-graphs` for host-launch concepts. These describe
cuTile/Megatron-specific APIs that are **not** installed or authorized here;
they are conceptual reference only and cannot override this program.

Before a reference is used to motivate a candidate, the supervisor should add
its URL, revision/date, and any material constraint to the dated attempt entry
below. Do not commit large downloaded documents, reference implementations,
private paths, or hostnames.

## Prime-Agent handoff

Parent/supervisor prompt: read `program_kda_training_speed.md`, this stream,
and `research speed-supervisor summary`; create an isolated candidate worktree;
spawn a child worker; then own intake, run, summary, and runbook recording.

Child/candidate prompt: read the same briefing and the supplied summary; make
one KDA-only committed hypothesis in the assigned worktree; message its parent
with the commit SHA; never invoke the supervisor as a substitute for the
parent's decision and never edit this stream.

## 2026-08-07 [agent] initialize Prime-Agent speed-loop briefing

**Context**

- The user selected Prime Agent as the intended model interface and requested a
  supervisor-owned runbook, task briefing, and reference map.

**Commands**

```bash
uv run --no-sync research speed-supervisor init \
  --config configs/research/kda_training_speed.toml
```

**Artifacts**

- `program_kda_training_speed.md`
- This stream and the ignored SQLite ledger under `runs/`.

**Result**

- No candidate implementation or GPU benchmark was launched. The ledger has no
  attempts. The briefing separates Prime parent/supervisor authority from
  spawned candidate-worker authority.

**Next**

- Add local reference provenance when material is actually supplied, then let a
  Prime supervisor create the first candidate worktree and child task.


## 2026-08-07 [agent] require bounded CUDA-event operator-region profiles

**Context**

- Every valid speed attempt needs comparable attribution without the prior
  multi-gigabyte Chrome-trace failure or an agent-selected profiling mode.

**Commands**

```bash
TORCH_COMPILE_DISABLE=1 <two-update 4k KDA profile smoke>
uv run --no-sync python -m pytest -q
```

**Artifacts**

- Protected profiler: `nanochat/research/speed_profile.py`.
- Ignored smoke artifact: `runs/speed-profile-smoke/profile.json` (4,348 bytes).

**Result**

- `speed-supervisor run` now requires baseline-pre and candidate CUDA-event
  operator-region profiles after their successful systems timings. A missing,
  malformed, timed-out, or oversized profile invalidates the attempt.
- The profile instruments the actual eager 4k training update, including its
  four gradient-accumulation microbatches, with protected CUDA events around
  forward/backward, optimizer, each KDA layer, q/k/v projections and short
  convolutions, FLA KDA forward, and output components. It synchronizes once
  and stores a compact JSON region table; it never invokes `torch.profiler` or
  exports a Chrome trace.
- The raw KDA smoke completed in 82.9 seconds end-to-end. Its profiled update
  took 39.639 s and produced the expected attribution: q/k/v short
  convolutions consumed about 8.558 s in aggregate and FLA KDA forward about
  47 ms. This is diagnostic evidence only, not an autoresearch candidate run.

**Next**

- The Prime supervisor can now require profile deltas for every candidate.
  Add a runbook entry after every terminal attempt using the immutable ledger
  and profile artifacts; do not reintroduce an unrestricted per-kernel trace.


## 2026-08-07 [agent] complete end-to-end speed-supervisor dry run

**Context**

- Validated the full protected loop with a documentation-only candidate commit
  that changed only `nanochat/mixers/kda.py` and deliberately preserved all KDA
  execution behavior.

**Commands**

```bash
research speed-supervisor intake --base-ref c051775 --candidate-ref ec97019 \
  --idea "Dry-run documentation-only KDA diff to validate supervisor plumbing"
research speed-supervisor run --attempt 1
research speed-supervisor summary --attempt 1
```

**Artifacts**

- Ledger attempt: `1`; candidate branch: `kda-speed-dry-run` at `ec97019`.
- Ignored artifacts: `runs/speed-supervisor/attempt-00001/`.
- The temporary candidate worktree was removed after completion; the branch
  preserves the reviewed commit for ledger provenance.

**Result**

- The complete sequence passed: fixed correctness suite; baseline-pre systems
  timing and required profile; candidate systems timing and required profile;
  baseline-post systems timing; SQLite summary generation.
- Baseline median was 827 tok/s and the documentation-only candidate was
  842 tok/s (+1.81%). Baseline drift was 4.11%, above the frozen 3% limit, so
  the supervisor correctly emitted `retest`, not an improvement. No quality
  metric was run and no candidate implementation was retained on `main`.
- This is plumbing validation, not a KDA speed conclusion. It occupies one of
  the bounded ledger attempts and remains visible to later candidate models.

**Next**

- Start actual Prime candidate work from the protected `main` baseline; use the
  recorded dry run as evidence that small timing differences require retest,
  not as a performance target.


## 2026-08-07 [agent] freeze and push speed-loop Git topology

**Context**

- The Prime-Agent loop needs durable baseline and candidate provenance across
  brittle local hardware, while keeping `main` free of unreviewed code.

**Commands**

```bash
git tag -a kda-speed-baseline-20260807 -m "Frozen baseline for protected KDA training-speed autoresearch" 531c16d
git push origin kda-speed-baseline-20260807
git push origin kda-speed-dry-run
```

**Artifacts**

- Pushed annotated baseline tag: `kda-speed-baseline-20260807` at `531c16d`.
- Pushed dry-run provenance branch: `origin/kda-speed-dry-run` at `ec97019`.
- Updated `program_kda_training_speed.md` with the mandatory topology.

**Result**

- `main` is the clean coordinator. Every actual candidate must use its own
  named worktree/branch created from the retained baseline tag, be committed
  and pushed before intake, and remain retained as an immutable ledger ref.
  The Prime parent alone invokes the supervisor from the coordinator.

**Next**

- Launch the first Prime candidate child from a separate
  `kda-speed/attempt-002` worktree when authorized.


## 2026-08-07 [agent] publish Prime worktree baseline

**Context**

- The Git-topology briefing itself must be present in every candidate worktree,
  so the earlier pre-briefing baseline tag is retained as provenance while a
  new immutable Prime-loop tag anchors future candidates.

**Commands**

```bash
git tag -a kda-speed-prime-baseline-20260807 -m "Prime Agent worktree baseline for protected KDA training-speed autoresearch" 2e8cd30
git push origin kda-speed-prime-baseline-20260807
```

**Artifacts**

- Pushed annotated tag: `kda-speed-prime-baseline-20260807` at `2e8cd30`.

**Result**

- Future candidate worktrees must branch from
  `kda-speed-prime-baseline-20260807`, which includes the mandatory Prime
  parent/child and coordinator/worktree contract.

**Next**

- The first real candidate is `kda-speed/attempt-002` from this tag.


## 2026-08-07 [agent] pin NVIDIA GPU-performance agent reference

**Context**

- The Prime candidate workers need a high-quality local CUDA/GPU-performance
  concept source without allowing external reference code to become runtime
  code or override the protected protocol.

**Commands**

```bash
git clone --depth 1 https://github.com/NVIDIA/skills.git ref/nvidia-skills
cd ref/nvidia-skills && git rev-parse HEAD
chmod -R a-w ref/nvidia-skills
```

**Artifacts**

- Ignored, read-only reference clone: `ref/nvidia-skills/` (64 MiB), commit
  `a9ea6f987f3a2e5efbd6f94ad394dc0b64265719`.
- Upstream license: Apache-2.0 and CC-BY-4.0.

**Result**

- Added the pinned catalog to the source map. Its relevant materials cover
  profile-led GPU-kernel iteration, autotuning, Triton/layout concepts, and
  CUDA-graph launch-overhead concepts. The child must read selectively and
  treat it as conceptual evidence only; project constraints and measured local
  artifacts always take precedence.

**Next**

- A Prime parent may expose this read-only reference root to its candidate
  child alongside the program, runbook briefing, and current ledger summary.
