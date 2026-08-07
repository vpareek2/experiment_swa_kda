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
| GB10 platform facts | `ref/nvidia-docs/dgx-spark-user-guide.txt` | [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/dgx-spark.pdf) |
| CUDA kernel concepts | `ref/nvidia-docs/{blackwell-tuning-guide,cuda-best-practices-guide,cuda-programming-guide}.txt` | NVIDIA CUDA 13.3 documentation |
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


## 2026-08-07 [agent] pin DGX Spark and CUDA kernel documentation

**Context**

- Added official NVIDIA platform and CUDA reference material so Prime candidate
  workers can reason from local documented concepts rather than generic GPU
  priors alone.

**Commands**

```bash
# Downloaded official NVIDIA documentation snapshots and extracted searchable text.
# See ref/nvidia-docs/SOURCES.json for URLs and SHA-256 values.
```

**Artifacts**

- Ignored, read-only `ref/nvidia-docs/` (31 MiB), with PDFs, HTML snapshots,
  extracted text, `SOURCES.json`, and `AGENT_INDEX.md`.
- DGX guide PDF SHA-256:
  `96ec0bed663a954ee5e4a1e6a7c55d9f1bf3d6223302e6245c6a5043e516983c`.
- CUDA Programming Guide 13.3 SHA-256:
  `1a0659b7324d10f1c0a57fc0c82aa83ee1aba437fa85dabc90ee8f736e439bea`.
- CUDA Best Practices Guide 13.3 SHA-256:
  `38ed26226ea3f71f92a69516128cfea31a309a110ec75bc2a43777c440479def`.
- Blackwell Tuning Guide 13.3 SHA-256:
  `1697938ef20db2127bb2874ce8bff1df2a07d57a435fcef5055eb521774e13da`.

**Result**

- Verified DGX Spark guide facts added to the local agent index: Grace
  Blackwell integrated platform; 20-core Arm CPU (10 Cortex-X925 + 10
  Cortex-A725); 128 GB LPDDR5x unified memory on a 256-bit, 4266 MHz interface
  with documented 273 GB/s bandwidth; Blackwell GPU with fifth-generation
  Tensor Cores; 10 GbE/ConnectX-7 connectivity.
- The Blackwell guide labels its architecture-specific discussion as compute
  capability 10.0, whereas the live GB10 reports 12.1. Candidates may use its
  general concepts but must not assume its specific limits apply to GB10; local
  profile and systems artifacts remain authoritative.

**Next**

- The Prime parent should expose `ref/nvidia-docs/AGENT_INDEX.md` and relevant
  source text read-only to children. A child should consult the concise tuning
  guide first, then targeted sections, rather than ingesting the full corpus.


## 2026-08-07 [agent] record GB10 SM121 tensor-core instruction constraint

**Context**

- DGX Spark GB10 is not interchangeable with server Blackwell for low-level
  instruction assumptions. The candidate briefing needs a verified constraint,
  not inference from marketing terminology.

**Commands**

```bash
# Snapshot official NVIDIA/dgx-spark-playbooks issue 22 and maintainer comments
# into ignored ref/nvidia-docs/dgx-spark-playbooks-issue-22.json.
```

**Artifacts**

- Local source snapshot SHA-256:
  `7bb0f5ca3eeafe1ad31dc7d7d206df51d7f5ad97852c2220fc2b4f878bb7f786`.
- Canonical source: [NVIDIA/dgx-spark-playbooks issue 22](https://github.com/NVIDIA/dgx-spark-playbooks/issues/22), including NVIDIA maintainer comments dated 2025-12-19 and 2025-12-22.

**Result**

- NVIDIA states that DGX Spark/GB10 is SM121; `tcgen05` and TMEM are not
  supported. DSMEM and TMA/multicast are stated to be available. TMA/DSMEM
  availability does not prove that the installed PyTorch/Triton/FLA stack has a
  usable path.
- Candidate workers must not target tcgen05, TMEM, SM100 cubin patching, or
  server-Blackwell instruction assumptions. Any proposed TMA/DSMEM path needs
  explicit local toolchain support and a measured systems/profile justification.

**Next**

- Use the ordinary eager KDA/PyTorch/FLA path as the baseline constraint; let
  mandatory profile artifacts, not unsupported instruction speculation, select
  subsequent candidate hypotheses.


## 2026-08-07 [agent] prepare context-complete Prime baseline

**Context**

- The earlier Prime baseline tag predates the pinned NVIDIA/CUDA/DGX source
  context and the verified SM121 instruction constraint.

**Commands**

```bash
git tag -a kda-speed-prime-context-baseline-20260807 -m "Context-complete Prime baseline for KDA speed autoresearch" <this commit>
git push origin kda-speed-prime-context-baseline-20260807
```

**Artifacts**

- The new immutable tag is created after this documentation commit.

**Result**

- Future Prime candidate worktrees must use `kda-speed-prime-context-baseline-20260807` so their checked-out
  program includes the complete source/context and tcgen05/TMEM constraint.

**Next**

- Launch `kda-speed/attempt-002` from this tag when authorized.


## 2026-08-07 [agent] authorize 15-attempt implementation-language-neutral loop

**Context**

- The user authorized the supervising agent to run the protected loop with a
  hard cap of 15 attempts and allowed any implementation language that works in
  the existing uv-managed project environment.

**Commands**

```bash
# Freeze a new protocol: max_attempts=15, new ledger, candidate scope nanochat/mixers/.
```

**Artifacts**

- Frozen ledger path: `runs/kda-training-autoresearch-15.sqlite3`.
- Candidate scope: files under `nanochat/mixers/`; all research/config/test and
  trainer files remain protected.

**Result**

- Candidates may use Python, Triton, CUDA/PTX, CuTe DSL, cuTile, Gluon, or
  another implementation approach only when it is self-contained under the
  mixer directory and runs through the existing uv environment. They may not
  add unreviewed project dependencies, patch the toolchain, use tcgen05/TMEM,
  or weaken correctness/fallback rules.
- The previous documentation-only dry run remains historical plumbing evidence
  in the older ledger and does not consume the new 15-attempt experimental cap.

**Next**

- Commit, push, and tag this new protocol baseline; launch attempt 1 through a
  separate candidate worktree and child agent.


## 2026-08-07 [agent] namespace artifacts across speed protocols

**Context**

- The first launch in the new 15-attempt ledger stopped before correctness or
  GPU work because attempt ID 1 collided with the historical dry-run artifact
  directory from the older protocol.

**Commands**

```bash
uv run --no-sync python -m pytest -q tests/test_speed_supervisor.py
```

**Artifacts**

- No candidate phase artifact was created and no GPU job launched.

**Result**

- Protected supervisor artifacts are now namespaced by the first 12 hex digits
  of the frozen protocol hash before the attempt ID. Tests pass (7). The empty
  `testing` transition is reset to `accepted` with an append-only audit event
  so the same immutable candidate can resume without consuming an experimental
  attempt for a harness-only collision.

**Next**

- Resume attempt 1 at commit `a503132` under the corrected artifact namespace.


## 2026-08-07 [agent] retain attempt 1 vectorized short convolution

**Context**

- Candidate `a503132` replaced the per-token Python short-convolution loop with
  one equivalent grouped `conv1d` over cached history plus the full sequence.
  It was produced and pushed from isolated branch `kda-speed/attempt-001`.

**Commands**

```bash
research speed-supervisor intake --base-ref kda-speed-autoresearch-15-baseline-20260807 \
  --candidate-ref a503132 --idea "Vectorize ShortConvolution ..."
research speed-supervisor run --attempt 1
```

**Artifacts**

- Ledger: `runs/kda-training-autoresearch-15.sqlite3`, attempt 1.
- Ignored artifact root:
  `runs/speed-supervisor/c50c1dfdddc6/attempt-00001/`.
- Candidate branch/commit: `origin/kda-speed/attempt-001` at `a503132`.

**Result**

- Protected correctness passed: 52 tests, including the fixed CPU/operator/
  CUDA/integration suite; FLA resolved without fallback.
- Baseline-pre median: 845 tok/s (38.766 s/step). Baseline-post median:
  822 tok/s (39.818 s/step). Their supervisor median was 833.5 tok/s and drift
  was 2.76%, within the frozen 3% limit.
- Candidate median: 41,413 tok/s (0.7912 s/step), a 48.69x relative increase
  over the supervisor baseline median. Timed candidate samples were
  41,248/41,413/41,465 tok/s.
- Mandatory profile training update fell from 38,160.5 ms to 787.8 ms. The
  q/k/v short-convolution regions fell from 2,917.9/2,648.4/2,598.5 ms to
  6.17/6.22/6.17 ms. FLA forward remained small (42.3 to 32.1 ms). The
  improvement is far beyond timing noise and passed all frozen gates.
- Decision: `improved`; the candidate was merged as the next retained baseline.
  This is a training-speed systems result only; quality remains unevaluated.

**Next**

- Tag the retained baseline. Attempt 2 should use its mandatory profile to
  identify a new primary bottleneck rather than revisiting the removed Python
  token loop.


## 2026-08-07 [agent] reject attempt 2 fused QKV projection

**Context**

- Candidate `782b932` concatenated existing q/k/v weights, executed one wider
  `F.linear`, and split the outputs while preserving public modules and state.

**Commands**

```bash
research speed-supervisor run --attempt 2
```

**Artifacts**

- Ledger attempt 2; ignored artifacts:
  `runs/speed-supervisor/c50c1dfdddc6/attempt-00002/`.
- Candidate branch: `origin/kda-speed/attempt-002` at `782b932`.

**Result**

- Protected correctness passed. Baseline medians produced supervisor baseline
  41,538.5 tok/s with only 0.022% drift. Candidate median was 41,759 tok/s,
  +0.53%, below the frozen 3% threshold.
- Mandatory profile update improved by only 3.52 ms; forward by 1.78 ms and
  backward by 0.85 ms. The q/k/v projection regions disappeared as expected,
  but their prior contribution was too small for a material step gain.
- Decision: `not_improved`; branch retained for provenance, not merged.

**Next**

- Attempt 3 may test fusion of the three depthwise short-convolution dispatches,
  whose retained profile contribution is larger than input projection dispatch.


## 2026-08-07 [agent] reject attempt 3 fused short convolutions

**Context**

- Candidate `1ac8183` fused the three retained vectorized q/k/v depthwise
  convolutions into one wider grouped convolution while preserving modules and
  cache semantics.

**Commands**

```bash
research speed-supervisor run --attempt 3
```

**Artifacts**

- Ledger attempt 3; ignored artifacts:
  `runs/speed-supervisor/c50c1dfdddc6/attempt-00003/`.
- Candidate branch: `origin/kda-speed/attempt-003` at `1ac8183`.

**Result**

- Correctness passed. Baseline was 41,529 tok/s with 0.21% drift. Candidate
  was 40,876 tok/s, a 1.57% regression.
- Individual q/k/v convolution profile regions disappeared as expected, but
  model backward increased 9.64 ms and the full update increased 12.61 ms.
  The wider grouped depthwise kernel was counterproductive at this GB10 shape.
- Decision: `not_improved`; not merged.

**Next**

- Target FLA backward recomputation: test `disable_recompute=True` in chunk
  training as a speed-for-activation-memory tradeoff, with exact correctness
  and peak-memory evidence.


## 2026-08-07 [agent] reject attempt 4 disable FLA recomputation

**Context**

- Candidate `2dde1f2` passed `disable_recompute=True` only to FLA chunk KDA,
  intentionally trading saved activations for less backward recomputation.

**Commands**

```bash
research speed-supervisor run --attempt 4
```

**Artifacts**

- Ledger attempt 4; ignored artifacts:
  `runs/speed-supervisor/c50c1dfdddc6/attempt-00004/`.
- Candidate branch: `origin/kda-speed/attempt-004` at `2dde1f2`.

**Result**

- Correctness passed. Baseline was 41,460.5 tok/s with 0.080% drift.
  Candidate was 41,992 tok/s, +1.28%, below the 3% threshold.
- Profile update improved 13.65 ms and backward improved 12.75 ms. Peak
  allocated training memory increased from 5,664.25 MiB to 5,986.39 MiB.
- Decision: `not_improved`; the real speed-for-memory gain was too small to
  retain independently.

**Next**

- Reassess the remaining KDA forward/backward kernel and layout costs before
  attempt 5; do not combine this rejected axis without a distinct interaction
  hypothesis.
