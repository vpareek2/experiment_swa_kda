# KDA CUDA-Ownership Autoresearch

This stream is supervisor-owned durable context. Candidate workers read it but
never edit it.

## Launch briefing

- **Goal:** record the entire sequential-PyTorch → naive native CUDA → optimized
  FLA-free project CUDA journey, while keeping performance evidence matched and
  auditable.
- **Historical context:** clean anchor
  `0b4b24773c2696c23338d7600101d7072b592aa9`; completed speed ledger, artifacts,
  branches, protocol hash, and tags remain unchanged.
- **Foundation tag required after commit:** `kda-cuda-ownership-foundation`.
- **Protocol:** `configs/research/kda_cuda_ownership.toml`, independent schema 2.
- **CLI:** `research cuda-ownership-supervisor
  {init,calibrate,intake,run,retain,verify-release,report,recover,summary}`.
- **Candidate scope:** `nanochat/mixers/cuda_kda/` only.
- **Lanes:** automatically derived bootstrap → migration → optimization.
- **Hypotheses:** freely chosen by the model within the derived maturity lane.
- **Bootstrap:** simplest correct native CUDA unit, no optimization, no PTX;
  slow/timeout performance is observational rather than invalid.
- **Migration:** strict retained ownership supersets; FLA only for unclaimed
  units; first complete FLA-free backend becomes the naive-CUDA anchor.
- **Optimization:** all units project-owned, no FLA, repeated/interleaved 4k
  parent-vs-candidate measurements and strict statistical/resource retention.
- **Release:** separate fifteen-pair fixed-anchor verification with cumulative
  3% lower-confidence-bound gate.
- **Not evaluated:** BPB, CORE, RULER, or architecture quality.

## 2026-08-08 [agent] implement staged naive-to-optimized campaign

**Context**

- The first protected foundation was safe for a complete backend challenger but
  made a naive first implementation face production 4k gates immediately.
- The desired educational evidence requires correct slow milestones to survive,
  incremental component migration, immutable retained lineage, and compatible
  cumulative reporting without weakening final promotion.

**Commands**

```bash
uv run --no-sync python -m py_compile nanochat/research/cuda_config.py nanochat/research/cuda_supervisor.py nanochat/research/cuda_worker.py nanochat/mixers/kda.py
uv run --no-sync python -m pytest -q
uv run --no-sync research cuda-ownership-supervisor --help
uv build --out-dir /tmp/swa-kda-dist
# also launched the protected worker against a temporary 0b4b247 worktree
# with CUDA hidden; it reached the expected "CUDA is required" result
git diff --check
```

**Artifacts**

- Independent schema/config: `nanochat/research/cuda_config.py` and
  `configs/research/kda_cuda_ownership.toml`.
- Protected staged supervisor/worker:
  `nanochat/research/{cuda_supervisor,cuda_worker}.py`.
- Protected hybrid dispatcher and fail-closed candidate ABI:
  `nanochat/mixers/kda.py`, `nanochat/mixers/cuda_kda/`, `nanochat/gpt.py`,
  `scripts/base_train.py`, and CUDA package-data rules in `pyproject.toml`.
- Program/reporting: `program_kda_cuda_ownership.md`, this stream, and hash-pinned
  `configs/research/archive/kda_training_speed_context.json`.
- Compatibility snapshots remain under `configs/research/archive/`.

**Result**

- Added automatically derived bootstrap, migration, and optimization lanes.
  Candidate models—not humans—produce the naive CUDA implementation.
- Protected routing records exact component/backend events: claimed units cannot
  use FLA, while only unclaimed transitional units may use protected FLA.
- Added append-only anchors/milestones, stale-head protection, explicit human
  retention, separate release runs, Python/FLA calibration, raw paired blocks,
  and JSON/Markdown naive-to-optimized waterfall reports.
- Bootstrap/migration performance is observational; correctness, provenance,
  native build/operator evidence, and sanitizer failures remain invalid.
- Final optimization is stricter: all frozen units project-owned and runtime
  FLA-free. Release still requires the cumulative fixed-anchor 3% lower bound.
- Ownership audit now scans every candidate source, observes distinct registered
  CUDA operators during protected correctness, consumes their return values
  directly, and binds declared native kernel symbols in a bounded profiler pass. Fixed-anchor release reruns correctness,
  runtime ownership, profiling, all four sanitizers, kernel, and interleaved
  training gates.
- Validation passed: `162 passed, 10 skipped`; the skips are the existing
  optional CUDA test markers. Python compilation, CLI help, wheel/sdist packaging, the fixed-anchor worker
  bridge, golden legacy protocol hashes, and `git diff --check` also passed.
- No candidate, calibration, costly GPU campaign, ledger, commit, or tag was
  created while implementing the foundation.

**Next**

- Review the complete diff, then commit and create the immutable foundation tag.
- Initialize and calibrate anchors from that clean tag; the first child can then
  write the simplest correct naive CUDA unit.

## 2026-08-08 [agent] freeze protected foundation

**Context**

- The staged supervisor, worker, routing, candidate ABI, report, documentation,
  and test suite completed internal and independent review.
- Review blockers were closed before freezing: protected routing consumes native
  operator outputs directly; fixed-anchor workers can execute against the older
  anchor; calibration/retention are atomic; and release attempts are retriable.

**Commands**

```bash
git commit -m "Add staged KDA CUDA ownership campaign"
git tag -a kda-cuda-ownership-foundation \
  -m "Protected KDA CUDA-ownership autoresearch foundation" \
  07d8996eb0fe104e6b07d9a5ae4f2aa31e9f49e6
git rev-parse kda-cuda-ownership-foundation^{}
```

**Artifacts**

- Foundation commit and peeled tag target:
  `07d8996eb0fe104e6b07d9a5ae4f2aa31e9f49e6`.
- Immutable tag: `kda-cuda-ownership-foundation`.

**Result**

- The protected foundation is committed and tagged from a clean tree. Existing
  `kda-speed-*` refs were not moved.
- No CUDA campaign ledger, calibration, candidate, or GPU campaign was launched.

**Next**

- Initialize the independent ledger, calibrate Python/FLA operator anchors, and
  give the bootstrap instruction to the first autonomous candidate worker.

## 2026-08-08 [agent] validate real pinned-ref initialization

**Context**

- A temporary ledger smoke was run only after the immutable foundation tag
  existed, to validate real ref peeling without initializing the campaign ledger.

**Commands**

```bash
research cuda-ownership-supervisor init \
  --ledger /tmp/kda-cuda-foundation-smoke.sqlite3
research cuda-ownership-supervisor summary \
  --ledger /tmp/kda-cuda-foundation-smoke.sqlite3
rm /tmp/kda-cuda-foundation-smoke.sqlite3
```

**Artifacts**

- Ephemeral `/tmp` ledger only; it was deleted after the smoke.

**Result**

- Initialization resolved foundation
  `07d8996eb0fe104e6b07d9a5ae4f2aa31e9f49e6`, fixed anchor
  `0b4b24773c2696c23338d7600101d7072b592aa9`, and revision-2 protocol
  `ebb24c0069bd7a760e102dd88cab94f7b6d5253ef469b5b98196beaf929ef4a7`.
- The first summary smoke exposed a set-valued JSON serialization defect. It was
  fixed and regression-tested; the repeated CLI smoke reported
  `bootstrap`, `anchors_calibrated=false`, and `foundation`.
- The actual configured campaign ledger remains uninitialized.

**Next**

- Initialize the configured ledger and calibrate anchors when the human starts
  the campaign, then spawn the first naive-CUDA candidate.


## 2026-08-08 [agent] add ledger-free staged candidate onboarding

**Context**

- A candidate worktree had no safe way to reuse the coordinator environment and
  current protected worker before committing, so early native-build/runtime
  mistakes would consume supervisor turnaround.
- Candidate instructions also needed an exact source/math map, lexical hazards,
  handoff format, and an explicitly narrow first task.

**Commands**

```bash
uv run --no-sync python -m py_compile nanochat/research/cuda_candidate.py nanochat/research/cli.py
uv run --no-sync research cuda-candidate-check --help
uv run --no-sync python -m pytest -q tests/test_cuda_supervisor.py
uv run --no-sync python -m pytest -q
git diff --check
```

**Artifacts**

- Ledger-free checker: `nanochat/research/cuda_candidate.py` and protected CLI
  command `research cuda-candidate-check`.
- Candidate guidance: `program_kda_cuda_ownership.md` and the read-only
  `nanochat/mixers/cuda_kda/README.md`.
- Checker diagnostics default to isolated, retained
  `/tmp/nanochat-cuda-candidate-*` directories; no GPU candidate check was
  launched while implementing the command.

**Result**

- The coordinator command accepts an explicit candidate worktree and derived
  lane, requires an exact staged snapshot with no unstaged/untracked files,
  validates scope/suffixes/forbidden lexical tokens, and invokes the current
  protected worker with the coordinator interpreter.
- Runtime audit performs the candidate build in isolated extension/CUDA caches;
  bounded ownership profiling follows, with all four sanitizers optional for
  iteration. The command does not import/open the supervisor or ledger and
  cannot intake or retain a result.
- Documentation now uses `uv run --no-sync research`, recommends naive
  recurrent-only bootstrap first, distinguishes coordination deadlines from
  protected subprocess and campaign deadlines, and specifies the exact pushed
  commit handoff.
- Validation passed: `166 passed, 10 skipped`; CUDA skips remain the declared
  optional test lane. Focused tests passed `29 passed`. CLI help, compilation,
  and `git diff --check` passed.

**Next**

- Review the combined uncommitted implementation. Before the first child, run
  the checker from the coordinator against a disposable staged recurrent-only
  worktree, then use the emitted absolute `summary.json` in the handoff.


## 2026-08-08 [agent] harden the autonomous CUDA launch boundary

**Context**

- Fresh candidate worktrees could create empty local environments, the host
  lacked `setuptools` for PyTorch extensions, protected backward dispatch was
  ambiguous, and Kineto exposed no SM121 CUDA kernel events on this host.
- Transitional Compute Sanitizer runs also needed to isolate only claimed
  atomic units rather than fail on unclaimed FLA kernels.

**Commands**

```bash
uv sync --extra gpu --group dev
uv run --no-sync research cuda-toolchain-preflight \
  --cache-dir /tmp/nanochat-sm121-final-preflight --sanitizers
uv run --no-sync python -m pytest -q
uv build --out-dir /tmp/nanochat-build-check
```

**Artifacts**

- Protected build/preflight: `nanochat/research/cuda_build.py` and
  `nanochat/research/cuda_preflight.py`.
- Exact protected ABI and candidate checker in
  `nanochat/mixers/cuda_kda/README.md` and
  `nanochat/research/cuda_candidate.py`.
- Ephemeral successful receipt:
  `/tmp/nanochat-sm121-final-preflight.json`; it contains no campaign result and
  is not an initialized ledger or candidate artifact.

**Result**

- A content-addressed `TORCH_LIBRARY` hello op compiled with CUDA 13.1 using
  `compute_121,sm_121`, loaded from the isolated cache, produced exact output,
  and exposed `nanochat_sm121_hello_kernel` through Nsight Systems.
- `memcheck`, `racecheck`, `synccheck`, and `initcheck` all returned zero-error
  summaries. Racecheck's distinct `RACECHECK SUMMARY` format is now handled
  explicitly by the preflight, candidate checker, and supervisor.
- Candidate execution reuses the coordinator interpreter and absolute protected
  worker, pins the protected dispatcher, bridges only candidate `cuda_kda`, and
  profiles declared kernel symbols with Nsight rather than unavailable Kineto
  events. Fixed historical FLA operator timing uses an explicit historical
  bridge instead.
- Chunk and convolution backward dispatch now cross separately registered
  native operators. Transitional sanitizers execute only complete claimed
  atomic units; optimization requires every unit.
- Full validation passed: `172 passed, 10 skipped`. Wheel/sdist construction
  passed and includes the build, preflight, checker, and read-only ABI guide.
- The configured campaign ledger, anchor calibration, and first candidate have
  not been launched.

**Next**

- Commit this launch foundation, create immutable launch-foundation and
  controller tags, push and verify their peeled remote SHAs, then perform a
  temporary initialization smoke before the configured ledger is created.


## 2026-08-08 [agent] publish and verify immutable CUDA launch refs

**Context**

- Launch hardening had passed the full suite, native SM121 preflight, four
  sanitizers, packaging, and a clean controller commit.

**Commands**

```bash
git tag -a kda-cuda-ownership-launch-foundation 0d7d3be... \
  -m "Launch-ready KDA CUDA ownership candidate foundation"
git tag -a kda-cuda-ownership-controller 0d7d3be... \
  -m "Pinned KDA CUDA ownership protected controller"
git push origin main
git push origin refs/tags/kda-cuda-ownership-foundation \
  refs/tags/kda-cuda-ownership-launch-foundation \
  refs/tags/kda-cuda-ownership-controller
research cuda-ownership-supervisor init \
  --ledger /tmp/kda-cuda-launch-smoke.sqlite3
research cuda-ownership-supervisor summary \
  --ledger /tmp/kda-cuda-launch-smoke.sqlite3
```

**Artifacts**

- The temporary initialization ledger was deleted. The configured ledger under
  `runs/` remains uninitialized.

**Result**

- Launch foundation and protected controller both peel to
  `0d7d3be43baad65bf6effc8dff3ea6ce9daf27b8`.
- The original immutable foundation remains
  `07d8996eb0fe104e6b07d9a5ae4f2aa31e9f49e6` and was not moved.
- Remote `main` and all three peeled tag SHAs were independently verified.
- Temporary initialization pinned controller and launch foundation `0d7d3be...`,
  fixed cumulative FLA anchor `0b4b24773c2696c23338d7600101d7072b592aa9`,
  revision 2, and protocol
  `d30336f61d10a40740667c81f37a7916f93b7f8634a3e7584a35ce68ca99aa24`.
  Summary reported uncalibrated anchors, the foundation milestone, and
  `bootstrap` as the next lane.

**Next**

- Initialize the configured ledger and calibrate its anchors only when the
  campaign is explicitly started. Attempt 1 should branch from the launch
  foundation and implement only naive native-CUDA `recurrent_decode`.


## 2026-08-08 [agent] initialize campaign and launch recurrent bootstrap attempt

**Context**

- The user explicitly authorized the protected CUDA autoresearch campaign to run
  autonomously overnight.
- The coordinator was clean at pushed `main`; the configured campaign ledger did
  not yet exist and no GPU process was active.

**Commands**

```bash
.venv/bin/research cuda-ownership-supervisor init \
  --config configs/research/kda_cuda_ownership.toml
.venv/bin/research cuda-ownership-supervisor calibrate \
  --config configs/research/kda_cuda_ownership.toml
git worktree add -b kda-cuda/recurrent-bootstrap-001 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_001 \
  kda-cuda-ownership-launch-foundation
```

**Artifacts**

- Ledger: ignored `runs/kda-cuda-ownership.sqlite3`.
- Calibration:
  `runs/cuda-ownership-supervisor/d30336f61d10/anchor-calibrations/attempt-0001`.
- Candidate worktree:
  `/home/veer/Master/projects/experiment_swa_kda_cuda_attempt_001`.

**Result**

- Initialization pinned launch foundation/controller
  `0d7d3be43baad65bf6effc8dff3ea6ce9daf27b8`, fixed cumulative FLA anchor
  `0b4b24773c2696c23338d7600101d7072b592aa9`, and protocol
  `d30336f61d10a40740667c81f37a7916f93b7f8634a3e7584a35ce68ca99aa24`.
- Python and FLA anchors calibrated successfully. Comparable length-1 recurrent
  medians were 0.170736 ms (Python) and 0.046528 ms (FLA); complete per-shape
  evidence remains in the calibration artifact.
- Autonomous candidate `attempt1-recurrent` was launched from the exact retained
  base with scope limited to a naive native-CUDA `recurrent_decode`; chunk,
  convolution, PTX, optimization, and supervisor actions are excluded.
- No candidate has yet been intaken, run, or retained.

**Next**

- Require a clean pushed candidate, full ledger-free checker including all four
  sanitizers, and exact handoff. Independently verify it before intake.


## 2026-08-08 [agent] repair protected transitional launch blockers

**Context**

- The first staged recurrent candidate compiled and produced complete 20%
  ownership provenance, but ledger-free checking exposed three deterministic
  protected-controller defects before any candidate could pass.
- No candidate had been intaken, so the initialized/calibrated protocol could be
  preserved as blocked evidence and restarted without rewriting an attempt.

**Commands**

```bash
.venv/bin/research cuda-candidate-check ...   --artifact-dir /tmp/kda-cuda-attempt-001-check
.venv/bin/research cuda-candidate-check ...   --artifact-dir /tmp/kda-cuda-attempt-001-check-002
mv runs/kda-cuda-ownership.sqlite3   runs/kda-cuda-ownership-blocked-d30336f6.sqlite3
uv run --no-sync python -m pytest -q
```

**Artifacts**

- First ledger-free failure: `/tmp/kda-cuda-attempt-001-check/summary.json`.
- Second ledger-free failure:
  `/tmp/kda-cuda-attempt-001-check-002/summary.json`.
- Preserved blocked ledger: ignored
  `runs/kda-cuda-ownership-blocked-d30336f6.sqlite3`; its calibration artifact
  remains under protocol directory `d30336f61d10`.

**Result**

- The mandatory build helper returned target `12.1` while provenance used a
  non-canonical lexical `121` check. The helper now reports `sm_121`, and the
  audit requires both `compute_121` and `sm_121` in the real compiler command.
- The runtime import blocker deleted/blocked protected FLA needed by explicitly
  unclaimed transitional units. It now blocks forbidden imports only when a
  candidate-source frame is active, while allowing protected transitional
  routing.
- Profile audit instantiated `_NativeOperatorRecorder` with an unsupported
  argument. The call now matches the protected recorder API.
- Regression tests cover scoped import blocking, canonical build receipts,
  compiler evidence, and recorder invocation. Full validation passed:
  `174 passed, 10 skipped`.
- The controller ref is intentionally changed to a new immutable tag, so the
  previous protocol/ledger is not reused. Candidate sources remain staged and
  unchanged; direct native-op parity checks passed random/extreme shapes and
  chained recurrent state with zero BF16 output delta and FP32 state delta at
  most 3.58e-07.

**Next**

- Commit/tag/push the repaired controller, initialize and calibrate its new
  protocol, remove the candidate's temporary receipt normalization, and rerun
  the complete checker with sanitizers.


## 2026-08-08 [agent] recalibrate repaired controller and qualify recurrent handoff

**Context**

- The protected transitional repair was committed, tagged, pushed, and required
  a fresh protocol/ledger before candidate intake.
- The candidate removed its temporary receipt normalization and reran the exact
  staged snapshot against the repaired controller.

**Commands**

```bash
.venv/bin/research cuda-ownership-supervisor init \
  --config configs/research/kda_cuda_ownership.toml
.venv/bin/research cuda-ownership-supervisor calibrate \
  --config configs/research/kda_cuda_ownership.toml
.venv/bin/research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_001 \
  --lane bootstrap \
  --artifact-dir /tmp/kda-cuda-attempt-001-check-003 \
  --extension-cache /tmp/kda-cuda-attempt-001-check-003/torch-extensions \
  --cuda-cache /tmp/kda-cuda-attempt-001-check-003/cuda-cache \
  --sanitizers
```

**Artifacts**

- Active protocol: `df188a2c3f05c6fa399974dce9d6dcd748ac5d8ef8b00b80f389795654620fd9`.
- Calibration:
  `runs/cuda-ownership-supervisor/df188a2c3f05/anchor-calibrations/attempt-0001`.
- Ledger-free candidate evidence:
  `/tmp/kda-cuda-attempt-001-check-003/summary.json`.
- Candidate commit: `da1dea938d37618306c8dfaf82b5f06ac8628c6c`, pushed branch
  `kda-cuda/recurrent-bootstrap-001`, parent launch foundation `0d7d3be...`.

**Result**

- The new ledger pins controller
  `6cb0c1b6f68fe436d884675081812c5d322dd299`. Python and FLA anchors calibrated;
  comparable recurrent medians were 0.173040 ms and 0.046768 ms respectively.
- Ledger-free runtime audit completed in 113.26 seconds with all 16 protected
  checks, 20% ownership, four observed project recurrent events, and FLA only
  for explicitly unclaimed units.
- Nsight independently observed
  `nanochat_kda_recurrent_decode_kernel` and the registered
  `nanochat_kda::recurrent_decode` operator.
- Memcheck, racecheck, synccheck, and initcheck all completed with their genuine
  zero-error summaries and exercised only the claimed recurrent unit.
- The 10,703-byte recorded patch and both source hashes exactly match the clean
  pushed candidate commit. This is a qualified handoff, not yet an authoritative
  campaign attempt or retained result.

**Next**

- Intake the exact candidate/hypothesis, run the protected bootstrap supervisor,
  inspect the complete saved artifact, and retain only if eligibility is
  `correct_bootstrap`.
