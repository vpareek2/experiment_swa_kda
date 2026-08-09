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


## 2026-08-08 [agent] retain correct recurrent CUDA bootstrap

**Context**

- Candidate `da1dea9...` had a clean, independently verified staged handoff, but
  retention required a separate authoritative supervisor attempt.

**Commands**

```bash
.venv/bin/research cuda-ownership-supervisor intake \
  --base-ref 0d7d3be43baad65bf6effc8dff3ea6ce9daf27b8 \
  --candidate-ref da1dea938d37618306c8dfaf82b5f06ac8628c6c \
  --hypothesis '<verbatim candidate hypothesis>'
.venv/bin/research cuda-ownership-supervisor run --attempt 1
.venv/bin/research cuda-ownership-supervisor retain --attempt 1 \
  --label 'naive recurrent decode bootstrap' \
  --reason '<reviewed protected evidence>'
```

**Artifacts**

- Authoritative attempt:
  `runs/cuda-ownership-supervisor/df188a2c3f05/attempt-00001`.
- Final summary is stored in the append-only ledger and materialized as
  `attempt-00001/summary.json` for direct artifact inspection.
- Retained milestone 2, ordinal 1, commit
  `da1dea938d37618306c8dfaf82b5f06ac8628c6c`.

**Result**

- Every authoritative phase completed: protected tests, 16-check runtime audit,
  Nsight ownership profile, memcheck, racecheck, synccheck, initcheck, and
  baseline/candidate bounded kernel observations.
- Decision was `correct_bootstrap`: ownership progressed from 0% to 20% with
  exactly `recurrent_decode` project-owned. Runtime correctly remained non-FLA-
  free because chunk and convolution are explicitly unclaimed.
- The registered operator, actual SM121 compiler command, mapped-library SHA,
  tracked source SHA, native runtime events, and GPU kernel symbol were bound in
  saved evidence. No forbidden candidate import was attempted.
- Recurrent length-1 median was 0.044976 ms versus the matched Python baseline
  0.172656 ms. Performance is observational in bootstrap and was not used for
  retention; other measurements include transitional FLA units and are not
  attributed to the recurrent intervention.
- The ledger head now derives the `migration` lane. No Git merge or default-
  backend switch was performed.

**Next**

- Start a fresh candidate from retained commit `da1dea9...` and add the smaller
  remaining atomic unit: native causal-convolution forward plus backward. Keep
  recurrent ownership and leave chunk forward/backward unclaimed.


## 2026-08-08 [agent] launch causal-convolution migration attempt

**Context**

- Retained milestone 2 owns recurrent decode only and derives the migration lane.
- The smaller remaining atomic unit is causal-convolution forward plus backward;
  choosing it preserves incremental educational ownership evidence before the
  larger chunk autograd unit.

**Commands**

```bash
git worktree add -b kda-cuda/convolution-migration-002 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_002 \
  da1dea938d37618306c8dfaf82b5f06ac8628c6c
```

**Artifacts**

- Candidate worktree:
  `/home/veer/Master/projects/experiment_swa_kda_cuda_attempt_002`.
- Candidate branch: `kda-cuda/convolution-migration-002`.

**Result**

- Autonomous candidate `attempt2-convolution` was launched from the exact
  retained SHA. Its scope is naive native convolution forward/backward, including
  cache-tail gradients, while preserving recurrent ownership and leaving both
  chunk components unclaimed.
- Independent no-edit reviewer `attempt2-reviewer` was launched for ABI, indexing,
  gradient, provenance, profile, and sanitizer review.
- No attempt-2 candidate has been checked, committed, intaken, run, or retained.

**Next**

- Require a clean pushed strict ownership superset and genuine runtime/Nsight/all-
  sanitizer checker pass before authoritative migration intake.


## 2026-08-08 [agent] add protected convolution cache-gradient gates

**Context**

- Independent attempt-2 review found that the protected runtime audit compared
  convolution output gradients only at `T=65,W=4`; the sanitizer mixed final
  state into the loss but checked only gradient presence.
- A candidate with a dropped or misindexed `grad_final_state` contribution could
  therefore pass every existing protected gate, especially for `T<W` cache
  tails. Attempt 2 was paused before checker execution or intake.

**Commands**

```bash
mv runs/kda-cuda-ownership.sqlite3 \
  runs/kda-cuda-ownership-pre-state-gradients-df188a2c.sqlite3
uv run --no-sync python -m pytest -q
# Protected worker runtime audit against retained recurrent milestone:
PYTHONPATH="$COORD" TORCH_EXTENSIONS_DIR=/tmp/kda-state-gradient-controller-cache \
  "$COORD/.venv/bin/python" "$COORD/nanochat/research/cuda_worker.py" \
  runtime-audit --implementation-root "$RETAINED_WORKTREE" \
  --lane migration --config "$COORD/configs/research/kda_cuda_ownership.toml" \
  --output /tmp/kda-state-gradient-controller-runtime.json
```

**Artifacts**

- Preserved superseded ledger:
  `runs/kda-cuda-ownership-pre-state-gradients-df188a2c.sqlite3`, including the
  authoritative retained recurrent attempt.
- Direct protected validation:
  `/tmp/kda-state-gradient-controller-runtime.json`.

**Result**

- Protected runtime audit now numerically compares convolution output, weight,
  x, and initial-state gradients against the independent oracle for five cases:
  `T=2<W`, `T=4=W`, `T=5>W` without initial state, `T=65>W`, and
  `T=3` with `output_final_state=False`.
- Requested final-state cases mix a nonzero cache-tail cotangent into the loss;
  the false-state case forces the `None` final-state/cotangent ABI. Every initial
  cache is checked for bitwise nonmutation.
- Full validation passed: `175 passed, 10 skipped`. A real protected migration
  runtime audit against retained recurrent commit `da1dea9...` completed all 21
  checks, including all five new cache-gradient cases through protected FLA.
- The controller ref is changed again intentionally. The `df188a2c...` ledger is
  preserved but will not be reused; the recurrent milestone must be reconstructed
  under the newly pinned protocol before attempt 2 can proceed.

**Next**

- Commit/tag/push the state-gradient controller, initialize/calibrate its fresh
  ledger, rerun and retain exact recurrent attempt 1 under that protocol, then
  resume the paused convolution candidate and require it to pass the new cases.

## 2026-08-08 [agent] reconstruct recurrent retention and qualify convolution migration

**Context**

- The state-gradient-gated controller is immutable at
  `56faf5da7f5080c50a22fc666eba32dd6cab9981` under tag
  `kda-cuda-ownership-controller-state-gradient-gates`; remote `main` and the
  peeled tag were already verified at that SHA.
- Its fresh protocol
  `ba64643fd7fd764bab39f99ea83ecf3805522fab3005516f29079806e32a46cf`
  calibrated Python and FLA recurrent medians at 0.1708479971 ms and
  0.0468800012 ms. The superseded `d30336f6...` and `df188a2c...` ledgers
  remain preserved and were not reused or deleted.
- Reconstructed attempt 1 was complete and eligible as `correct_bootstrap`, but
  had not yet advanced the append-only retained head. Attempt 2 remained an
  uncommitted three-file convolution migration candidate from that recurrent
  SHA.

**Commands**

```bash
WORKTREE=../experiment_swa_kda_cuda_attempt_002
.venv/bin/research cuda-ownership-supervisor summary \
  --config configs/research/kda_cuda_ownership.toml --attempt 1
.venv/bin/research cuda-ownership-supervisor retain \
  --config configs/research/kda_cuda_ownership.toml --attempt 1 \
  --label "naive recurrent decode bootstrap" \
  --reason "Reconstructed under the state-gradient-gated controller; protected correctness, 20% native ownership, build/runtime/Nsight evidence, and all four claimed-unit sanitizer gates completed."
git -C "$WORKTREE" add -- \
  nanochat/mixers/cuda_kda/__init__.py \
  nanochat/mixers/cuda_kda/causal_convolution_forward.cu \
  nanochat/mixers/cuda_kda/causal_convolution_backward.cu
.venv/bin/research cuda-candidate-check \
  --config configs/research/kda_cuda_ownership.toml \
  --worktree "$WORKTREE" \
  --lane migration \
  --artifact-dir /tmp/kda-cuda-attempt-002-check-001 \
  --extension-cache /tmp/kda-cuda-attempt-002-check-001/torch-extensions \
  --cuda-cache /tmp/kda-cuda-attempt-002-check-001/cuda-cache --sanitizers
git -C "$WORKTREE" \
  commit -m "Add naive CUDA causal convolution"
git -C "$WORKTREE" \
  push origin HEAD:kda-cuda/convolution-migration-002
```

**Artifacts**

- Reconstructed authoritative attempt:
  `runs/cuda-ownership-supervisor/ba64643fd7fd/attempt-00001`; the missing
  `summary.json` was materialized byte-for-byte from
  `/tmp/kda-ba64643f-attempt1.json` with SHA-256
  `d99c2c76038e6bdf13aca4e95c81d5ad8d258eef095b9237286ba904cbce1dfd`.
- Preserved frozen-tolerance failure:
  `/tmp/kda-cuda-attempt-002-direct-001/parity.log`.
- Corrected direct evidence:
  `/tmp/kda-cuda-attempt-002-direct-002/{exact-t3-after,all-direct-gradients,new-controller-conv-gates}.log`.
- Complete ledger-free checker:
  `/tmp/kda-cuda-attempt-002-check-001/summary.json`.
- Candidate commit `146e9090a6823a9e87c91114e4eec1b8852a6836`, pushed branch
  `kda-cuda/convolution-migration-002`, exact parent
  `da1dea938d37618306c8dfaf82b5f06ac8628c6c`.

**Result**

- Reconstructed attempt 1's runtime, Nsight, build, ownership, and all four
  sanitizer artifacts were independently inspected. The four raw logs report
  genuine zero-error/zero-hazard summaries, the runtime artifact includes all
  five state-gradient cases, and no candidate forbidden import was attempted.
  Retention advanced only the ledger head to recurrent commit `da1dea9...`, 20%
  ownership, owner set `recurrent_decode`, and derived the `migration` lane; no
  candidate was merged.
- The first convolution source, SHA-256
  `a44674dacc151649c2e19593c037c15fe9a542175baeba5261fd4d70c1cfd03e`,
  genuinely failed the exact `T=3` sample: 1/42 unequal outputs, maximum
  absolute difference 0.03125, and maximum relative difference
  0.0076904296875 at frozen 0.005 tolerances. The evidence remains preserved.
- Rounding each BF16 product before FP32 reduction and the reduced
  preactivation to BF16 before FP32 SiLU, then recomputing backward from that
  same rounded preactivation, fixed the failure without relaxing tolerances.
  The corrected forward/backward source hashes are
  `02ef44d8e46d4f0d0ce5d8ae0afa95afbe4d3415362209ff3d29a918aae6d4e6`
  and `c8b88abdc4fed560d4c64ec0fc5806cd5df59273af397a8d94375f4aefffb98c`;
  retained recurrent remained byte-identical at
  `c6c1b5704d062b18c5af93092c380c09ca5d3420d1dfaf2a0c44f1a0bc319f91`.
- The exact staged snapshot passed the current ledger-free migration checker.
  Its runtime audit completed 21 checks including all five cache-gradient
  cases, with exact 40% ownership: recurrent and both convolution units project-
  owned, both chunk units third-party. It recorded no forbidden candidate import
  attempts, the isolated-cache library SHA-256
  `3bd4add77927e9d1fc691df0302717b4fce4c5d7e281b79f95639288e6acd322`,
  actual `compute_121,sm_121` compiler commands, and all reviewed source hashes.
- Protected operator tracing observed three distinct `nanochat_kda::` operators;
  Nsight independently observed recurrent, convolution-forward, and convolution-
  backward kernel symbols. Memcheck, racecheck, synccheck, and initcheck executed
  only the three claimed components and reported genuine zero-error/zero-hazard
  summaries. The recorded staged patch was 20,691 bytes with SHA-256
  `a68f5bf2c91d11a8897c2fca902cd93d11a4b270641e59c3d27c1f9749dd0b1e`,
  exactly matching the committed snapshot.
- Independent static review found no ABI, indexing, gradient, BF16, provenance,
  scope, or ownership blocker. The candidate branch is pushed, its worktree is
  clean, and the remote SHA equals the local SHA.

**Next**

- Commit and push this runbook update so the coordinator is clean. Intake exact
  candidate `146e9090...` from retained base `da1dea9...` with the declared
  hypothesis, run every protected supervisor phase, and retain only after a
  complete reviewed `validated_component` ownership-progress decision.

## 2026-08-08 [agent] retain validated causal-convolution migration

**Context**

- Ledger-free qualification and the pushed exact candidate are diagnostic only;
  retention required a separate authoritative supervisor intake and complete
  attempt from the current clean coordinator.
- The retained head before intake was recurrent-only commit `da1dea9...` at 20%
  ownership in the migration lane. Performance remained explicitly advisory.

**Commands**

```bash
.venv/bin/research cuda-ownership-supervisor intake \
  --config configs/research/kda_cuda_ownership.toml \
  --base-ref da1dea938d37618306c8dfaf82b5f06ac8628c6c \
  --candidate-ref 146e9090a6823a9e87c91114e4eec1b8852a6836 \
  --hypothesis 'Naive independent-element CUDA kernels with oracle-matched BF16 product/preactivation rounding, FP32 accumulation, exact cache-tail indexing, and recomputed SiLU backward should match convolution output/state and x/weight/initial-state gradients while exposing distinct native forward/backward operators and kernels.'
.venv/bin/research cuda-ownership-supervisor run \
  --config configs/research/kda_cuda_ownership.toml --attempt 2
.venv/bin/research cuda-ownership-supervisor retain \
  --config configs/research/kda_cuda_ownership.toml --attempt 2 \
  --label 'naive causal convolution migration' --reason '<reviewed protected evidence>'
```

**Artifacts**

- Authoritative attempt:
  `runs/cuda-ownership-supervisor/ba64643fd7fd/attempt-00002`.
- Its self-contained `summary.json` was materialized from the complete run CLI
  result without changing any evidence value; SHA-256
  `c1bd6025ae001fdc91d22005a7102d5e3d241bae23b455b5c140341b3ec4e57f`.
- Retained milestone 3, ordinal 2, commit
  `146e9090a6823a9e87c91114e4eec1b8852a6836`.

**Result**

- Intake accepted exact parent `da1dea9...`, exact direct-child candidate
  `146e9090...`, the declared hypothesis verbatim, and only the three candidate
  paths. Protected normalized patch SHA-256
  `261e8e107227c1185ff6064921bc1973db73b4bb2d2df13a109792eb337a608c`
  equals the committed binary diff after the controller's documented terminal-
  newline stripping.
- The authoritative attempt completed all mandatory phases: 52 protected tests;
  a 21-check runtime audit including all five cache-gradient cases; ownership
  profile; memcheck, racecheck, synccheck, and initcheck; and matched bounded
  baseline/candidate kernel observations.
- Eligibility was `validated_component` and migration decision was
  `ownership_progress`. Exact ownership is 40% with project-owned
  `recurrent_decode`, `causal_convolution_forward`, and
  `causal_convolution_backward`; only `chunk_forward` and `chunk_backward`
  remain third-party and use explicitly transitional FLA.
- The authoritative isolated library has SHA-256
  `7e5e7f8e3c2e5dc20a4cff57b0eba9cac700463c131706313a62f6a0ad953f5f`.
  Saved provenance binds the mapped library, all three exact source hashes,
  actual `compute_121,sm_121` commands, three registered operators, and three
  independently observed Nsight kernel symbols. No candidate forbidden import
  was attempted.
- Raw sanitizer evidence is genuine: memcheck, synccheck, and initcheck each
  report zero errors; racecheck reports zero hazards, errors, and warnings.
  Every sanitizer executed only the three claimed project units.
- Performance decision was `observed`, advisory, with no paired interval; it was
  not used for retention and no speed or quality claim is made.
- Explicit retention advanced only the append-only ledger head to milestone 3.
  It did not merge the candidate, move a tag, change the default backend, or
  claim a fully FLA-free implementation. The derived next lane remains
  `migration` for the atomic chunk forward/backward pair.

**Next**

- Commit and push this runbook update, then create a fresh attempt-3 worktree and
  branch from exact retained commit `146e9090...`. Implement the simplest correct
  naive native `chunk_forward` plus `chunk_backward` atomic unit, preserve all
  three retained operators, and require the same staged checker, exact handoff,
  authoritative supervisor run, evidence review, and explicit retention.

## 2026-08-08 [agent] launch final naive chunk migration

**Context**

- Retained milestone 3 owns recurrent decode and causal-convolution
  forward/backward for exact 40% project-owned CUDA coverage.
- The only remaining atomic unit is `chunk_forward` plus `chunk_backward` (60%).
  Completing it correctly should produce the first fully project-owned,
  runtime-FLA-free naive CUDA milestone before any optimization.

**Commands**

```bash
git worktree add -b kda-cuda/chunk-migration-003 \
  ../experiment_swa_kda_cuda_attempt_003 \
  146e9090a6823a9e87c91114e4eec1b8852a6836
```

**Artifacts**

- Candidate worktree: `../experiment_swa_kda_cuda_attempt_003`.
- Candidate branch: `kda-cuda/chunk-migration-003`.
- Exact base: `146e9090a6823a9e87c91114e4eec1b8852a6836`.

**Result**

- A fresh clean attempt-3 worktree was created from the exact retained head.
- Autonomous implementation work was delegated with candidate scope limited to
  ordinary naive CUDA source under `nanochat/mixers/cuda_kda/`. It must preserve
  the three retained units, implement chunk forward/backward atomically, use no
  PTX or optimization, and leave protected code, the README, ledger, supervisor,
  and runbook untouched.
- No attempt-3 source change, staged checker, commit, intake, supervisor run, or
  retention exists yet.

**Next**

- Review the source and narrow direct evidence, stage only the intended candidate
  files, and require a fresh complete migration checker with runtime FLA-free
  100% ownership, five distinct operators/kernel symbols, and all four genuine
  claimed-only sanitizer summaries before exact commit/push and supervisor
  intake.

## 2026-08-08 [agent] qualify complete naive CUDA backend

**Context**

- Attempt 3 adds the last atomic unit, native `chunk_forward` plus
  `chunk_backward`, while preserving the retained recurrent and convolution
  units. Correctness and honest ownership remain the migration gates;
  performance is observational and the implementation intentionally uses a
  serial, memory-heavy naive backward.
- Candidate direct testing and the first staged checker exposed two genuine
  candidate correctness/compatibility defects. Their artifacts and caches are
  preserved rather than reused.

**Commands**

```bash
WORKTREE=../experiment_swa_kda_cuda_attempt_003
git -C "$WORKTREE" add -- \
  nanochat/mixers/cuda_kda/__init__.py \
  nanochat/mixers/cuda_kda/chunk.cu
.venv/bin/research cuda-candidate-check \
  --config configs/research/kda_cuda_ownership.toml \
  --worktree "$WORKTREE" --lane migration \
  --artifact-dir /tmp/kda-cuda-attempt-003-check-001 \
  --extension-cache /tmp/kda-cuda-attempt-003-check-001/torch-extensions \
  --cuda-cache /tmp/kda-cuda-attempt-003-check-001/cuda-cache --sanitizers
# After candidate-only fixes and fresh direct tests:
.venv/bin/research cuda-candidate-check \
  --config configs/research/kda_cuda_ownership.toml \
  --worktree "$WORKTREE" --lane migration \
  --artifact-dir /tmp/kda-cuda-attempt-003-check-002 \
  --extension-cache /tmp/kda-cuda-attempt-003-check-002/torch-extensions \
  --cuda-cache /tmp/kda-cuda-attempt-003-check-002/cuda-cache --sanitizers
git -C "$WORKTREE" commit -m "Add naive CUDA chunk KDA"
git -C "$WORKTREE" push origin HEAD:kda-cuda/chunk-migration-003
```

**Artifacts**

- Original normal direct evidence and harness-import failure:
  `/tmp/kda-attempt-003-direct-20260808-a`.
- Genuine tiny-normalization-gradient failure:
  `/tmp/kda-attempt-003-direct-tiny-001/tiny.log`; corrected exact-source pass:
  `/tmp/kda-attempt-003-direct-tiny-003/tiny.log`.
- Final exact-source normal parity, cotangent matrix, noncontiguous input, and
  production-shape safety evidence:
  `/tmp/kda-attempt-003-direct-20260808-c`,
  `/tmp/kda-attempt-003-direct-cotangents-003`, and
  `/tmp/kda-attempt-003-direct-noncontiguous-001`.
- Preserved failed checker: `/tmp/kda-cuda-attempt-003-check-001/summary.json`.
- Complete final checker: `/tmp/kda-cuda-attempt-003-check-002/summary.json`.
- Clean pushed candidate commit `4d1a3b231da2c99882324efbda5306a1815e21c7`,
  exact parent `146e9090a6823a9e87c91114e4eec1b8852a6836`, branch
  `kda-cuda/chunk-migration-003`.

**Result**

- The initial chunk implementation used the projection-form normalization
  derivative even when `F.normalize` clamps a tiny norm at epsilon. A genuine
  BF16 tiny-norm case failed 9/48 query-gradient elements with maximum absolute
  difference 4,194,304 and maximum relative difference 0.134765625. The raw
  failure remains preserved.
- The candidate fixed the derivative by suppressing the projection term below
  the `1e-12` norm clamp while retaining the `1/eps` linear derivative. The
  exact final tiny-norm evidence passes all gradients; Q/K/V/raw/beta were exact,
  with maximum A/dt absolute deltas below `3e-11`.
- The first full checker then failed validly during the protected prefix-
  causality case with `RuntimeError: q must be contiguous CUDA bfloat16`; profile
  and sanitizers were skipped. The current protected audit passes B>1 prefix
  views whose leading-length slices are noncontiguous. The failed staged source
  SHA-256 was
  `88a443a349e8dd8465fec4c5c55d58c13a14b243890d17a8ad1360f9570ce052`
  and its 29,347-byte staged patch SHA-256 was
  `c825f45212453a085aaf34ee9b761abc027ca92ac006280a88ef6b98e090faad`.
- The candidate-only fix safely accepts the protected views by materializing
  private contiguous activation copies inside the native forward/backward
  wrappers and returning contiguous shape-correct gradients. A direct B=2,
  H=2 noncontiguous prefix forward/backward comparison then passed without
  changing any numerical, ownership, or protected gate.
- Exact final source hashes are
  `8e0d83f9a4349f9e1d045de7e02056a43bdbf74536a2b2d06e0b10309b5fb53c`
  for `__init__.py` and
  `67c2d2f38ab471cd35c8d426ee481ea65b63d02f5613a988086344be4d2563fa`
  for `chunk.cu`. README and all three retained source files remain byte-
  identical. Final normal boundary/state parity through T=257, unequal K/V,
  seven public gradients, initial-state gradient, independent output/state/both
  cotangents, two-dimensional dt bias, None final cotangent, and production T=65
  safety all passed frozen tolerances.
- The fresh final checker completed every phase. Runtime completed 21 checks,
  owned fraction is exactly 1.0, `migration_ready=true`, `runtime_fla_free=true`,
  no FLA module remained loaded or executed, and no forbidden candidate import
  was attempted. All five components are honestly project-owned.
- Protected operator tracing and Nsight independently observed five distinct
  operators and the five declared recurrent, convolution-forward,
  convolution-backward, chunk-forward, and chunk-backward kernel symbols. The
  isolated library SHA-256 is
  `4ca61a85ea9d6a4716c57163a9649ecfc2a05a677d6b99e117fb3be2b44558eb`;
  its receipt records actual `compute_121,sm_121` commands and all exact source
  hashes.
- Memcheck, racecheck, synccheck, and initcheck each executed all five claimed
  units only. Raw logs report zero errors; racecheck reports zero hazards,
  errors, and warnings. The exact final staged patch is 30,031 bytes with
  SHA-256 `e70f250ce86ad981549398482eb48d845553df9652dba2c8d4d2edfd660464fb`.
- Candidate commit parent and remote branch SHA were independently verified and
  the worktree is clean. This is a qualified diagnostic handoff, not yet an
  authoritative campaign result or retained FLA-free milestone.
- Two additional preserved failures were diagnostic-script defects rather than
  candidate failures: an incorrect protected-module import in the first script,
  and a state-only comparison that treated an oracle `None` gradient as
  different from the native ABI's required explicit zero tensor. Corrected fresh
  scripts passed; no gate or candidate math was weakened.

**Next**

- Commit and push this runbook update so the coordinator is clean. Intake exact
  candidate `4d1a3b2...` from retained base `146e9090...` with the mechanistic
  hypothesis, run and inspect every authoritative supervisor phase, and retain
  the immutable naive fully project-owned CUDA milestone only if eligibility is
  `fla_free_naive`, runtime is FLA-free, and all mandatory evidence completes.

## 2026-08-08 [agent] retain immutable naive FLA-free CUDA milestone

**Context**

- The complete staged checker was diagnostic; the first fully project-owned
  CUDA backend still required exact clean intake, an authoritative supervisor
  attempt, explicit evidence review, and append-only retention.
- Migration performance is observational. The naive chunk backward is expected
  to be unsuitable for optimization until parallelized, so a frozen performance
  timeout must be reported as a censored observation rather than a poor score or
  a correctness failure.

**Commands**

```bash
.venv/bin/research cuda-ownership-supervisor intake \
  --config configs/research/kda_cuda_ownership.toml \
  --base-ref 146e9090a6823a9e87c91114e4eec1b8852a6836 \
  --candidate-ref 4d1a3b231da2c99882324efbda5306a1815e21c7 \
  --hypothesis 'A deterministic naive CUDA migration unit can implement chunk forward with one independent thread per [B,H,V] recurrence and chunk backward with one deterministic thread per head, recomputing FP32 state history and reverse-mode adjoints without atomics. This should exactly preserve V-first state/FP32 recurrence semantics, expose distinct chunk forward/backward kernels, make all five components project-owned and runtime FLA-free, while accepting intentionally poor observational performance/memory.'
.venv/bin/research cuda-ownership-supervisor run \
  --config configs/research/kda_cuda_ownership.toml --attempt 3
.venv/bin/research cuda-ownership-supervisor retain \
  --config configs/research/kda_cuda_ownership.toml --attempt 3 \
  --label 'naive FLA-free project CUDA milestone' \
  --reason '<exact reviewed protected evidence and advisory-timeout qualification>'
.venv/bin/research cuda-ownership-supervisor report \
  --config configs/research/kda_cuda_ownership.toml --format json
```

**Artifacts**

- Authoritative attempt:
  `runs/cuda-ownership-supervisor/ba64643fd7fd/attempt-00003`.
- Complete materialized run result: `attempt-00003/summary.json`, SHA-256
  `648dad978e0ded68bfe3ecd0d38edb66468562ab9c4c1177d573b7eab36c1f44`.
- Candidate performance log is preserved as
  `attempt-00003/candidate-kernel.log`; the censored run emitted no
  `candidate-kernel.json`.
- Retained milestone 4, ordinal 3, commit
  `4d1a3b231da2c99882324efbda5306a1815e21c7`.
- Immutable report anchor `fla_free_naive` points to that exact commit and
  authoritative artifact.

**Result**

- Intake accepted only `__init__.py` and `chunk.cu` from exact direct-child
  commit `4d1a3b2...`. Its 30,030-byte controller-normalized patch has SHA-256
  `10837dd19f68a5e374d64efaa745f7092b9971ab37599e30b34b5bd3e809c32a`,
  exactly matching the committed 30,031-byte binary diff after documented
  terminal-newline stripping.
- The authoritative attempt completed 52 protected tests, all 21 runtime checks,
  ownership profile, all four sanitizers, and the baseline observation.
  Eligibility is `fla_free_naive`; migration is `migration_ready`; exact
  ownership is 100% across all five units; runtime is FLA-free; loaded forbidden
  modules and forbidden candidate attempts are both empty.
- The authoritative isolated mapped library has SHA-256
  `12f839d387771707f6905e25cc0933f0f3df503e3c0b95ccac62756e8b22433e`.
  It binds the exact four `.cu` source hashes and actual
  `compute_121,sm_121` compiler commands. Five registered operators and all five
  declared kernel symbols were independently observed by protected tracing and
  Nsight.
- Memcheck, synccheck, and initcheck each report zero errors. Racecheck reports
  zero hazards, errors, and warnings. Each sanitizer ran exactly the five
  project-owned components and recorded no forbidden import attempt.
- The candidate kernel observation reached the frozen 180-second limit and
  emitted no JSON. The controller explicitly treats this as an advisory timeout
  in migration, so the overall attempt correctly remains complete with
  performance decision `observed`, `advisory=true`, and a 180-second censored
  observation. It supports no performance, memory, throughput, quality, or
  stability claim.
- Explicit retention advanced only the append-only ledger head. The report now
  contains immutable `fla_free_naive` anchor commit `4d1a3b2...`, labeled
  `First complete FLA-free naive CUDA backend`, and derives the `optimization`
  lane. No Git merge, tag movement, default-backend switch, or quality training
  occurred.

**Next**

- Commit and push this runbook update. Begin optimization only from exact
  retained commit `4d1a3b2...` in a fresh worktree, keep all five units
  project-owned and runtime FLA-free, and choose a profile-supported hypothesis.
  First remove the naive chunk backward's serial-head trajectory bottleneck so
  the protected 4k matched candidate observation can complete; then require the
  full optimization correctness/profile/sanitizer gates and nine alternating
  matched 4k parent/candidate blocks before any retention claim.

## 2026-08-08 [agent] profile naive bottleneck and launch first optimization

**Context**

- Milestone 4 is the immutable first complete project-owned CUDA backend and the
  ledger derives the strict optimization lane.
- Its migration performance observation was censored, so the first optimization
  needs a profile-supported single-axis hypothesis without treating the timeout
  as a numeric score.

**Commands**

```bash
# Exact retained implementation, one T=65 production-shape forward/backward.
nsys profile --trace=cuda,nvtx,osrt --sample=none \
  -o /tmp/kda-naive-profile-001/naive-t65 \
  .venv/bin/python /tmp/kda-naive-profile-001/profile_naive.py
nsys stats --report cuda_gpu_kern_sum --format csv \
  /tmp/kda-naive-profile-001/naive-t65.nsys-rep
git worktree add -b kda-cuda/chunk-backward-parallel-004 \
  ../experiment_swa_kda_cuda_attempt_004 \
  4d1a3b231da2c99882324efbda5306a1815e21c7
```

**Artifacts**

- Nsight report: `/tmp/kda-naive-profile-001/naive-t65.nsys-rep`.
- Kernel summary: `/tmp/kda-naive-profile-001/cuda_gpu_kern_sum.csv`.
- Candidate worktree: `../experiment_swa_kda_cuda_attempt_004`.
- Branch: `kda-cuda/chunk-backward-parallel-004`; exact base
  `4d1a3b231da2c99882324efbda5306a1815e21c7`.

**Result**

- Nsight attributes 418.410944 ms, or 99.2% of captured GPU kernel time, to the
  single-thread-per-head naive chunk-backward kernel at T=65, K=V=128. Chunk
  forward used 3.249216 ms, or 0.8%. This is profile localization, not a matched
  optimization result or quality claim.
- The first optimization candidate was delegated with one primary axis:
  parallelize chunk backward/state-history work using ordinary project-owned
  CUDA while preserving all five ownership claims, runtime FLA freedom, exact
  ABI/math, and sanitizer safety. PTX and unrelated changes remain excluded.
- No optimization source change, staged checker, commit, intake, matched
  training block, or retention exists yet.

**Next**

- Require exact direct parity and matched local timing, then a full staged
  optimization checker with all ownership/profile/sanitizer gates. Before
  authoritative intake, account explicitly for the retained parent's censored
  kernel behavior; do not weaken the strict optimization baseline, candidate,
  nine-pair training, memory, drift, or confidence gates.

## 2026-08-08 [agent] qualify and preserve parallel chunk-backward candidate

**Context**

- The retained naive profile localized 99.2% of T=65 GPU kernel time to the
  one-thread-per-head chunk-backward kernel. Attempt 4 changes only that primary
  axis while preserving the complete five-unit project-owned backend.
- Exact hypothesis prepared for intake:
  `Nsight localized 99.2% of retained T=65 GPU kernel time (418.410944 ms) in the one-thread-per-head chunk backward kernel. Replacing that launch with one 128-thread block per head while preserving sequential time, V-first FP32 state recurrence, phase barriers, and deterministic ordered reductions should expose independent value-row, key-column, and state-matrix work, preserve all five project-owned/runtime-FLA-free correctness and sanitizer gates, leave the dominant history workspace unchanged, and let the strict 4k candidate observation complete.`

**Commands**

```bash
.venv/bin/research cuda-candidate-check \
  --config configs/research/kda_cuda_ownership.toml \
  --worktree ../experiment_swa_kda_cuda_attempt_004 \
  --lane optimization \
  --artifact-dir /tmp/kda-cuda-attempt-004-check-001/artifact \
  --extension-cache /tmp/kda-cuda-attempt-004-check-001/torch-extensions \
  --cuda-cache /tmp/kda-cuda-attempt-004-check-001/cuda-cache \
  --sanitizers
git -C ../experiment_swa_kda_cuda_attempt_004 commit \
  -m "Parallelize CUDA chunk KDA backward"
git -C ../experiment_swa_kda_cuda_attempt_004 push -u origin \
  kda-cuda/chunk-backward-parallel-004
```

**Artifacts**

- Direct/runtime/timing/profile diagnostics:
  `/tmp/kda-cuda-attempt-004-direct-001`,
  `/tmp/kda-cuda-attempt-004-runtime-002`,
  `/tmp/kda-cuda-attempt-004-timing-001`, and
  `/tmp/kda-cuda-attempt-004-profile-001`.
- Full ledger-free checker:
  `/tmp/kda-cuda-attempt-004-check-001/artifact`; summary SHA-256
  `5f16181e4d2a095750a7b88091d463efe4df324e298e2cbaaf1b01490f795126`.
- Exact one-file staged patch was 17,476 bytes, SHA-256
  `b0692916b06b7051396e924373f55f346331daf656258dc011c9e7e30c0e3003`.
- Candidate source `chunk.cu` SHA-256
  `ca75d9102f44f202227d4ff25d67181d6ed6b6d3fa5beb33bb51c7183dc1564d`;
  checker isolated library SHA-256
  `b59a2979f49d242a853540082aaef3cd4b005c87a921907b9c9b03a5763f8a10`.
- Candidate commit `613b0759d3b954cae984274b267a83a5c4ac46ff`, exact parent
  `4d1a3b231da2c99882324efbda5306a1815e21c7`, pushed branch
  `kda-cuda/chunk-backward-parallel-004`.

**Result**

- The source keeps time sequential but uses a 128-thread block per head to
  parallelize independent value rows, key columns, and V-by-K adjoint updates.
  Barriers separate every dependency; reductions are deterministic and no
  atomics, PTX, FLA route, or extra full-history workspace was introduced.
- Direct oracle comparisons passed all boundaries, unequal B/H/K/V, nonzero
  initial state, independent cotangents, tiny normalization, and noncontiguous
  prefixes. Matched local chunk-backward medians improved from 414.931 to 7.507
  ms at T=65, 1628.804 to 29.593 ms at T=256, and 6657.708 to 129.976 ms at
  T=1024, with identical peak allocations. These are diagnostic operator
  measurements, not protected optimization or training evidence.
- The full optimization checker completed all 21 runtime checks, exact 1.0
  ownership, runtime FLA freedom, five operator and five Nsight symbols,
  provenance, and genuine zero memcheck/racecheck/synccheck/initcheck summaries.
  Two independent read-only reviews found no staging or checker blocker.
- Commit and remote branch identity were verified; the candidate worktree is
  clean. Nothing was merged and the default backend remains unchanged.
- A frozen-gate entry problem remains separate from candidate validity: the
  exact retained parent already exhausted migration's 180-second kernel worker
  at at most T=1024/25 iterations, while optimization demands T=4096/50. Direct
  evidence measures the parent at 6.658 seconds for one T=1024 backward. The
  supervisor requires both parent and candidate kernel workers to complete
  before any of nine matched blocks, and the program freezes those timeouts.
  This cannot be bypassed or treated as a numeric score.

**Next**

- Intake the exact pushed child and run the authoritative supervisor unchanged
  to materialize the protected outcome. If the retained-parent kernel worker
  times out as predicted, record attempt 4 as invalid with exact censored
  evidence, do not retain or claim optimization, and stop rather than weaken a
  frozen timeout or skip the nine matched blocks.

## 2026-08-08 [agent] preserve strict optimization entry failure

**Context**

- The exact checker-qualified and pushed attempt-4 candidate was intaken to
  exercise the frozen optimization supervisor without changing or bypassing any
  parent, kernel, training, timeout, memory, drift, or confidence gate.
- The retained parent is intentionally the immutable naive milestone. Its
  migration kernel observation had already been censored at 180 seconds.

**Commands**

```bash
.venv/bin/research cuda-ownership-supervisor intake \
  --config configs/research/kda_cuda_ownership.toml \
  --base-ref 4d1a3b231da2c99882324efbda5306a1815e21c7 \
  --candidate-ref 613b0759d3b954cae984274b267a83a5c4ac46ff \
  --hypothesis "<verbatim profile-supported hypothesis above>"
.venv/bin/research cuda-ownership-supervisor run \
  --config configs/research/kda_cuda_ownership.toml --attempt 4
.venv/bin/research cuda-ownership-supervisor summary \
  --config configs/research/kda_cuda_ownership.toml
```

**Artifacts**

- Authoritative attempt:
  `runs/cuda-ownership-supervisor/ba64643fd7fd/attempt-00004`.
- Candidate kernel payload SHA-256
  `a319cf1d14f704c2731347e2118faad8596be03685a9620b59e0d3582c4a9af2`.
- Runtime audit SHA-256
  `d6c75798cd7377b614af130010691832a9a577d6288e9331c281f56c01df9d5d`;
  ownership profile SHA-256
  `a90136d3719a871cfbf69896fc4baffd55b27ae033134a4aff25475e94127243`.
- Authoritative isolated mapped library SHA-256
  `694fbdcf1ea81dfdc312237aaae360466f1d4330b5adf4e31a8e0eae17facd83`.
- Intake normalized patch SHA-256
  `fcc11b13153fed7fab7a7162f49cf02cd840979f08c95cf6a004a289356e8950`.

**Result**

- Intake accepted attempt 4 in the optimization lane from exact milestone 4.
  Candidate `613b0759...` has exact parent `4d1a3b2...` and changes only
  `nanochat/mixers/cuda_kda/chunk.cu`.
- All mandatory pre-timing candidate evidence completed: 52 protected tests,
  all 21 runtime checks, exact 1.0 ownership, runtime FLA freedom, five operator
  traces, five Nsight kernel symbols, complete SM121 build/source/library
  provenance, and genuine zero memcheck/racecheck/synccheck/initcheck results.
- The exact retained-parent kernel worker then timed out after the frozen 180.0
  seconds and emitted no `baseline-kernel.json`. This is a censored invalid
  worker, not a numeric baseline score. The candidate worker completed all 22
  rows and 50 samples per row; its protected but unpaired T=4096 medians were
  193.697 ms for chunk forward and 727.902 ms for chunk forward/backward.
- Because optimization requires both kernel workers to complete before matched
  training, the supervisor launched zero of the nine required paired blocks.
  No paired confidence interval, peak-memory comparison, kernel-regression
  comparison, or baseline-drift measurement exists.
- The authoritative terminal row is exactly `status=invalid`,
  `migration_decision=invalid`, `performance_decision=invalid`,
  `eligibility_decision=not_retainable`, with reason `protected
  correctness/safety/evidence gate failed`; summary and milestone id are null.
  An independent reviewer confirmed retention is categorically forbidden.
- The retained head and immutable naive anchor remain
  `4d1a3b231da2c99882324efbda5306a1815e21c7`, all five units owned,
  ownership 1.0, runtime FLA-free; the derived lane remains optimization.
  Candidate and coordinator worktrees are clean, remote refs are verified, and
  no supervisor, worker, sanitizer, compiler, or GPU process remains.

**Next**

- Stop the current frozen optimization lane. Every new candidate would inherit
  the same exact retained baseline, and the protected supervisor requires that
  baseline's stricter T=4096/50 kernel worker to complete before block 0. A
  retry, advisory reinterpretation, shortened payload, raised timeout, skipped
  parent, or retention would weaken/change the frozen protocol.
- Preserve attempt 4 and branch `kda-cuda/chunk-backward-parallel-004` as
  diagnostic evidence only. Do not retain, rerun, merge, score, or claim an
  optimized backend. Further work requires explicit human authorization for a
  protected protocol redesign and re-freeze; the complete naive project-owned,
  runtime-FLA-free milestone remains the valid campaign result.

## 2026-08-08 [agent] size and prepare human-authorized timeout re-freeze

**Context**

- Authoritative attempt 4 proved that the exact retained naive parent, not the
  parallel-backward candidate, exhausted the original 180-second optimization
  kernel worker before block 0. The human explicitly authorized a protected
  timeout-only re-freeze and directed that the slow parent be allowed to finish.
- Original protocol `ba64643fd7fd764bab39f99ea83ecf3805522fab3005516f29079806e32a46cf`,
  old ledger, namespace, controller tags, candidates, and invalid attempt remain
  immutable. Before the diagnostic, the old ledger was 647,168 bytes with
  SHA-256 `14bce91dee58aaa21a59c87be24eb37b821472d8711ce2fc310553d77331267c`.

**Commands**

```bash
# Run once from each exact clean candidate worktree with distinct extension
# caches; these are timeout-sizing diagnostics, not campaign comparisons.
.venv/bin/python -m scripts.base_train --seed 42 --depth 6 --head-dim 128 \
  --window-pattern K --kda-backend project_cuda --no-force-final-full \
  --max-seq-len 4096 --device-batch-size 2 --total-batch-size 32768 \
  --num-iterations 7 --eval-every -1 --core-metric-every -1 \
  --sample-every -1 --save-every -1 --model-tag <diagnostic-label> \
  --run dummy
```

**Artifacts**

- `/tmp/kda-timeout-refreeze-block-diagnostic-001/parent.log` from exact
  `4d1a3b231da2c99882324efbda5306a1815e21c7`.
- `/tmp/kda-timeout-refreeze-block-diagnostic-001/candidate.log` from exact
  `613b0759d3b954cae984274b267a83a5c4ac46ff`.
- Diagnostic manifest:
  `/tmp/kda-timeout-refreeze-block-diagnostic-001/diagnostic-manifest.json`,
  SHA-256 `5b6a58a09f4ece057f796b9dd284d2f8195ca39dc8b286c3fdaa09ecd56bee0a`.

**Result**

- Both exact seven-step, 4k, batch-32768, depth-6 project-CUDA trainers
  completed. The retained parent used 8,932.104 summed step seconds; the
  candidate used 239.756 seconds. Both reported the same 5,511.408 MiB peak.
  These numbers size process ceilings only and are not paired campaign evidence
  or a retention result.
- The parent process survived a notebook wait interruption and was recovered by
  PID rather than killed or duplicated. Reopening the persisted writer during
  kernel recovery truncated the first 2,107 already-written warning/header
  bytes; the final file preserves a 2,107-byte NUL hole followed by all seven
  `RESEARCH_TRAIN_STEP` records and the complete `RESEARCH_TRAIN_RESULT`. This
  exact recovery defect is recorded in the manifest and no source or gate was
  changed because of it.
- The declared `ceil_to_300(1.5 * slower summed step seconds)` rule selects a
  13,500-second training-block ceiling. The kernel ceiling is 3,600 seconds,
  based on the retained T=65/T=256/T=1024 scaling and the unchanged T=4096,
  10-warmup, 50-sample worker payload.
- The prepared config changes only the controller ref, fresh ledger path, and
  these two ceilings. All shapes, samples, nine discovery pairs, 15 release
  pairs, order, seed, correctness/ownership/profile/sanitizer gates, confidence
  interval, effect floor, memory limit, kernel-regression limit, and drift limit
  remain unchanged.
- The first read-only allowed-diff helper compared old canonical JSON lists with
  new dataclass tuples and failed after reporting representation-only collection
  paths. It mutated nothing. JSON-roundtripping the new value before comparison
  fixed the diagnostic; the corrected helper reported exactly the four declared
  fields and new protocol SHA `6fdb0ec11d7efb82ae67bf39997f3601eae08026f5ec12f719f21f1c7c916e7c`.

**Next**

- Run protected/static tests and an exact resolved-config diff against the old
  ledger. If clean, commit once, create the new immutable annotated controller
  tag `kda-cuda-ownership-controller-timeout-refreeze`, push and verify it, then
  initialize only `runs/kda-cuda-ownership-timeout-refreeze.sqlite3`. Calibrate
  fresh anchors and rerun the exact retained milestone chain; never copy old
  SQLite rows or reuse old artifacts.

## 2026-08-08 [agent] freeze and calibrate extended-timeout protocol

**Context**

- The timeout-only diff passed 40 focused supervisor tests and the full suite
  with 177 passed and 10 skipped. An independent pre-tag review returned PASS.
- The corrected canonical config comparison reported exactly four changes from
  the old ledger: controller ref, ledger path, kernel ceiling, and training-block
  ceiling. No experimental payload or decision gate changed.

**Commands**

```bash
git commit -m "Refreeze CUDA ownership timeouts"
git tag -a kda-cuda-ownership-controller-timeout-refreeze \
  c74bc7553d7c518899f71daef9a33b3464d08e2a \
  -m "Protected KDA CUDA timeout-only re-freeze"
git push origin main
git push origin refs/tags/kda-cuda-ownership-controller-timeout-refreeze
.venv/bin/research cuda-ownership-supervisor init \
  --config configs/research/kda_cuda_ownership.toml
.venv/bin/research cuda-ownership-supervisor calibrate \
  --config configs/research/kda_cuda_ownership.toml
```

**Artifacts**

- Immutable controller commit/tag:
  `c74bc7553d7c518899f71daef9a33b3464d08e2a` /
  `kda-cuda-ownership-controller-timeout-refreeze`.
- New protocol SHA:
  `6fdb0ec11d7efb82ae67bf39997f3601eae08026f5ec12f719f21f1c7c916e7c`.
- Fresh ledger: `runs/kda-cuda-ownership-timeout-refreeze.sqlite3`.
- Fresh anchor calibration:
  `runs/cuda-ownership-supervisor/6fdb0ec11d7e/anchor-calibrations/attempt-0001`;
  calibration summary SHA-256
  `8be236ed23a4468d9e4a6bb7207f7ec916da580c9bba4493832f747e3156dbc4`.

**Result**

- Remote `main` and the annotated tag were pushed and the tag peels exactly to
  `c74bc755...`. Every `CONTROLLER_PATHS` byte in the coordinator matches that
  immutable tag.
- Fresh initialization pinned unchanged foundation
  `0d7d3be43baad65bf6effc8dff3ea6ce9daf27b8`, unchanged cumulative anchor
  `0b4b24773c2696c23338d7600101d7072b592aa9`, and controller `c74bc755...`.
- Python and FLA anchors completed in the new namespace. The new ledger is
  independent and initially 122,880 bytes after calibration. The old ledger
  remains 647,168 bytes with unchanged SHA-256
  `14bce91dee58aaa21a59c87be24eb37b821472d8711ce2fc310553d77331267c`.
- No candidate evidence has yet been imported or copied. The fresh retained head
  is the unchanged protected foundation, and reconstruction must rerun every
  candidate under this protocol.

**Next**

- Reconstruct the exact retained chain with fresh intake/run/review/retain:
  `da1dea938...`, then `146e9090...`, then `4d1a3b231...`. Use the original
  verbatim hypotheses but new evidence-backed retention reasons. Stop on any
  mismatch. Only after the fresh FLA-free milestone is retained may exact
  candidate `613b0759...` re-enter optimization.

## 2026-08-08 [agent] reconstruct recurrent and convolution milestones

**Context**

- The fresh timeout-refreeze ledger was calibrated and contained only its
  foundation. Reconstruction reruns exact historical candidate commits and
  hypotheses; it does not import decisions or artifacts from the old ledger.

**Commands**

```bash
# Attempt 1: exact foundation -> recurrent-only candidate.
.venv/bin/research cuda-ownership-supervisor intake ... \
  --base-ref 0d7d3be43baad65bf6effc8dff3ea6ce9daf27b8 \
  --candidate-ref da1dea938d37618306c8dfaf82b5f06ac8628c6c
.venv/bin/research cuda-ownership-supervisor run ... --attempt 1
.venv/bin/research cuda-ownership-supervisor retain ... --attempt 1 \
  --label "naive recurrent decode bootstrap"

# Attempt 2: retained recurrent -> causal-convolution atomic unit.
.venv/bin/research cuda-ownership-supervisor intake ... \
  --base-ref da1dea938d37618306c8dfaf82b5f06ac8628c6c \
  --candidate-ref 146e9090a6823a9e87c91114e4eec1b8852a6836
.venv/bin/research cuda-ownership-supervisor run ... --attempt 2
.venv/bin/research cuda-ownership-supervisor retain ... --attempt 2 \
  --label "naive causal convolution migration"
```

**Artifacts**

- Fresh attempt 1:
  `runs/cuda-ownership-supervisor/6fdb0ec11d7e/attempt-00001`.
- Fresh attempt 2:
  `runs/cuda-ownership-supervisor/6fdb0ec11d7e/attempt-00002`.

**Result**

- Attempt 1 completed 52 protected tests, all 20 applicable runtime checks,
  recurrent-only 20% ownership, isolated SM121 receipt, one operator/Nsight
  symbol, transitional FLA only for unclaimed units, all four zero sanitizer
  summaries, and complete advisory timing. Independent review passed; exact
  `da1dea938...` is retained as milestone 2 / `correct_bootstrap`.
- Attempt 2 completed 52 tests, all 21 runtime checks, strict ownership increase
  to 40% by adding the convolution forward/backward atomic pair, three exact
  operators/Nsight symbols and receipts, FLA only for unclaimed chunk units,
  all four zero sanitizer summaries, and complete advisory timing. Independent
  review passed; exact `146e9090...` is retained as milestone 3 /
  `validated_component`.
- Neither retention used advisory performance as a veto or claim. No candidate
  was merged and the default backend was not changed.

**Next**

- Intake exact `4d1a3b231...` from retained `146e9090...` with its original
  chunk hypothesis. Under the 3,600-second kernel ceiling, require the formerly
  censored naive candidate observation to complete or fail honestly; review all
  five-unit/runtime-FLA-free evidence before retaining the fresh immutable
  milestone.

## 2026-08-08 [agent] reconstruct complete FLA-free milestone

**Context**

- Fresh milestones 2 and 3 retained recurrent and convolution ownership. The
  final migration reconstruction reran exact chunk candidate `4d1a3b231...`
  under the extended kernel ceiling rather than copying the old censored result.

**Commands**

```bash
.venv/bin/research cuda-ownership-supervisor intake \
  --config configs/research/kda_cuda_ownership.toml \
  --base-ref 146e9090a6823a9e87c91114e4eec1b8852a6836 \
  --candidate-ref 4d1a3b231da2c99882324efbda5306a1815e21c7 \
  --hypothesis "<original verbatim deterministic naive chunk hypothesis>"
.venv/bin/research cuda-ownership-supervisor run \
  --config configs/research/kda_cuda_ownership.toml --attempt 3
.venv/bin/research cuda-ownership-supervisor retain \
  --config configs/research/kda_cuda_ownership.toml --attempt 3 \
  --label "naive FLA-free project CUDA milestone" --reason "<fresh evidence>"
```

**Artifacts**

- Fresh attempt:
  `runs/cuda-ownership-supervisor/6fdb0ec11d7e/attempt-00003`.
- Complete candidate-kernel payload SHA-256
  `cac190f7bea0f915fc7c2d8f5d099868295743ca406e3d29df510b8e2c2236e0`.
- Authoritative isolated mapped library SHA-256
  `4bb92fe51b22d181bef08894a0d3a5673fe233fae74c97e71448b51cdf99b4e8`.

**Result**

- The fresh attempt completed 52 protected tests, all 21 runtime checks, exact
  ownership increase from 40% to 100%, runtime FLA freedom, no forbidden
  attempts or selective PTX, all five operator traces and independent Nsight
  symbols, exact SM121 source/library provenance, and genuine zero summaries
  from all four sanitizer tools.
- Unlike the old protocol, the naive candidate kernel worker completed rather
  than being censored: 18 migration rows in 286.865 seconds. At T=1024 its
  chunk forward/backward median was 6,700.671 ms; maximum regression was
  2,210.945x. Migration performance is advisory, so this is an honest slowness
  observation and no speed, memory, stability, or quality improvement claim.
- Independent review passed. Exact `4d1a3b231...` is retained as fresh milestone
  4 / `fla_free_naive`, with all five owner units and runtime FLA freedom. The
  derived lane is now optimization.

**Next**

- Intake only exact pushed child `613b0759...` from this retained parent with
  its original Nsight-supported hypothesis. Run all strict optimization gates
  unchanged. The parent kernel worker may take roughly 35 minutes and each of
  nine parent training blocks roughly 2.5 hours; let them finish within the
  new 3,600/13,500-second ceilings. Retain only if both kernel workers, all nine
  alternating pairs, confidence interval, memory, kernel, drift, ownership,
  correctness, profile, and sanitizer gates pass.

## 2026-08-08 [agent] terminate impractical naive-parent replay

**Context**

- The timeout-only protocol successfully removed the 180-second entry deadlock,
  but the exact sizing diagnostic showed the retained naive parent needs about
  2.5 hours for one seven-step block. Repeating that parent nine times would
  consume roughly 22 hours before considering later candidates.
- The human explicitly rejected that evaluation strategy and directed immediate
  termination and redesign. This supersedes continuing the costly replay; it
  does not authorize treating partial work as a result or retaining attempt 4.

**Commands**

```bash
# Supervisor and worker owned distinct process groups.
kill -TERM -- -312660
kill -TERM -- -313575
.venv/bin/research cuda-ownership-supervisor recover \
  --config configs/research/kda_cuda_ownership.toml --attempt 4 \
  --reason "Human-directed termination ... no baseline JSON or paired block"
```

**Artifacts**

- Interrupted fresh attempt:
  `runs/cuda-ownership-supervisor/6fdb0ec11d7e/attempt-00004`.
- Fresh timeout-refreeze ledger SHA-256 after terminal recovery:
  `8fcf16b48f9b5bb844e69970dd3ce4cdca066a1c08425b83d2fbf5405281f556`.

**Result**

- SIGTERM to the supervisor group stopped PID 312660. Its bounded kernel worker
  had its own session, survived as an orphan, and was then explicitly stopped by
  process group 313575. No supervisor, worker, trainer, compiler, sanitizer, or
  GPU compute process remains.
- Candidate correctness, runtime audit, ownership profile, and all four
  sanitizer phases had completed. The naive baseline kernel worker was stopped
  after about 14 minutes, emitted no `baseline-kernel.json`, and created no
  phase row. Zero paired blocks exist.
- Protected recovery terminally records fresh attempt 4 as `invalid`,
  `not_retainable`, with null summary/milestone and an exact human-directed
  interruption reason. The partial baseline is censored diagnostic work, not a
  score or performance comparison.
- The fresh retained head remains exact `4d1a3b231...`, ownership 1.0 and runtime
  FLA-free. Exact candidate `613b0759...` remains pushed and checker-qualified
  but is not retained or merged. The original ledger remains byte-identical at
  SHA-256 `14bce91dee58aaa21a59c87be24eb37b821472d8711ce2fc310553d77331267c`.
- The internal monitoring heartbeat was cancelled after termination.

**Next**

- Do not rerun the fixed nine-pair naive-parent strategy. Redesign the first
  optimization transition so the deliberately pathological correctness anchor
  is not replayed for roughly 22 hours. Preserve exact candidate correctness,
  ownership, provenance, sanitizer, full candidate-kernel, and full candidate
  training evidence; distinguish any bridge eligibility from a statistical
  optimization-retention claim. Obtain explicit agreement on the new comparison
  and decision rule before another protected protocol freeze.

## 2026-08-08 [agent] replace per-edit confirmation with an overnight development funnel

**Context**

- The human rejected spending about 95% of kernel-development time in the
  confirmation evaluator and authorized an immediate fast loop. Pure kernel
  measurement and one bounded full-model throughput block are now the normal
  development gates; the nine-pair confirmation suite is reserved for a
  plateau, a major strategy boundary, a roughly four-hour checkpoint, or the
  final candidate.
- The exact naive parent has already been measured at great cost. It must never
  run in the inner loop again. Exact `613b0759...` is the non-official
  development baseline because it is project-owned/runtime-FLA-free,
  checker-qualified, and has complete kernel and exact 4k model-throughput
  diagnostics. This designation makes no quality or confirmation claim and
  does not change the official retained milestone.

**Commands**

```bash
# Create the first fast-loop worktree from the checker-qualified implementation.
git worktree add -b kda-cuda/batch-parallel-backward-005 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_005 613b0759...
uv run --no-sync python -m pytest -q tests/test_kda_cuda_development.py
# Result: 13 passed
```

**Artifacts**

- Fast-loop policy: `runs/kda-cuda-development/development-policy.json`, SHA-256
  `dec9c1cce7ec4456b333ec6c5e1e1df07c2e4026883b9b94d8ce05bc55ef848f`.
- Development-baseline manifest:
  `runs/kda-cuda-development/baseline/613b0759.json`, SHA-256
  `1b5636aa0824e5ebafae231a203227adcb4dbe7d1a96e502679b0fb90ba42ff9`.
- Slow-loop presentation archive:
  `runs/kda-cuda-development/evidence/slow-loop-archive-001/index.json`,
  initial SHA-256
  `25114ba28a0d2a7901a6af090ca20769c5135a673f0f625be9dbe3811f0dd2dc`.
  It copies key timing, Nsight, checker, and timeout-sizing files from ephemeral
  roots and references both authoritative ledgers/supervisor namespaces. The
  seven-step diagnostic logs were recovered from the immutable session JSONL
  after revived persisted file handles re-truncated the original `/tmp` logs;
  this limitation is explicit and the diagnostic remains non-conclusion-bearing.
- Public/offline reference manifest:
  `runs/kda-cuda-development/reference-sources.json`. Exact local pins include
  FlashKDA `1ce47ea3...`, FLA `a3edffc3...`, and MIT Lethe
  `e3ed0ccb...`; none may be runtime-imported by a candidate.

**Result**

- Added `scripts/kda_cuda_development.py`, a non-conclusion-bearing append-only
  Level-1 harness. It refuses artifact collisions, captures both worktrees'
  commit/status/diff/source hashes and all subprocess logs, keeps each root in
  an isolated process/cache, safely terminates process groups, runs protected
  candidate runtime audit, and measures exact production B=2, H=3, K=V=128 at
  T=256/1024/4096 with three warmups and ten samples.
- Level 1 advances only when T=4096 forward+backward improves at least 3%, every
  important latency row is within 5%, peak allocation is within 3%, and the
  audit/process payloads are complete. Its subprocess ceiling is 300 seconds.
- Level 2 is deliberately not auto-launched: the harness records a fresh UUID
  namespace and exact baseline/candidate six-layer 4k trainer commands in an
  explicitly alternating order. One baseline/candidate block is used only for
  a Level-1 winner and requires at least 2% model-throughput improvement,
  finite steps, and peak memory within 3%.
- Online/local review agrees that the low-risk first change is B*H backward
  parallelism with deterministic per-batch parameter-gradient partials. The
  larger non-naive direction is a project-owned CUDA chunkwise WY/UT training
  path modeled from the published KDA/DeltaNet equations and offline FLA
  structure: C=64 gives 384 independent chunk CTAs at the target shape and
  replaces roughly 1.5 GiB of per-token FP32 history with compact chunk
  boundaries. Lethe corroborates a reverse state-adjoint scan plus WY VJP but
  its SM100 `tcgen05`/TMEM implementation cannot run on GB10 SM121.

**Next**

- Complete attempt 5's single-axis B*H backward split, run the fast Level-1
  artifact once, and run one bounded Level-2 baseline/candidate pair only if it
  clears Level 1. Preserve the attempt regardless of outcome.
- Continue with shared invariant hoisting as a separate low-risk axis. In
  parallel, design the C=64 ordinary-CUDA WY/UT path from equations/offline
  references rather than repeatedly polishing the token-serial algorithm.
- Never launch the naive parent or the nine-pair confirmation suite in the
  inner loop. Confirm only at the agreed sparse cadence.

## 2026-08-08 [agent] accept batch-parallel backward as the fast development baseline

**Context**

- Attempt 5 is the first candidate evaluated through the new short funnel. Its
  only axis is removing the serial batch loop from chunk backward: one CTA now
  owns each `(B,H)` recurrence, writes deterministic per-batch FP32 parameter
  partials, and two small fixed-order kernels reduce `dA_log` and `ddt_bias`.
- Parent is exact development baseline `613b0759...`; candidate branch is
  `kda-cuda/batch-parallel-backward-005`. No naive-parent or confirmation run
  was launched.

**Commands**

```bash
# First harness invocation (preserved invalid software failure).
.venv/bin/python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_004 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_005 \
  runs/kda-cuda-development/attempts/attempt-00005-level1
# Fixed isolated import with importlib.import_module; 13 focused tests passed.
# Fresh rerun:
.venv/bin/python scripts/kda_cuda_development.py ... \
  runs/kda-cuda-development/attempts/attempt-00005-level1-rerun-001
# One recorded Level-2 baseline-first pair executed from level2-plan.json.
```

**Artifacts**

- Preserved invalid harness attempt:
  `runs/kda-cuda-development/attempts/attempt-00005-level1`.
- Valid attempt and every provenance/diff/raw sample/log:
  `runs/kda-cuda-development/attempts/attempt-00005-level1-rerun-001`.
- Append-only attempt index SHA-256:
  `ca7fe3b04e074f4b49390d3d10436caa6349f37410825063a886e8a7031fa68c`.
- New development-baseline manifest:
  `runs/kda-cuda-development/baseline/c5b36f8dc.json`, SHA-256
  `4d86f8d217096eec295f0c58383805bf3b4bfb7a53518ba06df1ae25f48300a3`.
- Candidate commit/source: `c5b36f8dc0018f28d242ad1a656625b7aa94cb7b` /
  `chunk.cu` SHA-256
  `5ad66f545331a6d592095f5f244235e74ae6fac427f0fa4495103b56db288db8`.

**Result**

- The first Level-1 invocation is invalid and unscored: both benchmark helpers
  imported the package-level `kda` function instead of the dispatcher module
  and emitted no measurements. It remains preserved. Coordinator fix
  `322eb5d` uses `importlib.import_module`; no artifact was overwritten.
- Fresh exact B=2, H=3, K=V=128 Level 1 completed in about 135 seconds. Candidate
  runtime audit passed, ownership was 1.0, runtime was FLA-free, and production
  helper events were purely project CUDA.
- Forward+backward medians (parent -> candidate):
  - T=256: `81.374 -> 48.767 ms` (40.07% lower).
  - T=1024: `329.343 -> 197.904 ms` (39.91% lower).
  - T=4096: `1318.981 -> 792.127 ms` (39.94% lower).
  T=4096 forward was effectively unchanged (`243.027 -> 242.015 ms`). Every
  important row stayed within the 5% guard and T=4096 peak allocation changed
  by only 3,584 bytes (`1.0000021x`). Level 1 advanced.
- A preserved changed-axis diagnostic compared joint B=2/H=3 and B=3/unequal-V
  candidate gradients against the exact parent under independent output/final
  state cotangents. All BF16 outputs/activation/initial-state gradients were
  bit-identical; only FP32 parameter reduction association changed (`dA` max
  abs `4.77e-7`, `ddt` max abs `1.19e-7`). Candidate repeats were bitwise
  deterministic.
- The single exact six-layer 4k Level-2 pair completed in baseline-first order:
  parent timed steps `[960,959,959,958,956]`, median `959 tok/s`; candidate
  `[1516,1516,1517,1517,1516]`, median `1516 tok/s`. The observed single-pair
  improvement is 58.08%; both reported exactly `5511.408 MiB` peak. This is a
  development observation, not a confidence interval or quality result.
- Attempt 5 passed the agreed Level-1 and Level-2 development rules and becomes
  the new fast development baseline. It is pushed but not merged, not the
  default backend, and not an officially confirmed retained milestone.

**Next**

- Branch attempt 6 from `c5b36f8...` and cache per-token decay/normalized-key
  invariants in chunk backward as one separate low-risk axis. Continue the
  kernel-only Level 1 and one-pair Level 2 funnel.
- Start the larger C=64 WY/UT CUDA design in parallel. Do not invoke nine-pair
  confirmation until a plateau/strategy boundary/four-hour checkpoint/end of
  night.

## 2026-08-09 [agent] reject shared-cache barrier cost and accept 256-thread backward

**Context**

- Two independent low-risk axes followed accepted attempt 5. Both used exact
  B=2/H=3 production-shaped Level 1 against `c5b36f8...`; neither invoked the
  naive parent or confirmation.

**Commands**

```bash
.venv/bin/python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_005 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_006 \
  runs/kda-cuda-development/attempts/attempt-00006-level1 \
  --level2-order candidate-first
.venv/bin/python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_005 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_007 \
  runs/kda-cuda-development/attempts/attempt-00007-level1 \
  --level2-order candidate-first
```

**Artifacts**

- Attempt 6: `runs/kda-cuda-development/attempts/attempt-00006-level1`;
  exact rejected commit `1ec0bc4889c1982dc2d415df98859e2a8b5433d0`.
- Attempt 7: `runs/kda-cuda-development/attempts/attempt-00007-level1`;
  exact accepted commit `90e87cf86723d19dd26906f3ede567c7ba1c2268`.
- Append-only attempt index SHA-256 after both:
  `de0e0e0eca721c072fdb88fdfbaa8316abd29a8326a9a0ef220f2ab735114c33`.
- New development-baseline manifest:
  `runs/kda-cuda-development/baseline/90e87cf86.json`, SHA-256
  `f558e50224b7a920dd163e01add0863741e96f4e30fe46c7ad5a628bc11893f9`.
- Preserved C=64 project-CUDA WY/UT implementation blueprint:
  `runs/kda-cuda-development/research/chunkwise-wy-cuda-blueprint-001.md`,
  SHA-256
  `d0d04136381e27acff28dff5fd8683aa803021a44b77ddb38112c768ee954bb3`.

**Result**

- Attempt 6 cached normalized keys, gate sigmoids, and decay exponentials in
  dynamic shared memory during backward. Correctness/runtime audit passed and
  memory was unchanged, but its extra per-token barriers outweighed redundant
  transcendental removal: T=4096 forward+backward regressed
  `792.369 -> 822.806 ms` (3.84%). Level 1 rejected it in about 105 seconds;
  Level 2 was not run. The exact source/artifact/branch remain preserved.
- Attempt 7 changed only backward block size from 128 to 256 threads. Level 1
  completed in about 91 seconds and advanced: T=256/T=1024/T=4096
  forward+backward improved 8.29%/7.63%/7.01%; T=4096 was
  `792.674 -> 737.126 ms`. All latency/memory/runtime gates passed.
- Its one candidate-first exact six-layer 4k Level-2 pair observed candidate
  timed steps `[1636,1631,1629,1628,1629]`, median `1629 tok/s`, versus parent
  `[1515,1515,1517,1517,1515]`, median `1515 tok/s`: 7.52% higher with identical
  `5511.408 MiB` peak. This is one development pair, not statistical
  confirmation or a quality result.
- Attempt 7 becomes the current development baseline. It is pushed but not
  merged or set as the default backend.
- The offline/public blueprint derives a fixed-shape C=64 FP32 WY/UT path with
  only 24.375 MiB of chunk-boundary state instead of about 1536 MiB token
  history. Its first complete path must remain FP32 SIMT/ATen BMM; BF16 WMMA is
  explicitly a later separate precision/performance axis.

**Next**

- Finish/evaluate attempt 8's independently authored global token-preprocessing
  axis. Because it branched from `c5b36f8...` in parallel, any win must be
  replayed as a fresh child of `90e87cf...` before becoming the baseline.
- Continue building the FP32 C=64 forward path as small independently testable
  stages; keep sparse confirmation reserved for the agreed cadence.

## 2026-08-09 [agent] preserve four near-misses and advance two cumulative development baselines

**Context**

- Attempts 8-13 explored invariant preprocessing, workspace-free forward
  caching, their evidence-motivated combination, and backward CTA size. All
  compared only to the current fast development baseline. The expensive
  confirmation suite and naive parent remained unused.

**Artifacts**

- Complete append-only artifacts:
  `runs/kda-cuda-development/attempts/attempt-00008-level1` through
  `attempt-00013-level1`; Level 2 exists only for attempts 11 and 13.
- Append-only attempt index SHA-256 after attempt 13:
  `bf26b8c5e376c956a86860747067cad5d9c533edf6f6aaf39a7294809c560aea`.
- Development baseline `fe0411f...` manifest:
  `runs/kda-cuda-development/baseline/fe0411f36.json`, SHA-256
  `593991bd6a3c668d4d0b7f4b97171a35449e8a49fd3cbb686d9ee8dce0c1ba7c`.
- Current development baseline `69a5ee6...` manifest:
  `runs/kda-cuda-development/baseline/69a5ee68e.json`, SHA-256
  `cbba2c8d2407502b655af5a3f5b4e4397d1b5567d870dd63a5ee32ab57c0a96d`.

**Result**

- Attempt 8 (`610cb318...`) globally materialized normalized q/k and gate/decay
  invariants for forward and backward. It lowered T=4096 forward 8.90% and
  forward+backward 6.42%, but forward-only allocation grew `2.590x`; the fixed
  per-row 3% Level-1 memory guard rejected it. No Level 2 ran.
- Attempt 9 (`897c843...`) restricted the same 48.375 MiB workspace to backward
  and combined it with the accepted 256-thread parent. Memory passed (`1.029845x`)
  and latency fell 2.896%, but this honestly missed the predeclared 3% threshold;
  no retest or Level 2 was used.
- Attempt 10 (`b393ea4...`) used only 1.5 KiB dynamic shared memory in a K=V=128
  forward fast path while preserving the generic fallback. T=4096 forward fell
  4.57%, but total forward+backward improved only 1.62%; it was preserved and
  rejected without Level 2.
- Attempt 11 deliberately combined those two related, independently measured
  subthreshold mechanisms. Exact Level 1 passed: T=4096 forward+backward
  `736.944 -> 703.009 ms` (4.60% lower), forward `243.176 -> 231.328 ms`, and
  kernel peak `1.029845x`. Its exact baseline-first Level-2 pair observed
  `1631 -> 1670 tok/s` (2.39% higher) with identical `5511.408 MiB` full-model
  peak. Commit `fe0411f3685378ce51bb084a2741998cd418250f` became the development
  baseline; the observation is neither statistical confirmation nor quality.
- Attempt 12 (`eb17ed6...`) changed 256 to 512 threads. It improved T=4096
  forward+backward 2.915%, narrowly below 3%, so Level 2 did not run.
- Attempt 13 changed 256 to the hardware maximum 1024 threads. Level 1 passed:
  `703.825 -> 682.268 ms` (3.06% lower), no memory change. Its candidate-first
  Level-2 pair observed median `1716 tok/s` versus `1669 tok/s` (2.82% higher),
  identical `5511.408 MiB` full-model peak. Exact pushed commit
  `69a5ee6...` is the current development baseline, still unmerged/unconfirmed.
- The fast funnel made six source experiments, two full-model pairs, and four
  honest rejections in a fraction of the former single-evaluation time. Every
  branch, source, sample, raw log, decision, and failed gate is preserved.

**Next**

- Stop spending major effort on scalar launch tuning. Profile/decompose the
  remaining roughly `451 ms` backward portion, then split history replay from
  reverse recurrence or begin the staged FP32 C=64 WY/UT forward path from the
  preserved blueprint.
- Continue Level 1/2 for development; invoke confirmation only at the agreed
  plateau/time boundary.

## 2026-08-09 [agent] row-parallel and tiled reverse boundaries pass sparse development confirmation

**Context**

- Continued the approved fast funnel from the confirmed 1024-thread scalar
  development baseline. The sealed naive parent was never executed.
- This interval crossed two major implementation boundaries, so each was
  checked with the full protected correctness/profile/runtime checker, all four
  compute-sanitizer tools, and nine alternating exact Level-2 pairs. Neither
  confirmation evaluated language-model quality or changed official retention.

**Artifacts**

- Attempt artifacts: `runs/kda-cuda-development/attempts/attempt-00014-level1`
  through `attempt-00022-level1`, including raw profiles and Level-2 logs only
  where the frozen funnel allowed them.
- First sparse confirmation:
  `runs/kda-cuda-development/confirmations/confirmation-00001-row-parallel-boundary`;
  final manifest SHA-256
  `a8f1e497ea01d9d9d5f1104cd32e0134217a8ea1c4c611d7120b5277f955c202`.
- Second sparse confirmation:
  `runs/kda-cuda-development/confirmations/confirmation-00002-tiled-reverse-boundary`;
  final manifest SHA-256
  `4d95ee0e0d64684904151b4a5bc80116e2f79afd1728fce032cd621fd2dde0db`.
- Current confirmed development baseline manifest:
  `runs/kda-cuda-development/baseline/6c8475157-confirmed.json`, SHA-256
  `c7a1980d7765fef1587d08094c233469708b4dc1095556313630e6038c297971`.
- Append-only attempt index after attempt 22/confirmation 2: SHA-256
  `5c06a2d665ce7ddd5ff1571fe9e5d5ae99dc0ba23488f70f4d0eb399fde0b4a9`.

**Result**

- Attempt 14 (`2485f177...`) split backward history replay into a 768-CTA
  value-row kernel plus reverse-only recurrence. Level 1 improved T=4096
  forward+backward `680.720 -> 577.269 ms` (15.20%); Level 2 observed
  `1721 -> 1978 tok/s` (14.93%) at identical `5511.408 MiB`. Nsight measured
  history at only `11.390 ms`, leaving reverse at `332.353 ms`.
- Attempt 15 (`886712ad...`) applied the same value-row decomposition to
  forward without a global workspace. T=4096 forward fell
  `231.429 -> 19.754 ms` (91.46%) and forward+backward fell 36.37%. Level 2
  observed `1979 -> 2866 tok/s` (44.82%), unchanged full-model peak.
- The first protected confirmation preflight honestly preserved three invalid
  infrastructure invocations: wrong checker cwd, a clean worktree where a
  staged candidate was required, and then a valid correctness/runtime pass
  whose profile could not see the frozen canonical kernel names. Candidate
  attempt 16 (`5ce9e192...`) renamed the actual fast production kernels to the
  audited canonical symbols. This symbol-only fix changed T=4096 by 0.041%,
  within every regression/resource guard.
- Confirmation 1 then passed all protected runtime/correctness/profile checks,
  memcheck, racecheck, synccheck, and initcheck. All nine alternating pairs were
  valid and faster: baseline medians `1973-1986 tok/s`, candidate medians
  `2851-2866 tok/s`, median improvement 44.742%, exact two-sided sign-test
  `p=0.00390625`, and every full-model peak exactly `5511.408 MiB`.
- Attempt 17 (`a460dfd...`) introduced deterministic C=64/G=8 value-tiled
  reverse and cut T=4096 kernel time 73.39%, but its fixed 4.512 MiB workspace
  made the T=256 memory row 4.414% larger. The candidate was rejected without
  Level 2; no favorable latency overrode the frozen memory gate.
- Attempt 18 (`83e99411...`) reduced chunk capacity to 32, preserving the
  algorithm while bounding workspace. Every memory row passed (T=256 ratio
  `1.021476`); T=4096 forward+backward fell `367.119 -> 97.911 ms` (73.33%).
  Its Level-2 pair observed `2863 -> 6689 tok/s` (133.64%), identical full-model
  peak. Profiling isolated `18.656 ms` in a serial parameter finalizer.
- Attempt 19 (`6c847515...`) parallelized deterministic parameter reductions,
  reusing dead q-norm workspace and adding no allocation. T=4096 fell another
  `97.169 -> 79.449 ms` (18.24%); Level 2 observed `6684 -> 7394 tok/s`
  (10.62%), identical full-model peak. Nsight reduced parameter work to under
  `0.36 ms`; the value-tiled reverse is now `42.43 ms`, forward `19.84 ms`,
  history `11.85 ms`, and preprocess `2.61 ms`.
- Confirmation 2 on exact commit `6c847515733078fc6485e10713f981dfa20a5ffd`
  again passed the full protected checker and all four sanitizers. All nine
  alternating pairs were valid and faster: baseline `2855-2863 tok/s`,
  candidate `7325-7394 tok/s`, median improvement 156.708%, exact sign-test
  `p=0.00390625`, and every full-model peak exactly `5511.408 MiB`.
- Attempts 20 (`547ce9c...`) and 21 (`9eb2ada...`) tested sixteen value tiles
  with constant or BF16-compressed workspace. They regressed T=4096 by 50.59%
  and 46.14%, respectively, and were rejected without Level 2. Attempt 22
  (`acffe2f...`) cached the owned adjoint tile in 8.14 KiB shared memory; it
  regressed 8.96% and was likewise rejected. All sources/artifacts remain.
- Exact `6c847515...` is therefore the current statistically confirmed
  **development** baseline. Its source SHA-256 is
  `c50becb95a700bd981e2e294d4a41dd51bacadf7312019ede9408410f15f58e5`.
  It remains unmerged, non-default, and is not an LM-quality or official
  retention claim. Official protected retention remains exact naive
  `4d1a3b231da2c99882324efbda5306a1815e21c7`.

**Next**

- Treat the C32/G8 tiled reverse as the scalar/value-decomposition plateau.
  Further major gains require the staged FP32 C=64 WY/UT algorithmic path (or
  another dependency-breaking reverse formulation), not more tile-count or
  shared-cache launch tuning.
- Keep `6c847515...` as the confirmed comparison anchor, preserve every future
  attempt, and defer another nine-pair confirmation until the next major
  strategy boundary, four-hour checkpoint, plateau, or final candidate.

## 2026-08-09 [agent] establish matched FLA Triton throughput target

**Context**

- Before beginning the FP32 C=64 WY/UT rewrite, measured the external FLA
  implementation as a separate reference ceiling. FLA remains forbidden as a
  candidate runtime dependency; this benchmark did not modify candidate source,
  merge a branch, change the default backend, or evaluate LM quality.
- Used exact current Level-2 model conditions: six KDA layers, B=2, H=3 derived
  from depth 6/head dimension 128, sequence length 4096, global batch 32768,
  BF16, seed 42, 2 warmup plus 5 measured optimizer steps. Set
  `FLA_FLASH_KDA=0`, `FLA_TILELANG=0`, and `TORCH_COMPILE_DISABLE=1`; the
  resolved backend was `fla_triton` with `fla-core 0.5.2`.

**Artifacts**

- `runs/kda-cuda-development/reference-benchmarks/fla-triton-001`.
- Predeclared plan SHA-256:
  `30675429377db428ed683f1583b32dd9d9ad44bce852892d165f565a4ec0e412`.
- Final manifest SHA-256:
  `d0bbd7a4759d03299e194b5a8ffcb17f84041aa12c2fe0ad23791cfb8926df4d`.
- Append-only attempt/reference index SHA-256 after recording this benchmark:
  `df7cea5e95cf21d163f49833da5559c7db54b0d76420f37010b1d7848868d1c2`.

**Result**

- Five valid run medians were `43750`, `43697`, `43613`, `43629`, and
  `43680 tok/s`; the median-of-medians target is therefore **43680 tok/s**
  with observed run-median range `43613-43750 tok/s`.
- Median full-model peak allocation was `5550.471 MiB` (first compiled run
  `5548.471 MiB`; subsequent runs `5550.471 MiB`).
- Exact confirmed project-owned development commit `6c847515...` has a
  confirmation-2 candidate median of `7333 tok/s` and `5511.408 MiB` peak.
  The FLA target is `5.9566x` current project throughput, so the project-owned
  path needs approximately 495.7% additional throughput to exceed it. The
  project path currently uses about `39.063 MiB` less peak allocation.
- Step 0 of the first FLA process included Triton compilation (`310 tok/s`),
  but the two declared warmups excluded it from the five measured steps; all
  measured steps were stable around 43.4-43.9k tok/s. Later processes reused
  the reference cache and produced the same steady-state band.

**Next**

- Use `43680 tok/s` as the matched full-model throughput goal for project-owned
  CUDA. Continue reporting both absolute tok/s and fraction of this target.
- Do not import or link FLA in a candidate. Pursue the independently implemented
  C=64 FP32 WY/UT path and preserve the current correctness, ownership,
  sanitizer, memory, and confirmation gates.

## 2026-08-09 [agent] attribute the warmed project training step

**Context**

- Before starting the C=64 rewrite, the approved execution plan required a
  warmed whole-step trace to reconcile the large gap between the isolated KDA
  operator and the matched full-model result.
- Profiled exact confirmed development commit `6c847515...` at the frozen
  six-layer B=2/H=3/T=4096 training shape. This was attribution-only evidence,
  not a new throughput score or LM-quality evaluation.

**Commands**

```bash
# Warm an isolated extension cache, then trace all seven exact Level-2 steps.
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none   --output runs/kda-cuda-development/profiles/whole-step-attribution-001/whole-step   <exact seven-step project_cuda trainer command>
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,osrt_sum <report>
```

**Artifacts**

- `runs/kda-cuda-development/profiles/whole-step-attribution-001` contains the
  predeclared plan, exact command/environment, raw trainer log, `.nsys-rep`,
  exported CSV summaries, structured summary, provenance, and file manifest.
- Plan SHA-256:
  `b1cdfd90aec3d13e7ad3d9d043255dc5e387692121e2b8eb62953e3b6d42f978`.

**Result**

- Five post-warmup steps were stable at `4.4509-4.4638 s`; median was
  `4.46148 s` / `7344 tok/s`, consistent with prior confirmation.
- GPU kernels account for `4.42486 s/step`, or 99.18% of the warmed median.
  The trace observed exactly 504 causal-convolution backward launches and 168
  KDA calls across seven steps: 72 convolution backwards and 24 KDA calls per
  optimizer step as predicted.
- Causal-convolution backward alone consumed `1.93032 s/step` (43.62% of all
  GPU kernel time, `26.81 ms/call`). Project KDA work consumed about
  `1.859 s/step`; its major components were reverse `1.0267 s`, forward
  `0.4561 s`, history `0.2829 s`, and preprocess `0.0643 s`.
- The former unexplained residual is therefore reconciled. The source-level
  O(T) search around width-four dependencies is the correct first target before
  WY/UT. Convolution forward was only `5.90 ms/step` and is not a priority.

**Next**

- Evaluate attempt 23, which changes only causal-convolution backward dependency
  bounds while preserving accumulation order and the generic ABI.
- Use a convolution-specific microbenchmark plus the exact Level-2 pair; the
  raw KDA-only Level-1 latency is not applicable to this intervention.
- Reprofile after the convolution family plateaus, then proceed to the staged
  project-owned FP32 C=64 WY/UT path toward the overall 45k tok/s aim.

## 2026-08-09 [agent] accept bounded convolution dependencies as development baseline

**Context**

- Whole-step attribution showed causal-convolution backward consumed 43.62% of
  project GPU kernel time. Attempt 23 changed only the invalid O(T) dependency
  search for `dx` and initial-state gradients; all valid contributing tokens
  remain in the original ascending accumulation order.
- The raw KDA-only Level-1 lane is blind to convolution and is explicitly not
  applicable to this intervention. The human authorized a matched convolution
  microgate and unchanged Level-2 pair instead, with intermediate targets used
  as aims rather than automatic campaign stop conditions.

**Commands**

```bash
# Preserved KDA-only audit/provenance run; timing decision not applicable.
.venv/bin/python scripts/kda_cuda_development.py <6c847515 worktree>   <efd41da1 worktree> runs/kda-cuda-development/attempts/attempt-00023-level1-kda-not-applicable
# Four-process A/B/B/A convolution microgate, then the exact planned Level-2 pair.
# Reprofile exact candidate with the same seven-step nsys command as its parent.
```

**Artifacts**

- KDA-only audit/provenance artifact:
  `runs/kda-cuda-development/attempts/attempt-00023-level1-kda-not-applicable`.
- Preserved invalid first microgate import:
  `runs/kda-cuda-development/attempts/attempt-00023-convolution-microgate`.
  Its ignored script imported coordinator `nanochat` and raised
  `NotImplementedError`; it emitted no measurement and is unscored.
- Valid fresh microgate:
  `runs/kda-cuda-development/attempts/attempt-00023-convolution-microgate-rerun-001`,
  final manifest SHA-256
  `293c1191c6aebb76270fb202e816c4a525fb4862039f6d56bfb9afd011a37ed6`.
- Candidate reprofile:
  `runs/kda-cuda-development/profiles/whole-step-attribution-002-attempt23`,
  final manifest SHA-256
  `bdd7b92b6cb6481b3c40272f9bbb29efbb071a800c0e9d9086611edcf22d3494`.
- Development-baseline manifest:
  `runs/kda-cuda-development/baseline/efd41da1e.json`, SHA-256
  `654070b19072b2d49219e80b9f24913aa869800d5ec080999a7eb712549ecc25`.
- Append-only attempt/reference index SHA-256:
  `9b4f3efe9ab53b7bbd4d61735f053ac6ca282eb6d213c5a2deac541665d3eb64`.

**Result**

- Exact commit `efd41da1e94cf5b1c7b3194231add04e73ba90a7`, source SHA-256
  `0e05bf9b12a266a3e98c29f319e3e1f8544127db77e5bd9f3476d07fa5dd8727`,
  passed the protected 21-check runtime audit, ownership 1.0, runtime FLA
  freedom, focused build/test checks, and generic cache/final-state paths.
- Across two processes per side, T=4096 convolution backward fell
  `26.815 -> 6.502 ms` (`4.124x`); forward+backward fell
  `27.657 -> 7.078 ms`. All output/dx/dweight hashes were bit-identical and
  measured peak allocation was identical at every T=256/1024/4096 row.
- The exact baseline-first six-layer pair observed `7356 -> 10957 tok/s`, a
  48.95% development improvement, with identical `5511.408 MiB` peak. This is
  25.08% of the 43,680 FLA target and is not a confidence interval or quality
  result.
- Reprofiling reconciled the end-to-end delta: convolution backward fell from
  `1.9303` to `0.4873 s/step`, removing `1.4431 s`; total GPU kernel time fell
  from `4.4249` to `2.9610 s/step` and still explained 98.99% of the warmed
  step. KDA now dominates, but convolution has another factorizable stage.
- Attempt 23 is pushed and becomes the next development comparison baseline. It
  remains unmerged, non-default, and separate from official retention.

**Next**

- Evaluate attempt 24's single-axis FP32 preactivation-gradient factoring and
  deterministic tiled weight reduction. Preserve the bounded generic fallback.
- After the convolution family plateaus, run its full sanitizer boundary and
  begin the staged project-owned FP32 C=64 WY/UT forward path.

## 2026-08-09 [agent] complete and validate convolution optimization boundary

**Context**

- Attempt 23 removed the asymptotic dependency search but still recomputed the
  same SiLU preactivation gradient in both `dx` and `dweight`. Attempt 24
  isolates that remaining hot-path redundancy without changing schemas or the
  generic state/final-cotangent implementation.

**Commands**

```bash
# KDA-only audit/provenance run (timing decision not applicable), then an
# A/B/B/A convolution microgate and exact candidate-first Level-2 pair.
# Reprofile all seven steps with nsys.
uv run --no-sync research cuda-candidate-check   --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_validation_024   --lane optimization --sanitizers <isolated artifact/cache arguments>
```

**Artifacts**

- Attempt/audit:
  `runs/kda-cuda-development/attempts/attempt-00024-level1-kda-not-applicable`.
- Convolution microgate:
  `runs/kda-cuda-development/attempts/attempt-00024-convolution-microgate`,
  final manifest SHA-256
  `16c7fc20288372abfe414a72d280a666c82fc0f57229db80c19659a9d59a8f04`.
- Level-2 execution manifest SHA-256:
  `f8793cba03cdaeb2a545d2a1a663250a1a0014821319e44aa196cd8e23613858`.
- Whole-step profile:
  `runs/kda-cuda-development/profiles/whole-step-attribution-003-attempt24`,
  final manifest SHA-256
  `0a28ed022afc24fa23ff875f528feab04be9f327b5b0ae4efc5bf4554ca79906`.
- Full checker/sanitizer boundary:
  `runs/kda-cuda-development/validations/validation-00001-convolution-boundary`,
  final manifest SHA-256
  `d01e580cf803cd86dcea8a2bce45e641f906d315f68b73e8812f2684705f9f71`.
- Development-baseline manifest:
  `runs/kda-cuda-development/baseline/2a0a08e25.json`, SHA-256
  `6edaeaa9b563aebd0d993e99fbcd904ee044f1b93b1a977f21f91b33158631bc`.
- Append-only attempt/reference index SHA-256:
  `fcee0353d5cc4bf9bf5270b1298365df76f4b3e8313aea5314a762f6618413a8`.

**Result**

- Exact commit `2a0a08e254e3af81a57b4ee472aea4d8e8b76a56`, source SHA-256
  `7a356fd341e07aeea376dabdd3716b2f265e2ca85ab2322b2bdc7dfef5b8e586`,
  materializes allocator-visible FP32 `dz`, computes `dx` from it, and uses
  fixed 256-token FP32 weight partials plus a deterministic finalizer. All other
  shapes/state cases route to the bounded project fallback.
- Against attempt 23, T=4096 convolution backward fell
  `6.348 -> 0.371 ms` (`17.10x`) and forward+backward fell
  `7.111 -> 0.466 ms`. `dx` remained bit-identical. Weight reassociation changed
  one of 1,536 BF16 values at production shape, max abs `0.03125` / max relative
  `0.00437`, within protected gradient tolerance.
- Isolated convolution peak rose about 29% because the 12 MiB FP32 `dz` and
  partials are visible. The exact full model remained unchanged at
  `5511.408 MiB`, so the enabling scratch did not move the training peak.
- Exact Level 2 observed `10961 -> 12977 tok/s` (18.39%), reaching 29.71% of
  the 43,680 FLA target. This remains a development pair, not quality evidence
  or a confidence interval.
- Full protected runtime/profile checking passed at ownership 1.0 and runtime
  FLA freedom. Memcheck, racecheck, synccheck, and initcheck each completed with
  genuine zero-error summaries.
- The fresh whole-step profile accounted for 98.88% of the `2.531 s` warmed
  step. All convolution backward stages now total only `30.75 ms/step` (1.23%
  of GPU time), while project KDA forward/reverse/history/preprocess consume
  about 72.7%. Convolution has reached its useful development plateau.
- Attempt 24 is pushed and becomes the new development baseline. It remains
  unmerged, non-default, non-quality, and separate from official retention.

**Next**

- Start the independently authored project FP32 C=64 WY/UT forward stage from
  exact `2a0a08e...`, keeping the existing project backward as the analytical
  recurrence derivative and all non-production shapes on the legacy project
  path.
- Continue toward the overall 45k target without treating intermediate latency
  aims as automatic stop conditions; preserve every diagnostic and rollback.

## 2026-08-09 [agent] preserve correct FP32 C64 WY forward scaffold

**Context**

- With convolution reduced to 1.23% of GPU time, attempt 25 began the planned
  algorithmic transition. It changed only exact production training forward;
  the project token recurrence remains the analytical backward and every other
  shape/state/inference call remains on the project compatibility kernels.

**Commands**

```bash
# Protected runtime/profile preflight, exact parent-recurrence equation checks,
# then the normal Level-1 artifact against 2a0a08e.
.venv/bin/python scripts/kda_cuda_development.py <attempt24> <attempt25>   runs/kda-cuda-development/attempts/attempt-00025-level1
# Nsight-stage diagnostic on the preserved rejected candidate.
```

**Artifacts**

- `runs/kda-cuda-development/attempts/attempt-00025-level1`.
- Stage profile:
  `runs/kda-cuda-development/attempts/attempt-00025-level1/diagnostic-stage-profile`,
  final manifest SHA-256
  `f248e42237f0a730d30abe5fab2e50d62f0add7248588156b957474618a1740f`.
- Append-only attempt/reference index SHA-256:
  `6d48a29d4248fd2d6351275bffaf8869f7456212e834f7065eb55d6537ffd927`.

**Result**

- Exact pushed commit `a34661d8fd6e85d8ec2275f44aba71a32f26dece`
  added project-owned FP32 C=64 preprocessing, stable pair matrices, fixed-order
  unit-lower solve, NoTF32 ATen U/W BMMs, and a fixed-order SIMT boundary scan.
  It uses no PTX, WMMA, atomics, hidden allocation, or runtime reference.
- C=1/2/3/7/64 equation checks matched through `1.82e-6` output and `2.09e-7`
  state. Exact production output matched the project recurrence at max abs
  `6.1035e-5`, all gradients were finite, the protected runtime/profile audit
  passed, ownership remained 1.0, and runtime remained FLA-free.
- Level 1 honestly rejected the first correctness scaffold: T=4096
  forward+backward was `79.091 -> 87.362 ms` (10.46% regression), with a small
  `1.00486x` kernel-memory ratio. T=256/1024 stayed on the fallback and were
  effectively unchanged. Level 2 was not run.
- Profiling localized the issue rather than invalidating the equations. Per
  specialized call: scan `25.712 ms`, M/A build `1.891 ms`, preprocess
  `0.380 ms`, U/W BMM `0.257 ms`, and solve `0.083 ms`. The scan alone explains
  the regression; the other independently testable stages are already small.
- Attempt 25 is preserved as a correct equation milestone, not a development
  baseline or performance win. Exact `2a0a08e...` remains the comparison
  baseline.

**Next**

- Branch from the attempt-25 equation scaffold and replace only its forward
  scan with a faster matrix path; compare the cumulative candidate to exact
  `2a0a08e...`. Do not rewrite the validated F0-F3 stages.
- Keep intermediate targets advisory while continuing toward 45k, but never
  weaken correctness, ownership, sanitizer, or provenance requirements.
