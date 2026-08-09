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

## 2026-08-09 [agent] accept FP32 batched-matrix C64 scan as development baseline

**Context**

- Attempt 26 retained attempt 25's validated FP32 C=64 preprocessing, stable
  pair matrices, solve, and equations, replacing only the slow scalar boundary
  scan with a batched FP32 matrix scan. The exact convolution baseline
  `2a0a08e...` remained the matched comparison anchor.
- This was a Level-1 and single Level-2 development decision only. It did not
  run quality evaluation, nine-pair statistical confirmation, private seeds,
  or an official retention action.

**Commands**

```bash
.venv/bin/python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_024 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_026 \
  runs/kda-cuda-development/attempts/attempt-00026-level1
# Executed the predeclared candidate-first Level-2 pair, finalized its
# summary/manifest and baseline record, then pushed the exact clean branch.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_026 \
  push origin HEAD:kda-cuda/wy-fp32-bmm-scan-026
```

**Artifacts**

- Level 1 and runtime/profile evidence:
  `runs/kda-cuda-development/attempts/attempt-00026-level1`, manifest SHA-256
  `ec4daf9418fbc2d452aa80f6052b077db68eae96bed9f4db7bb200947ffd2a7e`.
- Level-2 candidate-first pair:
  `runs/kda-cuda-development/attempts/attempt-00026-level1/level2-execution`,
  final manifest SHA-256
  `cba8ef86d9dfeecae77d96ba8e7ff8e50bba11edbe16892bf3c3001ccfbc1b37`.
- Development-baseline manifest:
  `runs/kda-cuda-development/baseline/fb89c2605.json`, SHA-256
  `2db903d49dc32ef619b24e99ac8c326214e96c5556c9e9af218a2fd27ebf4149`.
- Append-only attempt/reference index SHA-256:
  `ab2b4926a800119b262c7197a2287592dc3345e4570ce669f3f392eb70160ee7`.

**Result**

- Exact pushed commit `fb89c260541db40662bc79929d8391081ed5fc5e`
  matched the project recurrence at max absolute output error
  `6.103515625e-05`; deterministic repeat passed, returned gradients were
  finite, and the protected runtime/profile audit passed at ownership 1.0 with
  no runtime FLA.
- T=4096 Level-1 forward+backward improved `79.074 -> 63.652 ms` (19.50%).
  The allocator-visible kernel memory ratio was `1.00486`, within the frozen
  development limit.
- The exact candidate-first Level-2 pair measured candidate
  `[14790,14778,14800,14973,14879]`, median `14800 tok/s`, versus baseline
  `[12950,12507,12949,12932,12917]`, median `12932 tok/s`: a 14.44% point
  improvement. Candidate peak was `5508.533 MiB` versus `5511.408 MiB`.
- This reaches 33.88% of the matched external `43680 tok/s` target. It is an
  accepted development baseline, not statistical confirmation, an LM-quality
  result, a default change, a merge, or official milestone retention.

**Next**

- Inspect the already-preserved independent BF16 WMMA attempt 27 once and run
  the normal bounded gate only if its correctness evidence is complete.
- Use attempt 26 as the fallback development baseline and begin replacing full
  token history with C64 chunk-boundary recomputation/reverse scan, followed by
  the complete WY/UT VJP. Keep sparse confirmation cadence and every existing
  correctness, ownership, sanitizer, deterministic-gradient, and provenance
  gate.

## 2026-08-09 [agent] preserve valid BF16 WMMA scan without selecting it

**Context**

- Recovered attempt 27 once after its parallel worker had stopped. The clean
  committed branch replaces attempt 25's scalar C64 scan with an eight-warp,
  32-value-tile BF16 WMMA scan and retains FP32 accumulation.
- Scratch evidence was complete enough to justify the normal bounded gate: the
  protected 21-check runtime and profile audits passed, runtime remained
  FLA-free at ownership 1.0, production output was deterministic and within
  tolerance, and all returned gradients were finite.

**Commands**

```bash
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_024 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_027 \
  runs/kda-cuda-development/attempts/attempt-00027-level1
# Executed only the predeclared baseline-first Level-2 pair after Level 1
# advanced, finalized the artifact, and pushed the exact clean branch.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_027 \
  push origin HEAD:kda-cuda/wy-bf16-wmma-scan-027
```

**Artifacts**

- Recovered one-shot diagnostics:
  `runs/kda-cuda-development/diagnostics/attempt-00027-recovered-preflight`,
  manifest SHA-256
  `6ff70ceaab3e8daf8b6332a2cc8015f8ec9cb72530d4bdaab06e8012aa642447`.
- Level 1:
  `runs/kda-cuda-development/attempts/attempt-00027-level1`, manifest SHA-256
  `f77e4f5072a25bf1dbab51884eae0cb022e6752c2c8545cc0e326f60f8041f98`.
- Level-2 baseline-first pair:
  `runs/kda-cuda-development/attempts/attempt-00027-level1/level2-execution`,
  final manifest SHA-256
  `d83ff71626fde1d00083f6409ec12557b36cc5f908f6c321733cddb8e1041e49`.
- Append-only attempt/reference index SHA-256:
  `a101165634df5f55d4aaf635e4c358f238286ce5ce3e55333981e36a5b95eac8`.

**Result**

- Exact pushed commit `f73a68a9efcbbb7e4de4ea7cdf23801b58d35378`
  executed `HMMA.16816.F32.BF16`. Production output matched within protected
  tolerance at max absolute `4.8828125e-4`, p99 `6.103515625e-5`; deterministic
  repeat and finite-gradient checks passed.
- Level 1 advanced: T=4096 forward+backward improved
  `79.164 -> 62.703 ms` (20.79%), with `1.00486x` memory. Its normal
  forward-only row was `19.730 ms`, slightly slower than attempt 26's
  `19.622 ms`; the much smaller recovered direct diagnostic used a different
  import/input setup and is not used for selection.
- The baseline-first Level-2 pair measured baseline
  `[13046,13034,13041,12998,13023]`, median `13034 tok/s`, and candidate
  `[14984,14916,14989,14929,14981]`, median `14981 tok/s`, a 14.94% point
  improvement with `5508.533 MiB` peak. This is 34.30% of the matched FLA goal.
- Attempt 27 is valid and preserved, but not selected over attempt 26. The
  cross-attempt whole-step delta is small and unconfirmed, the matched normal
  forward-only gate did not improve, and BF16 operand casting increased max
  output error while remaining within tolerance. No quality, statistical,
  default, merge, or official-retention claim is made.

**Next**

- Continue from exact FP32 BMM development baseline `fb89c260...` for the C64
  backward transition. Preserve attempt 27 as an independently valid tensor-
  core option that can be revisited after backward dominates less of the step.
- First eliminate full token-state history via chunk-boundary recomputation and
  reverse scan, then implement the complete WY/UT VJP. Do not run another
  confirmation until the next major strategy boundary, plateau, roughly
  four-hour checkpoint, or final candidate.

## 2026-08-09 [agent] accept chunk-boundary backward recomputation

**Context**

- Attempt 28 began the C64 backward transition from exact accepted FP32 BMM
  baseline `fb89c260...`. It changes only production K=V=128 backward history:
  instead of materializing `[B,H,T+1,V,K]`, it stores fixed 32-token boundary
  states and deterministically recomputes one local chunk before each reverse.
- The existing analytical token VJP and every generic/state path remain
  unchanged. This is the memory/recomputation scaffold for a later complete
  WY/UT VJP, not that VJP itself.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_028 \
  --lane optimization <isolated artifact/cache arguments>
uv run --no-sync python scripts/kda_cuda_development.py \
  --level2-order candidate-first \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_026 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_028 \
  runs/kda-cuda-development/attempts/attempt-00028-level1
# Exact B=2/H=3/T=4096 saved-output/gradient comparison, then the predeclared
# candidate-first Level-2 pair.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_028 \
  push origin HEAD:kda-cuda/wy-backward-boundaries-028
# Full checker plus memcheck/racecheck/synccheck/initcheck on an exact staged
# candidate tree in the validation worktree.
```

**Artifacts**

- Level 1:
  `runs/kda-cuda-development/attempts/attempt-00028-level1`, manifest SHA-256
  `3c9383e89ff7927a13a2ccc40c874d3da8652f20ccdb1c9468cf4632f41b3d1c`.
- Level-2 candidate-first pair:
  `runs/kda-cuda-development/attempts/attempt-00028-level1/level2-execution`,
  final manifest SHA-256
  `69f3f270893a237841e5156d97ad3ec5b12af72021013fd72e6e479dbed9f6a7`.
- Exact production gradient comparison:
  `runs/kda-cuda-development/diagnostics/attempt-00028-boundary-gradient-exact`,
  manifest SHA-256
  `fc4d483935a91de5ad6bf7280ee026bba25776ce71586f80fadbce8d72c7f263`.
- Preserved invalid clean-worktree sanitizer invocation:
  `runs/kda-cuda-development/validations/validation-00002-boundary-recompute`,
  manifest SHA-256
  `5226c420fcf7410a384e2696123d867c2104d6f0ee50b6e4460e6e237b3d164f`.
- Valid staged-tree full checker and sanitizer boundary:
  `runs/kda-cuda-development/validations/validation-00003-boundary-recompute`,
  manifest SHA-256
  `fdb8b0801b67878d115fb01ddff3e223ec43e872f35d6db6dd85f1f7a3968e0e`.
- Development-baseline manifest:
  `runs/kda-cuda-development/baseline/30a13734f.json`, SHA-256
  `56521ea7c3ac85c38c06a3c0eac765a5c23a08482d62389d9cff15a4e1600805`.
- Append-only attempt/reference index SHA-256:
  `9aed69ae1d079d5e8beae1f523a6b3beac337d1bceee49758c65e06ef8ae92a5`.

**Result**

- Exact pushed commit `30a13734f22e0639f9d9aa05417a4ac69dfe62f1`,
  source SHA-256
  `191a2e00dc107aaa184120eb899d3c018c451ebddfc9cbf9f3e984558fc1a93c`,
  passed the protected 21-check runtime audit, profile audit, ownership 1.0,
  runtime FLA freedom, and all four sanitizers with genuine zero-error
  summaries.
- Exact B=2/H=3/T=4096 output and all seven gradients were bitwise identical
  to parent `fb89c260...`; every result was finite.
- Level 1 measured T=4096 forward+backward `63.457 -> 58.205 ms`, an 8.28%
  improvement. Isolated peak allocation fell `1761199616 -> 201410048` bytes,
  a `0.11436x` ratio. No other row exceeded the 5% regression gate.
- The exact candidate-first Level-2 pair measured candidate
  `[15928,15864,15908,15883,15895]`, median `15895 tok/s`, versus baseline
  `[14860,14887,14913,14846,14842]`, median `14860 tok/s`: a 6.97% point
  improvement. Both reported `5508.533 MiB` full-model peak. The candidate is
  at 36.39% of the matched `43680 tok/s` FLA target.
- Two setup failures were preserved and are invalid, not measurements: the
  first draft benchmark accidentally ran `uv` from the candidate, created an
  isolated empty environment, and failed `ModuleNotFoundError: torch`; that
  environment was moved to `/tmp/kda028-accidental-venv-20260809`. The first
  saved-gradient helper imported the package-level function and failed
  `AttributeError`. A clean committed sanitizer invocation also emitted no
  result because the checker correctly requires a staged source snapshot.
- Attempt 28 is the accepted development baseline. It is not statistically
  confirmed, official retention, a merge/default change, or an LM-quality
  result.

**Next**

- Implement the complete independently authored C64 WY/UT VJP from exact
  `30a13734...`, using boundary states and reverse chunk scan rather than
  restoring token history. Keep the generic analytical recurrence fallback.
- Continue toward 45k with Level 1/2 development evidence and defer another
  confirmation to the next declared major strategy boundary, plateau,
  roughly four-hour checkpoint, or final candidate.

## 2026-08-09 [agent] preserve complete C64 WY/UT VJP equation milestone

**Context**

- Attempt 29 replaces the production-shape analytical token recurrence VJP
  with an independently derived reverse-mode implementation of the complete
  C64 WY/UT equations. It retains attempt 28's chunk-boundary discipline and
  keeps the generic/state paths unchanged.
- An FP64 multi-chunk equation prototype first matched PyTorch autograd to
  `7.11e-15` or better across q, k, v, beta, and gate increments. The committed
  CUDA scaffold then passed the protected generic runtime/profile audit before
  its production-only specialization was evaluated separately.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_029 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seeded B=2/H=3/T=4096 saved-output/gradient comparison against
# attempt 28, followed by a candidate-only production microbenchmark/profile.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_029 \
  push -u origin kda-cuda/wy-full-vjp-029
```

**Artifacts**

- Protected checker, including isolated caches and raw logs:
  `runs/kda-cuda-development/diagnostics/attempt-00029-protected-checker`,
  manifest SHA-256
  `b5f1de7c727b1115d0d686e6a4f3cb737394f8a2930fdf51de11a96ed170c3cd`.
- Exact production output/gradient comparison:
  `runs/kda-cuda-development/diagnostics/attempt-00029-wy-vjp-gradient-exact`,
  manifest SHA-256
  `39a88b8eebb4d48ddb7509e483d149fe984bc9b5f60ed2060325cfd447dc90d0`.
- Production microbenchmark and operator profile:
  `runs/kda-cuda-development/diagnostics/attempt-00029-wy-vjp-draft`,
  manifest SHA-256
  `c57ba5da88e14673c39eda2e2c243d78baed845670601b8c1374df0494219fee`.
- Append-only attempt/reference index SHA-256:
  `7a62eed577f43ffa99c9da7babbca042da9b3174d0717988c0afbdfd28110b21`.

**Result**

- Exact pushed commit `b54d72e6f36397c49772770fa336b2899e2f32ef`
  passed the protected 21-check runtime audit and profile audit at ownership
  1.0 with no runtime FLA. At B=2/H=3/T=4096 its output is bitwise identical
  to accepted parent `30a13734...`; all seven gradients are finite and the
  largest absolute parent delta is `3.637978807091713e-12` (`dk`). This clears
  the production equation gate for the complete WY/UT VJP.
- The scaffold is performance-rejected: T=4096 forward+backward measured
  `111.892 ms` versus attempt 28 diagnostic `58.973 ms` (`1.8973x`), while
  peak allocation rose `201410048 -> 305376768` bytes. The operator profile
  counted 1,284 `aten::bmm` calls and 23.612 ms of self CPU dispatch time.
- The first baseline gradient launch was invalid before CUDA because it used
  the coordinator as the build-helper working directory; its exact traceback
  is preserved as `baseline-invalid-001` and is not a measurement.
- Attempt 29 is a correct equation milestone, not the selected development
  baseline, statistical confirmation, official retention, a merge/default
  change, or an LM-quality result. Attempt 28 remains selected.

**Next**

- Start a new attempt from the exact attempt-29 equations. Batch all
  chunk-independent dR/dA/dE/dW/dT/dP/dQ/dM products across the 384 chunk rows,
  precompute the chunk-local dstate/dZ sources, and retain only the two
  genuinely recurrent matrix products per reverse chunk.
- Re-run the exact production gradient comparison before the normal Level-1
  gate. Do not launch Level 2 unless Level 1 advances, and keep the next full
  sanitizer/confirmation boundary sparse as declared.

## 2026-08-09 [agent] reject batched WY VJP on memory after speed breakthrough

**Context**

- Attempt 30 changes only the dispatch structure of attempt 29's validated
  equations. Chunk-independent VJP products are batched over all 384 chunk
  rows; the reverse loop retains only the two state-dependent products.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_030 \
  --lane optimization <isolated artifact/cache arguments>
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_028 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_030 \
  runs/kda-cuda-development/attempts/attempt-00030-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_030 \
  push -u origin kda-cuda/wy-batched-vjp-030
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00030-protected-checker`,
  manifest SHA-256 `330e05d948b617021d6074addceebfb5607eb2906f55f871434bef47e89ac619`.
- Bitwise production comparison:
  `runs/kda-cuda-development/diagnostics/attempt-00030-batched-vjp-gradient-exact`,
  manifest SHA-256 `517bda12eb207a299463f54981362f0d65f5b39f7949efde0100b9dc0d53629b`.
- Draft timing/profile:
  `runs/kda-cuda-development/diagnostics/attempt-00030-batched-vjp-draft`,
  manifest SHA-256 `8bc6082392cefc494aeabdf7c7c1e087c96559dfaf285a1f468f1370dce52d14`.
- Formal Level 1: `runs/kda-cuda-development/attempts/attempt-00030-level1`,
  manifest SHA-256 `d2a9105f17c9a118dbbe38d80029574cc985563b0cf5742f786a9205f42b8387`.
- Append-only index SHA-256:
  `ec066a29b2d93c66795e3ba6257461b615a58f68dc7d6d0a9d69331e205899fc`.

**Result**

- Exact pushed commit `14d8c8d4532302fa2a792d2b7b17743cbe8e4d59`
  passed the protected audit at ownership 1.0 with no runtime FLA. Output and
  all seven B=2/H=3/T=4096 gradients are bitwise identical to attempt 29.
- The operator profile fell from 1,284 to 528 BMM calls and BMM self CPU time
  from 23.612 to 3.846 ms. Formal Level 1 improved T=4096 forward+backward
  `58.693 -> 47.287 ms` (19.43%).
- Level 1 nevertheless rejected the candidate: peak allocation increased
  `201410048 -> 492151296` bytes (`2.4435x`), and T=256 forward+backward
  regressed 7.18%, beyond the 5% limit. Level 2 was not launched.
- This is a speed milestone and a validated optimization parent, not the
  accepted development baseline, confirmation, quality evidence, or official
  retention. Attempt 28 remains the accepted baseline.

**Next**

- Continue from attempt 30 only for memory-lifetime work: release dead forward
  products before allocating adjoints, reuse dZ storage, and materialize the
  independent VJP matrices in bounded groups. Re-run exact gradients and Level
  1; do not relax either the memory or short-length regression gate.

## 2026-08-09 [agent] reduce batched WY VJP peak through exact lifetimes

**Context**

- Attempt 31 changes only tensor lifetimes and storage reuse in attempt 30's
  bitwise-validated batched VJP. Dead forward matrices are released at phase
  boundaries, dZ and dM storage is reused, and large output-adjoint buffers are
  allocated only after the reverse state scan.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_031 \
  --lane optimization <isolated artifact/cache arguments>
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_028 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_031 \
  runs/kda-cuda-development/attempts/attempt-00031-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_031 \
  push -u origin kda-cuda/wy-vjp-lifetimes-031
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00031-protected-checker`,
  manifest SHA-256 `bc63e511e36d77a46c904ec3ecf467fcf158ae08f6aa0d56ac9f1792edbee73f`.
- Bitwise production comparison:
  `runs/kda-cuda-development/diagnostics/attempt-00031-lifetimes-gradient-exact`,
  manifest SHA-256 `c9cf3cc26b532a9fa8405024067501ef3d5cdfa1cdd287f641510b3fad93f106`.
- Draft timing/memory:
  `runs/kda-cuda-development/diagnostics/attempt-00031-lifetimes-draft`,
  manifest SHA-256 `3c9a892109ae570e7e37edb255aa4057c4801e44d323f7a527d7dbd32124e369`.
- Formal Level 1: `runs/kda-cuda-development/attempts/attempt-00031-level1`,
  manifest SHA-256 `c762d1a081a3eab6ce486312d972d05bbf94feca694ecbf271766df7a6b25685`.
- Append-only index SHA-256:
  `fd70fdd8c89e6216e0432792ea073644adc3e5caba2e27b45ce442d9aa438cea`.

**Result**

- Exact pushed commit `e7e597b3c55df7873fb3d7d7b015b7c3e273774c`
  passed the protected audit at ownership 1.0 with no runtime FLA. Output and
  all seven production gradients are bitwise identical to attempt 30.
- Candidate-only peak fell `492151296 -> 319572992` bytes (35.07%) while
  T=4096 forward+backward changed `47.024 -> 48.130 ms` (2.35% regression).
- Formal Level 1 retained a 19.60% T=4096 improvement
  (`59.156 -> 47.563 ms`) and every non-memory row stayed inside the 5% gate.
  It did not advance because peak remained `1.5867x` the accepted baseline.
  Level 2 was not launched.
- Attempt 31 is the next memory-optimization parent, not the accepted baseline,
  confirmation, quality evidence, or official retention. Attempt 28 remains
  accepted.

**Next**

- Full-sequence H, Z, dH-next, and matrix-adjoint outputs still coexist at the
  remaining peak. Replace them with bounded chunk-group boundary states,
  recompute each group, and consume its reverse/VJP products before moving to
  the previous group. Preserve exact ordering within each recurrence.

## 2026-08-09 [agent] preserve four-chunk grouped WY VJP scaffold

**Context**

- Attempt 32 replaces full-sequence H/Z/dH-next/dZ and matrix-adjoint storage
  with 16 four-chunk boundary groups. Each group is recomputed, reverse-scanned,
  differentiated, and consumed before the previous group begins.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_032 \
  --lane optimization <isolated artifact/cache arguments>
compute-sanitizer --tool memcheck <exact production gradient helper>
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_028 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_032 \
  runs/kda-cuda-development/attempts/attempt-00032-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_032 \
  push -u origin kda-cuda/wy-vjp-grouped-032
```

**Artifacts**

- Final protected checker: `runs/kda-cuda-development/diagnostics/attempt-00032-protected-checker`,
  manifest SHA-256 `4689033a9a67bb34c2012a53e61d34067da28c6d91d184441900f9ad6710f7f8`.
- Production comparison plus every invalid pre-fix run and raw memcheck log:
  `runs/kda-cuda-development/diagnostics/attempt-00032-grouped-gradient-exact`,
  manifest SHA-256 `7404cf3f8a13e55d086e7f76e3390a5d9cfe8aeb4d71205ed629378fd77da027`.
- Draft timing/memory:
  `runs/kda-cuda-development/diagnostics/attempt-00032-grouped-draft`,
  manifest SHA-256 `705af1fbac6189b0232949ae856b09c26a02757d94dbbb056f0c0be804580bbe`.
- Formal Level 1: `runs/kda-cuda-development/attempts/attempt-00032-level1`,
  manifest SHA-256 `e44614a555dd2401eee4cf9d5390f4041da41325bbf1ba104f607c3645c70da5`.
- Append-only index SHA-256:
  `436853935f74680fe54a17da2781555180e73305aa5081b41e7463e17d5f41ec`.

**Result**

- The first production launch was invalid. Synchronous rerun exposed an
  illegal address, and memcheck reported 130 errors in the grouped pair kernel:
  group-local dP/dQ were indexed with global `n`. Both indices were corrected
  to `local_n`; all failed logs are preserved and are not measurements.
- Exact pushed commit `941330842689a04a07dca14c6e2ce02686099337`
  passed the final protected audit at ownership 1.0 with no runtime FLA. After
  the fix, output and all seven production gradients are bitwise identical to
  attempt 31.
- Peak fell `319572992 -> 271764480` bytes, but candidate-only T=4096 rose to
  `61.681 ms`. Formal Level 1 measured `60.706 -> 60.578 ms` (0.21%) and a
  `1.3493x` memory ratio, so neither the 3% speed threshold nor memory gate
  passed. Level 2 was not launched.
- Attempt 32 is a correct grouped scaffold, not an accepted baseline,
  confirmation, quality evidence, or official retention.

**Next**

- The full U/W/R/E tensors and global dqbar/dkhat/dprefix/dv scratch dominate
  the remaining live set, while packing 16 groups dominates runtime. Recompute
  U/W/R/E inside wider groups and finalize each group's normalized/input
  gradients immediately so group-local adjoints do not accumulate globally.

## 2026-08-09 [agent] reject eight-chunk local-final WY VJP

**Context**

- Attempt 33 removes full-sequence U/W/R/E, FP32 upstream packing, and global
  dqbar/dkhat/dprefix/dv scratch from attempt 32. Eight-chunk groups recompute
  their WY products, consume local adjoints, write BF16 input gradients
  immediately, and carry deterministic parameter accumulators in reverse order.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_033 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seeded B=2/H=3/T=4096 output and gradient comparison plus repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_028 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_033 \
  runs/kda-cuda-development/attempt-00033-local-final-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_033 \
  push -u origin kda-cuda/wy-vjp-local-finalize-033
```

**Artifacts**

- Protected checker:
  `runs/kda-cuda-development/diagnostics/attempt-00033-protected-checker`,
  manifest SHA-256
  `fb443fc0e7b264e8082afd7e3eb4f53b38e76fcb4e3a68ee1ff427ebe71d7f6b`.
- Production comparison, deterministic repeat, and preserved invalid first
  invocation:
  `runs/kda-cuda-development/diagnostics/attempt-00033-local-final-gradient-exact`,
  manifest SHA-256
  `b2f4d3bf48459af9d4e4595177a1ee12094cc03dead3a29304a9908004fd05f5`.
- Unmatched B1/H1 protected-worker diagnostic, retained but excluded from the
  production decision:
  `runs/kda-cuda-development/diagnostics/attempt-00033-local-final-draft`,
  manifest SHA-256
  `f87d240f9c246abddd233fb4c0544bff5970e2e51312104fdaaa64df228260be`.
- Matched formal Level 1:
  `runs/kda-cuda-development/attempt-00033-local-final-level1`, manifest
  SHA-256
  `add2808ab9fd91407ba4f3b20b1c61818c1af2af00315aa05ae14b76d23d193f`.
- Append-only attempt/reference index SHA-256:
  `5d6d546b04328d8bb62950b33d226098584667a5a04a76a331bb6da73fc74ccb`.

**Result**

- Exact pushed commit `470a64eea53170e1a55ffc92875462ddc3602f68`
  passed the protected runtime/profile audit at ownership 1.0 with no runtime
  FLA. Output and all seven production gradients are bitwise identical to
  attempt 32; a second saved tensor bundle is also bitwise identical.
- The first saved-gradient invocation was invalid before CUDA because the
  build helper resolved sources from the coordinator working directory. Its
  traceback is preserved and excluded from evidence.
- Matched Level 1 rejected the candidate: T=4096 forward+backward regressed
  `59.363 -> 90.293 ms` (52.10%), while peak allocation changed
  `201410048 -> 204491264` bytes (`1.0153x`). Level 2 was not launched.
- Because Level 1 was decisively rejected, no sanitizer confirmation was
  launched and no sanitizer-valid claim is made for attempt 33. It is an exact
  equation/storage scaffold only, not an accepted baseline, confirmation,
  quality result, default, merge, or official retention. Attempt 28 remains
  the accepted development baseline.

**Next**

- Profile the exact B=2/H=3/T=4096 specialization to attribute the 4096-only
  regression. Optimize group recomputation/launch structure without restoring
  full token history or changing equation order, then rerun exact gradients and
  matched Level 1. Keep Level 2 and sanitizer confirmation gated on advancement
  or a declared strategy boundary.

## 2026-08-09 [agent] accept local raw-gradient WY VJP baseline

**Context**

- A bounded Nsight Systems production profile of attempt 33 attributed 67.9%
  of observed kernel time to eight launches of its parameter-gradient kernel:
  each group recomputed sigmoid and raw-gate derivatives already produced by
  finalization.
- Attempt 34 changes only that lifetime. Finalization retains one group-local
  FP32 raw-gate-gradient buffer, and the deterministic reverse parameter
  reduction consumes it before the group is released.

**Commands**

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  <exact B=2/H=3/T=4096 attempt-33 helper>
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_034 \
  --lane optimization <isolated artifact/cache arguments> --sanitizers
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_028 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_034 \
  runs/kda-cuda-development/attempt-00034-local-raw-gradient-level1
# Executed the predeclared baseline-first Level-2 pair once after all gates.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_034 \
  push -u origin kda-cuda/wy-vjp-local-raw-grad-034
```

**Artifacts**

- Attempt-33 production profile:
  `runs/kda-cuda-development/diagnostics/attempt-00033-production-profile`,
  manifest SHA-256
  `9c39f2c26fc4964415731aae28b1a8fd0193059e752310aa87c2e2d2507a8a3a`.
- Initial protected checker:
  `runs/kda-cuda-development/diagnostics/attempt-00034-protected-checker`,
  manifest SHA-256
  `6443c0e4f023f4dc0d89392297b8da1d0acdad3e5d7cbf407a996db3a5c7d17d`.
- Exact production comparison and repeat:
  `runs/kda-cuda-development/diagnostics/attempt-00034-local-raw-gradient-exact`,
  manifest SHA-256
  `cd510bea8cb202a1e463efa08186b874c31e9719f6585604ce17cb2a6a214589`.
- Full protected checker and all four sanitizers:
  `runs/kda-cuda-development/validations/validation-00004-local-raw-gradient`,
  manifest SHA-256
  `cae4643b1320262825eaf48d732c71da8e8751f1f36335fc808f245cbc943161`.
- Matched Level 1:
  `runs/kda-cuda-development/attempt-00034-local-raw-gradient-level1`,
  manifest SHA-256
  `08b67b5775ca80f1fee830ce7aae238aa0c1e86176b04bb0049f5fb324dea3e9`.
- Baseline-first Level-2 pair:
  `runs/kda-cuda-development/attempt-00034-local-raw-gradient-level1/level2-execution`,
  manifest SHA-256
  `19fc41ddd14d813ac1889e1ae6bb80c739b7b0eff840df0be9fb949c716007e5`.
- Development-baseline manifest:
  `runs/kda-cuda-development/baseline/c46ac8d5a.json`, SHA-256
  `f31fcbe614a3cad579460a87625b111abafdb06ba0b855055da1921048f6d912`.
- Append-only attempt/reference index SHA-256:
  `6c81673441cd5ad1867dd8d9736e2f1a67205dfcb0a61377d438005dc8d77004`.

**Result**

- Exact pushed commit `c46ac8d5aefdca82cd4fa2b38ecf8570f8cc1b13`
  passed ownership 1.0, runtime/profile audit, runtime FLA freedom, and all four
  sanitizers with zero-error summaries. Output and all seven production
  gradients are bitwise identical to attempt 33; the deterministic repeat has
  the same tensor-bundle SHA-256.
- Level 1 advanced: T=4096 forward+backward improved
  `58.732 -> 43.429 ms` (26.06%), with `1.0231x` peak allocation. Every other
  runtime row stayed within the 5% limit; T=4096 forward-only was 3.47% slower.
- The baseline-first Level-2 pair measured baseline
  `[15873,15865,15873,15889,15890]`, median `15873 tok/s`, and candidate
  `[19031,19014,18994,18990,19009]`, median `19009 tok/s`: a 19.76% point
  improvement. Both reported `5508.533 MiB` peak. The candidate is 43.52% of
  the matched `43680 tok/s` FLA target.
- Two non-measurement bookkeeping failures are preserved: the first Level-2
  wrapper resolved relative artifact paths from the baseline worktree and
  stopped before launching training; the first derived finalization padded an
  abbreviated candidate SHA with zeros. The raw pair was not rerun, and the
  final summary was regenerated with the Git-resolved full SHA.
- Attempt 34 is the accepted development baseline. It is not statistical
  confirmation, official retention, a quality result, merge, or default
  change. The official milestone remains `4d1a3b231...`.

**Next**

- Continue from `c46ac8d5...`. Profile the exact production backward after the
  dominant parameter kernel removal, then attack the remaining pair VJP and
  BMM/dispatch cost without restoring full histories or changing reduction
  order. Continue toward at least 45k; do not treat 19k as a stopping point.

## 2026-08-09 [agent] preserve parallel parameter reduction as uncertain speed milestone

**Context**

- Attempt 34's production profile left parameter reduction at 30.6% of kernel
  time and the pair VJP at 25.6%. Attempt 35 replaces 768 single-thread
  parameter blocks with six 128-thread recurrence blocks per group.
- Key lanes load raw and biased gate values in parallel; lane zero still sums
  `dA_log` in token-reverse/key-ascending order, while every lane accumulates
  its `dt_bias` gradient in token-reverse order.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_035 \
  --lane optimization <isolated artifact/cache arguments>
# Exact B=2/H=3/T=4096 saved-gradient comparison.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_034 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_035 \
  runs/kda-cuda-development/attempt-00035-parallel-params-level1
# One bounded retest after the unchanged generic T=256 row crossed the guard.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_035 \
  push -u origin kda-cuda/wy-vjp-parallel-params-035
```

**Artifacts**

- Attempt-34 production profile:
  `runs/kda-cuda-development/diagnostics/attempt-00034-production-profile`,
  manifest SHA-256
  `f181526cfd4805ec5d1348f5572f12d8b1d930a263774c6bddcffa0f386f6ab7`.
- Protected checker:
  `runs/kda-cuda-development/diagnostics/attempt-00035-protected-checker`,
  manifest SHA-256
  `78e68e71ceee62dd9e120de9aae690201956ab2e5433ad8cfa87bde93e61a1ea`.
- Exact production comparison:
  `runs/kda-cuda-development/diagnostics/attempt-00035-parallel-parameter-gradient-exact`,
  manifest SHA-256
  `c95106d013cd67eff4fd9d887085db004a301212dd73d38a27c827023c235844`.
- Initial matched Level 1:
  `runs/kda-cuda-development/attempt-00035-parallel-params-level1`, manifest
  SHA-256
  `81d42137a4ebf8039919aa65edb43da021182189e96ef31b6be7158b09779402`.
- Bounded Level-1 retest:
  `runs/kda-cuda-development/attempt-00035-parallel-params-level1-retest-001`,
  manifest SHA-256
  `4e7b986586840a9b27d36d269d8a2fd0beeb50af93b91a0e9d419b667e069ced`.
- Append-only attempt/reference index SHA-256:
  `91368878549aea494b90b48d92406f82723477f0c0d6f0d03f0982cc98ce13ac`.

**Result**

- Exact pushed commit `6812002afd62848b5278dfbfb4465b3939b89f29`
  passed the protected audit at ownership 1.0 with no runtime FLA. Output and
  all seven production gradients are bitwise identical to attempt 34.
- Initial Level 1 improved T=4096 forward+backward
  `43.816 -> 34.313 ms` (21.69%) at identical peak allocation, but T=256
  forward+backward regressed `4.128 -> 4.380 ms` (6.11%); the decision was
  therefore `do_not_advance`.
- One bounded retest reproduced the T=4096 gain (`43.661 -> 34.470 ms`, 21.05%)
  and measured T=256 as a 1.67% improvement. Under the contract, this is
  uncertainty rather than a win. Level 2 and sanitizer confirmation were not
  launched, and attempt 34 remains the accepted baseline.
- Attempt 35 is a bitwise-exact speed milestone and permissible optimization
  parent, not an accepted baseline, confirmation, quality result, merge,
  default change, or official retention.

**Next**

- Continue from attempt 35 for the pair VJP, now the largest known production
  kernel. Seek a candidate that preserves the parameter gain and clears the
  entire matched Level-1 gate without relying on a retest; only then reconsider
  Level 2 and sanitizer confirmation.

## 2026-08-09 [agent] accept row-parallel exact pair VJP baseline

**Context**

- Attempt 36 retains attempt 35's exact parallel parameter reduction and maps
  the pair VJP to one block per `(recurrence, chunk row)` instead of one block
  per recurrence serially traversing all 64 rows. Each output lane preserves
  its prior arithmetic order; only independent rows execute concurrently.
- The formal comparison remained anchored to accepted attempt 34, not the
  uncertain attempt-35 retest.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_036 \
  --lane optimization <isolated artifact/cache arguments> --sanitizers
# Exact seeded B=2/H=3/T=4096 saved-gradient comparison plus repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_034 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_036 \
  runs/kda-cuda-development/attempt-00036-row-parallel-level1
# Executed the predeclared baseline-first Level-2 pair once after all gates.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_036 \
  push -u origin kda-cuda/wy-vjp-row-parallel-036
```

**Artifacts**

- Initial protected checker:
  `runs/kda-cuda-development/diagnostics/attempt-00036-protected-checker`,
  manifest SHA-256
  `48cbdcf72cb259101e97926bb0df029e7f3fd4d88d89fac5364807af13ba21a5`.
- Exact production comparison, deterministic repeat, and both preserved invalid
  setup invocations:
  `runs/kda-cuda-development/diagnostics/attempt-00036-row-parallel-gradient-exact`,
  manifest SHA-256
  `bb683c793595354af1b7397e8dfd1347ca9468b9badb5cdd39cf51e93b6a523d`.
- Full protected checker and all four sanitizers:
  `runs/kda-cuda-development/validations/validation-00005-row-parallel`,
  manifest SHA-256
  `574f16f83771c656f3831fd6900678a3bca5cf227b6d5e8f997390a57c267f25`.
- Matched Level 1:
  `runs/kda-cuda-development/attempt-00036-row-parallel-level1`, manifest
  SHA-256
  `a0ba55b7bf76578bc0bdf61526cd18f3c82671ae4395650de23f9f3bd1b6749c`.
- Baseline-first Level-2 pair:
  `runs/kda-cuda-development/attempt-00036-row-parallel-level2`, manifest
  SHA-256
  `81d930f4e35315cc7fe655c11b32c8596c98bc2a2533d05e845db92dd26fb2c4`.
- Development-baseline manifest:
  `runs/kda-cuda-development/baseline/bb29a36f50.json`, SHA-256
  `c445633ec39761766e63b0c735e1358454a3b8b44ac7bc1715bd4f34f08d815c`.
- Append-only attempt/reference index SHA-256:
  `b9a349a0a75fc31c1105d3ecb4bcb0a92fae4a23893572e5fd8109c2dd9e896d`.

**Result**

- Exact pushed commit `bb29a36f502cc8f11880c5391267657d06a3ba4a`
  passed ownership 1.0, the protected runtime/profile audit, runtime FLA
  freedom, and all four sanitizers with zero-error summaries. Output and all
  seven production gradients are bitwise identical to attempt 35; the valid
  deterministic repeat has the same tensor-bundle SHA-256.
- Two setup failures are preserved and excluded from evidence: a worktree-local
  `uv` environment lacked Torch, then a coordinator-project invocation imported
  coordinator source before candidate `PYTHONPATH` was set. Neither emitted a
  measurement; the corrected absolute invocation passed.
- Level 1 advanced cleanly without a retest: T=4096 forward+backward improved
  `43.466 -> 35.144 ms` (19.15%) at identical peak allocation. All other rows
  stayed inside the 5% runtime guard; the largest regression was 2.32%.
- The baseline-first Level-2 pair measured baseline
  `[19045,19022,19031,18975,19001]`, median `19022 tok/s`, and candidate
  `[21804,21847,21888,21869,21800]`, median `21847 tok/s`: a 14.85% point
  improvement. Both reported `5508.533 MiB` peak. The candidate is 50.02% of
  the fixed `43680 tok/s` external FLA reference.
- Attempt 36 is the accepted development baseline. It is not statistical
  confirmation, official retention, a quality result, merge, or default
  change. The official milestone remains `4d1a3b231...`.

**Next**

- Profile exact production backward at attempt 36, attack the largest remaining
  exact kernel or dispatch cost, and continue toward at least 45k without
  relaxing correctness, ownership, provenance, or sanitizer gates.

## 2026-08-09 [agent] reject eight-row pair-VJP scheduling

**Context**

- Attempt 36's production profile showed the row-parallel pair VJP at 34.0% of
  GPU kernel time (`10.335 ms` per iteration). Attempt 37 reduced launch count
  by assigning eight consecutive rows to each block while preserving each
  output's arithmetic order.

**Commands**

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  <exact B=2/H=3/T=4096 attempt-36 helper>
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_037 \
  --lane optimization <isolated artifact/cache arguments>
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_036 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_037 \
  runs/kda-cuda-development/attempt-00037-eight-row-level1
```

**Artifacts**

- Attempt-36 profile: `runs/kda-cuda-development/diagnostics/attempt-00036-production-profile`, manifest `6024c5e904824d3e4bc191214d14c71d01feb38dee7960df038a2b70e495e487`.
- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00037-protected-checker`, manifest `9e8e1bf10346b5eea833d414cdbf11523e503bc66829346ebb6de1d0b732f834`.
- Exact gradients: `runs/kda-cuda-development/diagnostics/attempt-00037-eight-row-gradient-exact`, manifest `7b3ba09b2f5f6de9ea2e8fd79c1ca2433a3f1b8ac249a44081590a8d6e4dac01`.
- Level 1: `runs/kda-cuda-development/attempt-00037-eight-row-level1`, manifest `94dce8aaf4e5ba924d5c9db9cd73cf9897749ecd65833de84ddb01bf6e62fe8b`.

**Result**

- Pushed commit `22f9e321240da49eb0614250b8ccd848087aac54` is
  bitwise identical to attempt 36 for output and all seven gradients and passed
  ownership/runtime gates. The first checker invocation stopped before build
  because staging ran in the coordinator; that exact error is preserved.
- Level 1 rejected it: T=4096 forward+backward regressed
  `35.193 -> 44.034 ms` (25.12%) at identical memory. No retest, sanitizer, or
  Level 2 ran; no sanitizer-valid claim is made.

**Next**

- Keep one row per block and parallelize the lane-zero beta-gradient tail.

## 2026-08-09 [agent] accept parallel exact beta-gradient baseline

**Context**

- Attempt 38 distributes per-key `dbeta` terms across all 128 lanes, stores
  them in bounded shared scratch, and retains lane-zero key-ascending sums.
  The row-per-block pair schedule and all reported accumulation orders remain.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_038 \
  --lane optimization <isolated artifact/cache arguments> --sanitizers
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_036 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_038 \
  runs/kda-cuda-development/attempt-00038-parallel-beta-level1
# Executed the predeclared baseline-first Level-2 pair once.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_038 \
  push -u origin kda-cuda/wy-vjp-parallel-beta-038
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00038-protected-checker`, manifest `c7ba09fca477d87f4ef33fb9984748cf9765c06046357764c3122c51d366c1d5`.
- Exact comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00038-parallel-beta-gradient-exact`, manifest `f76616b60a4c11c1ec307f749eceb79400a1285880cd7f9b76a8bb0fac9ce8d9`.
- Full sanitizer validation: `runs/kda-cuda-development/validations/validation-00006-parallel-beta`, manifest `372f79ec6dfeff5e018ea13883b83287d4f3f41a5fb5cffb725fa1ffa7eb0039`.
- Level 1: `runs/kda-cuda-development/attempt-00038-parallel-beta-level1`, manifest `a34cb63c37a2b83b946ad8001d5e4bf9b08463fd30071e40daee47352b658bf8`.
- Level 2: `runs/kda-cuda-development/attempt-00038-parallel-beta-level2`, manifest `ffab3e47d4798c16618cf5ccb61b976c930c86f0fb096f4111b05d248745e26e`.
- Production profile: `runs/kda-cuda-development/diagnostics/attempt-00038-production-profile`, manifest `fb0395d946ba98395fbfb1a366efbff4ed9b5f2a86b31398cff1ba003c1b48fa`.
- Baseline manifest: `runs/kda-cuda-development/baseline/315cb4a19a.json`, SHA-256 `8c520aa8a8ed02aaad86f439e4e74f92f4b4c36c610ffe5809653516532a9693`.

**Result**

- Exact pushed commit `315cb4a19ac6cf94f815da7c1afd1f20caf15fe6`
  passed ownership 1.0, runtime/profile audit, runtime FLA freedom, all four
  sanitizers, and a bitwise deterministic repeat for output and seven gradients.
- Level 1 improved T=4096 forward+backward `35.452 -> 28.984 ms`
  (18.24%) at identical memory; all other rows stayed within the 5% guard.
- Level 2 measured baseline `[21634,21648,21717,21747,21653]`, median
  `21653 tok/s`, and candidate `[23875,23820,23867,23708,23851]`, median
  `23851 tok/s`: +10.15%, identical `5508.533 MiB`, and 54.60% of the fixed
  43,680 tok/s external reference.
- The new profile measured the pair kernel at `4.749 ms` per iteration, down
  54% from attempt 36, but it remains the largest named kernel. Attempt 38 is
  the accepted development baseline, not confirmation, quality evidence,
  official retention, a merge, or a default change.

**Next**

- Continue from attempt 38 toward 45k; optimize only profile-supported costs.

## 2026-08-09 [agent] reject shared pair-ratio cache

**Context**

- Attempt 39 reused attempt 38's already allocated shared buffer to cache decay
  ratios from the `dA` loops for the later `dM` loops, removing duplicate
  exponentials without changing accumulation order.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_039 \
  --lane optimization <isolated artifact/cache arguments>
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_038 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_039 \
  runs/kda-cuda-development/attempt-00039-cache-ratios-level1
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00039-protected-checker`, manifest `b9e1a1c458b06ce695ead8ee08f227fca30bd3522014d008e5a8294ce8afdfca`.
- Exact gradients: `runs/kda-cuda-development/diagnostics/attempt-00039-cache-ratios-gradient-exact`, manifest `2d6bd5c778b6c218ceff8d863bfe4de2423c32141e030c65193094a12221fd6f`.
- Level 1: `runs/kda-cuda-development/attempt-00039-cache-ratios-level1`, manifest `b679fade6f1f5dbf8632ecb997871d8dc710aad60243b2176454f6c144af79c4`.
- Append-only attempt/reference index SHA-256: `7331a52ca9a2a8ed11ad3601dfc422fa9c78f365837ab040a098a728579f79f9`.

**Result**

- Exact pushed commit `63752a3722454bdfbd613ba40be79dccf5894b2d`
  passed ownership/runtime gates and remained bitwise identical to attempt 38.
  Its first checker command used an empty worktree-local `uv` environment and
  stopped before build; the exact failure is preserved and excluded.
- Level 1 rejected it: T=4096 forward+backward changed
  `29.394 -> 29.637 ms` (-0.83%), T=1024 regressed 4.13%, and memory was
  unchanged. No retest, sanitizer, or Level 2 ran.

**Next**

- Keep attempt 38 accepted. Do not sweep ratio-cache variants; target the
  parameter kernel or the paired forward/backward M/A construction instead.

## 2026-08-09 [agent] reject direct parameter-gradient loads

**Context**

- Attempt 40 removed two per-token barriers and shared staging from the
  parameter kernel. Lane zero directly read raw gradients and gate inputs in
  the same token-reverse/key-ascending order, while all lanes retained their
  independent reverse-token bias reductions.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_040 \
  --lane optimization <isolated artifact/cache arguments>
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_038 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_040 \
  runs/kda-cuda-development/attempt-00040-direct-params-level1
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00040-protected-checker`, manifest `e124977248c52cdd997f36ed01f59320942912b34583b3f512ec5a2ba40a9fc1`.
- Exact gradients: `runs/kda-cuda-development/diagnostics/attempt-00040-direct-params-gradient-exact`, manifest `581d9cbba5353b93bbe814fc95208c9c996265ed212fd069e44d0f7fa9965bed`.
- Level 1: `runs/kda-cuda-development/attempt-00040-direct-params-level1`, manifest `fa51874c4bda671d626aaf72c911f2f06d3705979147e0f760b55f38041bc98f`.

**Result**

- Pushed commit `c854eee0c751a9a2931b06fe9695895e6f21b8ae` passed
  ownership/runtime gates and was bitwise identical to attempt 38.
- Level 1 rejected it: T=4096 forward+backward regressed
  `29.108 -> 38.046 ms` (30.71%) at identical memory. Removing barriers lost
  coalesced staging and was decisively harmful. No retest, sanitizer, or Level
  2 ran; no sanitizer-valid claim is made.

**Next**

- Retain the accepted staged parameter kernel; do not pursue direct-load forms.

## 2026-08-09 [agent] reject 128-thread M/A builders

**Context**

- Attempt 41 changed both exact C64 M/A builder launches from 256 to 128
  threads, matching the key width while preserving every pair's ascending-key
  dot product. This tested the remaining exact launch-shape axis after the
  attempt-38 profile attributed about `3.73 ms` per iteration to the two
  forward/backward builders.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_041 \
  --lane optimization <isolated artifact/cache arguments>
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_038 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_041 \
  runs/kda-cuda-development/attempt-00041-ma128-level1
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00041-protected-checker`, manifest `9a749e3fd7f786a331eb5eb1e3c7065fbbc9b63b7ffe2f4c7a9b187b9574aac6`.
- Exact gradients: `runs/kda-cuda-development/diagnostics/attempt-00041-ma128-gradient-exact`, manifest `fb6e113a59fef7fc215a59831c1d891ecab6a0ffccb87853d187d1729861f96b`.
- Level 1: `runs/kda-cuda-development/attempt-00041-ma128-level1`, manifest `ade7c814644d352624e7ca4a34a0a56b26d04cf0e86ea0feb54e978de73554b0`.
- Append-only attempt/reference index SHA-256: `7102a37423f36db3ec66a54edd6ba8d058a6bb3e77e66fd7e447decfe2c520d7`.

**Result**

- Pushed commit `26fb81c1692f12feef1fc5a60b52daa65335aa1d` passed
  ownership/runtime gates and remained bitwise identical to attempt 38.
- Level 1 did not advance: T=4096 forward+backward changed
  `29.094 -> 29.483 ms` (-1.34%), T=256 regressed 4.89%, and memory was
  unchanged. No retest, sanitizer, or Level 2 ran.
- Attempt 38 remains the accepted development baseline. Attempts 39-41 form an
  exact-optimization plateau, not confirmation, quality evidence, or official
  retention.

**Next**

- Stop launch-shape and shared-cache sweeps. The next credible boundary is an
  algebraic M/A construction redesign (for example, matched transformed BMMs)
  with explicit recurrence-output/gradient tolerances, deterministic repeats,
  and the full protected gates before any performance claim.

## 2026-08-09 [agent] preserve stable tiled-BMM M/A strategy boundary

**Context**

- Attempt 42 replaces both scalar C64 M/A builders with stable 16x16
  transformed FP32 BMM tiles. A per-channel boundary center keeps every
  exponent finite even at the frozen `lower_bound=-5`; the worst diagonal-tile
  span is 15 steps. Existing P storage supplies both transform buffers and is
  rebuilt exactly afterward, so no additional peak allocation is introduced.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_042 \
  --lane optimization <isolated artifact/cache arguments>
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_038 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_042 \
  runs/kda-cuda-development/attempt-00042-ma-tiled-bmm-level1
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  <exact B=2/H=3/T=4096 attempt-42 helper>
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00042-protected-checker`, manifest `2708de56323f9c3bce43d036cc808793e43cb3542d6520a69c5be3ef5b59b7d6`.
- Production output/gradients and repeat: `runs/kda-cuda-development/diagnostics/attempt-00042-ma-tiled-bmm-gradient`, manifest `be8ddc9ea877749682112c8aabffff151b17e551f50eca938df29f3f59fd442a`.
- Level 1: `runs/kda-cuda-development/attempt-00042-ma-tiled-bmm-level1`, manifest `5f52fa10b86591ff7ede9ab8b19f2528fecf5e743547da06c4d8d3a501624d74`.
- Profile: `runs/kda-cuda-development/diagnostics/attempt-00042-production-profile`, manifest `bb432bc30971ee7d67b5d71265a727edbd3e4c47a433596617441c73df582107`.

**Result**

- Pushed commit `332e0be865b67937d6d9231c4e4dd5495fb42e52` passed
  ownership 1.0, runtime/profile gates, runtime FLA freedom, extreme-gate
  finiteness, and a bitwise deterministic repeat. Against accepted attempt 38,
  output max absolute difference was `6.1035e-05` and the largest gradient
  difference was `1.819e-12`, far inside frozen `0.005/0.02` tolerances.
- Level 1 was a near miss, not a win: T=4096 forward+backward improved
  `29.071 -> 28.319 ms` (2.59%), below the 3% gate, with memory ratio 1.0 and
  all guard rows passing. No retest, sanitizer, or Level 2 ran.
- The boundary profile reduced M/A work from about `3.73 ms` at attempt 38 to
  about `1.85 ms` per iteration; separate target transforms still duplicated
  an exponent. Attempt 42 is a correct equation/performance milestone, not an
  accepted baseline or confirmation.

**Next**

- Fuse q-left and k-left target transforms, retain stable right transforms, and
  compare the cumulative result directly against accepted attempt 38.

## 2026-08-09 [agent] accept fused stable M/A transform baseline

**Context**

- Attempt 43 fuses q-left, k-left, and right tile operands into one pass and
  reuses three quarters of P storage as scratch. It is bitwise identical to
  attempt 42, so the change affects execution only.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_043 \
  --lane optimization <isolated artifact/cache arguments> --sanitizers
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_038 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_043 \
  runs/kda-cuda-development/attempt-00043-ma-fused-transform-level1
# Executed the predeclared baseline-first Level-2 pair once.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_043 \
  push -u origin kda-cuda/wy-ma-fused-transform-043
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00043-protected-checker`, manifest `9f195bdead6d7d7ee9b6bfda5b50a0af2d5eedc1b21c888971247f4c282a6ffd`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00043-ma-fused-transform-gradient`, manifest `dff463c1a4ff0bcc340a392cbea96045f7e16573757bac3ba3c98e1a23827c0b`.
- Full sanitizer validation: `runs/kda-cuda-development/validations/validation-00007-ma-fused-transform`, manifest `ffd7a950d9600cb69b397004acf9aa72d9337b4fbf9edf525efaa90d26369ea0`.
- Level 1: `runs/kda-cuda-development/attempt-00043-ma-fused-transform-level1`, manifest `143cee2a74fc2843371ecae25b6923cc45331e7ff5ff130dee520a0eb294e821`.
- Level 2: `runs/kda-cuda-development/attempt-00043-ma-fused-transform-level2`, manifest `8b2564c66f86c6895d77bba54ecb82ec631cfba63a5cb1e310fec3896f46331f`.
- Production profile: `runs/kda-cuda-development/diagnostics/attempt-00043-production-profile`, manifest `b71cb260838ab0cd46c562173e09b90b352071bc8b42beaee40ae6b0e9ac2273`.
- Baseline manifest: `runs/kda-cuda-development/baseline/223e58634c.json`, SHA-256 `3509decf116505bde745b7371590fb396dbff19dc0e93c8068d0ea311528ef04`.
- Append-only attempt/reference index SHA-256: `ec9ea49c574f672c5f1f100cfd5865dfd2bc3c447687b9770a447abd9311239c`.

**Result**

- Exact pushed commit `223e58634c6e03ef3cb4cc34960fc7be7c526af3`
  passed ownership 1.0, runtime/profile audit, runtime FLA freedom, all four
  sanitizers, and a bitwise deterministic production repeat. Its output and
  all gradients are bitwise identical to attempt 42 and tolerance-correct
  against accepted attempt 38.
- Level 1 advanced without a retest: T=4096 forward+backward improved
  `29.295 -> 28.152 ms` (3.90%) at memory ratio 1.0; all guard rows passed.
- Level 2 measured baseline `[23826,23841,23843,23526,23810]`, median
  `23826 tok/s`, and candidate `[24319,24334,24310,24282,24260]`, median
  `24310 tok/s`: +2.03%, identical `5508.533 MiB`, and 55.65% of the fixed
  43,680 tok/s reference.
- The production profile shows the fused three-output transform at about
  `1.46 ms` per iteration, slower than attempt 42's split transforms despite
  the formal matched gates passing. Attempt 43 is the accepted development
  baseline, not statistical confirmation, quality evidence, official
  retention, a merge, or a default change.

**Next**

- Continue from attempt 43. Split target (q/k) and right transforms to reduce
  three-output kernel pressure without reintroducing the duplicated target
  exponent, then attack the still-dominant pair and parameter VJP kernels.

## 2026-08-09 [agent] reject split M/A transforms

**Context**

- Attempt 44 split the fused target q/k transform from the right transform to
  test whether lower per-kernel storage pressure outweighed an extra launch.
  This was an execution-only intervention and remained bitwise equal to the
  accepted attempt-43 implementation.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_044 \
  --lane optimization <isolated artifact/cache arguments>
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_043 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_044 \
  runs/kda-cuda-development/attempt-00044-ma-split-target-right-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_044 \
  push -u origin kda-cuda/wy-ma-split-target-right-044
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00044-protected-checker`, manifest `334881d6222b0d99e7afad17e6f14524b1e5b91ed435086827784708198f3308`.
- Recovered production comparison: `runs/kda-cuda-development/diagnostics/attempt-00044-ma-split-target-right-gradient`, manifest `62b7290a9d6c03a2b3adf43970956da48652194c7cfaad827bfa0d002d1ce96d`.
- Level 1: `runs/kda-cuda-development/attempt-00044-ma-split-target-right-level1`, manifest `634566d7e9b3e921f98dd35d8deb60066b4572684c0c009cff6bb50c67c01c79`.

**Result**

- Pushed commit `a3423e8ac84b84c5c698bb50ace7adcfd4d3a979`
  passed ownership 1.0, the protected runtime/profile audit, runtime FLA
  freedom, and the deterministic comparison. Output and every gradient were
  bitwise equal to attempt 43.
- Level 1 rejected the split: T=4096 forward+backward changed
  `28.126 -> 28.364 ms` (-0.85%) at memory ratio 1.0. No retest, sanitizer,
  or Level 2 ran.

**Next**

- Retain the fused attempt-43 transform and optimize the parameter and pair
  VJP kernels instead.

## 2026-08-09 [agent] accept key-major parameter VJP baseline

**Context**

- Attempt 45 maps parameter accumulation by key lane: each lane accumulates
  reverse-token `A_log` work and the corresponding `dt_bias` contribution,
  followed by one ordered shared key reduction. This removes 1,024 barriers
  per production group while retaining deterministic accumulation.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_045 \
  --lane optimization <isolated artifact/cache arguments> --sanitizers
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_043 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_045 \
  runs/kda-cuda-development/attempt-00045-key-major-params-level1
# Executed the predeclared baseline-first Level-2 pair once.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_045 \
  push -u origin kda-cuda/wy-vjp-key-major-params-045
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00045-protected-checker`, manifest `5069a752e488cf6ce8462f33f8e9150199fec8a4d213e421b42bbaeb5c43fef2`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00045-key-major-params-gradient`, manifest `fcdc440ee48c1b5cb7a08efd7826a877375eac957380d41b2ebcd054c62c1515`.
- Full sanitizer validation: `runs/kda-cuda-development/validations/validation-00008-key-major-params`, manifest `7066b3ed32f7db87ee5542df625b96879e8c39c623be2e6f341e8d6e8b5f8518`.
- Level 1: `runs/kda-cuda-development/attempt-00045-key-major-params-level1`, manifest `350e48e05428dcc685bbc8b3f74d696f5bbd348310ebe008ec0e00917d57599f`.
- Level 2: `runs/kda-cuda-development/attempt-00045-key-major-params-level2`, manifest `43f45d3f96dc427558e71d75c4d249328cfb6b6221339af8f8e80a2899ebe898`.
- Baseline manifest: `runs/kda-cuda-development/baseline/f0a0c0aaf9.json`, SHA-256 `bbe291e76aa0e5ac58bc18c8ee54cfaa210e31471e26feeecd57ce5e50147c52`.
- Append-only attempt/reference index SHA-256: `9611e172fc3c5a0211a6ffaa849b374357e9b335a956754f2e0cd803e48f3773`.

**Result**

- Exact pushed commit `f0a0c0aaf9e8c1a98f14d2d47af5859e24fad3ed`
  passed ownership 1.0, runtime/profile audit, runtime FLA freedom, all four
  sanitizers, frozen gradient tolerances, and a bitwise deterministic repeat.
  Output and all gradients except `dA_log` were bitwise equal to attempt 43;
  `dA_log` differed only by `2.530e-12` maximum due to the declared reduction
  order change.
- Level 1 advanced without a retest: T=4096 forward+backward improved
  `28.126 -> 26.037 ms` (7.43%) at memory ratio 1.0; all guards passed.
- Level 2 measured baseline `[24399,24321,24385,24376,24461]`, median
  `24385 tok/s`, and candidate `[25394,25454,25420,25210,25491]`, median
  `25420 tok/s`: +4.24%, identical `5508.533 MiB`, and 58.20% of the fixed
  43,680 tok/s reference.
- Attempt 45 is the accepted development baseline. This is not statistical
  confirmation, LM-quality evidence, official retention, a merge, or a
  default change.

**Next**

- Profile attempt 45 once, then use the remaining dominant backward kernel to
  choose the next algebraic strategy toward the fixed 45k objective.

## 2026-08-09 [agent] attribute accepted attempt-45 backward

**Context**

- A single production-shape Nsight Systems attribution was scheduled after
  attempt 45 became the accepted development baseline. The initial invocation
  resolved sources from the coordinator directory and failed before model
  execution; it is preserved separately as invalid.

**Commands**

```bash
PYTHONPATH=/home/veer/Master/projects/experiment_swa_kda_cuda_attempt_045 \
TORCH_EXTENSIONS_DIR=/tmp/kda045-key-major-params-ext-001 \
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=<artifact>/trace \
  /home/veer/Master/projects/experiment_swa_kda/.venv/bin/python \
  /tmp/kda033_nsys.py
nsys stats --report cuda_gpu_kern_sum --format csv \
  --output <artifact>/kern <artifact>/trace.nsys-rep
```

**Artifacts**

- Valid profile: `runs/kda-cuda-development/diagnostics/attempt-00045-production-profile`, manifest `b1e35fbcee62d09147967629cd5fcc7b1a737ca036687b45541c3a12de472bcd`.
- Preserved invalid invocation: `runs/kda-cuda-development/diagnostics/attempt-00045-production-profile-invalid-001`, manifest `3765dfc0b0bbf2b74b25fb82c7923fcb276331fc0e856da87fe70e59b78bd61f`.

**Result**

- The parameter kernel fell from about `2.545 ms` at attempt 43 to
  `0.255 ms` per profiled iteration at attempt 45, a 90.0% reduction.
- The remaining dominant owned kernels are the complete pair/state VJP at
  `4.887 ms`, boundary terms at `1.773 ms`, and finalization at `1.292 ms`.
  This is profile attribution, not a matched speed, quality, or confirmation
  result.

**Next**

- Do not replay rejected attempt-39 pair-ratio caching. First remove redundant
  per-value decay exponentials from the boundary VJP, then return to an
  algebraic/tiled implementation of the dominant complete pair VJP.

## 2026-08-09 [agent] accept key-block boundary VJP baseline

**Context**

- Attempt 46 remaps each reverse boundary update to one block per
  recurrence/key. The block computes the chunk-end decay once, updates all 128
  value lanes, and preserves the original ordered `dD` dot-product reduction.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_046 \
  --lane optimization <isolated artifact/cache arguments>
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_validation_046 \
  --lane optimization <isolated artifact/cache arguments> --sanitizers
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_045 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_046 \
  runs/kda-cuda-development/attempt-00046-boundary-key-block-level1
# Executed the predeclared baseline-first Level-2 pair once, then one profile.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_046 \
  push -u origin kda-cuda/wy-vjp-boundary-key-block-046
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00046-protected-checker`, manifest `6bc8b1cc6bbee06cfae91746dd5f2a126aeab41bc7ea571ea3531d5866220041`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00046-boundary-key-block-gradient`, manifest `63a29653e5a874d1460c666d6277b029948a28cf706d52770fc3569e7f777177`.
- Full sanitizer validation: `runs/kda-cuda-development/validations/validation-00009-boundary-key-block`, manifest `1dc38d5016969445465df2583e0590d490bd82664fe8d82d5fc5f06571e10f8a`.
- Level 1: `runs/kda-cuda-development/attempt-00046-boundary-key-block-level1`, manifest `c3131a8085e1f16f22ee2e14c79c79802a5869184ee13a33e5017678dd7212ca`.
- Level 2: `runs/kda-cuda-development/attempt-00046-boundary-key-block-level2`, manifest `adfc19f2510453ddc0622f036cea58096de3e8eda3113561ae2e83f03864fe3a`.
- Production profile: `runs/kda-cuda-development/diagnostics/attempt-00046-production-profile`, manifest `432d0d51a4ff9624e5a084b1f547132cd6951ad94d0308f22ecbced5ace59bfe`.
- Baseline manifest: `runs/kda-cuda-development/baseline/5edf86274c.json`, SHA-256 `48e2d3dc6ce05811d8ac3d70a2629619ad225a8af34b4e8bde72328ae3c71ceb`.
- Append-only attempt/reference index SHA-256: `7eacaf62d3490e91e31ace86c3a67884f4be5422fbf5d8b954c0519a5c5d2cb2`.

**Result**

- Exact pushed commit `5edf86274c25cdfc473f13412337b048480a35a5`
  passed ownership 1.0, runtime/profile audit, runtime FLA freedom, all four
  sanitizers, and a bitwise deterministic production comparison. Output and
  every gradient were bitwise equal to attempt 45.
- Level 1 advanced: T=4096 forward+backward improved
  `25.667 -> 24.441 ms` (4.78%) at memory ratio 1.0. The T=256 backward row
  regressed 2.05%, within its frozen 5% guard.
- Level 2 measured baseline `[25530,25507,25566,25415,25608]`, median
  `25530 tok/s`, and candidate `[26278,26258,26315,26048,26236]`, median
  `26258 tok/s`: +2.85%, identical `5508.533 MiB`, and 60.11% of the fixed
  43,680 tok/s reference.
- The follow-up profile reduced boundary terms from `1.773 ms` to
  `0.170 ms` per iteration (90.4%). The complete pair/state VJP remains the
  dominant owned kernel at `4.769 ms`. Attempt 46 is the accepted development
  baseline, not confirmation, LM-quality evidence, or official retention.

**Next**

- Implement the complete A/M pair VJP with stable tiled transforms and matched
  BMMs, retaining the scalar kernel only for R/E/P/Q/D terms. Do not replay the
  rejected attempt-39 shared ratio cache.

## 2026-08-09 [agent] reject row-block finalization at Level 2

**Context**

- Attempt 47 maps each independent finalization row to one CUDA block while
  preserving the ordered q/k normalization reductions. It tests whether wider
  row-level occupancy can remove the remaining serial finalization loop.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_047 \
  --lane optimization <isolated artifact/cache arguments>
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_validation_047 \
  --lane optimization <isolated artifact/cache arguments> --sanitizers
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_046 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_047 \
  runs/kda-cuda-development/attempt-00047-finalize-row-block-level1
# Executed the predeclared baseline-first Level-2 pair once.
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00047-protected-checker`, manifest `86ba8b207525925d81c27c90b0c9723d890c5439b96462903c6152120a3769e0`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00047-finalize-row-block-gradient`, manifest `b0c1c58bb3eb297f604d0b2bf75f1ebde966a48146a2f326a31d6a2120fec856`.
- Full sanitizer validation: `runs/kda-cuda-development/validations/validation-00010-finalize-row-block`, manifest `71324bf2a05d5056aaf37fadf92465bf52ee55c63d435c23798d8eb1a089d9e9`.
- Level 1: `runs/kda-cuda-development/attempt-00047-finalize-row-block-level1`, manifest `7d47d2a551bf9fa5bf798fd306754649524a024c905821ff5728868d8c4a8edf`.
- Level 2: `runs/kda-cuda-development/attempt-00047-finalize-row-block-level2`, manifest `5af9e0de0b1b6fb43ea4fd6c79894d32fc58c54afb059875b411cabaa12cb486`.
- Append-only attempt/reference index SHA-256: `750dced9999fc7f3bb5f74b788b5cb0a04a1bafbb6f41d88e6633640d637f241`.

**Result**

- Pushed commit `a3dd2f236887b518c17c82e01d45c96ca9c25698`
  passed ownership 1.0, runtime FLA freedom, all four sanitizers, and remained
  bitwise equal to attempt 46 across output, all gradients, and its repeat.
- Level 1 advanced: T=4096 forward+backward improved
  `24.536 -> 23.214 ms` (5.39%) with memory ratio 1.0.
- Level 2 measured baseline `[26216,26171,26223,26040,26221]`, median
  `26216 tok/s`, and candidate `[26787,26685,26782,26453,26675]`, median
  `26685 tok/s`: +1.79%, below the frozen 2% gate, with identical
  `5508.533 MiB`. No retest was run.
- Attempt 47 is a correct rejected milestone. Attempt 46 remains the accepted
  development baseline; neither result is confirmation or quality evidence.

**Next**

- Stop independent finalization scheduling and move to the dominant complete
  A/M pair VJP with a bounded algebraic/tiled strategy.

## 2026-08-09 [agent] reject two-way scalar pair VJP

**Context**

- Attempt 48 splits each row/key's causal A/M source and target loops across
  two 128-lane halves, then deterministically combines the three adjoints. The
  existing ordered beta dot tail remains unchanged. This tests scalar temporal
  parallelism without replaying attempt-39 ratio caching.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_048 \
  --lane optimization <isolated artifact/cache arguments>
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_046 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_048 \
  runs/kda-cuda-development/attempt-00048-two-way-time-level1
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00048-protected-checker`, manifest `8604eebf63a7c5ddb471559bfea2b1de8031a50db3104842d682ef4db7002c5d`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00048-two-way-time-gradient`, manifest `f9846a01e682e67b0e213d2a74c1f118a6d8a2b830062f002bd7442f9b7ee5e3`.
- Preserved invalid artifact invocation: `runs/kda-cuda-development/diagnostics/attempt-00048-gradient-invalid-invocation-001`, manifest `11bdb3834a72a58fef4d7936c2d5ada6086a611f9c82d51adfd4ca4dfd10599a`.
- Level 1: `runs/kda-cuda-development/attempt-00048-two-way-time-level1`, manifest `f4399a19f3518861d90cdc3de7f7692d29ebf81fd13f2c40bff1218815a064e7`.
- Append-only attempt/reference index SHA-256: `4a6535bad56566e9e62c2e56188a610b9e941eae9f8594e41284a24d671fe89c`.

**Result**

- Pushed commit `d38eac1137a3f4eb8be6572f66d51d887edaea8d`
  passed ownership 1.0 and runtime FLA freedom. It was deterministic and well
  inside frozen tolerances; the largest gradient difference from attempt 46
  was `3.638e-12`, while output, `dv`, and `dbeta` remained bitwise equal.
- Level 1 rejected the intervention: T=4096 forward+backward changed
  `24.281 -> 25.711 ms` (-5.89%), exceeding the frozen 5% important-regression
  guard, with memory ratio 1.0. No sanitizer, Level 2, or retest ran.
- Attempt 46 remains the accepted baseline. This result is neither
  confirmation nor quality evidence.

**Next**

- Stop scalar pair scheduling/cache variants. The remaining credible strategy
  is a bounded stable algebraic A/M VJP, designed to avoid attempt 29's 1,284
  BMM calls and attempt 30's full-history memory expansion.

## 2026-08-09 [agent] reject split warp-reduced pair VJP

**Context**

- Attempt 49 separates base R/E/P/Q/D work from complete A/M pair work. Eight
  warps reduce causal time for interleaved key channels, while four source
  warps compute the beta pair dot. It avoids atomics, ratio caches, BMMs, and
  additional persistent history, but adds two row-granular launches per group.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_049 \
  --lane optimization <isolated artifact/cache arguments>
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_046 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_049 \
  runs/kda-cuda-development/attempt-00049-warp-pair-level1
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00049-protected-checker`, manifest `cffa528d45f4b1feda924d9b0dcfbdc71252e64d83e0a8de506e3a99a0119870`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00049-warp-pair-gradient`, manifest `8fba42ac6efec13190c51da27ea8ea56a75e90a28d0b89a9a1a7785a6bed979e`.
- Level 1: `runs/kda-cuda-development/attempt-00049-warp-pair-level1`, manifest `4d62b4c71a69b66e411ef9d44765c29e4ac3d4331828442cbebf746edb854cf3`.
- Append-only attempt/reference index SHA-256: `f235dfc4e079040430b10a6b6e1c39ac2e426a5f1dbd4512ca716db2620369a8`.

**Result**

- Pushed commit `1d377d656e240759959f0adafc0da9c682cfb897`
  passed ownership 1.0 and runtime FLA freedom. Its deterministic gradient
  differences were tiny (maximum `3.638e-12`), while output, `dv`, and `dbeta`
  remained bitwise equal to attempt 46.
- Level 1 rejected the split: T=4096 forward+backward changed
  `24.431 -> 30.829 ms` (-26.19%), with memory ratio 1.0. No sanitizer,
  Level 2, or retest ran.
- Attempts 48-49 show that extra pair scheduling overwhelms scalar arithmetic
  savings. Attempt 46 remains the accepted baseline, not confirmation or
  quality evidence.

**Next**

- Retain the fused row-per-block pair kernel. Reassess whole-operator launch
  and BMM structure from the accepted attempt-46 profile before another major
  strategy boundary; do not continue pair scheduling/cache variants.

## 2026-08-09 [agent] reject stacked shared-right forward scan

**Context**

- Attempt 50 stacks the two chunk-scan products that share recurrent state and
  the two products that share `z`. It reduces the forward scan from four to two
  BMM dispatches per chunk while aliasing scratch storage so peak allocation is
  unchanged. This is a forward launch-structure test on accepted attempt 46.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_050 \
  --lane optimization <isolated artifact/cache arguments>
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_046 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_050 \
  runs/kda-cuda-development/attempt-00050-stacked-scan-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_050 \
  push -u origin kda-cuda/wy-scan-stacked-bmm-050
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00050-protected-checker`, manifest `8f12182e4a79f98c36026cf6ca90d97be2e060bf8cc1f093f2f701f02ed55afe`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00050-stacked-scan-gradient`, manifest `8cfe4d7009c3b4d772901f3818614465376d61e8c95a9f0c2a72736a07e5deae`.
- Preserved invalid invocation 1: `runs/kda-cuda-development/diagnostics/attempt-00050-gradient-invalid-invocation-001`, manifest `756fed2271267954e8610574baa2ea53914543fc4d308e71ef836e94e2ccc0f0`.
- Preserved invalid invocation 2: `runs/kda-cuda-development/diagnostics/attempt-00050-gradient-invalid-invocation-002`, manifest `61a0b22a8891f8ed4de9e7d22b484b1361663f591af6d1bc70a9e60dfe58cd51`.
- Level 1: `runs/kda-cuda-development/attempt-00050-stacked-scan-level1`, manifest `6650200f66233ee6c1c73d9293bdd019798a3c0a8f05340e43ef01dcba7f7b4a`.
- Append-only attempt/reference index SHA-256: `1289e0d5ad20c7b0ffc76873eeff0955b86cc3ed450e6a4f8b88e073c82301ae`.

**Result**

- Pushed commit `b05de084634f49b481c9e92df6508e4a2da87392`
  passed ownership 1.0, runtime/profile audit, and runtime FLA freedom. Output
  and all seven gradients were bitwise equal to attempt 46 and to the
  deterministic repeat.
- Level 1 rejected the intervention: T=4096 forward+backward improved
  `24.209 -> 23.630 ms` (2.39%), below the frozen 3% gate, with memory ratio
  1.0. T=4096 forward alone improved only 0.17%. No sanitizers, Level 2, or
  retest ran.
- Two setup failures were invalid invocations before model execution and are
  preserved separately. Attempt 46 remains the accepted baseline. This result
  is neither confirmation nor quality evidence.

**Next**

- Stop forward BMM dispatch stacking. Return to the requested C64 backward
  transition: replace full token history with chunk-boundary recomputation and
  reverse scan, then implement the complete WY/UT VJP without replaying the
  pathological naive parent.

## 2026-08-09 [agent] accept bounded stable pair VJP baseline

**Context**

- Attempt 51 replaces the scalar complete A/M pair loops with stable 16x16
  transformed FP32 VJPs. It batches four of the ten causal tile pairs at a
  time inside each eight-chunk reverse group, reducing each batch to two
  stacked adjoint BMMs plus one pre-M BMM.
- The implementation retains attempt 46's group-boundary reconstruction and
  reverse state scan. Dead U/Q/H/state/W/R group buffers supply all pair
  operands and outputs, so the intervention adds no measured allocation and
  does not restore full token-state history.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_051 \
  --lane optimization <isolated artifact/cache arguments>
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_046 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_051 \
  runs/kda-cuda-development/attempt-00051-pair-batched-tiles-level1
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_validation_051 \
  --lane optimization <isolated artifact/cache arguments> --sanitizers
# Executed the saved baseline-first Level-2 plan exactly once, then one
# candidate production profile.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_051 \
  push -u origin kda-cuda/wy-pair-batched-tiles-051
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00051-protected-checker`, manifest `a50ee44a3833f4aae1e9b56d4f35c1e5d50527faa3e141dff1e4573b02d653e5`.
- Invalid compile checker: `runs/kda-cuda-development/diagnostics/attempt-00051-protected-checker-invalid-compile-001`, manifest `a015ffc0158b15312abcd769cd1258f989955c11f39e112fa7008e5e3b1c9f42`.
- Pre-equation-fix checker: `runs/kda-cuda-development/diagnostics/attempt-00051-protected-checker-pre-equation-fix-002`, manifest `a24ff5b1b2ccdb235cb8b3f6075bbff96f28e7d5895745b382cf83070b519161`.
- Invalid production equation bundle: `runs/kda-cuda-development/diagnostics/attempt-00051-pair-tiles-gradient-invalid-equation-001`, manifest `3f345070de64422746b425c33815ced26444389546bc39b0402bab5d69f30980`.
- Correct production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00051-pair-tiles-gradient`, manifest `ebbfa5fd406a0f3c0f6c5499c5b9cc309a7d1e1eeceefa62347ee303b1d89ae4`.
- Full sanitizer validation: `runs/kda-cuda-development/validations/validation-00011-pair-batched-tiles`, manifest `1524b99d5a259ef5ed1ce134b2e0beeb50ed8968cd00336b1817369d78d5340a`.
- Level 1: `runs/kda-cuda-development/attempt-00051-pair-batched-tiles-level1`, manifest `f94c24411514e01eaaf42271e2499c305a91e945c043225c76d545ccf1151478`.
- Level 2: `runs/kda-cuda-development/attempt-00051-pair-batched-tiles-level2`, manifest `0c576eb5138d1b14ac69bb27ffbcfc8e28cb6f99c6c005d200d3f3cf2a0bb6ab`.
- Production profile: `runs/kda-cuda-development/diagnostics/attempt-00051-production-profile`, manifest `f0fb168150323383fcc207685f031b3c0c3f88b7841c20b2664391a48ad05f20`.
- Baseline manifest: `runs/kda-cuda-development/baseline/336a084f4d.json`, SHA-256 `fc354f9424cec752c8f533bb362cdcc5e15360f2b4d0f9b6351614f8a65795b0`.
- Append-only attempt/reference index SHA-256: `d1fecf64a16e5547bcfcb7a097c656c1aded9110ddba567120ce4762595c1905`.

**Result**

- Pushed commit `336a084f4d8ef34b673619ee020076d9e77a7399`
  passed ownership 1.0, runtime/profile audit, runtime FLA freedom, all four
  sanitizers, and a bitwise deterministic repeat. Output, `dv`, and `dbeta`
  are bitwise equal to attempt 46; the largest gradient delta is `3.638e-12`.
- The first compiled snapshot omitted the transform factors when mapping BMM
  adjoints back to q/k and failed the production equation gate by orders of
  magnitude. It was never timed. The exact failed tensors and the preceding
  const-output compile failure are preserved.
- Level 1 advanced: T=4096 forward+backward improved
  `24.586 -> 22.662 ms` (7.83%) at memory ratio 1.0; every guard row passed.
- The single Level-2 pair measured baseline
  `[26073,26126,26105,26008,26010]`, median `26073 tok/s`, and candidate
  `[27043,27043,27070,26966,26945]`, median `27043 tok/s`: +3.72%, identical
  `5508.533 MiB`, and 61.91% of the fixed 43,680 tok/s reference.
- Profiling reduced the old scalar complete-pair kernel from `4.769 ms` to a
  `0.493 ms` base-vector kernel plus `0.923 ms` pair packing and `0.649 ms`
  pair accumulation per iteration, excluding the associated BMM kernels.
  Finalization is now the largest named project kernel at `1.163 ms`.
- Attempt 51 is the accepted development baseline. It is not statistically
  confirmed, official retention, a merge/default change, or quality evidence.

**Next**

- Test a five-pair batch with explicitly released dead H/state storage to cut
  pair pack/accumulate dispatches from three to two batches per reverse group,
  while retaining the 3% memory gate. If that does not advance, profile-guided
  finalization fusion is the next distinct axis. Continue toward 45k.

## 2026-08-09 [agent] reject five-pair WY batching

**Context**

- Attempt 52 increases attempt 51's pair batch from four to five, explicitly
  ending dead H/state/Q/W views before allocating the larger scratch. This
  reduces each reverse group from three pair batches to two.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_052 \
  --lane optimization <isolated artifact/cache arguments>
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_051 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_052 \
  runs/kda-cuda-development/attempt-00052-pair-batch5-level1
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00052-protected-checker`, manifest `76c944f0fc665f1707dffaf5096960ddca8bf0c3bd07099549da287fbd156720`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00052-pair-batch5-gradient`, manifest `71b73f4a4856a823bf05ffdf6b481ab1d5d21dddb45e098e2c68e08ab9992bf8`.
- Level 1: `runs/kda-cuda-development/attempt-00052-pair-batch5-level1`, manifest `0e2210d24d64e58c99a2daa537a760cf52a140f13e90730fbb3ff05c7a904af1`.
- Append-only attempt/reference index SHA-256: `0d1a02d8bb299f5c29bc8356fc9091181eb175a640a4a09470c1f63ce1d8adde`.

**Result**

- Pushed commit `c0f7d50ff69e369d086e7ece86b4f6f1abaf6e07`
  passed ownership/runtime gates, remained deterministic, and stayed inside
  frozen tolerances. Output, `dv`, and `dbeta` are bitwise equal to attempt 51;
  the largest gradient delta is `4.55e-13`.
- Level 1 did not advance: T=4096 forward+backward improved only
  `22.473 -> 22.327 ms` (0.65%). Peak allocation ratio was `1.00763`, within
  the resource gate but larger than attempt 51. No sanitizer, Level 2, or
  retest ran.
- Attempt 51 remains the accepted development baseline. Neither attempt is
  confirmation or quality evidence.

**Next**

- Stop pair-batch width tuning. From exact attempt 51, fuse finalization and
  local parameter accumulation to target the profile's largest named project
  kernel while preserving ordered reductions and BF16 rounding.

## 2026-08-09 [agent] accept chunk-partial finalization baseline

**Context**

- Attempt 53 starts from exact accepted attempt 51 and removes the full FP32
  raw-gate-gradient history. A row-parallel finalization kernel recomputes the
  gate derivative once per chunk and emits deterministic `[group rows, 2, 128]`
  parameter partials; the recurrence/key path reduces those chunks in fixed
  order without atomics.
- This is the requested chunk-boundary recomputation/reverse-scan complete
  WY/UT VJP path, with only `nanochat/mixers/cuda_kda/chunk_wy_backward.cu`
  changed.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_053 \
  --lane optimization <isolated artifact/cache arguments>
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_051 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_053 \
  runs/kda-cuda-development/attempt-00053-finalize-partials-level1
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_validation_053 \
  --lane optimization <isolated artifact/cache arguments> --sanitizers
# Executed the saved baseline-first Level-2 plan exactly once, then one
# candidate production profile.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_053 \
  push -u origin kda-cuda/wy-finalize-chunk-partials-053
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00053-protected-checker`, manifest `661ff0b763129c0b9b930089037c1968f786debf9730150e75170d116535d9c8`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00053-finalize-partials-gradient`, manifest `2cefa444a94a760c6708674191ca7d4ad7ef981ca7745004788fcaf793951b33`.
- Full sanitizer validation: `runs/kda-cuda-development/validations/validation-00012-finalize-partials`, manifest `4f32f08ff8449051607d2afc54b964059ea4aa3128af25c3cc48b1c105c9830e`.
- Level 1: `runs/kda-cuda-development/attempt-00053-finalize-partials-level1`, manifest `23d5fd882539d58790d7b0e2241dce8b4d2bb679ef92b854acb418edd99cd127`.
- Level 2: `runs/kda-cuda-development/attempt-00053-finalize-partials-level2`, manifest `a50d63b9d81d108271c228f2f526e7992483fd87d0a998d1d9110e2849ca3272`.
- Production profile: `runs/kda-cuda-development/diagnostics/attempt-00053-production-profile`, manifest `6cb95033a4a087e20ff4727330d89c3ea42b36b14046784b0d4a8c0a485e445b`.
- Baseline manifest: `runs/kda-cuda-development/baseline/0d168b5621.json`, SHA-256 `34464160df5af93905913a708dcc3e4a12216ded9b6709a4174b35b82aa7e4e8`.
- Append-only attempt/reference index SHA-256: `3676f47f861eb400fac9a240b1b826df76df475455cee430541d5e0b9fd5cd24`.

**Result**

- Pushed commit `0d168b5621c43218667754f55167601cc6a3f9d0`
  passed ownership 1.0, runtime/profile audit, runtime FLA freedom, all four
  sanitizers, frozen correctness, and a bitwise deterministic repeat. Output,
  `dq`, `dk`, `dv`, `draw_gate`, and `dbeta` are bitwise equal to attempt 51;
  the largest FP32 parameter-gradient delta is `2.842e-14`.
- Level 1 advanced: T=4096 forward+backward improved
  `22.626 -> 21.661 ms` (4.27%), and peak allocation fell from 206,064,128 to
  204,540,416 bytes (ratio 0.99261). The T=256 forward+backward regression was
  2.63%, within the frozen 5% guard; every guard passed.
- The single baseline-first Level-2 pair measured baseline
  `[27003,27071,27043,26787,27012]`, median `27012 tok/s`, and candidate
  `[27681,27671,27599,27425,27674]`, median `27671 tok/s`: +2.44%, identical
  `5508.533 MiB`, and 63.35% of the fixed 43,680 tok/s reference.
- Profiling reduced finalization plus parameter work from `1.417 ms` in attempt
  51 to `0.381 ms` per iteration. Pair packing plus accumulation is now the
  largest named owned backward path at `1.534 ms` per iteration, excluding its
  associated BMMs.
- Attempt 53 is the accepted development baseline. It is not statistically
  confirmed, official retention, a merge/default change, or quality evidence.

**Next**

- From exact attempt 53, target the pair pack/accumulate CUDA kernel structure
  without replaying attempt 52's pair-batch-width tuning. Preserve stable
  transforms, ordered reductions, BF16 rounding, and every ownership and
  correctness gate while continuing toward 45k.

## 2026-08-09 [agent] reject fast pair exponentials

**Context**

- Attempt 54 substitutes CUDA's fast FP32 exponential only inside the stable,
  tile-centered pair pack and accumulation transforms. Batch width, equations,
  reduction order, scratch storage, and BF16 writes remain identical to
  accepted attempt 53.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_054 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seeded B=2/H=3/T=4096 saved-gradient comparison plus repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_053 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_054 \
  runs/kda-cuda-development/attempt-00054-pair-fast-exp-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_054 \
  push -u origin kda-cuda/wy-pair-fast-exp-054
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00054-protected-checker`, manifest `ac91482ee89898e2c9887d844270dbf9055daa9b8b12a7d0969e9e9c98bdc79c`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00054-pair-fast-exp-gradient`, manifest `ce3bd72044733794515aade7b612612cbf4ff1607ede94fa3c2bcbde2546842b`.
- Invalid source-root invocation: `runs/kda-cuda-development/diagnostics/attempt-00054-gradient-invalid-source-root-001`, manifest `6717d6f6efa76cdfb0d1da612e2b8d51fbe046b321295d11321f385443090ce8`.
- Invalid package-precedence invocation: `runs/kda-cuda-development/diagnostics/attempt-00054-gradient-invalid-pythonpath-002`, manifest `da8eb5820dcfe3e516b20662065007f722240fa1b1ce6a7c36e6ae3fab2df5d0`.
- Level 1: `runs/kda-cuda-development/attempt-00054-pair-fast-exp-level1`, manifest `b2305a6982edb3e9d57c8da40c25036771224c909690a9c52d1c2f47381979fa`.
- Append-only attempt/reference index SHA-256: `766a92d78d5530a65818a1b9b53aa8216bb6a34906272339bea5647527b8b556`.

**Result**

- Pushed commit `576797a09c196a0035236f5948c0681a8e365b05`
  passed ownership 1.0, runtime/profile audit, and runtime FLA freedom. The
  production comparison was deterministic and inside frozen tolerances;
  output, `dv`, and `dbeta` are bitwise equal to attempt 53, and the largest
  gradient delta is `3.638e-12`.
- Two setup failures occurred before CUDA or measurement: first the source root
  resolved against the coordinator, then `uv --project` imported the
  coordinator package before the candidate. Both are preserved as invalid
  invocations; the valid run pinned candidate working directory and
  `PYTHONPATH` while using the coordinator environment.
- Level 1 rejected the intervention: T=4096 forward+backward changed
  `21.402 -> 22.156 ms` (-3.52%) with memory ratio 1.0. No sanitizer, Level 2,
  profile, or retest ran.
- Attempt 53 remains the accepted development baseline. This result is neither
  confirmation nor quality evidence.

**Next**

- Stop pair exponential intrinsic substitution. Select a distinct structural
  backward axis from exact attempt 53; do not resume pair-batch-width tuning.

## 2026-08-09 [agent] reject fused row-owned beta dot

**Context**

- Attempt 55 removes the third pair BMM used only for the unscaled M dot in
  `dbeta`. The existing row-owned accumulation block instead assigns four
  source rows to four warps, reduces the 128-channel stable packed dot in a
  fixed order, and combines it without atomics or persistent M history.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_055 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seeded B=2/H=3/T=4096 saved-gradient comparison plus repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_053 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_055 \
  runs/kda-cuda-development/attempt-00055-beta-dot-fused-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_055 \
  push -u origin kda-cuda/wy-beta-dot-fused-055
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00055-protected-checker`, manifest `3f79c96193355e9a61f0649d389d5b864ee5a4d8e7b8f04fc3a30ca943726cdd`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00055-beta-dot-fused-gradient`, manifest `5eb4c91a943e71ed7b8bbf571e1a060019b41eae8779062c98d23f09dbcaacbb`.
- Level 1: `runs/kda-cuda-development/attempt-00055-beta-dot-fused-level1`, manifest `ed49b05fe7801c5c7682de4f7c39c12cd00a140c7056771d09d38cbd79bf2ddb`.
- Append-only attempt/reference index SHA-256: `510b5b29d7d4c46c271761ed2f2810224357bb86ed4ddfde9ffeff0a97e316c2`.

**Result**

- Pushed commit `1c9381d663d3bdc26f0c6325aa07c52d291e6092`
  passed ownership 1.0, runtime/profile audit, and runtime FLA freedom. Output,
  all seven gradients, and the deterministic repeat are bitwise identical to
  attempt 53.
- Level 1 did not advance: T=4096 forward+backward changed
  `21.580 -> 21.605 ms` (-0.12%) with memory ratio 1.0. The fixed-order fused
  warp reductions cost at least as much as the removed small BMM. No sanitizer,
  Level 2, profile, or retest ran.
- Attempt 53 remains the accepted development baseline. This result is neither
  confirmation nor quality evidence.

**Next**

- Keep the efficient third pair BMM. Choose a different structural backward
  path from exact attempt 53; do not continue pair-batch, exponential, or beta
  reduction variants.

## 2026-08-09 [agent] preserve hybrid WMMA scan below advance gate

**Context**

- With attempt 53's complete backward near 2 ms beyond forward-only time, the
  campaign reached the declared point for revisiting preserved attempt 27's
  tensor-core scan. Attempt 56 combines attempt 53's stable FP32 tiled A/M
  construction and accepted backward with only the prior 32-value BF16-WMMA,
  FP32-accumulator forward scan.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_056 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seeded B=2/H=3/T=4096 saved-gradient comparison plus repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_053 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_056 \
  runs/kda-cuda-development/attempt-00056-hybrid-wmma-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_056 \
  push -u origin kda-cuda/wy-hybrid-wmma-scan-056
```

**Artifacts**

- Invalid generated-patch compile: `runs/kda-cuda-development/diagnostics/attempt-00056-protected-checker-invalid-compile-001`, manifest `36b0c6bdb3e00f178cdb390f0566f78c088297da9400491c06a1f341b95a7ab1`.
- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00056-protected-checker`, manifest `8eb3d78c7241a48581c465e784f4767c3a045f1fbb73534ec29f0aba0c81c91e`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00056-hybrid-wmma-gradient`, manifest `443613143d8b28d2b635d511e8f63bd141ad9862b94971cba8a6d563ca42bd44`.
- Level 1: `runs/kda-cuda-development/attempt-00056-hybrid-wmma-level1`, manifest `535b6923d862e0183963f62dbf308956dcb82eb0a595d3d9b9e25b32887f7c62`.
- Append-only attempt/reference index SHA-256: `ef57cb5c7d7f975deb8a4c43ad8589de25365cb7f850635f97df164754310a1e`.

**Result**

- The first checker stopped at compilation because the generated transplant
  left one literal `+` line. It emitted no measurement and is preserved. The
  corrected pushed commit `f5146a5a7ecd6bc54772e91704dcd38e7c25eac3`
  passed ownership 1.0, runtime/profile audit, and runtime FLA freedom.
- Production output and gradients remained finite and deterministic. Maximum
  output delta was `4.8828125e-4`; the largest gradient delta was `5.821e-11`,
  both within frozen tolerances.
- Level 1 did not advance: T=4096 forward+backward improved
  `21.787 -> 21.221 ms` (2.60%), below the 3% gate, with memory ratio 1.0.
  Forward-only improved 0.33%. No retest, sanitizer, Level 2, or profile ran.
- Attempt 53 remains accepted. Attempt 56 is a valid preserved tensor-core
  strategy-boundary option, not confirmation, quality evidence, or retention.

**Next**

- Return to exact FP32 attempt 53. Stack the same-right tiled A/M construction
  products into one FP32 batched call without changing scan precision or
  allocation; do not tune WMMA tile geometry.

## 2026-08-09 [agent] reject stacked A/M construction lifetime

**Context**

- Attempt 57 stacks each tile's FP32 A and M products into one batched BMM in
  both forward and backward reconstruction. The duplicated right operand uses
  the previously unused quarter of P scratch, and combined A/M storage has the
  same nominal bytes as the two parent allocations.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_057 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seeded B=2/H=3/T=4096 saved-gradient comparison plus repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_053 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_057 \
  runs/kda-cuda-development/attempt-00057-stacked-am-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_057 \
  push -u origin kda-cuda/wy-stacked-am-build-057
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00057-protected-checker`, manifest `3a3f49921bd3b7fc276a174c350930ba9a2a5d2d9d410cb9692a0a959b426b95`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00057-stacked-am-gradient`, manifest `72bc41a473d9a37c2ed0da9394ece018b9fef2c1727cae8aa6739e74c08481cb`.
- Level 1: `runs/kda-cuda-development/attempt-00057-stacked-am-level1`, manifest `0d52381d0a080e1f63c2efddd5521690ca515dad19f02f2a973d37ed9abaaaf2`.
- Append-only attempt/reference index SHA-256: `7a0b935189defcfa445f035b58927aed0ef9d0821bd9b62d60e30ac60593b78e`.

**Result**

- Pushed commit `9b05e8e83f197f68b2fe968b460a7f1cddfa44da`
  passed ownership/runtime gates. Output, every gradient, and the repeat are
  bitwise identical to attempt 53.
- Level 1 rejected the committed form: T=4096 forward-only improved
  `19.677 -> 18.998 ms` (3.45%), but forward+backward regressed
  `21.786 -> 22.320 ms` (-2.45%). Peak allocation rose to 210,307,584 bytes,
  ratio 1.02820, because A's view retains the combined allocation after M is
  dead. No sanitizer, Level 2, profile, or retest ran.
- Attempt 53 remains accepted. This is neither confirmation nor quality
  evidence.

**Next**

- Preserve attempt 57 unchanged. In a separate child, write T in-place over M
  after the lower solve, eliminating the separate T allocation and the dead-M
  lifetime penalty while retaining stacked FP32 construction.

## 2026-08-09 [agent] invalidate unsynchronized in-place T solve

**Context**

- Attempt 58 aliases T over M in attempt 57's forward and backward lower
  solves, intending to remove the separate T allocation. The first production
  comparison exposed that the existing end-of-row barrier is insufficient for
  aliasing: short column sums can write T into an M row while longer column
  sums are still reading that row.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_058 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seeded B=2/H=3/T=4096 saved-gradient comparison plus repeat only.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_058 \
  push -u origin kda-cuda/wy-stacked-am-inplace-t-058
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00058-protected-checker`, manifest `7d1b05ba52d527cd8ad5899acd3655d9dd3bc6cae9b8a81c1fd1f987ae829d02`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00058-inplace-t-gradient`, manifest `8c0764657e6aea8da620a80ddcffd0ffd0c488c5e95dff9c4127e973a2d8905c`.
- Append-only attempt/reference index SHA-256: `9649c6b38ee7fca651c40c8b846f5a85946e5eae760f56fafa880659a3e5d782`.

**Result**

- Pushed commit `1c12e671d82b44035a5760d44d85f622ea130851`
  passed the bounded protected checker and was deterministic, but it is not a
  race-free production candidate. Maximum output delta was `1.221e-4` and the
  largest gradient delta was `7.276e-12`; tolerance does not excuse the
  unsynchronized read/write dependency.
- No Level 1, sanitizer-valid claim, Level 2, profile, or retest exists. The
  exact branch and tensors are preserved as an invalid dependency experiment.
  Attempt 53 remains accepted.

**Next**

- Add a block barrier after every row's triangular reads and before its T
  writes in a separate child. Require bitwise production restoration and the
  normal ownership, sanitizer, determinism, and performance gates.

## 2026-08-09 [agent] reject synchronized in-place T repair

**Context**

- Attempt 59 adds the required pre-write block barrier to both in-place lower
  solves from attempt 58. It retains attempt 57's stacked FP32 A/M constructor
  and removes the read/write dependency before any timing.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_059 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seeded B=2/H=3/T=4096 saved-gradient comparison plus repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_053 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_059 \
  runs/kda-cuda-development/attempt-00059-inplace-t-sync-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_059 \
  push -u origin kda-cuda/wy-stacked-am-inplace-t-sync-059
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00059-protected-checker`, manifest `792aa271d6e0ed11f3d4bbaafac163b92f0aae449db4e2a656bb3d1c692bcc20`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00059-inplace-t-sync-gradient`, manifest `d533a7770a9f90c3305d432ad114b0480a2d72ba86cf257708d91966dfcf4ad3`.
- Level 1: `runs/kda-cuda-development/attempt-00059-inplace-t-sync-level1`, manifest `e7c4e28f940fcb8f8f974a490038c0c52ebd89f838046e23db9888ae792c45c6`.
- Append-only attempt/reference index SHA-256: `2ce8aa47d1c77bb03d05f09e6e6d2098e9626f9a2daa4c0abb399c109859d222`.

**Result**

- Pushed commit `f269c84d97c578080a0d68a36d31f83dab56e00a`
  passed ownership/runtime gates. The pre-write barrier restored bitwise
  equality against attempt 53 for output, all gradients, and repeat.
- Level 1 rejected the repaired design: T=4096 forward+backward changed
  `21.639 -> 21.899 ms` (-1.20%), forward-only regressed 0.68%, and peak
  allocation ratio was 1.00449. The required row barriers erase the stacked
  construction gain. No sanitizer, Level 2, profile, or retest ran.
- Attempt 53 remains the accepted development baseline. Attempts 57-59 are a
  fully preserved exact dispatch/lifetime/repair chain, not confirmation or
  quality evidence.

**Next**

- Abandon stacked A/M plus in-place T. Return to exact attempt 53 and choose a
  distinct whole-forward structural path; do not recombine rejected WMMA,
  stacked-construction, or synchronization variants without new evidence.

## 2026-08-09 [agent] reject fused scan state update

**Context**

- Attempt 60 computes the independent `E^T Z` product before state mutation,
  then combines output writing, boundary decay, and delta addition in one
  kernel. Explicit `__fmul_rn` and `__fadd_rn` preserve the two parent FP32
  rounding points while removing 64 state-add launches and one state pass.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_060 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seeded B=2/H=3/T=4096 saved-gradient comparison plus repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_053 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_060 \
  runs/kda-cuda-development/attempt-00060-state-update-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_060 \
  push -u origin kda-cuda/wy-scan-state-update-fused-060
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00060-protected-checker`, manifest `1da3fd67aa432649bd2bb4be713482061ea4857e63d3012cd557b631c7e28246`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00060-state-update-gradient`, manifest `338b9a0cb461727e64729e566eb252d912fe5428cd96547fa23258c6c3059773`.
- Level 1: `runs/kda-cuda-development/attempt-00060-state-update-level1`, manifest `d38dcd12d335594bf665ebec6b11217f3dfe37f029846f881df74faadde1fd9b`.
- Append-only attempt/reference index SHA-256: `145a79aa0d879a7463efa3bc4dc617562fe21582c063513686b04a1e789c54e5`.

**Result**

- Pushed commit `ad5d1087cbeaf9a18e83f1ac078de05e2e7173fc`
  passed ownership/runtime gates. Output, all seven gradients, and repeat are
  bitwise identical to attempt 53.
- Level 1 rejected the intervention: T=4096 forward improved 1.09%, but
  forward+backward changed only `21.575 -> 21.566 ms` (+0.04%). T=256
  forward+backward regressed 5.42%, beyond the 5% guard. Memory ratio was 1.0.
  No sanitizer, Level 2, profile, or retest ran.
- Attempt 53 remains accepted. This is neither confirmation nor quality
  evidence.

**Next**

- Preserve separate output-decay and state-add kernels. Return to exact attempt
  53 and choose a new whole-forward structural axis rather than further
  launch-only fusion.

## 2026-08-09 [agent] reject configured TF32 for WY BMMs

**Context**

- Attempt 61 removes the explicit `at::NoTF32Guard` from the attempt-53 WY
  forward and backward operators. This allows their homogeneous ATen FP32 BMMs
  to follow the repository-wide `torch.set_float32_matmul_precision("high")`
  setting while leaving all project CUDA kernels and configurations unchanged.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_061 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seeded B=2/H=3/T=4096 saved-gradient comparison plus repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_053 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_061 \
  runs/kda-cuda-development/attempt-00061-tf32-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_061 \
  push -u origin kda-cuda/wy-tf32-bmm-061
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00061-protected-checker`, manifest `0297748fece66352890f0da00d0c8362ef459eb3d2d913f42d6463d553154728`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00061-tf32-gradient`, manifest `aeb50a654783ca5f6ec83b224516d21a46cc5243b6e9ff0a678b8f84b5537ff1`.
- Level 1: `runs/kda-cuda-development/attempt-00061-tf32-level1`, manifest `df5df339964b2f46436b8fdbb075ae88d82b67978446a1e7e13afd51a59ac238`.
- Append-only attempt/reference index SHA-256: `558879e87e98726cec239840c4c3bf09f7f9faff32d36e74706b28c265aa8d16`.

**Result**

- Pushed commit `6e1a5faae045058029280534ffb1662d660719b5`
  passed the protected ownership and runtime audits. Output, all seven
  production-shape gradients, and the independent repeat are bitwise identical
  to attempt 53.
- Level 1 rejected the intervention: T=4096 forward+backward changed
  `21.427 -> 21.692 ms` (-1.24%), forward-only regressed 0.40%, and peak
  allocation was unchanged. Removing the guard did not expose a useful faster
  path under the configured `high` matmul precision on this workload.
- No sanitizers, Level 2, profile, confirmation, or retest ran. Attempt 53
  remains the accepted development baseline; this is not quality or
  statistically confirmed evidence.

**Next**

- Preserve attempt 61 unchanged. Return to attempt 53 and target a structural
  reduction in the dominant generic BMM or backward pair-pack/accumulate work;
  do not pursue global TF32 permission further without new kernel evidence.

## 2026-08-09 [agent] reject reverse-pass transform reuse

**Context**

- Attempt 62 computes each group's `U=T*P` and `W=T*Q` once during boundary
  reconstruction, writes those results into the global P/Q storage after its
  last original-value use, and reuses them in the reverse pass. Original P/Q
  are rebuilt exactly into group-local buffers, removing 16 repeated BMMs
  without increasing peak allocation.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_062 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seeded B=2/H=3/T=4096 saved-gradient comparison plus repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_053 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_062 \
  runs/kda-cuda-development/attempt-00062-reuse-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_062 \
  push -u origin kda-cuda/wy-reuse-transforms-062
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00062-protected-checker`, manifest `22c16ef482c7990b76733b9e44470bf1d7fd6021e56e6c3d07483ed2d747587e`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00062-reuse-gradient`, manifest `013b4dc086f08f1436e3a43da78f9f29f441e5f22383146ca0f6483507c20b46`.
- Level 1: `runs/kda-cuda-development/attempt-00062-reuse-level1`, manifest `a2bca532d208ef393ed1977192d63067cd07c4e2077add9839774e4d05ba9e21`.
- Append-only attempt/reference index SHA-256: `5d69648aa85eaf520a100a827e421c3b27ad1a7765de563d1d7dd1be3ff4a8fc`.

**Result**

- Pushed commit `b9b35c202d12dcad6ee836a214c840b74791f24b`
  passed protected ownership/runtime gates. Output, all seven gradients, and
  the independent repeat are bitwise identical to attempt 53.
- Level 1 rejected the intervention: T=4096 forward+backward changed only
  `21.696 -> 21.688 ms` (+0.04%). The T=1024 row improved 3.30%, but the
  production T=4096 gate did not. Peak allocation was unchanged. Exact P/Q
  reconstruction and buffer-copy traffic consume the redundant-BMM saving.
- No sanitizers, Level 2, profile, confirmation, or retest ran. Attempt 53
  remains the accepted development baseline; this is not quality or
  statistically confirmed evidence.

**Next**

- Preserve attempt 62 unchanged. Do not reuse U/W through global P/Q in this
  form. Return to attempt 53 and target the dominant pair-pack/accumulate path
  or a group-state formulation that removes work without replacement traffic.

## 2026-08-09 [agent] reject warp-parallel normalization reductions

**Context**

- Attempt 63 replaces lane-zero serial 128-channel normalization sums with
  deterministic four-warp reductions in forward preprocessing, backward
  preprocessing, and backward normalization finalization. This changes FP32
  association but leaves equations, configs, ownership, and storage unchanged.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_063 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seeded B=2/H=3/T=4096 saved-gradient comparison plus repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_053 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_063 \
  runs/kda-cuda-development/attempt-00063-warpnorm-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_063 \
  push -u origin kda-cuda/wy-warp-norm-reduce-063
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00063-protected-checker`, manifest `e614edf5a0e4c1e51bc2b6b1878e9ffa6b240ddd76bd1c0a195f78eaddac4044`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00063-warpnorm-gradient`, manifest `f26d4023b82cee857989c358a9d1529c0734fae06099284db79d29c1709628d3`.
- Level 1: `runs/kda-cuda-development/attempt-00063-warpnorm-level1`, manifest `3acfdf2b744bdc86e5cf61ee8e9a2f3aeec138b2fb435577f7b22f3d0e158088`.
- Append-only attempt/reference index SHA-256: `815712e577b29690d8c5e95931cc56852746fc760fac005c86efe17f6068492e`.

**Result**

- Pushed commit `c753cba267eaf6b4bc0bc39374428eb24df4a995`
  passed protected ownership/runtime gates and the production frozen-tolerance
  gate. Maximum output difference was `1.220703125e-4`; the largest gradient
  difference was `7.276e-12`; the repeat was bitwise deterministic.
- Level 1 rejected the intervention: T=4096 forward improved 1.61%, but
  forward+backward changed only `21.658 -> 21.564 ms` (+0.43%). T=256
  forward+backward regressed 6.67%, exceeding the 5% guard. Peak allocation
  was unchanged.
- No sanitizers, Level 2, profile, confirmation, or retest ran. Attempt 53
  remains the accepted development baseline; this is not quality or
  statistically confirmed evidence.

**Next**

- Preserve attempt 63 unchanged. Keep serial normalization reductions in the
  accepted baseline. Continue from attempt 53 with a larger structural change
  to pair-pack/accumulate or recurrent group-state work rather than reduction
  scheduling alone.

## 2026-08-09 [agent] preserve near-threshold WMMA boundary scan

**Context**

- Attempt 64 replaces only backward group-boundary reconstruction with an
  eight-chunk persistent WMMA scan. Twenty-four CTAs retain FP32 state and
  accumulators while explicitly casting WMMA operands to BF16. The complete
  reverse WY/UT VJP remains unchanged.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_064 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seeded B=2/H=3/T=4096 saved-gradient comparison plus repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_053 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_064 \
  runs/kda-cuda-development/attempt-00064-wmma-boundary-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_064 \
  push -u origin kda-cuda/wy-wmma-boundary-scan-064
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00064-protected-checker`, manifest `5c55811313fa67cf5c13f6aa8f4433263ac4b3c8e530525f50358227c1b2cbe6`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00064-wmma-boundary-gradient`, manifest `f91a7409f6c9d3dd5b59a0413c006d09b044a0b5004a35e1beab45cc73fb3845`.
- Level 1: `runs/kda-cuda-development/attempt-00064-wmma-boundary-level1`, manifest `30b33d456c252472eba06235679b54f9350979d562186677fa8e2ff7563af888`.
- Append-only attempt/reference index SHA-256: `c9dbd9c357ba91f8ef2ad71e450a58077eb2317cde9acf03a34bb006d650e74d`.

**Result**

- Pushed commit `0fc6bf77c57ce59d9eb995c8bc4ebcc2fee6d4df`
  passed protected ownership/runtime gates and the production frozen-tolerance
  gate. Output, `dv`, and `dbeta` are bitwise equal to attempt 53; the largest
  changed gradient difference is `1.819e-12`; repeat is bitwise deterministic.
- Level 1 classified the candidate `do_not_advance`: T=4096
  forward+backward improved `21.550 -> 20.967 ms` (2.70%), below the frozen 3%
  gate. Peak allocation was unchanged and all important regressions remained
  within 5%. No sanitizer, Level 2, profile, confirmation, or retest ran.
- Attempt 53 remains accepted. This is neither quality nor statistically
  confirmed evidence.

**Next**

- Attempt 64 supplies new independent strategy evidence: attempt 56's forward
  WMMA scan improved 2.60%, while attempt 64's backward boundary scan improves
  2.70%. Test both validated substitutions together from exact attempt 53 as a
  unified scan-backend boundary; do not add unrelated changes.

## 2026-08-09 [agent] accept unified forward and backward WMMA scans

**Context**

- Attempt 65 combines the independently validated attempt-56 forward scan and
  attempt-64 backward group-boundary scan. Both persistent kernels retain FP32
  recurrent state and accumulators while explicitly casting WMMA operands to
  BF16. The complete reverse WY/UT VJP and attempt-53 chunk-partial
  finalization remain unchanged.
- The candidate parent is attempt 64, but every matched performance and
  correctness comparison uses accepted attempt 53. Only
  `nanochat/mixers/cuda_kda/chunk_wy_forward.cu` and
  `nanochat/mixers/cuda_kda/chunk_wy_backward.cu` differ from that baseline.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_065 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seeded B=2/H=3/T=4096 saved-gradient comparison plus repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_053 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_065 \
  runs/kda-cuda-development/attempt-00065-unified-wmma-level1
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_validation_065 \
  --lane optimization <isolated artifact/cache arguments> --sanitizers
# Executed the saved baseline-first Level-2 plan exactly once.
PYTHONPATH=/home/veer/Master/projects/experiment_swa_kda_cuda_attempt_065 \
TORCH_EXTENSIONS_DIR=/tmp/kda065-unified-wmma-profile-ext-002 \
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=<artifact>/trace \
  /home/veer/Master/projects/experiment_swa_kda/.venv/bin/python \
  /tmp/kda033_nsys.py
nsys stats --report cuda_gpu_kern_sum --format csv \
  --output <artifact>/kern <artifact>/trace.nsys-rep
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_065 \
  push -u origin kda-cuda/wy-unified-wmma-scan-065
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00065-protected-checker`, manifest `267663b051697e30fe6b21c14b43ab69a302c6ee57b95a6b23425f01244870b7`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00065-unified-wmma-gradient`, manifest `e06e7092bea3da9c66b1d3fb38db510b5ef8ffa4b42bc8cf80998fa9efa66b09`.
- Full sanitizer validation: `runs/kda-cuda-development/validations/validation-00013-unified-wmma-scan`, manifest `84865ec0ad9455649d1022f9a3a61c353765874685d84925ed2c0361781e28d4`.
- Level 1: `runs/kda-cuda-development/attempt-00065-unified-wmma-level1`, manifest `745d3fae0c4cf32f5a74a95b0b0402522a65f713e98320e404c6c7faa7918d50`.
- Level 2: `runs/kda-cuda-development/attempt-00065-unified-wmma-level2`, manifest `262b79b79eadb742789c099322895aa3a343ad512d0c9367b1cde1e997c9b5cd`.
- Production profile: `runs/kda-cuda-development/diagnostics/attempt-00065-production-profile`, manifest `5f390377ce61e65ba8a4d8c5c16e00201d861fbbe10854f18f7af364a34b1ac0`.
- Invalid coordinator-CWD profile invocation: `runs/kda-cuda-development/diagnostics/attempt-00065-production-profile-invalid-001`, manifest `8cd5abcdcb7ce0232b27ab61284edd3e0f1a05b8c164fb7c0e90f0455fc8756c`. The empty trace was overwritten before a failed relative-path move was noticed; exact captured stderr and that preservation limitation are recorded.
- Baseline manifest: `runs/kda-cuda-development/baseline/dbfe809a82.json`, SHA-256 `1fad6301b3a002eab582f244ea0a6bac59bb464ca2579b95b14bef07c33f503a`.
- Append-only attempt/reference index SHA-256: `748b145b198f427af39e7362e0bf3284c2cd01606344afffc172a11d3644c3b6`.

**Result**

- Pushed commit `dbfe809a822084c849b344a6d479c90f48e1474d`
  passed ownership 1.0, protected runtime/profile audit, runtime FLA freedom,
  frozen production correctness, a bitwise deterministic repeat, and all four
  sanitizers. Maximum output difference versus attempt 53 was
  `4.8828125e-4`; the largest gradient difference was `5.821e-11`.
- Level 1 advanced: T=4096 forward+backward improved
  `21.596 -> 20.249 ms` (6.23%) with memory ratio 1.0. Every important
  regression guard passed.
- The single baseline-first Level-2 pair measured baseline
  `[27575,27546,27684,27462,27654]`, median `27575 tok/s`, and candidate
  `[28344,28307,28325,27996,28358]`, median `28325 tok/s`: +2.72%, identical
  `5508.533 MiB`, and 64.85% of the fixed 43,680 tok/s reference.
- The production profile identifies the forward WMMA scan as the largest named
  owned kernel at `2.725 ms` per iteration. Backward pair packing plus
  accumulation totals `1.786 ms`; the backward group-boundary WMMA scan is
  `0.834 ms`.
- Attempt 65 is the accepted development baseline. It is not statistically
  confirmed, official retention, a merge/default change, or LM-quality
  evidence.

**Next**

- Continue from exact attempt 65. First target the persistent forward WMMA
  scan's intra-kernel work or state/output staging, while preserving explicit
  BF16 operand rounding and FP32 accumulation. Treat pair-pack/accumulate as
  the next backward target; do not replay rejected pair-width or global
  transform-reuse variants.

## 2026-08-09 [agent] reject separate WMMA decay precompute

**Context**

- Attempt 66 starts from accepted attempt 65 and computes the value-independent
  `q*exp(G)` and `k*exp(G_end-G)` WMMA operands once into dead P/Q storage
  after U/W are formed. This removes four-CTA redundant exponentials without
  adding an allocation or changing the explicit BF16 WMMA casts.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_validation_066 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seeded B=2/H=3/T=4096 saved-gradient comparison plus repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_065 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_066 \
  runs/kda-cuda-development/attempt-00066-precompute-decay-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_066 \
  push -u origin kda-cuda/wy-precompute-wmma-decay-066
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00066-protected-checker`, manifest `46d209cfcc89bc4030a8daf444812dc02d1c901dd505b0e10abd2a298e636f3d`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00066-precompute-decay-gradient`, manifest `ed90089e550d9d91e76f533307ce2a5fbb9c7031cedd10cb5d43e62836d5eae1`.
- Preserved invalid candidate-local uv invocation: `runs/kda-cuda-development/diagnostics/attempt-00066-gradient-invalid-001`, manifest `553855be3e164f244a9ef66377fdef3ddc6b3b3bc5c6e3110ac7df79be0d56cd`. It failed before importing Torch or starting GPU work; the ignored empty candidate `.venv` remains preserved.
- Level 1: `runs/kda-cuda-development/attempt-00066-precompute-decay-level1`, manifest `bf68dc3e04626e60435697fa50f624f9f377540d14a78063a7595fa8a7e60f2c`.
- Append-only attempt/reference index SHA-256: `a3fa840a229e8c7ac497a87bb86498a882d24a9ce0ac8bcd8b2706193d4106b1`.

**Result**

- Pushed commit `b420cb1291e7010b6079d80453d1e4e747f2cd31`
  passed ownership 1.0, protected runtime/profile audit, and runtime FLA
  freedom. Production output, all seven gradients, and the independent repeat
  are bitwise identical to attempt 65; peak allocation is unchanged.
- Level 1 rejected the intervention: T=4096 forward+backward changed
  `20.639 -> 20.683 ms` (-0.21%), forward-only regressed 0.25%, and T=256
  forward+backward regressed 6.86%, exceeding the 5% guard. The separate
  whole-history pass adds enough traffic to erase the redundant-exponential
  saving.
- No sanitizers, Level 2, profile, confirmation, or retest ran. Attempt 65
  remains accepted. This is neither quality nor statistically confirmed
  evidence.

**Next**

- Preserve attempt 66 unchanged. Do not materialize whole-history decay
  operands. Return to attempt 65 and target work entirely inside the persistent
  scan, or shift to the measured backward pair-pack/accumulate path.

## 2026-08-09 [agent] preserve pair-pack parallelism without advancing

**Context**

- Attempt 67 starts from accepted attempt 65 and flattens each 16x128 stable
  pair operand across a 256-thread CTA instead of making 128 channel lanes loop
  over all 16 rows. Every output element retains a single writer; equations,
  pair-batch width, reductions, storage, and allocation are unchanged.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_validation_067 \
  --lane optimization <isolated artifact/cache arguments>
# Exact production-shape saved-gradient comparison plus repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_065 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_067 \
  runs/kda-cuda-development/attempt-00067-pair-pack-level1
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_validation_067 \
  --lane optimization <isolated artifact/cache arguments> --sanitizers
# Executed one valid saved baseline-first Level-2 plan after preserving an
# artifact-invalid baseline invocation.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_067 \
  push -u origin kda-cuda/wy-pair-pack-256-067
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00067-protected-checker`, manifest `0011f899c1de10609465e3c60050f7b0a891b8796e4ecbe4650ff8a666a90f94`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00067-pair-pack-gradient`, manifest `98d86cd20ffb80301ecb5c33f7a1379fdfc2911b7051f59958f4ec43b6a00de7`.
- Full sanitizer validation: `runs/kda-cuda-development/validations/validation-00014-pair-pack-256`, manifest `f7f62b84582ea6c9e8512bf27f8ed76277df70ee5a742e287717b2762c6941d9`.
- Level 1: `runs/kda-cuda-development/attempt-00067-pair-pack-level1`, manifest `081d5f764c21538292f49e3fc9f7cb79194ef98a41fc120b7f6456251e1241d2`.
- Level 2: `runs/kda-cuda-development/attempt-00067-pair-pack-level2`, manifest `dbc2d39c4c71bc5da296b135e754b6a4ae3d7866493bffc0489b53773d7ba1fa`.
- Invalid gradient-save invocation: `runs/kda-cuda-development/diagnostics/attempt-00067-gradient-save-invalid-001`, manifest `dc13a482315e57e052f41b0be80b37ccf8d4c0938e89ea280cfc0a203d7906df`.
- Invalid missing-tee Level-2 baseline: `runs/kda-cuda-development/diagnostics/attempt-00067-level2-invalid-001`, manifest `7434de87e45a325e8460e2dd676d73bc69645ef55e1c75f09b333100a4747d44`.
- Append-only attempt/reference index SHA-256: `300fe34a987553b6cbe5030da9f52ede32a1dac85962bddbcf9e73decab9afe5`.

**Result**

- Pushed commit `edcf90c12c8f4ea19e08f85009e51ae7a3c65add`
  passed ownership 1.0, runtime/profile audit, runtime FLA freedom, bitwise
  production correctness and repeat, and all four sanitizers. Peak allocation
  is unchanged.
- Level 1 advanced: T=4096 forward+backward improved
  `21.047 -> 20.077 ms` (4.61%); all important regression guards passed.
- The valid Level-2 pair measured baseline
  `[28741,28656,28565,28339,28448]`, median `28565 tok/s`, and candidate
  `[28724,28675,28556,28378,28663]`, median `28663 tok/s`: +0.34%, identical
  `5508.533 MiB`, and 65.62% of the fixed 43,680 tok/s reference. This small
  matched gain is not a defensible new baseline.
- The first Level-2 baseline completed but its raw-log tee target did not exist;
  it is invalid and excluded. A fresh complete baseline-first pair supplied the
  reported evidence. Attempt 65 remains accepted. No confirmation, profile, or
  retest ran; this is not quality or statistically confirmed evidence.

**Next**

- Preserve attempt 67 unchanged. Continue from attempt 65 with a larger
  structural reduction in backward pair transforms/BMMs or forward persistent
  scan work; thread-count scheduling alone does not materially move trainer
  throughput.

## 2026-08-09 [agent] accept eight-warp forward WMMA scheduling

**Context**

- Attempt 68 starts from accepted attempt 65 and activates all eight resident
  warps for the eight `(16-row tile, 16-value half)` products in the forward Z
  and output phases. It does not change the 32-value CTA geometry, the
  eight-warp state update, equations, allocation, FP32 accumulators, or explicit
  BF16 operand casts.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --config configs/research/kda_cuda_ownership.toml \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_068 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seeded B=2/H=3/T=4096 saved-gradient comparison plus repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_065 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_068 \
  runs/kda-cuda-development/attempt-00068-eight-warp-level1
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_validation_068 \
  --lane optimization <isolated artifact/cache arguments> --sanitizers
# Executed the saved baseline-first Level-2 plan exactly once, then one
# production-shape Nsight Systems profile.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_068 \
  push -u origin kda-cuda/wy-eight-warp-forward-068
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00068-protected-checker`, manifest `63fc0d948e5bacaa5a8f4636e3a7b2f0a12c99777c72c8c44a59f1cda0c5120b`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00068-eight-warp-gradient`, manifest `c56ac0fab5497eaa56716b57720a5cc875760695fc1b727ce7b45d8d04adc57e`.
- Full sanitizer validation: `runs/kda-cuda-development/validations/validation-00015-eight-warp-forward`, manifest `e8a409d0bb6c679272c84e221eb85307947d68292c19b3a6341da35628441d8c`.
- Level 1: `runs/kda-cuda-development/attempt-00068-eight-warp-level1`, manifest `1e98cf7e15152c8e0a2d14d1b431a40b30c1b71121fe40ed37bed76ce77e35fa`.
- Level 2: `runs/kda-cuda-development/attempt-00068-eight-warp-level2`, manifest `a7d68f59592a7754c331b4fbd6de293239730051569a99954b0c02e43249ed49`.
- Production profile: `runs/kda-cuda-development/diagnostics/attempt-00068-production-profile`, manifest `47055719564743de998ccf11fc1cca338bd6c43e63c4b3b8817817723f41c223`.
- Baseline manifest: `runs/kda-cuda-development/baseline/705c607440.json`, SHA-256 `049a65ae602f18e5298e08f7f0ca3f8382bba632fa867c0ebf04ad7c47ddddb2`.
- Append-only attempt/reference index SHA-256: `203b3ed1321bc5c2d34043772b637a7920bd1e2268a702fd00dcbfd71b39e409`.

**Result**

- Pushed commit `705c60744015ae054cf116f64e8bd7a1c4fd844b`
  passed ownership 1.0, protected runtime/profile audit, runtime FLA freedom,
  bitwise production correctness and repeat, and all four sanitizers. Peak
  allocation is unchanged.
- Level 1 advanced: T=4096 forward improved 2.47% and forward+backward improved
  `20.421 -> 19.666 ms` (3.70%). Every important regression guard passed.
- The single baseline-first Level-2 pair measured baseline
  `[28373,28494,28376,28157,28299]`, median `28373 tok/s`, and candidate
  `[28784,28788,28772,28481,28839]`, median `28784 tok/s`: +1.45%, identical
  `5508.533 MiB`, and 65.90% of the fixed 43,680 tok/s reference.
- Profiling confirms the targeted forward WMMA kernel fell from `2.725 ms` in
  attempt 65 to `2.167 ms` per iteration, a 20.46% reduction. Backward pair
  pack plus accumulation now totals `1.654 ms` per iteration.
- Attempt 68 is the accepted development baseline. It is not statistically
  confirmed, official retention, a merge/default change, or LM-quality
  evidence.

**Next**

- Continue from exact attempt 68. The next large owned paths are the remaining
  forward WMMA scan, backward pair pack/accumulate, and generic FP32 BMM groups.
  Prefer an algebraic or fusion change over further thread-count tuning.

## 2026-08-09 [agent] preserve persistent group recomputation behind short guard

**Context**

- Attempt 69 starts from accepted attempt 68 and makes the existing persistent
  backward group-boundary WMMA scan emit the already allocated per-chunk H and
  Z buffers during reverse recomputation. It replaces eight BMM/copy/decay/add
  sequences per group, removes two local temporary buffers, and does not store
  full-history state.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --config configs/research/kda_cuda_ownership.toml \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_069 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seeded production comparison plus deterministic repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_068 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_069 \
  runs/kda-cuda-development/attempt-00069-group-recompute-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_069 \
  push -u origin kda-cuda/wy-persistent-group-recompute-069
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00069-protected-checker`, manifest `708791cde7c56f1aee6fe3d0f5d0a7980b35eb16dfb7bded7851134379a00894`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00069-group-recompute-gradient`, manifest `fbe2675eb3dd7f9d17bae6c77f0eb00bafda7ba9f7152e3e14293a9828837f2b`.
- Level 1: `runs/kda-cuda-development/attempt-00069-group-recompute-level1`, manifest `65fd0c842e00cda23fd51bb2e78972b39a62c1abe2de3d38ddb71f628c7078ac`.
- Append-only attempt/reference index SHA-256: `13d261c5af41a908d8c19d53f3042aa9fc02bdd2b5e016b08668c31277af9f27`.

**Result**

- Pushed commit `ff37d9ea6d9b9cfd26c23a62b76e76686b5fbf19`
  passed ownership 1.0, protected runtime/profile audit, runtime FLA freedom,
  frozen production correctness, and a bitwise repeat. Output, `dv`, and
  `dbeta` are bitwise exact; the largest changed gradient delta is `1.455e-11`.
- The production objective improved strongly: T=4096 forward+backward fell
  `19.674 -> 18.516 ms` (5.89%), and peak allocation fell from 204,540,416 to
  203,950,592 bytes (ratio 0.99712).
- Level 1 nevertheless classified `do_not_advance`: T=256 forward+backward
  regressed 8.27%, exceeding the frozen 5% guard. The changed C64 specialization
  is dispatched only at T=4096, so the short regression is extrinsic timing
  noise, but it is not overridden or retested.
- No sanitizers, Level 2, profile, confirmation, or retest ran. Attempt 68
  remains accepted. This is neither quality nor statistically confirmed
  evidence.

**Next**

- Preserve attempt 69 unchanged as strong strategy evidence. Refine its
  persistent group scan with an additional mechanistic improvement in a new
  child, compare to accepted attempt 68, and continue to honor all Level-1
  guards rather than treating the short-path result as a win.

## 2026-08-09 [agent] accept eight-warp persistent group recomputation

**Context**

- Attempt 70 is a child of preserved attempt 69, while every matched
  correctness and performance comparison uses accepted attempt 68. It maps the
  persistent backward group scan's eight `(16-row, 16-value)` Z products to all
  eight resident warps instead of making four warps serialize both value
  halves. Equations, rounding, geometry, state/history allocation, and the
  chunk-boundary recomputation strategy are unchanged.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --config configs/research/kda_cuda_ownership.toml \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_070 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seeded B=2/H=3/T=4096 saved-gradient comparison against the existing
# attempt-68 tensor artifact, plus one independent deterministic repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_068 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_070 \
  runs/kda-cuda-development/attempt-00070-eight-warp-group-level1
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_validation_070 \
  --lane optimization <isolated artifact/cache arguments> --sanitizers
# Executed one artifact-complete saved baseline-first Level-2 pair after
# preserving a baseline whose raw-log tee target was missing.
PYTHONPATH=/home/veer/Master/projects/experiment_swa_kda_cuda_attempt_070 \
TORCH_EXTENSIONS_DIR=/tmp/kda070-production-profile-ext-001 \
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=<artifact>/trace \
  /home/veer/Master/projects/experiment_swa_kda/.venv/bin/python \
  /tmp/kda033_nsys.py
nsys stats --report cuda_gpu_kern_sum --format csv \
  --output <artifact>/kern <artifact>/trace.nsys-rep
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_070 \
  push -u origin kda-cuda/wy-eight-warp-group-recompute-070
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00070-protected-checker`, manifest `88f4392c5d0486612906265ef4ce232105241f02e4dc8d7b798e490d5ca57fd2`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00070-eight-warp-group-gradient`, manifest `1ca5b18d2df2b1e4cce82eab54282da2bf71fa0169005093bce874ad621de995`.
- Full sanitizer validation: `runs/kda-cuda-development/validations/validation-00016-eight-warp-group-recompute`, manifest `7dd61dfbc536f6bf3f557a6fba5c9e95d890fcb50143202cb31595b6cc2a1c8c`.
- Level 1: `runs/kda-cuda-development/attempt-00070-eight-warp-group-level1`, manifest `e2e9a8ee229217be0c24d033bf457a3d0f9f850b633d3067f3dce2532dcf872a`.
- Level 2: `runs/kda-cuda-development/attempt-00070-eight-warp-group-level2`, manifest `6d4b4c556d507305f91366431673dc2e47b02cf991dde5451f55fdf49092478b`.
- Production profile: `runs/kda-cuda-development/diagnostics/attempt-00070-production-profile`, manifest `e143779f072c9f0af5ea8d977eff515b3e497bfd94c2283d9cb14bea66584d35`.
- Invalid gradient invocations: `runs/kda-cuda-development/diagnostics/attempt-00070-gradient-invalid-001`, manifest `01c3efc62faa95b6a6190f7ede31721b757fe590ead028827a4df9a8cadfb5f2`, and `attempt-00070-gradient-invalid-002`, manifest `35083d9b699fcd38352880388b934176b3df44097adf3d3f22ffdac924431c68`. Both failed before GPU work because module/source roots resolved from the coordinator; the second raw directory was overwritten by the corrected absolute-path launch, so its exact captured traceback and that limitation are recorded explicitly.
- Invalid missing-tee Level-2 baseline: `runs/kda-cuda-development/diagnostics/attempt-00070-level2-invalid-001`, manifest `87918ceaac0a9266e1955d688a5fce184c50fccdb8522f71340a6d24f44231df`. The completed baseline samples are preserved and excluded; no candidate ran in that namespace.
- Baseline manifest: `runs/kda-cuda-development/baseline/c6532b46e4.json`, SHA-256 `431bdc8d2a460778c2ae456759aab9b49e7b968ed78182b0e8b23e9f932f5d5b`.
- Append-only attempt/reference index SHA-256: `ceece07dc7132df879085ac9832df8e6e0b9fea65d111173aad3978e991bf41d`.

**Result**

- Pushed commit `c6532b46e4ad2f4ad81072dc0016e94eeb4f7f84`
  passed ownership 1.0, protected runtime/profile audit, runtime FLA freedom,
  frozen production correctness, a bitwise deterministic repeat, and all four
  sanitizers. Output, `dv`, and `dbeta` are bitwise exact against attempt 68;
  the largest changed-gradient delta is `1.455e-11`.
- Level 1 advanced with every guard passing: T=4096 forward+backward improved
  `19.726 -> 18.286 ms` (7.30%), peak allocation ratio was `0.99712`, and the
  formerly noisy T=256 forward+backward row improved 3.82%.
- The valid Level-2 pair measured baseline
  `[28641,28600,28754,28378,28687]`, median `28641 tok/s`, and candidate
  `[29727,29699,29715,29315,29656]`, median `29699 tok/s`: +3.69%, identical
  `5508.533 MiB`, and 67.99% of the fixed 43,680 tok/s reference.
- The fresh profile places the forward WMMA scan at `2.183 ms/iteration`, the
  expanded persistent group scan at `1.695 ms`, pair pack at `0.915 ms`, pair
  transforms at about `0.751-0.757 ms` each, and pair accumulation at
  `0.658 ms`. The persistent group kernel is larger because it now emits the
  reverse-recompute H/Z histories, but it removes a greater amount of generic
  BMM/copy/decay/add work.
- Attempt 70 is the accepted development baseline. Official retention remains
  `4d1a3b231da2c99882324efbda5306a1815e21c7`; no confirmation or quality
  evaluation ran, and this is not statistically confirmed evidence.

**Next**

- Continue from exact attempt 70. Prioritize eliminating duplicated work or
  traffic inside the persistent group scan, then the remaining forward WMMA
  scan, pair transforms/pack-accumulate, and generic FP32 BMM groups. Preserve
  the no-full-history boundary-recomputation design and all frozen gates.

## 2026-08-09 [agent] preserve sixteen-warp group state compile failure

**Context**

- Attempt 71 starts from accepted attempt 70 and assigns each of the 16
  `(16-key, 16-value)` `E^T Z` products in the persistent group scan to its own
  warp. It raises the CTA from 256 to 512 threads and enlarges only the
  per-warp BF16 operand staging; equations and persistent state are unchanged.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --config configs/research/kda_cuda_ownership.toml \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_071 \
  --lane optimization <isolated artifact/cache arguments>
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_071 \
  push -u origin kda-cuda/wy-sixteen-warp-group-state-071
```

**Artifacts**

- Invalid protected checker: `runs/kda-cuda-development/diagnostics/attempt-00071-sixteen-warp-group-check-invalid`, manifest `bf8e262f4ba201c77f7711dac4e479c95cd0c5356b9f6f1395a073c7de7d9b18`.
- Append-only attempt/reference index SHA-256: `85aa6505c5abd7275694d19ea0f82dcab7219a76e887b1311980445c7fdc7935`.

**Result**

- Pushed commit `4cd6f3aad2f72e4df25309c871cefc110fdc77cd`
  is invalid: `ptxas` reports that the persistent group kernel uses `0xe000`
  bytes of static shared memory, exceeding the `0xc000` limit. Runtime audit
  failed at build, profile audit was skipped, and no GPU measurement ran.
- Attempt 70 remains accepted. No correctness, sanitizer, Level-1, Level-2,
  confirmation, or quality claim is attached to attempt 71.

**Next**

- Preserve attempt 71 unchanged. Test any alternative value-tile geometry in a
  new child of attempt 70; do not exceed the static shared-memory ceiling.

## 2026-08-09 [agent] accept 16-column persistent group state tiling

**Context**

- Attempt 72 starts from accepted attempt 70 and changes the persistent
  backward group scan from one 32-value-column CTA to two 16-value-column CTAs.
  Eight warps now compute all eight `E^T Z` products concurrently without the
  shared-memory overflow of attempt 71. Equations, FP32 state/accumulators,
  BF16 operand casts, histories, and chunk-boundary recomputation are unchanged.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_072 \
  --lane optimization <isolated artifact/cache arguments>
# Exact production-shape saved-gradient comparison plus repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_070 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_072 \
  runs/kda-cuda-development/attempt-00072-value-tile16-level1
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_validation_072 \
  --lane optimization <isolated artifact/cache arguments> --sanitizers
# Executed the saved baseline-first Level-2 plan exactly once, then one
# production-shape Nsight Systems profile.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_072 \
  push -u origin kda-cuda/wy-value-tile16-group-state-072
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00072-value-tile16-protected-checker`, manifest `ed06e62b6ea2580704fb55489a7f6e2da8f91747e442df839181fdd1d6ad61ad`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00072-value-tile16-gradient`, manifest `c6d690ec808526ac5cea971c36ed966a8a54939a4f28ea80bb38ed6f8c96c7f1`.
- Full sanitizer validation: `runs/kda-cuda-development/validations/validation-00017-value-tile16-group-state`, manifest `eb0efab08c268b3ab3c2694bb885fa3ed08f31841055a69acb6dcbb3d9b1d030`.
- Level 1: `runs/kda-cuda-development/attempt-00072-value-tile16-level1`, manifest `66cd13b2e6c9fb83a7ff32efa77547153a9cf34eb8e49917f1fdbb25f800ae63`.
- Level 2: `runs/kda-cuda-development/attempt-00072-value-tile16-level2`, manifest `e8ad96391ef7b55d29b8295333ac023f47fd2da4463c37390a39b026e754aeec`.
- Production profile: `runs/kda-cuda-development/diagnostics/attempt-00072-production-profile`, manifest `9af73270c6e9070e3bc952a00d09f11f212c65a85477b9b903261bcf1c1bc71c`.
- Baseline manifest: `runs/kda-cuda-development/baseline/66e432607e.json`, SHA-256 `f79cb593a52e305af3b3282682396a90b0c2d36f3b80e62165f19ff89b333311`.
- Append-only attempt/reference index SHA-256: `055cc1b0f90241779356ee577c42d8cd4ecb998fb213702ec33836a448c48af9`.

**Result**

- Pushed commit `66e432607ed619ccbac36900e1d5c648e213beb2`
  passed ownership 1.0, runtime/profile audit, runtime FLA freedom, bitwise
  output and all-gradient equality against attempt 70, a bitwise repeat, and
  all four sanitizers.
- Level 1 advanced with every guard passing: T=4096 forward+backward improved
  `18.838 -> 18.068 ms` (4.09%) with unchanged peak allocation.
- The single baseline-first Level-2 pair measured baseline
  `[29619,29477,29496,29436,29484]`, median `29484 tok/s`, and candidate
  `[29896,29748,29868,29412,29891]`, median `29868 tok/s`: +1.30%, identical
  `5508.533 MiB`, and 68.38% of the fixed 43,680 tok/s reference.
- Profiling confirms the targeted persistent group kernel fell from
  `1.695 -> 1.366 ms/iteration`, a 19.39% reduction. Attempt 72 is the accepted
  development baseline. Official retention remains `4d1a3b231da2c99882324efbda5306a1815e21c7`;
  no confirmation or quality evaluation ran, and the result is not
  statistically confirmed.

**Next**

- Continue from exact attempt 72. The largest named path remains the forward
  WMMA scan (`2.174 ms/iteration`), followed by pair pack/transforms and the
  remaining generic FP32 BMM groups. Preserve 16-column group tiling and the
  no-full-history reverse-recomputation design.

## 2026-08-09 [agent] reject 16-column forward WMMA tiling

**Context**

- Attempt 73 starts from accepted attempt 72 and applies 16-value-column CTA
  tiling to the persistent forward WMMA scan. It removes value-half
  serialization from `Hnext`, while doubling CTA count and using four of eight
  warps during the Z and output phases. Equations, casts, accumulators, and
  allocation are unchanged.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_073 \
  --lane optimization <isolated artifact/cache arguments>
# Exact production comparison/repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_072 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_073 \
  runs/kda-cuda-development/attempt-00073-forward-tile16-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_073 \
  push -u origin kda-cuda/wy-forward-value-tile16-073
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00073-forward-tile16-protected-checker`, manifest `a879902b025789e8ecbd40625f8725103531b8a42098e996641994149cd7c6ef`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00073-forward-tile16-gradient`, manifest `69001bd3be68599c1d38f26e4fc1d15c867a63e77b48ce03dac9147d6702bcf2`.
- Level 1: `runs/kda-cuda-development/attempt-00073-forward-tile16-level1`, manifest `f12f728187900645c192a3b72d56a9b0d0422656287d11f232995b9215351843`.
- Invalid no-staged-source checker invocation: `runs/kda-cuda-development/diagnostics/attempt-00073-check-invalid-001`, manifest `62d5331b5e807ff77f66317f033c5639e5f058ded8ad1b4fcb22788753980dc1`.
- Invalid missing-destination gradient invocation: `runs/kda-cuda-development/diagnostics/attempt-00073-gradient-invalid-001`, manifest `c0c16f61d607c505bc0059c711481ca83b54fbf47f0d6227da91a33665e179d7`. Computation completed, but no tensors or raw log could be saved; it is excluded.
- Append-only attempt/reference index SHA-256: `202034652979f44d42a0284de1309512cb46b9402c96652d5ded556e1b670569`.

**Result**

- Pushed commit `1d2c66a7339c3855d795f2673839cb9ce45628be`
  passed ownership 1.0, runtime/profile audit, runtime FLA freedom, and bitwise
  output/all-gradient correctness plus a bitwise repeat.
- Level 1 rejected the geometry: T=4096 forward-only improved 1.37%, but
  forward+backward regressed `18.313 -> 19.305 ms` (5.42%), exceeding the
  frozen 5% guard. Peak allocation is unchanged.
- No sanitizers, Level 2, profile, confirmation, or retest ran. Attempt 72
  remains accepted; this is neither quality nor statistically confirmed
  evidence.

**Next**

- Preserve attempt 73 unchanged. Continue from attempt 72 and target pair
  pack/transforms or generic FP32 BMM groups rather than doubling forward CTA
  count.

## 2026-08-09 [agent] reject pair-factor caching traffic

**Context**

- Attempt 74 starts from accepted attempt 72. The pair-pack kernel writes its
  already computed target/source exponential factors into `dR_group` and
  `dE_group` after their last reads; pair accumulation reloads those factors
  instead of recomputing exponentials. The buffers are group-local and dead,
  so allocation and history are unchanged.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_074 \
  --lane optimization <isolated artifact/cache arguments>
# Exact production comparison/repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_072 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_074 \
  runs/kda-cuda-development/attempt-00074-pair-factor-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_074 \
  push -u origin kda-cuda/wy-pair-factor-cache-074
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00074-pair-factor-protected-checker`, manifest `1380487dd92ff420ed0aac69f5a02a14ba4e074c576f135216d3bb994f00add2`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00074-pair-factor-gradient`, manifest `5264a67b26eaae5d3addc25724736bce24cfdd7629d315a7194216233c7e49f8`.
- Level 1: `runs/kda-cuda-development/attempt-00074-pair-factor-level1`, manifest `1bb0c0119131e7669572e75aa7937d55ef9c9f15a3876b7389f34de8f2e4435c`.
- Append-only attempt/reference index SHA-256: `d57206bd61fe859d2fa37dc7294f5a5fd4392b88b72a8741710901b8b788612d`.

**Result**

- Pushed commit `3b6ca582a008f9255393d2279a8e3ad1c74441a3`
  passed ownership 1.0, runtime/profile audit, runtime FLA freedom, bitwise
  output/all-gradient correctness, and a bitwise repeat.
- Level 1 rejected the cache: T=4096 forward+backward regressed
  `18.374 -> 18.584 ms` (1.15%), and T=256 forward+backward regressed 6.32%,
  exceeding the frozen guard. The factor writes and reloads cost more than the
  removed exponentials save; allocation is unchanged.
- No sanitizers, Level 2, profile, confirmation, or retest ran. Attempt 72
  remains accepted, without quality or statistical claims.

**Next**

- Preserve attempt 74 unchanged. Continue from attempt 72; avoid materializing
  factor caches and favor eliminating a BMM or fusing existing traffic.

## 2026-08-09 [agent] reject compact exact pre-M history

**Context**

- Attempt 75 starts from accepted attempt 72. It copies the exact unscaled
  strict-lower C64 pair matrix before in-place beta scaling into a compact
  2,016-float-per-chunk history, then reconstructs group-local `pre_m` during
  pair packing. This removes the third generic FP32 pair BMM without restoring
  full recurrent token-state history. The expected production allocation
  increase was bounded by the frozen 3% memory gate.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_075 \
  --lane optimization <isolated artifact/cache arguments>
# Exact production-shape gradient capture and fresh-cache deterministic repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_072 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_075 \
  runs/kda-cuda-development/attempt-00075-compact-pre-m-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_075 \
  push -u origin kda-cuda/wy-compact-pre-m-075
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00075-compact-pre-m-protected-checker`, manifest `4d718be11459e2e406b705f62cb78fbf802c8fc9091198fbf82c734ef1e3b56f`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00075-compact-pre-m-gradient`, manifest `4ab5aa38854e9939ca780ebf67b4d0dd89b26dbb67b7f0e8aaeab75ccc051652`.
- Level 1: `runs/kda-cuda-development/attempt-00075-compact-pre-m-level1`, manifest `4a8bf454dc1d434fa1259aeb019fbd3f8ef92e78ee41f00b5beed7d72801ba3c`.
- Append-only attempt/reference index SHA-256: `fd44d6b57cd969d348a1947c744c705a886894c271961691b4748bc6380af012`.

**Result**

- Pushed commit `2ca3135a0ea096a5a6beb8fcd44847082ea0bf11`
  passed ownership 1.0, runtime/profile audit, runtime FLA freedom, bitwise
  output/all-gradient correctness, and a bitwise fresh-cache repeat.
- Level 1 rejected the candidate: T=4096 forward+backward improved only
  `18.008 -> 17.861 ms` (0.81%), below the frozen 3% advancement threshold.
  Peak allocation rose 1.52%, within the 3% cap, and all per-length regression
  guards passed.
- No sanitizers, Level 2, profile, confirmation, or retest ran. Attempt 72
  remains accepted; this is neither quality nor statistically confirmed
  evidence.

**Next**

- Preserve attempt 75 unchanged. Continue from attempt 72 and seek a
  higher-leverage backward transition, especially fusion or replacement of
  the remaining generic FP32 BMM groups rather than storing additional pair
  history.

## 2026-08-09 [agent] reject fused group-product batch

**Context**

- Attempt 76 starts from accepted attempt 72. One owned packing kernel gathers
  group-local `P/Q` and duplicates group-local `T`, allowing `T@P` and `T@Q`
  to run as one doubled-batch FP32 BMM in both boundary construction and
  reverse replay. The complete WY/UT VJP and FP32 arithmetic are unchanged.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_076 \
  --lane optimization <isolated artifact/cache arguments>
# Exact production-shape gradient capture and fresh-cache deterministic repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_072 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_076 \
  runs/kda-cuda-development/attempt-00076-fused-group-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_076 \
  push -u origin kda-cuda/wy-fused-group-inputs-076
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00076-fused-group-protected-checker`, manifest `7b6effb8a327e5c3e859c0cc59981db1cf4d8eb6df27844a085a8d3d7885c6d5`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00076-fused-group-gradient`, manifest `99ab297b4b13f67fe870b7f321fe1c5e8ed5dab5207245c67cbfe26c7dcd04be`.
- Level 1: `runs/kda-cuda-development/attempt-00076-fused-group-level1`, manifest `994bc4a7c694d55f301c082371e1faed6ed1da811407584d623dc01f7e413c99`.
- Append-only attempt/reference index SHA-256: `d247b2a3448b9c7986f270fd26b762ff7a2d2560cd254c977a7de912dc1f82b3`.

**Result**

- Pushed commit `622cbc812e9fb7f8338c9a0007c2dc62c24e9f58`
  passed ownership 1.0, runtime/profile audit, runtime FLA freedom, bitwise
  output/all-gradient correctness, and a bitwise fresh-cache repeat.
- Level 1 rejected the candidate: T=4096 forward+backward regressed
  `18.145 -> 18.425 ms` (1.54%). Peak allocation rose 0.39%; all frozen
  per-length regression and memory guards still passed.
- No sanitizers, Level 2, profile, confirmation, or retest ran. Attempt 72
  remains accepted; this is neither quality nor statistically confirmed
  evidence.

**Next**

- Preserve attempt 76 unchanged. Continue from attempt 72. Do not pursue
  doubled-batch concatenation; target the reverse chunk scan itself or fuse
  BMM-adjacent elementwise/copy work without changing efficient BMM geometry.

## 2026-08-09 [agent] accept fused reverse chunk transfers

**Context**

- Attempt 77 starts from accepted attempt 72 and preserves every FP32 BMM
  shape and arithmetic path. One owned prepare kernel gathers the strided
  recurrence-major `dstate_base` and `dZ` slices while saving the incoming
  boundary adjoint; one finish kernel scatters updated `dZ` and advances the
  state adjoint. This replaces five generic copy launches with two owned
  transfers per reverse chunk. Chunk-boundary recomputation and the complete
  WY/UT VJP remain unchanged.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_077 \
  --lane optimization <isolated artifact/cache arguments>
# Exact production-shape gradient capture and fresh-cache deterministic repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_072 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_077 \
  runs/kda-cuda-development/attempt-00077-reverse-transfer-level1
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_validation_077 \
  --lane optimization <isolated artifact/cache arguments> --sanitizers
# Executed one saved-shape candidate-first Level-2 pair, then one bounded
# production-shape Nsight Systems profile.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_077 \
  push -u origin kda-cuda/wy-inplace-reverse-077
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00077-reverse-transfer-protected-checker`, manifest `b3ddf084f6e68026277b27cdde9d8a3467ee94e6344c4ad54d1101716909900d`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00077-reverse-transfer-gradient`, manifest `9e167c5cb423232f904b1af2404de44d32fddfef99a4408dd82232076340df9c`.
- Full sanitizer validation: `runs/kda-cuda-development/validations/validation-00018-reverse-transfer`, manifest `d4489f04821ee38b18c8584bb3193fad040c232c2c6833711f9d9f19f6c40581`.
- Level 1: `runs/kda-cuda-development/attempt-00077-reverse-transfer-level1`, manifest `532ba94bdb5af15c853ccae114729433dbc2b3871c54c6d3ac8c738f2df15041`.
- Level 2: `runs/kda-cuda-development/attempt-00077-reverse-transfer-level2`, manifest `00072fe8c49cedd4e95cb230aba0b46d038fb189c0cf763900f9222afec1584f`.
- Production profile: `runs/kda-cuda-development/diagnostics/attempt-00077-production-profile`, manifest `01dcb4ef84999db5b2872b3c4d99bd1cfe43d6de8f8de9727c3856a50c327d03`.
- Invalid pre-launch sanitizer lane invocation: `runs/kda-cuda-development/diagnostics/attempt-00077-sanitizer-invalid-lane-001`, manifest `6d1ffc25fb9e4aaaa3ee6f8f0c3b4ec8ef2ce965adc30fb24424cffda32998bf`.
- Baseline manifest: `runs/kda-cuda-development/baseline/8cfff8e89f.json`, SHA-256 `5ba063cc6e6c8a56c4f11f67712fd7a5f396bce46ace81e2d6296fa56b416ad7`.
- Append-only attempt/reference index SHA-256: `b0ab3348cc31100d1969591bc098dfcad7b50d81f8320d31254846633c5f0f34`.

**Result**

- Pushed commit `8cfff8e89f994c038d01566f2ae9b0c1626d959e`
  passed ownership 1.0, protected runtime/profile audit, runtime FLA freedom,
  bitwise output/all-gradient correctness, a bitwise fresh-cache repeat, and
  all four sanitizers with zero errors.
- Level 1 advanced: T=4096 forward+backward improved
  `18.286 -> 17.414 ms` (4.77%) with memory ratio 1.0. Every frozen
  per-length regression and memory guard passed.
- The candidate-first Level-2 pair measured candidate
  `[29921,29956,29873,29825,29996]`, median `29921 tok/s`, and baseline
  `[29656,29547,29642,29454,29581]`, median `29581 tok/s`: +1.15%, identical
  `5508.533 MiB`, and 68.50% of the fixed 43,680 tok/s reference.
- The production profile shows generic direct-copy instances falling from 656
  to 144 across two profiled iterations. The new prepare/finish transfers total
  `0.368 ms` per iteration. Remaining named targets are forward WMMA
  (`2.164 ms`), group-boundary WMMA (`1.342 ms`), pair pack (`0.957 ms`), and
  pair accumulation (`0.684 ms`).
- The first sanitizer command used a nonexistent `validation` lane and failed
  in argument parsing before GPU work. It is preserved as invalid and excluded;
  the corrected optimization-lane sanitizer run passed.
- Attempt 77 is the accepted development baseline. It is not statistically
  confirmed, official retention, a merge/default change, or LM-quality
  evidence.

**Next**

- Continue from attempt 77. Keep the efficient FP32 BMM geometries and target
  the remaining reverse-loop add/sub launches, pair pack/accumulate traffic,
  group-boundary WMMA, or forward WMMA. The campaign remains well below the
  fixed 43,680 tok/s reference and the >=45k objective.

## 2026-08-09 [agent] preserve sub-threshold reverse arithmetic fusion

**Context**

- Attempt 78 starts from accepted attempt 77. It folds `dZ += temp_vector`
  into the existing boundary-term kernel and folds `dstate -= temp_state` into
  the reverse finish-transfer kernel. This removes two launches per reverse
  chunk and one state-buffer read/write round trip without changing arithmetic
  order, equations, BMM geometry, history, or allocation.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_078 \
  --lane optimization <isolated artifact/cache arguments>
# Exact production-shape gradient capture and fresh-cache deterministic repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_077 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_078 \
  runs/kda-cuda-development/attempt-00078-reverse-arithmetic-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_078 \
  push -u origin kda-cuda/wy-fused-reverse-arithmetic-078
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00078-reverse-arithmetic-protected-checker`, manifest `60fe7c30c80dd3e1d73c4d305b6cfd7a6959938599bdc6d87b304922ee48b516`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00078-reverse-arithmetic-gradient`, manifest `536273e66c2b77fb696ae0fdb7e037c22f58e6e17d22f66215d4cd7034544028`.
- Level 1: `runs/kda-cuda-development/attempt-00078-reverse-arithmetic-level1`, manifest `19717d22d3473301d13757f0c531213f65e1f5d8dc2e47d9dcd0d719a26784f8`.
- Append-only attempt/reference index SHA-256: `9f0b98e57af70297f3a57789eb44adc965881032b864b6481967e6c16434b708`.

**Result**

- Pushed commit `1d255e75088c3bbc435949dc3df8a1c4f9fdaf2f`
  passed ownership 1.0, runtime/profile audit, runtime FLA freedom, bitwise
  output/all-gradient correctness, and a bitwise fresh-cache repeat.
- Level 1 measured a real but sub-threshold T=4096 forward+backward gain:
  `17.509 -> 17.243 ms` (1.52%), below the frozen 3% advancement threshold.
  Allocation is unchanged and every per-length regression guard passed.
- No sanitizers, Level 2, profile, confirmation, or retest ran. Attempt 77
  remains accepted; this is neither quality nor statistically confirmed
  evidence.

**Next**

- Preserve attempt 78 unchanged. Continue from attempt 77; the arithmetic
  fusion may be reconsidered only as part of a separately gated larger
  reverse-loop fusion, not replayed alone.

## 2026-08-09 [agent] reject warp-pair forward master-tile sharing

**Context**

- Attempt 79 starts from accepted attempt 77 and changes only
  `nanochat/mixers/cuda_kda/chunk_wy_forward.cu`. In the forward WMMA Z and
  output q/A phases, each adjacent warp pair forms a 64-thread cooperative
  group. The even warp loads and converts the shared master A tile once, then
  both value-half warps consume it after pair synchronization.
- The candidate preserves WMMA geometry, equations, FP32 accumulation, BF16
  casts, allocation, and output partitioning. It tests whether halving the
  duplicated W/q/A master loads and exponent work outweighs the new pair
  barriers.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_079 \
  --lane optimization <isolated artifact/cache arguments>
# Exact production-shape gradient capture and fresh-cache deterministic repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_077 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_079 \
  runs/kda-cuda-development/attempt-00079-warp-pair-master-level1
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_079 \
  push -u origin kda-cuda/wy-warp-pair-master-079
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00079-warp-pair-master-protected-checker`, manifest `c54e5b4c12b8056a2bfa33e7f612bcc10d6c2ff6f07c6b1d8760247b098622cc`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00079-warp-pair-master-gradient`, manifest `14e51f3c6c5a8282cd2c248e177d6b389d8a0b7bcfea1fc71f2a3af8f3ddab86`.
- Level 1: `runs/kda-cuda-development/attempt-00079-warp-pair-master-level1`, manifest `c0b5ffd9cfe97185e73c939a7262a36df73b0d9889bb9bf610806a303505ff6c`.
- Append-only attempt/reference index SHA-256: `0d76016f12677c1131813018935a649eac559c12b5d18117a268aa8762227b3a`.

**Result**

- Pushed commit `2ea89549dc01eb725fe6d9f24245ed376d04e1ca`
  passed ownership 1.0, protected runtime/profile audit, runtime FLA freedom,
  bitwise output/all-gradient correctness against attempt 77, and a bitwise
  fresh-cache repeat.
- Level 1 rejected the candidate. T=4096 forward+backward regressed
  `17.279 -> 18.500 ms` (7.07%), exceeding the frozen 5% important-regression
  limit. T=4096 forward also regressed 0.12%; memory ratio remained 1.0.
  Although shorter shapes improved by 0.05% to 1.48%, they do not override the
  protected long-shape failure.
- No sanitizer, Level 2, profile, confirmation, or retest ran. The evidence
  indicates the 64-thread pair barriers cost more than the eliminated duplicate
  master-tile work. Attempt 77 remains the accepted development baseline; this
  is neither quality nor statistically confirmed evidence.

**Next**

- Preserve attempt 79 unchanged and continue from attempt 77. Do not replay
  warp-pair barriers in the forward kernels. Prefer an axis that reduces
  pair-pack/accumulate traffic or group-boundary WMMA work without adding
  cross-warp synchronization.

## 2026-08-09 [agent] reject shared group-boundary decay

**Context**

- Attempt 80 starts from accepted attempt 77 and changes only
  `nanochat/mixers/cuda_kda/chunk_wy_backward.cu`. The group-boundary WMMA
  kernel computes each chunk-end decay once per key into 128 shared FP32
  values instead of calling the same `expf` independently for all 16 value
  columns. Every state update retains the same multiply/add expression.
- The intervention adds one CTA barrier per reconstructed chunk but does not
  change WMMA, pair scheduling, equations, history, or allocation.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_080 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seed-4101 B=2/H=3/T=4096 production capture against attempt 77,
# followed by an independent candidate capture with fresh compiler caches.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_077 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_080 \
  runs/kda-cuda-development/attempt-00080-boundary-decay-level1 \
  --level2-order candidate-first
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_080 \
  push -u origin kda-cuda/wy-boundary-decay-share-080
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00080-boundary-decay-protected-checker`, manifest `ba45e216610fe57aade33bd6bf3971d419c8a0691a81c70987c0b48ad6e8e081`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00080-boundary-decay-gradient`, manifest `03847a6c9f92536bb933ce72a8e645b8e3ba84ee621bfed8fac57e47e0110160`.
- Level 1: `runs/kda-cuda-development/attempt-00080-boundary-decay-level1`, manifest `cf0d026441130484d626d0ef84369290a46ed6c47d18408773b7495775023870`.
- Append-only attempt/reference index SHA-256: `ff192523ac8337cf564f5273a550de44b272f1c637e90b66d56843c0413e58cb`.

**Result**

- Pushed commit `1efc9391316e236796a59f62e8a47da2adfa4048`
  passed ownership 1.0, protected runtime/profile audit, runtime FLA freedom,
  bitwise output/all-gradient correctness against attempt 77, and a bitwise
  fresh-cache repeat.
- Level 1 rejected the candidate. T=4096 forward+backward regressed
  `17.257 -> 17.758 ms` (2.90%), while memory ratio remained 1.0. The frozen
  important-regression guard also failed because unchanged T=1024 forward
  measured `4.115 -> 4.944 ms` (+20.17%). That unrelated row is preserved as
  uncertainty and was not retested or used to rescue the backward result.
- No sanitizer, Level 2, profile, confirmation, or retest ran. Sharing the
  decay exponentials did not overcome the added per-chunk CTA barrier. Attempt
  77 remains the accepted development baseline; this is neither quality nor
  statistically confirmed evidence.

**Next**

- Preserve attempt 80 unchanged and continue from attempt 77. Avoid adding a
  CTA-wide barrier solely to share the group-boundary decay. Target a larger
  structural reduction in group-boundary WMMA or pair/BMM traffic.

## 2026-08-09 [agent] reject sixteen-chunk backward groups

**Context**

- Attempt 81 starts from accepted attempt 77 and changes only the backward
  recompute group width from eight chunks to sixteen in
  `nanochat/mixers/cuda_kda/chunk_wy_backward.cu`. This halves group-level
  pack, FP32 BMM, pair-transform, and boundary-kernel dispatch counts while
  preserving all 64 ordered chunk updates.
- The candidate intentionally tests the frozen memory gate because its bounded
  group-local operands and scratch double in size.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_081 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seed-4101 B=2/H=3/T=4096 production capture against attempt 77,
# followed by an independent candidate capture with fresh compiler caches.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_077 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_081 \
  runs/kda-cuda-development/attempt-00081-group16-level1 \
  --level2-order baseline-first
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_081 \
  push -u origin kda-cuda/wy-group16-081
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00081-group16-protected-checker`, manifest `b71f5576a932b85b9e4b0068bcccefb87eb23534df20a8ec74954944148e2fc7`.
- Invalid preflight staging invocations: `runs/kda-cuda-development/diagnostics/attempt-00081-group16-protected-checker-invalid-001`, manifest `1bf97ac7d707dd7cf181c516b60bd1952db9b5435e61012bb863862aa653a852`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00081-group16-gradient`, manifest `15a7896a9a22b9acb6deb5aafbef6085be1949f5b5fc3bbaf30795ca66e55e54`.
- Level 1: `runs/kda-cuda-development/attempt-00081-group16-level1`, manifest `47717929247b008b9396d990276f598e22f6bf755a66be14cbc1d6ba1ea174a5`.
- Append-only attempt/reference index SHA-256: `02e10f069739fad2e111c567b0523b40aac9d65e25fa0b3b9c82eedc0b037957`.

**Result**

- Pushed commit `bec34bb6364f3f330dd9879b7c5582c9dde5d175`
  passed ownership 1.0, protected runtime/profile audit, and runtime FLA
  freedom. Output and six gradients are bitwise equal to attempt 77;
  `dA_log` differs by at most `1.705e-13`, passes the frozen tolerance, and is
  bitwise deterministic across a fresh-cache repeat.
- Level 1 decisively rejected the candidate. T=4096 forward+backward regressed
  `17.353 -> 18.982 ms` (9.39%), and peak allocation rose from 203,950,592 to
  243,071,488 bytes (ratio 1.19182), far beyond the frozen 3% memory cap.
  T=256 forward+backward also regressed 8.36%.
- Two initial commands incorrectly staged from the coordinator. Both stopped
  before build/GPU work with `pathspec ... did not match any files` and
  `candidate checker requires at least one staged source change`; the second
  chain also reported that the nonexistent first artifact could not be moved.
  These failures are preserved as invalid and excluded from evidence.
- No sanitizer, Level 2, profile, confirmation, or retest ran. Attempt 77
  remains the accepted development baseline; this is neither quality nor
  statistically confirmed evidence.

**Next**

- Preserve attempt 81 unchanged and continue from attempt 77. Keep eight-chunk
  bounded scratch; reduce group-boundary or pair/BMM traffic without widening
  the live group or replaying pair-batch-width tuning.

## 2026-08-09 [agent] preserve sub-threshold in-kernel group-boundary stores

**Context**

- Attempt 82 starts from accepted attempt 77 and changes only
  `nanochat/mixers/cuda_kda/chunk_wy_backward.cu`. It makes saved group
  boundaries group-major and writes each incoming boundary from the existing
  group WMMA kernel while that kernel loads the same state into shared memory.
- This removes the explicit forward `copy_` and reverse `.contiguous()` copy
  for each of eight groups without changing equations, arithmetic, group
  width, pair/BMM geometry, or bounded scratch.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_082 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seed-4101 production capture and fresh-cache repeat against attempt 77.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_077 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_082 \
  runs/kda-cuda-development/attempt-00082-group-boundary-store-level1 \
  --level2-order candidate-first
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_082 \
  push -u origin kda-cuda/wy-group-boundary-store-082
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00082-group-boundary-store-protected-checker`, manifest `b00e7410efce5161951c96f86317747dfbe0e477155ad5bbe019cd07dcb1f227`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00082-group-boundary-store-gradient`, manifest `7354cd5b396f625ec04ddfb337584218b04bece3fade5089bdebd215996997c7`.
- Level 1: `runs/kda-cuda-development/attempt-00082-group-boundary-store-level1`, manifest `25aeb73f93b82fbee4853ac813b7cf286aa404b7a91768acb63f4d8c008c2d4a`.
- Append-only attempt/reference index SHA-256: `9c0dc889e2e93011190837ae95c1981daa21f292580ad26e0ac469809d51ca19`.

**Result**

- Pushed commit `2eeb942aab9492e91b12a79dfcfb8bd7ef2eda41`
  passed ownership 1.0, protected runtime/profile audit, runtime FLA freedom,
  bitwise output/all-gradient correctness, and a bitwise fresh-cache repeat.
- Level 1 measured only a sub-threshold T=4096 forward+backward gain:
  `17.649 -> 17.601 ms` (0.27%), below the frozen 3% advancement gate. Peak
  allocation fell from 203,950,592 to 203,557,376 bytes (ratio 0.99807), and
  every regression and memory guard passed.
- No sanitizer, Level 2, profile, confirmation, or retest ran. Attempt 77
  remains accepted; this is neither quality nor statistically confirmed
  evidence.

**Next**

- Preserve attempt 82 unchanged. Continue from attempt 77 and seek a larger
  structural removal of pair/BMM or group-boundary work; the copy fusion alone
  is too small to retain.

## 2026-08-09 [agent] accept fused pair-WMMA VJP as development baseline

**Context**

- Attempt 83 starts from accepted attempt 77 and changes only
  `nanochat/mixers/cuda_kda/chunk_wy_backward.cu`. One 256-thread CTA per
  stable tile pair packs the upstream, right, and left operands into shared
  BF16 storage and computes three VJP products with BF16 WMMA and FP32
  accumulation. The existing ordered accumulation kernel remains.
- This removes three `at::bmm_out` calls per pair batch without changing the
  recurrence, bounded eight-chunk scratch, forward path, or declared model
  configuration. The fixed external FLA reference remains 43,680 tok/s and
  the campaign target remains at least 45,000 tok/s.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_083 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seed-4101 production capture and fresh-cache repeat against attempt 77.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_077 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_083 \
  runs/kda-cuda-development/attempt-00083-pair-wmma-level1 \
  --level2-order baseline-first
# One invalid baseline-only Level-2 invocation was excluded, then one fresh,
# fully logged baseline-first matched pair was run under a new namespace.
# The candidate patch was staged in detached validation worktree 083 before:
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_validation_083 \
  --lane optimization --sanitizers <isolated artifact/cache arguments>
# One bounded two-iteration production Nsight Systems profile followed.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_083 \
  push -u origin kda-cuda/wy-pair-wmma-vjp-083
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00083-pair-wmma-protected-checker`, manifest `50dc85d44ed089bd686e0699a0da55d08fed0c0d41382378bf4049d5daffa7f8`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00083-pair-wmma-gradient`, manifest `7ec0697289cc86d7b5ecb465aad8bf10ad2ba282bc7a5e36ee3a37b926f6e77c`.
- Level 1: `runs/kda-cuda-development/attempt-00083-pair-wmma-level1`, manifest `f6a04ee8f6cd2e195a2ec43e18f4d2d93007b8407186d6c68f16a3eef1bc0c83`.
- Full sanitizer validation: `runs/kda-cuda-development/validations/validation-00019-pair-wmma`, manifest `3734607272d1a3f8b6e5a69cd1177360d4babd30dab4ac8fd0836b5492162806`.
- Invalid first Level-2 baseline capture: `runs/kda-cuda-development/diagnostics/attempt-00083-level2-invalid-missing-log-001`, manifest `fcd964ebb291109c223d2293a0632c97b46678dc4998d3b0a13bf6a33ed9e07e`.
- Valid fresh Level-2 pair: `runs/kda-cuda-development/attempt-00083-pair-wmma-level2-valid-001`, manifest `28122606f52e6f852807459bf07722220281362c2cc2d317bcfed6ac3c3ed003`.
- Production profile: `runs/kda-cuda-development/diagnostics/attempt-00083-production-profile`, manifest `91a37ff76394f2dda839b8922c5f6a0b4396265436e767023740331d1d5b7999`.
- Development-baseline manifest: `runs/kda-cuda-development/baseline/17dc8f662e.json`, SHA-256 `8522ac4b1fe477ad0ea4a251d127b4043c0b506b440a87114c12523dce853078`.
- Append-only attempt/reference index SHA-256: `578ea0811bc8762973ed6050949671a1399320de5ec830154981b40fce1f7897`.

**Result**

- Pushed commit `17dc8f662e1fea264aa7a42423ebfc5353623361`
  passed ownership 1.0, protected runtime/profile audit, runtime FLA freedom,
  and all four sanitizers with zero errors. Output and `dv` are bitwise equal
  to attempt 77. The maximum gradient delta is `5.090e-09`; every tensor
  passes the frozen tolerance and the fresh-cache repeat is bitwise
  deterministic.
- Level 1 advanced: T=4096 forward+backward improved `17.543 -> 16.523 ms`
  (5.82%) with memory ratio 1.0 and all guards true. T=4096 forward improved
  0.70%; T=1024 forward+backward improved 0.47%; T=256 forward+backward
  regressed 1.70%, within the declared limit.
- The valid fresh Level-2 baseline-first pair measured baseline
  `[30100, 30093, 30050, 29899, 29972]` tok/s, median 30,050, and candidate
  `[30656, 30608, 30647, 30390, 30637]` tok/s, median 30,637. This is a 1.95%
  matched development improvement with identical 5,508.533 MiB peak memory
  and 70.14% of the fixed external FLA reference.
- The first Level-2 baseline process completed, but relative artifact setup
  occurred from the baseline worktree, so the absolute `tee` target did not
  exist and no raw stdout log was saved. Its recovered payload and stderr are
  preserved, but the measurement is invalid and excluded.
- The two-iteration profile measured the new fused pair WMMA at 1.004 ms/iter
  and ordered pair accumulation at 0.590 ms/iter. Relative to attempt 77, each
  of three generic pair-BMM families lost 48 instances over two iterations.
- Attempt 83 is accepted only as the new development baseline. The official
  retained milestone remains `4d1a3b231da2c99882324efbda5306a1815e21c7`.
  No confirmation or LM-quality evaluation ran, and this is not statistically
  confirmed evidence.

**Next**

- Continue from attempt 83. Use the saved profile to target forward WMMA
  (2.178 ms/iter), group-boundary WMMA (1.327 ms/iter), the forward/backward
  pair transforms (0.727/0.720 ms/iter), and the combined fused-pair plus
  ordered-accumulation path (1.594 ms/iter). Preserve all gates and use sparse
  confirmation only at a major strategy boundary, plateau, checkpoint, or
  final candidate.

## 2026-08-09 [agent] accept fused stable A/M builder as development baseline

**Context**

- Attempt 84 starts from accepted attempt 83 and changes only
  `nanochat/mixers/cuda_kda/chunk_wy_forward.cu` and
  `nanochat/mixers/cuda_kda/chunk_wy_backward.cu`. In both forward construction
  and backward recomputation, one CTA packs each stable 16x16 tile pair once
  and evaluates the A and M products with explicit BF16 WMMA operands and FP32
  accumulators.
- The fused builder masks/scales directly into the final matrices. This removes
  the two generic pair BMMs, transform scratch, the finish kernel, and the P
  rebuild from each phase without changing the equations, recurrence, scan
  geometry, bounded backward state, pair-batch width, or model configuration.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --config configs/research/kda_cuda_ownership.toml \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_084 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seed-4101 B=2/H=3/T=4096 matched production capture and fresh-cache
# repeat against attempt 83.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_083 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_084 \
  runs/kda-cuda-development/attempt-00084-build-pair-wmma-level1 \
  --level2-order candidate-first
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_validation_084 \
  --lane optimization --sanitizers <isolated artifact/cache arguments>
# Executed the saved candidate-first Level-2 plan exactly once with absolute
# raw-log paths, then one bounded production-shape Nsight Systems profile.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_084 \
  push -u origin kda-cuda/wy-build-pair-wmma-084
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00084-build-pair-wmma-protected-checker`, manifest `5578c1d29d0517de668bb0bae7eabe6b517fc425710168a0cf6ad5202d586ca4`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00084-build-pair-wmma-gradient`, manifest `2d38acdfa738e431592e494a87dde98e8c28f2fdab210abe8bde24e000368ed4`.
- Invalid production captures: `runs/kda-cuda-development/diagnostics/attempt-00084-build-pair-wmma-gradient-invalid-001`, manifest `cf8f38c62e994982f812f0041c3a3e88a45b9f76112ccbb0bbd134f27e828ea1`, and `attempt-00084-build-pair-wmma-gradient-invalid-002`, manifest `d598963bbaf8bf0e798e0d0f5db2e6ff99f188b513b75a1d80055462413e306d`.
- Level 1: `runs/kda-cuda-development/attempt-00084-build-pair-wmma-level1`, manifest `edfc63b15555ebafec6b92583718df13fe51f787a1a84d949da771cdd1b7e2e7`.
- Invalid empty-stage sanitizer launch: `runs/kda-cuda-development/diagnostics/attempt-00084-sanitizer-invalid-empty-stage-001`, manifest `efbebb6e2ee5b16ddfd4f4eeb14fcac54070852d79fae43f2170db1cd172e680`.
- Full sanitizer validation: `runs/kda-cuda-development/validations/validation-00020-build-pair-wmma`, manifest `6f8c7db814db9a9863d2de8913a43a791bdbbd0f3087c12da98b3b9cd869c042`.
- Level 2: `runs/kda-cuda-development/attempt-00084-build-pair-wmma-level2`, manifest `5162210f4c7d6bd26a5dd2307001ea981b291e1386a6516eaa28c60592f47a80`.
- Production profile: `runs/kda-cuda-development/diagnostics/attempt-00084-production-profile`, manifest `bb9883c6601493e9f937d1671ec344e2385398bf6e4a63436dc81bdc683e707f`.
- Development-baseline manifest: `runs/kda-cuda-development/baseline/af8d2198e7.json`, SHA-256 `1898346162216d0d8d3e71dac1bc7dd62439dd7cf480127b699d1360d5f149fe`.
- Append-only attempt/reference index SHA-256: `5a3f19d4fa7897378a3af2686eb921938e2072cb9530eddc6b17749cac5fd212`.

**Result**

- Pushed commit `af8d2198e75bb7977b2448b6f516e8185bf44ffd`
  passed ownership 1.0, the complete protected runtime/profile audit, runtime
  FLA freedom, and all four sanitizers with zero errors. Against attempt 83,
  maximum output delta is `4.8828125e-4` and maximum gradient delta is
  `2.055e-09`; all tensors are finite and within frozen tolerances. The
  fresh-cache candidate repeat is bitwise identical.
- Level 1 advanced: T=4096 forward+backward improved
  `16.526 -> 14.769 ms` (10.63%) at memory ratio 1.0. T=4096 forward improved
  0.54%, T=1024 forward+backward improved 1.38%, and T=256 forward+backward
  improved 5.84%; every regression and memory guard passed.
- The candidate-first Level-2 pair measured candidate
  `[31747, 31903, 31823, 31724, 31724]` tok/s, median 31,747, and baseline
  `[30675, 30587, 30703, 30535, 30598]` tok/s, median 30,598. This is a 3.76%
  matched development improvement with identical 5,508.533 MiB peak memory
  and 72.68% of the fixed 43,680 tok/s external FLA reference.
- The profile removed 80 small-N TN BMM launches, 40 transform launches, four
  finish launches, and four P-rebuild launches over two iterations. The new
  forward/backward builders total 0.794 ms/iter versus 2.416 ms/iter for the
  removed attempt-83 path, an estimated targeted reduction of 1.622 ms/iter.
- Two capture harness invocations failed before the project operator ran: the
  first selected an empty candidate-local environment and could not import
  PyTorch; the second imported the package-level function instead of the KDA
  dispatcher module. Both raw logs are preserved and excluded. A sanitizer
  invocation also stopped before build/GPU work because the detached staging
  command used invalid revision `af8d2192`; the checker created no raw artifact,
  and that preservation limitation is recorded explicitly.
- Attempt 84 is accepted only as the new development baseline. The official
  retained milestone remains `4d1a3b231da2c99882324efbda5306a1815e21c7`.
  No confirmation or LM-quality evaluation ran, and this is not statistically
  confirmed evidence.

**Next**

- Continue from attempt 84. The largest named owned paths are the forward WMMA
  scan (2.410 ms/iter), group-boundary WMMA (1.363 ms/iter), and fused pair
  WMMA plus ordered accumulation (1.546 ms/iter). Prefer structural work that
  reduces operands, synchronization, or launches; do not replay prior tile,
  group-width, pair-batch, or fast-exponential variants.

## 2026-08-09 [agent] preserve sub-threshold warp-owned output stores

**Context**

- Attempt 85 starts from accepted attempt 84 and changes only
  `nanochat/mixers/cuda_kda/chunk_wy_forward.cu`. Each forward WMMA warp
  converts and writes its disjoint 16x16 output tile immediately after storing
  the FP32 accumulator, removing the leading block-wide barrier and separate
  CTA output pass before H-next reuses the shared buffer.
- Arithmetic, scan geometry, recurrence state, output layout, allocation, and
  all experiment-defining configuration remain unchanged.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --config configs/research/kda_cuda_ownership.toml \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_085 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seed-4101 production capture against the saved attempt-84 bundle and
# one fresh-cache deterministic repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_084 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_085 \
  runs/kda-cuda-development/attempt-00085-warp-output-level1 \
  --level2-order baseline-first
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_085 \
  push -u origin kda-cuda/wy-warp-output-store-085
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00085-warp-output-protected-checker`, manifest `902d90eb7d1badbc00996f8393e9bb1fffb4d9fc87c376995a39af5749d89fcc`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00085-warp-output-gradient`, manifest `2517566da11a956fc000c8f38c02fc1f6ad0bfa7572ebfeb1e4dd7564f576ffd`.
- Level 1: `runs/kda-cuda-development/attempt-00085-warp-output-level1`, manifest `ec0dfa686c44742fe6e21863fa7429b5266552f49c58d8417666885993c5c847`.
- Append-only attempt/reference index SHA-256: `60f8fc56259299b01fa288ddd620f10bca82bdf8f07cb4152898dcd643a5b882`.

**Result**

- Pushed commit `f1c67f4a78ecca9f9cc4ad2fe56c3e00ab62c2d2`
  passed ownership 1.0, the complete protected runtime/profile audit, runtime
  FLA freedom, bitwise production output/all-gradient correctness against
  attempt 84, and a bitwise fresh-cache repeat.
- Level 1 did not advance: T=4096 forward+backward improved only
  `15.124 -> 15.092 ms` (0.21%), below the frozen 3% gate, at memory ratio
  1.0. Forward-only improved 0.76%. T=1024 forward+backward regressed 2.34%
  but remained inside the declared guard; all memory and important-regression
  guards passed.
- No sanitizers, Level 2, profile, confirmation, or retest ran. Attempt 84
  remains the accepted development baseline. This is neither quality nor
  statistically confirmed evidence.

**Next**

- Preserve attempt 85 unchanged and continue from attempt 84. A single forward
  barrier is too small to retain; target a larger structural reduction in the
  persistent forward scan, group-boundary scan, or pair accumulation without
  replaying prior scheduling variants.

## 2026-08-09 [agent] preserve persistent build/solve fusion after sub-threshold Level 2

**Context**

- Attempt 86 starts from accepted attempt 84 and changes only
  `nanochat/mixers/cuda_kda/chunk_wy_forward.cu` and
  `nanochat/mixers/cuda_kda/chunk_wy_backward.cu`. One persistent CTA per
  chunk computes all ten stable A/M tile pairs, retains M and T in shared
  memory, performs the ordered unit-lower triangular solve, and writes A/T.
- This removes the global M tensor, eighteen extra pair-builder launches, and
  two separate solve launches over forward construction plus backward
  recomputation. The recurrence, C64 chunk geometry, FP32 state, bounded
  backward state, and declared model configuration remain unchanged.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --config configs/research/kda_cuda_ownership.toml \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_086 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seed-4101 production capture and fresh-cache repeat against attempt 84.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_084 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_086 \
  runs/kda-cuda-development/attempt-00086-build-solve-level1 \
  --level2-order candidate-first
# After staging the exact committed patch in detached validation worktree 086:
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_validation_086 \
  --lane optimization --sanitizers <isolated artifact/cache arguments>
# Executed the saved candidate-first Level-2 plan exactly once with absolute
# raw-log paths, followed by one bounded two-iteration production profile.
PYTHONPATH=/home/veer/Master/projects/experiment_swa_kda_cuda_attempt_086 \
TORCH_EXTENSIONS_DIR=/tmp/kda086-production-profile-ext-001 \
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=<artifact>/trace \
  /home/veer/Master/projects/experiment_swa_kda/.venv/bin/python \
  /tmp/kda033_nsys.py
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_086 \
  push -u origin kda-cuda/wy-persistent-build-solve-086
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00086-build-solve-protected-checker`, manifest `d0ea75134907bcdb15c246e9b5f4fd3460bfab39ca74eaa2b8ad645c4695cdcd`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00086-build-solve-gradient`, manifest `4209d1dd5fecad4fcccb3f68b1ce930f2f389a23ccb26d0877c86df7eba75f2d`.
- Invalid pre-GPU sanitizer staging: `runs/kda-cuda-development/diagnostics/attempt-00086-validation-stage-invalid-001`, manifest `0cdeca256e8bcd0f030847958812df2efa1bd4e4d7b16d472fade0ac87c67f44`.
- Level 1: `runs/kda-cuda-development/attempt-00086-build-solve-level1`, manifest `266fb4e60294b5962aa4c4c13b0219c3ee904691ec2ee8264456cae0392d8764`.
- Full sanitizer validation: `runs/kda-cuda-development/validations/validation-00021-build-solve`, manifest `3193d8d31c0e84a8f336dd9c240f8347fb0a540a7dc463143aef3e3b071ef7ca`.
- Level 2: `runs/kda-cuda-development/attempt-00086-build-solve-level2`, manifest `d08c693cb5d6b457d09cf98e710efdb68840806aec33d59c71a11ae8bdc2e4f5`.
- Production profile: `runs/kda-cuda-development/diagnostics/attempt-00086-production-profile`, manifest `4aed887bb92d09077a1f11b52510cd05509611f2f5583bb4d4605787b69fa44b`.
- Append-only attempt/reference index SHA-256: `24a80445ce7437d6bc6478b1b9ba0bd6324440b32c5605cea29ae053ee8b31bb`.

**Result**

- Pushed commit `8ce6d2ad8b1f5cf43e2477f5a111b947442d2d6e` passed ownership 1.0,
  the protected runtime/profile audit, runtime FLA freedom, and all four
  sanitizers with zero errors or hazards. Output and `dq` are bitwise equal to
  attempt 84; the maximum gradient delta is `2.056e-9`, every tensor is finite
  and within the frozen tolerance, and the fresh-cache repeat is bitwise exact.
- Level 1 advanced: T=4096 forward+backward improved
  `15.285 -> 14.691 ms` (3.89%) with memory ratio 1.0 and all guards true.
  T=256 forward+backward improved 5.24%; T=1024 forward+backward regressed
  1.64%, within the declared guard.
- The candidate-first Level-2 pair measured candidate
  `[32177, 31846, 32043, 31981, 31872]` tok/s, median 31,981, and baseline
  `[31853, 31882, 31877, 31782, 31805]` tok/s, median 31,853. The 0.40%
  development improvement is sub-threshold, with peak memory 5,507.033 versus
  5,508.533 MiB and candidate throughput at 73.22% of the fixed 43,680 tok/s
  external FLA reference.
- The profile measured build+solve at 0.709 ms/iteration versus 0.962 ms for
  attempt 84, an estimated targeted saving of only 0.253 ms/iteration. The
  largest remaining owned kernels are forward WMMA at 2.185 ms/iteration,
  group-boundary WMMA at 1.357 ms, and pair-WMMA plus ordered accumulation at
  1.566 ms.
- The first detached validation staging command used invalid guessed revision
  `8ce6d2aa95b3daf73b891306405d5bf9a50c4f95` and failed before checker or GPU
  work with `fatal: bad object` and `No valid patches in input`. No redirected
  raw log exists; that preservation limitation and the exact errors are saved.
- Attempt 86 is preserved but does not replace attempt 84 as the accepted
  development baseline. The official retained milestone remains
  `4d1a3b231da2c99882324efbda5306a1815e21c7`. No confirmation or LM-quality
  evaluation ran, and this is not statistically confirmed evidence.

**Next**

- Continue from attempt 84. FlashKDA's offline forward design separates a
  token-parallel preparation kernel from the persistent recurrence and feeds
  the latter pre-decayed/pre-restored BF16 operands. Test the corresponding
  project-owned mechanism without importing or linking FlashKDA: compact the
  FP32 `W`, decayed q, restored k, and A scan operands to BF16 once after
  construction, then remove their repeated per-value-tile conversions while
  retaining FP32 state and accumulation. Preserve every correctness and
  provenance gate.

## 2026-08-09 [agent] preserve rejected BF16 forward-scan prepack boundary

**Context**

- Attempt 87 starts from accepted attempt 84 and changes only
  `nanochat/mixers/cuda_kda/chunk_wy_forward.cu`. Following the offline
  FlashKDA separation between parallel preparation and persistent recurrence,
  one preparation CTA per C64 chunk snapshots and compacts the FP32 `W`,
  decayed q, restored k, and A masters into BF16 views inside their existing
  allocations. The persistent value-tile scan then loads those exact BF16
  values directly rather than converting the same masters four times.
- FP32 recurrent state, WMMA accumulation, equations, backward path, allocation
  sizes, and model configuration remain unchanged. No reference source is
  imported, linked, or used at runtime.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_087 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seed-4101 production capture and fresh-cache repeat against attempt 84.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_084 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_087 \
  runs/kda-cuda-development/attempt-00087-prepack-level1 \
  --level2-order baseline-first
PYTHONPATH=/home/veer/Master/projects/experiment_swa_kda_cuda_attempt_087 \
TORCH_EXTENSIONS_DIR=/tmp/kda087-production-profile-ext-001 \
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=<artifact>/trace \
  /home/veer/Master/projects/experiment_swa_kda/.venv/bin/python \
  /tmp/kda033_nsys.py
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_087 \
  push -u origin kda-cuda/wy-prepack-scan-bf16-087
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00087-prepack-protected-checker`, manifest `57b53a6ad6d4bd5f5ae0af77d540e6471e3b8caf0c50bae39b43a5f923e17719`.
- Invalid coordinator-CWD production capture: `runs/kda-cuda-development/diagnostics/attempt-00087-prepack-gradient-invalid-cwd-001`, manifest `f2f623412782fa61b1a4dea2fd152a227672e4c7ba4ad7d72d9e627f7afd4031`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00087-prepack-gradient`, manifest `4e8cb499c5fb834afa2c5f7ca546abb42e8438466a8298ee6850bdf683310a0d`.
- Level 1: `runs/kda-cuda-development/attempt-00087-prepack-level1`, manifest `97dfaf56d2f560a05aff130c3c29085db831cef6a6c5bdc40bf98f832774e246`.
- Production profile: `runs/kda-cuda-development/diagnostics/attempt-00087-production-profile`, manifest `3f293fe07e50aedf130d7195e43276dd47e6c8589d8562cc64e9e833088b92d7`.
- Append-only attempt/reference index SHA-256: `5abce26bc3875292febb13c62b3ed2422add0bd2205418b6f3ff502c3747ccc9`.

**Result**

- Pushed commit `a36b0474d5111471731cf9c69940c764181ef921` passed ownership 1.0,
  protected runtime/profile audit, and runtime FLA freedom. Production output
  and `dq` are bitwise equal to attempt 84; maximum gradient delta is
  `1.871e-9`, all tensors pass the frozen tolerance, and the fresh-cache repeat
  is bitwise exact.
- Level 1 rejected the implementation: T=4096 forward improved
  `19.474 -> 19.170 ms` (1.56%), but forward+backward regressed
  `15.015 -> 15.513 ms` (3.32%). Memory ratio is 1.0 and all declared
  regression guards remain within their limits. No sanitizer, Level 2,
  confirmation, or retest ran.
- The boundary profile confirms the mechanism but rejects the realization.
  The forward scan fell from 2.410 to 1.985 ms/iteration, saving 0.426 ms, but
  the shared-memory compaction pass costs 0.320 ms/iteration, leaving only
  about 0.106 ms net in the targeted path.
- The first production capture was launched from the coordinator CWD, causing
  project-relative sources to resolve in the source-free coordinator and a
  `FileNotFoundError` before build/GPU work. Its raw log is preserved and
  excluded.
- Attempt 84 remains the accepted development baseline and the official
  retained milestone remains `4d1a3b231da2c99882324efbda5306a1815e21c7`.
  This is neither statistically confirmed nor LM-quality evidence.

**Next**

- Replace shared-memory in-place compaction with direct parallel writes into
  forward buffers whose original values are dead after the U/W products, and
  compose that with attempt 86's correct persistent build/solve. The measured
  upper bound is roughly 0.426 ms/iteration from scan prepacking plus 0.253 ms
  from build/solve before the new direct-pack cost.

## 2026-08-09 [agent] preserve sub-threshold direct dead-buffer prepack composition

**Context**

- Attempt 88 composes attempt 86's correct persistent A/M build+solve with a
  cheaper realization of attempt 87's BF16 scan prepack. After U/W are formed,
  two flat kernels write qgamma/restored-k/W/A directly into dead P/Q/T/qbar
  backing storage. No new allocation or shared snapshot is used.
- The candidate differs from accepted attempt 84 only in
  `nanochat/mixers/cuda_kda/chunk_wy_forward.cu` and
  `nanochat/mixers/cuda_kda/chunk_wy_backward.cu`; FP32 recurrent state and
  accumulation remain unchanged.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_088 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seed-4101 production comparison/repeat against attempt 84.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_084 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_088 \
  runs/kda-cuda-development/attempt-00088-direct-prepack-level1 \
  --level2-order candidate-first
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_088 \
  push -u origin kda-cuda/wy-direct-prepack-build-solve-088
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00088-direct-prepack-protected-checker`, manifest `3a8a2257437120d616ff932b7cb701a5fde5b1c8d41e48f81cf5a9ec2bd529e7`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00088-direct-prepack-gradient`, manifest `10c90f2d9297a4ca42fdb45f1c38351e730a80d6b66d17849b93fe8602560d67`.
- Level 1: `runs/kda-cuda-development/attempt-00088-direct-prepack-level1`, manifest `335956cac5afb2c69f52c1e3bfc7d80785ff9f2451248bc43e1eb9a62f59726a`.
- Append-only attempt/reference index SHA-256: `a626cae78d7daad50a19431b5e5b3bba46546c7151ed4a6d9c4f772bf8606d4d`.

**Result**

- Pushed commit `eeb08e6c927b43ccc6bdce5019b885edaef0bf4e` passed ownership 1.0,
  protected runtime/profile audit, and runtime FLA freedom. Output and `dq` are
  bitwise equal to attempt 84, maximum gradient delta is `2.056e-9`, all
  tensors pass frozen tolerances, and the fresh-cache repeat is bitwise exact.
- Level 1 is directionally positive but sub-threshold: T=4096 forward improved
  `19.579 -> 19.086 ms` (2.52%) and forward+backward improved
  `14.708 -> 14.445 ms` (1.79%). Memory ratio is 1.0 and every regression guard
  passed. No sanitizer, Level 2, profile, confirmation, or retest ran.
- Attempt 84 remains the accepted development baseline. The official retained
  milestone remains `4d1a3b231da2c99882324efbda5306a1815e21c7`; this is not
  statistically confirmed or LM-quality evidence.

**Next**

- Test BF16 storage for the inter-chunk recurrent state as a separate axis on
  top of attempt 88. The scan already rounds that state to BF16 before every
  WMMA, while BF16 storage can halve its shared footprint and remove repeated
  state casts. Keep the frozen forward/gradient tolerances unchanged and reject
  immediately if the inter-chunk decay/update rounding exceeds them.

## 2026-08-09 [agent] preserve correct BF16 state after sub-threshold Level 2

**Context**

- Attempt 89 starts from attempt 88 and changes only nine lines in
  `nanochat/mixers/cuda_kda/chunk_wy_forward.cu`: the persistent scan stores
  inter-chunk state in BF16, loads it directly for both state-consuming WMMAs,
  and explicitly rounds the FP32 chunk-boundary update back to BF16.
- This is the strongest directly transferable mechanism from the offline
  FlashKDA forward path. It halves shared state storage while preserving FP32
  WMMA accumulation and all frozen equations/tolerances.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_089 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seed-4101 production comparison/repeat against attempt 84.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_084 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_089 \
  runs/kda-cuda-development/attempt-00089-bf16-state-level1 \
  --level2-order baseline-first
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_validation_089 \
  --lane optimization --sanitizers <isolated artifact/cache arguments>
# Executed the saved baseline-first Level-2 pair exactly once, then one bounded
# two-iteration production profile.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_089 \
  push -u origin kda-cuda/wy-bf16-interchunk-state-089
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00089-bf16-state-protected-checker`, manifest `545d022af0f65c27e6904c6979392870d5a18efab5ad52cc28321f9a8ee3d6d7`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00089-bf16-state-gradient`, manifest `e8a9a950829ad4ce577e896cbb7a637013aca1768df71643e7b3929ae2dcd684`.
- Level 1: `runs/kda-cuda-development/attempt-00089-bf16-state-level1`, manifest `a282e372a16aa35c80ea2468fc9ce35f163236b2e9a6ccd36a3bd5a6f998af03`.
- Full sanitizer validation: `runs/kda-cuda-development/validations/validation-00022-bf16-state`, manifest `24b6e1e5f34ee9ecd27b8f5de8c6314d814e0d91811f14a61299e452eaf8a1e0`.
- Level 2: `runs/kda-cuda-development/attempt-00089-bf16-state-level2`, manifest `a1642bb9a4fff9a52e5c1b683fb605bc071f25016ef977c5345966a169d4785f`.
- Production profile: `runs/kda-cuda-development/diagnostics/attempt-00089-production-profile`, manifest `257e9657d9c19171aac6c3ece8a303c700f7310da5c1e7faae7bcc0fd8044b24`.
- Append-only attempt/reference index SHA-256: `dcca263ccc22878fd5d98f75b3549dfef22aa0513c1f25de0d27a3d9e339072a`.

**Result**

- Pushed commit `5feae8573343c69d41bc7c2eeabb166244d414e6` passed ownership 1.0,
  protected runtime/profile audit, runtime FLA freedom, and all four sanitizers
  with zero errors/hazards. Production output and `dq` remain bitwise equal to
  attempt 84; maximum gradient delta is `2.056e-9`, all tensors pass frozen
  tolerances, and the fresh-cache repeat is bitwise exact.
- Level 1 advanced: T=4096 forward+backward improved
  `14.995 -> 14.421 ms` (3.82%) and forward improved 2.08%, with memory ratio
  1.0 and every guard true. T=256 forward+backward regressed 4.25%, still
  within the frozen 5% guard.
- The single baseline-first Level-2 pair measured baseline
  `[31696, 31697, 31656, 31480, 31621]` tok/s, median 31,656, and candidate
  `[31856, 31862, 31918, 31591, 31798]` tok/s, median 31,856. The 0.63%
  development improvement is sub-threshold; candidate peak is 5,507.033 MiB
  versus 5,508.533 MiB, and candidate throughput is 72.93% of the fixed
  43,680 tok/s external reference.
- The profile measures the forward scan at 1.939 versus 2.410 ms/iteration,
  direct packing at 0.312 ms, and build+solve at 0.704 versus 0.962 ms. Their
  estimated targeted net saving is 0.417 ms/iteration. Remaining backward
  targets are group-boundary WMMA at 1.417 ms and pair-WMMA plus ordered
  accumulation at 1.530 ms.
- Attempt 89 is preserved but does not replace attempt 84 as the development
  baseline. The official retained milestone remains
  `4d1a3b231da2c99882324efbda5306a1815e21c7`. No confirmation or LM-quality
  evaluation ran, and this is not statistically confirmed evidence.

**Next**

- Keep attempt 89 as a correct equation/precision milestone, but continue
  matched development from attempt 84. Re-read attempts 69–82 before changing
  the group-boundary or pair-VJP paths; do not replay rejected group-width,
  boundary-copy, recompute, or scheduling variants.

## 2026-08-09 [agent] preserve bitwise-exact backward operand prepack below Level-1 gate

**Context**

- Attempt 90 starts from accepted development baseline attempt 84 and applies
  the offline FlashKDA preparation/recurrence split to backward group-boundary
  recomputation only. The existing group-pack kernel rounds `W` and `E` to
  BF16 once, and the eight value-tile CTAs consume the packed operands instead
  of redundantly converting their FP32 copies inside every WMMA tile.
- No runtime FlashKDA/FLA code is imported or linked. The packed pair occupies
  an existing FP32-sized dead buffer exactly: the unused `R_group` allocation
  during the forward boundary sweep, and `dO_group` before its delayed gradient
  gather during the reverse sweep. Peak live allocation and kernel count are
  unchanged; FP32 `W/E` remain intact for the later complete VJP BMMs.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --config configs/research/kda_cuda_ownership.toml \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_090 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seed-4101 B=2/H=3/T=4096 production comparison against the saved
# attempt-84 bundle, followed by one independent fresh-extension-cache repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_084 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_090 \
  runs/kda-cuda-development/attempt-00090-bf16-group-operands-level1 \
  --level2-order candidate-first
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_090 \
  push -u origin kda-cuda/wy-bf16-group-operands-090
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00090-bf16-group-operands-protected-checker`, manifest `09ad6504b472c2a283a63715da173820c4392b330e2b0edbbdd3a82ae8634476`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00090-bf16-group-operands-gradient`, manifest `29a25eca943ede33f2c9a99bc427a5c3b63740df1c8c34ba78c74b2b5ce39831`.
- Level 1: `runs/kda-cuda-development/attempt-00090-bf16-group-operands-level1`, manifest `fd8e6d32661931c7bf98e31ec872a1a55ba6fc8f2d1a3ba194163d4d0297ffcf`.
- Append-only attempt/reference index SHA-256: `09c3d28dd990796801ce8b0686d30539ad51216db0a0a594843867913b38198e`.

**Result**

- Pushed commit `49d7096e72b4f373aa693d9f28a9afac94a53a91` passed ownership 1.0,
  the complete protected runtime/profile audit, and runtime FLA freedom.
  Production output and every saved gradient are bitwise equal to attempt 84;
  the independent fresh-cache repeat is also bitwise equal for every tensor.
- Level 1 is directionally positive but below the frozen advance threshold:
  T=4096 forward+backward improved `14.8913 -> 14.6631 ms` (1.53%), while
  forward-only improved 0.09%. T=256 forward+backward improved 1.31%, and
  T=1024 forward+backward regressed 0.12%; memory ratio is 1.0 and every
  important-regression and memory guard passed.
- No sanitizer, Level 2, production profile, confirmation, or LM-quality
  evaluation ran. Attempt 84 remains the accepted development baseline and
  the official retained milestone remains
  `4d1a3b231da2c99882324efbda5306a1815e21c7`; this is not statistically
  confirmed evidence.

**Next**

- Preserve the prepack result as evidence that repeated backward operand
  conversion is real but too small alone. Continue from attempt 84 with a
  larger group-boundary fusion that removes shared staging and/or launch and
  global-memory traffic in addition to the conversions; do not compose attempt
  90 into a baseline unless the larger fused axis independently clears the
  frozen gates.

## 2026-08-09 [agent] accept persistent reverse group recurrence as development baseline

**Context**

- Attempt 91 starts from accepted attempt 84 and replaces the eight sequential
  reverse-chunk launch chains inside each backward group. One 256-thread CTA
  owns sixteen value columns for one recurrence, keeps the state adjoint in
  shared memory across all eight reverse chunks, and evaluates both dependent
  products with BF16 WMMA operands and FP32 accumulation.
- The intervention replaces sixteen FP32 BMMs plus prepare/add/boundary/subtract/
  finish launches per group with one persistent reverse kernel. A second
  deterministic kernel reconstructs `dD` from the exact incoming-state history;
  no atomics, full token-state history, wider group, runtime FLA code, or
  rejected attempt-90 composition is used.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --config configs/research/kda_cuda_ownership.toml \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_091 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seed-4101 B=2/H=3/T=4096 production comparison against attempt 84 and
# one independent fresh-extension-cache repeat.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_084 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_091 \
  runs/kda-cuda-development/attempt-00091-persistent-reverse-level1 \
  --level2-order baseline-first
uv run --no-sync research cuda-candidate-check \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_validation_091 \
  --lane optimization --sanitizers <isolated artifact/cache arguments>
# Executed the saved baseline-first Level-2 commands exactly once, followed by
# one bounded two-iteration production Nsight Systems profile.
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_091 \
  push -u origin kda-cuda/wy-persistent-reverse-group-091
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00091-persistent-reverse-protected-checker`, manifest `241fd8fad2720bf1dc2ae0de332a604748e2562f90fc964b9a4e7ba623292274`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00091-persistent-reverse-gradient`, manifest `03605d1c0318a32193346be7d506a9b188e065b205b1cd175705e9cc816f0ece`.
- Level 1: `runs/kda-cuda-development/attempt-00091-persistent-reverse-level1`, manifest `f4f10a3f613da08cbc5d5e4da416f5425f47a83c6ac2d4ad5835f1ca90d7b2a2`.
- Full sanitizer validation: `runs/kda-cuda-development/validations/validation-00023-persistent-reverse`, manifest `3d1293ea30db09e0c72b47106ab872425b54507dfd1175bb67e121de2726ea54`.
- Level 2: `runs/kda-cuda-development/attempt-00091-persistent-reverse-level2`, manifest `a28971c24570062ded2fbd8586c8da9a20e15d55cc011e8605e8d219f9e26bbd`.
- Production profile: `runs/kda-cuda-development/diagnostics/attempt-00091-production-profile`, manifest `d73306f82e09475076708b301decd767834ed2ec980276998e8e7e736480c341`.
- Invalid pre-launch profile artifact: `runs/kda-cuda-development/diagnostics/attempt-00091-production-profile-invalid-cwd-001`, manifest `e35b7d60b249231be8630b8ecf0e5e0bdb138dc88c28f3962a441d7bf49750e9`.
- Development-baseline manifest: `runs/kda-cuda-development/baseline/41858b66f7.json`, SHA-256 `9c0855dc8fa2c714439cc5dd2d1b57a05ca4d4a3eb9a26aaf1692b6998e43c85`.
- Append-only attempt/reference index SHA-256: `9bf32f24181bf6719e71d39e67d1d152211bb975570fcbea2d5ac05f5e41e1a6`.

**Result**

- Pushed commit `41858b66f7cf982007bdf11918efa2b8d99113fe` passed ownership 1.0,
  the full protected runtime/profile audit, runtime FLA freedom, and all four
  sanitizers with zero errors or hazards. Output and `dq` are bitwise equal to
  attempt 84; the maximum gradient delta is `3.638e-12`, all tensors pass the
  frozen tolerance, and the independent repeat is bitwise exact.
- Level 1 advanced decisively: T=4096 forward+backward improved
  `15.0520 -> 13.5989 ms` (9.65%), T=1024 forward+backward improved 3.58%,
  and T=256 forward+backward improved 0.85%. Candidate T=4096 peak allocation
  is 202,770,944 versus 203,950,592 bytes (ratio 0.99422); every guard passed.
- The single baseline-first Level-2 pair measured baseline
  `[31833, 31904, 31680, 31672, 31912]` tok/s, median 31,833, and candidate
  `[32994, 32949, 32827, 32682, 32914]` tok/s, median 32,914. The development
  gain is 3.40%, peak memory is identical at 5,508.533 MiB, and candidate
  throughput is 75.35% of the fixed 43,680 tok/s reference.
- Across two profiled iterations, generic GEMM instances fell from 516 to 260,
  exactly 128 removed per iteration. The new persistent reverse kernel costs
  0.769 ms/iteration and deterministic `dD` costs 0.072 ms; the estimated net
  targeted saving is 0.603 ms/iteration. Remaining named kernels are forward
  WMMA at 2.187 ms, group-boundary WMMA at 1.310 ms, pair WMMA at 0.988 ms,
  and ordered pair accumulation at 0.582 ms per iteration.
- The first profile command used a candidate-relative `mkdir` with an absolute
  coordinator log target and failed before profiler/GPU work with
  `/bin/bash: .../profile.log: No such file or directory`. No raw log could be
  created; the exact failure and limitation are preserved. The first offline
  stats packaging chain later emitted valid `stats.log` but guessed the wrong
  generated CSV name and ended with `cp: cannot stat ...trace_cuda_gpu_kern_sum.csv`.
  A bounded offline stats rerun extracted the CSV from the same immutable trace;
  it launched no GPU work and the failed output was not treated as evidence.
- Attempt 91 is the new accepted development baseline only. The official
  retained milestone remains `4d1a3b231da2c99882324efbda5306a1815e21c7`.
  No confirmation or LM-quality evaluation ran, and this is not statistically
  confirmed evidence.

**Next**

- Continue from attempt 91 while preserving the persistent reverse recurrence.
  The largest measured opportunities are forward WMMA, group-boundary WMMA,
  and the pair-WMMA/ordered-accumulation path. Favor another structural fusion
  or producer/consumer layout that eliminates launches/global traffic; do not
  replay isolated prepacking, barrier sharing, group widening, or copy-only
  variants. The fixed reference remains 43,680 tok/s and the objective remains
  at least 45,000 tok/s.

## 2026-08-09 [agent] reject prepared BF16 group-boundary recurrence

**Context**

- Pair-WMMA production and ordered accumulation cannot be directly fused while
  preserving determinism: the producer owns tile pairs, whereas the consumer
  owns output rows and ordered contributions. Direct fusion would require
  atomics, duplicate products, or rejected pair-batch widening.
- Attempt 92 therefore starts from accepted attempt 91 and tests the next
  structural group-boundary axis. `W/E` are rounded once during the existing
  group pack into dead scratch, and the persistent eight-chunk boundary kernel
  keeps its local state in BF16 while retaining FP32 WMMA accumulation and
  explicit FP32-to-BF16 chunk-boundary rounding.

**Commands**

```bash
uv run --no-sync research cuda-candidate-check \
  --config configs/research/kda_cuda_ownership.toml \
  --worktree /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_092 \
  --lane optimization <isolated artifact/cache arguments>
# Exact seed-4101 production comparison/repeat against attempt 91.
uv run --no-sync python scripts/kda_cuda_development.py \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_091 \
  /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_092 \
  runs/kda-cuda-development/attempt-00092-bf16-group-recurrence-level1 \
  --level2-order candidate-first
git -C /home/veer/Master/projects/experiment_swa_kda_cuda_attempt_092 \
  push -u origin kda-cuda/wy-bf16-group-recurrence-092
```

**Artifacts**

- Protected checker: `runs/kda-cuda-development/diagnostics/attempt-00092-bf16-group-recurrence-protected-checker`, manifest `7d22c49f5e3df5c98aa47f60be44cf409e1e9506f0c8fcf996800015a214dd6b`.
- Production comparison/repeat: `runs/kda-cuda-development/diagnostics/attempt-00092-bf16-group-recurrence-gradient`, manifest `5e87e5c241ae4094f43e62b8067ba7adbf7c082c5f8febb6ecaeb3b95a6be1f6`.
- Level 1: `runs/kda-cuda-development/attempt-00092-bf16-group-recurrence-level1`, manifest `e963639540fafe6b3db8b520f38c181d474a72ddaeb7ac0977e2c7cd1ea42a60`.
- Append-only attempt/reference index SHA-256: `f3783635ea56d1def333ab6855946a129993b6587def62c1cec146c36f68b573`.

**Result**

- Pushed commit `82b397b1f4492b8d412d78514918ef9701776b3c` passed ownership 1.0,
  the complete protected runtime/profile audit, and runtime FLA freedom.
  Output, `dv`, and `dbeta` are bitwise equal to attempt 91; the maximum
  gradient delta is `3.638e-12`, every tensor passes the frozen tolerance, and
  the independent repeat is bitwise exact.
- Level 1 rejected the candidate. T=4096 forward+backward regressed
  `13.2552 -> 13.4394 ms` (1.39%); T=1024 regressed 1.26% and T=256 regressed
  3.40%. Memory ratio is 1.0 and all frozen guards remain within their limits.
- No sanitizer, Level 2, profile, confirmation, or LM-quality evaluation ran.
  Attempt 91 remains the accepted development baseline and the official
  retained milestone remains `4d1a3b231da2c99882324efbda5306a1815e21c7`.
  This is not statistically confirmed evidence.

**Next**

- Preserve FP32 local state and in-kernel operand conversion in the
  group-boundary path. Continue from attempt 91; do not compose attempt 92 or
  retry direct pair producer/consumer fusion. Target a larger ownership-aligned
  fusion, especially the post-pair accumulation/prefix/finalize chain or a
  custom producer that removes group `T@P/T@Q` materialization entirely.
