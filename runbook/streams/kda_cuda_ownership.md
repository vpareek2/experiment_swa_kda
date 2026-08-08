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
