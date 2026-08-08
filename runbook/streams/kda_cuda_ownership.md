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
