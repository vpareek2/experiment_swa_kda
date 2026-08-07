# KDA Training-Speed Autoresearch Program

## Purpose

Improve **only steady-state eager 4k KDA training throughput** through one
committed implementation hypothesis at a time. This is systems research, not
architecture-quality research: it does not evaluate BPB, CORE, RULER, LongBench,
memory probes, or downstream quality.

Read `AGENTS.md`, this file, and `runbook/streams/kda_training_speed.md` before
working. The supervisor also supplies the latest `research speed-supervisor
summary` JSON.

## Prime-Agent roles

A Prime Agent parent is the **supervisor**. It owns the ledger, worktree
coordination, benchmark invocation, result interpretation, and runbook entry.
It must not edit KDA candidate implementation files.

A spawned Prime Agent child is a **candidate worker**. It may work only in its
assigned candidate worktree and may edit only `nanochat/mixers/kda.py`. It must
not edit the supervisor, benchmark/configuration code, tests, runbook, data,
tokenizer, fallback policy, or generated artifacts. It returns its hypothesis,
commit SHA, changed-file list, and risks to its parent.

## Candidate cycle

1. The supervisor chooses a clean baseline commit and creates a candidate
   worktree/branch from it.
2. The child reads the briefing, source map, latest ledger summary, and relevant
   read-only local references. It states one primary hypothesis, makes one
   implementation change, runs narrow local checks if useful, and commits.
3. The supervisor records the immutable refs and idea with:

   ```bash
   research speed-supervisor intake --base-ref <base> --candidate-ref <candidate> --idea "<hypothesis>"
   ```

4. Only an accepted attempt may run:

   ```bash
   research speed-supervisor run --attempt <id>
   research speed-supervisor summary
   ```

   The supervisor runs fixed correctness tests, then baseline-pre/candidate/
   baseline-post measurements in detached worktrees. It records every outcome,
   including invalid attempts, in SQLite and in the runbook.
5. The supervisor gives the compact summary to the next candidate worker. It
   retains no candidate automatically; a human/supervisor promotion decision is
   required after an `improved` result.

## Frozen score and invalidity

The only score is the median of the declared warm 4k training token/s samples.
The lane is explicitly eager (`TORCH_COMPILE_DISABLE=1`) for both model and
optimizer; cold setup is diagnostic only. An improvement needs the frozen
throughput threshold and stable baseline-pre/baseline-post measurements.

Every systems-valid baseline-pre and candidate run also requires the protected
CUDA-event operator-region profile. It records fixed named regions (KDA layers,
projections, short convolutions, FLA forward, output components, forward,
backward, and optimizer) for one warmed full update. It has a hard timeout and
byte cap, never exports a Chrome trace, and is not model-selectable. Profile
failure is invalid, not missing feedback.

A failed correctness test, fallback, NaN, OOM, crash, timeout, malformed metric,
profile failure, or excessive baseline drift is invalid/retest, never a slow
numeric score and never a win.

## References and evidence

Use the tracked briefing for verified facts. Large local material belongs in
ignored `ref/` and is read-only reference material: never import it at runtime
or copy it into candidate code. Prefer local profiler/ledger evidence over
assumptions from external guides. Do not place hostnames, private paths,
credentials, datasets, checkpoints, or private seeds in Git or the runbook.

## Stop conditions

The protected ledger limits this protocol to 24 attempts. Stop when that budget
is exhausted, a systems invariant fails, or the supervisor cannot attribute an
observed regression/improvement from saved evidence. Do not start quality or
long-training campaigns from this program.
