"""Fail-closed, speed-only KDA candidate supervisor.

This module intentionally does not generate or edit candidates. An external model
proposes one committed change; this supervisor records the idea, validates the
commit, runs fixed correctness tests and A-B-A systems measurements, and returns
small evidence-backed feedback for the next model turn.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from nanochat.research.artifacts import canonical_json_bytes
from nanochat.research.config import ResearchConfig
from nanochat.research.systems import run_system_benchmark


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _git(root: Path, args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    if result.returncode:
        raise ValueError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _commit(root: Path, ref: str) -> str:
    return _git(root, ["rev-parse", "--verify", f"{ref}^{{commit}}"])


def _clean(root: Path) -> None:
    if _git(root, ["status", "--porcelain"]):
        raise ValueError("speed supervisor coordinator requires a clean worktree")


def _ledger_path(root: Path, config: ResearchConfig, override: str | Path | None) -> Path:
    path = Path(override or config.speed_supervisor.ledger_path)
    return path if path.is_absolute() else root / path


def _open(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30, isolation_level=None)
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=30000")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS attempts (
      id INTEGER PRIMARY KEY, created_at REAL NOT NULL, status TEXT NOT NULL,
      base_sha TEXT, candidate_sha TEXT, idea TEXT NOT NULL, changed_paths_json TEXT,
      patch_sha TEXT, patch_text TEXT, protocol_sha TEXT NOT NULL, decision TEXT, reason TEXT, summary_json TEXT
    );
    CREATE UNIQUE INDEX IF NOT EXISTS attempt_commit_protocol
      ON attempts(candidate_sha, protocol_sha) WHERE candidate_sha IS NOT NULL;
    CREATE TABLE IF NOT EXISTS phases (
      id INTEGER PRIMARY KEY, attempt_id INTEGER NOT NULL REFERENCES attempts(id),
      role TEXT NOT NULL, phase TEXT NOT NULL, ordinal INTEGER NOT NULL,
      status TEXT NOT NULL, started_at REAL NOT NULL, finished_at REAL NOT NULL,
      returncode INTEGER, timeout_seconds REAL, artifact_dir TEXT, metric_json TEXT,
      reason TEXT, UNIQUE(attempt_id, role, phase, ordinal)
    );
    CREATE TABLE IF NOT EXISTS events (
      id INTEGER PRIMARY KEY, attempt_id INTEGER NOT NULL REFERENCES attempts(id),
      at REAL NOT NULL, from_status TEXT, to_status TEXT NOT NULL, detail_json TEXT NOT NULL
    );
    """)
    columns = {row[1] for row in db.execute("PRAGMA table_info(attempts)")}
    for name in ("patch_sha", "patch_text"):
        if name not in columns:
            db.execute(f"ALTER TABLE attempts ADD COLUMN {name} TEXT")
    return db


def _protocol_sha(config: ResearchConfig) -> str:
    # The candidate code is deliberately excluded; this identifies frozen policy.
    return _sha(config.to_dict())


def initialize(root: str | Path, config: ResearchConfig, ledger: str | Path | None = None) -> dict[str, Any]:
    repo = Path(root).resolve()
    if not config.speed_supervisor.enabled:
        raise ValueError("speed supervisor is disabled in this config")
    path = _ledger_path(repo, config, ledger)
    protocol_sha = _protocol_sha(config)
    db = _open(path)
    try:
        previous = db.execute("SELECT value FROM metadata WHERE key='protocol_sha'").fetchone()
        if previous and previous[0] != protocol_sha:
            raise ValueError("ledger protocol hash differs from the frozen config")
        db.execute("INSERT OR IGNORE INTO metadata(key,value) VALUES('protocol_sha',?)", (protocol_sha,))
        db.execute("INSERT OR IGNORE INTO metadata(key,value) VALUES('schema','speed-supervisor')")
    finally:
        db.close()
    return {"status": "ready", "ledger": str(path), "protocol_sha": protocol_sha}


def _allowed(path: str, allowed: tuple[str, ...]) -> bool:
    return any(path == item or (item.endswith("/") and path.startswith(item)) for item in allowed)


def _record_event(db: sqlite3.Connection, attempt_id: int, old: str | None, new: str, detail: dict[str, Any]) -> None:
    db.execute("INSERT INTO events(attempt_id,at,from_status,to_status,detail_json) VALUES(?,?,?,?,?)",
               (attempt_id, time.time(), old, new, json.dumps(detail, sort_keys=True)))


def intake(root: str | Path, config: ResearchConfig, base_ref: str, candidate_ref: str,
           idea: str, ledger: str | Path | None = None) -> dict[str, Any]:
    repo = Path(root).resolve(); _clean(repo)
    ready = initialize(repo, config, ledger); path = Path(ready["ledger"])
    db = _open(path)
    try:
        base, candidate = _commit(repo, base_ref), _commit(repo, candidate_ref)
        if base == candidate:
            reason = "candidate commit must differ from base commit"; changed: list[str] = []
        elif subprocess.run(["git", "merge-base", "--is-ancestor", base, candidate], cwd=repo).returncode:
            reason = "base commit must be an ancestor of candidate"; changed = []
        else:
            changed = [line for line in _git(repo, ["diff", "--name-only", f"{base}..{candidate}"]).splitlines() if line]
            reason = None if changed and all(_allowed(item, config.speed_supervisor.candidate_paths) for item in changed) else (
                "candidate has no changes" if not changed else "candidate changes paths outside the frozen allowlist"
            )
        patch = _git(repo, ["diff", "--binary", f"{base}..{candidate}"]) if changed else ""
        patch_sha = hashlib.sha256(patch.encode("utf-8")).hexdigest()
        existing = db.execute("SELECT id,status FROM attempts WHERE candidate_sha=? AND protocol_sha=?", (candidate, ready["protocol_sha"])).fetchone()
        if existing:
            return {"attempt_id": existing[0], "status": "already_recorded", "reason": f"candidate already {existing[1]}",
                    "base_sha": base, "candidate_sha": candidate, "changed_paths": changed}
        used = db.execute("SELECT COUNT(*) FROM attempts WHERE protocol_sha=?", (ready["protocol_sha"],)).fetchone()[0]
        if reason is None and used >= config.speed_supervisor.max_attempts:
            reason = "frozen maximum attempt budget exhausted"
        status = "accepted" if reason is None else "rejected"
        db.execute("BEGIN IMMEDIATE")
        try:
            cursor = db.execute("INSERT INTO attempts(created_at,status,base_sha,candidate_sha,idea,changed_paths_json,patch_sha,patch_text,protocol_sha,reason) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (time.time(), status, base, candidate, idea, json.dumps(changed), patch_sha, patch, ready["protocol_sha"], reason))
            attempt_id = int(cursor.lastrowid); _record_event(db, attempt_id, None, status, {"reason": reason, "changed_paths": changed})
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK"); raise
        return {"attempt_id": attempt_id, "status": status, "reason": reason, "base_sha": base, "candidate_sha": candidate, "changed_paths": changed}
    finally:
        db.close()


def _bounded(command: list[str], cwd: Path, log: Path, timeout: float) -> dict[str, Any]:
    started = time.monotonic(); log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            return {"status": "complete" if code == 0 else "crash", "returncode": code, "seconds": time.monotonic() - started}
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL)
            return {"status": "timeout", "seconds": time.monotonic() - started, "timeout_seconds": timeout}


@contextmanager
def _worktree(repo: Path, commit: str) -> Iterator[Path]:
    directory = Path(tempfile.mkdtemp(prefix="kda-speed-"))
    try:
        _git(repo, ["worktree", "add", "--detach", str(directory), commit])
        if _commit(directory, "HEAD") != commit:
            raise ValueError("detached worktree HEAD did not bind to reviewed commit")
        yield directory
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(directory)], cwd=repo, text=True, capture_output=True)
        if directory.exists():
            import shutil; shutil.rmtree(directory, ignore_errors=True)


def _phase(db: sqlite3.Connection, attempt: int, role: str, phase: str, ordinal: int,
           result: dict[str, Any], artifact: Path, metric: dict[str, Any] | None = None) -> None:
    db.execute("INSERT OR REPLACE INTO phases(attempt_id,role,phase,ordinal,status,started_at,finished_at,returncode,timeout_seconds,artifact_dir,metric_json,reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
       (attempt, role, phase, ordinal, result["status"], time.time() - result.get("seconds", 0), time.time(), result.get("returncode"), result.get("timeout_seconds"), str(artifact), json.dumps(metric, sort_keys=True) if metric else None, result.get("reason")))


def _benchmark(root: Path, config: ResearchConfig, output: Path) -> dict[str, Any]:
    try:
        result = run_system_benchmark(root, config, output)
    except Exception as error:
        return {"status": "launch_error", "reason": str(error)}
    if result.get("status") != "complete" or not result.get("warm_training"):
        return {"status": "invalid", "reason": result.get("status", "missing warm timing"), "systems": result}
    values = result["warm_training"]["tokens_per_second"].get("values", [])
    if len(values) != config.systems.timed_steps or any(not isinstance(item, (int, float)) or item <= 0 for item in values):
        return {"status": "invalid", "reason": "missing or nonpositive timed steps", "systems": result}
    return {"status": "complete", "systems": result, "tokens_per_second": float(result["warm_training"]["tokens_per_second"]["median"])}


def run_attempt(root: str | Path, config: ResearchConfig, attempt_id: int, ledger: str | Path | None = None) -> dict[str, Any]:
    repo = Path(root).resolve(); _clean(repo); ready = initialize(repo, config, ledger); db = _open(Path(ready["ledger"]))
    try:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT status,base_sha,candidate_sha,idea FROM attempts WHERE id=?", (attempt_id,)).fetchone()
        if not row: db.execute("ROLLBACK"); raise ValueError(f"unknown attempt {attempt_id}")
        status, base, candidate, idea = row
        if status != "accepted": db.execute("ROLLBACK"); raise ValueError(f"attempt {attempt_id} is {status}, not accepted")
        db.execute("UPDATE attempts SET status='testing' WHERE id=?", (attempt_id,)); _record_event(db, attempt_id, status, "testing", {"idea": idea}); db.execute("COMMIT")
        artifact = repo / config.run.artifact_root / "speed-supervisor" / f"attempt-{attempt_id:05d}"
        if artifact.exists(): raise ValueError(f"attempt artifact already exists: {artifact}")
        with _worktree(repo, candidate) as candidate_root:
            test = _bounded([sys.executable, "-m", "pytest", "-q", *config.speed_supervisor.correctness_tests], candidate_root,
                            artifact / "candidate" / "correctness.log", config.speed_supervisor.test_timeout_seconds)
            _phase(db, attempt_id, "candidate", "correctness", 0, test, artifact / "candidate")
            if test["status"] != "complete":
                return _finish(db, attempt_id, "invalid", "correctness gate failed", artifact)
            pre = None; post = None
            with _worktree(repo, base) as base_root:
                pre = _benchmark(base_root, config, artifact / "baseline-pre")
            _phase(db, attempt_id, "baseline", "systems", 0, pre, artifact / "baseline-pre", {"tokens_per_second": pre.get("tokens_per_second")})
            candidate_result = _benchmark(candidate_root, config, artifact / "candidate")
            _phase(db, attempt_id, "candidate", "systems", 0, candidate_result, artifact / "candidate", {"tokens_per_second": candidate_result.get("tokens_per_second")})
            with _worktree(repo, base) as base_root:
                post = _benchmark(base_root, config, artifact / "baseline-post")
            _phase(db, attempt_id, "baseline", "systems", 1, post, artifact / "baseline-post", {"tokens_per_second": post.get("tokens_per_second")})
        if any(item["status"] != "complete" for item in (pre, candidate_result, post)):
            return _finish(db, attempt_id, "invalid", "systems benchmark invalid", artifact)
        baseline = statistics.median([pre["tokens_per_second"], post["tokens_per_second"]])
        drift = abs(post["tokens_per_second"] - pre["tokens_per_second"]) / baseline
        relative = candidate_result["tokens_per_second"] / baseline - 1.0
        if drift > config.speed_supervisor.max_baseline_drift_fraction:
            decision, reason = "retest", "baseline drift exceeded frozen threshold"
        elif relative >= config.speed_supervisor.min_relative_throughput_improvement:
            decision, reason = "improved", "candidate exceeded frozen throughput threshold"
        else:
            decision, reason = "not_improved", "candidate did not exceed frozen throughput threshold"
        summary = {"objective": "steady_state_training_tokens_per_second_only", "quality_not_evaluated": True,
                   "attempt_id": attempt_id, "base_sha": base, "candidate_sha": candidate, "idea": idea,
                   "baseline_tokens_per_second": baseline, "candidate_tokens_per_second": candidate_result["tokens_per_second"],
                   "relative_change": relative, "baseline_drift": drift, "decision": decision, "reason": reason,
                   "artifact_dir": str(artifact)}
        return _finish(db, attempt_id, "complete", reason, artifact, decision, summary)
    finally:
        db.close()


def _finish(db: sqlite3.Connection, attempt: int, status: str, reason: str, artifact: Path,
            decision: str | None = None, summary: dict[str, Any] | None = None) -> dict[str, Any]:
    db.execute("UPDATE attempts SET status=?,decision=?,reason=?,summary_json=? WHERE id=?",
               (status, decision, reason, json.dumps(summary, sort_keys=True) if summary else None, attempt))
    _record_event(db, attempt, "testing", status, {"reason": reason})
    return {"attempt_id": attempt, "status": status, "decision": decision, "reason": reason,
            "summary": summary, "artifact_dir": str(artifact)}


def summary(root: str | Path, config: ResearchConfig, attempt_id: int | None = None,
            ledger: str | Path | None = None) -> dict[str, Any]:
    repo = Path(root).resolve(); ready = initialize(repo, config, ledger); db = _open(Path(ready["ledger"]))
    try:
        where, values = ("WHERE id=?", (attempt_id,)) if attempt_id else ("", ())
        rows = db.execute(f"SELECT id,status,base_sha,candidate_sha,idea,changed_paths_json,patch_sha,decision,reason,summary_json FROM attempts {where} ORDER BY id DESC LIMIT 12", values).fetchall()
        attempts = [{"attempt_id": row[0], "status": row[1], "base_sha": row[2], "candidate_sha": row[3], "idea": row[4],
                     "changed_paths": json.loads(row[5] or "[]"), "patch_sha256": row[6], "decision": row[7], "reason": row[8],
                     "measurement": json.loads(row[9]) if row[9] else None} for row in rows]
        return {"objective": "steady_state_training_tokens_per_second_only", "quality_not_evaluated": True,
                "ledger": ready["ledger"], "protocol_sha": ready["protocol_sha"], "attempts": attempts,
                "next_model_instruction": "Propose one committed KDA-only implementation hypothesis; do not edit supervisor, tests, config, or benchmark code."}
    finally:
        db.close()
