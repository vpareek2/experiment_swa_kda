"""Bounded, protected systems benchmark for architecture candidates."""
from __future__ import annotations

import json
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from nanochat.research.artifacts import atomic_write_json, environment_provenance, git_provenance, make_run_id
from nanochat.research.config import ResearchConfig

STEP_PREFIX = "RESEARCH_TRAIN_STEP "
RESULT_PREFIX = "RESEARCH_TRAIN_RESULT "


def trainer_command(config: ResearchConfig, tag: str, iterations: int) -> list[str]:
    train = config.training
    return [
        sys.executable, "-m", "scripts.base_train", "--seed", str(config.run.seed),
        "--depth", str(train.depth), "--head-dim", str(train.head_dim),
        "--window-pattern", train.window_pattern, "--kda-backend", train.kda_backend,
        "--sliding-window", str(train.sliding_window),
        "--force-final-full" if train.force_final_full else "--no-force-final-full",
        "--max-seq-len", str(train.sequence_length), "--device-batch-size", str(train.device_batch_size),
        "--total-batch-size", str(train.total_batch_size), "--num-iterations", str(iterations),
        "--eval-every", "-1", "--core-metric-every", "-1", "--sample-every", "-1", "--save-every", "-1",
        "--model-tag", tag, "--run", "dummy",
    ]


def parse_steps(log: Path) -> list[dict[str, Any]]:
    return [json.loads(line[len(STEP_PREFIX):]) for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith(STEP_PREFIX)]


def summarize_warm_steps(steps: list[dict[str, Any]], warmup_steps: int) -> dict[str, Any]:
    timed = steps[warmup_steps:]
    if not timed:
        raise ValueError("systems run produced no timed training steps")
    seconds = [float(item["step_seconds"]) for item in timed]
    throughput = [float(item["tokens_per_second"]) for item in timed]
    return {"timed_steps": len(timed), "step_seconds": {"median": statistics.median(seconds), "values": seconds},
            "tokens_per_second": {"median": statistics.median(throughput), "values": throughput}}


def _run(command: list[str], root: Path, log: Path, timeout: float) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("NANOCHAT_DTYPE", "bfloat16")
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            return {"status": "complete" if code == 0 else "crash", "returncode": code, "seconds": time.monotonic() - started}
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
            return {"status": "compile_timeout", "seconds": time.monotonic() - started, "timeout_seconds": timeout}


def run_system_benchmark(root: str | Path, config: ResearchConfig, artifact_root: str | Path | None = None) -> dict[str, Any]:
    repo = Path(root).resolve()
    provenance = git_provenance(repo)
    if provenance["commit"] is None or provenance["dirty"]:
        raise ValueError("systems benchmarks require a committed, clean worktree")
    systems = config.systems
    if not systems.enabled:
        raise ValueError("systems benchmark is disabled in the frozen config")
    output = Path(artifact_root or config.run.artifact_root)
    if not output.is_absolute(): output = repo / output
    run_id = make_run_id(config.run.name, config.run.seed, provenance["commit"])
    run_dir = output / f"systems-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(run_dir / "resolved-config.json", config.to_dict())
    atomic_write_json(run_dir / "manifest.json", {"run_id": run_id, "git": provenance, "environment": environment_provenance()})
    cold = _run(trainer_command(config, f"{run_id}-cold", 1), repo, run_dir / "compile.log", systems.compile_timeout_seconds)
    cold["log"] = str(run_dir / "compile.log")
    if cold["status"] != "complete":
        result = {"schema_version": 1, "run_id": run_id, "status": cold["status"], "compile": cold}
        atomic_write_json(run_dir / "summary.json", result); return result
    warm_steps = systems.warmup_steps + systems.timed_steps
    warm = _run(trainer_command(config, f"{run_id}-warm", warm_steps), repo, run_dir / "train.log", systems.compile_timeout_seconds * 2)
    warm["log"] = str(run_dir / "train.log")
    result = {"schema_version": 1, "run_id": run_id, "status": warm["status"], "compile": cold, "training": warm,
              "warm_training": summarize_warm_steps(parse_steps(run_dir / "train.log"), systems.warmup_steps) if warm["status"] == "complete" else None,
              "prefill": {"status": "not_run"}, "decode": {"status": "not_run"}}
    atomic_write_json(run_dir / "summary.json", result)
    return result
