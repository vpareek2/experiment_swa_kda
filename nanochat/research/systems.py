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

from nanochat.research.artifacts import atomic_write_json, environment_provenance, git_provenance, make_run_id, select_triton_ptxas
from nanochat.research.config import ResearchConfig

STEP_PREFIX = "RESEARCH_TRAIN_STEP "
RESULT_PREFIX = "RESEARCH_TRAIN_RESULT "


def trainer_command(config: ResearchConfig, tag: str, iterations: int, profile_output: str | None = None) -> list[str]:
    train = config.training
    command = [
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
    if profile_output:
        command.extend([
            "--speed-profile-output", profile_output,
            "--speed-profile-warmup-steps", "1",
            "--speed-profile-max-bytes", str(config.speed_supervisor.profile_max_bytes),
            "--speed-profile-operator-rows", str(config.speed_supervisor.profile_operator_rows),
        ])
    return command


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


def _run(command: list[str], root: Path, log: Path, timeout: float, execution_mode: str) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("NANOCHAT_DTYPE", "bfloat16")
    if execution_mode == "eager":
        # This is a frozen benchmark mode, deliberately not an implicit
        # recovery from a compiler failure. It disables model and optimizer
        # compilation for every candidate and baseline subprocess.
        env["TORCH_COMPILE_DISABLE"] = "1"
    elif execution_mode != "full_compile":
        raise ValueError(f"unsupported systems execution mode: {execution_mode}")
    ptxas = select_triton_ptxas()
    if ptxas:
        env["TRITON_PTXAS_PATH"] = ptxas
    env["FLA_FLASH_KDA"] = "0"
    env["FLA_TILELANG"] = "0"
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


def run_speed_profile(root: str | Path, config: ResearchConfig, artifact_dir: str | Path, label: str) -> dict[str, Any]:
    """Run the required bounded profile in a clean, reviewed worktree."""
    repo = Path(root).resolve()
    output = Path(artifact_dir); output.mkdir(parents=True, exist_ok=False)
    profile_path, log_path = output / "profile.json", output / "profile.log"
    result = _run(
        trainer_command(config, f"{label}-profile", 2, str(profile_path)), repo, log_path,
        config.speed_supervisor.profile_timeout_seconds, config.systems.execution_mode,
    )
    result["log"] = str(log_path); result["artifact"] = str(profile_path)
    if result["status"] != "complete":
        return result
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        if profile_path.stat().st_size > config.speed_supervisor.profile_max_bytes:
            raise ValueError("profile artifact exceeds frozen byte cap")
        if not payload.get("regions") or not payload.get("operators"):
            raise ValueError("profile artifact has no regions or operators")
        result["profile"] = payload
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        result["status"] = "invalid_profile"; result["reason"] = str(error)
    return result


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
    cold = _run(
        trainer_command(config, f"{run_id}-cold", 1), repo, run_dir / "compile.log",
        systems.compile_timeout_seconds, systems.execution_mode,
    )
    cold["log"] = str(run_dir / "compile.log")
    cold["phase"] = "cold_setup" if systems.execution_mode == "eager" else "cold_compile"
    if cold["status"] != "complete":
        result = {
            "schema_version": 1, "run_id": run_id, "status": cold["status"],
            "execution_mode": systems.execution_mode, "compile": cold,
        }
        atomic_write_json(run_dir / "summary.json", result); return result
    warm_steps = systems.warmup_steps + systems.timed_steps
    warm = _run(
        trainer_command(config, f"{run_id}-warm", warm_steps), repo, run_dir / "train.log",
        systems.warm_timeout_seconds, systems.execution_mode,
    )
    warm["log"] = str(run_dir / "train.log")
    result = {"schema_version": 1, "run_id": run_id, "status": warm["status"],
              "execution_mode": systems.execution_mode, "compile": cold, "training": warm,
              "warm_training": summarize_warm_steps(parse_steps(run_dir / "train.log"), systems.warmup_steps) if warm["status"] == "complete" else None,
              "prefill": {"status": "not_run"}, "decode": {"status": "not_run"}}
    atomic_write_json(run_dir / "summary.json", result)
    return result
