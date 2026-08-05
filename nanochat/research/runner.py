from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import torch

from nanochat.common import get_base_dir
from nanochat.research.artifacts import (
    append_jsonl,
    atomic_write_json,
    environment_provenance,
    git_provenance,
    make_run_id,
    protected_fingerprint,
)
from nanochat.research.config import ResearchConfig
from nanochat.research.decision import classify_candidate, objectives_from_config
from nanochat.research.probe import run_memory_probe


TRAIN_RESULT_PREFIX = "RESEARCH_TRAIN_RESULT "
TRAIN_STEP_PREFIX = "RESEARCH_TRAIN_STEP "


def doctor(root: str | Path, config: ResearchConfig) -> dict[str, Any]:
    repo = Path(root).resolve()
    base_dir = Path(get_base_dir())
    tokenizer_files = [base_dir / "tokenizer" / "tokenizer.pkl", base_dir / "tokenizer" / "token_bytes.pt"]
    data_dir = base_dir / "base_data_climbmix"
    parquet = sorted(data_dir.glob("*.parquet")) if data_dir.exists() else []
    environment = environment_provenance()
    provenance = git_provenance(repo)
    checks = {
        "git_repository": (repo / ".git").exists(),
        "uv": shutil.which("uv") is not None,
        "tokenizer": all(path.exists() for path in tokenizer_files),
        "dataset_shards": len(parquet),
        "cuda": environment["cuda_available"],
        "bfloat16": not environment["cuda_available"] or tuple(environment.get("compute_capability", (0, 0))) >= (8, 0),
        "clean_commit": provenance["commit"] is not None and not provenance["dirty"],
    }
    environment_valid = all(value for key, value in checks.items() if key not in {"dataset_shards", "clean_commit"}) and checks["dataset_shards"] >= 1
    return {
        "schema_version": 1,
        "valid": environment_valid,
        "research_ready": environment_valid and checks["clean_commit"],
        "checks": checks,
        "environment": environment,
        "config_suite": config.run.suite,
        "base_dir": str(base_dir),
    }


def prepare_data(execute: bool = False) -> dict[str, Any]:
    commands = [
        [sys.executable, "-m", "nanochat.dataset", "-n", "8"],
        [sys.executable, "-m", "scripts.tok_train", "--max-chars", "2000000000"],
        [sys.executable, "-m", "scripts.tok_eval"],
    ]
    if execute:
        for command in commands:
            subprocess.run(command, check=True)
    return {"executed": execute, "commands": commands, "base_dir": get_base_dir()}


def _trainer_command(config: ResearchConfig, run_id: str) -> list[str]:
    train = config.training
    command = [
        sys.executable, "-m", "scripts.base_train",
        "--seed", str(config.run.seed),
        "--depth", str(train.depth),
        "--head-dim", str(train.head_dim),
        "--window-pattern", train.window_pattern,
        "--sliding-window", str(train.sliding_window),
        "--force-final-full" if train.force_final_full else "--no-force-final-full",
        "--max-seq-len", str(train.sequence_length),
        "--device-batch-size", str(train.device_batch_size),
        "--total-batch-size", str(train.total_batch_size),
        "--eval-every", "1000000000",
        "--eval-tokens", str(train.eval_tokens),
        "--core-metric-every", "-1",
        "--sample-every", "-1",
        "--save-every", "-1",
        "--model-tag", run_id,
        "--run", "dummy",
    ]
    if train.tokens > 0:
        command.extend(["--num-iterations", str(train.tokens // train.total_batch_size)])
    else:
        command.extend(["--max-training-seconds", str(train.seconds)])
    return command


def _parse_training_result(log_path: Path) -> dict[str, Any]:
    result = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(TRAIN_RESULT_PREFIX):
            result = json.loads(line[len(TRAIN_RESULT_PREFIX):])
    if result is None:
        raise ValueError(f"training log contains no {TRAIN_RESULT_PREFIX.strip()} record")
    return result


def _extract_training_metrics(log_path: Path, output_path: Path) -> int:
    count = 0
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(TRAIN_STEP_PREFIX):
            append_jsonl(output_path, json.loads(line[len(TRAIN_STEP_PREFIX):]))
            count += 1
    return count


def _state_bytes(config: ResearchConfig) -> int:
    train = config.training
    model_dim = math.ceil((train.depth * 64) / train.head_dim) * train.head_dim
    heads = model_dim // train.head_dim
    pattern = train.window_pattern.upper()
    lengths = []
    for layer in range(train.depth):
        char = pattern[layer % len(pattern)]
        lengths.append(train.sequence_length if char == "L" else train.sliding_window)
    if train.force_final_full:
        lengths[-1] = train.sequence_length
    return int(sum(2 * heads * train.head_dim * 2 * length for length in lengths))


def _frontier_summaries(artifact_root: Path, exclude_run_id: str) -> list[dict[str, Any]]:
    frontier = []
    if not artifact_root.exists():
        return frontier
    for summary_path in artifact_root.glob("*/summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("run_id") != exclude_run_id and summary.get("decision", {}).get("status") in {"frontier", "confirmed"}:
            frontier.append(summary)
    return frontier


def run_experiment(
    root: str | Path,
    config: ResearchConfig,
    artifact_root_override: str | Path | None = None,
    skip_training: bool = False,
    skip_probe: bool = False,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    provenance = git_provenance(repo)
    artifact_root = Path(artifact_root_override or config.run.artifact_root)
    if not artifact_root.is_absolute():
        artifact_root = repo / artifact_root
    run_id = make_run_id(config.run.name, config.run.seed, provenance["commit"])
    run_dir = artifact_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(run_dir / "resolved-config.json", config.to_dict())
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "git": provenance,
        "environment": environment_provenance(),
        "protected_files": protected_fingerprint(repo, config.protection.protected_paths),
    }
    atomic_write_json(run_dir / "manifest.json", manifest)

    if provenance["commit"] is None or provenance["dirty"]:
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "invalid",
            "failure_reason": "research runs require a committed, clean worktree",
            "manifest": str(run_dir / "manifest.json"),
        }
        atomic_write_json(run_dir / "summary.json", summary)
        append_jsonl(artifact_root / "index.jsonl", summary)
        return summary

    training: dict[str, Any] = {}
    if not skip_training:
        log_path = run_dir / "train.log"
        environment = os.environ.copy()
        environment["NANOCHAT_DTYPE"] = config.training.precision
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                _trainer_command(config, run_id), cwd=repo, env=environment,
                stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        if completed.returncode != 0:
            summary = {
                "schema_version": 1, "run_id": run_id, "status": "crash",
                "failure_reason": f"trainer exited with code {completed.returncode}",
                "log": str(log_path),
            }
            atomic_write_json(run_dir / "summary.json", summary)
            append_jsonl(artifact_root / "index.jsonl", summary)
            return summary
        training = _parse_training_result(log_path)
        training["metric_records"] = _extract_training_metrics(log_path, run_dir / "train-metrics.jsonl")

    probe: dict[str, Any] = {}
    if config.memory_probe.enabled and not skip_probe:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        probe = run_memory_probe(
            config.memory_probe, config.training.window_pattern, config.run.seed, device,
            config.training.force_final_full,
        )
        atomic_write_json(run_dir / "memory-probe.json", probe)

    if skip_training or (config.memory_probe.enabled and skip_probe):
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "diagnostic",
            "training": training,
            "memory_probe": probe,
            "manifest": str(run_dir / "manifest.json"),
        }
        atomic_write_json(run_dir / "summary.json", summary)
        append_jsonl(artifact_root / "index.jsonl", summary)
        return summary

    objectives = {
        "val_bpb": float(training.get("val_bpb", float("inf"))),
        "memory_auc": float(probe.get("evaluation", {}).get("memory_auc", 0.0)),
        "update_accuracy": float(probe.get("evaluation", {}).get("update_accuracy", 0.0)),
        "tokens_per_second": float(training.get("median_tokens_per_second", 0.0)),
        "peak_memory_mb": float(training.get("peak_memory_mb", float("inf"))),
        "state_bytes": float(_state_bytes(config)),
    }
    frontier = _frontier_summaries(artifact_root, run_id)
    candidate = {"status": "complete", "run_id": run_id, "objectives": objectives}
    decision = classify_candidate(candidate, frontier, objectives_from_config(config.decision))
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "complete",
        "suite": config.run.suite,
        "objectives": objectives,
        "training": training,
        "memory_probe": probe.get("evaluation", {}),
        "decision": decision,
        "manifest": str(run_dir / "manifest.json"),
    }
    atomic_write_json(run_dir / "summary.json", summary)
    append_jsonl(artifact_root / "index.jsonl", summary)
    return summary


def render_report(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Research comparison", "",
        "| Run | Decision | BPB ↓ | Memory AUC ↑ | Update ↑ | tok/s ↑ | Peak MB ↓ | State MB ↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        obj = summary.get("objectives", {})
        lines.append(
            f"| {summary.get('run_id', '?')} | {summary.get('decision', {}).get('status', summary.get('status', '?'))} "
            f"| {obj.get('val_bpb', float('nan')):.6f} | {obj.get('memory_auc', float('nan')):.4f} "
            f"| {obj.get('update_accuracy', float('nan')):.4f} | {obj.get('tokens_per_second', float('nan')):.0f} "
            f"| {obj.get('peak_memory_mb', float('nan')):.1f} | {obj.get('state_bytes', 0) / 2**20:.2f} |"
        )
    return "\n".join(lines) + "\n"
