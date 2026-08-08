#!/usr/bin/env python3
"""Fast, non-conclusion-bearing development loop for project-owned KDA CUDA.

Level 1 retains the protected runtime audit and runs isolated production-shape
microbenchmarks. Level 2 only records and prints one ordered trainer block; this
program never launches training.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import signal
import statistics
import subprocess
import sys
import uuid
from typing import Any, Sequence

COORDINATOR = Path(__file__).resolve().parents[1]
BASE_CONFIG = COORDINATOR / "configs/research/kda_cuda_ownership.toml"
LENGTHS = (256, 1024, 4096)
OPERATIONS = ("chunk_forward", "chunk_forward_backward")
PRODUCTION_SHAPE = {"batch": 2, "heads": 3, "key_dim": 128, "value_dim": 128}
WARMUP_ITERATIONS = 3
TIMED_ITERATIONS = 10
SOURCE_SUFFIXES = {".py", ".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp", ".ptx"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_worktree", type=Path)
    parser.add_argument("candidate_worktree", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--level2-order", choices=("baseline-first", "candidate-first"),
        default="baseline-first",
    )
    return parser.parse_args(argv)


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def create_output_dir(path: Path) -> Path:
    """Create a fresh artifact directory, refusing every collision."""
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args], text=True, capture_output=True, check=False
    )


def _worktree(path: Path) -> Path:
    root = path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"worktree is not a directory: {root}")
    result = _run_git(root, "rev-parse", "--show-toplevel")
    if result.returncode or Path(result.stdout.strip()).resolve() != root:
        raise ValueError(f"path must be the exact root of a Git worktree: {root}")
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes(root: Path) -> dict[str, str]:
    paths: set[Path] = set()
    for relative in ("nanochat/gpt.py", "nanochat/mixers/kda.py", "scripts/base_train.py"):
        path = root / relative
        if path.is_file():
            paths.add(path)
    source_root = root / "nanochat/mixers/cuda_kda"
    if source_root.is_dir():
        paths.update(
            path for path in source_root.rglob("*")
            if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
        )
    return {str(path.relative_to(root)): _sha256(path) for path in sorted(paths)}


def capture_worktree(root: Path, destination: Path) -> dict[str, Any]:
    destination.mkdir()
    commands = {
        "head": ("rev-parse", "HEAD"),
        "tree": ("rev-parse", "HEAD^{tree}"),
        "status": ("status", "--short", "--untracked-files=all"),
        "diff": ("diff", "--binary", "HEAD", "--"),
        "diff_stat": ("diff", "--stat", "HEAD", "--"),
    }
    results: dict[str, Any] = {}
    for name, args in commands.items():
        result = _run_git(root, *args)
        (destination / f"git-{name}.txt").write_text(result.stdout)
        (destination / f"git-{name}.stderr.txt").write_text(result.stderr)
        results[name] = {
            "command": ["git", "-C", str(root), *args],
            "returncode": result.returncode,
            "sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        }
    hashes = _source_hashes(root)
    _json(destination / "source-hashes.json", hashes)
    results.update({
        "root": str(root),
        "commit": (destination / "git-head.txt").read_text().strip(),
        "source_hashes": hashes,
    })
    return results


def resolved_config() -> tuple[dict[str, Any], Any]:
    """Resolve the protected config with only the authorized Level-1 budget."""
    from nanochat.research.cuda_config import (
        cuda_campaign_config_from_dict,
        load_cuda_campaign_config,
    )

    raw = asdict(load_cuda_campaign_config(BASE_CONFIG))
    raw["kernel_gates"].update({
        "sequence_lengths": list(LENGTHS),
        "warmup_iterations": 3,
        "timed_iterations": 10,
        "timeout_seconds": 300.0,
    })
    config = cuda_campaign_config_from_dict(raw)
    return raw, config


def production_shape(config: Any) -> dict[str, Any]:
    """Derive the KDA kernel shape used by the exact protected trainer."""
    trainer = COORDINATOR / "scripts/base_train.py"
    tree = ast.parse(trainer.read_text(), filename=str(trainer))
    aspect_ratio = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        if isinstance(node.args[0], ast.Constant) and node.args[0].value == "--aspect-ratio":
            for keyword in node.keywords:
                if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                    aspect_ratio = int(keyword.value.value)
    if aspect_ratio is None:
        raise ValueError("could not derive protected trainer --aspect-ratio default")
    measurement = config.measurement
    base_dim = measurement.depth * aspect_ratio
    model_dim = ((base_dim + measurement.head_dim - 1) // measurement.head_dim) * measurement.head_dim
    heads = model_dim // measurement.head_dim
    return {
        "B": measurement.device_batch_size, "H": heads,
        "K": measurement.head_dim, "V": measurement.head_dim,
        "dtype": "bfloat16", "depth": measurement.depth,
        "aspect_ratio": aspect_ratio, "model_dim": model_dim,
        "derivation": "model_dim=ceil(depth*aspect_ratio/head_dim)*head_dim; H=model_dim/head_dim",
        "source": str(trainer), "source_sha256": _sha256(trainer),
    }


def _cache_env(output: Path, label: str, *, pythonpath: Path = COORDINATOR) -> dict[str, str]:
    cache = output / "caches" / label
    values = {
        "TORCH_EXTENSIONS_DIR": cache / "torch-extensions",
        "CUDA_CACHE_PATH": cache / "cuda",
        "TRITON_CACHE_DIR": cache / "triton",
        "XDG_CACHE_HOME": cache / "xdg",
        "PYTHONPYCACHEPREFIX": cache / "pycache",
    }
    for path in values.values():
        path.mkdir(parents=True, exist_ok=True)
    return {
        **{name: str(path) for name, path in values.items()},
        "PYTHONPATH": str(pythonpath),
        "PYTHONNOUSERSITE": "1",
        "TORCH_COMPILE_DISABLE": "1",
        "NANOCHAT_DTYPE": "bfloat16",
        "FLA_FLASH_KDA": "0",
        "FLA_TILELANG": "0",
    }


def _communicate_bounded(
    argv: list[str], cwd: Path, env: dict[str, str], timeout: float,
) -> tuple[dict[str, Any], str, str]:
    """Capture a subprocess, terminating its whole fresh session on timeout."""
    process = subprocess.Popen(
        argv, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        status = "complete" if process.returncode == 0 else "invalid"
        return {"status": status, "returncode": process.returncode}, stdout, stderr
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=5.0)
            termination = "SIGTERM"
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            termination = "SIGKILL"
        return {
            "status": "invalid", "reason": "worker timeout",
            "timeout_seconds": timeout, "termination": termination,
            "returncode": process.returncode,
        }, stdout, stderr


def run_worker(
    command: str,
    root: Path,
    config_path: Path,
    artifact: Path,
    cache_label: str,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one protected worker process and retain all process evidence."""
    artifact.mkdir()
    python = COORDINATOR / ".venv/bin/python"
    if not python.is_file():
        raise RuntimeError(f"coordinator interpreter is missing: {python}")
    payload_path = artifact / "payload.json"
    argv = [
        str(python), str(COORDINATOR / "nanochat/research/cuda_worker.py"), command,
        "--implementation-root", str(root), "--backend", "project_cuda",
        "--lane", "optimization", "--config", str(config_path),
        "--output", str(payload_path),
    ]
    overrides = _cache_env(artifact.parent, cache_label)
    inherited = {
        name: os.environ[name] for name in
        ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "PATH") if name in os.environ
    }
    _json(artifact / "invocation.json", {
        "argv": argv,
        "cwd": str(root),
        "environment_overrides": overrides,
        "inherited_runtime_environment": inherited,
        "timeout_seconds": timeout,
    })
    env = os.environ.copy()
    env.update(overrides)
    process, stdout, stderr = _communicate_bounded(argv, root, env, timeout)
    (artifact / "stdout.log").write_text(stdout)
    (artifact / "stderr.log").write_text(stderr)
    (artifact / "worker.log").write_text(
        "===== STDOUT =====\n" + stdout + "\n===== STDERR =====\n" + stderr
    )
    try:
        payload = json.loads(payload_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        payload = {}
        process.update({"status": "invalid", "payload_error": f"{type(error).__name__}: {error}"})
    if payload.get("status") != "complete":
        process.update({"status": "invalid", "worker_status": payload.get("status", "missing")})
    _json(artifact / "process.json", process)
    return process, payload


def _production_inputs(torch: Any, length: int, *, gradients: bool) -> tuple[Any, ...]:
    shape = PRODUCTION_SHAPE
    torch.manual_seed(4100 + length + int(gradients))
    args = (
        torch.randn(shape["batch"], length, shape["heads"], shape["key_dim"], device="cuda", dtype=torch.bfloat16),
        torch.randn(shape["batch"], length, shape["heads"], shape["key_dim"], device="cuda", dtype=torch.bfloat16),
        torch.randn(shape["batch"], length, shape["heads"], shape["value_dim"], device="cuda", dtype=torch.bfloat16),
        torch.randn(shape["batch"], length, shape["heads"], shape["key_dim"], device="cuda", dtype=torch.bfloat16),
        torch.randn(shape["batch"], length, shape["heads"], device="cuda", dtype=torch.bfloat16),
        torch.zeros(shape["heads"], device="cuda", dtype=torch.float32),
        torch.zeros(shape["heads"] * shape["key_dim"], device="cuda", dtype=torch.float32),
    )
    if gradients:
        for tensor in args:
            tensor.requires_grad_(True)
    return args


def _measure_production_operation(torch: Any, implementation: Any, length: int, operation: str) -> dict[str, Any]:
    gradients = operation == "chunk_forward_backward"
    values = _production_inputs(torch, length, gradients=gradients)

    def clear_gradients() -> None:
        if gradients:
            for value in values:
                value.grad = None

    def one() -> None:
        if gradients:
            output, _ = implementation.kda(
                *values, mode="project_chunk", allow_fallback=False,
            )
            output.float().square().mean().backward()
        else:
            with torch.no_grad():
                implementation.kda(
                    *values, mode="project_chunk", allow_fallback=False,
                )

    for _ in range(WARMUP_ITERATIONS):
        clear_gradients()
        one()
    torch.cuda.synchronize()
    elapsed_ms: list[float] = []
    peaks: list[int] = []
    for _ in range(TIMED_ITERATIONS):
        clear_gradients()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        one()
        end.record()
        end.synchronize()
        elapsed_ms.append(float(start.elapsed_time(end)))
        peaks.append(int(torch.cuda.max_memory_allocated()))
    return {
        "operation": operation,
        "length": length,
        **PRODUCTION_SHAPE,
        "warmup_iterations": WARMUP_ITERATIONS,
        "timed_iterations": TIMED_ITERATIONS,
        "elapsed_ms": elapsed_ms,
        "median_ms": statistics.median(elapsed_ms),
        "peak_allocated_bytes_samples": peaks,
        "peak_allocated_bytes": max(peaks),
    }


def production_worker(implementation_root: Path, output: Path) -> int:
    """GPU-only helper. A process imports exactly one implementation root."""
    payload: dict[str, Any] = {
        "status": "running", "backend": "project_cuda",
        "shape": dict(PRODUCTION_SHAPE), "lengths": list(LENGTHS),
        "warmup_iterations": WARMUP_ITERATIONS,
        "timed_iterations": TIMED_ITERATIONS, "microbenchmarks": [],
    }
    try:
        root = implementation_root.resolve(strict=True)
        if Path.cwd().resolve() != root:
            raise RuntimeError("production helper cwd must equal its implementation root")
        import torch
        implementation = importlib.import_module("nanochat.mixers.kda")
        module_path = Path(implementation.__file__).resolve()
        if root != module_path and root not in module_path.parents:
            raise RuntimeError(f"loaded KDA dispatcher outside implementation root: {module_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        implementation.prepare_kda_backend("project_cuda")
        provenance = implementation.kda_backend_provenance()
        for component in ("chunk_forward", "chunk_backward"):
            owner = provenance.get("components", {}).get(component, {}).get("owner")
            if owner != "project":
                raise RuntimeError(f"pure project_cuda benchmark requires project-owned {component}")
        implementation.reset_project_runtime_events()
        for length in LENGTHS:
            for operation in OPERATIONS:
                before = len(implementation.project_runtime_events())
                row = _measure_production_operation(torch, implementation, length, operation)
                events = implementation.project_runtime_events()[before:]
                if not events or any(event.get("backend") != "project" for event in events):
                    raise RuntimeError(f"non-project runtime event for {operation} T={length}: {events}")
                row["runtime_events"] = events
                payload["microbenchmarks"].append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
        payload["pure_project_cuda"] = True
        payload["status"] = "complete"
        _json(output, payload)
        return 0
    except BaseException as error:
        import traceback
        payload.update({"status": "invalid", "reason": f"{type(error).__name__}: {error}"})
        _json(output, payload)
        traceback.print_exc()
        return 1


def run_production_benchmark(
    root: Path, artifact: Path, cache_label: str, timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact.mkdir()
    python = COORDINATOR / ".venv/bin/python"
    payload_path = artifact / "payload.json"
    argv = [
        str(python), str(Path(__file__).resolve()), "_production-worker",
        str(root), str(payload_path),
    ]
    overrides = _cache_env(artifact.parent, cache_label, pythonpath=root)
    inherited = {
        name: os.environ[name] for name in
        ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "PATH") if name in os.environ
    }
    _json(artifact / "invocation.json", {
        "argv": argv, "cwd": str(root), "environment_overrides": overrides,
        "inherited_runtime_environment": inherited, "timeout_seconds": timeout,
        "isolated_implementation_root": str(root),
    })
    env = os.environ.copy()
    env.update(overrides)
    process, stdout, stderr = _communicate_bounded(argv, root, env, timeout)
    (artifact / "stdout.log").write_text(stdout)
    (artifact / "stderr.log").write_text(stderr)
    (artifact / "worker.log").write_text(
        "===== STDOUT =====\n" + stdout + "\n===== STDERR =====\n" + stderr
    )
    try:
        payload = json.loads(payload_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        payload = {}
        process.update({"status": "invalid", "payload_error": f"{type(error).__name__}: {error}"})
    if payload.get("status") != "complete" or not payload.get("pure_project_cuda"):
        process.update({"status": "invalid", "worker_status": payload.get("status", "missing")})
    _json(artifact / "process.json", process)
    return process, payload


def _indexed_rows(payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    if payload.get("status") != "complete":
        raise ValueError(f"benchmark payload is not complete: {payload.get('status', 'missing')}")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    rows = payload.get("microbenchmarks")
    if not isinstance(rows, list):
        raise ValueError("microbenchmarks must be a list")
    for row in rows:
        try:
            key = (str(row["operation"]), int(row["length"]))
            median = float(row["median_ms"])
            peak = int(row["peak_allocated_bytes"])
            shape = {name: int(row[name]) for name in PRODUCTION_SHAPE}
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"invalid microbenchmark row: {row!r}") from error
        if key[0] not in OPERATIONS or key[1] not in LENGTHS:
            raise ValueError(f"unexpected microbenchmark row: {key}")
        if not math.isfinite(median) or median <= 0.0 or peak <= 0:
            raise ValueError(f"non-finite/non-positive latency or memory for {key}")
        if shape != PRODUCTION_SHAPE:
            raise ValueError(f"non-production shape for {key}: {shape}")
        if key in indexed:
            raise ValueError(f"duplicate microbenchmark row: {key}")
        indexed[key] = row
    return indexed


def compare_microbenchmarks(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Apply the Level-1 production-shape latency and memory gates."""
    base, cand = _indexed_rows(baseline), _indexed_rows(candidate)
    required = {(operation, length) for operation in OPERATIONS for length in LENGTHS}
    missing = sorted((required - base.keys()) | (required - cand.keys()))
    if missing:
        raise ValueError(f"missing required paired rows: {missing}")
    rows = []
    for operation, length in sorted(required, key=lambda item: (item[1], item[0])):
        base_row, cand_row = base[(operation, length)], cand[(operation, length)]
        baseline_ms, candidate_ms = float(base_row["median_ms"]), float(cand_row["median_ms"])
        baseline_memory = int(base_row["peak_allocated_bytes"])
        candidate_memory = int(cand_row["peak_allocated_bytes"])
        latency_ratio = candidate_ms / baseline_ms
        memory_ratio = candidate_memory / baseline_memory
        rows.append({
            "operation": operation, "length": length,
            "baseline_median_ms": baseline_ms, "candidate_median_ms": candidate_ms,
            "candidate_over_baseline": latency_ratio,
            "improvement_fraction": 1.0 - latency_ratio,
            "regression_fraction": max(0.0, latency_ratio - 1.0),
            "baseline_peak_allocated_bytes": baseline_memory,
            "candidate_peak_allocated_bytes": candidate_memory,
            "candidate_memory_over_baseline": memory_ratio,
            "memory_regression_fraction": max(0.0, memory_ratio - 1.0),
        })
    lookup = {(row["operation"], row["length"]): row for row in rows}
    target = lookup[("chunk_forward_backward", 4096)]
    target_ok = target["candidate_over_baseline"] <= 0.97
    regression_ok = all(row["candidate_over_baseline"] <= 1.05 for row in rows)
    memory_ok = all(row["candidate_memory_over_baseline"] <= 1.03 for row in rows)
    advance = target_ok and regression_ok and memory_ok
    return {
        "shape": dict(PRODUCTION_SHAPE), "rows": rows,
        "t4096_forward_backward": {
            "improvement_fraction": target["improvement_fraction"],
            "meets_three_percent": target_ok,
        },
        "important_regression_limit_fraction": 0.05,
        "important_regressions_within_limit": regression_ok,
        "memory_regression_limit_fraction": 0.03,
        "memory_regressions_within_limit": memory_ok,
        "level1_decision": "advance" if advance else "do_not_advance",
        "advance": advance,
    }


def level2_plan(
    baseline: Path, candidate: Path, output: Path, config: Any,
    order: str = "baseline-first",
) -> dict[str, Any]:
    """Build, but never execute, one ordered protected 4k trainer block."""
    from nanochat.research.cuda_supervisor import _trainer_command

    roots = {"baseline": baseline, "candidate": candidate}
    labels = ["baseline", "candidate"] if order == "baseline-first" else ["candidate", "baseline"]
    if order not in {"baseline-first", "candidate-first"}:
        raise ValueError(f"unknown Level-2 order: {order}")
    namespace = f"kda-cuda-development-{uuid.uuid4().hex}"
    python = str(COORDINATOR / ".venv/bin/python")
    entries = []
    for label in labels:
        root = roots[label]
        model_tag = f"{namespace}-{label}"
        command = _trainer_command(config, "project_cuda", model_tag)
        command[0] = python
        env = {
            "TORCH_COMPILE_DISABLE": "1", "NANOCHAT_DTYPE": "bfloat16",
            "TORCH_EXTENSIONS_DIR": str(output / "caches" / f"trainer-{label}"),
            "FLA_FLASH_KDA": "0", "FLA_TILELANG": "0",
        }
        entries.append({
            "label": label, "cwd": str(root), "argv": command,
            "environment": env, "model_tag": model_tag,
        })
    plan = {
        "status": "not_run", "training_launched": False,
        "requested_order": order, "order": labels, "namespace": namespace,
        "sequence_length": 4096, "backend": "project_cuda", "block_count": 1,
        "commands": entries,
    }
    _json(output / "level2-plan.json", plan)
    for entry in entries:
        rendered_env = " ".join(f"{key}={shlex.quote(value)}" for key, value in entry["environment"].items())
        rendered_command = shlex.join(entry["argv"])
        print(f"[{entry['label']}] cd {shlex.quote(entry['cwd'])} && env {rendered_env} {rendered_command}")
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    baseline = _worktree(args.baseline_worktree)
    candidate = _worktree(args.candidate_worktree)
    output = create_output_dir(args.output_dir)

    manifest: dict[str, Any] = {
        "schema": "kda_cuda_development", "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running", "conclusion_bearing": False,
        "quality_not_evaluated": True,
        "coordinator": str(COORDINATOR),
        "coordinator_python": str(COORDINATOR / ".venv/bin/python"),
        "platform": platform.platform(), "python": sys.version,
        "protocol": {
            "lengths": list(LENGTHS), "warmups": WARMUP_ITERATIONS,
            "timed": TIMED_ITERATIONS, "backend": "project_cuda",
            "production_shape": dict(PRODUCTION_SHAPE),
            "level2_order": args.level2_order,
        },
    }
    _json(output / "manifest.json", manifest)
    try:
        manifest["worktrees"] = {
            "baseline": capture_worktree(baseline, output / "baseline-provenance"),
            "candidate": capture_worktree(candidate, output / "candidate-provenance"),
        }
        raw_config, config = resolved_config()
        config_path = output / "resolved-config.json"
        _json(config_path, raw_config)
        manifest["base_config"] = {"path": str(BASE_CONFIG), "sha256": _sha256(BASE_CONFIG)}
        manifest["resolved_config_sha256"] = _sha256(config_path)
        _json(output / "manifest.json", manifest)

        derived_shape = production_shape(config)
        expected = {
            "B": PRODUCTION_SHAPE["batch"], "H": PRODUCTION_SHAPE["heads"],
            "K": PRODUCTION_SHAPE["key_dim"], "V": PRODUCTION_SHAPE["value_dim"],
        }
        actual = {name: derived_shape[name] for name in expected}
        if actual != expected:
            raise ValueError(f"protected trainer shape changed: expected {expected}, found {actual}")
        manifest["production_kernel_shape"] = {**dict(PRODUCTION_SHAPE), **derived_shape}
        _json(output / "manifest.json", manifest)
        audit_process, audit = run_worker(
            "runtime-audit", candidate, config_path, output / "candidate-runtime-audit",
            "candidate-audit", config.correctness.runtime_audit_timeout_seconds,
        )
        base_process, base_payload = run_production_benchmark(
            baseline, output / "baseline-production-microbenchmark",
            "baseline-production-microbenchmark", config.kernel_gates.timeout_seconds,
        )
        cand_process, cand_payload = run_production_benchmark(
            candidate, output / "candidate-production-microbenchmark",
            "candidate-production-microbenchmark", config.kernel_gates.timeout_seconds,
        )
        plan = level2_plan(baseline, candidate, output, config, args.level2_order)
        processes_ok = all(item.get("status") == "complete" for item in
                           (audit_process, base_process, cand_process))
        try:
            comparison = compare_microbenchmarks(base_payload, cand_payload)
        except ValueError as error:
            comparison = {"advance": False, "level1_decision": "invalid", "reason": str(error), "rows": []}
        valid = processes_ok and comparison["level1_decision"] != "invalid"
        if not valid:
            comparison["advance"] = False
            comparison["level1_decision"] = "invalid"
        summary = {
            "status": "complete" if valid else "invalid",
            "conclusion_bearing": False, "quality_not_evaluated": True,
            "quality_evaluation": "not_evaluated",
            "production_kernel_shape": manifest["production_kernel_shape"],
            "runtime_audit_status": audit.get("status", "missing"),
            "processes": {"candidate_runtime_audit": audit_process,
                          "baseline_production_microbenchmark": base_process,
                          "candidate_production_microbenchmark": cand_process},
            "comparison": comparison,
            "level2": plan,
        }
        _json(output / "summary.json", summary)
        manifest.update({"status": summary["status"], "summary": "summary.json"})
        _json(output / "manifest.json", manifest)
        return 0 if valid else 1
    except BaseException as error:
        manifest.update({"status": "invalid", "failure": f"{type(error).__name__}: {error}"})
        _json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "_production-worker":
        raise SystemExit(production_worker(Path(sys.argv[2]), Path(sys.argv[3])))
    raise SystemExit(main())
