from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


def _tool_output(path: str, flag: str) -> str:
    result = subprocess.run([path, flag], text=True, capture_output=True, check=False)
    return (result.stdout + result.stderr).strip()


def select_triton_ptxas() -> str | None:
    """Select a system assembler when Triton's bundled one lacks the GPU target."""
    explicit = os.environ.get("TRITON_PTXAS_PATH")
    if explicit:
        return explicit
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (12, 1):
        return None
    candidates = ["/usr/local/cuda/bin/ptxas", shutil.which("ptxas")]
    for candidate in dict.fromkeys(path for path in candidates if path):
        if "sm_121a" in _tool_output(candidate, "--help"):
            return candidate
    return None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def append_jsonl(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json_bytes(value).decode("utf-8") + "\n")


def _git(repo: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_provenance(repo: str | Path = ".") -> dict[str, Any]:
    root = Path(repo).resolve()
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain=v1")
    diff = _git(root, "diff", "--binary", "HEAD") if commit else None
    return {
        "root": str(root),
        "commit": commit,
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
        "diff_sha256": hashlib.sha256((diff or "").encode()).hexdigest(),
    }


def environment_provenance() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda,
    }
    if cuda:
        selected_ptxas = select_triton_ptxas()
        result.update({
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "device_count": torch.cuda.device_count(),
            "compiled_architectures": torch.cuda.get_arch_list(),
            "triton_ptxas_path": selected_ptxas,
            "triton_ptxas_version": _tool_output(selected_ptxas, "--version") if selected_ptxas else None,
        })
    return result


def make_run_id(name: str, seed: int, commit: str | None = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-") or "run"
    revision = (commit or "uncommitted")[:8]
    return f"{stamp}-{safe_name}-{revision}-s{seed}"


def protected_fingerprint(root: str | Path, paths: Iterable[str]) -> dict[str, str]:
    base = Path(root).resolve()
    files: list[Path] = []
    for raw in paths:
        path = base / raw
        if path.is_dir():
            files.extend(sorted(
                item for item in path.rglob("*")
                if item.is_file() and "__pycache__" not in item.parts and item.suffix not in {".pyc", ".pyo"}
            ))
        elif path.is_file():
            files.append(path)
    return {str(path.relative_to(base)): sha256_file(path) for path in sorted(set(files))}
