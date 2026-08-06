"""CUDA toolchain compatibility helpers shared by training and research runs."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import torch


SM121_TARGET = "sm_121a"


def tool_output(path: str, flag: str) -> str:
    result = subprocess.run([path, flag], text=True, capture_output=True, check=False)
    return (result.stdout + result.stderr).strip()


def ptxas_supports(path: str, target: str = SM121_TARGET) -> bool:
    candidate = Path(path)
    return candidate.is_file() and target in tool_output(str(candidate), "--help")


def select_triton_ptxas() -> str | None:
    """Select an assembler that supports the active GPU's Triton target."""

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (12, 1):
        return os.environ.get("TRITON_PTXAS_PATH")

    explicit = os.environ.get("TRITON_PTXAS_PATH")
    candidates = [explicit, "/usr/local/cuda/bin/ptxas", shutil.which("ptxas")]
    for candidate in dict.fromkeys(path for path in candidates if path):
        if ptxas_supports(candidate):
            return candidate
    return None


def configure_triton_ptxas(*, required: bool = False) -> str | None:
    """Configure Triton before compilation, failing closed for SM121 when requested."""

    selected = select_triton_ptxas()
    capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    if capability >= (12, 1) and selected is None and required:
        explicit = os.environ.get("TRITON_PTXAS_PATH")
        detail = f" configured path {explicit!r}" if explicit else ""
        raise RuntimeError(
            f"SM {capability[0]}.{capability[1]} requires a Triton assembler with "
            f"{SM121_TARGET} support;{detail} is unavailable. Install CUDA 13.1+ or set "
            "TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas."
        )
    if selected is not None:
        os.environ["TRITON_PTXAS_PATH"] = selected
    return selected


__all__ = [
    "SM121_TARGET",
    "configure_triton_ptxas",
    "ptxas_supports",
    "select_triton_ptxas",
    "tool_output",
]
