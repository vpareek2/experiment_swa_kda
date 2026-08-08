"""Reproducible, fail-closed builds for project-owned CUDA extensions.

The helper intentionally loads a shared library rather than a Python module.
Native sources must register their operators with ``TORCH_LIBRARY``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Sequence

_TARGET_ARCH = "12.1"
_SM121 = re.compile(r"(?<![A-Za-z0-9_])sm_121(?![A-Za-z0-9_])", re.IGNORECASE)


def _content_name(
    prefix: str,
    sources: Sequence[Path],
    extra_cflags: Sequence[str],
    extra_cuda_cflags: Sequence[str],
    extra_ldflags: Sequence[str],
) -> str:
    import torch

    safe_prefix = re.sub(r"[^A-Za-z0-9_]", "_", prefix).strip("_") or "nanochat_cuda"
    manifest = {
        "schema": "nanochat-cuda-extension/1",
        "target_arch": _TARGET_ARCH,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "extra_cflags": list(extra_cflags),
        "extra_cuda_cflags": list(extra_cuda_cflags),
        "extra_ldflags": list(extra_ldflags),
        "sources": [
            {"name": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            for source in sources
        ],
    }
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{safe_prefix}_{digest[:20]}"


def _mapped_library(build_directory: Path, extension_name: str) -> Path:
    matches = sorted(build_directory.glob(f"{extension_name}*.so"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {extension_name} shared library in {build_directory}, found {len(matches)}"
        )
    library = matches[0].resolve()
    if not library.is_file() or library.suffix != ".so":
        raise RuntimeError(f"CUDA extension output is not a shared library: {library}")
    try:
        mappings = Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise RuntimeError("cannot verify the loaded CUDA library in /proc/self/maps") from error
    mapped_paths = {
        line.split(maxsplit=5)[-1].removesuffix(" (deleted)")
        for line in mappings.splitlines()
        if len(line.split(maxsplit=5)) == 6
    }
    if str(library) not in mapped_paths:
        raise RuntimeError(f"CUDA extension was built but is not mapped in this process: {library}")
    return library


def _verify_sm121(library: Path) -> None:
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        raise RuntimeError("cuobjdump is required to verify SM121 device code")
    result = subprocess.run(
        [cuobjdump, "--list-elf", str(library)],
        text=True,
        capture_output=True,
        check=False,
    )
    evidence = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(f"cuobjdump failed for {library}: {evidence.strip()}")
    if _SM121.search(evidence) is None:
        raise RuntimeError(f"cuobjdump found no SM121 cubin in {library}")


def _compiler_command(build_directory: Path) -> str:
    ninja = shutil.which("ninja")
    if ninja is None:
        raise RuntimeError("ninja is required to record the CUDA compiler command")
    result = subprocess.run(
        [ninja, "-t", "commands"],
        cwd=build_directory,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot recover CUDA compiler command: {(result.stdout + result.stderr).strip()}")
    commands = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and ("nvcc" in line or (".cu" in line and "-gencode" in line))
    ]
    if not commands:
        raise RuntimeError("build receipt has no CUDA compiler command")
    command = "\n".join(dict.fromkeys(commands))
    if "compute_121" not in command or "sm_121" not in command:
        raise RuntimeError("CUDA compiler command does not target compute_121/sm_121")
    return command


def build_cuda_extension(
    sources: Sequence[str | Path],
    *,
    name: str = "nanochat_cuda",
    cache_dir: str | Path | None = None,
    extra_cflags: Sequence[str] = (),
    extra_cuda_cflags: Sequence[str] = (),
    extra_ldflags: Sequence[str] = (),
    verbose: bool = False,
) -> dict[str, object]:
    """Build and load a content-addressed SM121 ``TORCH_LIBRARY`` extension.

    The returned receipt deliberately has exactly the four keys consumed by the
    protected CUDA ownership audit. ``source_paths`` retain the caller's path
    spelling so reviewed project-relative paths can be bound into provenance.
    """
    if not sources:
        raise ValueError("at least one CUDA extension source is required")
    declared_sources = [str(Path(source)) for source in sources]
    resolved_sources = [Path(source).expanduser().resolve() for source in sources]
    if len({source.name for source in resolved_sources}) != len(resolved_sources):
        raise ValueError("CUDA extension source basenames must be unique")
    for source in resolved_sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    if not any(source.suffix.lower() == ".cu" for source in resolved_sources):
        raise ValueError("a native .cu translation unit is required")

    import torch
    from torch.utils.cpp_extension import get_default_build_root, load

    extension_name = _content_name(
        name, resolved_sources, extra_cflags, extra_cuda_cflags, extra_ldflags
    )
    configured_cache = cache_dir if cache_dir is not None else os.environ.get("TORCH_EXTENSIONS_DIR")
    cache_root = Path(configured_cache or get_default_build_root()).expanduser().resolve()
    build_directory = cache_root / extension_name
    build_directory.mkdir(parents=True, exist_ok=True)

    # PyTorch derives -gencode flags from this variable. Overwrite, rather than
    # append, so ambient developer settings cannot silently make a non-SM121 build.
    os.environ["TORCH_CUDA_ARCH_LIST"] = _TARGET_ARCH
    load(
        name=extension_name,
        sources=[str(source) for source in resolved_sources],
        extra_cflags=list(extra_cflags),
        extra_cuda_cflags=list(extra_cuda_cflags),
        extra_ldflags=list(extra_ldflags),
        build_directory=str(build_directory),
        with_cuda=True,
        is_python_module=False,
        verbose=verbose,
    )
    library = _mapped_library(build_directory, extension_name)
    _verify_sm121(library)
    compiler_command = _compiler_command(build_directory)

    receipt: dict[str, object] = {
        "library_path": str(library),
        "source_paths": declared_sources,
        "compiler_command": compiler_command,
        "target_arch": _TARGET_ARCH,
    }
    if set(receipt) != {"library_path", "source_paths", "compiler_command", "target_arch"}:
        raise AssertionError("internal build receipt schema changed")
    return receipt
