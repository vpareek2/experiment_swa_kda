"""Ledger-free preflight for staged CUDA candidate worktrees.

This command deliberately runs the coordinator's interpreter and protected
``cuda_worker.py`` against candidate implementation files.  It never imports
or opens the campaign supervisor or ledger.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import io
import tokenize
from typing import Any

from nanochat.research.cuda_config import KdaCudaCampaignConfig
from nanochat.research.cuda_preflight import capture_nsys_cuda_symbols, sanitizer_zero_summary


FORBIDDEN_GENERATED_SUFFIXES = {
    ".a", ".bin", ".cubin", ".dll", ".dylib", ".fatbin", ".o", ".obj", ".so",
}
FORBIDDEN_SOURCE_TOKENS = (
    b"import fla",
    b"from fla",
    b"_run_fla",
    b"_fla_ops",
    b"_fla_causal_conv1d",
    b"_reference_kda",
    b"tests.kda_oracle",
    b"/ref/",
    b"tcgen05",
    b"tmem",
)
PYTHON_REGISTRATION_TOKENS = (b"torch.library", b"from torch import library")
SOURCE_SCAN_SUFFIXES = {".py", ".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp", ".ptx"}
READ_ONLY_CANDIDATE_PATHS = {"nanochat/mixers/cuda_kda/README.md"}


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, errors="replace"
    )
    if check and result.returncode:
        raise ValueError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def _zero_paths(root: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args, "-z"], cwd=root, capture_output=True
    )
    if result.returncode:
        raise ValueError(
            f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace').strip()}"
        )
    return [item.decode(errors="surrogateescape") for item in result.stdout.split(b"\0") if item]


def _allowed(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in prefixes)


def _staged_blob(root: Path, path: str) -> bytes | None:
    result = subprocess.run(["git", "show", f":{path}"], cwd=root, capture_output=True)
    if result.returncode:
        # A staged deletion has no index blob.
        return None
    return result.stdout


def _comment_free_source(path:Path,data:bytes)->bytes:
    if path.suffix==".py":
        try:
            tokens=[token for token in tokenize.tokenize(io.BytesIO(data).readline) if token.type not in (tokenize.COMMENT,tokenize.ENCODING)]
            rendered=tokenize.untokenize(tokens)
            return rendered.encode() if isinstance(rendered,str) else rendered
        except (tokenize.TokenError,IndentationError):
            return data
    text=data.decode("utf-8",errors="replace"); output=[]; index=0; quote=None
    while index<len(text):
        if quote:
            output.append(text[index])
            if text[index]=="\\" and index+1<len(text): output.append(text[index+1]); index+=2; continue
            if text[index]==quote: quote=None
            index+=1; continue
        if text[index] in ("'",'"'): quote=text[index]; output.append(text[index]); index+=1; continue
        if text.startswith("//",index):
            newline=text.find("\n",index); index=len(text) if newline<0 else newline; continue
        if text.startswith("/*",index):
            end=text.find("*/",index+2); index=len(text) if end<0 else end+2; continue
        output.append(text[index]); index+=1
    return "".join(output).encode()


def inspect_staged_candidate(worktree: str | Path, config: KdaCudaCampaignConfig) -> dict[str, Any]:
    """Validate that the executable worktree is exactly its staged candidate snapshot."""
    root = Path(worktree).resolve()
    if not root.is_dir():
        raise ValueError(f"candidate worktree does not exist: {root}")
    if _git(root, "rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        raise ValueError(f"not a Git worktree: {root}")

    staged = _zero_paths(root, "diff", "--cached", "--no-renames", "--name-only")
    unstaged = _zero_paths(root, "diff", "--no-renames", "--name-only")
    untracked = _zero_paths(root, "ls-files", "--others", "--exclude-standard")
    if not staged:
        raise ValueError("candidate checker requires at least one staged source change")
    if unstaged or untracked:
        raise ValueError(
            "candidate worktree must have no unstaged or untracked files; "
            f"unstaged={unstaged}, untracked={untracked}"
        )
    outside = [path for path in staged if not _allowed(path, config.campaign.candidate_paths)]
    if outside:
        raise ValueError(f"staged paths outside candidate scope: {outside}")
    protected=[path for path in staged if path in READ_ONLY_CANDIDATE_PATHS]
    if protected:
        raise ValueError(f"staged paths include protected read-only onboarding files: {protected}")

    allowed_suffixes = set(config.ownership.source_extensions) | {".json"}
    violations: list[str] = []
    receipts: list[dict[str, Any]] = []
    for path in staged:
        suffix = Path(path).suffix.lower()
        blob = _staged_blob(root, path)
        if blob is None:
            receipts.append({"path": path, "status": "deleted", "sha256": None})
            continue
        stage = _git(root, "ls-files", "--stage", "--", path).stdout.split()
        mode = stage[0] if stage else ""
        if mode in {"120000", "160000"}:
            violations.append(f"{path}: symlinks and submodules are forbidden")
        if suffix in FORBIDDEN_GENERATED_SUFFIXES or suffix not in allowed_suffixes:
            violations.append(f"{path}: suffix is outside the CUDA/build-source allowlist")
        lowered = _comment_free_source(Path(path),blob).lower()
        if suffix in SOURCE_SCAN_SUFFIXES:
            found = [token.decode("ascii") for token in FORBIDDEN_SOURCE_TOKENS if token in lowered]
            if suffix == ".py":
                found += [token.decode("ascii") for token in PYTHON_REGISTRATION_TOKENS if token in lowered]
            if found:
                violations.append(f"{path}: forbidden lexical token(s): {sorted(set(found))}")
        receipts.append({
            "path": path,
            "status": "staged",
            "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
        })

    patch = subprocess.run(
        ["git", "diff", "--cached", "--binary", "--no-renames"], cwd=root, capture_output=True
    )
    if patch.returncode:
        raise ValueError("could not read staged candidate patch")
    if len(patch.stdout) > config.campaign.max_patch_bytes:
        violations.append("staged candidate patch exceeds the frozen byte cap")
    if violations:
        raise ValueError("; ".join(violations))
    return {
        "head_sha": _git(root, "rev-parse", "HEAD").stdout.strip(),
        "staged_paths": staged,
        "staged_patch_sha256": hashlib.sha256(patch.stdout).hexdigest(),
        "staged_patch_bytes": len(patch.stdout),
        "sources": receipts,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"status": "invalid", "reason": "worker output is not an object"}
    except (OSError, json.JSONDecodeError) as error:
        return {"status": "invalid", "reason": f"cannot read worker output: {error}"}


def _run_logged(
    command: list[str], cwd: Path, log: Path, timeout: float, env: dict[str, str]
) -> dict[str, Any]:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("wb") as handle:
        try:
            completed = subprocess.run(
                command, cwd=cwd, env={**os.environ, **env}, stdout=handle,
                stderr=subprocess.STDOUT, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "returncode": None, "seconds": time.monotonic() - started}
    return {
        "status": "complete" if completed.returncode == 0 else "invalid",
        "returncode": completed.returncode,
        "seconds": time.monotonic() - started,
    }


def _phase(
    coordinator: Path,
    worktree: Path,
    config_path: Path,
    artifact: Path,
    caches: dict[str, Path],
    lane: str,
    command: str,
    timeout: float,
    expected_symbols: list[str] | None = None,
) -> dict[str, Any]:
    output = artifact / f"{command}.json"
    log = artifact / f"{command}.log"
    worker = coordinator / "nanochat" / "research" / "cuda_worker.py"
    args = [
        sys.executable, str(worker), command,
        "--implementation-root", str(worktree),
        "--backend", "project_cuda",
        "--lane", lane,
        "--config", str(config_path),
        "--output", str(output),
    ]
    env = {
        "PYTHONPATH": str(coordinator) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
        "TORCH_EXTENSIONS_DIR": str(caches["extensions"]),
        "CUDA_CACHE_PATH": str(caches["cuda"]),
    }
    if expected_symbols is None:
        process = _run_logged(args, worktree, log, timeout, env)
        nsys_evidence={}
    else:
        started=time.monotonic()
        try:
            nsys_evidence=capture_nsys_cuda_symbols(args,expected_symbols=expected_symbols,cwd=worktree,env={**os.environ,**env},timeout=timeout)
            process={"status":"complete","returncode":0,"seconds":time.monotonic()-started}
            log.write_text(json.dumps({"profiler_backend":"nsys","kernel_evidence":nsys_evidence},indent=2,sort_keys=True)+"\n")
        except Exception as error:
            nsys_evidence={}; process={"status":"invalid","returncode":None,"seconds":time.monotonic()-started,"reason":f"Nsight profile failed: {type(error).__name__}: {error}"}
            log.write_text(process["reason"]+"\n")
    payload = _read_json(output)
    if process["status"]=="complete" and expected_symbols is not None:
        payload={**payload,"profiler_backend":"nsys","observed_kernel_symbols":sorted(nsys_evidence),"nsys_kernel_evidence":nsys_evidence}
        output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    status = "complete" if process["status"] == "complete" and payload.get("status") == "complete" else process["status"]
    if process["status"] == "complete" and payload.get("status") != "complete":
        status = "invalid"
    return {"name": command, "status": status, "process": process, "result": payload, "log": str(log), "output": str(output)}


def _sanitizer_phase(
    coordinator: Path,
    worktree: Path,
    config_path: Path,
    artifact: Path,
    caches: dict[str, Path],
    lane: str,
    tool: str,
    timeout: float,
) -> dict[str, Any]:
    executable = shutil.which("compute-sanitizer")
    output = artifact / f"sanitizer-{tool}.json"
    log = artifact / f"compute-sanitizer-{tool}.log"
    if executable is None:
        return {"name": f"compute-sanitizer-{tool}", "status": "invalid", "reason": "compute-sanitizer is not available", "log": str(log), "output": str(output)}
    worker = coordinator / "nanochat" / "research" / "cuda_worker.py"
    args = [
        executable, "--tool", tool, "--error-exitcode=99",
        sys.executable, str(worker), "sanitizer-smoke",
        "--implementation-root", str(worktree),
        "--backend", "project_cuda", "--lane", lane,
        "--config", str(config_path), "--output", str(output),
    ]
    env = {
        "PYTHONPATH": str(coordinator) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
        "TORCH_EXTENSIONS_DIR": str(caches["extensions"]),
        "CUDA_CACHE_PATH": str(caches["cuda"]),
    }
    process = _run_logged(args, worktree, log, timeout, env)
    payload = _read_json(output)
    log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    zero_errors = sanitizer_zero_summary(tool,log_text)
    status = "complete" if process["status"] == "complete" and payload.get("status") == "complete" and zero_errors else "invalid"
    if process["status"] == "timeout":
        status = "timeout"
    return {
        "name": f"compute-sanitizer-{tool}", "status": status, "process": process,
        "result": payload, "zero_error_summary": zero_errors, "log": str(log), "output": str(output),
    }


def run_cuda_candidate_check(
    coordinator_root: str | Path,
    config: KdaCudaCampaignConfig,
    *,
    worktree: str | Path,
    lane: str,
    artifact_dir: str | Path | None = None,
    extension_cache: str | Path | None = None,
    cuda_cache: str | Path | None = None,
    sanitizers: bool = False,
) -> dict[str, Any]:
    """Run protected build/runtime/profile preflight without campaign state."""
    if lane not in {"bootstrap", "migration", "optimization"}:
        raise ValueError(f"unknown CUDA candidate lane: {lane}")
    coordinator = Path(coordinator_root).resolve()
    candidate = Path(worktree).resolve()
    if candidate == coordinator:
        raise ValueError("--worktree must name a candidate worktree, not the coordinator")
    worker = coordinator / "nanochat" / "research" / "cuda_worker.py"
    if not worker.is_file():
        raise ValueError(f"protected CUDA worker is missing: {worker}")

    staged = inspect_staged_candidate(candidate, config)
    if artifact_dir is None:
        artifact = Path(tempfile.mkdtemp(prefix="nanochat-cuda-candidate-")).resolve()
    else:
        artifact = Path(artifact_dir).expanduser().resolve()
        if artifact.exists() and any(artifact.iterdir()):
            raise ValueError(f"artifact directory must be new or empty: {artifact}")
        artifact.mkdir(parents=True, exist_ok=True)
    cache_paths = {
        "extensions": Path(extension_cache).expanduser().resolve() if extension_cache else artifact / "extension-cache",
        "cuda": Path(cuda_cache).expanduser().resolve() if cuda_cache else artifact / "cuda-cache",
    }
    for label, path in {"artifact": artifact, **cache_paths}.items():
        if _inside(path, candidate):
            raise ValueError(f"{label} path must be outside the candidate worktree: {path}")
        path.mkdir(parents=True, exist_ok=True)

    resolved_config = artifact / "resolved-config.json"
    resolved_config.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {
        "schema": "kda_cuda_candidate_check", "schema_version": 1,
        "status": "running", "lane": lane,
        "coordinator_root": str(coordinator), "worktree": str(candidate),
        "interpreter": sys.executable, "protected_worker": str(worker),
        "artifact_dir": str(artifact),
        "extension_cache": str(cache_paths["extensions"]), "cuda_cache": str(cache_paths["cuda"]),
        "staged": staged, "sanitizers_requested": sanitizers,
        "ledger_accessed": False,
    }
    (artifact / "invocation.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    phases: list[dict[str, Any]] = []
    runtime = _phase(
        coordinator, candidate, resolved_config, artifact, cache_paths, lane,
        "runtime-audit", config.correctness.runtime_audit_timeout_seconds,
    )
    if lane == "bootstrap" and runtime.get("result", {}).get("provenance", {}).get("selective_ptx"):
        runtime["status"] = "invalid"
        runtime["reason"] = "bootstrap forbids selective PTX"
    phases.append(runtime)
    if runtime["status"] == "complete":
        expected_symbols=[symbol for component in runtime.get("result",{}).get("provenance",{}).get("components",{}).values() if component.get("owner")=="project" for symbol in component.get("kernel_symbols",[])]
        profile = _phase(
            coordinator, candidate, resolved_config, artifact, cache_paths, lane,
            "profile-audit", config.correctness.runtime_audit_timeout_seconds,
            expected_symbols=expected_symbols,
        )
        profile_output = Path(profile["output"])
        if profile_output.exists() and profile_output.stat().st_size > config.kernel_gates.profile_max_bytes:
            profile["status"] = "invalid"
            profile["reason"] = "ownership profile exceeds profile_max_bytes"
        phases.append(profile)
    else:
        phases.append({"name": "profile-audit", "status": "skipped", "reason": "runtime audit did not complete"})

    if sanitizers:
        if all(phase["status"] == "complete" for phase in phases):
            for tool in config.correctness.sanitizer_tools:
                phases.append(_sanitizer_phase(
                    coordinator, candidate, resolved_config, artifact, cache_paths,
                    lane, tool, config.correctness.compute_sanitizer_timeout_seconds,
                ))
        else:
            phases.append({"name": "compute-sanitizers", "status": "skipped", "reason": "build/runtime/profile preflight did not complete"})

    complete = all(phase["status"] == "complete" for phase in phases)
    result = {**metadata, "status": "complete" if complete else "invalid", "phases": phases}
    summary = artifact / "summary.json"
    result["summary_path"] = str(summary)
    summary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result
