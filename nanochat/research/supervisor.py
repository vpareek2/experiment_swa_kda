from __future__ import annotations

import json
from pathlib import Path
import shlex
from typing import Any

from nanochat.common import get_base_dir
from nanochat.research.artifacts import atomic_write_json
from nanochat.research.config import ResearchConfig
from nanochat.research.protected import (
    changed_path_violations,
    changed_paths,
    derive_confirmation_seeds,
    initialize_supervisor,
    sign_summary,
    verify_protected,
)


def verify_candidate(root: str | Path, config: ResearchConfig, base_ref: str, candidate_ref: str) -> dict[str, Any]:
    protected = verify_protected(root)
    paths = changed_paths(root, base_ref, candidate_ref)
    violations = changed_path_violations(paths, config.protection.allowed_paths)
    return {
        "valid": protected["valid"] and not violations,
        "protected": protected,
        "changed_paths": paths,
        "violations": violations,
    }


def sign_result(summary_path: str | Path) -> dict[str, Any]:
    path = Path(summary_path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    signed = {"summary": summary, "signature": sign_summary(summary)}
    atomic_write_json(path.with_name("signed-summary.json"), signed)
    return signed


def sandbox_command(
    image: str,
    source: str | Path,
    output: str | Path,
    config_path: str,
    confirmation: bool = False,
    cache: str | Path | None = None,
    candidate_path: str | None = None,
) -> list[str]:
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    cache_path = Path(cache or get_base_dir()).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    command = [
        "docker", "run", "--rm", "--gpus=all", "--network=none", "--read-only",
        "--security-opt=no-new-privileges", "--cap-drop=ALL",
        "--tmpfs", "/tmp:rw,nosuid,size=8g",
        "--env", "NANOCHAT_BASE_DIR=/nanochat-cache",
        "--mount", f"type=bind,src={source_path},dst=/workspace,readonly",
        "--mount", f"type=bind,src={cache_path},dst=/nanochat-cache,readonly",
        "--mount", f"type=bind,src={checkpoint_path},dst=/nanochat-cache/base_checkpoints",
        "--mount", f"type=bind,src={output_path},dst=/output",
        "--workdir", "/workspace", image,
        "/opt/research-venv/bin/python", "-m", "nanochat.research.cli", "run",
        "--config", config_path, "--artifact-root", "/output",
    ]
    if candidate_path:
        command.extend(["--candidate", candidate_path])
    if confirmation:
        seeds = derive_confirmation_seeds(5)
        command.extend(["--confirmation-seeds", ",".join(str(seed) for seed in seeds)])
    return command


def format_command(command: list[str]) -> str:
    return shlex.join(command)
