from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import subprocess
from typing import Any, Iterable

from nanochat.research.artifacts import atomic_write_json, canonical_json_bytes, protected_fingerprint


def default_state_dir() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "swa-kda-research" / "supervisor"


def initialize_supervisor(root: str | Path, protected_paths: Iterable[str], state_dir: str | Path | None = None) -> dict[str, Any]:
    state = Path(state_dir) if state_dir else default_state_dir()
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    secret_path = state / "secret.key"
    if not secret_path.exists():
        secret_path.write_bytes(secrets.token_bytes(32))
        secret_path.chmod(0o600)
    protected_paths = tuple(protected_paths)
    manifest = {
        "schema_version": 1,
        "protected_paths": list(protected_paths),
        "protected_files": protected_fingerprint(root, protected_paths),
    }
    manifest["fingerprint"] = hashlib.sha256(canonical_json_bytes(manifest["protected_files"])).hexdigest()
    atomic_write_json(state / "protected-manifest.json", manifest)
    confirmation = {
        "schema_version": 1,
        "seed_nonce": secrets.token_hex(32),
        "released": False,
    }
    if not (state / "confirmation.json").exists():
        atomic_write_json(state / "confirmation.json", confirmation)
        (state / "confirmation.json").chmod(0o600)
    return {"state_dir": str(state), **manifest}


def verify_protected(root: str | Path, state_dir: str | Path | None = None) -> dict[str, Any]:
    state = Path(state_dir) if state_dir else default_state_dir()
    manifest_path = state / "protected-manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("supervisor is not initialized")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = protected_fingerprint(root, manifest.get("protected_paths", manifest["protected_files"].keys()))
    expected = manifest["protected_files"]
    changed = sorted(path for path in set(expected) | set(current) if expected.get(path) != current.get(path))
    return {
        "valid": not changed,
        "fingerprint": manifest["fingerprint"],
        "changed": changed,
    }


def derive_confirmation_seeds(count: int, state_dir: str | Path | None = None, context: str = "confirmation-v1") -> list[int]:
    state = Path(state_dir) if state_dir else default_state_dir()
    secret = (state / "secret.key").read_bytes()
    confirmation = json.loads((state / "confirmation.json").read_text(encoding="utf-8"))
    seeds: list[int] = []
    for index in range(count):
        message = f"{context}:{confirmation['seed_nonce']}:{index}".encode()
        digest = hmac.new(secret, message, hashlib.sha256).digest()
        seeds.append(int.from_bytes(digest[:8], "big") & 0x7FFFFFFF)
    return seeds


def sign_summary(summary: dict[str, Any], state_dir: str | Path | None = None) -> str:
    state = Path(state_dir) if state_dir else default_state_dir()
    secret = (state / "secret.key").read_bytes()
    return hmac.new(secret, canonical_json_bytes(summary), hashlib.sha256).hexdigest()


def verify_signature(summary: dict[str, Any], signature: str, state_dir: str | Path | None = None) -> bool:
    return hmac.compare_digest(sign_summary(summary, state_dir), signature)


def changed_paths(root: str | Path, base_ref: str, candidate_ref: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...{candidate_ref}"],
        cwd=root, text=True, capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    return [line for line in result.stdout.splitlines() if line]


def path_allowed(path: str, allowed_paths: Iterable[str]) -> bool:
    return any(path == allowed or (allowed.endswith("/") and path.startswith(allowed)) for allowed in allowed_paths)


def changed_path_violations(paths: Iterable[str], allowed_paths: Iterable[str]) -> list[str]:
    allowed = tuple(allowed_paths)
    return sorted(path for path in paths if not path_allowed(path, allowed))
