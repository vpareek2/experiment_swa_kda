#!/usr/bin/env bash

# Source this file for copy/paste command templates. These functions become
# usable as the corresponding project files are added. Do not store secrets or
# host-specific credentials here.

set -euo pipefail

swa_kda_environment() {
  uv run python - <<'PY'
import platform
import torch

print("machine", platform.machine())
print("torch", torch.__version__)
print("cuda", torch.version.cuda, torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
}

swa_kda_check() {
  uv run pytest -q
  git diff --check
}

swa_kda_cpu_check() {
  NANOCHAT_DTYPE=float32 uv run python -m pytest -q tests/test_research_harness.py
}

swa_kda_config_run() {
  local config_path="${1:?config path required}"
  uv run research doctor --config "${config_path}"
  uv run research run --config "${config_path}"
}
