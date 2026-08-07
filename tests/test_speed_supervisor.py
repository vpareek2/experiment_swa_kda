from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from nanochat.research.config import ConfigError, load_config, validate_config
from nanochat.research.speed_supervisor import initialize, intake, summary

ROOT = Path(__file__).resolve().parents[1]


def git(root, *args):
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def make_repo(tmp_path):
    root = tmp_path / "repo"
    (root / "nanochat" / "mixers").mkdir(parents=True)
    (root / "nanochat" / "mixers" / "kda.py").write_text("baseline\n", encoding="utf-8")
    git(tmp_path, "init", str(root)); git(root, "config", "user.email", "test@example.com"); git(root, "config", "user.name", "Test")
    git(root, "add", "."); git(root, "commit", "-m", "base")
    return root, git(root, "rev-parse", "HEAD")


def config(tmp_path):
    value = load_config(ROOT / "configs/research/kda_training_speed.toml")
    return replace(value, speed_supervisor=replace(value.speed_supervisor, ledger_path=str(tmp_path / "ledger.sqlite3")))


def test_speed_supervisor_intake_binds_ancestry_and_allowed_diff(tmp_path):
    root, base = make_repo(tmp_path)
    (root / "nanochat" / "mixers" / "kda.py").write_text("candidate\n", encoding="utf-8")
    git(root, "add", "."); git(root, "commit", "-m", "candidate"); candidate = git(root, "rev-parse", "HEAD")
    result = intake(root, config(tmp_path), base, candidate, "fuse local convolution")
    assert result["status"] == "accepted"
    assert result["changed_paths"] == ["nanochat/mixers/kda.py"]
    view = summary(root, config(tmp_path))
    assert view["quality_not_evaluated"] is True
    assert view["attempts"][0]["candidate_sha"] == candidate


def test_speed_supervisor_records_rejected_protected_diff(tmp_path):
    root, base = make_repo(tmp_path)
    (root / "tests").mkdir(); (root / "tests" / "test_cheat.py").write_text("pass\n", encoding="utf-8")
    git(root, "add", "."); git(root, "commit", "-m", "cheat"); candidate = git(root, "rev-parse", "HEAD")
    result = intake(root, config(tmp_path), base, candidate, "change test")
    assert result["status"] == "rejected"
    assert "outside" in result["reason"]


def test_speed_profile_never_exports_a_chrome_trace():
    source = (ROOT / "nanochat/research/speed_profile.py").read_text(encoding="utf-8")
    assert "export_chrome_trace" not in source
    assert "torch.profiler" not in source
    assert "mandatory_cuda_event_operator_regions" in source


def test_speed_supervisor_profile_contract_is_frozen(tmp_path):
    value = config(tmp_path)
    assert value.speed_supervisor.profile_timeout_seconds == 240
    assert value.speed_supervisor.profile_max_bytes == 262144
    assert value.speed_supervisor.profile_operator_rows == 30
    bad = replace(value, speed_supervisor=replace(value.speed_supervisor, profile_operator_rows=0))
    with pytest.raises(ConfigError, match="profile limits"):
        validate_config(bad)


def test_speed_supervisor_rejects_non_eager_lane(tmp_path):
    value = config(tmp_path)
    bad = replace(value, systems=replace(value.systems, execution_mode="full_compile"))
    with pytest.raises(ConfigError, match="eager"):
        validate_config(bad)


def test_speed_artifacts_are_namespaced_by_protocol():
    source = (ROOT / "nanochat/research/speed_supervisor.py").read_text(encoding="utf-8")
    assert 'ready["protocol_sha"][:12]' in source
    assert 'f"attempt-{attempt_id:05d}"' in source


def test_speed_supervisor_init_detects_protocol_change(tmp_path):
    root, _ = make_repo(tmp_path)
    value = config(tmp_path)
    assert initialize(root, value)["status"] == "ready"
    changed = replace(value, speed_supervisor=replace(value.speed_supervisor, max_attempts=3))
    with pytest.raises(ValueError, match="protocol hash"):
        initialize(root, changed)
