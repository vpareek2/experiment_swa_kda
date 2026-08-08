from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import kda_cuda_development as development


def _payload(*, latency_ratio: float = 1.0, memory_ratio: float = 1.0):
    rows = []
    for length in development.LENGTHS:
        for operation in development.OPERATIONS:
            baseline_ms = 20.0 if operation == "chunk_forward_backward" else 10.0
            rows.append({
                "operation": operation, "length": length,
                **development.PRODUCTION_SHAPE,
                "median_ms": baseline_ms * latency_ratio,
                "peak_allocated_bytes": int(1_000_000 * memory_ratio),
            })
    return {"status": "complete", "microbenchmarks": rows}


def _row(payload, operation, length):
    return next(row for row in payload["microbenchmarks"]
                if row["operation"] == operation and row["length"] == length)


def test_forward_need_not_improve_for_advance():
    baseline = _payload()
    candidate = _payload()
    for row in candidate["microbenchmarks"]:
        if row["operation"] == "chunk_forward_backward" and row["length"] == 4096:
            row["median_ms"] *= 0.96
    result = development.compare_microbenchmarks(baseline, candidate)
    assert result["advance"] is True
    assert result["t4096_forward_backward"]["meets_three_percent"] is True


def test_latency_regression_over_five_percent_blocks():
    baseline, candidate = _payload(), _payload()
    _row(candidate, "chunk_forward", 256)["median_ms"] *= 1.051
    result = development.compare_microbenchmarks(baseline, candidate)
    assert result["advance"] is False
    assert result["important_regressions_within_limit"] is False


def test_memory_regression_over_three_percent_blocks():
    baseline, candidate = _payload(), _payload()
    _row(candidate, "chunk_forward_backward", 4096)["median_ms"] *= 0.96
    _row(candidate, "chunk_forward", 1024)["peak_allocated_bytes"] = 1_030_001
    result = development.compare_microbenchmarks(baseline, candidate)
    assert result["advance"] is False
    assert result["memory_regressions_within_limit"] is False


@pytest.mark.parametrize("mutation", ["missing", "invalid", "wrong_shape", "duplicate"])
def test_invalid_or_missing_rows_are_rejected(mutation):
    baseline, candidate = _payload(), _payload()
    if mutation == "missing":
        candidate["microbenchmarks"].pop()
    elif mutation == "invalid":
        candidate["microbenchmarks"][0]["peak_allocated_bytes"] = 0
    elif mutation == "wrong_shape":
        candidate["microbenchmarks"][0]["heads"] = 6
    else:
        candidate["microbenchmarks"].append(copy.deepcopy(candidate["microbenchmarks"][0]))
    with pytest.raises(ValueError):
        development.compare_microbenchmarks(baseline, candidate)


def test_incomplete_payload_is_rejected():
    with pytest.raises(ValueError, match="not complete"):
        development.compare_microbenchmarks({"status": "invalid"}, _payload())


def test_output_collision_never_overwrites(tmp_path):
    output = tmp_path / "artifacts"
    output.mkdir()
    sentinel = output / "sentinel.json"
    sentinel.write_text(json.dumps({"keep": True}))
    with pytest.raises(FileExistsError):
        development.create_output_dir(output)
    assert json.loads(sentinel.read_text()) == {"keep": True}


def test_production_shape_is_b2_h3_and_level1_is_bounded():
    _, config = development.resolved_config()
    shape = development.production_shape(config)
    assert (shape["B"], shape["H"], shape["K"], shape["V"]) == (2, 3, 128, 128)
    assert config.kernel_gates.timeout_seconds == 300.0


@pytest.mark.parametrize(
    ("requested", "expected"),
    [("baseline-first", ["baseline", "candidate"]),
     ("candidate-first", ["candidate", "baseline"])],
)
def test_level2_order_and_fresh_namespace(tmp_path, requested, expected):
    _, config = development.resolved_config()
    first_dir = tmp_path / "first"; first_dir.mkdir()
    second_dir = tmp_path / "second"; second_dir.mkdir()
    first = development.level2_plan(
        tmp_path / "base", tmp_path / "candidate", first_dir, config, requested,
    )
    second = development.level2_plan(
        tmp_path / "base", tmp_path / "candidate", second_dir, config, requested,
    )
    assert first["training_launched"] is False
    assert first["order"] == expected
    assert [entry["label"] for entry in first["commands"]] == expected
    assert first["namespace"] != second["namespace"]
    assert len({entry["model_tag"] for entry in first["commands"]}) == 2


def test_cli_level2_order():
    args = development.parse_args([
        "/tmp/base", "/tmp/candidate", "/tmp/output",
        "--level2-order", "candidate-first",
    ])
    assert args.level2_order == "candidate-first"
