from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.research.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    protected_fingerprint,
    select_triton_ptxas,
)
from nanochat.research.config import ConfigError, MemoryProbeConfig, ProbeStageConfig, apply_candidate, load_config
from nanochat.research.decision import (
    aggregate_objectives,
    calibrate_objectives,
    classify_candidate,
    compare_metrics,
    objectives_from_config,
)
from nanochat.research.memory import (
    PAIR,
    QUERY,
    MemoryTaskSpec,
    generate_memory_batch,
    generate_memory_example,
    score_answer_tokens,
    score_oracle,
)
from nanochat.research.probe import calibration_checks, probe_protocol_hash, train_probe, wilson_interval
from nanochat.research.protected import (
    changed_path_violations,
    initialize_supervisor,
    sign_summary,
    verify_protected,
    verify_signature,
)
from nanochat.research.runner import (
    _extract_training_metrics,
    _frontier_summaries,
    _parse_training_result,
    _probe_calibration_identity,
    load_probe_calibration,
    probe_calibration_path,
    render_report,
)


ROOT = Path(__file__).resolve().parents[1]


def test_research_configs_load_and_use_distinct_budget_lanes():
    discovery = load_config(ROOT / "configs/research/discovery.toml")
    promotion = load_config(ROOT / "configs/research/promotion.toml")
    assert discovery.training.seconds == 300
    assert discovery.training.tokens == 0
    assert promotion.training.seconds == 0
    assert promotion.training.tokens == 100_663_296
    assert promotion.training.tokens % promotion.training.total_batch_size == 0
    assert discovery.schema_version == promotion.schema_version == 2
    assert sum(stage.answer_budget for stage in discovery.memory_probe.stages) == 544_768
    assert probe_protocol_hash(discovery.memory_probe) == probe_protocol_hash(promotion.memory_probe)


def test_config_rejects_unknown_keys(tmp_path):
    source = (ROOT / "configs/research/discovery.toml").read_text(encoding="utf-8")
    path = tmp_path / "bad.toml"
    path.write_text(source.replace("depth = 6", "depth = 6\nunknown_knob = 1"), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown_knob"):
        load_config(path)


def test_candidate_config_can_only_change_architecture(tmp_path):
    protocol = load_config(ROOT / "configs/research/discovery.toml")
    swa = apply_candidate(ROOT / "configs/candidates/baseline_swa.toml", protocol)
    assert swa.run.name == "baseline-swa"
    assert swa.training.window_pattern == "S"
    assert swa.training.seconds == protocol.training.seconds
    bad = tmp_path / "bad-candidate.toml"
    bad.write_text('[candidate]\nname="cheat"\nseconds=1\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="seconds"):
        apply_candidate(bad, protocol)


def test_explicit_swa_window_can_remain_local_in_final_layer():
    config = GPTConfig(sequence_len=1024, n_layer=3, n_head=2, n_kv_head=2, n_embd=64,
                       window_pattern="S", sliding_window=128, force_final_full=False)
    with torch.device("meta"):
        model = GPT(config)
    assert model.window_sizes == [(128, 0)] * 3
    config = replace(config, force_final_full=True)
    with torch.device("meta"):
        model = GPT(config)
    assert model.window_sizes == [(128, 0), (128, 0), (1024, 0)]


def _latest_writes(inputs):
    latest = {}
    for index in range(len(inputs) - 2):
        if inputs[index].item() == PAIR:
            latest[inputs[index + 1].item()] = inputs[index + 2].item()
    return latest


def test_memory_generator_is_deterministic_masked_and_last_write_wins():
    spec = MemoryTaskSpec(128, 8, num_queries=3, updates_per_query_key=4, distractor_ratio=1)
    first = generate_memory_example(spec, 17)
    second = generate_memory_example(spec, 17)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    inputs, labels, metadata = first
    assert int((labels >= 0).sum()) == spec.num_queries
    latest = _latest_writes(inputs)
    for key_position in metadata.query_positions:
        assert inputs[key_position - 1].item() == QUERY
        assert labels[key_position].item() == latest[inputs[key_position].item()]


@pytest.mark.parametrize("distance", [31, 32, 33, 64])
def test_memory_generator_controls_last_write_distance(distance):
    spec = MemoryTaskSpec(128, 8, num_queries=1, target_distance=distance)
    _, _, metadata = generate_memory_example(spec, 91)
    assert metadata.query_distances == (distance,)


def test_memory_batch_and_answer_scoring():
    spec = MemoryTaskSpec(64, 4, num_queries=2)
    inputs, labels, metadata = generate_memory_batch(spec, [1, 2])
    logits = torch.zeros(2, 64, spec.vocab_size)
    mask = labels >= 0
    batch_index, token_index = mask.nonzero(as_tuple=True)
    logits[batch_index, token_index, labels[mask]] = 10
    scores = score_answer_tokens(logits, labels)
    assert inputs.shape == labels.shape == (2, 64)
    assert len(metadata) == 2
    assert scores["answer_tokens"] == 4
    assert scores["accuracy"] == 1.0
    assert score_oracle(inputs, labels)["accuracy"] == 1.0


def test_wilson_interval_contains_observed_accuracy():
    low, high = wilson_interval(80, 100)
    assert low < 0.8 < high
    assert wilson_interval(0, 1)[0] == 0.0


def _objectives(**overrides):
    values = {
        "val_bpb": 2.0,
        "memory_auc": 0.5,
        "update_accuracy": 0.5,
        "tokens_per_second": 100.0,
        "peak_memory_mb": 1000.0,
        "state_bytes": 1000.0,
    }
    values.update(overrides)
    return values


def test_pareto_decision_distinguishes_dominated_frontier_and_retest():
    incumbent = {"run_id": "base", "objectives": _objectives()}
    dominated = {"status": "complete", "run_id": "bad", "objectives": _objectives(val_bpb=2.1)}
    assert classify_candidate(dominated, [incumbent])["status"] == "dominated"
    tradeoff = {"status": "complete", "run_id": "trade", "objectives": _objectives(val_bpb=1.9, peak_memory_mb=1200)}
    assert classify_candidate(tradeoff, [incumbent])["status"] == "retest"
    winner = {"status": "complete", "run_id": "win", "objectives": _objectives(val_bpb=1.9)}
    assert classify_candidate(winner, [incumbent])["status"] == "frontier"
    assert "tokens_per_second" in compare_metrics(_objectives(tokens_per_second=102), _objectives())["equivalent"]
    decision = load_config(ROOT / "configs/research/discovery.toml").decision
    assert objectives_from_config(decision)[0].tolerance == decision.bpb_floor


def test_seed_aggregation_and_noise_calibration():
    summaries = [
        {"status": "complete", "memory_probe_protocol_version": "associative_recall_v2",
         "objectives": _objectives(val_bpb=value)}
        for value in (1.99, 2.00, 2.01, 2.02, 1.98)
    ]
    aggregate = aggregate_objectives(summaries)
    assert aggregate["runs"] == 5
    assert aggregate["metrics"]["val_bpb"]["mean"] == pytest.approx(2.0)
    decision = load_config(ROOT / "configs/research/discovery.toml").decision
    calibrated = calibrate_objectives(summaries, decision)
    assert calibrated["tolerances"]["val_bpb"] > decision.bpb_floor


def test_artifacts_are_canonical_atomic_and_fingerprinted(tmp_path):
    assert canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
    path = tmp_path / "nested/result.json"
    atomic_write_json(path, {"ok": True})
    assert json.loads(path.read_text()) == {"ok": True}
    (tmp_path / "nested/__pycache__").mkdir()
    (tmp_path / "nested/__pycache__/ignored.pyc").write_bytes(b"cache")
    fingerprint = protected_fingerprint(tmp_path, ["nested/"])
    assert fingerprint["nested/result.json"]
    assert "nested/__pycache__/ignored.pyc" not in fingerprint


def test_gb10_selects_an_assembler_with_sm121a_support():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (12, 1):
        pytest.skip("SM 12.1 compatibility check")
    selected = select_triton_ptxas()
    assert selected is not None


def test_structured_training_log_is_extracted(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        'noise\nRESEARCH_TRAIN_STEP {"step": 1, "loss": 2.0}\n'
        'RESEARCH_TRAIN_RESULT {"status": "complete", "step": 2}\n',
        encoding="utf-8",
    )
    metrics = tmp_path / "metrics.jsonl"
    assert _extract_training_metrics(log, metrics) == 1
    assert json.loads(metrics.read_text()) == {"step": 1, "loss": 2.0}
    assert _parse_training_result(log)["step"] == 2


def test_supervisor_detects_changes_and_signs(tmp_path):
    root = tmp_path / "repo"
    state = tmp_path / "state"
    (root / "protected").mkdir(parents=True)
    target = root / "protected/eval.py"
    target.write_text("original\n", encoding="utf-8")
    initialize_supervisor(root, ["protected/"], state)
    assert verify_protected(root, state)["valid"]
    target.write_text("changed\n", encoding="utf-8")
    assert verify_protected(root, state)["changed"] == ["protected/eval.py"]
    target.write_text("original\n", encoding="utf-8")
    (root / "protected/new.py").write_text("added\n", encoding="utf-8")
    assert verify_protected(root, state)["changed"] == ["protected/new.py"]
    summary = {"run_id": "x", "score": 1}
    signature = sign_summary(summary, state)
    assert verify_signature(summary, signature, state)
    assert not verify_signature({**summary, "score": 2}, signature, state)
    assert changed_path_violations(["nanochat/gpt.py", "tests/x.py"], ["nanochat/gpt.py"]) == ["tests/x.py"]


def test_tiny_memory_probe_training_step_runs_on_cpu():
    config = MemoryProbeConfig(
        vocab_size=256,
        depth=1,
        width=32,
        head_dim=16,
        window_size=32,
        examples_per_cell=1,
        lengths=(128,),
        loads=(4,),
        updates=(1,),
        distractor_ratios=(0,),
        stages=(ProbeStageConfig("tiny", 8, 2, 128, (4,)),),
    )
    model, result = train_probe(config, "S", 7, torch.device("cpu"))
    assert result["training_steps"] == 1
    assert result["supervised_answers"] == 8
    assert result["stage_results"][0]["final_loss"] is not None
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def _calibration_result(boundary, *, easy=1.0, auc=1.0, updates=1.0, seconds=1.0):
    return {
        "easy_control": {"accuracy": easy},
        "evaluation": {
            "memory_auc": auc,
            "update_accuracy": updates,
            "cells": {"boundary": [
                {"spec": {"target_distance": distance}, "accuracy": accuracy}
                for distance, accuracy in boundary.items()
            ]},
        },
        "total_seconds": seconds,
    }


def test_calibration_requires_learning_and_swa_boundary_discrimination():
    config = MemoryProbeConfig()
    full = _calibration_result({255: 1.0, 256: 1.0, 1024: 0.95})
    swa = _calibration_result({255: 0.95, 256: 0.90, 1024: 0.10})
    checks = calibration_checks(config, [11], {"accuracy": 1.0}, {"accuracy": 1.0}, [full], [swa])
    assert all(item["passed"] for item in checks)
    broken = _calibration_result({255: 1.0, 256: 1.0, 1024: 0.90}, easy=0.2)
    checks = calibration_checks(config, [11], {"accuracy": 1.0}, {"accuracy": 1.0}, [broken], [swa])
    assert next(item for item in checks if item["name"].endswith("full_easy"))["passed"] is False


def test_legacy_probe_runs_are_excluded_from_frontier_and_report(tmp_path):
    legacy = {"run_id": "v1", "status": "complete", "decision": {"status": "frontier"},
              "objectives": _objectives(memory_auc=0.99)}
    current = {"run_id": "v2", "status": "complete", "decision": {"status": "frontier"},
               "memory_probe_protocol_version": "associative_recall_v2", "objectives": _objectives()}
    for item in (legacy, current):
        atomic_write_json(tmp_path / item["run_id"] / "summary.json", item)
    assert [item["run_id"] for item in _frontier_summaries(tmp_path, "new", "associative_recall_v2")] == ["v2"]
    report = render_report([legacy, current])
    assert "legacy/ineligible" in report


def test_probe_calibration_loader_rejects_missing_and_stale_artifacts(tmp_path):
    config = load_config(ROOT / "configs/research/discovery.toml")
    config = replace(config, protection=replace(config.protection, protected_paths=()))
    with pytest.raises(FileNotFoundError, match="missing memory-probe calibration"):
        load_probe_calibration(tmp_path, config)
    path = probe_calibration_path(tmp_path, config)
    identity = _probe_calibration_identity(tmp_path, config)
    atomic_write_json(path, {"status": "valid", **identity})
    assert load_probe_calibration(tmp_path, config)["protocol_hash"] == identity["protocol_hash"]
    atomic_write_json(path, {"status": "valid", **identity, "protected_hash": "stale"})
    with pytest.raises(ValueError, match="protected_hash mismatch"):
        load_probe_calibration(tmp_path, config)
