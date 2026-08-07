import hashlib
import json
import math

import pytest
import torch

from nanochat.research.config import ConfigError, GeneralEvaluationConfig, ResearchConfig, TrainingConfig, validate_config
from nanochat.research.general_eval import collect_suffix_examples, prepared_core_bundle, ruler_match_score, score_context_curve


def test_collect_suffix_examples_is_deterministic_and_uses_only_long_documents():
    selected = collect_suffix_examples([[1, 2], [0, 1, 2, 3, 4, 5], [9] * 8], (4, 6), 2, 2)
    assert selected == [[0, 1, 2, 3, 4, 5], [9] * 6]


def test_context_curve_scores_the_same_suffix_at_every_length():
    def uniform_model(tokens):
        return torch.zeros(tokens.shape[0], tokens.shape[1], 8, device=tokens.device)

    result = score_context_curve(
        uniform_model,
        [[0, 1, 2, 3, 4, 5], [0, 2, 3, 4, 5, 6]],
        (4, 6), 2, torch.ones(8, dtype=torch.long), torch.device("cpu"),
    )
    assert result["contexts"]["4"]["documents"] == 2
    assert result["contexts"]["6"]["documents"] == 2
    assert result["context_bpb"] == pytest.approx(3.0)


def test_general_evaluation_rejects_untrained_or_ambiguous_context_lengths():
    config = ResearchConfig(
        training=TrainingConfig(sequence_length=256),
        memory_probe=__import__("dataclasses").replace(ResearchConfig().memory_probe, enabled=False),
        evaluation=GeneralEvaluationConfig(context_lengths=(128, 512)),
    )
    with pytest.raises(ConfigError, match="trained sequence length"):
        validate_config(config)
    config = ResearchConfig(
        memory_probe=__import__("dataclasses").replace(ResearchConfig().memory_probe, enabled=False),
        evaluation=GeneralEvaluationConfig(context_lengths=(256, 128)),
    )
    with pytest.raises(ConfigError, match="unique and ascending"):
        validate_config(config)


def test_enabled_ruler_requires_a_hash_pinned_manifest():
    config = ResearchConfig(
        memory_probe=__import__("dataclasses").replace(ResearchConfig().memory_probe, enabled=False),
        evaluation=GeneralEvaluationConfig(ruler_enabled=True),
    )
    with pytest.raises(ConfigError, match="manifest path and SHA-256"):
        validate_config(config)


def test_core_bundle_requires_a_pinned_complete_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    bundle = tmp_path / "eval_bundle"
    bundle.mkdir()
    files = {}
    for name, contents in (("core.yaml", b"tasks: []\n"), ("eval_meta_data.csv", b"header\n")):
        target = bundle / name
        target.write_bytes(contents)
        files[name] = hashlib.sha256(contents).hexdigest()
    manifest = {"schema_version": 1, "files": files}
    manifest_path = tmp_path / "core.manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    config = GeneralEvaluationConfig(core_enabled=True, core_manifest="core.manifest.json", core_manifest_sha256=manifest_hash)
    assert prepared_core_bundle(config) == bundle
    (bundle / "core.yaml").write_text("changed\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="changed"):
        prepared_core_bundle(config)


def test_ruler_matching_matches_official_all_and_partial_semantics():
    prediction = "The values are Alpha and beta."
    assert ruler_match_score(prediction, ["alpha", "beta", "missing"], "all") == pytest.approx(2 / 3)
    assert ruler_match_score(prediction, ["missing", "BETA"], "partial") == 1.0
    assert ruler_match_score("", ["alpha"], "partial") == 0.0
