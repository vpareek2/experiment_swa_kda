import math

import pytest
import torch

from nanochat.research.config import ConfigError, GeneralEvaluationConfig, ResearchConfig, TrainingConfig, validate_config
from nanochat.research.general_eval import collect_suffix_examples, score_context_curve


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
