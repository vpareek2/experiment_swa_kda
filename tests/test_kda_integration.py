"""Contracts for integrating KDA into nanochat and the research harness."""

from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache
from nanochat.gpt import CausalSelfAttention, GPT, GPTConfig
from nanochat.research.config import (
    ConfigError,
    MemoryProbeConfig,
    ProbeStageConfig,
    apply_candidate,
    load_config,
    validate_config,
)
from nanochat.research.memory import MemoryTaskSpec, generate_memory_batch
from nanochat.research.probe import CanonicalProbeModel
from nanochat.research.runner import _state_bytes


ROOT = Path(__file__).resolve().parents[1]


def require_kda():
    try:
        return importlib.import_module("nanochat.kda")
    except ModuleNotFoundError as error:
        if error.name == "nanochat.kda":
            pytest.fail("nanochat.kda is not implemented (expected red TDD failure)")
        raise


def tiny_config(pattern, *, layers=4, force_final_full=False):
    return GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=layers,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern=pattern,
        sliding_window=4,
        force_final_full=force_final_full,
    )


def materialize_model(config):
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


def make_cache(model, *, batch_size, seq_len):
    return KVCache(
        batch_size=batch_size,
        num_heads=model.config.n_kv_head,
        seq_len=seq_len,
        head_dim=model.config.n_embd // model.config.n_head,
        num_layers=model.config.n_layer,
        mixer_types=model.mixer_types,
        conv_size=4,
        device="cpu",
        dtype=COMPUTE_DTYPE,
    )


@pytest.mark.parametrize(
    ("pattern", "layers", "expected"),
    [
        ("K", 4, ["K", "K", "K", "K"]),
        ("SK", 5, ["S", "K", "S", "K", "S"]),
        ("SSK", 7, ["S", "S", "K", "S", "S", "K", "S"]),
        ("KL", 5, ["K", "L", "K", "L", "K"]),
    ],
)
def test_mixer_pattern_tiles_exactly(pattern, layers, expected):
    module = require_kda()
    config = tiny_config(pattern, layers=layers)
    with torch.device("meta"):
        model = GPT(config)

    assert model.mixer_types == expected
    for index, kind in enumerate(expected):
        mixer = model.transformer.h[index].attn
        expected_type = module.KimiDeltaAttention if kind == "K" else CausalSelfAttention
        assert isinstance(mixer, expected_type)
        if kind == "K":
            assert model.window_sizes[index] is None
        elif kind == "S":
            assert model.window_sizes[index] == (4, 0)
        else:
            assert model.window_sizes[index] == (16, 0)


def test_force_final_full_has_explicit_existing_semantics_for_kda():
    module = require_kda()
    with torch.device("meta"):
        exact = GPT(tiny_config("K", layers=3, force_final_full=False))
        forced = GPT(tiny_config("K", layers=3, force_final_full=True))
    assert exact.mixer_types == ["K", "K", "K"]
    assert all(isinstance(block.attn, module.KimiDeltaAttention) for block in exact.transformer.h)
    assert forced.mixer_types == ["K", "K", "L"]
    assert isinstance(forced.transformer.h[-1].attn, CausalSelfAttention)
    assert forced.window_sizes[-1] == (16, 0)


def test_tiny_all_kda_gpt_forward_backward():
    model = materialize_model(tiny_config("K", layers=2))
    token_ids = torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(47))
    targets = torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(53))
    loss = model(token_ids, targets=targets)
    loss.backward()

    assert loss.ndim == 0 and torch.isfinite(loss)
    kda_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if ".attn." in name
    ]
    assert kda_parameters
    assert all(parameter.grad is not None for parameter in kda_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in kda_parameters)


def test_tiny_swa_kda_gpt_forward_backward():
    model = materialize_model(tiny_config("SK", layers=2))
    token_ids = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(59))
    targets = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(61))
    loss = model(token_ids, targets=targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        block.attn.o_proj.weight.grad is not None
        if hasattr(block.attn, "o_proj")
        else block.attn.c_proj.weight.grad is not None
        for block in model.transformer.h
    )


@pytest.mark.parametrize("pattern", ["K", "SK"])
def test_cached_prefill_and_decode_match_one_shot_logits(pattern):
    model = materialize_model(tiny_config(pattern, layers=2)).eval()
    tokens = torch.randint(0, 64, (1, 9), generator=torch.Generator().manual_seed(67))
    cache = make_cache(model, batch_size=1, seq_len=tokens.shape[1])

    with torch.inference_mode():
        prefill_logits = model(tokens[:, :4], kv_cache=cache)
        expected_prefill = model(tokens[:, :4])
        torch.testing.assert_close(prefill_logits, expected_prefill, atol=2e-4, rtol=2e-4)
        assert cache.get_pos() == 4

        for position in range(4, tokens.shape[1]):
            decoded = model(tokens[:, position:position + 1], kv_cache=cache)
            one_shot = model(tokens[:, :position + 1])[:, -1:]
            torch.testing.assert_close(decoded, one_shot, atol=3e-4, rtol=3e-4)
            assert cache.get_pos() == position + 1


def test_hybrid_cache_allocates_only_the_state_each_mixer_needs():
    model = materialize_model(tiny_config("SK", layers=4)).eval()
    cache = make_cache(model, batch_size=1, seq_len=16)
    assert tuple(cache.allocated_kv_layers) == (0, 2)
    assert tuple(cache.allocated_kda_layers) == (1, 3)
    for layer_index in cache.allocated_kda_layers:
        assert cache.get_kda_state(layer_index) is None
    for layer_index in cache.allocated_kv_layers:
        key, value = cache.get_layer_cache(layer_index)
        assert key.shape == value.shape == (1, 16, 2, 16)
    for layer_index in cache.allocated_kda_layers:
        with pytest.raises(ValueError, match="KDA"):
            cache.get_layer_cache(layer_index)


def test_kda_cache_state_size_is_independent_of_maximum_sequence_length():
    model = materialize_model(tiny_config("K", layers=2)).eval()
    prompt = torch.randint(0, 64, (1, 6), generator=torch.Generator().manual_seed(71))
    short_cache = make_cache(model, batch_size=1, seq_len=8)
    long_cache = make_cache(model, batch_size=1, seq_len=256)
    with torch.inference_mode():
        model(prompt, kv_cache=short_cache)
        model(prompt, kv_cache=long_cache)
    assert short_cache.state_nbytes() == long_cache.state_nbytes()
    assert short_cache.state_nbytes() > 0
    assert short_cache.get_kda_state(0).memory.shape == (1, 2, 16, 16)


def test_kda_cache_prefill_expands_batch_and_reset_clears_every_state():
    model = materialize_model(tiny_config("K", layers=2)).eval()
    tokens = torch.randint(0, 64, (1, 5), generator=torch.Generator().manual_seed(73))
    source = make_cache(model, batch_size=1, seq_len=5)
    destination = make_cache(model, batch_size=3, seq_len=12)
    with torch.inference_mode():
        model(tokens, kv_cache=source)
    destination.prefill(source)

    assert destination.get_pos() == 5
    for layer_index in destination.allocated_kda_layers:
        source_state = source.get_kda_state(layer_index)
        copied_state = destination.get_kda_state(layer_index)
        assert copied_state.memory.shape[0] == 3
        torch.testing.assert_close(
            copied_state.memory,
            source_state.memory.expand(3, -1, -1, -1),
        )
        assert copied_state.memory.data_ptr() != source_state.memory.data_ptr()
    assert destination.prev_embedding is not None and destination.prev_embedding.shape[0] == 3

    destination.reset()
    assert destination.get_pos() == 0
    assert destination.prev_embedding is None
    assert all(destination.get_kda_state(index) is None for index in destination.allocated_kda_layers)


def test_candidate_and_protocol_validation_accept_kda_patterns(tmp_path):
    protocol = load_config(ROOT / "configs/research/discovery.toml")
    candidate = tmp_path / "candidate.toml"
    candidate.write_text(
        '[candidate]\nname = "swa-kda"\nwindow_pattern = "SK"\n'
        "sliding_window = 256\nforce_final_full = false\n",
        encoding="utf-8",
    )
    resolved = apply_candidate(candidate, protocol)
    assert resolved.training.window_pattern == "SK"
    assert resolved.training.force_final_full is False

    invalid = replace(resolved, training=replace(resolved.training, window_pattern="SKX"))
    with pytest.raises(ConfigError, match="L, S, and K|LSK"):
        validate_config(invalid)


def test_research_state_accounting_counts_fixed_kda_and_conv_states():
    protocol = load_config(ROOT / "configs/research/discovery.toml")
    base = replace(
        protocol.training,
        depth=2,
        head_dim=8,
        sequence_length=128,
        sliding_window=16,
        window_pattern="K",
        force_final_full=False,
    )
    config = replace(protocol, training=base)
    # model_dim = ceil(depth*64/head_dim)*head_dim = 128, H=16, D=8.
    # Per KDA layer: FP32 H*V*K memory plus three BF16 width*conv_size caches.
    expected_per_layer = 16 * 8 * 8 * 4 + 3 * 128 * 4 * 2
    assert _state_bytes(config) == 2 * expected_per_layer

    longer = replace(config, training=replace(base, sequence_length=1024, sliding_window=16))
    assert _state_bytes(longer) == _state_bytes(config)

    mixed = replace(config, training=replace(base, window_pattern="SK"))
    expected_swa = 2 * 16 * 8 * 2 * 16
    assert _state_bytes(mixed) == expected_swa + expected_per_layer


def test_canonical_memory_probe_accepts_kda_and_optimizes_one_step():
    probe = MemoryProbeConfig(
        vocab_size=256,
        depth=2,
        width=32,
        head_dim=16,
        window_size=8,
        lengths=(16,),
        loads=(2,),
        updates=(1,),
        distractor_ratios=(0,),
        stages=(ProbeStageConfig("smoke", 4, 2, 16, (2,), 1, (1,), (0,)),),
    )
    model = CanonicalProbeModel(probe, "K", force_final_full=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs, labels, _ = generate_memory_batch(
        MemoryTaskSpec(16, 2, num_queries=1, updates_per_query_key=1, vocab_size=256),
        seeds=(79, 83),
        device="cpu",
    )
    logits = model(inputs)
    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-1)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.parameters()
    )
