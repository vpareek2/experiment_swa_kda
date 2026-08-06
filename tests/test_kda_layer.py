"""Parity contracts for the complete K3 KimiDeltaAttention mixer."""

from __future__ import annotations

import importlib
import inspect

import pytest
import torch
import torch.nn as nn

from kda_oracle import OracleKDAState, full_layer as oracle_full_layer


def require_kda():
    try:
        return importlib.import_module("nanochat.kda")
    except ModuleNotFoundError as error:
        if error.name == "nanochat.kda":
            pytest.fail("nanochat.kda is not implemented (expected red TDD failure)")
        raise


def make_layer(module, *, hidden_size=16, num_heads=2, head_dim=8):
    return module.KimiDeltaAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        conv_size=4,
        layer_idx=0,
        mode="reference",
        norm_eps=1e-5,
    )


def deterministic_parameters(layer):
    generator = torch.Generator().manual_seed(314159)
    with torch.no_grad():
        for parameter in layer.parameters():
            values = torch.randn(parameter.shape, generator=generator, dtype=torch.float32) * 0.08
            parameter.copy_(values.to(parameter.dtype))
        layer.A_log.copy_(torch.tensor([-0.2, 0.15], dtype=torch.float32)[: layer.num_heads])
        layer.dt_bias.copy_(
            torch.linspace(-0.4, 0.3, layer.num_heads * layer.head_dim, dtype=torch.float32)
        )


def conv_weight(layer, name):
    weight = getattr(layer, name).weight
    assert weight.ndim == 3 and weight.shape[1] == 1
    return weight[:, 0]


def oracle_from_layer(layer, x, state=None):
    return oracle_full_layer(
        x,
        q_weight=layer.q_proj.weight,
        k_weight=layer.k_proj.weight,
        v_weight=layer.v_proj.weight,
        q_conv_weight=conv_weight(layer, "q_conv1d"),
        k_conv_weight=conv_weight(layer, "k_conv1d"),
        v_conv_weight=conv_weight(layer, "v_conv1d"),
        forget_a_weight=layer.f_a_proj.weight,
        forget_b_weight=layer.f_b_proj.weight,
        beta_weight=layer.b_proj.weight,
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        output_gate_weight=layer.g_proj.weight,
        norm_weight=layer.o_norm.weight,
        output_weight=layer.o_proj.weight,
        num_heads=layer.num_heads,
        norm_eps=layer.o_norm.eps,
        initial_state=state,
        lower_bound=layer.lower_bound,
    )


def assert_state_close(actual, expected, *, atol=2e-5, rtol=2e-5):
    torch.testing.assert_close(actual.memory, expected.memory, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual.q_conv, expected.q_conv, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual.k_conv, expected.k_conv, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual.v_conv, expected.v_conv, atol=atol, rtol=rtol)


def test_k3_layer_has_required_full_rank_structure():
    module = require_kda()
    layer = make_layer(module)

    assert layer.num_heads == 2
    assert layer.head_dim == 8
    assert layer.lower_bound == -5.0
    for projection in (layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj):
        assert isinstance(projection, nn.Linear)
        assert projection.bias is None
    for convolution in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d):
        assert convolution.groups == 16
        assert convolution.kernel_size == (4,)
        assert convolution.bias is None

    assert layer.A_log.shape == (2,)
    assert layer.dt_bias.shape == (16,)
    assert layer.A_log.dtype == torch.float32
    assert layer.dt_bias.dtype == torch.float32
    assert layer.b_proj.weight.shape == (2, 16)
    assert layer.f_a_proj.weight.shape == (8, 16)
    assert layer.f_b_proj.weight.shape == (16, 8)

    # K3 replaced Kimi Linear's two-layer low-rank output gate with one direct
    # input-dependent full-rank projection.
    assert isinstance(layer.g_proj, nn.Linear)
    assert layer.g_proj.weight.shape == (16, 16)
    assert layer.g_proj.bias is None
    assert not hasattr(layer, "g_a_proj")
    assert not hasattr(layer, "g_b_proj")


def test_k3_layer_does_not_accept_attention_position_or_value_embedding_inputs():
    module = require_kda()
    layer = make_layer(module)
    parameters = inspect.signature(layer.forward).parameters
    assert "cos_sin" not in parameters
    assert "window_size" not in parameters
    assert "ve" not in parameters
    assert not hasattr(layer, "ve_gate")


def test_k3_layer_full_forward_matches_independent_composition():
    module = require_kda()
    layer = make_layer(module)
    deterministic_parameters(layer)
    x = torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(7))

    actual, actual_state = layer(
        x, state=None, output_final_state=True, mode="reference"
    )
    expected, expected_state = oracle_from_layer(layer, x)

    assert actual.shape == x.shape
    assert isinstance(actual_state, module.KDAState)
    assert actual_state.memory.shape == (2, 2, 8, 8)
    assert actual_state.memory.dtype == torch.float32
    for conv_state in (actual_state.q_conv, actual_state.k_conv, actual_state.v_conv):
        assert conv_state.shape == (2, 16, 4)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert_state_close(actual_state, expected_state)


def test_k3_layer_full_sequence_matches_token_decode():
    module = require_kda()
    layer = make_layer(module)
    deterministic_parameters(layer)
    layer.eval()
    x = torch.randn(2, 11, 16, generator=torch.Generator().manual_seed(19))

    full, full_state = layer(x, state=None, output_final_state=True, mode="reference")
    state = None
    pieces = []
    for index in range(x.shape[1]):
        piece, state = layer(
            x[:, index:index + 1],
            state=state,
            output_final_state=True,
            mode="recurrent",
        )
        pieces.append(piece)
    decoded = torch.cat(pieces, dim=1)

    torch.testing.assert_close(decoded, full, atol=3e-5, rtol=3e-5)
    assert_state_close(state, full_state, atol=3e-5, rtol=3e-5)


def test_k3_layer_nonzero_state_matches_oracle_and_is_not_mutated():
    module = require_kda()
    layer = make_layer(module)
    deterministic_parameters(layer)
    generator = torch.Generator().manual_seed(23)
    x = torch.randn(2, 5, 16, generator=generator)
    state = module.KDAState(
        memory=torch.randn(2, 2, 8, 8, generator=generator),
        q_conv=torch.randn(2, 16, 4, generator=generator),
        k_conv=torch.randn(2, 16, 4, generator=generator),
        v_conv=torch.randn(2, 16, 4, generator=generator),
    )
    original = OracleKDAState(
        state.memory.clone(), state.q_conv.clone(), state.k_conv.clone(), state.v_conv.clone()
    )

    actual, actual_state = layer(x, state=state, output_final_state=True, mode="reference")
    expected, expected_state = oracle_from_layer(layer, x, original)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert_state_close(actual_state, expected_state)
    assert_state_close(state, original, atol=0, rtol=0)


def test_k3_layer_reset_and_batch_independence():
    module = require_kda()
    layer = make_layer(module)
    deterministic_parameters(layer)
    x = torch.randn(2, 7, 16, generator=torch.Generator().manual_seed(29))

    batched, _ = layer(x, state=None, output_final_state=True, mode="reference")
    reset, _ = layer(x, state=None, output_final_state=True, mode="reference")
    torch.testing.assert_close(batched, reset, atol=0, rtol=0)
    for batch_index in range(2):
        separate, _ = layer(
            x[batch_index:batch_index + 1],
            state=None,
            output_final_state=True,
            mode="reference",
        )
        torch.testing.assert_close(
            batched[batch_index:batch_index + 1], separate, atol=2e-5, rtol=2e-5
        )


def test_k3_layer_is_causal():
    module = require_kda()
    layer = make_layer(module)
    deterministic_parameters(layer)
    x = torch.randn(1, 9, 16, generator=torch.Generator().manual_seed(31))
    altered = x.clone()
    altered[:, 5:] = torch.randn(
        altered[:, 5:].shape, generator=torch.Generator().manual_seed(37)
    )
    original, _ = layer(x, state=None, output_final_state=False, mode="reference")
    changed, _ = layer(altered, state=None, output_final_state=False, mode="reference")
    torch.testing.assert_close(original[:, :5], changed[:, :5], atol=0, rtol=0)


def test_k3_layer_backward_is_finite_for_input_and_all_parameters():
    module = require_kda()
    layer = make_layer(module)
    deterministic_parameters(layer)
    x = torch.randn(2, 6, 16, generator=torch.Generator().manual_seed(41), requires_grad=True)
    output, _ = layer(x, state=None, output_final_state=False, mode="reference")
    output.square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    missing = []
    nonfinite = []
    for name, parameter in layer.named_parameters():
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            nonfinite.append(name)
    assert not missing, f"parameters without gradients: {missing}"
    assert not nonfinite, f"parameters with non-finite gradients: {nonfinite}"


def test_k3_layer_can_omit_final_state():
    module = require_kda()
    layer = make_layer(module)
    x = torch.randn(1, 3, 16)
    output, state = layer(x, state=None, output_final_state=False, mode="reference")
    assert output.shape == x.shape
    assert state is None
