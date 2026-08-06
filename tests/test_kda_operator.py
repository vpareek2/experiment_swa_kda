"""CPU contracts for K3 gating and the KDA recurrence."""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn.functional as F

from kda_oracle import (
    causal_depthwise_conv,
    default_scale,
    expected_single_token_state,
    k3_decay_gate as oracle_decay_gate,
    recurrent_kda as oracle_recurrent_kda,
)


def require_kda():
    try:
        return importlib.import_module("nanochat.kda")
    except ModuleNotFoundError as error:
        if error.name == "nanochat.kda":
            pytest.fail("nanochat.kda is not implemented (expected red TDD failure)")
        raise


def make_inputs(*, batch=2, length=7, heads=2, key_dim=4, value_dim=3, dtype=torch.float32):
    generator = torch.Generator().manual_seed(1729)
    tensors = {
        "q": torch.randn(batch, length, heads, key_dim, generator=generator, dtype=dtype),
        "k": torch.randn(batch, length, heads, key_dim, generator=generator, dtype=dtype),
        "v": torch.randn(batch, length, heads, value_dim, generator=generator, dtype=dtype),
        "g": torch.randn(batch, length, heads, key_dim, generator=generator, dtype=dtype),
        "beta": torch.randn(batch, length, heads, generator=generator, dtype=dtype),
        "A_log": torch.randn(heads, generator=generator, dtype=dtype) * 0.2,
        "dt_bias": torch.randn(heads, key_dim, generator=generator, dtype=dtype) * 0.3,
        "initial_state": torch.randn(
            batch, heads, value_dim, key_dim, generator=generator, dtype=dtype
        ) * 0.1,
    }
    return tensors


def call_operator(module, inputs, **kwargs):
    return module.kda(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["A_log"],
        inputs["dt_bias"],
        initial_state=inputs.get("initial_state"),
        output_final_state=True,
        mode="reference",
        **kwargs,
    )


def test_oracle_gate_is_bounded_monotone_and_matches_formula():
    raw = torch.tensor([-100.0, -2.0, 0.0, 2.0, 100.0], dtype=torch.float64).view(1, 1, 1, 5)
    A_log = torch.tensor([0.3], dtype=torch.float64)
    bias = torch.zeros(1, 5, dtype=torch.float64)
    actual = oracle_decay_gate(raw, A_log, bias)
    expected = -5.0 * torch.sigmoid(A_log.exp().view(1, 1) * raw)

    torch.testing.assert_close(actual, expected)
    assert torch.all(actual >= -5.0)
    assert torch.all(actual <= 0.0)
    assert torch.all(actual[..., 1:] < actual[..., :-1])
    assert actual[..., 0].item() == pytest.approx(0.0, abs=1e-12)
    assert actual[..., -1].item() == pytest.approx(-5.0, abs=1e-12)


def test_oracle_one_token_zero_state_has_closed_form():
    inputs = make_inputs(batch=1, length=1, heads=2, key_dim=4, value_dim=3, dtype=torch.float64)
    inputs["initial_state"] = None
    output, state = oracle_recurrent_kda(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"],
        inputs["A_log"], inputs["dt_bias"], initial_state=None, compute_dtype=torch.float64,
    )
    key = F.normalize(inputs["k"][:, 0], dim=-1)
    beta = inputs["beta"][:, 0].sigmoid()
    expected_state = expected_single_token_state(key, inputs["v"][:, 0], beta)
    query = F.normalize(inputs["q"][:, 0], dim=-1) * default_scale(inputs["q"].shape[-1])
    expected_output = torch.einsum("bhk,bhvk->bhv", query, expected_state).unsqueeze(1)

    torch.testing.assert_close(state, expected_state, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(output, expected_output, atol=1e-10, rtol=1e-10)


def test_oracle_exact_key_overwrites_and_orthogonal_key_survives():
    # With zero log-decay and beta=1, writing the same unit key replaces its
    # associated value, while a write to an orthogonal key leaves it intact.
    state = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    state[0, 0, :, 0] = torch.tensor([3.0, -1.0])
    # Construct two sequential keys e0 and e1.
    q = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=torch.float64)
    k = q.clone()
    v = torch.tensor([[[[5.0, 7.0]], [[11.0, 13.0]]]], dtype=torch.float64)
    raw_gate = torch.full_like(q, -100.0)  # lower-bounded gate approaches zero
    beta = torch.full((1, 2, 1), 100.0, dtype=torch.float64)
    out, final_state = oracle_recurrent_kda(
        q, k, v, raw_gate, beta, torch.zeros(1, dtype=torch.float64),
        torch.zeros(1, 2, dtype=torch.float64), initial_state=state,
        compute_dtype=torch.float64,
    )

    torch.testing.assert_close(final_state[0, 0, :, 0], v[0, 0, 0], atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(final_state[0, 0, :, 1], v[0, 1, 0], atol=1e-8, rtol=1e-8)
    assert torch.isfinite(out).all()


def test_oracle_applies_decay_before_delta_correction():
    q = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    k = torch.ones_like(q)
    v = torch.full((1, 1, 1, 1), 10.0, dtype=torch.float64)
    raw_gate = torch.zeros_like(q)  # A_log=0 and no bias gives log-decay -2.5.
    beta = torch.zeros(1, 1, 1, dtype=torch.float64)  # sigmoid(0) = 0.5.
    initial_state = torch.full((1, 1, 1, 1), 2.0, dtype=torch.float64)
    output, final_state = oracle_recurrent_kda(
        q,
        k,
        v,
        raw_gate,
        beta,
        torch.zeros(1, dtype=torch.float64),
        torch.zeros(1, 1, dtype=torch.float64),
        initial_state=initial_state,
        compute_dtype=torch.float64,
    )
    decayed = 2.0 * torch.exp(torch.tensor(-2.5, dtype=torch.float64))
    expected = decayed + 0.5 * (10.0 - decayed)
    torch.testing.assert_close(final_state.flatten(), expected.reshape(1))
    torch.testing.assert_close(output.flatten(), expected.reshape(1))


def test_oracle_causal_prefix_and_batch_independence():
    inputs = make_inputs(batch=2, length=7)
    inputs["initial_state"] = None
    full, _ = oracle_recurrent_kda(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"],
        inputs["A_log"], inputs["dt_bias"], initial_state=None,
    )
    prefix = 4
    short, _ = oracle_recurrent_kda(
        inputs["q"][:, :prefix], inputs["k"][:, :prefix], inputs["v"][:, :prefix],
        inputs["g"][:, :prefix], inputs["beta"][:, :prefix], inputs["A_log"],
        inputs["dt_bias"], initial_state=None,
    )
    torch.testing.assert_close(full[:, :prefix], short)

    for batch_index in range(2):
        single, _ = oracle_recurrent_kda(
            inputs["q"][batch_index:batch_index + 1],
            inputs["k"][batch_index:batch_index + 1],
            inputs["v"][batch_index:batch_index + 1],
            inputs["g"][batch_index:batch_index + 1],
            inputs["beta"][batch_index:batch_index + 1],
            inputs["A_log"], inputs["dt_bias"], initial_state=None,
        )
        torch.testing.assert_close(full[batch_index:batch_index + 1], single)


def test_oracle_short_convolution_full_sequence_matches_decode():
    generator = torch.Generator().manual_seed(99)
    x = torch.randn(2, 9, 6, generator=generator)
    weight = torch.randn(6, 4, generator=generator)
    full, full_state = causal_depthwise_conv(x, weight)

    state = None
    pieces = []
    for index in range(x.shape[1]):
        piece, state = causal_depthwise_conv(x[:, index:index + 1], weight, initial_state=state)
        pieces.append(piece)
    decoded = torch.cat(pieces, dim=1)
    torch.testing.assert_close(decoded, full)
    torch.testing.assert_close(state, full_state)


def test_production_k3_gate_matches_independent_formula_and_gradients():
    module = require_kda()
    raw = torch.randn(2, 3, 2, 4, dtype=torch.float64, requires_grad=True)
    A_log = torch.randn(2, dtype=torch.float64, requires_grad=True)
    dt_bias = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    actual = module.k3_decay_gate(raw, A_log, dt_bias, lower_bound=-5.0)
    expected = oracle_decay_gate(raw, A_log, dt_bias)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)

    cotangent = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (raw, A_log, dt_bias), cotangent, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (raw, A_log, dt_bias), cotangent)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-9, rtol=1e-7)


@pytest.mark.parametrize(
    ("batch", "length", "heads", "key_dim", "value_dim"),
    [(1, 1, 1, 4, 3), (2, 7, 2, 4, 3), (1, 64, 2, 16, 11)],
)
def test_production_reference_forward_and_state_match_oracle(batch, length, heads, key_dim, value_dim):
    module = require_kda()
    inputs = make_inputs(
        batch=batch, length=length, heads=heads, key_dim=key_dim, value_dim=value_dim
    )
    actual, actual_state = call_operator(module, inputs)
    expected, expected_state = oracle_recurrent_kda(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"],
        inputs["A_log"], inputs["dt_bias"], initial_state=inputs["initial_state"],
    )

    assert actual.shape == (batch, length, heads, value_dim)
    assert actual_state.shape == (batch, heads, value_dim, key_dim)
    assert actual_state.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_state, expected_state, atol=1e-5, rtol=1e-5)


def test_production_reference_gradients_match_oracle():
    module = require_kda()
    base = make_inputs(batch=1, length=5, heads=2, key_dim=4, value_dim=3)
    actual_inputs = {name: value.clone().requires_grad_(True) for name, value in base.items()}
    expected_inputs = {name: value.clone().requires_grad_(True) for name, value in base.items()}

    actual, actual_state = call_operator(module, actual_inputs)
    expected, expected_state = oracle_recurrent_kda(
        expected_inputs["q"], expected_inputs["k"], expected_inputs["v"],
        expected_inputs["g"], expected_inputs["beta"], expected_inputs["A_log"],
        expected_inputs["dt_bias"], initial_state=expected_inputs["initial_state"],
    )
    output_cotangent = torch.randn_like(expected)
    state_cotangent = torch.randn_like(expected_state)
    names = ("q", "k", "v", "g", "beta", "A_log", "dt_bias", "initial_state")
    actual_grads = torch.autograd.grad(
        (actual, actual_state), tuple(actual_inputs[name] for name in names),
        (output_cotangent, state_cotangent),
    )
    expected_grads = torch.autograd.grad(
        (expected, expected_state), tuple(expected_inputs[name] for name in names),
        (output_cotangent, state_cotangent),
    )
    for name, actual_grad, expected_grad in zip(names, actual_grads, expected_grads):
        torch.testing.assert_close(
            actual_grad, expected_grad, atol=2e-4, rtol=2e-4,
            msg=lambda message, name=name: f"gradient mismatch for {name}: {message}",
        )


def test_production_reference_is_causal_and_resettable():
    module = require_kda()
    inputs = make_inputs(batch=2, length=8)
    inputs["initial_state"] = None
    full, _ = call_operator(module, inputs)
    prefix_inputs = {
        name: value[:, :4] if name in {"q", "k", "v", "g", "beta"} else value
        for name, value in inputs.items()
    }
    prefix, _ = call_operator(module, prefix_inputs)
    reset, _ = call_operator(module, inputs)
    torch.testing.assert_close(full[:, :4], prefix, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(full, reset, atol=0, rtol=0)


def test_production_rejects_k_first_initial_state_layout():
    module = require_kda()
    inputs = make_inputs(batch=1, length=2, heads=2, key_dim=4, value_dim=3)
    inputs["initial_state"] = torch.zeros(1, 2, 4, 3)
    with pytest.raises(ValueError, match=r"V-first|\[B, H, V, K\]"):
        call_operator(module, inputs)
