"""Marked CUDA parity lane for optimized KDA training and decode backends."""

from __future__ import annotations

import importlib

import pytest
import torch

from kda_oracle import recurrent_kda as oracle_recurrent_kda
from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="KDA optimized parity requires CUDA"),
]


def require_kda():
    try:
        return importlib.import_module("nanochat.kda")
    except ModuleNotFoundError as error:
        if error.name == "nanochat.kda":
            pytest.fail("nanochat.kda is not implemented (expected red TDD failure)")
        raise


def make_cuda_inputs(*, length, dim, batch=1, heads=2, requires_grad=False):
    torch.manual_seed(20260805 + length + dim)
    device = torch.device("cuda")

    def activation(shape):
        return torch.randn(*shape, device=device, dtype=torch.bfloat16).requires_grad_(requires_grad)

    q = activation((batch, length, heads, dim))
    k = activation((batch, length, heads, dim))
    v = activation((batch, length, heads, dim))
    g = activation((batch, length, heads, dim))
    beta = activation((batch, length, heads))
    A_log = (torch.randn(heads, device=device, dtype=torch.float32) * 0.2).requires_grad_(requires_grad)
    dt_bias = (torch.randn(heads, dim, device=device, dtype=torch.float32) * 0.3).requires_grad_(requires_grad)
    initial_state = (
        torch.randn(batch, heads, dim, dim, device=device, dtype=torch.float32) * 0.05
    ).requires_grad_(requires_grad)
    return q, k, v, g, beta, A_log, dt_bias, initial_state


def call_backend(module, tensors, *, mode):
    q, k, v, g, beta, A_log, dt_bias, initial_state = tensors
    return module.kda(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        initial_state=initial_state,
        output_final_state=True,
        mode=mode,
        lower_bound=-5.0,
        allow_fallback=False,
    )


def make_cuda_gpt(*, layers=1):
    config = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=layers,
        n_head=1,
        n_kv_head=1,
        n_embd=128,
        window_pattern="K",
        sliding_window=4,
        force_final_full=False,
        kda_backend="fla_triton",
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cuda")
    model.init_weights()
    generator = torch.Generator(device="cuda").manual_seed(20260806)
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.o_proj.weight.normal_(std=0.01, generator=generator)
            block.mlp.c_proj.weight.normal_(std=0.01, generator=generator)
    return model


@pytest.mark.parametrize("length", [31, 32, 33, 63, 64, 65])
def test_chunk_bf16_matches_oracle_across_boundaries(length):
    module = require_kda()
    tensors = make_cuda_inputs(length=length, dim=64)
    actual, actual_state = call_backend(module, tensors, mode="chunk")
    expected, expected_state = oracle_recurrent_kda(
        *tensors[:7], initial_state=tensors[7], compute_dtype=torch.float32
    )

    assert actual.dtype == torch.bfloat16
    assert actual_state.dtype == torch.float32
    assert actual_state.shape == (1, 2, 64, 64)
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(actual_state, expected_state, atol=5e-3, rtol=5e-3)


def test_chunk_production_head_dimension_128_matches_oracle():
    module = require_kda()
    tensors = make_cuda_inputs(length=65, dim=128)
    actual, actual_state = call_backend(module, tensors, mode="chunk")
    expected, expected_state = oracle_recurrent_kda(
        *tensors[:7], initial_state=tensors[7], compute_dtype=torch.float32
    )
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(actual_state, expected_state, atol=5e-3, rtol=5e-3)


def test_chunk_backward_matches_independent_oracle():
    module = require_kda()
    actual_inputs = make_cuda_inputs(length=33, dim=64, requires_grad=True)
    expected_inputs = tuple(value.detach().clone().requires_grad_(True) for value in actual_inputs)

    actual, actual_state = call_backend(module, actual_inputs, mode="chunk")
    expected, expected_state = oracle_recurrent_kda(
        *expected_inputs[:7], initial_state=expected_inputs[7], compute_dtype=torch.float32
    )
    output_cotangent = torch.randn_like(actual)
    state_cotangent = torch.randn_like(actual_state)
    actual_grads = torch.autograd.grad(
        (actual, actual_state), actual_inputs, (output_cotangent, state_cotangent)
    )
    expected_grads = torch.autograd.grad(
        (expected, expected_state), expected_inputs, (output_cotangent, state_cotangent)
    )

    names = ("q", "k", "v", "raw_gate", "beta", "A_log", "dt_bias", "initial_state")
    for name, actual_grad, expected_grad in zip(names, actual_grads, expected_grads):
        tolerance = 2e-2 if name in {"raw_gate", "beta", "A_log", "dt_bias"} else 8e-3
        torch.testing.assert_close(
            actual_grad,
            expected_grad,
            atol=tolerance,
            rtol=tolerance,
            msg=lambda message, name=name: f"optimized gradient mismatch for {name}: {message}",
        )


def test_recurrent_decode_matches_chunk_with_same_nonzero_state():
    module = require_kda()
    tensors = make_cuda_inputs(length=17, dim=128)
    chunk_output, chunk_state = call_backend(module, tensors, mode="chunk")

    q, k, v, g, beta, A_log, dt_bias, state = tensors
    pieces = []
    for index in range(q.shape[1]):
        piece_inputs = (
            q[:, index:index + 1],
            k[:, index:index + 1],
            v[:, index:index + 1],
            g[:, index:index + 1],
            beta[:, index:index + 1],
            A_log,
            dt_bias,
            state,
        )
        piece, state = call_backend(module, piece_inputs, mode="recurrent")
        pieces.append(piece)
    recurrent_output = torch.cat(pieces, dim=1)

    torch.testing.assert_close(recurrent_output, chunk_output, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(state, chunk_state, atol=5e-3, rtol=5e-3)


def test_extreme_k3_gate_logits_remain_finite_over_long_sequence():
    module = require_kda()
    tensors = list(make_cuda_inputs(length=256, dim=64, requires_grad=True))
    raw_gate = tensors[3]
    with torch.no_grad():
        raw_gate[..., ::2].fill_(-100.0)
        raw_gate[..., 1::2].fill_(100.0)
    output, state = call_backend(module, tuple(tensors), mode="chunk")
    loss = output.float().square().mean() + state.square().mean()
    gradients = torch.autograd.grad(loss, tensors, allow_unused=False)

    assert torch.isfinite(output).all()
    assert torch.isfinite(state).all()
    for gradient in gradients:
        assert torch.isfinite(gradient).all()


def test_chunk_is_fail_closed_and_does_not_use_reference(monkeypatch):
    module = require_kda()
    implementation = importlib.import_module("nanochat.mixers.kda")

    def forbidden_reference(*args, **kwargs):
        raise AssertionError("optimized no-fallback call reached the reference loop")

    monkeypatch.setattr(implementation, "_reference_kda", forbidden_reference)
    output, state = call_backend(module, make_cuda_inputs(length=33, dim=64), mode="chunk")
    assert torch.isfinite(output).all()
    assert torch.isfinite(state).all()


def test_chunk_preserves_nonzero_initial_state_tensor():
    module = require_kda()
    tensors = make_cuda_inputs(length=33, dim=64)
    initial_state = tensors[-1]
    original = initial_state.clone()
    call_backend(module, tensors, mode="chunk")
    torch.testing.assert_close(initial_state, original, atol=0, rtol=0)


def test_optimized_modes_reject_unsupported_execution_contracts():
    module = require_kda()
    tensors = list(make_cuda_inputs(length=1, dim=64))
    tensors[0] = tensors[0].float()
    with pytest.raises(RuntimeError, match="requires bfloat16"):
        call_backend(module, tuple(tensors), mode="chunk")

    tensors = make_cuda_inputs(length=1, dim=64, requires_grad=True)
    with pytest.raises(RuntimeError, match="inference-only"):
        call_backend(module, tensors, mode="recurrent")


def test_optimized_gpt_prefill_and_recurrent_decode_match_one_shot():
    model = make_cuda_gpt().eval()
    tokens = torch.randint(0, 64, (1, 9), device="cuda")
    cache = KVCache(
        batch_size=1,
        num_heads=1,
        seq_len=16,
        head_dim=128,
        num_layers=1,
        mixer_types=model.mixer_types,
        conv_size=4,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    with torch.inference_mode():
        prefill = model(tokens[:, :4], kv_cache=cache)
        expected = model(tokens[:, :4])
        torch.testing.assert_close(prefill, expected, atol=5e-3, rtol=5e-3)
        for position in range(4, tokens.shape[1]):
            decoded = model(tokens[:, position:position + 1], kv_cache=cache)
            one_shot = model(tokens[:, :position + 1])[:, -1:]
            torch.testing.assert_close(decoded, one_shot, atol=5e-3, rtol=5e-3)


@pytest.mark.slow
def test_compiled_production_dim_gpt_forward_backward_is_finite():
    module = require_kda()
    module.prepare_kda_backend()
    model = make_cuda_gpt()
    compiled = torch.compile(model, dynamic=False)
    tokens = torch.randint(0, 64, (1, 8), device="cuda")
    targets = torch.randint(0, 64, (1, 8), device="cuda")
    loss = compiled(tokens, targets=targets)
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
