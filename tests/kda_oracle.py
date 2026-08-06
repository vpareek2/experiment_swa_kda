"""Independent mathematical oracle for Kimi Delta Attention tests.

This module intentionally depends only on PyTorch.  It must not import the
production KDA implementation or anything under ``ref/``: sharing code with the
implementation would make parity tests incapable of detecting shared bugs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass
class OracleKDAState:
    """Canonical decode state used by the tests.

    Memory is V-first to match K3/FlashKDA.  Convolution states retain the last
    ``kernel_size`` projected inputs, including left-padding zeros.
    """

    memory: torch.Tensor
    q_conv: torch.Tensor
    k_conv: torch.Tensor
    v_conv: torch.Tensor


def k3_decay_gate(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float = -5.0,
) -> torch.Tensor:
    """Apply the K3 lower-bounded log-decay parameterization."""

    if raw_gate.ndim != 4:
        raise ValueError("raw_gate must have shape [B, T, H, K]")
    heads, key_dim = raw_gate.shape[-2:]
    if A_log.shape != (heads,):
        raise ValueError(f"A_log must have shape {(heads,)}")
    if dt_bias.numel() != heads * key_dim:
        raise ValueError(f"dt_bias must contain {heads * key_dim} values")
    bias = dt_bias.reshape(heads, key_dim)
    return lower_bound * torch.sigmoid(A_log.exp().reshape(heads, 1) * (raw_gate + bias))


def recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    lower_bound: float = -5.0,
    scale: float | None = None,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-loop KDA recurrence with a V-first public state.

    Shapes are q/k ``[B,T,H,K]``, v ``[B,T,H,V]``, and state
    ``[B,H,V,K]``. Q/K normalization, beta sigmoid, and K3 gate activation
    are deliberately included so this is an oracle for the fused public
    operator rather than only its already-activated inner recurrence.
    """

    if q.shape != k.shape or q.shape != raw_gate.shape:
        raise ValueError("q, k, and raw_gate must have identical shapes")
    if q.shape[:3] != v.shape[:3]:
        raise ValueError("q/k and v must share B, T, and H")
    if beta_logits.shape != q.shape[:3]:
        raise ValueError("beta_logits must have shape [B, T, H]")

    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    scale = key_dim**-0.5 if scale is None else scale

    qn = F.normalize(q.to(compute_dtype), p=2, dim=-1) * scale
    kn = F.normalize(k.to(compute_dtype), p=2, dim=-1)
    values = v.to(compute_dtype)
    decay = k3_decay_gate(
        raw_gate.to(compute_dtype),
        A_log.to(compute_dtype),
        dt_bias.to(compute_dtype),
        lower_bound,
    )
    beta = beta_logits.to(compute_dtype).sigmoid()

    if initial_state is None:
        state_vk = torch.zeros(
            batch,
            heads,
            value_dim,
            key_dim,
            dtype=compute_dtype,
            device=q.device,
        )
    else:
        expected = (batch, heads, value_dim, key_dim)
        if initial_state.shape != expected:
            raise ValueError(f"initial_state must have shape {expected}")
        state_vk = initial_state.to(compute_dtype)

    # The recurrence is clearest in K-first [B,H,K,V] layout.  Its public
    # boundary remains V-first, which also catches accidental transpositions.
    state = state_vk.transpose(-1, -2)
    outputs = []
    for index in range(length):
        qi = qn[:, index]
        ki = kn[:, index]
        vi = values[:, index]
        gi = decay[:, index]
        bi = beta[:, index]

        state = state * gi.exp().unsqueeze(-1)
        prediction = torch.einsum("bhk,bhkv->bhv", ki, state)
        residual = vi - prediction
        state = state + torch.einsum("bhk,bhv->bhkv", bi.unsqueeze(-1) * ki, residual)
        outputs.append(torch.einsum("bhk,bhkv->bhv", qi, state))

    output = torch.stack(outputs, dim=1) if outputs else values.new_empty(batch, 0, heads, value_dim)
    return output.to(v.dtype), state.transpose(-1, -2).contiguous()


def causal_depthwise_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    activation: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit token-loop causal depthwise convolution.

    ``x`` is ``[B,T,D]``, weight is ``[D,W]``, and state is ``[B,D,W]``.
    The state includes the current token, matching the short-convolution cache
    convention used by Kimi Linear and FLA.
    """

    if x.ndim != 3 or weight.ndim != 2 or x.shape[-1] != weight.shape[0]:
        raise ValueError("expected x [B,T,D] and weight [D,W]")
    batch, length, channels = x.shape
    width = weight.shape[-1]
    if initial_state is None:
        state = x.new_zeros(batch, channels, width)
    else:
        if initial_state.shape != (batch, channels, width):
            raise ValueError("invalid convolution state shape")
        state = initial_state.clone()

    outputs = []
    for index in range(length):
        state = torch.cat((state[..., 1:], x[:, index].unsqueeze(-1)), dim=-1)
        y = (state * weight.unsqueeze(0)).sum(dim=-1)
        outputs.append(F.silu(y) if activation else y)
    output = torch.stack(outputs, dim=1) if outputs else x.new_empty(batch, 0, channels)
    return output, state


def rmsnorm_gated(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """K3 output normalization followed by a sigmoid gate."""

    normalized = F.rms_norm(x, (x.shape[-1],), weight=weight.to(x.dtype), eps=eps)
    return normalized * gate.sigmoid()


def full_layer(
    x: torch.Tensor,
    *,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    q_conv_weight: torch.Tensor,
    k_conv_weight: torch.Tensor,
    v_conv_weight: torch.Tensor,
    forget_a_weight: torch.Tensor,
    forget_b_weight: torch.Tensor,
    beta_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    output_gate_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    output_weight: torch.Tensor,
    num_heads: int,
    norm_eps: float,
    initial_state: OracleKDAState | None = None,
    lower_bound: float = -5.0,
) -> tuple[torch.Tensor, OracleKDAState]:
    """Independently compose the complete K3 mixer from explicit weights."""

    batch, _, _ = x.shape
    q_projected = F.linear(x, q_weight)
    k_projected = F.linear(x, k_weight)
    v_projected = F.linear(x, v_weight)

    q_cache = None if initial_state is None else initial_state.q_conv
    k_cache = None if initial_state is None else initial_state.k_conv
    v_cache = None if initial_state is None else initial_state.v_conv
    q, q_final = causal_depthwise_conv(q_projected, q_conv_weight, initial_state=q_cache)
    k, k_final = causal_depthwise_conv(k_projected, k_conv_weight, initial_state=k_cache)
    v, v_final = causal_depthwise_conv(v_projected, v_conv_weight, initial_state=v_cache)

    key_dim = q.shape[-1] // num_heads
    value_dim = v.shape[-1] // num_heads
    q = q.reshape(batch, -1, num_heads, key_dim)
    k = k.reshape(batch, -1, num_heads, key_dim)
    v = v.reshape(batch, -1, num_heads, value_dim)
    raw_gate = F.linear(F.linear(x, forget_a_weight), forget_b_weight)
    raw_gate = raw_gate.reshape(batch, -1, num_heads, key_dim)
    beta_logits = F.linear(x, beta_weight)
    memory = None if initial_state is None else initial_state.memory
    mixed, final_memory = recurrent_kda(
        q,
        k,
        v,
        raw_gate,
        beta_logits,
        A_log,
        dt_bias,
        initial_state=memory,
        lower_bound=lower_bound,
    )

    output_gate = F.linear(x, output_gate_weight).reshape(batch, -1, num_heads, value_dim)
    mixed = rmsnorm_gated(mixed, output_gate, norm_weight, norm_eps)
    output = F.linear(mixed.flatten(2), output_weight)
    return output, OracleKDAState(final_memory, q_final, k_final, v_final)


def expected_single_token_state(
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Closed-form zero-state update, returned in V-first layout."""

    state_kv = torch.einsum("bhk,bhv->bhkv", beta.unsqueeze(-1) * key, value)
    return state_kv.transpose(-1, -2).contiguous()


def default_scale(key_dim: int) -> float:
    return 1.0 / math.sqrt(key_dim)
