"""Correctness-first PyTorch implementation of Kimi K3 Delta Attention.

# Speed-supervisor dry-run candidate: documentation-only change; no executable
# KDA behavior, parameterization, backend selection, or fallback policy changes.

The implementation in this module is intentionally straightforward.  It is the
production CPU oracle and fallback against which optimized chunk/recurrent GPU
kernels are validated.  It does not import code from the local ``ref/`` tree.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import math
import os
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.cuda_compat import configure_triton_ptxas


@dataclass
class KDAState:
    """Fixed-size state for recurrent KDA decoding.

    ``memory`` uses the V-first ``[B,H,V,K]`` layout expected by K3 and
    FlashKDA.  The convolution states are ``[B,D,W]`` and contain the most
    recent projected inputs, including the current token.
    """

    memory: torch.Tensor
    q_conv: torch.Tensor
    k_conv: torch.Tensor
    v_conv: torch.Tensor


class Linear(nn.Linear):
    """Keep master weights in FP32 while matching activation dtype for matmul."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, self.weight.to(dtype=x.dtype), bias)


class RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        normalized = F.rms_norm(
            x,
            (x.shape[-1],),
            weight=self.weight.to(dtype=x.dtype),
            eps=self.eps,
        )
        return normalized * gate.sigmoid()


class ShortConvolution(nn.Conv1d):
    """Depthwise causal convolution with explicit, non-mutating cache state."""

    def __init__(self, hidden_size: int, kernel_size: int = 4):
        super().__init__(
            hidden_size,
            hidden_size,
            kernel_size,
            groups=hidden_size,
            bias=False,
            padding=kernel_size - 1,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.ndim != 3 or x.shape[-1] != self.in_channels:
            raise ValueError(f"x must have shape [B,T,{self.in_channels}]")
        batch, length, channels = x.shape
        width = self.kernel_size[0]
        if state is None:
            current = x.new_zeros(batch, channels, width)
        else:
            expected = (batch, channels, width)
            if state.shape != expected:
                raise ValueError(f"convolution state must have shape {expected}")
            current = state.to(device=x.device, dtype=x.dtype).clone()

        weight = self.weight[:, 0].to(dtype=x.dtype)
        outputs: list[torch.Tensor] = []
        for index in range(length):
            current = torch.cat((current[..., 1:], x[:, index].unsqueeze(-1)), dim=-1)
            outputs.append(F.silu((current * weight.unsqueeze(0)).sum(dim=-1)))
        output = torch.stack(outputs, dim=1) if outputs else x.new_empty(batch, 0, channels)
        return output, current if output_final_state else None


def _gate_shapes(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[int, int]:
    if raw_gate.ndim != 4:
        raise ValueError("raw_gate must have shape [B, T, H, K]")
    heads, key_dim = raw_gate.shape[-2:]
    if A_log.shape != (heads,):
        raise ValueError(f"A_log must have shape {(heads,)}")
    if dt_bias.numel() != heads * key_dim:
        raise ValueError(f"dt_bias must contain {heads * key_dim} values")
    return heads, key_dim


def k3_decay_gate(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float = -5.0,
) -> torch.Tensor:
    """K3 lower-bounded log decay: ``lb * sigmoid(exp(A) * (z+b))``."""

    heads, key_dim = _gate_shapes(raw_gate, A_log, dt_bias)
    if lower_bound >= 0:
        raise ValueError("lower_bound must be negative")
    return lower_bound * torch.sigmoid(
        A_log.exp().reshape(heads, 1) * (raw_gate + dt_bias.reshape(heads, key_dim))
    )


def _validate_operator_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta_logits: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    if q.shape != k.shape or q.shape != raw_gate.shape or q.ndim != 4:
        raise ValueError("q, k, and raw_gate must share shape [B, T, H, K]")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("v must have shape [B, T, H, V]")
    if beta_logits.shape != q.shape[:3]:
        raise ValueError("beta logits must have shape [B, T, H]")
    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if initial_state is not None:
        expected = (batch, heads, value_dim, key_dim)
        if initial_state.shape != expected:
            raise ValueError(f"initial_state must be V-first [B, H, V, K] with shape {expected}")
    return batch, length, heads, key_dim, value_dim


def _reference_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    lower_bound: float,
    scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, length, heads, key_dim, value_dim = _validate_operator_inputs(
        q, k, v, raw_gate, beta_logits, initial_state
    )
    _gate_shapes(raw_gate, A_log, dt_bias)
    query_scale = key_dim**-0.5 if scale is None else scale

    q_float = F.normalize(q.float(), p=2, dim=-1) * query_scale
    k_float = F.normalize(k.float(), p=2, dim=-1)
    v_float = v.float()
    decay = k3_decay_gate(
        raw_gate.float(), A_log.float(), dt_bias.float(), lower_bound=lower_bound
    )
    beta = beta_logits.float().sigmoid()

    if initial_state is None:
        state_vk = torch.zeros(
            batch,
            heads,
            value_dim,
            key_dim,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        state_vk = initial_state.float()
    state = state_vk.transpose(-1, -2)

    outputs: list[torch.Tensor] = []
    for index in range(length):
        qi = q_float[:, index]
        ki = k_float[:, index]
        vi = v_float[:, index]
        state = state * decay[:, index].exp().unsqueeze(-1)
        prediction = torch.einsum("bhk,bhkv->bhv", ki, state)
        residual = vi - prediction
        state = state + torch.einsum(
            "bhk,bhv->bhkv", beta[:, index].unsqueeze(-1) * ki, residual
        )
        outputs.append(torch.einsum("bhk,bhkv->bhv", qi, state))

    output = (
        torch.stack(outputs, dim=1)
        if outputs
        else v_float.new_empty(batch, 0, heads, value_dim)
    )
    return output.to(v.dtype), state.transpose(-1, -2).contiguous()


_FLA_OPS: tuple[Callable, Callable] | None = None


def _load_fla_ops() -> tuple[Callable, Callable]:
    """Load the pinned Triton operators only after SM121 is configured."""

    global _FLA_OPS
    if _FLA_OPS is not None:
        return _FLA_OPS

    configure_triton_ptxas(required=True)
    # Keep this research phase deterministic if optional accelerator packages
    # happen to be installed in the environment later.
    os.environ["FLA_FLASH_KDA"] = "0"
    os.environ["FLA_TILELANG"] = "0"
    try:
        fla_version = importlib.metadata.version("fla-core")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "the optimized KDA backend requires fla-core==0.5.2; "
            "run `uv sync --extra gpu`"
        ) from error
    if fla_version != "0.5.2":
        raise RuntimeError(f"optimized KDA requires fla-core==0.5.2, found {fla_version}")
    try:
        from fla.ops.kda import chunk_kda, fused_recurrent_kda
    except ModuleNotFoundError as error:
        if error.name == "fla" or (error.name and error.name.startswith("fla.")):
            raise RuntimeError(
                "the optimized KDA backend requires fla-core==0.5.2; "
                "run `uv sync --extra gpu`"
            ) from error
        raise
    _FLA_OPS = chunk_kda, fused_recurrent_kda
    return _FLA_OPS


def prepare_kda_backend() -> None:
    """Fail early and stabilize FLA imports before compiling a KDA model."""

    _load_fla_ops()


def _optimized_backend_reason(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    lower_bound: float,
) -> str | None:
    if not q.is_cuda:
        return "FLA KDA requires CUDA"
    activations = (q, k, v, raw_gate, beta_logits)
    if any(tensor.device != q.device for tensor in (*activations, A_log, dt_bias)):
        return "all KDA tensors must be on the same CUDA device"
    if any(tensor.dtype != torch.bfloat16 for tensor in activations):
        return "FLA KDA currently requires bfloat16 activations"
    if A_log.dtype != torch.float32 or dt_bias.dtype != torch.float32:
        return "A_log and dt_bias must be float32"
    if q.shape[-1] > 256:
        return "FLA KDA supports key head dimensions up to 256"
    if not (-5.0 <= lower_bound < 0.0):
        return "FLA safe-gate lower_bound must be in [-5, 0)"
    if initial_state is not None:
        if initial_state.device != q.device:
            return "initial_state must be on the same CUDA device as q"
        if initial_state.dtype != torch.float32:
            return "initial_state must be float32"
    return None


@torch.compiler.disable
def _run_fla_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    mode: str,
    lower_bound: float,
    scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    chunk_kda, fused_recurrent_kda = _load_fla_ops()
    common = {
        "A_log": A_log,
        # FLA's public gradient contract is flat [H*K]. The view preserves the
        # caller's original dt_bias shape through autograd.
        "dt_bias": dt_bias.reshape(-1),
        "scale": scale,
        "initial_state": None if initial_state is None else initial_state.contiguous(),
        "output_final_state": output_final_state,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "lower_bound": lower_bound,
        "state_v_first": True,
    }
    inputs = tuple(tensor.contiguous() for tensor in (q, k, v, raw_gate, beta_logits))
    if mode == "chunk":
        return chunk_kda(*inputs, safe_gate=True, **common)
    return fused_recurrent_kda(*inputs, **common)


def kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    mode: str = "reference",
    lower_bound: float = -5.0,
    scale: float | None = None,
    allow_fallback: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run KDA through the requested backend.

    ``reference`` always uses the correctness implementation. ``chunk`` and
    ``recurrent`` select the pinned FLA Triton kernels when their CUDA/BF16
    contract is satisfied. Unsupported calls use the reference only when
    ``allow_fallback=True``; kernel failures are never swallowed.
    """

    if mode not in {"reference", "chunk", "recurrent"}:
        raise ValueError(f"unknown KDA mode: {mode}")
    if mode != "reference":
        reason = _optimized_backend_reason(
            q, k, v, raw_gate, beta_logits, A_log, dt_bias, initial_state,
            lower_bound=lower_bound,
        )
        recurrent_gradients = mode == "recurrent" and torch.is_grad_enabled() and any(
            tensor.requires_grad
            for tensor in (q, k, v, raw_gate, beta_logits, A_log, dt_bias)
        )
        if recurrent_gradients:
            reason = "FLA recurrent KDA is inference-only"
        if reason is None:
            return _run_fla_kda(
                q,
                k,
                v,
                raw_gate,
                beta_logits,
                A_log,
                dt_bias,
                initial_state=initial_state,
                output_final_state=output_final_state,
                mode=mode,
                lower_bound=lower_bound,
                scale=scale,
            )
        if not allow_fallback:
            raise RuntimeError(f"optimized {mode} KDA backend unavailable: {reason}")
    output, final_state = _reference_kda(
        q,
        k,
        v,
        raw_gate,
        beta_logits,
        A_log,
        dt_bias,
        initial_state=initial_state,
        lower_bound=lower_bound,
        scale=scale,
    )
    return output, final_state if output_final_state else None


class KimiDeltaAttention(nn.Module):
    """K3-compatible KDA mixer with short convolution and full-rank output gate."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_size: int = 4,
        layer_idx: int | None = None,
        mode: str = "reference",
        norm_eps: float = 1e-5,
        lower_bound: float = -5.0,
        allow_fallback: bool = True,
    ):
        super().__init__()
        if hidden_size != num_heads * head_dim:
            raise ValueError("initial KDA implementation requires hidden_size = num_heads * head_dim")
        if conv_size <= 0:
            raise ValueError("conv_size must be positive")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.conv_size = conv_size
        self.layer_idx = layer_idx
        self.mode = mode
        self.lower_bound = lower_bound
        self.allow_fallback = allow_fallback

        self.q_proj = Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = Linear(hidden_size, hidden_size, bias=False)
        self.q_conv1d = ShortConvolution(hidden_size, conv_size)
        self.k_conv1d = ShortConvolution(hidden_size, conv_size)
        self.v_conv1d = ShortConvolution(hidden_size, conv_size)

        self.A_log = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
        self.f_a_proj = Linear(hidden_size, head_dim, bias=False)
        self.f_b_proj = Linear(head_dim, hidden_size, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.b_proj = Linear(hidden_size, num_heads, bias=False)

        self.g_proj = Linear(hidden_size, hidden_size, bias=False)
        self.o_norm = RMSNormGated(head_dim, eps=norm_eps)
        self.o_proj = Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: KDAState | None = None,
        output_final_state: bool = False,
        mode: str | None = None,
        allow_fallback: bool | None = None,
    ) -> tuple[torch.Tensor, KDAState | None]:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden_states must have shape [B,T,{self.hidden_size}]")
        batch, length, _ = hidden_states.shape
        requested_mode = self.mode if mode is None else mode
        requested_fallback = self.allow_fallback if allow_fallback is None else allow_fallback
        want_state = output_final_state

        q_state = None if state is None else state.q_conv
        k_state = None if state is None else state.k_conv
        v_state = None if state is None else state.v_conv
        memory = None if state is None else state.memory
        q, q_final = self.q_conv1d(
            self.q_proj(hidden_states), q_state, output_final_state=want_state
        )
        k, k_final = self.k_conv1d(
            self.k_proj(hidden_states), k_state, output_final_state=want_state
        )
        v, v_final = self.v_conv1d(
            self.v_proj(hidden_states), v_state, output_final_state=want_state
        )

        q = q.reshape(batch, length, self.num_heads, self.head_dim)
        k = k.reshape(batch, length, self.num_heads, self.head_dim)
        v = v.reshape(batch, length, self.num_heads, self.head_dim)
        raw_gate = self.f_b_proj(self.f_a_proj(hidden_states)).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        beta_logits = self.b_proj(hidden_states)
        mixed, final_memory = kda(
            q,
            k,
            v,
            raw_gate,
            beta_logits,
            self.A_log,
            self.dt_bias,
            initial_state=memory,
            output_final_state=want_state,
            mode=requested_mode,
            lower_bound=self.lower_bound,
            allow_fallback=requested_fallback,
        )

        output_gate = self.g_proj(hidden_states).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        mixed = self.o_norm(mixed, output_gate)
        output = self.o_proj(mixed.reshape(batch, length, self.hidden_size))
        final_state = None
        if want_state:
            final_state = KDAState(final_memory, q_final, k_final, v_final)
        return output, final_state


__all__ = ["KDAState", "KimiDeltaAttention", "k3_decay_gate", "kda", "prepare_kda_backend"]
