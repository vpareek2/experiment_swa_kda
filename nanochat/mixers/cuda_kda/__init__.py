"""Minimal loader and provenance for the complete naive project CUDA backend."""
from __future__ import annotations

import os
from typing import Any

_RECEIPT: dict[str, object] | None = None
_CHUNK_SOURCE = "nanochat/mixers/cuda_kda/chunk.cu"
_CHUNK_WY_FORWARD_SOURCE = "nanochat/mixers/cuda_kda/chunk_wy_forward.cu"
_CHUNK_WY_BACKWARD_SOURCE = "nanochat/mixers/cuda_kda/chunk_wy_backward.cu"
_RECURRENT_SOURCE = "nanochat/mixers/cuda_kda/recurrent_decode.cu"
_CONVOLUTION_FORWARD_SOURCE = "nanochat/mixers/cuda_kda/causal_convolution_forward.cu"
_CONVOLUTION_BACKWARD_SOURCE = "nanochat/mixers/cuda_kda/causal_convolution_backward.cu"
_SOURCES = (
    _CHUNK_SOURCE,
    _CHUNK_WY_FORWARD_SOURCE,
    _CHUNK_WY_BACKWARD_SOURCE,
    _RECURRENT_SOURCE,
    _CONVOLUTION_FORWARD_SOURCE,
    _CONVOLUTION_BACKWARD_SOURCE,
)
_COMPONENTS = (
    "chunk_forward",
    "chunk_backward",
    "recurrent_decode",
    "causal_convolution_forward",
    "causal_convolution_backward",
)


def prepare() -> None:
    global _RECEIPT
    if _RECEIPT is not None:
        return
    from nanochat.research.cuda_build import build_cuda_extension

    extra_cuda_cflags = ["--use_fast_math"]
    if os.environ.get("NANOCHAT_DISABLE_SELECTIVE_PTX") == "1":
        extra_cuda_cflags.append("-DNANOCHAT_DISABLE_SELECTIVE_PTX=1")
    _RECEIPT = build_cuda_extension(
        list(_SOURCES),
        name="nanochat_kda_complete_naive",
        extra_cuda_cflags=tuple(extra_cuda_cflags),
    )


def provenance() -> dict[str, Any]:
    components = {
        name: {
            "owner": "third_party",
            "sources": [],
            "kernel_symbols": [],
            "torch_operator": None,
        }
        for name in _COMPONENTS
    }
    components["chunk_forward"] = {
        "owner": "project",
        "sources": [_CHUNK_SOURCE, _CHUNK_WY_FORWARD_SOURCE],
        # The canonical symbol remains the compatibility-path profile anchor;
        # the exact-shape WY scan has its own descriptive CUDA symbol.
        "kernel_symbols": ["nanochat_kda_chunk_forward_kernel"],
        "torch_operator": "nanochat_kda::chunk_forward",
    }
    components["chunk_backward"] = {
        "owner": "project",
        "sources": [_CHUNK_SOURCE, _CHUNK_WY_BACKWARD_SOURCE],
        "kernel_symbols": ["nanochat_kda_chunk_backward_kernel"],
        "torch_operator": "nanochat_kda::chunk_backward",
    }
    components["recurrent_decode"] = {
        "owner": "project",
        "sources": [_RECURRENT_SOURCE],
        "kernel_symbols": ["nanochat_kda_recurrent_decode_kernel"],
        "torch_operator": "nanochat_kda::recurrent_decode",
    }
    components["causal_convolution_forward"] = {
        "owner": "project",
        "sources": [_CONVOLUTION_FORWARD_SOURCE],
        "kernel_symbols": ["nanochat_kda_causal_convolution_forward_kernel"],
        "torch_operator": "nanochat_kda::causal_convolution_forward",
    }
    components["causal_convolution_backward"] = {
        "owner": "project",
        "sources": [_CONVOLUTION_BACKWARD_SOURCE],
        "kernel_symbols": ["nanochat_kda_causal_convolution_backward_kernel"],
        "torch_operator": "nanochat_kda::causal_convolution_backward",
    }
    return {
        "schema_version": 1,
        "components": components,
        "build": None if _RECEIPT is None else dict(_RECEIPT),
        "selective_ptx": [{
            "source": _CHUNK_WY_FORWARD_SOURCE,
            "rationale": (
                "Use documented warp mma.sync m16n8k16 and ldmatrix only for "
                "the GB10 register-held forward state recurrence"
            ),
            "architecture_guard": (
                "The project extension builds a native sm_121 image with CUDA "
                "13.x; the selected instructions require baseline sm_80+"
            ),
            "cuda_fallback": (
                "NANOCHAT_DISABLE_SELECTIVE_PTX=1 builds the exact standard-CUDA "
                "WMMA shared-state recurrence and launches its 256-thread CTA"
            ),
            "profile_evidence": (
                "runs/kda-cuda-development/profiles/attempt-00211-gb10-register-"
                "forward-state-profile-001"
            ),
        }],
    }
