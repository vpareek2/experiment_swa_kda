"""Minimal loader and provenance for the native recurrent decode unit."""
from __future__ import annotations

from typing import Any

_RECEIPT: dict[str, object] | None = None
_SOURCE = "nanochat/mixers/cuda_kda/recurrent_decode.cu"
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

    _RECEIPT = build_cuda_extension([_SOURCE], name="nanochat_kda_recurrent")


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
    components["recurrent_decode"] = {
        "owner": "project",
        "sources": [_SOURCE],
        "kernel_symbols": ["nanochat_kda_recurrent_decode_kernel"],
        "torch_operator": "nanochat_kda::recurrent_decode",
    }
    return {
        "schema_version": 1,
        "components": components,
        "build": None if _RECEIPT is None else dict(_RECEIPT),
        "selective_ptx": [],
    }
