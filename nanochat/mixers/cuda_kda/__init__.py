"""Candidate-owned native CUDA KDA backend boundary.

The foundation is deliberately fail-closed. Autoresearch replaces this module
and adds tracked CUDA/build sources in this package. ``provenance()`` is an
immutable claim consumed by protected routing: each atomic unit is either
``project`` or ``third_party``. Protected code calls FLA only for unclaimed
units during bootstrap/migration; optimization requires every unit project.
"""
from __future__ import annotations
from typing import Any

_COMPONENTS = (
    "chunk_forward", "chunk_backward", "recurrent_decode",
    "causal_convolution_forward", "causal_convolution_backward",
)


def prepare() -> None:
    raise NotImplementedError("project-owned CUDA KDA backend is not implemented")


def provenance() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "components": {
            name: {"owner": "third_party", "sources": [], "kernel_symbols": [], "torch_operator": None}
            for name in _COMPONENTS
        },
        "build": None,
        "selective_ptx": [],
    }
