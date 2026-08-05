from __future__ import annotations

from dataclasses import asdict
import math
import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import Block, GPTConfig, Linear, norm
from nanochat.research.config import MemoryProbeConfig
from nanochat.research.memory import MemoryTaskSpec, generate_memory_batch, score_answer_tokens


class CanonicalProbeModel(nn.Module):
    """Small protected shell that reuses nanochat's actual sequence-mixing block."""

    def __init__(self, config: MemoryProbeConfig, window_pattern: str, force_final_full: bool = True):
        super().__init__()
        if any(char not in "LS" for char in window_pattern.upper()):
            raise ValueError("the v1 probe supports full and sliding-window mixers; KDA registers later")
        n_head = config.width // config.head_dim
        model_config = GPTConfig(
            sequence_len=max(config.lengths),
            vocab_size=config.vocab_size,
            n_layer=config.depth,
            n_head=n_head,
            n_kv_head=n_head,
            n_embd=config.width,
            window_pattern=window_pattern,
        )
        self.config = config
        self.model_config = model_config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList([Block(model_config, index) for index in range(config.depth)])
        self.lm_head = Linear(config.width, config.vocab_size, bias=False)
        self.window_sizes = [
            (-1, 0) if window_pattern[index % len(window_pattern)].upper() == "L" else (config.window_size, 0)
            for index in range(config.depth)
        ]
        if force_final_full:
            self.window_sizes[-1] = (-1, 0)
        head_dim = config.head_dim
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (100000 ** (channel_range / head_dim))
        positions = torch.arange(max(config.lengths), dtype=torch.float32)
        frequencies = torch.outer(positions, inv_freq)
        self.register_buffer("cos", frequencies.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin", frequencies.sin()[None, :, None, :], persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        scale = math.sqrt(3) * self.config.width ** -0.5
        for block in self.blocks:
            nn.init.uniform_(block.attn.c_q.weight, -scale, scale)
            nn.init.uniform_(block.attn.c_k.weight, -scale, scale)
            nn.init.uniform_(block.attn.c_v.weight, -scale, scale)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.c_fc.weight, -0.4 * scale, 0.4 * scale)
            nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.ve_gate is not None:
                nn.init.zeros_(block.attn.ve_gate.weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        length = token_ids.shape[1]
        if length > self.cos.shape[1]:
            raise ValueError(f"sequence length {length} exceeds probe maximum {self.cos.shape[1]}")
        compute_dtype = COMPUTE_DTYPE if token_ids.device.type == "cuda" else torch.float32
        x = norm(self.embedding(token_ids).to(compute_dtype))
        cos_sin = (
            self.cos[:, :length].to(device=x.device, dtype=x.dtype),
            self.sin[:, :length].to(device=x.device, dtype=x.dtype),
        )
        for index, block in enumerate(self.blocks):
            x = block(x, None, cos_sin, self.window_sizes[index], None)
        return self.lm_head(norm(x)).float()


def _training_spec(config: MemoryProbeConfig, rng: random.Random) -> MemoryTaskSpec:
    load = rng.choice((4, 16, 32))
    queries = min(4, load)
    return MemoryTaskSpec(
        sequence_length=config.train_sequence_length,
        num_pairs=load,
        num_queries=queries,
        updates_per_query_key=rng.choice((1, 2, 4)),
        distractor_ratio=rng.choice((0, 1)),
        vocab_size=config.vocab_size,
    )


def train_probe(
    config: MemoryProbeConfig,
    window_pattern: str,
    seed: int,
    device: torch.device,
    force_final_full: bool = True,
) -> tuple[CanonicalProbeModel, dict[str, Any]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    model = CanonicalProbeModel(config, window_pattern, force_final_full).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    steps = config.training_tokens // (config.batch_size * config.train_sequence_length)
    rng = random.Random(seed)
    losses: list[float] = []
    model.train()
    for step in range(steps):
        spec = _training_spec(config, rng)
        seeds = [seed * 10_000_000 + step * config.batch_size + index for index in range(config.batch_size)]
        inputs, labels, _ = generate_memory_batch(spec, seeds, device=device)
        logits = model(inputs)
        mask = labels >= 0
        loss = F.cross_entropy(logits[mask], labels[mask])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite memory-probe loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return model, {
        "training_steps": steps,
        "training_tokens": steps * config.batch_size * config.train_sequence_length,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "num_params": sum(parameter.numel() for parameter in model.parameters()),
    }


@torch.inference_mode()
def _evaluate_cell(
    model: nn.Module,
    spec: MemoryTaskSpec,
    examples: int,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    total_answers = 0
    weighted_nll = 0.0
    weighted_accuracy = 0.0
    all_distances: list[int] = []
    model.eval()
    for offset in range(0, examples, batch_size):
        count = min(batch_size, examples - offset)
        seeds = [seed + offset + index for index in range(count)]
        inputs, labels, metadata = generate_memory_batch(spec, seeds, device=device)
        scores = score_answer_tokens(model(inputs), labels)
        total_answers += scores["answer_tokens"]
        weighted_nll += scores["nll"] * scores["answer_tokens"]
        weighted_accuracy += scores["accuracy"] * scores["answer_tokens"]
        for item in metadata:
            all_distances.extend(item["query_distances"])
    return {
        "spec": asdict(spec),
        "examples": examples,
        "answer_tokens": total_answers,
        "nll": weighted_nll / total_answers,
        "accuracy": weighted_accuracy / total_answers,
        "mean_query_distance": sum(all_distances) / len(all_distances),
        "chance_accuracy": 1.0 / 64.0,
    }


def evaluate_probe(
    model: nn.Module,
    config: MemoryProbeConfig,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    cells: dict[str, list[dict[str, Any]]] = {"load": [], "updates": [], "interference": [], "boundary": []}
    cell_seed = seed * 100_000_000
    ordinal = 0

    def evaluate(group: str, spec: MemoryTaskSpec):
        nonlocal ordinal
        result = _evaluate_cell(
            model, spec, config.examples_per_cell, cell_seed + ordinal * config.examples_per_cell,
            config.batch_size, device,
        )
        cells[group].append(result)
        ordinal += 1

    for length in config.lengths:
        for load in config.loads:
            if 1 + 3 * load + 3 * min(4, load) <= length + 1:
                evaluate("load", MemoryTaskSpec(length, load, min(4, load), vocab_size=config.vocab_size))
    for updates in config.updates:
        evaluate("updates", MemoryTaskSpec(512, 16, 1, updates, vocab_size=config.vocab_size))
    for ratio in config.distractor_ratios:
        evaluate("interference", MemoryTaskSpec(1024, 16, 4, 1, ratio, vocab_size=config.vocab_size))
    for distance in (config.window_size - 1, config.window_size, config.window_size + 1, 2 * config.window_size, 4 * config.window_size):
        length = max(config.train_sequence_length, distance + 128)
        if length <= max(config.lengths):
            evaluate("boundary", MemoryTaskSpec(length, 16, 1, 1, 0, distance, config.vocab_size))

    load_accuracies = [cell["accuracy"] for cell in cells["load"]]
    by_length: dict[int, list[float]] = {}
    for cell in cells["load"]:
        by_length.setdefault(cell["spec"]["sequence_length"], []).append(cell["accuracy"])
    memory_curve = [
        {"sequence_length": length, "accuracy": sum(values) / len(values)}
        for length, values in sorted(by_length.items())
    ]
    if len(memory_curve) == 1:
        memory_auc = memory_curve[0]["accuracy"]
    else:
        x = [math.log2(point["sequence_length"]) for point in memory_curve]
        area = sum(
            (x[index] - x[index - 1]) * (memory_curve[index]["accuracy"] + memory_curve[index - 1]["accuracy"]) / 2
            for index in range(1, len(x))
        )
        memory_auc = area / (x[-1] - x[0])
    update_accuracies = [cell["accuracy"] for cell in cells["updates"]]
    all_accuracies = [cell["accuracy"] for group in cells.values() for cell in group]
    return {
        "schema_version": 1,
        "cells": cells,
        "memory_auc": memory_auc,
        "mean_load_accuracy": sum(load_accuracies) / len(load_accuracies),
        "memory_curve": memory_curve,
        "update_accuracy": sum(update_accuracies) / len(update_accuracies),
        "worst_slice_accuracy": min(all_accuracies),
    }


def run_memory_probe(config: MemoryProbeConfig, window_pattern: str, seed: int, device: torch.device, force_final_full: bool = True):
    model, training = train_probe(config, window_pattern, seed, device, force_final_full)
    evaluation = evaluate_probe(model, config, seed + 1, device)
    return {
        "schema_version": 1,
        "topology": {"window_pattern": window_pattern, "window_sizes": model.window_sizes, "force_final_full": force_final_full},
        "training": training,
        "evaluation": evaluation,
    }
