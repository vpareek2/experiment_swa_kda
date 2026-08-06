from __future__ import annotations

from dataclasses import asdict
import math
import random
import time
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import Block, GPTConfig, Linear, norm
from nanochat.research.artifacts import sha256_json
from nanochat.research.config import MemoryProbeConfig, ProbeStageConfig
from nanochat.research.memory import (
    MemoryTaskSpec,
    generate_memory_batch,
    score_answer_tokens,
    score_oracle,
)


PROBE_PROTOCOL_VERSION = "associative_recall_v2"
ProgressCallback = Callable[[dict[str, Any]], None]


def probe_protocol_payload(config: MemoryProbeConfig) -> dict[str, Any]:
    payload = asdict(config)
    # Promotion increases statistical power without changing the learned task.
    payload.pop("examples_per_cell", None)
    return payload


def probe_protocol_hash(config: MemoryProbeConfig) -> str:
    return sha256_json(probe_protocol_payload(config))


def _emit(callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    if callback is not None:
        callback({"protocol_version": PROBE_PROTOCOL_VERSION, **event})


def wilson_interval(correct: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError("Wilson interval requires 0 <= correct <= total and total > 0")
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


class CanonicalProbeModel(nn.Module):
    """Protected shell in which only the configured sequence mixer varies."""

    def __init__(
        self,
        config: MemoryProbeConfig,
        window_pattern: str,
        force_final_full: bool = True,
        kda_backend: str = "reference",
    ):
        super().__init__()
        if any(char not in "LSK" for char in window_pattern.upper()):
            raise ValueError("memory probe v2 mixer pattern must contain only L, S, and K")
        n_head = config.width // config.head_dim
        maximum_length = max((*config.lengths, *(stage.sequence_length for stage in config.stages)))
        model_config = GPTConfig(
            sequence_len=maximum_length,
            vocab_size=config.vocab_size,
            n_layer=config.depth,
            n_head=n_head,
            n_kv_head=n_head,
            n_embd=config.width,
            window_pattern=window_pattern,
            sliding_window=config.window_size,
            force_final_full=force_final_full,
            kda_backend=kda_backend,
        )
        self.config = config
        self.model_config = model_config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList([Block(model_config, index) for index in range(config.depth)])
        self.lm_head = Linear(config.width, config.vocab_size, bias=False)
        mixer_types = [
            window_pattern[index % len(window_pattern)].upper()
            for index in range(config.depth)
        ]
        if force_final_full:
            mixer_types[-1] = "L"
        self.window_sizes = [
            (-1, 0) if mixer_type == "L" else ((config.window_size, 0) if mixer_type == "S" else None)
            for mixer_type in mixer_types
        ]
        channel_range = torch.arange(0, config.head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (100000 ** (channel_range / config.head_dim))
        positions = torch.arange(maximum_length, dtype=torch.float32)
        frequencies = torch.outer(positions, inv_freq)
        self.register_buffer("cos", frequencies.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin", frequencies.sin()[None, :, None, :], persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        scale = math.sqrt(3) * self.config.width ** -0.5
        for block in self.blocks:
            if block.mixer_type == "K":
                mixer = block.attn
                for projection in (
                    mixer.q_proj, mixer.k_proj, mixer.v_proj,
                    mixer.f_a_proj, mixer.f_b_proj, mixer.b_proj, mixer.g_proj,
                ):
                    nn.init.uniform_(projection.weight, -scale, scale)
                for convolution in (mixer.q_conv1d, mixer.k_conv1d, mixer.v_conv1d):
                    nn.init.uniform_(convolution.weight, -scale, scale)
                nn.init.zeros_(mixer.A_log)
                dt = torch.exp(
                    torch.rand_like(mixer.dt_bias) * (math.log(0.1) - math.log(0.001))
                    + math.log(0.001)
                ).clamp(min=1e-4)
                mixer.dt_bias.data.copy_(dt + torch.log(-torch.expm1(-dt)))
                nn.init.ones_(mixer.o_norm.weight)
                nn.init.zeros_(mixer.o_proj.weight)
            else:
                nn.init.uniform_(block.attn.c_q.weight, -scale, scale)
                nn.init.uniform_(block.attn.c_k.weight, -scale, scale)
                nn.init.uniform_(block.attn.c_v.weight, -scale, scale)
                nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.c_fc.weight, -0.4 * scale, 0.4 * scale)
            nn.init.zeros_(block.mlp.c_proj.weight)
            if block.mixer_type != "K" and block.attn.ve_gate is not None:
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


def _stage_spec(stage: ProbeStageConfig, rng: random.Random, vocab_size: int) -> MemoryTaskSpec:
    load = rng.choice(stage.loads)
    return MemoryTaskSpec(
        sequence_length=stage.sequence_length,
        num_pairs=load,
        num_queries=stage.num_queries,
        updates_per_query_key=rng.choice(stage.updates),
        distractor_ratio=rng.choice(stage.distractor_ratios),
        vocab_size=vocab_size,
    )


def _train_batch(model, optimizer, spec, seeds, device) -> tuple[float, int]:
    inputs, labels, _ = generate_memory_batch(spec, seeds, device=device)
    logits = model(inputs)
    mask = labels >= 0
    answer_count = int(mask.sum().item())
    loss = F.cross_entropy(logits[mask], labels[mask])
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite memory-probe loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.item()), answer_count


def train_probe(
    config: MemoryProbeConfig,
    window_pattern: str,
    seed: int,
    device: torch.device,
    force_final_full: bool = True,
    progress: ProgressCallback | None = None,
    kda_backend: str = "reference",
) -> tuple[CanonicalProbeModel, dict[str, Any]]:
    # Initialization is fixed so this diagnostic varies the mixer and task
    # stream, not an unrelated optimization lottery.
    torch.manual_seed(config.initialization_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(config.initialization_seed)
    model = CanonicalProbeModel(config, window_pattern, force_final_full, kda_backend).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    start = time.perf_counter()
    total_answers = 0
    total_steps = 0
    stage_results: list[dict[str, Any]] = []
    model.train()
    for stage_index, stage in enumerate(config.stages):
        steps = stage.answer_budget // (stage.batch_size * stage.num_queries)
        rng = random.Random(seed * 1_000_003 + stage_index * 10_007)
        initial_loss = None
        final_loss = None
        for stage_step in range(steps):
            spec = _stage_spec(stage, rng, config.vocab_size)
            seed_base = seed * 1_000_000_000 + stage_index * 10_000_000 + stage_step * stage.batch_size
            seeds = [seed_base + index for index in range(stage.batch_size)]
            final_loss, answer_count = _train_batch(model, optimizer, spec, seeds, device)
            initial_loss = final_loss if initial_loss is None else initial_loss
            if answer_count != stage.batch_size * stage.num_queries:
                raise RuntimeError(f"stage {stage.name} produced {answer_count} answers; expected {stage.batch_size * stage.num_queries}")
            total_answers += answer_count
            total_steps += 1
            report_every = max(steps // 4, 1)
            if stage_step == 0 or (stage_step + 1) % report_every == 0 or stage_step + 1 == steps:
                _emit(progress, {
                    "event": "probe_train",
                    "stage": stage.name,
                    "stage_index": stage_index,
                    "stage_step": stage_step + 1,
                    "stage_steps": steps,
                    "total_steps": total_steps,
                    "supervised_answers": total_answers,
                    "loss": final_loss,
                    "elapsed_seconds": time.perf_counter() - start,
                })
        stage_results.append({
            "name": stage.name,
            "steps": steps,
            "supervised_answers": stage.answer_budget,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
        })
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start
    return model, {
        "training_steps": total_steps,
        "supervised_answers": total_answers,
        "stage_results": stage_results,
        "training_seconds": training_seconds,
        "answers_per_second": total_answers / training_seconds,
        "num_params": sum(parameter.numel() for parameter in model.parameters()),
    }


@torch.inference_mode()
def _evaluate_cell(model, spec, examples, seed, batch_size, device) -> dict[str, Any]:
    total_answers = 0
    total_correct = 0
    weighted_nll = 0.0
    all_distances: list[int] = []
    model.eval()
    for offset in range(0, examples, batch_size):
        count = min(batch_size, examples - offset)
        seeds = [seed + offset + index for index in range(count)]
        inputs, labels, metadata = generate_memory_batch(spec, seeds, device=device)
        scores = score_answer_tokens(model(inputs), labels)
        total_answers += scores["answer_tokens"]
        total_correct += scores["correct"]
        weighted_nll += scores["nll"] * scores["answer_tokens"]
        for item in metadata:
            all_distances.extend(item["query_distances"])
    return {
        "spec": asdict(spec),
        "examples": examples,
        "answer_tokens": total_answers,
        "correct": total_correct,
        "nll": weighted_nll / total_answers,
        "accuracy": total_correct / total_answers,
        "accuracy_ci95": wilson_interval(total_correct, total_answers),
        "mean_query_distance": sum(all_distances) / len(all_distances),
        "chance_accuracy": 1.0 / 64.0,
    }


def _easy_control(model, config, seed, device) -> dict[str, Any]:
    started = time.perf_counter()
    cells = []
    for ordinal, length in enumerate((32, 64, 128, 256)):
        cells.append(_evaluate_cell(
            model,
            MemoryTaskSpec(length, 4, 4, vocab_size=config.vocab_size),
            config.easy_control_examples,
            seed + ordinal * config.easy_control_examples,
            config.evaluation_batch_size,
            device,
        ))
    correct = sum(cell["correct"] for cell in cells)
    answers = sum(cell["answer_tokens"] for cell in cells)
    return {
        "cells": cells,
        "accuracy": correct / answers,
        "accuracy_ci95": wilson_interval(correct, answers),
        "answer_tokens": answers,
        "evaluation_seconds": time.perf_counter() - started,
    }


def evaluate_probe(model, config: MemoryProbeConfig, seed: int, device: torch.device, progress: ProgressCallback | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    cells: dict[str, list[dict[str, Any]]] = {"load": [], "updates": [], "interference": [], "boundary": []}
    cell_seed = seed * 100_000_000
    ordinal = 0

    def evaluate(group: str, spec: MemoryTaskSpec):
        nonlocal ordinal
        result = _evaluate_cell(model, spec, config.examples_per_cell, cell_seed + ordinal * config.examples_per_cell,
                                config.evaluation_batch_size, device)
        cells[group].append(result)
        ordinal += 1
        _emit(progress, {"event": "probe_eval", "group": group, "cell": ordinal, "accuracy": result["accuracy"],
                         "elapsed_seconds": time.perf_counter() - start})

    for length in config.lengths:
        for load in config.loads:
            if 1 + 3 * load + 3 * min(4, load) <= length + 1:
                evaluate("load", MemoryTaskSpec(length, load, min(4, load), vocab_size=config.vocab_size))
    for updates in config.updates:
        evaluate("updates", MemoryTaskSpec(512, 16, 1, updates, vocab_size=config.vocab_size))
    for ratio in config.distractor_ratios:
        evaluate("interference", MemoryTaskSpec(1024, 16, 4, 1, ratio, vocab_size=config.vocab_size))
    maximum_training_length = max(stage.sequence_length for stage in config.stages)
    for distance in (config.window_size - 1, config.window_size, config.window_size + 1,
                     2 * config.window_size, 4 * config.window_size):
        length = max(maximum_training_length, distance + 128)
        if length <= max(config.lengths):
            evaluate("boundary", MemoryTaskSpec(length, 16, 1, 1, 0, distance, config.vocab_size))

    by_length: dict[int, list[float]] = {}
    for cell in cells["load"]:
        by_length.setdefault(cell["spec"]["sequence_length"], []).append(cell["accuracy"])
    memory_curve = [{"sequence_length": length, "accuracy": sum(values) / len(values)}
                    for length, values in sorted(by_length.items())]
    if len(memory_curve) == 1:
        memory_auc = memory_curve[0]["accuracy"]
    else:
        x = [math.log2(point["sequence_length"]) for point in memory_curve]
        area = sum((x[index] - x[index - 1]) *
                   (memory_curve[index]["accuracy"] + memory_curve[index - 1]["accuracy"]) / 2
                   for index in range(1, len(x)))
        memory_auc = area / (x[-1] - x[0])
    load_accuracies = [cell["accuracy"] for cell in cells["load"]]
    update_accuracies = [cell["accuracy"] for cell in cells["updates"]]
    all_accuracies = [cell["accuracy"] for group in cells.values() for cell in group]
    return {
        "schema_version": 2,
        "cells": cells,
        "memory_auc": memory_auc,
        "mean_load_accuracy": sum(load_accuracies) / len(load_accuracies),
        "memory_curve": memory_curve,
        "update_accuracy": sum(update_accuracies) / len(update_accuracies),
        "worst_slice_accuracy": min(all_accuracies),
        "evaluation_seconds": time.perf_counter() - start,
    }


def _probe_result(config, model, training, window_pattern, seed, device, force_final_full, progress):
    easy_control = _easy_control(model, config, seed * 100_000_000 + 70_000_000, device)
    evaluation = evaluate_probe(model, config, seed + 1, device, progress)
    total_seconds = training["training_seconds"] + easy_control["evaluation_seconds"] + evaluation["evaluation_seconds"]
    return {
        "schema_version": 2,
        "status": "complete",
        "protocol_version": config.protocol_version,
        "protocol_hash": probe_protocol_hash(config),
        "evaluation_seed": seed,
        "training_seed": config.training_seed,
        "topology": {"window_pattern": window_pattern, "window_sizes": model.window_sizes,
                     "force_final_full": force_final_full,
                     "kda_backend": model.model_config.kda_backend},
        "training": training,
        "easy_control": easy_control,
        "evaluation": evaluation,
        "total_seconds": total_seconds,
    }


def run_memory_probe(
    config,
    window_pattern,
    seed,
    device,
    force_final_full=True,
    progress=None,
    kda_backend="reference",
):
    model, training = train_probe(
        config, window_pattern, config.training_seed, device, force_final_full, progress, kda_backend,
    )
    return _probe_result(
        config, model, training, window_pattern, seed, device, force_final_full, progress,
    )


def run_overfit_control(config: MemoryProbeConfig, seed: int, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(config.initialization_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(config.initialization_seed)
    model = CanonicalProbeModel(config, "L", True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    spec = MemoryTaskSpec(32, 4, 4, vocab_size=config.vocab_size)
    inputs, labels, _ = generate_memory_batch(
        spec, [config.training_seed * 1000 + index for index in range(16)], device,
    )
    mask = labels >= 0
    achieved_step = None
    scores = None
    for step in range(config.overfit_max_steps + 1):
        logits = model(inputs)
        scores = score_answer_tokens(logits, labels)
        if scores["accuracy"] >= 0.99:
            achieved_step = step
            break
        if step == config.overfit_max_steps:
            break
        loss = F.cross_entropy(logits[mask], labels[mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {"accuracy": scores["accuracy"], "achieved_step": achieved_step,
            "max_steps": config.overfit_max_steps, "passed": achieved_step is not None}


def _boundary_accuracy(result: dict[str, Any], distance: int) -> float:
    for cell in result["evaluation"]["cells"]["boundary"]:
        if cell["spec"]["target_distance"] == distance:
            return cell["accuracy"]
    raise ValueError(f"calibration result has no boundary distance {distance}")


def calibration_checks(
    config: MemoryProbeConfig,
    seeds: list[int],
    oracle: dict[str, Any],
    overfit: dict[str, Any],
    full_results: list[dict[str, Any]],
    swa_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not (len(seeds) == len(full_results) == len(swa_results)):
        raise ValueError("calibration seeds and topology results must have equal lengths")
    checks: list[dict[str, Any]] = []

    def check(name: str, observed: float, threshold: float, comparator: str = ">="):
        passed = observed >= threshold if comparator == ">=" else observed <= threshold
        checks.append({"name": name, "observed": observed, "threshold": threshold,
                       "comparator": comparator, "passed": passed})

    check("oracle_accuracy", oracle["accuracy"], 1.0)
    check("overfit_accuracy", overfit["accuracy"], 0.99)
    for seed, full, swa in zip(seeds, full_results, swa_results):
        prefix = f"seed_{seed}"
        check(f"{prefix}_full_easy", full["easy_control"]["accuracy"], config.full_easy_min_accuracy)
        check(f"{prefix}_full_auc", full["evaluation"]["memory_auc"], config.full_memory_auc_min)
        check(f"{prefix}_full_updates", full["evaluation"]["update_accuracy"], config.full_update_accuracy_min)
        full_long = _boundary_accuracy(full, 4 * config.window_size)
        swa_local = min(_boundary_accuracy(swa, config.window_size - 1),
                        _boundary_accuracy(swa, config.window_size))
        swa_long = _boundary_accuracy(swa, 4 * config.window_size)
        check(f"{prefix}_full_long", full_long, config.full_long_accuracy_min)
        check(f"{prefix}_swa_local", swa_local, config.swa_local_accuracy_min)
        check(f"{prefix}_discrimination", full_long - swa_long, config.discrimination_margin_min)
        check(f"{prefix}_full_latency", full["total_seconds"], config.max_probe_seconds, "<=")
        check(f"{prefix}_swa_latency", swa["total_seconds"], config.max_probe_seconds, "<=")
    return checks


def run_probe_calibration(config: MemoryProbeConfig, seeds: list[int], device: torch.device,
                          progress: ProgressCallback | None = None) -> dict[str, Any]:
    if len(seeds) < 3:
        raise ValueError("probe calibration requires at least three seeds")
    oracle_spec = MemoryTaskSpec(1024, 16, 4, 4, 1, vocab_size=config.vocab_size)
    oracle_inputs, oracle_labels, _ = generate_memory_batch(oracle_spec, list(range(16)))
    oracle = score_oracle(oracle_inputs, oracle_labels)
    overfit = run_overfit_control(config, seeds[0], device)
    _emit(progress, {"event": "calibration", "topology": "full", "seed": config.training_seed})
    full_model, full_training = train_probe(config, "L", config.training_seed, device, True, progress)
    full_results = [
        _probe_result(config, full_model, full_training, "L", seed, device, True, progress)
        for seed in seeds
    ]
    _emit(progress, {"event": "calibration", "topology": "swa", "seed": config.training_seed})
    swa_model, swa_training = train_probe(config, "S", config.training_seed, device, False, progress)
    swa_results = [
        _probe_result(config, swa_model, swa_training, "S", seed, device, False, progress)
        for seed in seeds
    ]

    checks = calibration_checks(config, seeds, oracle, overfit, full_results, swa_results)
    return {
        "schema_version": 2,
        "protocol_version": config.protocol_version,
        "protocol_hash": probe_protocol_hash(config),
        "status": "valid" if all(item["passed"] for item in checks) else "invalid",
        "seeds": seeds,
        "oracle": oracle,
        "overfit": overfit,
        "checks": checks,
        "full_results": full_results,
        "swa_results": swa_results,
    }
