"""Frozen, architecture-neutral language-model evaluation helpers.

The evaluator deliberately scores ordinary held-out documents.  It uses the
same target suffix at every context length, so a context curve measures the
value of additional natural-language prefix rather than a different sample.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from nanochat.dataloader import _document_batches
from nanochat.loss_eval import evaluate_bpb
from nanochat.research.artifacts import sha256_file


def _token_lists(tokenizer, texts: Sequence[str]) -> list[list[int]]:
    bos = tokenizer.get_bos_token_id()
    encoded = tokenizer.encode(list(texts), prepend=bos, num_threads=1)
    return [list(tokens) for tokens in encoded]


def collect_suffix_examples(
    token_lists: Iterable[Sequence[int]], context_lengths: Sequence[int], target_tokens: int,
    max_documents: int,
) -> list[list[int]]:
    """Select deterministic document suffixes usable at every requested length."""
    longest = max(context_lengths)
    examples: list[list[int]] = []
    for tokens in token_lists:
        if len(tokens) < longest:
            continue
        examples.append(list(tokens[-longest:]))
        if len(examples) == max_documents:
            break
    return examples


def _bpb(total_nats: float, total_bytes: int) -> float:
    return total_nats / (math.log(2) * total_bytes) if total_bytes else float("inf")


@torch.no_grad()
def score_context_curve(
    model: Callable[[torch.Tensor], torch.Tensor], examples: Sequence[Sequence[int]],
    context_lengths: Sequence[int], target_tokens: int, token_bytes: torch.Tensor, device: torch.device,
) -> dict[str, Any]:
    """Score equal natural-text targets after increasingly long prefixes."""
    if not examples:
        raise ValueError("no held-out documents are long enough for the requested context lengths")
    result: dict[str, Any] = {"schema_version": 1, "target_tokens": target_tokens, "contexts": {}}
    for context_length in context_lengths:
        total_nats = 0.0
        total_bytes = 0
        for example in examples:
            tokens = torch.tensor(example[-context_length:], dtype=torch.long, device=device).unsqueeze(0)
            logits = model(tokens)
            start = context_length - target_tokens
            targets = tokens[:, start:]
            target_logits = logits[:, start - 1:-1, :]
            losses = F.cross_entropy(
                target_logits.reshape(-1, target_logits.size(-1)), targets.reshape(-1), reduction="none")
            byte_lengths = token_bytes[targets.reshape(-1)]
            valid = byte_lengths > 0
            total_nats += float(losses[valid].sum().item())
            total_bytes += int(byte_lengths[valid].sum().item())
        result["contexts"][str(context_length)] = {
            "bpb": _bpb(total_nats, total_bytes),
            "nats": total_nats,
            "bytes": total_bytes,
            "documents": len(examples),
        }
    longest = str(max(context_lengths))
    result["context_bpb"] = result["contexts"][longest]["bpb"]
    return result


def heldout_context_curve(model, tokenizer, config, device: torch.device) -> dict[str, Any]:
    """Evaluate deterministic suffixes from the held-out validation shard."""
    texts: list[str] = []
    batches = _document_batches("val", None, tokenizer_batch_size=128)
    # Gather more than requested because short documents are excluded.
    limit = max(config.max_documents * 8, config.max_documents)
    while len(texts) < limit:
        batch, _ = next(batches)
        texts.extend(batch)
    examples = collect_suffix_examples(
        _token_lists(tokenizer, texts), config.context_lengths, config.target_tokens, config.max_documents,
    )
    from nanochat.tokenizer import get_token_bytes
    token_bytes = get_token_bytes(device=device)
    return score_context_curve(model, examples, config.context_lengths, config.target_tokens, token_bytes, device)


def _normalise_answer(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _prepared_core_bundle() -> Path:
    from nanochat.common import get_base_dir
    bundle = Path(get_base_dir()) / "eval_bundle"
    required = (bundle / "core.yaml", bundle / "eval_data", bundle / "eval_meta_data.csv")
    if not all(path.exists() for path in required):
        raise FileNotFoundError("CORE bundle is not prepared locally; evaluation will not download data")
    return bundle


def _load_ruler_manifest(config) -> tuple[dict[str, Any], Path]:
    from nanochat.common import get_base_dir
    path = Path(get_base_dir()) / config.ruler_manifest
    if not path.is_file() or sha256_file(path) != config.ruler_manifest_sha256:
        raise FileNotFoundError("prepared RULER manifest is missing or its SHA-256 does not match the frozen config")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("tasks"), list) or not manifest["tasks"]:
        raise ValueError("prepared RULER manifest requires a non-empty tasks list")
    return manifest, path.parent


@torch.no_grad()
def evaluate_prepared_ruler(model, tokenizer, config) -> dict[str, Any]:
    """Run a hash-verified local RULER export without a runtime downloader.

    The manifest is deliberately narrow: each task names a JSONL file, its
    SHA-256, a max_new_tokens limit, and an official-answer-compatible scorer
    (``exact`` or ``contains``). Rows contain ``prompt``/``answers``; the
    common ``input``/``outputs`` aliases are accepted for prepared exports.
    """
    manifest, parent = _load_ruler_manifest(config)
    from nanochat.engine import Engine
    engine = Engine(model, tokenizer)
    max_sequence = model.config.sequence_len
    tasks: dict[str, Any] = {}
    for task in manifest["tasks"]:
        required = {"name", "path", "sha256", "max_new_tokens", "scorer"}
        if not isinstance(task, dict) or set(task) - required or required - set(task):
            raise ValueError("each RULER task must contain only name/path/sha256/max_new_tokens/scorer")
        if task["scorer"] not in {"exact", "contains"} or int(task["max_new_tokens"]) <= 0:
            raise ValueError("unsupported RULER scorer or generation limit")
        data_path = parent / task["path"]
        if not data_path.is_file() or sha256_file(data_path) != task["sha256"]:
            raise FileNotFoundError(f"RULER task is missing or changed: {task['name']}")
        correct = total = 0
        for line in data_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            prompt = row.get("prompt", row.get("input"))
            answers = row.get("answers", row.get("outputs"))
            if not isinstance(prompt, str) or not isinstance(answers, list) or not all(isinstance(a, str) for a in answers):
                raise ValueError(f"invalid RULER row in {task['name']}")
            tokens = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
            if len(tokens) + int(task["max_new_tokens"]) > max_sequence:
                raise ValueError(f"RULER prompt exceeds trained context: {task['name']}")
            generated, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=int(task["max_new_tokens"]), temperature=0)
            answer = _normalise_answer(tokenizer.decode(generated[0][len(tokens):]))
            expected = [_normalise_answer(item) for item in answers]
            if task["scorer"] == "exact":
                passed = answer in expected
            else:
                passed = any(item in answer for item in expected)
            correct += int(passed)
            total += 1
        if not total:
            raise ValueError(f"empty RULER task: {task['name']}")
        tasks[task["name"]] = {"accuracy": correct / total, "correct": correct, "examples": total,
                               "scorer": task["scorer"], "max_new_tokens": task["max_new_tokens"],
                               "sha256": task["sha256"]}
    return {"manifest_sha256": config.ruler_manifest_sha256, "tasks": tasks}


def run_general_evaluation(model, tokenizer, config, device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": config.protocol,
        "natural_context": heldout_context_curve(model, tokenizer, config, device),
    }
    if config.core_enabled:
        _prepared_core_bundle()
        # The inherited CORE code owns its frozen bundle and task formatting.
        from scripts.base_eval import evaluate_core
        result["core"] = evaluate_core(model, tokenizer, device, max_per_task=config.core_max_per_task)
    if config.ruler_enabled:
        result["ruler"] = evaluate_prepared_ruler(model, tokenizer, config)
    return result
