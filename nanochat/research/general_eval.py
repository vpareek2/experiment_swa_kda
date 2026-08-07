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
    # Scan a frozen count of source documents because long documents are sparse.
    limit = config.source_documents
    while len(texts) < limit:
        batch, _ = next(batches)
        texts.extend(batch)
    examples = collect_suffix_examples(
        _token_lists(tokenizer, texts), config.context_lengths, config.target_tokens, config.max_documents,
    )
    from nanochat.tokenizer import get_token_bytes
    token_bytes = get_token_bytes(device=device)
    return score_context_curve(model, examples, config.context_lengths, config.target_tokens, token_bytes, device)


def _ruler_prediction(value: str) -> str:
    # Matches the upstream RULER postprocess_pred behavior before matching.
    import re
    return re.sub(r"[\x00-\x1f]", "\n", value).strip().lower()


def ruler_match_score(prediction: str, references: Sequence[str], scorer: str) -> float:
    """Return the upstream RULER per-example all/partial match contribution."""
    if not references:
        raise ValueError("RULER reference list must be non-empty")
    prediction = _ruler_prediction(prediction)
    references = [reference.lower() for reference in references]
    if scorer == "all":
        return sum(reference in prediction for reference in references) / len(references)
    if scorer == "partial":
        return float(any(reference in prediction for reference in references))
    raise ValueError("unsupported RULER scorer")


def prepared_core_bundle(config) -> Path:
    """Return a complete, hash-verified local CORE bundle without downloading."""
    from nanochat.common import get_base_dir
    base = Path(get_base_dir())
    manifest_path = base / config.core_manifest
    if not manifest_path.is_file() or sha256_file(manifest_path) != config.core_manifest_sha256:
        raise FileNotFoundError("CORE manifest is missing or its SHA-256 does not match the frozen config")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    bundle = base / "eval_bundle"
    if manifest.get("schema_version") != 1 or not isinstance(files, dict):
        raise ValueError("invalid CORE manifest")
    required = {"core.yaml", "eval_meta_data.csv"}
    if not required.issubset(files):
        raise ValueError("CORE manifest is missing required files")
    for relative, expected_hash in files.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError("invalid CORE manifest file entry")
        target = bundle / relative
        try:
            target.relative_to(bundle)
        except ValueError as error:
            raise ValueError("CORE manifest path escapes its bundle") from error
        if not target.is_file() or sha256_file(target) != expected_hash:
            raise FileNotFoundError(f"CORE bundle file is missing or changed: {relative}")
    return bundle


def prepared_ruler_manifest(config) -> tuple[dict[str, Any], Path]:
    """Return a complete hash-verified local RULER export without downloading."""
    from nanochat.common import get_base_dir
    path = Path(get_base_dir()) / config.ruler_manifest
    if not path.is_file() or sha256_file(path) != config.ruler_manifest_sha256:
        raise FileNotFoundError("prepared RULER manifest is missing or its SHA-256 does not match the frozen config")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("prepared RULER manifest requires a non-empty tasks list")
    parent = path.parent
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("path"), str) or not isinstance(task.get("sha256"), str):
            raise ValueError("invalid RULER manifest task entry")
        target = parent / task["path"]
        try:
            target.relative_to(parent)
        except ValueError as error:
            raise ValueError("RULER manifest path escapes its bundle") from error
        if not target.is_file() or sha256_file(target) != task["sha256"]:
            raise FileNotFoundError(f"RULER task is missing or changed: {task.get('name', '?')}")
    return manifest, parent


@torch.no_grad()
def evaluate_prepared_ruler(model, tokenizer, config) -> dict[str, Any]:
    """Run a hash-verified export from the official RULER generator.

    Rows use the upstream ``input``, ``answer_prefix``, and ``outputs`` fields.
    Matching exactly follows the upstream synthetic evaluator: all-reference
    containment for retrieval/tracking/aggregation and any-reference containment
    for QA. The manifest fixes generated files, prompt budgets, and scorers.
    """
    manifest, parent = prepared_ruler_manifest(config)
    from nanochat.engine import Engine
    engine = Engine(model, tokenizer)
    max_sequence = model.config.sequence_len
    tasks: dict[str, Any] = {}
    allowed = {"name", "path", "sha256", "max_new_tokens", "scorer", "context_tokens"}
    required = allowed
    for task in manifest["tasks"]:
        if not isinstance(task, dict) or set(task) != required:
            raise ValueError("each RULER task must contain the frozen task fields")
        if task["scorer"] not in {"all", "partial"} or int(task["max_new_tokens"]) <= 0:
            raise ValueError("unsupported RULER scorer or generation limit")
        data_path = parent / task["path"]
        if not data_path.is_file() or sha256_file(data_path) != task["sha256"]:
            raise FileNotFoundError(f"RULER task is missing or changed: {task['name']}")
        score = 0.0
        total = 0
        for line in data_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            prompt = row.get("input", row.get("prompt"))
            answer_prefix = row.get("answer_prefix", "")
            answers = row.get("outputs", row.get("answers"))
            if not isinstance(prompt, str) or not isinstance(answer_prefix, str):
                raise ValueError(f"invalid RULER prompt in {task['name']}")
            if not isinstance(answers, list) or not all(isinstance(answer, str) for answer in answers):
                raise ValueError(f"invalid RULER references in {task['name']}")
            tokens = tokenizer.encode(prompt + answer_prefix, prepend=tokenizer.get_bos_token_id())
            if len(tokens) + int(task["max_new_tokens"]) > max_sequence:
                raise ValueError(f"RULER prompt exceeds trained context: {task['name']}")
            generated, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=int(task["max_new_tokens"]), temperature=0)
            prediction = tokenizer.decode(generated[0][len(tokens):])
            score += ruler_match_score(prediction, answers, task["scorer"])
            total += 1
        if not total:
            raise ValueError(f"empty RULER task: {task['name']}")
        tasks[task["name"]] = {
            "accuracy": score / total,
            "score_sum": score,
            "examples": total,
            "scorer": task["scorer"],
            "max_new_tokens": task["max_new_tokens"],
            "context_tokens": task["context_tokens"],
            "sha256": task["sha256"],
        }
    return {"manifest_sha256": config.ruler_manifest_sha256, "tasks": tasks}


def run_general_evaluation(model, tokenizer, config, device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": config.protocol,
        "natural_context": heldout_context_curve(model, tokenizer, config, device),
    }
    if config.core_enabled:
        prepared_core_bundle(config)
        # The inherited CORE code owns its frozen bundle and task formatting.
        from scripts.base_eval import evaluate_core
        result["core"] = evaluate_core(model, tokenizer, device, max_per_task=config.core_max_per_task)
    if config.ruler_enabled:
        result["ruler"] = evaluate_prepared_ruler(model, tokenizer, config)
    return result
