from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any

import torch


BOS = 0
PAIR = 1
QUERY = 2
KEY_START = 8
KEY_STOP = 72
VALUE_START = 72
VALUE_STOP = 136
NOISE_START = 136


@dataclass(frozen=True)
class MemoryTaskSpec:
    sequence_length: int
    num_pairs: int
    num_queries: int = 4
    updates_per_query_key: int = 1
    distractor_ratio: int = 0
    target_distance: int | None = None
    vocab_size: int = 256


@dataclass(frozen=True)
class MemoryExampleMetadata:
    seed: int
    sequence_length: int
    num_pairs: int
    num_queries: int
    updates_per_query_key: int
    distractor_ratio: int
    query_distances: tuple[int, ...]
    query_positions: tuple[int, ...]


def _validate_spec(spec: MemoryTaskSpec) -> None:
    if spec.vocab_size <= NOISE_START:
        raise ValueError(f"vocab_size must exceed {NOISE_START}")
    if spec.sequence_length < 16:
        raise ValueError("sequence_length must be at least 16")
    if not 1 <= spec.num_pairs <= KEY_STOP - KEY_START:
        raise ValueError("num_pairs exceeds the disjoint key vocabulary")
    if not 1 <= spec.num_queries <= spec.num_pairs:
        raise ValueError("num_queries must be between one and num_pairs")
    if spec.updates_per_query_key < 1:
        raise ValueError("updates_per_query_key must be positive")
    if spec.distractor_ratio < 0:
        raise ValueError("distractor_ratio must be non-negative")
    if spec.target_distance is not None and (spec.num_queries != 1 or spec.target_distance < 2):
        raise ValueError("target_distance requires one query and must be at least two")


def generate_memory_example(spec: MemoryTaskSpec, seed: int) -> tuple[torch.Tensor, torch.Tensor, MemoryExampleMetadata]:
    """Generate a masked next-token associative-recall example.

    Only answer tokens following ``QUERY key`` contribute to the loss. Repeated
    writes use last-write-wins semantics. The returned inputs and labels both
    have ``sequence_length`` elements; non-answer labels are -1.
    """
    _validate_spec(spec)
    rng = random.Random(seed)
    key_pool = list(range(KEY_START, KEY_STOP))
    value_pool = list(range(VALUE_START, VALUE_STOP))
    rng.shuffle(key_pool)
    rng.shuffle(value_pool)
    signal_keys = key_pool[:spec.num_pairs]
    query_keys = signal_keys[:spec.num_queries]

    extra_distractors = min(
        spec.num_pairs * spec.distractor_ratio,
        len(key_pool) - spec.num_pairs,
    )
    distractor_keys = key_pool[spec.num_pairs:spec.num_pairs + extra_distractors]
    writes: list[tuple[int, int]] = []
    latest: dict[int, int] = {}
    last_write_value_position: dict[int, int] = {}

    for index, key in enumerate(signal_keys):
        value = value_pool[index % len(value_pool)]
        writes.append((key, value))
        latest[key] = value
    for index, key in enumerate(distractor_keys):
        value = value_pool[(spec.num_pairs + index) % len(value_pool)]
        writes.append((key, value))
    rng.shuffle(writes)
    for update_index in range(1, spec.updates_per_query_key):
        for query_index, key in enumerate(query_keys):
            value = value_pool[(spec.num_pairs + extra_distractors + update_index * spec.num_queries + query_index) % len(value_pool)]
            writes.append((key, value))
            latest[key] = value
    if spec.target_distance is not None:
        query_key = query_keys[0]
        final_query_write = max(index for index, (key, _) in enumerate(writes) if key == query_key)
        writes.append(writes.pop(final_query_write))
    write_tokens: list[int] = []
    for key, value in writes:
        write_tokens.extend((PAIR, key, value))
        last_write_value_position[key] = len(write_tokens)  # position after BOS is added

    query_tokens: list[int] = []
    for key in query_keys:
        query_tokens.extend((QUERY, key, latest[key]))

    minimum_length = 1 + len(write_tokens) + len(query_tokens)
    if minimum_length > spec.sequence_length + 1:
        raise ValueError(
            f"task requires {minimum_length - 1} tokens but sequence_length={spec.sequence_length}"
        )

    filler_count = spec.sequence_length + 1 - minimum_length
    if spec.target_distance is not None:
        key = query_keys[0]
        natural_distance = len(query_tokens[:2]) + filler_count
        # Writes are kept before filler and the query is at the end. Move excess
        # filler to the prefix so the requested last-write/query distance is exact.
        between = spec.target_distance - 2
        if between < 0 or between > filler_count:
            raise ValueError(
                f"target_distance={spec.target_distance} cannot fit; available range is 2..{natural_distance}"
            )
        prefix_noise_count = filler_count - between
        between_noise_count = between
    else:
        prefix_noise_count = rng.randint(0, filler_count)
        between_noise_count = filler_count - prefix_noise_count

    noise = lambda count: [rng.randrange(NOISE_START, spec.vocab_size) for _ in range(count)]
    stream = [BOS] + noise(prefix_noise_count) + write_tokens + noise(between_noise_count) + query_tokens
    assert len(stream) == spec.sequence_length + 1

    inputs = torch.tensor(stream[:-1], dtype=torch.long)
    labels = torch.full((spec.sequence_length,), -1, dtype=torch.long)
    query_positions: list[int] = []
    query_distances: list[int] = []
    for position in range(spec.sequence_length - 1):
        if inputs[position].item() == QUERY:
            key_position = position + 1
            key = inputs[key_position].item()
            labels[key_position] = stream[key_position + 1]
            query_positions.append(key_position)
            absolute_write_position = 1 + prefix_noise_count + last_write_value_position[key] - 1
            query_distances.append(key_position - absolute_write_position)

    metadata = MemoryExampleMetadata(
        seed=seed,
        sequence_length=spec.sequence_length,
        num_pairs=spec.num_pairs,
        num_queries=spec.num_queries,
        updates_per_query_key=spec.updates_per_query_key,
        distractor_ratio=spec.distractor_ratio,
        query_distances=tuple(query_distances),
        query_positions=tuple(query_positions),
    )
    return inputs, labels, metadata


def generate_memory_batch(spec: MemoryTaskSpec, seeds: list[int], device: str | torch.device = "cpu"):
    examples = [generate_memory_example(spec, seed) for seed in seeds]
    inputs = torch.stack([example[0] for example in examples]).to(device)
    labels = torch.stack([example[1] for example in examples]).to(device)
    metadata = [asdict(example[2]) for example in examples]
    return inputs, labels, metadata


def score_answer_tokens(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"logits/labels shape mismatch: {logits.shape[:2]} vs {labels.shape}")
    mask = labels >= 0
    count = int(mask.sum().item())
    if count == 0:
        raise ValueError("no answer tokens in labels")
    selected_logits = logits[mask].float()
    selected_labels = labels[mask]
    nll = torch.nn.functional.cross_entropy(selected_logits, selected_labels, reduction="mean")
    correct = int((selected_logits.argmax(dim=-1) == selected_labels).sum().item())
    return {"answer_tokens": count, "correct": correct, "nll": float(nll.item()), "accuracy": correct / count}


def score_oracle(inputs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    """Solve generated examples by scanning writes with last-write-wins semantics."""
    if inputs.shape != labels.shape or inputs.ndim != 2:
        raise ValueError("oracle expects matching [batch, sequence] inputs and labels")
    correct = 0
    count = 0
    for row_inputs, row_labels in zip(inputs.cpu(), labels.cpu()):
        latest: dict[int, int] = {}
        for position in range(row_inputs.numel() - 2):
            if int(row_inputs[position]) == PAIR:
                latest[int(row_inputs[position + 1])] = int(row_inputs[position + 2])
        for position in (row_labels >= 0).nonzero(as_tuple=False).flatten().tolist():
            key = int(row_inputs[position])
            if key not in latest:
                raise ValueError(f"query key {key} has no preceding write")
            correct += int(latest[key] == int(row_labels[position]))
            count += 1
    if count == 0:
        raise ValueError("no answer tokens in labels")
    return {"answer_tokens": count, "correct": correct, "accuracy": correct / count}
