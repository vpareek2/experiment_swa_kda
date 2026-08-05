from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RunConfig:
    name: str = "discovery"
    suite: str = "discovery"
    artifact_root: str = "runs"
    seed: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    seconds: float = 300.0
    tokens: int = 0
    depth: int = 6
    head_dim: int = 128
    sequence_length: int = 1024
    device_batch_size: int = 8
    total_batch_size: int = 32768
    window_pattern: str = "L"
    sliding_window: int = 256
    force_final_full: bool = True
    eval_tokens: int = 1 << 20
    precision: str = "bfloat16"


@dataclass(frozen=True)
class MemoryProbeConfig:
    enabled: bool = True
    training_tokens: int = 1 << 21
    vocab_size: int = 256
    depth: int = 4
    width: int = 256
    head_dim: int = 128
    batch_size: int = 16
    train_sequence_length: int = 1024
    window_size: int = 256
    learning_rate: float = 3e-4
    examples_per_cell: int = 512
    lengths: tuple[int, ...] = (256, 512, 1024, 2048)
    loads: tuple[int, ...] = (4, 16, 64)
    updates: tuple[int, ...] = (1, 2, 4, 8)
    distractor_ratios: tuple[int, ...] = (0, 1, 4)


@dataclass(frozen=True)
class DecisionConfig:
    bpb_floor: float = 0.002
    accuracy_floor: float = 0.01
    throughput_floor_fraction: float = 0.03
    memory_floor_fraction: float = 0.02
    baseline_seeds: tuple[int, ...] = (11, 23, 37, 53, 71)
    internal_promotion_seeds: tuple[int, ...] = (101, 211, 307)
    confirmation_seed_count: int = 5


@dataclass(frozen=True)
class ProtectionConfig:
    allowed_paths: tuple[str, ...] = (
        "nanochat/gpt.py",
        "nanochat/mixers/",
        "configs/candidates/",
    )
    protected_paths: tuple[str, ...] = (
        "nanochat/research/",
        "configs/research/",
        "scripts/base_eval.py",
        "nanochat/core_eval.py",
        "nanochat/loss_eval.py",
        "nanochat/dataloader.py",
        "nanochat/dataset.py",
        "nanochat/tokenizer.py",
    )


@dataclass(frozen=True)
class ResearchConfig:
    schema_version: int = 1
    run: RunConfig = field(default_factory=RunConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    memory_probe: MemoryProbeConfig = field(default_factory=MemoryProbeConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    protection: ProtectionConfig = field(default_factory=ProtectionConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(cls, raw: dict[str, Any], name: str):
    values = raw.get(name, {})
    if not isinstance(values, dict):
        raise ConfigError(f"[{name}] must be a TOML table")
    known = set(cls.__dataclass_fields__)
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"Unknown [{name}] keys: {sorted(unknown)}")
    for key, value in list(values.items()):
        if key in {"lengths", "loads", "updates", "distractor_ratios", "baseline_seeds", "internal_promotion_seeds", "allowed_paths", "protected_paths"}:
            values[key] = tuple(value)
    return cls(**values)


def load_config(path: str | Path) -> ResearchConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    schema_version = raw.get("schema_version", 1)
    if schema_version != 1:
        raise ConfigError(f"Unsupported schema_version={schema_version}; expected 1")
    known = {"schema_version", "run", "training", "memory_probe", "decision", "protection"}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"Unknown top-level keys: {sorted(unknown)}")
    config = ResearchConfig(
        schema_version=schema_version,
        run=_section(RunConfig, raw, "run"),
        training=_section(TrainingConfig, raw, "training"),
        memory_probe=_section(MemoryProbeConfig, raw, "memory_probe"),
        decision=_section(DecisionConfig, raw, "decision"),
        protection=_section(ProtectionConfig, raw, "protection"),
    )
    validate_config(config)
    return config


def apply_candidate(path: str | Path, protocol: ResearchConfig) -> ResearchConfig:
    """Apply architecture-only candidate overrides to a frozen protocol."""
    candidate_path = Path(path)
    with candidate_path.open("rb") as handle:
        raw = tomllib.load(handle)
    if set(raw) != {"candidate"} or not isinstance(raw["candidate"], dict):
        raise ConfigError("candidate TOML must contain only a [candidate] table")
    values = raw["candidate"]
    allowed = {"name", "window_pattern", "sliding_window", "force_final_full"}
    unknown = set(values) - allowed
    if unknown:
        raise ConfigError(f"Unknown [candidate] keys: {sorted(unknown)}")
    if "name" not in values:
        raise ConfigError("[candidate].name is required")
    training_values = {key: value for key, value in values.items() if key != "name"}
    config = replace(
        protocol,
        run=replace(protocol.run, name=str(values["name"])),
        training=replace(protocol.training, **training_values),
    )
    validate_config(config)
    return config


def validate_config(config: ResearchConfig) -> None:
    train = config.training
    probe = config.memory_probe
    if config.run.suite not in {"discovery", "promotion", "confirmation"}:
        raise ConfigError(f"Unknown suite: {config.run.suite}")
    if (train.seconds <= 0) == (train.tokens <= 0):
        raise ConfigError("set exactly one positive training budget: seconds or tokens")
    if train.depth <= 0 or train.sequence_length <= 1:
        raise ConfigError("training depth and sequence length must be positive")
    if train.tokens > 0 and train.tokens % train.total_batch_size != 0:
        raise ConfigError("fixed training tokens must be divisible by total_batch_size")
    if train.total_batch_size % (train.device_batch_size * train.sequence_length) != 0:
        raise ConfigError("total_batch_size must divide evenly into device_batch_size * sequence_length")
    if train.precision not in {"bfloat16", "float32", "float16"}:
        raise ConfigError(f"Unsupported precision: {train.precision}")
    if not train.window_pattern or any(char not in "LS" for char in train.window_pattern.upper()):
        raise ConfigError("window_pattern must contain only L and S")
    if train.sliding_window <= 0 or train.sliding_window > train.sequence_length:
        raise ConfigError("sliding_window must be positive and no larger than sequence_length")
    if probe.enabled:
        if probe.vocab_size < 192:
            raise ConfigError("memory probe vocab_size must be at least 192")
        if probe.width % probe.head_dim != 0:
            raise ConfigError("memory probe width must be divisible by head_dim")
        if probe.training_tokens < probe.batch_size * probe.train_sequence_length:
            raise ConfigError("memory probe training_tokens must cover at least one batch")
        if probe.window_size <= 0 or probe.window_size > max(probe.lengths):
            raise ConfigError("memory probe window_size must be positive and fit an evaluation length")
        if any(length < 32 for length in probe.lengths):
            raise ConfigError("memory probe lengths must be at least 32")
