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


PROBE_KEY_COUNT = 64


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
    kda_backend: str = "fla_triton"


@dataclass(frozen=True)
class ProbeStageConfig:
    name: str
    answer_budget: int
    batch_size: int
    sequence_length: int
    loads: tuple[int, ...]
    num_queries: int = 4
    updates: tuple[int, ...] = (1,)
    distractor_ratios: tuple[int, ...] = (0,)


def _default_probe_stages() -> tuple[ProbeStageConfig, ...]:
    return (
        ProbeStageConfig("easy-32", 128_000, 64, 32, (4,)),
        ProbeStageConfig("easy-64", 128_000, 64, 64, (4,)),
        ProbeStageConfig("easy-128", 256_000, 64, 128, (4,)),
        ProbeStageConfig("mixed-1024", 32_768, 16, 1024, (4, 16, 32), 4, (1, 2, 4), (0, 1)),
    )


@dataclass(frozen=True)
class MemoryProbeConfig:
    enabled: bool = False
    protocol_version: str = "associative_recall_v2"
    vocab_size: int = 256
    depth: int = 4
    width: int = 256
    head_dim: int = 128
    window_size: int = 256
    learning_rate: float = 3e-4
    initialization_seed: int = 11
    training_seed: int = 11
    examples_per_cell: int = 512
    easy_control_examples: int = 512
    evaluation_batch_size: int = 16
    lengths: tuple[int, ...] = (256, 512, 1024, 2048)
    loads: tuple[int, ...] = (4, 16, 64)
    updates: tuple[int, ...] = (1, 2, 4, 8)
    distractor_ratios: tuple[int, ...] = (0, 1, 4)
    stages: tuple[ProbeStageConfig, ...] = field(default_factory=_default_probe_stages)
    calibration_seeds: tuple[int, ...] = (11, 23, 37)
    overfit_max_steps: int = 100
    full_easy_min_accuracy: float = 0.95
    full_memory_auc_min: float = 0.85
    full_update_accuracy_min: float = 0.60
    full_long_accuracy_min: float = 0.90
    swa_local_accuracy_min: float = 0.80
    discrimination_margin_min: float = 0.50
    max_probe_seconds: float = 90.0


@dataclass(frozen=True)
class GeneralEvaluationConfig:
    """Frozen, architecture-neutral evaluation on natural language and CORE."""
    enabled: bool = True
    protocol: str = "general_lm"
    context_lengths: tuple[int, ...] = (128, 256, 512, 1024)
    target_tokens: int = 64
    max_documents: int = 128
    source_documents: int = 1024
    core_enabled: bool = False
    core_manifest: str = ""
    core_manifest_sha256: str = ""
    core_max_per_task: int = 100
    ruler_enabled: bool = False
    ruler_manifest: str = ""
    ruler_manifest_sha256: str = ""


@dataclass(frozen=True)
class SystemsConfig:
    enabled: bool = False
    # "full_compile" preserves the inherited training behavior. "eager" is a
    # declared systems lane, not a fallback: Torch compilation is disabled for
    # both model and optimizer in every subprocess.
    execution_mode: str = "full_compile"
    compile_timeout_seconds: float = 120.0
    warm_timeout_seconds: float = 300.0
    warmup_steps: int = 3
    timed_steps: int = 10
    prefill_lengths: tuple[int, ...] = (1024,)
    decode_context_lengths: tuple[int, ...] = (1024,)


@dataclass(frozen=True)
class SpeedSupervisorConfig:
    """Protected contract for bounded KDA training-speed iteration."""
    enabled: bool = False
    ledger_path: str = "runs/kda-training-autoresearch.sqlite3"
    candidate_paths: tuple[str, ...] = ("nanochat/mixers/kda.py",)
    correctness_tests: tuple[str, ...] = (
        "tests/test_kda_layer.py",
        "tests/test_kda_operator.py",
        "tests/test_kda_cuda.py",
        "tests/test_kda_integration.py",
    )
    test_timeout_seconds: float = 300.0
    max_attempts: int = 24
    min_relative_throughput_improvement: float = 0.03
    max_baseline_drift_fraction: float = 0.03


@dataclass(frozen=True)
class DecisionConfig:
    bpb_floor: float = 0.002
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
    schema_version: int = 2
    run: RunConfig = field(default_factory=RunConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # Retained only to read historical artifacts; it is disabled in the current plan.
    memory_probe: MemoryProbeConfig = field(default_factory=MemoryProbeConfig)
    evaluation: GeneralEvaluationConfig = field(default_factory=GeneralEvaluationConfig)
    systems: SystemsConfig = field(default_factory=SystemsConfig)
    speed_supervisor: SpeedSupervisorConfig = field(default_factory=SpeedSupervisorConfig)
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
        if key in {"lengths", "loads", "updates", "distractor_ratios", "calibration_seeds", "context_lengths", "prefill_lengths", "decode_context_lengths", "baseline_seeds", "internal_promotion_seeds", "allowed_paths", "protected_paths", "candidate_paths", "correctness_tests"}:
            values[key] = tuple(value)
    if cls is MemoryProbeConfig and "stages" in values:
        stages = []
        for index, stage in enumerate(values["stages"]):
            if not isinstance(stage, dict):
                raise ConfigError(f"[[memory_probe.stages]] entry {index} must be a table")
            unknown_stage = set(stage) - set(ProbeStageConfig.__dataclass_fields__)
            if unknown_stage:
                raise ConfigError(f"Unknown memory probe stage keys: {sorted(unknown_stage)}")
            stage = dict(stage)
            for stage_key in ("loads", "updates", "distractor_ratios"):
                if stage_key in stage:
                    stage[stage_key] = tuple(stage[stage_key])
            stages.append(ProbeStageConfig(**stage))
        values["stages"] = tuple(stages)
    return cls(**values)


def load_config(path: str | Path) -> ResearchConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    schema_version = raw.get("schema_version", 1)
    if schema_version != 2:
        raise ConfigError(f"Unsupported schema_version={schema_version}; expected 2")
    known = {"schema_version", "run", "training", "memory_probe", "evaluation", "systems", "speed_supervisor", "decision", "protection"}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"Unknown top-level keys: {sorted(unknown)}")
    config = ResearchConfig(
        schema_version=schema_version,
        run=_section(RunConfig, raw, "run"),
        training=_section(TrainingConfig, raw, "training"),
        memory_probe=_section(MemoryProbeConfig, raw, "memory_probe"),
        evaluation=_section(GeneralEvaluationConfig, raw, "evaluation"),
        systems=_section(SystemsConfig, raw, "systems"),
        speed_supervisor=_section(SpeedSupervisorConfig, raw, "speed_supervisor"),
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
    if train.kda_backend not in {"reference", "fla_triton"}:
        raise ConfigError(f"Unsupported KDA backend: {train.kda_backend}")
    if not train.window_pattern or any(char not in "LSK" for char in train.window_pattern.upper()):
        raise ConfigError("window_pattern must contain only L, S, and K")
    if train.sliding_window <= 0 or train.sliding_window > train.sequence_length:
        raise ConfigError("sliding_window must be positive and no larger than sequence_length")
    evaluation = config.evaluation
    if evaluation.enabled:
        if evaluation.protocol != "general_lm":
            raise ConfigError("Unsupported general evaluation protocol")
        if not evaluation.context_lengths or min(evaluation.context_lengths) <= 0:
            raise ConfigError("evaluation context_lengths must be non-empty and positive")
        if tuple(sorted(set(evaluation.context_lengths))) != evaluation.context_lengths:
            raise ConfigError("evaluation context_lengths must be unique and ascending")
        if max(evaluation.context_lengths) > train.sequence_length:
            raise ConfigError("evaluation context_lengths must not exceed the trained sequence length")
        if evaluation.target_tokens <= 0 or evaluation.target_tokens >= min(evaluation.context_lengths):
            raise ConfigError("evaluation target_tokens must be positive and smaller than every context length")
        if evaluation.max_documents <= 0 or evaluation.source_documents < evaluation.max_documents:
            raise ConfigError("evaluation source_documents must be at least max_documents")
        if evaluation.core_max_per_task == 0 or evaluation.core_max_per_task < -1:
            raise ConfigError("evaluation core_max_per_task must be -1 or positive")
        if evaluation.core_enabled and (not evaluation.core_manifest or len(evaluation.core_manifest_sha256) != 64):
            raise ConfigError("enabled CORE evaluation requires a manifest path and SHA-256")
        if evaluation.ruler_enabled and train.sequence_length < 4096:
            raise ConfigError("the complete RULER task family requires a 4096-token training lane")
        if evaluation.ruler_enabled and (not evaluation.ruler_manifest or len(evaluation.ruler_manifest_sha256) != 64):
            raise ConfigError("enabled RULER evaluation requires a manifest path and SHA-256")
    systems = config.systems
    if systems.enabled:
        if systems.execution_mode not in {"full_compile", "eager"}:
            raise ConfigError("systems execution_mode must be full_compile or eager")
        if systems.compile_timeout_seconds <= 0 or systems.warm_timeout_seconds <= 0 or systems.warmup_steps < 0 or systems.timed_steps <= 0:
            raise ConfigError("systems timing budgets must be positive")
        if not systems.prefill_lengths or not systems.decode_context_lengths:
            raise ConfigError("systems prefill/decode lengths must be non-empty")
        if max((*systems.prefill_lengths, *systems.decode_context_lengths)) > train.sequence_length:
            raise ConfigError("systems lengths must not exceed the trained sequence length")
    speed = config.speed_supervisor
    if speed.enabled:
        if not systems.enabled or systems.execution_mode != "eager":
            raise ConfigError("training-speed supervisor requires an enabled eager systems lane")
        if "K" not in train.window_pattern.upper():
            raise ConfigError("training-speed supervisor requires a KDA training pattern")
        if not speed.ledger_path or not speed.candidate_paths or not speed.correctness_tests:
            raise ConfigError("speed supervisor requires ledger, candidate paths, and correctness tests")
        if speed.test_timeout_seconds <= 0 or speed.max_attempts <= 0:
            raise ConfigError("speed-supervisor budgets must be positive")
        if not 0 < speed.min_relative_throughput_improvement < 1 or not 0 < speed.max_baseline_drift_fraction < 1:
            raise ConfigError("speed-supervisor threshold fractions must be in (0, 1)")
    if probe.enabled:
        if probe.protocol_version != "associative_recall_v2":
            raise ConfigError(f"Unsupported memory probe protocol: {probe.protocol_version}")
        if probe.vocab_size < 192:
            raise ConfigError("memory probe vocab_size must be at least 192")
        if probe.width % probe.head_dim != 0:
            raise ConfigError("memory probe width must be divisible by head_dim")
        if min(probe.initialization_seed, probe.training_seed) < 0:
            raise ConfigError("memory probe initialization_seed and training_seed must be non-negative")
        if not all((probe.lengths, probe.loads, probe.updates, probe.distractor_ratios)):
            raise ConfigError("memory probe evaluation grids must be non-empty")
        if probe.window_size <= 0 or probe.window_size > max(probe.lengths):
            raise ConfigError("memory probe window_size must be positive and fit an evaluation length")
        if any(length < 32 for length in probe.lengths):
            raise ConfigError("memory probe lengths must be at least 32")
        if not probe.stages:
            raise ConfigError("memory probe requires at least one curriculum stage")
        stage_names = [stage.name for stage in probe.stages]
        if len(stage_names) != len(set(stage_names)):
            raise ConfigError("memory probe stage names must be unique")
        for stage in probe.stages:
            answers_per_step = stage.batch_size * stage.num_queries
            if min(stage.answer_budget, stage.batch_size, stage.sequence_length, stage.num_queries) <= 0:
                raise ConfigError(f"memory probe stage {stage.name} values must be positive")
            if stage.answer_budget % answers_per_step:
                raise ConfigError(f"memory probe stage {stage.name} answer_budget must divide by batch_size * num_queries")
            if not stage.loads or any(load < stage.num_queries or load > PROBE_KEY_COUNT for load in stage.loads):
                raise ConfigError(f"memory probe stage {stage.name} loads must fit query count and key vocabulary")
            if not stage.updates or min(stage.updates) < 1:
                raise ConfigError(f"memory probe stage {stage.name} updates must be positive")
            if not stage.distractor_ratios or min(stage.distractor_ratios) < 0:
                raise ConfigError(f"memory probe stage {stage.name} distractor ratios must be non-negative")
        thresholds = (
            probe.full_easy_min_accuracy, probe.full_memory_auc_min, probe.full_update_accuracy_min,
            probe.full_long_accuracy_min, probe.swa_local_accuracy_min, probe.discrimination_margin_min,
        )
        if any(value < 0 or value > 1 for value in thresholds):
            raise ConfigError("memory probe accuracy thresholds must be in [0, 1]")
        if len(probe.calibration_seeds) < 3:
            raise ConfigError("memory probe calibration requires at least three seeds")
