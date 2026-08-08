"""Independent frozen schema for the KDA CUDA-ownership campaign.

This schema deliberately does not extend :mod:`nanochat.research.config`.  The
completed speed campaign hashes the old ``ResearchConfig.to_dict()`` value, so
adding even a default field there would change its historical protocol hash.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


class CudaCampaignConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CampaignConfig:
    name: str = "kda-cuda-ownership"
    artifact_root: str = "runs/cuda-ownership-supervisor"
    ledger_path: str = "runs/kda-cuda-ownership.sqlite3"
    max_attempts: int = 24
    seed: int = 42
    candidate_paths: tuple[str, ...] = ("nanochat/mixers/cuda_kda/",)
    foundation_ref: str = "kda-cuda-ownership-launch-foundation"
    controller_ref: str = "kda-cuda-ownership-controller"
    cumulative_performance_anchor_ref: str = "0b4b24773c2696c23338d7600101d7072b592aa9"
    max_patch_bytes: int = 2_000_000


@dataclass(frozen=True)
class OwnershipConfig:
    metadata_callable: str = "kda_backend_provenance"
    required_components: tuple[str, ...] = (
        "chunk_forward", "chunk_backward", "recurrent_decode", "causal_convolution_forward", "causal_convolution_backward",
    )
    component_weights: tuple[float, ...] = (0.25, 0.35, 0.20, 0.10, 0.10)
    atomic_units: tuple[tuple[str, ...], ...] = (("chunk_forward", "chunk_backward"), ("recurrent_decode",), ("causal_convolution_forward", "causal_convolution_backward"))
    majority_core_components: tuple[str, ...] = ("chunk_forward", "chunk_backward", "recurrent_decode")
    project_source_roots: tuple[str, ...] = ("nanochat/mixers/cuda_kda/",)
    source_extensions: tuple[str, ...] = (".py", ".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp", ".ptx")
    native_cuda_extensions: tuple[str, ...] = (".cu", ".cuh")
    minimum_project_owned_fraction: float = 0.500001
    forbid_runtime_modules: tuple[str, ...] = ("fla",)
    forbid_runtime_fla_for_optimization: bool = True
    allow_fla_offline_oracle: bool = True


@dataclass(frozen=True)
class CorrectnessConfig:
    tests: tuple[str, ...] = (
        "tests/test_kda_operator.py", "tests/test_kda_layer.py",
        "tests/test_kda_cuda.py", "tests/test_kda_integration.py",
    )
    pytest_timeout_seconds: float = 360.0
    runtime_audit_timeout_seconds: float = 180.0
    compute_sanitizer_timeout_seconds: float = 300.0
    require_compute_sanitizer: bool = True
    sanitizer_tools: tuple[str, ...] = ("memcheck", "racecheck", "synccheck", "initcheck")
    boundary_lengths: tuple[int, ...] = (1, 2, 3, 4, 63, 64, 65, 127, 128, 129, 257)
    production_head_dim: int = 128
    forward_atol: float = 0.005
    forward_rtol: float = 0.005
    gradient_atol: float = 0.02
    gradient_rtol: float = 0.02


@dataclass(frozen=True)
class BootstrapConfig:
    enabled: bool = True
    maximum_sequence_length: int = 256
    timed_iterations: int = 10
    minimum_owned_fraction: float = 0.10
    performance_advisory: bool = True
    allow_fla_for_unclaimed_components: bool = True
    forbid_selective_ptx: bool = True


@dataclass(frozen=True)
class MigrationConfig:
    enabled: bool = True
    maximum_sequence_length: int = 1024
    timed_iterations: int = 25
    minimum_ownership_increase: float = 1e-9
    performance_advisory: bool = True
    allow_fla_for_unclaimed_components: bool = True
    require_strict_owner_superset: bool = True


@dataclass(frozen=True)
class OptimizationConfig:
    enabled: bool = True
    require_runtime_fla_free: bool = True
    require_all_components_project: bool = True
    require_full_training: bool = True


@dataclass(frozen=True)
class ReportingConfig:
    canonical_operation: str = "chunk_forward"
    canonical_length: int = 256
    retain_raw_attempts: bool = True
    require_python_anchor: bool = True
    require_fla_anchor: bool = True
    historical_context_manifest: str = "configs/research/archive/kda_training_speed_context.json"
    historical_context_sha256: str = "79d0384d516407ac4b6d36b20dbc317af7468921867560d0641b87510c6bc85f"


@dataclass(frozen=True)
class MeasurementConfig:
    sequence_length: int = 4096
    device_batch_size: int = 2
    total_batch_size: int = 32768
    depth: int = 6
    head_dim: int = 128
    warmup_steps: int = 2
    timed_steps: int = 5
    discovery_paired_blocks: int = 9
    promotion_paired_blocks: int = 15
    block_timeout_seconds: float = 180.0
    confidence_level: float = 0.95
    discovery_effect_fraction: float = 0.0075
    retention_margin_fraction: float = 0.0075
    promotion_cumulative_fraction: float = 0.03
    max_peak_memory_regression_fraction: float = 0.03
    max_kernel_latency_regression_fraction: float = 0.05
    max_baseline_drift_fraction: float = 0.01
    execution_mode: str = "eager"


@dataclass(frozen=True)
class KernelGateConfig:
    enabled: bool = True
    sequence_lengths: tuple[int, ...] = (1, 64, 65, 256, 1024, 4096)
    warmup_iterations: int = 10
    timed_iterations: int = 50
    timeout_seconds: float = 180.0
    profile_max_bytes: int = 262_144
    profile_rows: int = 40
    require_profile: bool = True
    selective_ptx_min_latency_improvement: float = 0.02


@dataclass(frozen=True)
class KdaCudaCampaignConfig:
    schema: str = "kda_cuda_ownership"
    schema_version: int = 2
    campaign: CampaignConfig = field(default_factory=CampaignConfig)
    ownership: OwnershipConfig = field(default_factory=OwnershipConfig)
    correctness: CorrectnessConfig = field(default_factory=CorrectnessConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    migration: MigrationConfig = field(default_factory=MigrationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    kernel_gates: KernelGateConfig = field(default_factory=KernelGateConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_TUPLE_FIELDS = {
    "candidate_paths", "required_components", "component_weights", "atomic_units", "majority_core_components", "project_source_roots",
    "source_extensions", "native_cuda_extensions", "forbid_runtime_modules", "tests", "sanitizer_tools", "boundary_lengths", "sequence_lengths",
}


def _section(cls, raw: dict[str, Any], name: str):
    values = raw.get(name, {})
    if not isinstance(values, dict):
        raise CudaCampaignConfigError(f"[{name}] must be a TOML table")
    unknown = set(values) - set(cls.__dataclass_fields__)
    if unknown:
        raise CudaCampaignConfigError(f"Unknown [{name}] keys: {sorted(unknown)}")
    values = dict(values)
    for key in set(values) & _TUPLE_FIELDS:
        values[key] = tuple(values[key])
    if cls is OwnershipConfig and "atomic_units" in values:
        values["atomic_units"] = tuple(tuple(unit) for unit in values["atomic_units"])
    return cls(**values)


def load_cuda_campaign_config(path: str | Path) -> KdaCudaCampaignConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    return cuda_campaign_config_from_dict(raw)


def cuda_campaign_config_from_dict(raw: dict[str, Any]) -> KdaCudaCampaignConfig:
    raw = {key: (dict(value) if isinstance(value, dict) else value) for key, value in raw.items()}
    known = {"schema", "schema_version", "campaign", "ownership", "correctness", "bootstrap", "migration", "optimization", "reporting", "measurement", "kernel_gates"}
    unknown = set(raw) - known
    if unknown:
        raise CudaCampaignConfigError(f"Unknown top-level keys: {sorted(unknown)}")
    if raw.get("schema") != "kda_cuda_ownership" or raw.get("schema_version") != 2:
        raise CudaCampaignConfigError("expected kda_cuda_ownership schema_version 2")
    config = KdaCudaCampaignConfig(
        schema=raw["schema"], schema_version=raw["schema_version"],
        campaign=_section(CampaignConfig, raw, "campaign"),
        ownership=_section(OwnershipConfig, raw, "ownership"),
        correctness=_section(CorrectnessConfig, raw, "correctness"),
        bootstrap=_section(BootstrapConfig, raw, "bootstrap"),
        migration=_section(MigrationConfig, raw, "migration"),
        optimization=_section(OptimizationConfig, raw, "optimization"),
        reporting=_section(ReportingConfig, raw, "reporting"),
        measurement=_section(MeasurementConfig, raw, "measurement"),
        kernel_gates=_section(KernelGateConfig, raw, "kernel_gates"),
    )
    validate_cuda_campaign_config(config)
    return config


def validate_cuda_campaign_config(config: KdaCudaCampaignConfig) -> None:
    campaign, ownership = config.campaign, config.ownership
    correctness, measurement, kernels = config.correctness, config.measurement, config.kernel_gates
    bootstrap, migration, optimization, reporting = config.bootstrap, config.migration, config.optimization, config.reporting
    if not campaign.ledger_path or not campaign.artifact_root or not campaign.candidate_paths or campaign.max_attempts <= 0 or campaign.max_patch_bytes <= 0:
        raise CudaCampaignConfigError("campaign paths and positive attempt budget are required")
    if not campaign.foundation_ref or not campaign.controller_ref or not campaign.cumulative_performance_anchor_ref:
        raise CudaCampaignConfigError("campaign foundation, controller, and cumulative performance refs are required")
    if not ownership.required_components or len(ownership.required_components) != len(ownership.component_weights):
        raise CudaCampaignConfigError("ownership components and weights must be non-empty and aligned")
    if len(set(ownership.required_components)) != len(ownership.required_components):
        raise CudaCampaignConfigError("ownership component names must be unique")
    flattened=[component for unit in ownership.atomic_units for component in unit]
    if sorted(flattened) != sorted(ownership.required_components) or len(flattened) != len(set(flattened)):
        raise CudaCampaignConfigError("atomic ownership units must partition every component exactly once")
    if tuple(ownership.atomic_units) != (("chunk_forward", "chunk_backward"), ("recurrent_decode",), ("causal_convolution_forward", "causal_convolution_backward")):
        raise CudaCampaignConfigError("frozen autograd ownership units do not match the protected ABI")
    if not ownership.majority_core_components or not set(ownership.majority_core_components) <= set(ownership.required_components):
        raise CudaCampaignConfigError("majority core components must be a non-empty ownership subset")
    if any(weight <= 0 for weight in ownership.component_weights) or abs(sum(ownership.component_weights) - 1.0) > 1e-9:
        raise CudaCampaignConfigError("ownership component weights must be positive and sum to one")
    if not 0.5 < ownership.minimum_project_owned_fraction <= 1.0:
        raise CudaCampaignConfigError("majority project ownership must be strictly greater than one half")
    if not ownership.forbid_runtime_fla_for_optimization or not ownership.allow_fla_offline_oracle:
        raise CudaCampaignConfigError("protected optimization FLA blocking and offline FLA anchoring are mandatory")
    if not ownership.project_source_roots or not ownership.source_extensions or not ownership.native_cuda_extensions:
        raise CudaCampaignConfigError("project source roots and CUDA/build extensions are required")
    if not set(ownership.native_cuda_extensions) <= set(ownership.source_extensions):
        raise CudaCampaignConfigError("native CUDA extensions must be allowed source extensions")
    if not correctness.tests or not correctness.boundary_lengths or min(correctness.boundary_lengths) < 1:
        raise CudaCampaignConfigError("strong correctness tests and positive boundary lengths are required")
    if correctness.require_compute_sanitizer and set(correctness.sanitizer_tools) != {"memcheck", "racecheck", "synccheck", "initcheck"}:
        raise CudaCampaignConfigError("strong CUDA safety requires memcheck, racecheck, synccheck, and initcheck")
    if min(correctness.pytest_timeout_seconds, correctness.runtime_audit_timeout_seconds,
           correctness.compute_sanitizer_timeout_seconds) <= 0:
        raise CudaCampaignConfigError("correctness timeouts must be positive")
    if min(correctness.forward_atol, correctness.forward_rtol, correctness.gradient_atol, correctness.gradient_rtol) < 0:
        raise CudaCampaignConfigError("correctness tolerances must be non-negative")
    if not all((bootstrap.enabled, migration.enabled, optimization.enabled)):
        raise CudaCampaignConfigError("bootstrap, migration, and optimization lanes must be enabled")
    if bootstrap.maximum_sequence_length <= 0 or bootstrap.maximum_sequence_length > migration.maximum_sequence_length:
        raise CudaCampaignConfigError("bootstrap maximum length must be positive and no larger than migration")
    if migration.maximum_sequence_length > measurement.sequence_length:
        raise CudaCampaignConfigError("migration maximum length must not exceed the optimization training length")
    if min(bootstrap.timed_iterations, migration.timed_iterations) <= 0:
        raise CudaCampaignConfigError("bootstrap and migration timed iterations must be positive")
    if not 0 < bootstrap.minimum_owned_fraction <= ownership.minimum_project_owned_fraction:
        raise CudaCampaignConfigError("bootstrap ownership floor must be positive and no greater than majority")
    if migration.minimum_ownership_increase <= 0:
        raise CudaCampaignConfigError("migration requires a positive ownership increase")
    if not bootstrap.performance_advisory or not migration.performance_advisory:
        raise CudaCampaignConfigError("bootstrap and migration performance must remain advisory")
    if not bootstrap.forbid_selective_ptx:
        raise CudaCampaignConfigError("bootstrap must forbid selective PTX")
    if not migration.require_strict_owner_superset:
        raise CudaCampaignConfigError("migration must require a strict owner superset")
    if not bootstrap.allow_fla_for_unclaimed_components or not migration.allow_fla_for_unclaimed_components:
        raise CudaCampaignConfigError("transitional lanes must allow FLA only for unclaimed components")
    if not (optimization.require_runtime_fla_free and optimization.require_all_components_project and optimization.require_full_training):
        raise CudaCampaignConfigError("optimization requires all components project-owned, zero runtime FLA, and full training")
    if reporting.canonical_operation not in {"chunk_forward", "recurrent_decode"} or reporting.canonical_length <= 0:
        raise CudaCampaignConfigError("reporting canonical operation/length is invalid")
    if not (reporting.retain_raw_attempts and reporting.require_python_anchor and reporting.require_fla_anchor):
        raise CudaCampaignConfigError("educational reporting must retain raw attempts and both anchors")
    if not reporting.historical_context_manifest or len(reporting.historical_context_sha256) != 64:
        raise CudaCampaignConfigError("reporting requires a hash-pinned historical context manifest")
    if measurement.execution_mode != "eager":
        raise CudaCampaignConfigError("CUDA migration comparisons require the frozen eager lane")
    if min(measurement.sequence_length, measurement.device_batch_size, measurement.total_batch_size,
           measurement.depth, measurement.head_dim, measurement.timed_steps) <= 0 or measurement.warmup_steps < 1:
        raise CudaCampaignConfigError("measurement shapes and step counts must be positive")
    if measurement.total_batch_size % (measurement.device_batch_size * measurement.sequence_length):
        raise CudaCampaignConfigError("total batch must divide evenly by device batch times sequence length")
    if measurement.discovery_paired_blocks < 7 or measurement.promotion_paired_blocks < measurement.discovery_paired_blocks:
        raise CudaCampaignConfigError("credible interleaving requires at least seven discovery blocks and no fewer promotion blocks")
    if measurement.confidence_level != 0.95:
        raise CudaCampaignConfigError("the implemented paired interval is frozen at 95% confidence")
    if not 0.005 <= measurement.discovery_effect_fraction <= 0.01:
        raise CudaCampaignConfigError("discovery effect must be in the requested 0.5%-1% range")
    if not 0.005 <= measurement.retention_margin_fraction <= 0.01:
        raise CudaCampaignConfigError("retention margin must be in the requested 0.5%-1% range")
    if measurement.promotion_cumulative_fraction < 0.03:
        raise CudaCampaignConfigError("promotion requires at least three percent cumulative improvement")
    if not all(0 < value < 1 for value in (measurement.max_peak_memory_regression_fraction, measurement.max_kernel_latency_regression_fraction, measurement.max_baseline_drift_fraction)):
        raise CudaCampaignConfigError("memory, kernel, and drift retention limits must be fractions in (0, 1)")
    if measurement.block_timeout_seconds <= 0:
        raise CudaCampaignConfigError("measurement timeout must be positive")
    if not kernels.enabled:
        raise CudaCampaignConfigError("kernel and bounded-profile evidence is mandatory")
    if kernels.enabled:
        if not kernels.sequence_lengths or min(kernels.sequence_lengths) < 1:
            raise CudaCampaignConfigError("kernel microbenchmark lengths must be positive")
        if not 0 < kernels.selective_ptx_min_latency_improvement < 1:
            raise CudaCampaignConfigError("selective PTX materiality must be a fraction in (0, 1)")
        if min(kernels.warmup_iterations, kernels.timed_iterations, kernels.timeout_seconds,
               kernels.profile_max_bytes, kernels.profile_rows) <= 0:
            raise CudaCampaignConfigError("kernel/profile budgets must be positive")
