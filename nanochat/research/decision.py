from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Objective:
    name: str
    direction: str
    tolerance: float
    relative: bool = False

    def __post_init__(self):
        if self.direction not in {"min", "max"}:
            raise ValueError("direction must be 'min' or 'max'")
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative")


DEFAULT_OBJECTIVES = (
    Objective("val_bpb", "min", 0.002),
    Objective("memory_auc", "max", 0.01),
    Objective("update_accuracy", "max", 0.01),
    Objective("tokens_per_second", "max", 0.03, relative=True),
    Objective("peak_memory_mb", "min", 0.02, relative=True),
    Objective("state_bytes", "min", 0.02, relative=True),
)


def objectives_from_config(config) -> tuple[Objective, ...]:
    """Build the frozen decision contract from a ResearchConfig decision table."""
    return (
        Objective("val_bpb", "min", config.bpb_floor),
        Objective("memory_auc", "max", config.accuracy_floor),
        Objective("update_accuracy", "max", config.accuracy_floor),
        Objective("tokens_per_second", "max", config.throughput_floor_fraction, relative=True),
        Objective("peak_memory_mb", "min", config.memory_floor_fraction, relative=True),
        Objective("state_bytes", "min", config.memory_floor_fraction, relative=True),
    )


def aggregate_objectives(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = [item for item in summaries if item.get("status") == "complete" and item.get("objectives")]
    if len(complete) < 2:
        raise ValueError("aggregation requires at least two complete runs")
    names = tuple(complete[0]["objectives"])
    if any(set(item["objectives"]) != set(names) for item in complete):
        raise ValueError("all runs must contain the same objectives")
    metrics: dict[str, Any] = {}
    for name in names:
        values = [float(item["objectives"][name]) for item in complete]
        mean = statistics.fmean(values)
        stdev = statistics.stdev(values)
        half_width = 1.96 * stdev / math.sqrt(len(values))
        metrics[name] = {
            "values": values,
            "mean": mean,
            "stdev": stdev,
            "ci95": [mean - half_width, mean + half_width],
        }
    return {"schema_version": 1, "runs": len(complete), "metrics": metrics}


def calibrate_objectives(summaries: Sequence[Mapping[str, Any]], config) -> dict[str, Any]:
    """Raise declared floors to the observed 95% baseline mean uncertainty."""
    aggregate = aggregate_objectives(summaries)
    declared = {objective.name: objective for objective in objectives_from_config(config)}
    calibrated: dict[str, float] = {}
    for name, stats in aggregate["metrics"].items():
        if name not in declared:
            continue
        observed = (stats["ci95"][1] - stats["ci95"][0]) / 2
        if declared[name].relative:
            observed /= max(abs(stats["mean"]), 1e-12)
        calibrated[name] = max(declared[name].tolerance, observed)
    return {"schema_version": 1, "method": "max(declared_floor, baseline_95pct_mean_half_width)", "aggregate": aggregate, "tolerances": calibrated}


def _delta(candidate: float, reference: float, objective: Objective) -> float:
    raw = candidate - reference
    if objective.relative:
        raw /= max(abs(reference), 1e-12)
    return raw if objective.direction == "max" else -raw


def compare_metrics(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
    objectives: Sequence[Objective] = DEFAULT_OBJECTIVES,
) -> dict[str, list[str]]:
    missing = [objective.name for objective in objectives if objective.name not in candidate or objective.name not in reference]
    if missing:
        raise ValueError(f"missing objectives: {missing}")
    improved: list[str] = []
    regressed: list[str] = []
    equivalent: list[str] = []
    for objective in objectives:
        delta = _delta(float(candidate[objective.name]), float(reference[objective.name]), objective)
        if delta > objective.tolerance:
            improved.append(objective.name)
        elif delta < -objective.tolerance:
            regressed.append(objective.name)
        else:
            equivalent.append(objective.name)
    return {"improved": improved, "regressed": regressed, "equivalent": equivalent}


def classify_candidate(
    candidate: Mapping[str, Any],
    frontier: Sequence[Mapping[str, Any]],
    objectives: Sequence[Objective] = DEFAULT_OBJECTIVES,
) -> dict[str, Any]:
    if candidate.get("status") not in {None, "complete"}:
        return {"status": "invalid", "reason": candidate.get("failure_reason", "candidate run is incomplete")}
    metrics = candidate.get("objectives", candidate)
    for incumbent in frontier:
        incumbent_metrics = incumbent.get("objectives", incumbent)
        comparison = compare_metrics(incumbent_metrics, metrics, objectives)
        if comparison["improved"] and not comparison["regressed"]:
            return {
                "status": "dominated",
                "dominated_by": incumbent.get("run_id"),
                "comparison": comparison,
            }
    if not frontier:
        return {"status": "frontier", "reason": "first valid candidate"}
    comparisons = [
        compare_metrics(metrics, incumbent.get("objectives", incumbent), objectives)
        for incumbent in frontier
    ]
    if any(item["improved"] and not item["regressed"] for item in comparisons):
        return {"status": "frontier", "comparisons": comparisons}
    return {"status": "retest", "comparisons": comparisons}
