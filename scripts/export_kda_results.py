#!/usr/bin/env python3
"""Export deterministic, release-facing summaries from the KDA attempt ledger."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "runs/kda-cuda-development/attempt-index.jsonl"
CSV_OUTPUT = ROOT / "results/attempts.csv"
SUMMARY_OUTPUT = ROOT / "results/campaign-summary.json"


def _get(record: dict[str, Any], *path: str) -> Any:
    value: Any = record
    for part in path:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _pct(value: Any) -> str:
    return "" if not isinstance(value, (int, float)) else f"{100 * value:.6f}"


def _scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def load_records() -> list[dict[str, Any]]:
    return [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]


FIELDS = [
    "loop_id",
    "status",
    "preserved",
    "candidate_commit",
    "parent_commit",
    "baseline_commit",
    "artifact",
    "correctness_recorded",
    "level1_decision",
    "level1_forward_backward_improvement_pct",
    "level2_baseline_median_tps",
    "level2_candidate_median_tps",
    "level2_improvement_pct",
    "trainer_evaluated",
    "quality_evaluated",
    "statistically_confirmed",
    "next",
]


def csv_text(records: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        level1_gain = _get(record, "level1", "t4096_forward_backward_improvement_fraction")
        if level1_gain is None:
            level1_gain = _get(record, "level1", "improvement_fraction")
        row = {
            "loop_id": record.get("loop_id"),
            "status": record.get("status"),
            "preserved": record.get("preserved"),
            "candidate_commit": record.get("candidate_commit"),
            "parent_commit": record.get("parent_commit"),
            "baseline_commit": record.get("baseline_commit"),
            "artifact": record.get("artifact"),
            "correctness_recorded": isinstance(record.get("correctness"), dict),
            "level1_decision": _get(record, "level1", "decision"),
            "level1_forward_backward_improvement_pct": _pct(level1_gain),
            "level2_baseline_median_tps": _get(record, "level2", "baseline_median_tps"),
            "level2_candidate_median_tps": _get(record, "level2", "candidate_median_tps"),
            "level2_improvement_pct": _pct(_get(record, "level2", "improvement_fraction")),
            "trainer_evaluated": record.get("trainer_evaluated"),
            "quality_evaluated": record.get("quality_evaluated"),
            "statistically_confirmed": record.get("statistically_confirmed"),
            "next": record.get("next"),
        }
        writer.writerow({key: _scalar(value) for key, value in row.items()})
    return output.getvalue()


def summary_text(records: list[dict[str, Any]]) -> str:
    statuses = Counter(str(record.get("status", "missing")) for record in records)
    summary = {
        "schema_version": 1,
        "source": str(LEDGER.relative_to(ROOT)),
        "record_count": len(records),
        "first_loop_id": records[0].get("loop_id") if records else None,
        "last_loop_id": records[-1].get("loop_id") if records else None,
        "evidence_counts": {
            "candidate_commit": sum(bool(record.get("candidate_commit")) for record in records),
            "correctness": sum(isinstance(record.get("correctness"), dict) for record in records),
            "level1": sum(isinstance(record.get("level1"), dict) for record in records),
            "level2": sum(isinstance(record.get("level2"), dict) for record in records),
            "profile": sum(isinstance(record.get("profile"), dict) for record in records),
            "statistically_confirmed": sum(record.get("statistically_confirmed") is True for record in records),
        },
        "status_counts": dict(sorted(statuses.items())),
        "notes": [
            "Counts describe append-only ledger events, not necessarily unique source candidates.",
            "Status names are preserved verbatim; invalid, rejected, neutral, and corrected events remain visible.",
            "Historical throughput values are comparable only within their declared matched block.",
        ],
    }
    return json.dumps(summary, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if tracked exports are stale")
    args = parser.parse_args()
    records = load_records()
    expected = {
        CSV_OUTPUT: csv_text(records),
        SUMMARY_OUTPUT: summary_text(records),
    }
    stale = [path for path, text in expected.items() if not path.exists() or path.read_text() != text]
    if args.check:
        if stale:
            for path in stale:
                print(f"stale: {path.relative_to(ROOT)}")
            return 1
        print(f"KDA result exports are current ({len(records)} ledger events)")
        return 0
    for path, text in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
