"""Lightweight per-run scratch log, distinct from RESULTS.md"""

from __future__ import annotations

import csv
import datetime

from . import paths

RUNS_TSV = paths.REPO_ROOT / "runs.tsv"

FIELDS = ["timestamp_utc", "phase", "experiment", "dataset", "metric", "value", "note"]


def append_run(
    phase: str, experiment: str, dataset: str, metric: str, value, note: str = ""
) -> None:
    """Appends one row to runs.tsv, creating the file with a header if it doesn't exist yet"""
    is_new = not RUNS_TSV.exists()
    with open(RUNS_TSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
                "phase": phase,
                "experiment": experiment,
                "dataset": dataset,
                "metric": metric,
                "value": value,
                "note": note,
            }
        )


def read_runs() -> list[dict]:
    if not RUNS_TSV.exists():
        return []
    with open(RUNS_TSV, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))
