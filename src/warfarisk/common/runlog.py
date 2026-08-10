"""Lightweight per-run scratch log, distinct from RESULTS.md.

Mirrors a pattern confirmed across three sibling shared-task projects
(halluscoring/stanceeval/dialectsenteval's autoresearch/results.tsv): every
call to a phase's run_*/fit_* function appends one row here, gitignored, no
curation required. This is NOT the verified-results ledger — runs.tsv is
allowed to accumulate failed/abandoned/in-progress attempts freely. A row
only gets promoted into RESULTS.md once it has real evidence and someone
has looked at it, per RESULTS.md's own rule.
"""

from __future__ import annotations

import csv
import datetime
from pathlib import Path

from . import paths

RUNS_TSV = paths.REPO_ROOT / "runs.tsv"

FIELDS = ["timestamp_utc", "phase", "experiment", "dataset", "metric", "value", "note"]


def append_run(phase: str, experiment: str, dataset: str, metric: str, value, note: str = "") -> None:
    """Appends one row to runs.tsv, creating the file with a header if it
    doesn't exist yet. Call this from inside a phase script right after
    computing a result — it costs nothing and means a crash mid-experiment
    doesn't lose the runs that already finished.
    """
    is_new = not RUNS_TSV.exists()
    with open(RUNS_TSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
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
