from __future__ import annotations
"""
Structured processing log writer for all pipeline scripts.

Each script should call log_run() on completion to append a row to
outputs/reports/processing_log.csv.
"""

import csv
import time
from datetime import datetime
from pathlib import Path

from src.utils.paths import get_output_path, ensure_dir

LOG_COLUMNS = [
    "timestamp",
    "script",
    "input_paths",
    "output_paths",
    "parameters",
    "status",
    "warnings",
    "runtime_seconds",
]


def log_run(
    script: str,
    input_paths: list[str],
    output_paths: list[str],
    parameters: dict,
    status: str,
    warnings: list[str] | None = None,
    runtime_seconds: float = 0.0,
) -> None:
    """Append a processing record to outputs/reports/processing_log.csv."""
    log_path = get_output_path("reports/processing_log.csv")
    ensure_dir(log_path.parent)

    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "script": script,
                "input_paths": "; ".join(str(p) for p in input_paths),
                "output_paths": "; ".join(str(p) for p in output_paths),
                "parameters": str(parameters),
                "status": status,
                "warnings": "; ".join(warnings or []),
                "runtime_seconds": round(runtime_seconds, 2),
            }
        )


class Timer:
    """Simple context-manager timer."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
