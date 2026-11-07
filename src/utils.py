"""Shared helpers: seeding, JSON/CSV I/O, timing and logging.

Owner: Muhammad Abdullah Saeed (see docs/DELIVERABLE_2_PLAN.md, Section 3.6).
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
LOG_DATEFMT = "%H:%M:%S"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed `random`, `numpy` and `torch` (torch is optional)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # sklearn-only machines (e.g. Zia's fallback path)
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op on CPU-only machines


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    """Make numpy scalars/arrays, Paths, sets and dataclasses JSON-serialisable."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _ensure_parent(path: str | os.PathLike) -> None:
    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_json(path: str | os.PathLike, obj: Any, indent: int = 2) -> None:
    """Write `obj` as UTF-8 JSON, creating parent folders if needed."""
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=_json_default)
        f.write("\n")


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(
    path: str | os.PathLike,
    rows: Sequence[dict],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    """Write a list of dicts as CSV. Column order = `fieldnames` or first row's keys."""
    _ensure_parent(path)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


# ---------------------------------------------------------------------------
# Timing and logging
# ---------------------------------------------------------------------------
class Timer:
    """Context manager measuring wall-clock seconds with `time.perf_counter()`.

    >>> with Timer("epoch") as t:
    ...     do_work()
    >>> t.elapsed   # seconds
    """

    def __init__(self, label: str = "", logger: Optional[logging.Logger] = None) -> None:
        self.label = label
        self.logger = logger
        self.elapsed = 0.0
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self.logger is not None:
            self.logger.info("%s took %.2f s", self.label or "block", self.elapsed)


def configure_logging(
    level: int | str = logging.INFO, log_file: Optional[str] = None
) -> logging.Logger:
    """Configure root logging (console, plus an optional file) and return a logger."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        _ensure_parent(log_file)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("nhr")
