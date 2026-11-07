"""Metrics of the paper's tables.

The paper reports, for every method, a **drop rate relative to KaMIS** and the
(inference / wall-clock) **time**.  Table 1 uses the drop for MIS, DS and VC on
RB/ER graphs and for MIS on real-world graphs; Table 2 uses it for the
partitioner ablation.

Sign convention (``docs/decisions.md`` item D-6): the drop rate always means
"how much worse than the reference", so smaller is better and the reference
itself is ``0%``.

    MIS (maximise):   drop = (|S_ref| - |S_ours|) / |S_ref| * 100
    DS/VC (minimise): drop = (|S_ours| - |S_ref|) / |S_ref| * 100
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def solution_size(labels: np.ndarray) -> int:
    return int(np.count_nonzero(labels == 1))


def drop_percent(ours: int, reference: int, maximize: bool) -> float:
    """Drop rate in percent; positive means worse than the reference."""
    if reference <= 0:
        raise ValueError("reference solution size must be positive")
    if maximize:
        return (reference - ours) / reference * 100.0
    return (ours - reference) / reference * 100.0


@dataclass
class RunResult:
    dataset: str
    problem: str
    method: str
    instance: str
    size: int
    drop: float | None
    time_s: float
    feasible: bool | None = None
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        row = {
            "dataset": self.dataset,
            "problem": self.problem,
            "method": self.method,
            "instance": self.instance,
            "size": self.size,
            "drop_%": None if self.drop is None else round(self.drop, 2),
            "time_s": round(self.time_s, 1),
            "feasible": self.feasible,
        }
        row.update(self.extra)
        return row
