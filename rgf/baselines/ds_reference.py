"""Reference solutions for the Dominating Set columns of Table 1.

The paper's baseline paragraph says the DS and VC instances are reduced to MIS
problems and handed to KaMIS.  For VC that is the standard complement relation
(``rgf/baselines/kamis.py::vc_via_mis``); for DS no such identity exists —
minimum dominating set is a hitting-set problem on the closed neighbourhoods and
is not the complement of an independent set — so the DS reference here is built
directly:

* small instances: HiGHS MILP, proven optimal;
* large instances: the same MILP under a wall-clock limit, keeping the incumbent;
* if the solver returns no incumbent in time: a greedy set-cover construction
  followed by a redundancy-removal pass.

Tables produced by this repository annotate the DS reference with the method
returned here, so the reported drop is read with the right baseline.
See ``docs/decisions.md`` item D-6.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from rgf.graph import Graph
from rgf.problems.ds import redundant_prune
from rgf.problems.exact import solve_ds_timed


@dataclass
class ReferenceSolution:
    labels: np.ndarray
    method: str
    optimal: bool
    objective: int
    time_s: float

    def as_dict(self) -> dict:
        return {
            "reference_method": self.method,
            "reference_optimal": self.optimal,
            "reference_size": self.objective,
            "reference_time_s": round(self.time_s, 2),
        }


def greedy_ds(g: Graph) -> np.ndarray:
    """Reverse-greedy dominating set: start from ``V`` and delete vertices whose
    closed neighbourhood stays dominated without them."""
    if g.n == 0:
        return np.zeros(0, dtype=np.int8)
    selected = np.ones(g.n, dtype=bool)
    counts = np.array([1 + g.neighbors(v).size for v in range(g.n)], dtype=np.int64)
    order = np.argsort(-g.degrees(), kind="stable")
    for v in order:
        v = int(v)
        if not selected[v]:
            continue
        closed = np.concatenate([[v], g.neighbors(v)])
        if np.all(counts[closed] >= 2):
            selected[v] = False
            counts[closed] -= 1
    return selected.astype(np.int8)


def ds_reference(g: Graph, time_limit_s: float | None = None) -> ReferenceSolution:
    """Reference solution for a DS instance (see the module docstring).

    2026-09-21 修正（docs/decisions.md D-18）：**不再无条件采用 MILP 的返回值**。
    实测 ER-[700-800]（n≈750）上 15 s 的 HiGHS incumbent 规模 194，而仓库里的逆向贪心
    只要 29、贪心解再做冗余删除只要 22.4 —— 直接采用 incumbent 会让参照解比平凡启发式
    还差 8.7 倍。现在对 MILP 解与逆向贪心解**统一做冗余删除并取更小者**（两者都是合法
    支配集，取更优不会破坏合法性）。
    """
    t0 = time.perf_counter()
    solved, optimal = None, False
    try:
        solved, optimal = solve_ds_timed(
            g, np.full(g.n, -1, dtype=np.int8), np.zeros(g.n, dtype=bool), time_limit_s=time_limit_s
        )
    except RuntimeError:
        pass

    candidates: list[tuple[str, np.ndarray]] = [("greedy+redundancy", redundant_prune(g, greedy_ds(g)))]
    if solved is not None:
        src = "milp-exact" if optimal else "milp-incumbent"
        candidates.append((f"{src}+redundancy", redundant_prune(g, solved)))

    best_method, best_labels = candidates[0]
    for name, labels in candidates[1:]:
        if int(np.count_nonzero(labels == 1)) < int(np.count_nonzero(best_labels == 1)):
            best_labels, best_method = labels, name

    return ReferenceSolution(
        labels=best_labels,
        method=best_method,
        optimal=bool(optimal and best_method.startswith("milp-exact")),
        objective=int(np.count_nonzero(best_labels == 1)),
        time_s=time.perf_counter() - t0,
    )
