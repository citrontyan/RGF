"""Exact MILP solvers for the three ECSP sub-instances.

The paper writes ``S*`` for "the exact solver" invoked when ``|V| <= alpha``
(Algorithm 1, line 3) without naming an implementation; we use HiGHS through
``scipy.optimize.milp``.  See ``docs/decisions.md`` item D-3.

The ``*_timed`` variants additionally report whether the optimum was proven;
they back the DS reference solutions of Table 1 (``docs/decisions.md`` item D-6).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

from rgf.graph import Graph


def _solve_min(
    n: int,
    c: np.ndarray,
    a_ub: sp.spmatrix | None,
    b_ub,
    a_eq,
    b_eq,
    time_limit_s: float | None = None,
) -> tuple[np.ndarray, bool]:
    """Solve a 0-1 ILP; returns ``(solution, optimal)``.

    ``optimal`` is False when the solver stopped early (time limit) but still
    returned an incumbent.
    """
    constraints = []
    if a_ub is not None:
        constraints.append(LinearConstraint(a_ub, -np.inf, b_ub))
    if a_eq is not None and a_eq.shape[0]:
        constraints.append(LinearConstraint(a_eq, b_eq, b_eq))
    options: dict = {"presolve": True}
    if time_limit_s is not None:
        options["time_limit"] = float(time_limit_s)
    res = milp(
        c=c,
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
        constraints=constraints,
        options=options,
    )
    if res.x is None:
        raise RuntimeError(f"MILP failed: {res.message}")
    solution = np.rint(res.x).astype(np.int8)
    return solution, bool(res.status == 0)


def _fixed_constraints(labels: np.ndarray) -> tuple[sp.spmatrix, np.ndarray, np.ndarray]:
    fixed = np.flatnonzero(labels >= 0)
    if fixed.size == 0:
        return sp.csr_matrix((0, labels.size)), np.empty(0), fixed
    a_eq = sp.csr_matrix((np.ones(fixed.size), (np.arange(fixed.size), fixed)), shape=(fixed.size, labels.size))
    return a_eq, labels[fixed].astype(float), fixed


def _merge(labels: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.array(labels, dtype=np.int8, copy=True)
    free = labels < 0
    out[free] = x[free]
    return out


def _edge_ub(g: Graph, sign: float, rhs: float):
    if g.m == 0:
        return None, None
    rows = np.repeat(np.arange(g.m), 2)
    cols = g.edges.reshape(-1)
    a_ub = sign * sp.csr_matrix((np.ones(2 * g.m), (rows, cols)), shape=(g.m, g.n))
    return a_ub, rhs * np.ones(g.m)


# ------------------------------------------------------------------ MIS
def solve_mis_timed(g: Graph, labels: np.ndarray, time_limit_s: float | None = None):
    a_eq, b_eq, _ = _fixed_constraints(labels)
    a_ub, b_ub = _edge_ub(g, 1.0, 1.0)  # x_u + x_v <= 1
    x, optimal = _solve_min(g.n, -np.ones(g.n), a_ub, b_ub, a_eq, b_eq, time_limit_s)
    return _merge(labels, x), optimal


def solve_mis(g: Graph, labels: np.ndarray) -> np.ndarray:
    return solve_mis_timed(g, labels)[0]


# ------------------------------------------------------------------- VC
def solve_vc_timed(g: Graph, labels: np.ndarray, time_limit_s: float | None = None):
    a_eq, b_eq, _ = _fixed_constraints(labels)
    a_ub, b_ub = _edge_ub(g, -1.0, -1.0)  # x_u + x_v >= 1
    x, optimal = _solve_min(g.n, np.ones(g.n), a_ub, b_ub, a_eq, b_eq, time_limit_s)
    return _merge(labels, x), optimal


def solve_vc(g: Graph, labels: np.ndarray) -> np.ndarray:
    return solve_vc_timed(g, labels)[0]


# ------------------------------------------------------------------- DS
def _ds_rows(g: Graph, dominated: np.ndarray):
    active = np.flatnonzero(~dominated)
    if active.size == 0:
        return None, None
    cols = []
    counts = []
    for v in active:
        nb = g.neighbors(v)
        cols.append(np.concatenate([[v], nb]))
        counts.append(1 + nb.size)
    counts = np.array(counts)
    rows = np.repeat(np.arange(active.size), counts)
    cols = np.concatenate(cols)
    a_ub = -sp.csr_matrix((np.ones(cols.size), (rows, cols)), shape=(active.size, g.n))
    return a_ub, -np.ones(active.size)


def solve_ds_timed(g: Graph, labels: np.ndarray, dominated: np.ndarray, time_limit_s: float | None = None):
    a_eq, b_eq, _ = _fixed_constraints(labels)
    a_ub, b_ub = _ds_rows(g, dominated)
    x, optimal = _solve_min(g.n, np.ones(g.n), a_ub, b_ub, a_eq, b_eq, time_limit_s)
    return _merge(labels, x), optimal


def solve_ds(g: Graph, labels: np.ndarray, dominated: np.ndarray) -> np.ndarray:
    return solve_ds_timed(g, labels, dominated)[0]
