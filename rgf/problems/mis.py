"""Maximum Independent Set (paper Def. 1).

``S`` is independent iff no edge has both endpoints in ``S``; the objective is
``max |S|``.

Cut rule (Eq. ``mis_r``): after the CDSP fixes ``V_C`` to ``S_C``, the vertices

    {v | v in V_C  or  exists (u, v) in E with u in S_C}

leave the instance — the cut vertices themselves and every neighbour of a
vertex decided to 1, which the independence constraint forces to 0.
"""

from __future__ import annotations

import numpy as np

from rgf.graph import Graph
from rgf.problems.base import CutOutcome, ECSP
from rgf.problems.exact import solve_mis


class MaximumIndependentSet(ECSP):
    name = "mis"
    maximize = True

    def feasible(self, g: Graph, labels: np.ndarray) -> bool:
        if labels.shape != (g.n,) or np.any((labels != 0) & (labels != 1)):
            return False
        if g.m == 0:
            return True
        return bool(np.all(labels[g.edges[:, 0]] + labels[g.edges[:, 1]] <= 1))

    def apply_cut(self, g: Graph, vc_ids: np.ndarray, s_c: np.ndarray) -> CutOutcome:
        labels = np.full(g.n, -1, dtype=np.int8)
        labels[vc_ids] = s_c.astype(np.int8)
        chosen = vc_ids[s_c.astype(bool)]
        removed = np.zeros(g.n, dtype=bool)
        removed[vc_ids] = True
        if chosen.size:
            nb = np.unique(np.concatenate([g.neighbors(v) for v in chosen]))
            labels[nb] = 0
            removed[nb] = True
        return CutOutcome(labels=labels, removed=removed, dominated=np.zeros(g.n, dtype=bool))

    def repair_cdsp(self, g: Graph, cut, s_c: np.ndarray) -> np.ndarray:
        """Drop decisions that would put two adjacent ``V_C`` vertices in the set."""
        s_c = np.array(s_c, dtype=bool)
        vc = cut.vc_ids
        order = np.argsort(-g.degrees()[vc], kind="stable")
        chosen = np.zeros(g.n, dtype=bool)
        out = s_c.copy()
        for idx in order:
            v = int(vc[idx])
            if not s_c[idx]:
                continue
            nb = g.neighbors(v)
            if np.any(chosen[nb]):
                out[idx] = False
            else:
                chosen[v] = True
        return out

    def solve_exact(self, g: Graph, labels: np.ndarray, dominated: np.ndarray) -> np.ndarray:
        return solve_mis(g, labels)
