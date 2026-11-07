"""Minimum Vertex Cover (paper Def. 3).

``S`` covers iff every edge has an endpoint in ``S``; the objective is
``min |S|``.

Cut rule (Eq. ``vc_r``): the vertices

    {v | v in V_C  or  exists (u, v) in E with u in V_C \\ S_C}

leave the instance — the cut vertices themselves plus every neighbour of a cut
vertex the CDSP decided **not** to take, which the coverage constraint forces
to 1.
"""

from __future__ import annotations

import numpy as np

from rgf.graph import Graph
from rgf.problems.base import CutOutcome, ECSP
from rgf.problems.exact import solve_vc


class VertexCover(ECSP):
    name = "vc"
    maximize = False

    def feasible(self, g: Graph, labels: np.ndarray) -> bool:
        if labels.shape != (g.n,) or np.any((labels != 0) & (labels != 1)):
            return False
        if g.m == 0:
            return True
        return bool(np.all(labels[g.edges[:, 0]] + labels[g.edges[:, 1]] >= 1))

    def apply_cut(self, g: Graph, vc_ids: np.ndarray, s_c: np.ndarray) -> CutOutcome:
        labels = np.full(g.n, -1, dtype=np.int8)
        labels[vc_ids] = s_c.astype(np.int8)
        removed = np.zeros(g.n, dtype=bool)
        removed[vc_ids] = True
        unchosen = vc_ids[~s_c.astype(bool)]
        if unchosen.size:
            nb = np.unique(np.concatenate([g.neighbors(v) for v in unchosen]))
            nb = nb[labels[nb] < 0]
            labels[nb] = 1
            removed[nb] = True
        return CutOutcome(labels=labels, removed=removed, dominated=np.zeros(g.n, dtype=bool))

    def repair_cdsp(self, g: Graph, cut, s_c: np.ndarray) -> np.ndarray:
        """Cover every edge whose two endpoints are both decided by the CDSP.

        Those vertices leave the instance, so if such an edge ends up with both
        endpoints at 0 nothing downstream can cover it.  This includes the cut
        edges themselves and any other edge internal to ``V_C``.
        """
        s_c = np.array(s_c, dtype=bool)
        pos = np.full(g.n, -1, dtype=np.int64)
        pos[cut.vc_ids] = np.arange(cut.vc_ids.size)
        deg = g.degrees()
        both = (pos[g.edges[:, 0]] >= 0) & (pos[g.edges[:, 1]] >= 0)
        for u, v in g.edges[both]:
            iu, iv = int(pos[u]), int(pos[v])
            if not s_c[iu] and not s_c[iv]:
                s_c[iu if deg[u] >= deg[v] else iv] = True
        return s_c

    def solve_exact(self, g: Graph, labels: np.ndarray, dominated: np.ndarray) -> np.ndarray:
        return solve_vc(g, labels)
