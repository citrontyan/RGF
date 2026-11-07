"""Single-vertex partitioning (paper §Experimental Settings, Table 2).

"selects one vertex as ``V_C`` in each recursion, effectively reducing our
framework to an autoregressive model".  The paper does not say how the single
vertex is picked; we use the maximum-degree vertex with random tie-breaking,
the natural choice for a solver that wants to fix influential decisions first
(``docs/decisions.md`` item D-5).
"""

from __future__ import annotations

import numpy as np

from rgf.graph import Graph
from rgf.partition.base import Cut, Partitioner


class SingleVertexPartitioner(Partitioner):
    name = "single-vertex"

    def partition(self, g: Graph, rng: np.random.Generator) -> Cut:
        if g.n == 0 or g.m == 0:
            return self.empty_cut(g)
        deg = g.degrees().astype(np.float64)
        v = int(np.argmax(deg + rng.random(g.n) * 1e-9))
        mask = (g.edges[:, 0] == v) | (g.edges[:, 1] == v)
        return Cut(edges=g.edges[mask], vc_ids=np.array([v], dtype=np.int64), edge_mask=mask)
