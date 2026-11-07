"""Size-balanced random partitioning (paper §Experimental Settings, Table 2).

"randomly assigns vertices to two partitions while ensuring balanced sizes".
The paper does not fix the balance tolerance or the RNG seed; we use an exact
split (``ceil(n/2)`` / ``floor(n/2)``) drawn uniformly at random, which is the
strongest reading of "balanced" (``docs/decisions.md`` item D-5).
"""

from __future__ import annotations

import numpy as np

from rgf.graph import Graph
from rgf.partition.base import Cut, Partitioner


class RandomBalancedPartitioner(Partitioner):
    name = "random"

    def partition(self, g: Graph, rng: np.random.Generator) -> Cut:
        if g.n < 2:
            return self.empty_cut(g)
        side = np.zeros(g.n, dtype=np.int8)
        perm = rng.permutation(g.n)
        side[perm[: g.n // 2]] = 1
        return self.cut_from_assignment(g, side)
