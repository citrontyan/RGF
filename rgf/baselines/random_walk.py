"""Random Walk baseline (paper Table 1, real-world large graphs).

The paper: "we also evaluate the performance of Random Walk (RW) with greedy
initialization on the same graphs.  At each instance, we first run our approach
and record the wall clock time.  Then we run RW with the same wall clock time."

The walk itself is not specified in the paper (``docs/decisions.md`` item D-5):
we use the standard randomised local search for MIS -- greedy maximal
independent set as the initial state, then repeated (1) pick a vertex not in the
set; (2) if it has no neighbour in the set, insert it; (3) otherwise eject one
conflicting neighbour and insert it -- with uniform restarts.
"""

from __future__ import annotations

import time

import numpy as np

from rgf.graph import Graph, greedy_maximal_independent_set


def random_walk_mis(
    g: Graph,
    time_budget_s: float,
    rng: np.random.Generator | None = None,
    greedy_init: bool = True,
) -> tuple[np.ndarray, int]:
    """Return ``(labels, iterations)`` for MIS under a wall-clock budget."""
    rng = rng or np.random.default_rng(0)
    if greedy_init:
        labels = greedy_maximal_independent_set(g, rng)
    else:
        labels = np.zeros(g.n, dtype=np.int8)
    best = labels.copy()
    best_size = int(np.count_nonzero(best == 1))

    if g.m == 0:
        return np.ones(g.n, dtype=np.int8), 0

    # adjacency as an array of arrays for fast membership tests
    neighbours = [g.neighbors(v) for v in range(g.n)]
    in_set = labels == 1
    # 增量维护规模：大图上 ``in_set.sum()`` 是 O(n)、每轮都做会让迭代率低几个数量级
    size = int(in_set.sum())
    deadline = time.perf_counter() + time_budget_s
    iters = 0
    while time.perf_counter() < deadline:
        iters += 1
        v = int(rng.integers(g.n))
        if in_set[v]:
            continue
        nb = neighbours[v]
        conflict = nb[in_set[nb]] if nb.size else nb
        if conflict.size == 0:
            in_set[v] = True
            size += 1
        elif conflict.size == 1:
            # 1-1 交换（规模不变，用于脱离平台）+ 对腾出的顶点做贪心回填
            # （回填成功即增长 1）。只弹一个冲突邻居时解仍合法。
            u = int(conflict[0])
            in_set[u] = False
            in_set[v] = True
            cand = neighbours[u]
            free = cand[(~in_set[cand]) & (cand != v)] if cand.size else cand
            if free.size:
                w = int(free[rng.integers(free.size)])
                wnb = neighbours[w]
                if not (wnb.size and in_set[wnb].any()):
                    in_set[w] = True
                    size += 1
        else:
            continue  # 多冲突：跳过（弹全部只减不增，见下）
        if size > best_size:
            best_size = size
            best = in_set.astype(np.int8)
    return best, iters
