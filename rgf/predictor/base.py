"""Predictor interface.

Algorithm 1 line 5 calls the learnable solver as ``F_theta(G, V_C)`` and expects
a CDSP solution ``S_C``: for every vertex of ``V_C``, whether it belongs to some
optimal solution of the *whole* instance (Def. 4).  A predictor therefore sees a
batch of ``(graph, V_C)`` pairs and returns one boolean vector per pair, aligned
with ``V_C``.

The paper's predictors are DIFUSCO and FastT2T (``rgf/predictor/difusco.py`` and
``rgf/predictor/fastt2t.py``, wired in later phases).  The two heuristics below
exist so that the framework, the partitioners and the tests can run before the
learned predictors are connected; they are **not** paper baselines and every
result produced with them is labelled as such.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from rgf.graph import Graph


class Predictor(Protocol):
    name: str

    def predict(self, batch: list[tuple[Graph, np.ndarray]]) -> list[np.ndarray]:
        """Return one boolean array per (graph, vc_ids) pair, aligned with vc_ids."""


class ConstantPredictor:
    """Always answers ``value``; useful for smoke tests."""

    name = "constant"

    def __init__(self, value: bool = True):
        self.value = value

    def predict(self, batch: list[tuple[Graph, np.ndarray]]) -> list[np.ndarray]:
        return [np.full(vc.size, self.value, dtype=bool) for _, vc in batch]


class GreedyDegreePredictor:
    """Degree heuristic over the cut vertices: take a vertex iff no neighbour in
    ``V_C`` was already taken (a maximal independent set inside the cut).

    D-20：batch 项可含第三元素 ``ctx=(labels, dominated)``；给出时跳过已判 0 的
    割顶点与已被高层决策支配的割顶点（它们无需再选）。仅在"无神经元冒烟/诊断"
    场景使用，所有结果均须标注。
    """

    name = "greedy-degree"

    def predict(self, batch: list[tuple]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for item in batch:
            g, vc = item[0], item[1]
            ctx = item[2] if len(item) > 2 else None
            skip = np.zeros(g.n, dtype=bool)
            dominated = np.zeros(g.n, dtype=bool)
            if ctx is not None:
                labels, pre_dom = ctx
                labels = np.asarray(labels)
                skip |= labels == 0
                one = labels == 1
                dominated |= one
                for v in np.flatnonzero(one):
                    dominated[g.neighbors(v)] = True
                if pre_dom is not None:
                    dominated |= np.asarray(pre_dom, dtype=bool)
            perm = np.argsort(-g.degrees()[vc], kind="stable")
            order = vc[perm]
            sub, _, _ = g.induced(order)
            chosen = np.zeros(sub.n, dtype=bool)
            blocked = np.zeros(sub.n, dtype=bool)
            for i, v in enumerate(order):
                # 全部用局部索引：blocked / sub 都是 induced(order) 的局部编号；
                # dominated / skip 用全局 id 查（ctx 的标签数组是按全局顶点编号的）。
                # 回归修复（HEAD 3662e9f 引入）：旧实现用全局 id 索引局部数组并采
                # g.neighbors(v)（全局邻居 id）写 blocked，v >= sub.n 时直接 IndexError。
                if blocked[i] or dominated[v] or skip[v]:
                    continue
                chosen[i] = True
                blocked[i] = True
                blocked[sub.neighbors(i)] = True
            # chosen 与 order 对齐，调用方预期与 vc 对齐：按 perm 还原。
            res = np.zeros(vc.size, dtype=bool)
            res[perm] = chosen
            out.append(res)
        return out
