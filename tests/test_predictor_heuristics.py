"""Regression tests for the heuristic predictors in ``rgf/predictor/base.py``.

``GreedyDegreePredictor`` is the CPU-only predictor used by smoke tests and
diagnostics.  A HEAD regression (commit 3662e9f, the D-20 ``ctx`` rework)
indexed local arrays with global vertex ids, so every ``run_rgf.py
--predictor degree`` run crashed with ``IndexError``.  These tests pin the
intended semantics: decisions are aligned with ``V_C``, form a maximal
independent set inside ``V_C`` (only V_C neighbours are blocked), and honour
the D-20 per-vertex context.
"""

from __future__ import annotations

import numpy as np

from rgf.graph import Graph
from rgf.predictor.base import GreedyDegreePredictor


def _path_graph(n: int) -> Graph:
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.int64)
    return Graph.from_edges(n, edges)


def _chosen(vc: np.ndarray, decisions: np.ndarray) -> set[int]:
    return set(np.asarray(vc)[np.asarray(decisions, dtype=bool)].tolist())


def test_greedy_degree_no_index_error_with_large_global_ids() -> None:
    """以全局 id 为下标写局部数组的崩溃回归（V_C 含大 id 顶点）。"""
    g = _path_graph(120)
    vc = np.array([119, 2, 118, 61, 60, 5], dtype=np.int64)
    out = GreedyDegreePredictor().predict([(g, vc)])

    assert len(out) == 1
    decisions = np.asarray(out[0], dtype=bool)
    assert decisions.shape == vc.shape
    chosen = _chosen(vc, decisions)
    for u in chosen:
        assert not chosen.intersection(g.neighbors(int(u)).tolist())


def test_greedy_degree_returns_independent_set_aligned_with_vc() -> None:
    """返回向量必须按 V_C 顺序对齐，且是 V_C 内部的极大独立集。"""
    g = _path_graph(20)
    vc = np.array([9, 3, 17, 4, 12, 0, 8], dtype=np.int64)
    decisions = np.asarray(
        GreedyDegreePredictor().predict([(g, vc)])[0], dtype=bool
    )
    chosen = _chosen(vc, decisions)

    # 独立集：V_C 内选中的顶点两两不相邻
    for u in chosen:
        assert not chosen.intersection(g.neighbors(int(u)).tolist())
    # 极大性：未被选中的 V_C 顶点必须已有 V_C 内邻居被选中（无 ctx 时无 skip/dominated）
    chosen_neighbor = np.zeros(g.n, dtype=bool)
    for u in chosen:
        chosen_neighbor[g.neighbors(int(u))] = True
    for v in vc:
        if int(v) not in chosen:
            assert chosen_neighbor[int(v)], f"V_C vertex {v} is neither chosen nor dominated"


def test_greedy_degree_respects_ctx_labels_and_dominated() -> None:
    """D-20 上下文：已判 0（skip）、已判 1（自身被支配）与预支配顶点都不得入选。"""
    g = _path_graph(20)
    vc = np.array([0, 1, 2, 3, 4], dtype=np.int64)

    labels = np.full(g.n, -1, dtype=np.int8)
    labels[0] = 0   # skip
    labels[1] = 1   # 已选，自身与邻居都应被视为已支配
    dominated = np.zeros(g.n, dtype=bool)
    dominated[4] = True  # 预支配（高层决策的支配先验）

    decisions = np.asarray(
        GreedyDegreePredictor().predict([(g, vc, (labels, dominated))])[0], dtype=bool
    )
    assert decisions.shape == vc.shape
    index = {int(v): i for i, v in enumerate(vc)}
    assert not decisions[index[0]], "label 0 的 V_C 顶点不应入选"
    assert not decisions[index[1]], "label 1 的 V_C 顶点已由高层决策覆盖"
    assert not decisions[index[4]], "pre-dominated 的 V_C 顶点不应入选"
    # 2 与 3 相邻；1 已支配 2，故 2 不选，3 可选
    assert not decisions[index[2]]


def test_greedy_degree_batching_and_empty_vc() -> None:
    g = _path_graph(10)
    empty = np.empty(0, dtype=np.int64)
    vc = np.array([4, 5, 6], dtype=np.int64)
    out = GreedyDegreePredictor().predict([(g, empty), (g, vc)])
    assert len(out) == 2
    assert out[0].shape == (0,)
    assert out[1].shape == vc.shape
