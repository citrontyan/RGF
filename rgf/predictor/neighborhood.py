"""Partial prediction: the neighbourhood subgraph of Algorithm 2.

For real-world instances the paper does not hand the whole graph to ``F_theta``
but only ``N_G(V_C)``, the neighbourhood of the cut vertices grown breadth-first
until ``beta`` vertices are reached (Algorithm 2).  ``beta`` is an engineering
parameter: the largest value that fits in GPU memory (paper §Scaling to
Real-world Large Graphs).  Predictions are then read off at the positions of
``V_C``.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from rgf.graph import Graph
from rgf.predictor.base import Predictor


def _neighborhood(
    g: Graph, vc_ids: np.ndarray, beta: int
) -> tuple[Graph, np.ndarray, np.ndarray]:
    """Alg.2 内部版：返回 ``(子图, 子图顶点→原图 id, V_C 在子图内的局部 id)``。"""
    vc_ids = np.asarray(vc_ids, dtype=np.int64)
    if vc_ids.size == 0:
        empty = Graph.from_edges(0, np.empty((0, 2), dtype=np.int64))
        return empty, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if vc_ids.size >= beta:
        # ``V_C`` alone already exceeds the budget: keep the cut vertices only.
        sub, ids, pos = g.induced(vc_ids)
        return sub, ids, pos[vc_ids]

    in_set = np.zeros(g.n, dtype=bool)
    in_set[vc_ids] = True
    queue: deque[int] = deque(int(v) for v in vc_ids)
    size = int(vc_ids.size)
    while queue and size < beta:
        v = queue.popleft()
        for u in g.neighbors(v):
            if in_set[u]:
                continue
            in_set[u] = True
            size += 1
            queue.append(int(u))
            if size >= beta:
                break
    vertices = np.flatnonzero(in_set)
    sub, ids, pos = g.induced(vertices)
    return sub, ids, pos[vc_ids]


def neighborhood_subgraph(g: Graph, vc_ids: np.ndarray, beta: int) -> tuple[Graph, np.ndarray]:
    """Algorithm 2: grow ``V_C`` breadth-first until ``beta`` vertices, induce.

    Returns the neighbourhood subgraph and the local ids of ``V_C`` inside it.
    """
    sub, _, local_vc = _neighborhood(g, vc_ids, beta)
    return sub, local_vc


def predict_with_neighborhood(
    predictor: Predictor,
    batch: list[tuple],
    beta: int | None,
) -> list[np.ndarray]:
    """Run ``predictor`` on ``N_G(V_C)`` when ``beta`` is set, else on the whole graph.

    batch 项可为 ``(graph, vc_ids)`` 或 ``(graph, vc_ids, ctx)``（D-20 覆盖感知解码的
    上下文）。

    **分块（2026-10-04 补，D-25）**：Alg.2 的 Ensure 是"size β 的邻域子图"，但伪码在
    ``|V_C| >= β`` 时只会返回 ``V_C`` 的诱导子图（不受 β 约束）。稠密图上单层 METIS 割
    可达 21 万顶点（coPapersDBLP），直接喂模型必然 OOM。此处把 ``V_C`` 切成
    ``β/2`` 大小的分片、每片各自做 BFS 邻域扩张（≤β 顶点）后分别预测，再按原顺序拼回：
    语义不变（预测仍只看局部邻域），显存有界。``ctx`` 按子图顶点映射回流。
    """
    if beta is None:
        return predictor.predict(batch)

    out: list[np.ndarray] = []
    for item in batch:
        g, vc = item[0], item[1]
        ctx = item[2] if len(item) > 2 else None
        vc = np.asarray(vc, dtype=np.int64)
        if vc.size == 0:
            out.append(np.empty(0, dtype=bool))
            continue
        chunk = max(1, int(beta) // 2)  # 一半留给邻域扩张
        labels = np.zeros(vc.size, dtype=bool)
        parts: list[tuple] = []
        spans: list[tuple[int, int]] = []
        for s in range(0, vc.size, chunk):
            part = vc[s:s + chunk]
            sub, sub_ids, local_vc = _neighborhood(g, part, beta)
            sub_ctx = None
            if ctx is not None:
                labels_full, dom_full = ctx
                sub_ctx = (np.asarray(labels_full)[sub_ids],
                           None if dom_full is None else np.asarray(dom_full)[sub_ids])
            parts.append((sub, local_vc, sub_ctx))
            spans.append((s, s + part.size))
        raw = predictor.predict(parts)
        # 分块预测会逐块抬高 caching allocator 的 reserved 池（稠密图上实测可累积到
        # 上百 GB → 下一块 OOM），每块后归还一次（D-25；代价毫秒级）。
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - 无 CUDA 环境（CPU 冒烟）忽略
            pass
        for (s, e), pred in zip(spans, raw):
            labels[s:e] = np.asarray(pred, dtype=bool)
        out.append(labels)
    return out
