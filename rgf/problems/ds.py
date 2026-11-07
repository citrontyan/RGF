"""Minimum Dominating Set (paper Def. 2).

``S`` dominates iff every vertex outside ``S`` has a neighbour in ``S``; the
objective is ``min |S|``.

Cut rule (paper §The Cut-Determination Sub-problem, DS paragraph): only the cut
edges are removed, **no vertex leaves the instance**, and the vertices of ``V_C``
keep the labels the CDSP gave them, so the sub-instances carry three-valued
labels ``{Unknown, 1, 0}``.

Because every cut edge joins two vertices of ``V_C``, a component can be adjacent
to a decided cut vertex that lives in another component.  ``dominated`` records
exactly that: a vertex is dominated by a decided ``1`` outside its own
sub-instance and its domination constraint is therefore already satisfied
(``docs/decisions.md`` item D-4).
"""

from __future__ import annotations

import numpy as np

from rgf.graph import Graph
from rgf.problems.base import CutOutcome, ECSP
from rgf.problems.exact import solve_ds


def redundant_prune(g: Graph, labels: np.ndarray) -> np.ndarray:
    """Delete redundant selected vertices from any legal dominating set.

    ``labels`` must dominate ``g``; the result still does and is never larger.
    """
    sel = np.asarray(labels, dtype=np.int8).astype(bool).copy()
    if g.n == 0:
        return np.zeros(0, dtype=np.int8)
    counts = np.zeros(g.n, dtype=np.int64)
    for v in np.flatnonzero(sel):
        counts[v] += 1
        counts[g.neighbors(v)] += 1
    for v in np.argsort(g.degrees(), kind="stable"):  # 先试删度数小的
        v = int(v)
        if not sel[v]:
            continue
        closed = np.concatenate([[v], g.neighbors(v)])
        if np.all(counts[closed] >= 2):
            sel[v] = False
            counts[closed] -= 1
    return sel.astype(np.int8)


class DominatingSet(ECSP):
    name = "ds"
    maximize = False

    def feasible(self, g: Graph, labels: np.ndarray) -> bool:
        if labels.shape != (g.n,) or np.any((labels != 0) & (labels != 1)):
            return False
        selected = labels == 1
        for v in range(g.n):
            if selected[v]:
                continue
            if not np.any(selected[g.neighbors(v)]):
                return False
        return True

    def apply_cut(self, g: Graph, vc_ids: np.ndarray, s_c: np.ndarray) -> CutOutcome:
        labels = np.full(g.n, -1, dtype=np.int8)
        labels[vc_ids] = s_c.astype(np.int8)
        selected = vc_ids[s_c.astype(bool)]
        dominated = np.zeros(g.n, dtype=bool)
        if selected.size:
            nb = np.unique(np.concatenate([g.neighbors(v) for v in selected]))
            dominated[nb] = True

        # 最小可行性恢复（D-20）：已判 0 的顶点若全部邻居都已判 0，则它永远无法
        # 被支配（连本可支配它的未判定邻居都没有了），必须改判 1。翻转只会增加
        # 支配，反复扫描至稳定即可；该扫描同时覆盖先前层级判 0 的顶点——它们
        # 依赖的未判定邻居可能在本层被判 0。
        changed = True
        while changed:
            changed = False
            for v in np.flatnonzero(labels == 0):
                v = int(v)
                nb = g.neighbors(v)
                if nb.size == 0:
                    needs = True
                else:
                    needs = not np.any(labels[nb] == 1) and not np.any(labels[nb] < 0)
                if needs:
                    labels[v] = 1
                    dominated[v] = True
                    dominated[g.neighbors(v)] = True
                    changed = True

        # No vertex is removed for DS: the cut edges simply disappear.
        return CutOutcome(labels=labels, removed=np.zeros(g.n, dtype=bool), dominated=dominated)

    def repair_cdsp(self, g: Graph, cut, s_c: np.ndarray) -> np.ndarray:
        """V_C 侧的最小可行性检查（D-20 修订）。

        仅当某个被判 0 的割顶点**没有任何未判定邻居、也没有判 1 邻居**（即再无
        人能支配它）时才翻成 1。旧实现要求"判 0 割顶点必须在 V_C 内有判 1 邻点"，
        忽略了非割邻居本可在子问题中被选中来支配它——稠密图上割顶点约 30/35 的
        邻居是非割顶点，旧条件几乎每层都触发 0→1 翻转，是 RGF-DS 解规模膨胀
        （12.4→15.7）的主因。递归内部的全局可行性由 :meth:`apply_cut` 的恢复
        扫描兜底（含"先前判 0 顶点的最后潜在支配者被后续层级判 0"的情形）。
        """
        s_c = np.array(s_c, dtype=bool)
        vc = cut.vc_ids
        in_vc = np.zeros(g.n, dtype=bool)
        in_vc[vc] = True
        for idx, v in enumerate(vc):
            if s_c[idx]:
                continue
            v = int(v)
            nb = g.neighbors(v)
            if nb.size == 0:
                s_c[idx] = True
                continue
            if np.any(in_vc[nb]):
                cut_nb = nb[in_vc[nb]]
                has_one = bool(np.any(s_c[np.searchsorted(vc, cut_nb)]))
            else:
                has_one = False
            has_free = bool(np.any(~in_vc[nb]))
            if not has_one and not has_free:
                s_c[idx] = True
        return s_c

    def solve_exact(self, g: Graph, labels: np.ndarray, dominated: np.ndarray) -> np.ndarray:
        return solve_ds(g, labels, dominated)

    def polish(self, g: Graph, labels: np.ndarray) -> np.ndarray:
        """重叠子实例合并后取并集会产生冗余选点，统一做一次冗余删除（D-21）。"""
        return redundant_prune(g, labels)
