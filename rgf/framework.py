"""Algorithm 1: the Recursive Generative Framework.

Structure (paper Algorithm 1):

1. ``|V| <= alpha``  -> hand the sub-instance to the exact solver ``S*`` (line 3).
2. otherwise        -> partitioner gives ``C, V_C`` (line 4), the predictor gives
   ``S_C`` (line 5), sub-instances are extracted with the cut edges and the
   decided vertices removed (line 6), and the framework recurses (line 7-9).

Two properties of the paper's analysis are implemented literally:

* **Batching of NN inferences.**  "Subgraphs at the same depth are independent
  ... we can parallelize inferences across all subgraphs at a given depth by
  treating them as a single (disconnected) graph, requiring only one NN
  inference" (paper §Framework Details).  Hence one predictor call per depth
  level, not one per sub-instance.
* **Partial prediction.**  When ``beta`` is given, the predictor only sees
  ``N_G(V_C)`` (Algorithm 2, ``rgf/predictor/neighborhood.py``).

Two implementation-level decisions that the paper leaves open are recorded in
``docs/decisions.md``: the exact solver (D-3), and the explicit ``dominated``
flag needed by the DS variant (D-4).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from rgf.graph import Graph
from rgf.partition.base import Partitioner
from rgf.predictor.base import Predictor
from rgf.predictor.neighborhood import predict_with_neighborhood
from rgf.problems.base import ECSP

if TYPE_CHECKING:  # 仅类型标注；记录器为 None 时模块 import 与行为都不受影响
    from rgf.decision_log import DecisionRecorder


@dataclass
class Task:
    """One recursive sub-instance."""

    graph: Graph
    ids: np.ndarray  # original vertex ids, aligned with graph local ids
    labels: np.ndarray  # int8 local, -1 = undecided
    dominated: np.ndarray  # bool local, DS only
    # D-21（覆盖型问题）：叶子精确求解时并入的"另一侧自由割顶点"外圈。
    #   collar_graph: 剩余子图 + 外圈（含跨侧边）的实例； None = 无外圈。
    #   collar_ids / collar_labels / collar_dominated / collar_free: 与 collar_graph
    #   局部编号对齐（collar_free=True 的顶点是外圈自由候选，不携带全局决策）。
    # 外圈只作用于叶子求解，不进入递归（避免每层累积导致子实例只增不减）。
    collar_graph: Graph | None = None
    collar_ids: np.ndarray | None = None
    collar_labels: np.ndarray | None = None
    collar_dominated: np.ndarray | None = None
    collar_free: np.ndarray | None = None


@dataclass
class SolveStats:
    levels: int = 0
    nn_inferences: int = 0
    exact_calls: int = 0
    cut_vertices: int = 0
    sub_instances: int = 0
    fallback_exact: int = 0
    wall_clock_s: float = 0.0
    per_level: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["wall_clock_s"] = round(self.wall_clock_s, 3)
        return d


class RGFSolver:
    """Recursive Generative Framework for one ECSP."""

    def __init__(
        self,
        problem: ECSP,
        partitioner: Partitioner,
        predictor: Predictor,
        alpha: int,
        beta: int | None = None,
        seed: int = 0,
        max_levels: int = 256,
        decode_scope: str = "full",
        overlap: bool = False,
        decision_recorder: "DecisionRecorder | None" = None,
    ):
        self.problem = problem
        self.partitioner = partitioner
        self.predictor = predictor
        self.alpha = int(alpha)
        self.beta = beta
        self.seed = seed
        self.max_levels = max_levels
        # "full"：预测器/解码看到含切边的完整子实例（论文原样）；
        # "component"：去掉切边后再送入（分解一致解码，D-20）——DS 的"判 0"决策
        # 依赖跨切边的支配上下文，切边移除后该前提失效，必须让解码与分解一致。
        if decode_scope not in ("full", "component"):
            raise ValueError(f"unknown decode_scope: {decode_scope}")
        self.decode_scope = decode_scope
        # 子实例重叠（D-21，覆盖型问题用）：每个子实例并入割边另一侧的 V_C 顶点
        # （作为自由顶点），使"判 0 边界顶点被另一侧支配"的决策在子实例内可满足；
        # 各子实例解的选中取并集，最后由 problem.polish 做冗余删除。
        self.overlap = bool(overlap)
        # 逐层预测决策记录（可选，纯测量；None 时与历史行为完全一致）。
        # 见 rgf/decision_log.py 与 scripts/analyze_predictor_errors.py。
        self.decision_recorder = decision_recorder

    # ------------------------------------------------ legality restoration (D-35)
    def _global_legality_restore(self, g: Graph, labels_out: np.ndarray) -> int:
        """Restore global legality of the partially merged solution, in place.

        2026-11-04（docs/decisions.md D-35）：``single-vertex`` 分区（以及任意会产生
        孤立单点分量的分区方式）会让"已判 0、但支配它的邻居要等后续层才决定"的顶点
        进入叶 MILP，导致 ``MILP infeasible``。这里在**每层结束**做一次全局扫描：

        * MIS：两端都判 1 的边 → 把其中一个翻 0；
        * VC：两端都判 0 的边 → 把度数较大的端点翻 1（只有 0→1，不会破坏覆盖）；
        * DS：已判 0、未被任何已判 1 顶点支配、且所有邻居都已决（或没有邻居）的顶点
          → 翻 1（0→1 只增强支配）。

        对本来合法的解（例如 METIS 路径的历史结果）这是 no-op，因此回归结果不变。
        返回翻转次数。
        """
        name = self.problem.name
        flips = 0
        changed = True
        while changed:
            changed = False
            if name == "mis":
                if g.m:
                    both = np.flatnonzero((labels_out[g.edges[:, 0]] == 1)
                                          & (labels_out[g.edges[:, 1]] == 1))
                    for e in both:
                        labels_out[int(g.edges[e, 1])] = 0
                        flips += 1
                        changed = True
            elif name == "vc":
                if g.m:
                    both = np.flatnonzero((labels_out[g.edges[:, 0]] == 0)
                                          & (labels_out[g.edges[:, 1]] == 0))
                    if both.size:
                        deg = g.degrees()
                        for e in both:
                            u, v = int(g.edges[e, 0]), int(g.edges[e, 1])
                            labels_out[u if deg[u] >= deg[v] else v] = 1
                            flips += 1
                            changed = True
            else:  # ds
                one = labels_out == 1
                dominated = one.copy()
                for v in np.flatnonzero(one):
                    dominated[g.neighbors(v)] = True
                for x in np.flatnonzero(labels_out == 0):
                    if dominated[x]:
                        continue
                    nb = g.neighbors(x)
                    no_future = nb.size == 0 or (
                        np.all(labels_out[nb] >= 0) and not np.any(labels_out[nb] == 1)
                    )
                    if no_future:
                        labels_out[x] = 1
                        flips += 1
                        changed = True
        return flips

    def _patch_task_after_restore(self, task: "Task", labels_out: np.ndarray, g: Graph) -> None:
        """把全局恢复对已决标签的改动同步回下一层 Task（labels / dominated / collar）。"""
        mask = task.labels >= 0
        if mask.any():
            task.labels[mask] = labels_out[task.ids[mask]]
        if self.problem.name == "ds":
            for idx in np.flatnonzero(task.labels == 1):
                task.dominated[task.graph.neighbors(idx)] = True
            # 外部（其它子任务/后续层）已判 1 的邻居同样支配本任务顶点，否则下一层
            # MILP 仍会误判"任务内无解"（D-35 的 singleton 叶任务即此情形）。
            for idx in range(task.ids.size):
                nb = g.neighbors(int(task.ids[idx]))
                if nb.size and np.any(labels_out[nb] == 1):
                    task.dominated[idx] = True
        if task.collar_graph is not None and task.collar_ids is not None:
            core = ~task.collar_free
            decided = core & (task.collar_labels >= 0)
            if decided.any():
                task.collar_labels[decided] = labels_out[task.collar_ids[decided]]
            if self.problem.name == "ds":
                for idx in np.flatnonzero(core & (task.collar_labels == 1)):
                    task.collar_dominated[task.collar_graph.neighbors(idx)] = True
                for pos in np.flatnonzero(task.collar_free):
                    if labels_out[int(task.collar_ids[pos])] == 1:
                        task.collar_dominated[task.collar_graph.neighbors(pos)] = True

    def _ds_fixed_zero_blocked(self, task: "Task", dom: np.ndarray, idx: int) -> bool:
        """该已判 0 顶点是否"任务内已无可能被支配"（邻居全决且无 1）。"""
        if np.any(task.graph.neighbors(idx)[task.labels[task.graph.neighbors(idx)] < 0]):
            return False  # 任务内还有自由邻居，MILP 可以把它们设成 1
        if np.any(task.labels[task.graph.neighbors(idx)] == 1):
            return False  # 任务内已有 1 邻居
        return True

    def _ds_task_needs_external_wait(self, task: "Task", g: Graph,
                                     labels_out: np.ndarray) -> bool:
        """DS 专用：任务内已"无解可救"的已判 0 顶点，是否还能被**未来外部** 1 支配。

        2026-11-04（D-35）：这类顶点若立即丢给 MILP，模型会强制它在任务内被支配而
        infeasible；但它的外部邻居可能后续被判 1，因此先 defer。只要外部已全部决定，
        就交给 :meth:`_resolve_deferred_ds`（全局恢复/翻转）。
        """
        if self.problem.name != "ds" or task.collar_graph is not None:
            return False
        dom = task.dominated.copy()
        for idx in np.flatnonzero(task.labels == 1):
            dom[task.graph.neighbors(idx)] = True
        for idx in range(task.ids.size):
            nb = g.neighbors(int(task.ids[idx]))
            if nb.size and np.any(labels_out[nb] == 1):
                dom[idx] = True
        task.dominated = dom  # 让随后的 solve_exact 看到外部支配
        for idx in np.flatnonzero(task.labels == 0):
            if dom[idx] or not self._ds_fixed_zero_blocked(task, dom, int(idx)):
                continue
            nb = g.neighbors(int(task.ids[idx]))
            if np.any(labels_out[nb] < 0):
                return True
        return False

    def _resolve_deferred_ds(self, deferred: list["Task"], g: Graph,
                             labels_out: np.ndarray) -> list["Task"]:
        """裁决 deferred 的 DS 任务：外部都已决 → 未支配的 0 翻 1，否则继续等。"""
        remaining = list(deferred)
        progress = True
        while progress:
            progress = False
            nxt: list[Task] = []
            for task in remaining:
                mask = task.labels >= 0
                task.labels[mask] = labels_out[task.ids[mask]]
                dom = task.dominated.copy()
                for idx in np.flatnonzero(task.labels == 1):
                    dom[task.graph.neighbors(idx)] = True
                for idx in range(task.ids.size):
                    nb = g.neighbors(int(task.ids[idx]))
                    if nb.size and np.any(labels_out[nb] == 1):
                        dom[idx] = True
                task.dominated = dom
                waiting = False
                to_flip: list[int] = []
                for idx in np.flatnonzero(task.labels == 0):
                    if dom[idx]:
                        continue
                    nb = g.neighbors(int(task.ids[idx]))
                    if np.any(labels_out[nb] < 0):
                        waiting = True
                    else:
                        to_flip.append(int(idx))
                if waiting:
                    nxt.append(task)
                    continue
                for idx in to_flip:
                    task.labels[idx] = 1
                    dom[task.graph.neighbors(idx)] = True
                task.dominated = dom
                if np.any(task.labels < 0):
                    # 任务内仍有自由顶点：现在外部已决定，交给精确求解（不再会误判 infeasible）
                    solved = self.problem.solve_exact(task.graph, task.labels, task.dominated)
                    task.labels = np.asarray(solved, dtype=np.int8)
                mask = task.labels >= 0
                labels_out[task.ids[mask]] = task.labels[mask]
                progress = True
            remaining = nxt
        return remaining

    # ------------------------------------------------------------------ solve
    def solve(self, g: Graph) -> tuple[np.ndarray, SolveStats]:
        rng = np.random.default_rng(self.seed)
        stats = SolveStats()
        started = time.perf_counter()

        labels_out = np.full(g.n, -1, dtype=np.int8)
        frontier = [Task(g, np.arange(g.n, dtype=np.int64), np.full(g.n, -1, dtype=np.int8), np.zeros(g.n, dtype=bool))]
        # 2026-11-04（D-35）：single-vertex 分区会产生"孤立单点分量"，其标签在第 0 层已判 0、
        # 但支配它的邻居要等后续层才决定；这样的叶任务不能立刻丢给 MILP（会把"可能由外部
        # 未来 1 支配"误判成 infeasible）。先 defer，等外部邻居都决定后再判定/翻转。
        deferred: list[Task] = []

        for level in range(self.max_levels):
            if not frontier:
                break
            stats.levels = level + 1
            # 逐层遥测（纯测量，不改任何算法分支）：n_total=本层所有任务的顶点数之和，
            # vc_total=本层 |V_C| 之和，predicted_positives=本层预测器在 V_C 上判 1 的
            # 个数（repair_cdsp 之后的生效决策），elapsed_s/exact_elapsed_s=本层墙钟与
            # 其中叶子精确求解的墙钟。
            level_started = time.perf_counter()
            level_n_total = int(sum(task.graph.n for task in frontier))
            level_vc_total = 0
            level_predicted_positives = 0
            pending: list[tuple[Task, object]] = []
            batch: list[tuple[Graph, np.ndarray]] = []
            small: list[Task] = []

            for task in frontier:
                if task.graph.n <= self.alpha or task.graph.m == 0:
                    small.append(task)
                    continue
                cut = self.partitioner.partition(task.graph, rng)
                if cut.vc_ids.size == 0:
                    small.append(task)
                    stats.fallback_exact += 1
                    continue
                pending.append((task, cut))
                ctx = (task.labels, task.dominated)  # D-20 覆盖感知解码的上下文
                if self.decode_scope == "component":
                    # 分解一致解码（D-20）：去掉切边、保留全部顶点（编号不变，
                    # cut.vc_ids 仍有效），预测器/解码看到的是递归实际求解的两侧。
                    decomp, _, _ = task.graph.induced(
                        np.arange(task.graph.n, dtype=np.int64), edge_mask=~cut.edge_mask
                    )
                    batch.append((decomp, cut.vc_ids, ctx))
                else:
                    batch.append((task.graph, cut.vc_ids, ctx))
                stats.cut_vertices += int(cut.vc_ids.size)
                level_vc_total += int(cut.vc_ids.size)

            level_exact_started = time.perf_counter()
            for task in small:
                if self._ds_task_needs_external_wait(task, g, labels_out):
                    deferred.append(task)
                    continue
                if task.collar_graph is None:
                    solved = self.problem.solve_exact(task.graph, task.labels, task.dominated)
                    labels_out[task.ids] = solved
                else:
                    # D-21（覆盖型问题）：叶子精确求解在外圈并入另一侧自由割顶点作为
                    # 支配候选。解回后：本侧顶点的赋值直接写回；外圈候选若被选中，
                    # 以并集记 1（它本来就是某邻接分量的成员，其支配关系成立）。
                    solved = self.problem.solve_exact(
                        task.collar_graph, task.collar_labels, task.collar_dominated
                    )
                    core = ~task.collar_free
                    labels_out[task.collar_ids[core]] = solved[core]
                    labels_out[task.collar_ids[(~core) & (solved == 1)]] = 1
                    # 未被选中的外圈顶点保持其全局取值（0），它们自身的支配由
                    # 自己所在分量的叶子精确解保证。
                stats.exact_calls += 1
            level_exact_elapsed = time.perf_counter() - level_exact_started

            next_frontier: list[Task] = []
            if batch:
                s_c_list = predict_with_neighborhood(self.predictor, batch, self.beta)
                stats.nn_inferences += 1
                for task_index, ((task, cut), s_c) in enumerate(zip(pending, s_c_list)):
                    # Legality safeguard: an approximate predictor may return an
                    # S_C that no solution is consistent with; repair it before
                    # the cut is resolved (docs/decisions.md item D-10).
                    # 记录器开启时先留一份 repair 前的原始预测（纯测量，不进算法分支）。
                    s_raw = (
                        np.asarray(s_c, dtype=bool)
                        if self.decision_recorder is not None
                        else None
                    )
                    s_c = self.problem.repair_cdsp(task.graph, cut, np.asarray(s_c, dtype=bool))
                    if self.decision_recorder is not None:
                        self.decision_recorder.record(
                            level=level,
                            task_index=task_index,
                            n=task.graph.n,
                            m=task.graph.m,
                            vertex=task.ids[cut.vc_ids],
                            degree=task.graph.degrees()[cut.vc_ids],
                            pred_pre=s_raw,
                            pred_post=s_c,
                        )
                    level_predicted_positives += int(np.count_nonzero(s_c))
                    outcome = self.problem.apply_cut(task.graph, cut.vc_ids, s_c)
                    decided = outcome.labels >= 0
                    labels_out[task.ids[decided]] = outcome.labels[decided]

                    keep = ~outcome.removed
                    if not keep.any():
                        continue
                    edge_keep = ~cut.edge_mask
                    keep_ids = np.flatnonzero(keep)
                    remaining, rem_ids, _ = task.graph.induced(keep_ids, edge_mask=edge_keep)
                    for comp in remaining.components():
                        comp_graph = rem_ids[comp]  # 本侧顶点（task.graph 编号）
                        rec, _, _ = remaining.induced(comp)
                        rec_ids = task.ids[comp_graph]
                        rec_labels = outcome.labels[comp_graph].astype(np.int8)
                        rec_labels[rec_labels < 0] = -1
                        rec_dom = outcome.dominated[comp_graph]

                        collar_graph = None
                        collar_ids = collar_labels = collar_dom = collar_free = None
                        if self.overlap and comp_graph.size:
                            # D-21（覆盖型问题）：叶子求解的外圈 = 割边另一侧的
                            # V_C 顶点（判 0 者放开为自由候选），从父图诱导以保留
                            # 跨侧割边。外圈只在叶子精确求解时存在，不进入递归——
                            # 否则每层新增的外圈不断累积，子实例永远缩不回 α 以下
                            # （递归深度与实例数双重爆炸）。
                            nbrs = np.concatenate(
                                [task.graph.neighbors(int(c)) for c in comp_graph]
                            )
                            outside_vc = np.setdiff1d(
                                np.intersect1d(nbrs, cut.vc_ids), comp_graph
                            )
                            if outside_vc.size:
                                aug = np.union1d(comp_graph, outside_vc)
                                collar_graph, aug_parent, _ = task.graph.induced(aug)
                                collar_ids = task.ids[aug_parent]
                                collar_labels = outcome.labels[aug_parent].astype(np.int8).copy()
                                own = np.isin(aug_parent, comp_graph)
                                collar_labels[~own & (collar_labels == 0)] = -1  # 判 0 外圈放开为自由
                                collar_labels[collar_labels < 0] = -1
                                collar_dom = outcome.dominated[aug_parent]
                                collar_free = ~own

                        next_frontier.append(
                            Task(rec, rec_ids, rec_labels, rec_dom,
                                 collar_graph, collar_ids, collar_labels, collar_dom, collar_free)
                        )
                        stats.sub_instances += 1

            # 2026-11-04（D-35）：每层结束做一次**全局**合法性恢复（不只是 task 内），
            # 再把恢复结果同步回下一层 Task 的 labels/dominated/collar。METIS 路径若本来
            # 合法则是 no-op（回归测试对比 checksum 不变）。
            flips = self._global_legality_restore(g, labels_out)
            if deferred:
                deferred = self._resolve_deferred_ds(deferred, g, labels_out)
                flips += self._global_legality_restore(g, labels_out)
            if flips:
                for _task in next_frontier:
                    self._patch_task_after_restore(_task, labels_out, g)

            stats.per_level.append(
                {
                    "level": level,
                    "tasks": len(frontier),
                    "predicted_instances": len(batch),
                    "exact_instances": len(small),
                    "next_tasks": len(next_frontier),
                    "n_total": level_n_total,
                    "vc_total": level_vc_total,
                    "predicted_positives": level_predicted_positives,
                    "elapsed_s": round(time.perf_counter() - level_started, 6),
                    "exact_elapsed_s": round(level_exact_elapsed, 6),
                }
            )
            frontier = next_frontier
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"maximum recursion depth ({self.max_levels}) exceeded")

        # 2026-11-04（D-35）：收尾再跑一次全局合法性恢复；deferred 任务在此刻所有顶点都已
        # 决定，若仍有未支配的已判 0 顶点则翻成 1（0→1 只增强支配，保持合法性）。
        self._global_legality_restore(g, labels_out)
        if deferred:
            deferred = self._resolve_deferred_ds(deferred, g, labels_out)
        for _task in deferred:  # 极端兜底：外部邻居仍 -1（不该发生）
            for _idx in np.flatnonzero(_task.labels == 0):
                _gid = int(_task.ids[_idx])
                _nb = g.neighbors(_gid)
                if labels_out[_gid] == 0 and not np.any(labels_out[_nb] == 1):
                    labels_out[_gid] = 1
        self._global_legality_restore(g, labels_out)

        if np.any(labels_out < 0):
            raise RuntimeError("RGF left vertices undecided (incomplete solution)")
        if self.overlap:
            # 重叠子实例取并集会产生冗余选点，由问题自带的收尾清理（D-21）
            labels_out = self.problem.polish(g, labels_out)

        stats.wall_clock_s = time.perf_counter() - started
        return labels_out, stats
