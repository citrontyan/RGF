"""逐层预测决策记录（仅测量，不改算法）。

论文附录的"误差传播分析"需要经验测量预测器错误率
:math:`\\rho_d = (\\mathrm{FP}_d + \\mathrm{FN}_d) / |V_C^{(d)}|`
（相对某个参照最优解 :math:`S^*`），以及 :math:`|V_{NN}| = \\sum_d |V_C^{(d)}|`。
框架本身只保留最后解，不保留"每个顶点是在哪一层、由哪次预测决定的"，
因此这里提供一个可选的记录器：``RGFSolver(..., decision_recorder=recorder)``。

设计约束（见 ``docs/decisions.md`` 的"只增不改"约定）：

* 记录器为 ``None`` 时求解器行为与历史逐位一致（无额外分配、无额外分支）；
* 记录内容只是决策审计所需的原始量（层号、任务序号、全局顶点 id、repair
  前后的 S_C、决策时该顶点在子图内的度数），不做任何与参照解的比对——
  比对在 ``scripts/analyze_predictor_errors.py`` 里做，避免把评测逻辑
  混进求解器。

文件格式（``npz``，压缩；数组为按 batch 拼接的扁平数组）：

==============  ==========  ==========================================
key              dtype/shape  含义
==============  ==========  ==========================================
batch_start      int64[nb+1]  第 j 个 batch 在扁平数组中的起点/终点
batch_level      int32[nb]    递归层号
batch_task       int32[nb]    该层内的任务序号（与 ``frontier`` 顺序一致）
batch_n          int64[nb]    决策时子图的顶点数
batch_m          int64[nb]    决策时子图的边数
vertex           int64[nr]    V_C 顶点的**全局** id（原始图编号）
degree           int32[nr]    决策时该顶点在子图内的度数
pred_pre         bool[nr]     repair_cdsp 之前的预测 S_C
pred_post        bool[nr]     repair_cdsp 之后、真正生效的 S_C
metadata         str          JSON（实例、α、seed、参照解规模、逐层遥测等）
==============  ==========  ==========================================

用法::

    from rgf.decision_log import DecisionRecorder

    rec = DecisionRecorder()
    solver = RGFSolver(..., decision_recorder=rec)
    labels, stats = solver.solve(g)
    rec.save(path, metadata={"instance": "luxembourg.osm", "alpha": 500, ...})
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class LevelDecisions:
    """一个子任务在一次 NN 调用中给出的 CDSP 决策。"""

    level: int
    task_index: int
    n: int
    m: int
    vertex: np.ndarray  # int64, 全局 id
    degree: np.ndarray  # int32, 决策时子图内度数
    pred_pre: np.ndarray  # bool
    pred_post: np.ndarray  # bool


@dataclass
class DecisionRecorder:
    """收集 :class:`rgf.framework.RGFSolver` 的逐层 CDSP 决策（纯测量）。"""

    batches: list[LevelDecisions] = field(default_factory=list)

    def record(
        self,
        *,
        level: int,
        task_index: int,
        n: int,
        m: int,
        vertex: np.ndarray,
        degree: np.ndarray,
        pred_pre: np.ndarray,
        pred_post: np.ndarray,
    ) -> None:
        vertex = np.asarray(vertex, dtype=np.int64)
        degree = np.asarray(degree, dtype=np.int32)
        pre = np.asarray(pred_pre, dtype=bool)
        post = np.asarray(pred_post, dtype=bool)
        if not (vertex.size == degree.size == pre.size == post.size):
            raise ValueError(
                "decision record arrays must be aligned: "
                f"|V_C|={vertex.size} |deg|={degree.size} |pre|={pre.size} |post|={post.size}"
            )
        self.batches.append(
            LevelDecisions(int(level), int(task_index), int(n), int(m), vertex, degree, pre, post)
        )

    @property
    def n_records(self) -> int:
        """已记录的 V_C 顶点总数（= 求解器统计里的 ``cut_vertices``）。"""
        return int(sum(b.vertex.size for b in self.batches))

    # ---------------------------------------------------------------- persist
    def save(self, path: str | Path, metadata: dict | None = None) -> Path:
        """把全部决策写成压缩 npz（含元数据 JSON 字符串）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        starts = np.zeros(len(self.batches) + 1, dtype=np.int64)
        if self.batches:
            starts[1:] = np.cumsum([b.vertex.size for b in self.batches])
        cat = lambda name: (  # noqa: E731 - 局部拼接小工具
            np.concatenate([getattr(b, name) for b in self.batches])
            if self.batches
            else np.empty(0)
        )
        np.savez_compressed(
            path,
            batch_start=starts,
            batch_level=np.asarray([b.level for b in self.batches], dtype=np.int32),
            batch_task=np.asarray([b.task_index for b in self.batches], dtype=np.int32),
            batch_n=np.asarray([b.n for b in self.batches], dtype=np.int64),
            batch_m=np.asarray([b.m for b in self.batches], dtype=np.int64),
            vertex=cat("vertex"),
            degree=cat("degree"),
            pred_pre=cat("pred_pre"),
            pred_post=cat("pred_post"),
            metadata=np.asarray(json.dumps(metadata or {}, ensure_ascii=False)),
        )
        return path


def load_decisions(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """读回 :meth:`DecisionRecorder.save` 写出的 npz。

    Returns
    -------
    arrays : dict
        ``batch_start`` / ``batch_level`` / ``batch_task`` / ``batch_n`` /
        ``batch_m`` / ``vertex`` / ``degree`` / ``pred_pre`` / ``pred_post``。
    metadata : dict
        写入时附带的元数据（JSON 解析后）。
    """
    with np.load(path, allow_pickle=False) as data:
        arrays = {
            "batch_start": data["batch_start"].astype(np.int64),
            "batch_level": data["batch_level"].astype(np.int32),
            "batch_task": data["batch_task"].astype(np.int32),
            "batch_n": data["batch_n"].astype(np.int64),
            "batch_m": data["batch_m"].astype(np.int64),
            "vertex": data["vertex"].astype(np.int64),
            "degree": data["degree"].astype(np.int32),
            "pred_pre": data["pred_pre"].astype(bool),
            "pred_post": data["pred_post"].astype(bool),
        }
        metadata = json.loads(str(data["metadata"]))
    return arrays, metadata
