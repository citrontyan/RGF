"""Legality regression tests for the D-35 global-legality repair (2026-11-04).

背景：``single-vertex`` 分区每层只剥离 1 个顶点，会把切点变成"孤立单点分量"。
``DominatingSet`` 下该顶点若被判 0，且支配它的邻居要等后续层才决定，就会让叶 MILP
收到"任务内必须支配、但任务内已无邻居"的固定 0 标签 → ``MILP infeasible``
（历史 single 消融只跑过 MIS，故未暴露；见 docs/decisions.md D-35）。

这些测试锁定两件事：
1. **修复合法的**：3 个 ECSP × partitioner=single × alpha∈{0,50} 的最终解必须合法；
2. **METIS 回归不变**：修复前本来合法的 METIS 结果（size/checksum/levels）逐位不变；
   修复前唯一一个非法的 METIS 小图用例（g8/ds/alpha=0）修复后必须合法（记录在案）。
"""

from __future__ import annotations

import itertools
import subprocess

import numpy as np
import pytest

from rgf.framework import RGFSolver
from rgf.graph import Graph
from rgf.partition import MetisPartitioner, SingleVertexPartitioner
from rgf.predictor.base import GreedyDegreePredictor
from rgf.problems import DominatingSet, MaximumIndependentSet, VertexCover

PROBLEMS = {"mis": MaximumIndependentSet, "ds": DominatingSet, "vc": VertexCover}

GRAPH_SPECS = {
    "g8": (8, 0.35, 0),
    "g10": (10, 0.30, 1),
    "g12": (12, 0.40, 2),
    "g20": (20, 0.25, 4),
    "g60": (60, 0.10, 3),
}

# 修复前（HEAD 3c4249a + D-35 patch 之前）用相同代码/相同 predictor 捕获的 METIS 结果：
# (graph, problem, alpha) -> (size, checksum, levels, feasible)
# 注意 g8/ds/0 是修复前唯一"跑完但解非法"的 METIS 用例；修复后必须 feasible=True。
METIS_PREFIX_FIXTURE = {
    ("g8", "mis", 50): (4, 17, 1, True),
    ("g8", "mis", 0): (3, 12, 2, True),
    ("g8", "ds", 50): (3, 14, 1, True),
    ("g8", "ds", 0): (4, 12, 4, False),
    ("g8", "vc", 50): (4, 19, 1, True),
    ("g8", "vc", 0): (4, 13, 2, True),
    ("g10", "mis", 50): (5, 23, 1, True),
    ("g10", "mis", 0): (5, 25, 2, True),
    ("g10", "ds", 50): (3, 15, 1, True),
    ("g10", "ds", 0): (5, 23, 5, True),
    ("g10", "vc", 50): (5, 32, 1, True),
    ("g10", "vc", 0): (6, 31, 2, True),
    ("g12", "mis", 50): (5, 30, 1, True),
    ("g12", "mis", 0): (4, 28, 2, True),
    ("g12", "ds", 50): (3, 28, 1, True),
    ("g12", "ds", 0): (6, 40, 5, True),
    ("g12", "vc", 50): (7, 48, 1, True),
    ("g12", "vc", 0): (8, 49, 2, True),
}


def make_graph(name: str) -> Graph:
    n, p, seed = GRAPH_SPECS[name]
    rng = np.random.default_rng(seed)
    edges = [(u, v) for u, v in itertools.combinations(range(n), 2) if rng.random() < p]
    return Graph.from_edges(n, np.array(edges, dtype=np.int64).reshape(-1, 2))


def checksum(labels: np.ndarray) -> int:
    return int(sum(int(v) * (i + 1) for i, v in enumerate(labels)) % (2 ** 61 - 1))


def solve(pname: str, g: Graph, partitioner, alpha: int, overlap: bool) -> dict:
    problem = PROBLEMS[pname]()
    solver = RGFSolver(problem, partitioner, GreedyDegreePredictor(), alpha=alpha, beta=None,
                       seed=0, decode_scope="full",
                       max_levels=2000 if alpha == 0 else 1000, overlap=overlap)
    labels, stats = solver.solve(g)
    return {
        "feasible": bool(problem.feasible(g, labels)),
        "size": int((labels == 1).sum()),
        "checksum": checksum(labels),
        "levels": stats.levels,
    }


@pytest.mark.parametrize("gname,pname,alpha", sorted(METIS_PREFIX_FIXTURE))
def test_metis_path_unchanged(gname: str, pname: str, alpha: int) -> None:
    """METIS 路径回归：修复前合法结果逐位不变；修复前非法用例必须被修复。"""
    expected = METIS_PREFIX_FIXTURE[(gname, pname, alpha)]
    g = make_graph(gname)
    try:
        got = solve(pname, g, MetisPartitioner(2, seed=0), alpha, False)
    except (FileNotFoundError, subprocess.CalledProcessError, RuntimeError) as exc:  # gpmetis 不可用
        pytest.skip(f"METIS partitioner unavailable: {exc}")
    size, cs, levels, pre_feasible = expected
    if not pre_feasible:
        # 修复前的退化用例：现在必须合法，且解不小于修复前（0→1 修复）
        assert got["feasible"] is True, (gname, pname, alpha, got)
        assert got["size"] >= size
        return
    assert got["feasible"] is True
    assert got["size"] == size, (gname, pname, alpha, got)
    assert got["checksum"] == cs, (gname, pname, alpha, got)
    assert got["levels"] == levels, (gname, pname, alpha, got)


@pytest.mark.parametrize("gname,pname,alpha", [
    ("g20", "mis", 0), ("g20", "mis", 50),
    ("g20", "vc", 0), ("g20", "vc", 50),
    ("g20", "ds", 0), ("g20", "ds", 50),
    ("g60", "mis", 0), ("g60", "mis", 50),
    ("g60", "vc", 0), ("g60", "vc", 50),
    ("g60", "ds", 0), ("g60", "ds", 50),
])
def test_single_vertex_final_solution_legal(gname: str, pname: str, alpha: int) -> None:
    """single-vertex 下三种 ECSP 的最终解必须合法（D-35 修复的验收条件）。"""
    g = make_graph(gname)
    got = solve(pname, g, SingleVertexPartitioner(), alpha, False)
    assert got["feasible"] is True, (gname, pname, alpha, got)


@pytest.mark.parametrize("gname,alpha", [("g20", 0), ("g20", 50), ("g60", 0), ("g60", 50)])
def test_single_vertex_ds_with_overlap_legal(gname: str, alpha: int) -> None:
    """DS 在 campaign 里默认开 overlap（D-21），single 分区下同样必须合法。"""
    g = make_graph(gname)
    got = solve("ds", g, SingleVertexPartitioner(), alpha, True)
    assert got["feasible"] is True, (gname, alpha, got)
