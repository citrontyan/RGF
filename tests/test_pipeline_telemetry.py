"""Self-checks for the pipeline hardening added for the large-graph experiments.

The alpha-runner / seeds-runner read these artifacts directly, so they are part of
the experimental contract rather than mere implementation details:

* per-level telemetry of ``RGFSolver`` (``rgf/framework.py``);
* ``--instances all`` must not treat derived ``*.label.npz``-style files as graphs;
* ``--deterministic`` must actually flip torch's determinism flags;
* ``run_large_table.run_one`` must write ``cut_vertices`` / ``per_level`` into the
  result JSON (the drop-vs-|V_NN| analysis depends on them).
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
import pytest

from rgf.datasets.io import save_graph_npz
from rgf.framework import RGFSolver
from rgf.graph import Graph
from rgf.partition import SingleVertexPartitioner
from rgf.problems import PROBLEMS

TELEMETRY_FIELDS = {
    "level",
    "tasks",
    "predicted_instances",
    "exact_instances",
    "next_tasks",
    "n_total",
    "vc_total",
    "predicted_positives",
    "elapsed_s",
    "exact_elapsed_s",
}


class OraclePredictor:
    """Complete predictor: solves each CDSP exactly (test-only)."""

    def __init__(self, problem):
        self.problem = problem

    def predict(self, batch):
        out = []
        for item in batch:
            g, vc = item[0], item[1]
            labels = np.full(g.n, -1, dtype=np.int8)
            solved = self.problem.solve_exact(g, labels, np.zeros(g.n, dtype=bool))
            out.append(solved[vc] == 1)
        return out


class ConstantPredictor:
    """Stand-in for FastT2TPredictor: no model, no GPU, deterministic heatmap."""

    name = "constant"

    def __init__(self):
        self.seed = None

    def _predict_graph(self, g):
        return np.full(g.n, 0.5, dtype=float)


def _random_graph(n: int, p: float, rng: np.random.Generator) -> Graph:
    upper = rng.random((n, n))[np.triu_indices(n, k=1)]
    iu, ju = np.triu_indices(n, k=1)
    keep = upper < p
    edges = np.stack([iu[keep], ju[keep]], axis=1).astype(np.int64)
    return Graph.from_edges(n, edges)


def test_per_level_telemetry_fields_and_invariants() -> None:
    problem = PROBLEMS["mis"]()
    g = _random_graph(14, 0.3, np.random.default_rng(7))
    solver = RGFSolver(problem, SingleVertexPartitioner(), OraclePredictor(problem),
                       alpha=2, seed=0)
    labels, stats = solver.solve(g)

    assert problem.feasible(g, labels)
    assert stats.per_level, "recursion must produce at least one level"
    assert [rec["level"] for rec in stats.per_level] == list(range(len(stats.per_level)))
    for rec in stats.per_level:
        assert TELEMETRY_FIELDS <= set(rec)
        assert rec["n_total"] > 0
        assert 0 <= rec["vc_total"]
        # repair_cdsp 只删不加，判 1 的顶点数不可能超过本层 |V_C| 之和
        assert 0 <= rec["predicted_positives"] <= rec["vc_total"]
        assert rec["elapsed_s"] >= 0.0
        assert rec["exact_elapsed_s"] >= 0.0
        assert rec["exact_elapsed_s"] <= rec["elapsed_s"] + 1e-4

    # 纯测量：逐层累计必须与既有 SolveStats 计数口径一致
    assert sum(rec["vc_total"] for rec in stats.per_level) == stats.cut_vertices
    assert sum(rec["exact_instances"] for rec in stats.per_level) == stats.exact_calls
    assert sum(rec["predicted_instances"] for rec in stats.per_level) == stats.nn_inferences


def test_instances_all_skips_derived_npz(tmp_path) -> None:
    pytest.importorskip("scipy")
    from scripts.run_large_table import select_instances

    for name in (
        "a.osm.npz",
        "a.osm.label.npz",
        "a.osm.rgf_labels.npz",
        "a.osm.dsref.npz",
        "a.osm.opt.npz",
        "b.npz",
    ):
        (tmp_path / name).write_bytes(b"not really a graph")

    args = argparse.Namespace(instances="all", data_dir=tmp_path, instance="")
    assert select_instances(args) == ["a.osm.npz", "b.npz"]

    listing = argparse.Namespace(instances="x.osm, y", data_dir=tmp_path, instance="")
    assert select_instances(listing) == ["x.osm.npz", "y.npz"]

    single = argparse.Namespace(instances="", data_dir=tmp_path, instance="z.osm")
    assert select_instances(single) == ["z.osm.npz"]


def test_configure_determinism_flips_torch_flags() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    import random

    from scripts.run_large_table import configure_determinism

    was_enabled = torch.are_deterministic_algorithms_enabled()
    try:
        configure_determinism(123)
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert random.random() == random.Random(123).random()  # random.seed(123)
    finally:
        torch.use_deterministic_algorithms(was_enabled)


def test_run_large_table_json_contains_cut_vertices_and_per_level(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("torch")  # the FastT2T adapter imports torch at module level
    pytest.importorskip("scipy")
    import rgf.predictor.fastt2t as fastt2t
    from scripts.run_large_table import run_one

    # 30 顶点路径图：MIS = 偶数顶点（15 个），参照标签直接落盘（不跑 KaMIS）
    n = 30
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    graph = Graph.from_edges(n, edges)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    save_graph_npz(graph, data_dir / "path30.npz")
    labels = np.zeros(n, dtype=np.int8)
    labels[::2] = 1
    np.savez_compressed(data_dir / "path30.label.npz", labels=labels)

    monkeypatch.setattr(fastt2t, "FastT2TPredictor", lambda *a, **k: ConstantPredictor())
    args = SimpleNamespace(
        deterministic=False, seed=0, data_dir=data_dir, kamis_time=1.0,
        predictor_seed="fixed", hidden=256, beta=8, partitioner="random",
        max_levels=0, alpha=2, cdsp_sampling=1, cdsp_rule="best", skip_rw=True,
        rw_budget=0.0, mem_gb=12.0,
    )
    out = tmp_path / "path30.json"
    run_one(data_dir / "path30.npz", args, out)

    payload = json.loads(out.read_text(encoding="utf-8"))
    # 既有字段保持不动
    for key in (
        "instance", "n", "m", "alpha", "beta", "seed", "max_levels", "vc_level0",
        "kamis_ref", "rgf_size", "rgf_drop", "rgf_levels", "rgf_exact_calls",
        "rgf_nn_inferences", "rgf_conflicts", "rgf_nonmax", "rgf_size_repaired",
        "rw_size", "rw_drop", "rw_iterations",
    ):
        assert key in payload

    # 硬性验收：cut_vertices（各层 |V_C| 之和）+ per_level（逐层遥测）
    assert payload["instance"] == "path30"
    assert isinstance(payload["cut_vertices"], int)
    assert payload["cut_vertices"] >= 0
    assert payload["cut_vertices"] == sum(rec["vc_total"] for rec in payload["per_level"])
    assert payload["per_level"]
    for rec in payload["per_level"]:
        assert TELEMETRY_FIELDS <= set(rec)

    # 确定性 / 预测器种子口径落盘（默认与历史一致：off / fixed / 未强制）
    assert payload["deterministic"] is False
    assert payload["predictor_seed"] == "fixed"
    assert payload["predictor_seed_value"] == 0
    assert payload["predictor_seed_enforced"] is False


def test_reseed_wrapper_makes_fixed_predictions_reproducible() -> None:
    """--deterministic + --predictor-seed fixed 时，每次推理前的重播种必须生效。"""
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from scripts.run_large_table import ReseedGraphPredictor

    class NoisyPredictor:
        name = "noisy"

        def _predict_graph(self, g):
            return torch.rand(g.n).numpy()

    edges = np.stack([np.arange(11), np.arange(1, 12)], axis=1)
    g = Graph.from_edges(12, edges)
    wrapped = ReseedGraphPredictor(NoisyPredictor(), seed=7)
    first = wrapped._predict_graph(g)
    second = wrapped._predict_graph(g)
    assert np.array_equal(first, second)

