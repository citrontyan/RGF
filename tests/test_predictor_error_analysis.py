"""自检：逐层预测决策记录 + FP/FN 误差率分析（theory 附录经验验证）。

覆盖三层契约：

1. ``DecisionRecorder`` 只增不改：开启后 ``RGFSolver`` 的计数（``cut_vertices`` /
   ``per_level``）与历史口径一致；关闭时原行为不变（由现有 telemetry 测试保证）。
2. ``scripts/analyze_predictor_errors.py`` 的 FP/FN/ρ 计算正确（手工构造的决策记录），
   且 L = ΣFN − ΣFP 恒等式与 V_NN 内/外分解自洽。
3. ``run_large_table.run_one`` 的 ``--dump-decisions`` 只在开启时写额外产物，
   且开启/关闭两次运行的 ``rgf_size`` / ``cut_vertices`` / ``per_level`` 完全一致。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rgf.decision_log import DecisionRecorder, load_decisions
from rgf.datasets.io import save_graph_npz
from rgf.framework import RGFSolver
from rgf.graph import Graph
from rgf.partition import RandomBalancedPartitioner
from rgf.problems import PROBLEMS


class AllOnePredictor:
    """对 V_C 全判 1 的朴素预测器（MIS 上必然被 repair_cdsp 删成合法独立集）。"""

    name = "all-one"

    def predict(self, batch):
        return [np.ones(np.asarray(item[1]).size, dtype=bool) for item in batch]


def _path_graph(n: int = 30) -> Graph:
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    return Graph.from_edges(n, edges)


def test_recorder_matches_framework_telemetry() -> None:
    problem = PROBLEMS["mis"]()
    g = _path_graph(18)
    recorder = DecisionRecorder()
    solver = RGFSolver(problem, RandomBalancedPartitioner(), AllOnePredictor(), alpha=2, seed=0,
                       decision_recorder=recorder)
    labels, stats = solver.solve(g)

    assert problem.feasible(g, labels)
    assert recorder.n_records == stats.cut_vertices
    assert recorder.n_records == sum(rec["vc_total"] for rec in stats.per_level)
    # 每个 batch 对应一次 NN 调用（predicted_instances 口径）
    assert len(recorder.batches) == sum(rec["predicted_instances"] for rec in stats.per_level)
    assert [b.level for b in recorder.batches] == sorted(b.level for b in recorder.batches)

    # repair_cdsp 对 MIS 只删不加：repair 后的 1 不可能多于 repair 前，
    # 且全 1 预测在相邻 V_C 顶点上必然被删掉一些 → 记录里应能看到改动。
    pre = np.concatenate([b.pred_pre for b in recorder.batches])
    post = np.concatenate([b.pred_post for b in recorder.batches])
    assert int(post.sum()) <= int(pre.sum())
    assert int(np.count_nonzero(pre != post)) > 0

    # 顶点 id 合法且互不重复（每个顶点只被预测一次，性质 3）
    vertex = np.concatenate([b.vertex for b in recorder.batches])
    assert vertex.min() >= 0 and vertex.max() < g.n
    assert np.unique(vertex).size == vertex.size

    # 度数就是决策时子图内的度数（路径图内部点度 2、端点度 1）
    degree = np.concatenate([b.degree for b in recorder.batches])
    assert set(np.unique(degree)) <= {0, 1, 2}


def test_recorder_npz_roundtrip(tmp_path: Path) -> None:
    problem = PROBLEMS["mis"]()
    g = _path_graph(12)
    recorder = DecisionRecorder()
    solver = RGFSolver(problem, RandomBalancedPartitioner(), AllOnePredictor(), alpha=2, seed=3,
                       decision_recorder=recorder)
    solver.solve(g)
    path = recorder.save(tmp_path / "dec.decisions.npz", metadata={"instance": "path12", "alpha": 2})

    arrays, metadata = load_decisions(path)
    assert metadata == {"instance": "path12", "alpha": 2}
    assert arrays["vertex"].size == recorder.n_records
    assert int(arrays["batch_start"][-1]) == recorder.n_records
    assert arrays["batch_start"].size == len(recorder.batches) + 1
    # 逐 batch 还原后与内存记录逐位一致
    for j, batch in enumerate(recorder.batches):
        sl = slice(int(arrays["batch_start"][j]), int(arrays["batch_start"][j + 1]))
        assert np.array_equal(arrays["vertex"][sl], batch.vertex)
        assert np.array_equal(arrays["degree"][sl], batch.degree)
        assert np.array_equal(arrays["pred_pre"][sl], batch.pred_pre)
        assert np.array_equal(arrays["pred_post"][sl], batch.pred_post)
        assert int(arrays["batch_level"][j]) == batch.level
        assert int(arrays["batch_task"][j]) == batch.task_index
        assert int(arrays["batch_n"][j]) == batch.n
        assert int(arrays["batch_m"][j]) == batch.m


def test_metrics_fp_fn_handcrafted() -> None:
    """手工构造决策记录，逐项核对 FP/FN/ρ、β、Δ̄ 与理论界。"""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from analyze_predictor_errors import metrics, whole_graph_decomposition

    rec = DecisionRecorder()
    # 第 0 层：V_C={0,1,2}，repair 前全 1，repair 后仅 {0}；参照 1-集为 {1,3,5}
    rec.record(level=0, task_index=0, n=8, m=7,
               vertex=np.array([0, 1, 2]), degree=np.array([2, 3, 4]),
               pred_pre=np.array([True, True, True]),
               pred_post=np.array([True, False, False]))
    # 第 1 层：V_C={3}，预测 1；参照里 3 属于 S*
    rec.record(level=1, task_index=0, n=4, m=3,
               vertex=np.array([3]), degree=np.array([1]),
               pred_pre=np.array([True]), pred_post=np.array([True]))
    arrays = {
        "batch_start": np.array([0, 3, 4], dtype=np.int64),
        "batch_level": np.array([0, 1], dtype=np.int32),
        "batch_task": np.array([0, 0], dtype=np.int32),
        "batch_n": np.array([8, 4], dtype=np.int64),
        "batch_m": np.array([7, 3], dtype=np.int64),
        "vertex": np.concatenate([b.vertex for b in rec.batches]),
        "degree": np.concatenate([b.degree for b in rec.batches]),
        "pred_pre": np.concatenate([b.pred_pre for b in rec.batches]),
        "pred_post": np.concatenate([b.pred_post for b in rec.batches]),
    }
    ref = np.zeros(8, dtype=np.int8)
    ref[[1, 3, 5]] = 1  # S_ref = {1, 3, 5}
    ref_pos = ref == 1
    metadata = {"kamis_ref": 3, "rgf_size": 2, "per_level": [
        {"level": 0, "predicted_instances": 1, "exact_instances": 0,
         "elapsed_s": 0.1, "exact_elapsed_s": 0.0},
        {"level": 1, "predicted_instances": 1, "exact_instances": 1,
         "elapsed_s": 0.2, "exact_elapsed_s": 0.1},
    ]}
    m = metrics(arrays, metadata, ref_pos, n_ref=8)

    # 决策与参照：v0 post=1/ref0 → FP；v1 post=0/ref1 → FN；v2 0/0；v3 1/1 ✓
    assert (m["fp"], m["fn"]) == (1, 1)
    assert m["v_nn"] == 4
    assert m["rho"] == pytest.approx(0.5)
    # pre：v0 与 v2 都预测 1 但参照 0 → 2 个 FP；v1 pre=1 与参照一致
    assert m["rho_pre"] == pytest.approx(0.5)
    assert m["beta_fp"] == pytest.approx(0.5)
    assert m["mean_degree_vc"] == pytest.approx((2 + 3 + 4 + 1) / 4)
    assert m["mean_degree_fp"] == pytest.approx(2.0)
    # 界 = ρ|V_NN|·((1-β)+βΔ̄) = 0.5*4*(0.5+0.5*2.5) = 3.5
    assert m["bound"] == pytest.approx(3.5)
    # L = 3 - 2 = 1；L/(ρ|V_NN|) = 1/2
    assert m["loss"] == 1
    assert m["loss_over_rho_vnn"] == pytest.approx(0.5)
    assert m["loss_over_bound"] == pytest.approx(1 / 3.5)
    per_level = {r["level"]: r for r in m["per_level"]}
    assert per_level[0]["vc_total"] == 3 and per_level[0]["fp"] == 1 and per_level[0]["fn"] == 1
    assert per_level[0]["rho"] == pytest.approx(2 / 3)
    assert per_level[0]["repair_changed"] == 2
    assert per_level[1]["vc_total"] == 1 and per_level[1]["rho"] == pytest.approx(0.0)

    # 最终解与决策一致（v0=1, v1=0, v2=0, v3=1），另在 V_NN 外多丢一个参照顶点 5：
    # |S_ref|=3、|S|=2 → L=1；恒等式 L = ΣFN − ΣFP = |{1,5}| − |{0}| = 1。
    final = np.zeros(8, dtype=np.int8)
    final[[0, 3]] = 1
    wg = whole_graph_decomposition(final, ref_pos, arrays["vertex"])
    assert wg["identity_ok"]
    assert wg["loss_identity"] == 1
    assert wg["loss_identity"] == wg["fn_all"] - wg["fp_all"]
    assert wg["fp_in_vnn"] == 1 and wg["fn_in_vnn"] == 1
    assert wg["loss_in_vnn"] == 0
    # 最终解在 V_C 之外的顶点（如 5）是叶子精确解/推断的结果
    assert wg["fn_outside_vnn"] == 1 and wg["loss_outside_vnn"] == 1


def test_run_one_dump_decisions_keeps_default_result(tmp_path: Path, monkeypatch) -> None:
    """--dump-decisions 开启 vs 关闭：核心结果完全一致，且只多写决策 npz。"""
    pytest.importorskip("torch")
    pytest.importorskip("scipy")
    import json

    import rgf.predictor.fastt2t as fastt2t
    from scripts.analyze_predictor_errors import analyze_one
    from scripts.run_large_table import run_one

    class ConstantPredictor:
        name = "constant"

        def __init__(self):
            self.seed = None

        def _predict_graph(self, g):
            return np.full(g.n, 0.5, dtype=float)

    n = 30
    graph = _path_graph(n)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    save_graph_npz(graph, data_dir / "path30.npz")
    labels = np.zeros(n, dtype=np.int8)
    labels[::2] = 1
    np.savez_compressed(data_dir / "path30.label.npz", labels=labels)

    monkeypatch.setattr(fastt2t, "FastT2TPredictor", lambda *a, **k: ConstantPredictor())

    def make_args(dump):
        return SimpleNamespace(
            deterministic=False, seed=0, data_dir=data_dir, kamis_time=1.0,
            predictor_seed="fixed", hidden=256, beta=8, partitioner="random",
            max_levels=0, alpha=2, cdsp_sampling=1, cdsp_rule="best", skip_rw=True,
            rw_budget=0.0, mem_gb=12.0, dump_decisions=dump,
        )

    out_a = tmp_path / "run_a" / "path30.json"
    out_a.parent.mkdir()
    run_one(data_dir / "path30.npz", make_args(None), out_a)
    payload_a = json.loads(out_a.read_text(encoding="utf-8"))
    assert "decisions_npz" not in payload_a and "decisions_records" not in payload_a

    dec_dir = tmp_path / "decisions"
    out_b = tmp_path / "run_b" / "path30.json"
    out_b.parent.mkdir()
    run_one(data_dir / "path30.npz", make_args(dec_dir), out_b)
    payload_b = json.loads(out_b.read_text(encoding="utf-8"))
    decisions_path = Path(payload_b["decisions_npz"])
    assert decisions_path.exists() and decisions_path.parent == dec_dir
    assert payload_b["decisions_records"] == payload_b["cut_vertices"]

    # 除墙钟时间外，两次 JSON 必须逐字段一致（新 flag 纯测量，不改算法）
    def strip_volatile(payload):
        clean = {k: v for k, v in payload.items()
                 if k not in ("rgf_time_s", "decisions_npz", "decisions_records")}
        clean["per_level"] = [
            {k: v for k, v in rec.items() if k not in ("elapsed_s", "exact_elapsed_s")}
            for rec in payload["per_level"]
        ]
        return clean

    assert strip_volatile(payload_a) == strip_volatile(payload_b)
    # 最终解标签也必须一致
    assert np.array_equal(
        np.load(out_a.with_suffix(".rgf_labels.npz"))["labels"],
        np.load(out_b.with_suffix(".rgf_labels.npz"))["labels"],
    )

    # 分析脚本把决策记录与参照解对上：|V_NN| 与 cut_vertices 一致，恒等式成立
    report = analyze_one(
        decisions_path,
        labels=data_dir / "path30.label.npz",
        rgf_labels=out_b.with_suffix(".rgf_labels.npz"),
        result_json=out_b,
        extra_labels=[data_dir / "path30.label.npz"],
    )
    m = report["metrics"]
    assert m["v_nn"] == payload_b["cut_vertices"]
    assert m["loss"] == payload_b["kamis_ref"] - payload_b["rgf_size"]
    assert report["whole_graph"] is not None and report["whole_graph"]["identity_ok"]
    assert report["whole_graph"]["fp_in_vnn"] == m["fp"]
    assert report["whole_graph"]["fn_in_vnn"] == m["fn"]
    assert report["cross_check"]["match"] is True
    assert report["cross_check"]["cut_vertices_match"] is True
    # 同一参照解当附加参照时 ρ 必然相同
    assert report["reference_sensitivity"][0]["rho"] == pytest.approx(m["rho"])


def test_aggregate_scans_analysis_json(tmp_path: Path) -> None:
    import json
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from analyze_predictor_errors import aggregate_report, render_aggregate_markdown

    sub = tmp_path / "alpha500"
    sub.mkdir()
    report = {
        "decisions": "/tmp/x.decisions.npz",
        "config": {"instance": "luxembourg.osm", "alpha": 500, "seed": 0, "rgf_drop": 0.3},
        "metrics": {
            "v_nn": 100, "fp": 4, "fn": 2, "rho": 0.06, "beta_fp": 2 / 3,
            "mean_degree_vc": 2.1, "mean_degree_fp": 2.0, "loss": 5,
            "loss_over_vnn": 0.05, "loss_over_rho_vnn": 0.8333,
            "bound": 9.0, "loss_over_bound": 5 / 9, "rho_pre": 0.07,
        },
    }
    (sub / "x.analysis.json").write_text(json.dumps(report), encoding="utf-8")
    agg = aggregate_report(tmp_path)
    assert agg["count"] == 1 and agg["runs"][0]["instance"] == "luxembourg.osm"
    md = render_aggregate_markdown(agg)
    assert "luxembourg.osm" in md and "α" in md
