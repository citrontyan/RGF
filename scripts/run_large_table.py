"""大图表格（论文 Table 3 / tab:ex）的完整驱动：KaMIS 参照 → RGF → 同墙钟 RW。

    wsl -e bash scripts/run_wsl.sh scripts/run_large_table.py \
        --instances luxembourg.osm,coAuthorsCiteseer --beta auto --alpha 500 --hidden 128

口径（论文 RQ4 / 算法 2 / D-9）：
* 预测器 = 预训练 Fast T2T，只喂 N_G(V_C) 邻域子图（β = 显存容得下的最大值）；
* α = 500（论文对大图的取值）；
* KaMIS 为 drop 的分母参照解，缓存在 `<data-dir>/<实例名>.label.npz`
  （`--data-dir` 默认 `/mnt/e/data/rgf/dimacs10`，与旧硬编码路径一致；
  外部数据用 `--data-dir /mnt/e/data/rgf/external`）；
* RW 与 RGF **同墙钟**（论文原话），贪心初始化。
结果写入 E:\\data\\rgf\\results\\large\\<实例名>.json；已存在则跳过（断点续跑）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

# 必须在 import torch 前设置（D-23：3090 上大图前向的碎片缓解，实测 2.3 倍提速）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
# --deterministic 需要的 cuBLAS 确定性工作区配置也必须在 import torch 之前生效。
# 这里直接扫 argv（不解析 argparse 结果，因为 torch 可能在 main() 之前就被间接导入）；
# 不带该 flag 时**不设置**该变量，默认运行环境与历史数字逐位一致。
if "--deterministic" in sys.argv[1:]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rgf.baselines.kamis import kamis_mis  # noqa: E402
from rgf.baselines.random_walk import random_walk_mis  # noqa: E402
from rgf.datasets.io import load_graph_npz  # noqa: E402
from rgf.decision_log import DecisionRecorder  # noqa: E402
from rgf.framework import RGFSolver  # noqa: E402
from rgf.partition import (  # noqa: E402
    MetisPartitioner,
    RandomBalancedPartitioner,
    SingleVertexPartitioner,
)
from rgf.predictor.neighborhood import neighborhood_subgraph  # noqa: E402
from rgf.problems import MaximumIndependentSet  # noqa: E402
from run_rgf import DecodedCutPredictor  # noqa: E402

DATA = Path("/mnt/e/data/rgf")
DIMACS = DATA / "dimacs10"
CKPTS = {
    256: DATA / "checkpoints/fastt2t/mis_rb.ckpt",   # RB 口径（hidden 256）
    128: DATA / "checkpoints/fastt2t/mis_er.ckpt",   # ER 口径（hidden 128）
}

# 实测的每边峰值显存（BFS 邻域子图，coPapersCiteseer，2026-10-04）：
#   hidden 256 约 15.5 KB/边；hidden 128 约 7.8 KB/边（显存与边数成正比）
BYTES_PER_EDGE = {256: 15.5e3, 128: 7.8e3}
MODEL_RESIDENT_GB = 1.5

# 派生文件后缀：这些 npz 不是图，`--instances all` 必须跳过（与 run_rgf.py 的
# 过滤写法对齐，并额外排除 RGF 自己写回的 *.rgf_labels.npz）。
DERIVED_NPZ_SUFFIXES = (".label.npz", ".rgf_labels.npz", ".dsref.npz", ".opt.npz")


def select_instances(args) -> list[str]:
    """把 ``--instance`` / ``--instances`` 解析成要跑的 npz 文件名。

    旧版 ``--instances all`` 直接 glob 全部 ``*.npz``，会把 KaMIS 参照
    ``*.label.npz``、DS 参照 ``*.dsref.npz``、精确最优 ``*.opt.npz``（以及结果目录里
    可能出现的 ``*.rgf_labels.npz``）当成图读入而报错；这里按后缀把它们排除。
    """
    if args.instances:
        if args.instances == "all":
            return sorted(
                p.name
                for p in args.data_dir.glob("*.npz")
                if not p.name.endswith(DERIVED_NPZ_SUFFIXES)
            )
        return [
            s.strip() if s.strip().endswith(".npz") else f"{s.strip()}.npz"
            for s in args.instances.split(",")
            if s.strip()
        ]
    return [args.instance if args.instance.endswith(".npz") else f"{args.instance}.npz"]


def _seed_torch_rngs(seed: int) -> None:
    """同时播种 CPU 与 CUDA 的 torch RNG（cuBLAS/TF32 由 configure_determinism 管）。"""
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(seed: int) -> None:
    """``--deterministic`` 的生效设置（必须在模型构造/推理之前调用）。

    * ``torch.use_deterministic_algorithms(True)``：ATen 算子级确定性；
    * 关闭 TF32 matmul（否则 3090 上的 fp32 matmul 会走 TF32，既非确定也损精度）；
    * 用 ``seed`` 播种 CPU / CUDA / Python RNG。

    cuBLAS 的 ``CUBLAS_WORKSPACE_CONFIG`` 已在模块 import 时（torch 之前）按 argv 设好。
    """
    import random

    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    _seed_torch_rngs(seed)
    random.seed(seed)


def seed_predictor_stream(seed: int) -> None:
    """``--predictor-seed fresh``：只播种一次全局 RNG，让每次推理取新噪声。"""
    _seed_torch_rngs(seed)


class ReseedGraphPredictor:
    """让 ``--predictor-seed fixed`` 在确定性模式下真正生效的薄包装。

    ``DecodedCutPredictor`` 直接调用 ``base._predict_graph``，绕过了
    ``FastT2TPredictor.predict`` 里唯一的 per-call 播种，因此构造时给的 ``seed``
    在这条路径上是惰性的（历史默认行为保持原样）。只在 ``--deterministic`` 下启用：
    每次推理前把 CPU+CUDA RNG 重播种到同一 seed，于是同一 ``--seed`` 的整条管线
    逐位可复现（P-5 的最小实验需求）。
    """

    def __init__(self, base, seed: int):
        self.base = base
        self.seed = int(seed)

    def _predict_graph(self, g):
        _seed_torch_rngs(self.seed)
        return self.base._predict_graph(g)


def choose_beta(g, vc_ids, hidden: int, budget_gb: float = 12.0) -> int:
    """按实测每边显存 + 候选子图的实际边数选 β（不跑模型，秒级完成）。

    显存由**边数**主导（实测线性），所以构造候选邻域子图并数它的边数即可估算，不必
    真做一次前向。旧版"翻倍试跑"标定过于保守：它在 coAuthorsCiteseer 上给出 β=16,000，
    而同一配置 β=100,000 实测只用约 2.8 GB 显存，drop 从 12.79% 改善到 2.29%。
    """
    bpe = BYTES_PER_EDGE.get(hidden, 15.5e3)
    best = 500
    for beta in (1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000):
        sample = vc_ids[: max(1, beta // 2)]
        sub, _ = neighborhood_subgraph(g, sample, beta)
        est_gb = sub.m * bpe / 1024 ** 3 + MODEL_RESIDENT_GB
        if est_gb > budget_gb:
            break
        best = beta
        if sub.n < min(beta, g.n):
            break  # 邻域已覆盖全图，再放大没有意义
    return best


def kamis_reference(g, path: Path, time_limit: float, seed: int = 1) -> tuple[int, str]:
    """KaMIS 参照解：已存在则读缓存，否则限时现跑并落盘。"""
    if path.exists():
        with np.load(path) as z:
            return int(z["labels"].sum()), "cached"
    t0 = time.perf_counter()
    labels = kamis_mis(g, time_limit_s=time_limit, seed=seed)
    el = time.perf_counter() - t0
    np.savez_compressed(path, labels=labels.astype(np.int8))
    return int(labels.sum()), f"ran {el:.0f}s"


def solution_health(g, labels: np.ndarray) -> dict:
    """解体检：冲突边数、非极大顶点数、贪心补全后的规模。

    极大性（MIS 必须满足）：任何未入选顶点都必须有邻居入选。递归框架把 V_C 的决策
    交给神经预测器，若预测器在这些顶点上给出的标注不是极大的，最终解就会"欠填"。
    稠密图上 V_C 占比高（coPapersCiteseer 26.7%），欠填会直接吃掉十几个点的质量。
    """
    sel = labels == 1
    src, dst = g.edges[:, 0], g.edges[:, 1]
    conflicts = int((sel[src] & sel[dst]).sum()) if g.m else 0
    neighbor_sel = np.zeros(g.n, dtype=bool)
    if g.m:
        neighbor_sel |= np.bincount(dst[sel[src]], minlength=g.n) > 0
        neighbor_sel |= np.bincount(src[sel[dst]], minlength=g.n) > 0
    nonmax = ~sel & ~neighbor_sel
    n_nonmax = int(nonmax.sum())
    size0 = int(sel.sum())
    repaired = size0
    if n_nonmax:
        adj = coo_matrix((np.ones(g.m, dtype=np.int8), (src, dst)), shape=(g.n, g.n)).tocsr()
        sel = sel.copy()
        neighbor_sel = neighbor_sel.copy()
        added = 0
        for v in np.flatnonzero(nonmax):
            if not sel[v] and not neighbor_sel[v]:
                sel[v] = True
                added += 1
                lo, hi = adj.indptr[v], adj.indptr[v + 1]
                if hi > lo:
                    neighbor_sel[adj.indices[lo:hi]] = True
        repaired = size0 + added
    return {"conflicts": conflicts, "nonmax": n_nonmax, "size": size0, "repaired": repaired}


def run_one(graph_path: Path, args, out: Path) -> None:
    if args.deterministic:
        configure_determinism(args.seed)

    from rgf.predictor.fastt2t import FastT2TPredictor

    g = load_graph_npz(graph_path)
    density = g.m / (g.n * (g.n - 1)) if g.n > 1 else 0.0
    print(f"=== {graph_path.name}: |V|={g.n:,} |E|={g.m:,} density={density:.2e} "
          f"平均度={2 * g.m / g.n:.1f} ===", flush=True)

    ref_path = args.data_dir / f"{graph_path.stem}.label.npz"
    ref, ref_note = kamis_reference(g, ref_path, args.kamis_time)
    print(f"  KaMIS 参照（限时 {args.kamis_time:.0f}s）|S|={ref:,}（{ref_note}）", flush=True)

    predictor_seed = args.seed if args.predictor_seed == "fixed" else None
    print(f"  随机种子 seed={args.seed}（划分器 / 求解器 / RW 共用）；"
          f"预测器种子模式={args.predictor_seed}"
          f"（{'seed=' + str(predictor_seed) if predictor_seed is not None else '不固定'}）；"
          f"deterministic={'on' if args.deterministic else 'off'}", flush=True)
    predictor = FastT2TPredictor(CKPTS[args.hidden], inference_diffusion_steps=1,
                                 hidden_dim=args.hidden, guided=False, rewrite_steps=0,
                                 seed=predictor_seed)
    if args.predictor_seed == "fresh":
        # 预测器自己不重置种子；这里只播种一次，等价于"每次推理取新噪声、同 seed 可复现"。
        seed_predictor_stream(args.seed)
    predictor_seed_enforced = bool(args.deterministic and args.predictor_seed == "fixed")
    if predictor_seed_enforced:
        # 确定性模式：fixed 口径必须真正落到每次推理（见 ReseedGraphPredictor 注释）。
        predictor = ReseedGraphPredictor(predictor, args.seed)
    if args.deterministic and args.predictor_seed == "none":
        print("  注意：--deterministic 与 --predictor-seed none 同用时预测器噪声流未播种，"
              "结果仍不可复现", flush=True)

    beta = args.beta
    if args.partitioner == "metis":
        make_part = lambda: MetisPartitioner(nparts=args.metis_nparts, seed=args.seed)  # noqa: E731
    elif args.partitioner == "random":
        make_part = RandomBalancedPartitioner  # 无参构造
    else:
        make_part = SingleVertexPartitioner  # 无参构造
    # 单顶点划分每次只切一个顶点 ⇒ 递归深度 ~ |V|/α 量级，必须把上限提到 ≥|V|
    max_levels = args.max_levels if args.max_levels > 0 else (
        max(256, g.n + 8) if args.partitioner == "single" else 256)
    cut = make_part().partition(g, np.random.default_rng(args.seed))
    print(f"  第 0 层 {args.partitioner} 分区：|V_C|={cut.vc_ids.size:,}"
          f"（占全图 {cut.vc_ids.size / g.n * 100:.2f}%）；max_levels={max_levels}", flush=True)
    if beta == "auto":
        beta = choose_beta(g, cut.vc_ids, args.hidden, budget_gb=args.mem_gb)
    beta = int(beta)
    per_chunk = max(1, beta // 2)
    print(f"  β = {beta:,}（每次送模型的邻域子图顶点数上限；显存预算 {args.mem_gb:.0f} GB；"
          f"本层需 ceil({cut.vc_ids.size:,}/{per_chunk}) = "
          f"{-(-cut.vc_ids.size // per_chunk)} 次模型调用）", flush=True)

    decoded = DecodedCutPredictor(predictor, "mis", cdsp_sampling=args.cdsp_sampling,
                                  cdsp_rule=args.cdsp_rule)
    recorder = None
    if getattr(args, "dump_decisions", None):
        # 纯测量（默认关）：把每层 CDSP 的 V_C 决策记下来，供误差率分析使用。
        recorder = DecisionRecorder()
    solver = RGFSolver(MaximumIndependentSet(), make_part(), decoded,
                       alpha=args.alpha, beta=beta, seed=args.seed, max_levels=max_levels,
                       decision_recorder=recorder)
    t0 = time.perf_counter()
    labels, stats = solver.solve(g)
    rgf_time = time.perf_counter() - t0
    rgf_size = int(np.count_nonzero(labels == 1))
    rgf_drop = (ref - rgf_size) / ref * 100
    print(f"  RGF : |S|={rgf_size:,}  drop={rgf_drop:.2f}%  用时 {rgf_time / 60:.1f} min "
          f"(递归层数={stats.levels} 精确求解调用={stats.exact_calls} "
          f"神经网络调用={stats.nn_inferences} 各层 |V_C| 之和={stats.cut_vertices:,})",
          flush=True)

    health = solution_health(g, labels)
    print(f"  RGF 体检：冲突边={health['conflicts']} 非极大顶点={health['nonmax']:,} "
          f"贪心补全后 |S|={health['repaired']:,}"
          f"（+{health['repaired'] - health['size']:,}）", flush=True)
    np.savez_compressed(out.with_suffix(".rgf_labels.npz"), labels=labels.astype(np.int8))

    # 逐层决策记录（仅 --dump-decisions 时写；元数据里带上参照解与最终 drop，
    # 分析脚本可以只用这一个 npz + 参照标签算 ρ，不必再猜运行配置）。
    decisions_path = None
    if recorder is not None:
        decisions_path = Path(args.dump_decisions) / f"{out.stem}.decisions.npz"
        recorder.save(decisions_path, metadata={
            "instance": graph_path.stem, "n": g.n, "m": g.m,
            "alpha": args.alpha, "beta": beta, "hidden": args.hidden,
            "seed": args.seed, "partitioner": args.partitioner,
            "metis_nparts": args.metis_nparts,
            "deterministic": bool(args.deterministic),
            "predictor_seed": args.predictor_seed,
            "predictor_seed_enforced": predictor_seed_enforced,
            "data_dir": str(args.data_dir), "graph_path": str(graph_path),
            "kamis_ref": ref, "kamis_note": ref_note,
            "rgf_size": rgf_size, "rgf_drop": rgf_drop, "rgf_time_s": rgf_time,
            "rgf_levels": stats.levels, "rgf_exact_calls": stats.exact_calls,
            "rgf_nn_inferences": stats.nn_inferences, "cut_vertices": stats.cut_vertices,
            "per_level": stats.per_level,
            "rgf_conflicts": health["conflicts"], "rgf_nonmax": health["nonmax"],
            "rgf_size_repaired": health["repaired"],
        })
        print(f"  逐层决策记录：{recorder.n_records:,} 条 → {decisions_path}", flush=True)

    rw_size = rw_iters = None
    rw_drop = None
    if not args.skip_rw:
        budget = args.rw_budget if args.rw_budget > 0 else rgf_time
        print(f"  RW  : 与 RGF 同墙钟 {budget / 60:.1f} min ...", flush=True)
        rw_labels, rw_iters = random_walk_mis(g, budget, rng=np.random.default_rng(args.seed))
        rw_size = int(np.count_nonzero(rw_labels == 1))
        rw_drop = (ref - rw_size) / ref * 100
        rw_bad = int((rw_labels[g.edges[:, 0]] + rw_labels[g.edges[:, 1]] > 1).sum()) if g.m else 0
        print(f"  RW  : |S|={rw_size:,}  drop={rw_drop:.2f}%  iterations={rw_iters:,}  "
              f"冲突边={rw_bad}", flush=True)
        assert rw_bad == 0, "RW 解不合法（有相邻顶点同时入选）"

    payload = {
        "instance": graph_path.stem, "n": g.n, "m": g.m, "density": density,
        "alpha": args.alpha, "beta": beta, "hidden": args.hidden,
        "mem_gb": args.mem_gb, "cdsp_sampling": args.cdsp_sampling,
        "cdsp_rule": args.cdsp_rule, "partitioner": args.partitioner,
        "metis_nparts": args.metis_nparts,
        "seed": args.seed,
        "max_levels": max_levels,
        "vc_level0": int(cut.vc_ids.size), "kamis_time_limit_s": args.kamis_time,
        "kamis_ref": ref, "kamis_note": ref_note,
        "rgf_size": rgf_size, "rgf_drop": rgf_drop, "rgf_time_s": rgf_time,
        "rgf_levels": stats.levels, "rgf_exact_calls": stats.exact_calls,
        "rgf_nn_inferences": stats.nn_inferences,
        # 理论分析的核心变量：各层 |V_C| 之和 ≈ |V_NN|（神经预测器实际处理的顶点数）。
        "cut_vertices": stats.cut_vertices,
        "per_level": stats.per_level,
        "deterministic": bool(args.deterministic),
        "predictor_seed": args.predictor_seed,
        "predictor_seed_value": predictor_seed,
        # 仅当 deterministic+fixed 时为 True：此时预测器噪声流被固定，同 seed 逐位可复现。
        "predictor_seed_enforced": predictor_seed_enforced,
        "rgf_conflicts": health["conflicts"], "rgf_nonmax": health["nonmax"],
        "rgf_size_repaired": health["repaired"],
        "rw_size": rw_size, "rw_drop": rw_drop, "rw_iterations": rw_iters,
    }
    if decisions_path is not None:  # 仅诊断打开时追加；默认 JSON 与历史逐字节一致
        payload["decisions_npz"] = str(decisions_path)
        payload["decisions_records"] = int(recorder.n_records)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  结果已写入 {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="", help="单个 dimacs10 npz 名")
    ap.add_argument("--instances", default="", help="逗号分隔的多个实例名（或 all）")
    ap.add_argument("--beta", default="auto", help="整数或 auto（按显存估算，D-9）")
    ap.add_argument("--mem-gb", type=float, default=12.0,
                    help="β=auto 时的显存预算（GB）；GPU1 独占时可上调以提高 β")
    ap.add_argument("--partitioner", default="metis", choices=["metis", "random", "single"],
                    help="第 0 层与递归用的划分器（论文 Table 2 的消融；single=单顶点划分）")
    ap.add_argument("--metis-nparts", type=int, default=2,
                    help="METIS 每层分区数 k（k-way 消融：默认 2 与历史口径一致；k=3 用于分区数实验）")
    ap.add_argument("--max-levels", type=int, default=0,
                    help="递归层数上限；0=自动（single 划分取 max(256, n+8)，其余 256）")
    ap.add_argument("--cdsp-sampling", type=int, default=1,
                    help="CDSP 决策的独立推理次数（D-14：×4 + best 能提升 MIS）")
    ap.add_argument("--cdsp-rule", default="best", choices=["best", "vote"])
    ap.add_argument("--seed", type=int, default=0,
                    help="随机种子，同时用于 METIS 划分 / 求解器 / 预测器 / RW（默认 0，"
                         "与历史结果逐位一致）；非 0 时结果文件名加 .seedN 后缀避免互相覆盖")
    ap.add_argument("--deterministic", action="store_true",
                    help="确定性模式（默认关）：torch.use_deterministic_algorithms(True)、关闭 "
                         "TF32 matmul、按 --seed 播种 RNG，并在 import torch 前设置 "
                         "CUBLAS_WORKSPACE_CONFIG=:4096:8；--predictor-seed fixed 时还会在每次"
                         "推理前重播种（CPU+CUDA）使其真正固定。结果 JSON 记录 deterministic=true")
    ap.add_argument("--predictor-seed", choices=["fixed", "fresh", "none"], default="fixed",
                    help="FastT2TPredictor 的种子口径（默认 fixed，保持历史数字）："
                         "fixed=构造时注入 --seed（历史口径；注意 DecodedCutPredictor 直接调 "
                         "_predict_graph，该 seed 在默认路径上不生效，只有配合 --deterministic 才被"
                         "强制到每次推理）；fresh=求解前用 --seed 播种全局 RNG，每次推理取新噪声"
                         "（同 seed 可复现）；none=完全不播种（沿用进程全局 RNG，不可复现，仅作对照）")
    ap.add_argument("--alpha", type=int, default=500)
    ap.add_argument("--hidden", type=int, default=256, choices=[128, 256])
    ap.add_argument("--kamis-time", type=float, default=1800.0)
    ap.add_argument("--rw-budget", type=float, default=0.0, help="0 = 与 RGF 同墙钟")
    ap.add_argument("--skip-rw", action="store_true")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的结果 JSON")
    ap.add_argument("--data-dir", type=Path, default=DIMACS,
                    help="实例 npz 与 <stem>.label.npz 所在目录（默认 /mnt/e/data/rgf/dimacs10）")
    ap.add_argument("--out-dir", type=Path, default=DATA / "results" / "large")
    ap.add_argument("--dump-decisions", type=Path, default=None, metavar="DIR",
                    help="（诊断，默认关）把逐层 CDSP 决策（层号/任务序号/V_C 全局 id/"
                         "repair 前后的 S_C/子图内度数）写成 DIR/<实例>.decisions.npz，"
                         "供 scripts/analyze_predictor_errors.py 计算逐层 FP/FN 错误率 ρ；"
                         "关闭时不写任何额外文件，结果 JSON 与历史逐字节一致")
    args = ap.parse_args()

    names = select_instances(args)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        stem = Path(name).stem
        stem_part = stem if args.seed == 0 else f"{stem}.seed{args.seed}"
        out = out_dir / f"{stem_part}.json"
        if out.exists() and not args.force:
            print(f"=== {name}: 已有结果，跳过 ===", flush=True)
            continue
        run_one(args.data_dir / name, args, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
