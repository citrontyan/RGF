"""Table-1 基线行：DIFUSCO / FastT2T 在 RB-[200-300] 与 ER-[700-800] 测试集上的多种子评测。

口径（`docs/decisions.md` D-8 / D-11）：

* DIFUSCO：Ts=100（Fast T2T 论文口径），贪心（``--sampling 1``）或采样×4（``--sampling 4``，逐图取最优）；
* FastT2T：RB 用 (Ts,Tg)=(1,1)、ratio 0.3、c1=c2=2、hidden 256；ER 用 (5,5)、ratio 0.2、c1=c2=2、hidden 128；
  两者均为官方 ``--rewrite --guided`` 口径；
* DIFUSCO 官方只发布 ER 与 SATLIB 权重，RB 行按 Fast T2T 论文的做法用 ER 权重迁移评测；
* 解对初始噪声高度敏感（D-11：seed 间 σ≈3.4）→ **必须多种子平均**（``--runs``，默认 5），
  单次采样数字不可与文献比较。参照解取同目录下的 ``*.label.npz``（KaMIS），drop = (ref - ours)/ref。

    set CUDA_VISIBLE_DEVICES=1
    .venv\\Scripts\\python.exe scripts\\eval_predictor_baselines.py --predictor difusco --dataset er --runs 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgf.datasets.io import load_graph_npz  # noqa: E402
from rgf.eval.metrics import drop_percent  # noqa: E402

# drop 的口径（D-6）：永远表示"比参照解差多少"，正的即更差；MIS 最大化、DS/VC 最小化
MAXIMIZE = {"mis": True, "vc": False, "ds": False}

# 大整数种子基（--seed-mode large）：避开 torch.manual_seed 小整数的 MT19937 弱播种
SEED_BASE = 987_654_321_987_654_321


def derive_seed(run: int, graph_idx: int, sample: int) -> int:
    """由 (run, 图序, 采样号) 派生 63 位大整数种子 → 每图噪声独立且良播种。"""
    digest = hashlib.blake2b(f"{run}:{graph_idx}:{sample}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**63 - 1)

DEFAULT_DATA_ROOT = Path(r"E:\data\rgf")
DATASETS = {"rb": "rb/rb200_300/test", "er": "er/er700_800/test"}

CALIBER = {
    ("difusco", "er"): dict(ckpt="checkpoints/difusco/mis_er_gaussian.ckpt", steps=100),
    ("difusco", "rb"): dict(ckpt="checkpoints/difusco/mis_er_gaussian.ckpt", steps=100),
    ("fastt2t", "rb"): dict(
        ckpt="checkpoints/fastt2t/mis_rb.ckpt", steps=1, rewrite_steps=1,
        rewrite_ratio=0.3, c1=2.0, c2=2.0, hidden_dim=256,
    ),
    ("fastt2t", "er"): dict(
        ckpt="checkpoints/fastt2t/mis_er.ckpt", steps=5, rewrite_steps=5,
        rewrite_ratio=0.2, c1=2.0, c2=2.0, hidden_dim=128,
    ),
}


def greedy_size(g, probs: np.ndarray, problem: str = "mis") -> int:
    """问题相关贪心解码后的解规模（预测器输出是 MIS 语义，VC/DS 由它导出）。

    * ``mis``：按概率降序的贪心极大独立集（等价于官方 ``mis_decode_np``）；
    * ``vc``：最小顶点覆盖 = 最大独立集的补（D-6）；
    * ``ds``：按概率降序的贪心支配集（先取概率高的顶点，跳过已被支配的）。
    """
    chosen = np.zeros(g.n, dtype=np.int8)
    blocked = np.zeros(g.n, dtype=bool)
    for v in np.argsort(-probs, kind="stable"):
        if blocked[v]:
            continue
        chosen[v] = 1
        blocked[v] = True
        blocked[g.neighbors(v)] = True
    if problem == "mis":
        return int(chosen.sum())
    if problem == "vc":
        return int(g.n - chosen.sum())
    return int(chosen.sum())  # ds：贪心规则与 MIS 相同（都在"未被覆盖"的顶点里取）


def greedy_solution(g, probs: np.ndarray) -> int:
    """MIS 口径的兼容包装（历史基线与结果字段不变）。"""
    return greedy_size(g, probs, "mis")


def load_reference(path: Path, problem: str, n: int) -> tuple[float, str, bool]:
    """读取参照解：(size, method, optimal)。文件名约定见 ``rgf.datasets.io`` / D-6。"""
    if problem == "ds":
        with np.load(path.with_suffix(".dsref.npz")) as data:
            return float(data["objective"]), str(data["method"]), bool(data["optimal"])
    with np.load(path.with_suffix(".label.npz")) as data:
        mis_size = float(data["labels"].sum())
    if problem == "vc":
        return float(n - mis_size), "KaMIS(complement)", True
    return mis_size, "KaMIS", True


def raw_probabilities(predictor, g, torch):
    """取预测器对整图的**原始概率**（``predict()`` 按协议已二值化为 CDSP 决策，
    直接基线需要概率本身来做贪心解码）。"""
    if predictor.seed is not None:
        torch.manual_seed(predictor.seed)
    return predictor._predict_graph(g)


def build_predictor(args, spec):
    if args.ts:
        spec = {**spec, "steps": args.ts}
    if args.tg is not None:
        spec = {**spec, "rewrite_steps": args.tg}
    if args.hidden_dim:
        spec = {**spec, "hidden_dim": args.hidden_dim}
    ckpt = Path(spec["ckpt"])
    if not ckpt.is_absolute():
        ckpt = args.data_root / ckpt
    if args.predictor == "difusco":
        from rgf.predictor.difusco import DifuscoPredictor

        return DifuscoPredictor(
            ckpt,
            inference_diffusion_steps=args.steps or spec["steps"],
            seed=0,
        )

    from rgf.predictor.fastt2t import FastT2TPredictor

    return FastT2TPredictor(
        ckpt,
        inference_diffusion_steps=args.steps or spec["steps"],
        hidden_dim=spec["hidden_dim"],
        guided=True,
        rewrite_steps=spec["rewrite_steps"],
        rewrite_ratio=spec["rewrite_ratio"],
        c1=spec["c1"],
        c2=spec["c2"],
        seed=0,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--predictor", choices=["difusco", "fastt2t"], required=True)
    ap.add_argument("--dataset", choices=["rb", "er"], required=True)
    ap.add_argument("--runs", type=int, default=5, help="随机种子重复次数（论文口径 5）")
    ap.add_argument("--sampling", type=int, default=1, help="每次运行内的独立采样数，逐图取最优（1=贪心，4=采样×4）")
    ap.add_argument("--steps", type=int, default=0, help="覆盖口径里的 Ts")
    ap.add_argument("--ts", type=int, default=0, help="等价于 --steps（与 run_rgf.py 命名一致）")
    ap.add_argument("--tg", type=int, default=None, help="覆盖 FastT2T 引导重写步数 Tg")
    ap.add_argument("--hidden-dim", type=int, default=0, help="覆盖预测器隐藏维度")
    ap.add_argument("--ckpt", default=None,
                    help="覆盖预测器权重路径（DS/VC 重训权重用；相对路径按 --data-root 解析）")
    ap.add_argument("--problem", choices=["mis", "vc", "ds"], default="mis",
                    help="解码与参照解口径：mis=贪心极大独立集/KaMIS；vc=取补/KaMIS 补；ds=贪心支配集/DS 参照解")
    ap.add_argument("--tag", default="", help="结果文件名后缀（区分不同权重/口径）")
    ap.add_argument("--limit", type=int, default=0, help="只评测前 N 张图（0=全部）")
    ap.add_argument("--out", type=Path, default=None, help="结果 JSON 路径")
    ap.add_argument(
        "--seed-mode",
        choices=["sequential", "large", "pergraph"],
        default="pergraph",
        help=(
            "噪声播种口径：sequential=历史行为，每图重播种且用小整数 r*K+j（全部图共用同一段噪声模式，"
            "与测试集顺序无关但有 MT19937 小整数弱播种之嫌）；large=同样共用模式但改用大整数种子（判别用）；"
            "pergraph=每张图用 (run,图序,采样号) 派生的独立大整数种子（每图噪声独立、与顺序无关、良播种）"
        ),
    )
    args = ap.parse_args()

    spec = dict(CALIBER[(args.predictor, args.dataset)])
    # 覆盖项就地生效，保证打印口径与结果文件名都反映真实配置
    if args.ts or args.steps:
        spec["steps"] = args.ts or args.steps
    if args.tg is not None and "rewrite_steps" in spec:
        spec["rewrite_steps"] = args.tg
    if args.hidden_dim and "hidden_dim" in spec:
        spec["hidden_dim"] = args.hidden_dim
    if args.ckpt:
        spec["ckpt"] = args.ckpt
    test_dir = args.data_root / DATASETS[args.dataset]
    # 排除派生文件：*.label.npz（KaMIS 参照）、*.dsref.npz（DS 参照解）、
    # *.opt.npz（精确最优解，D-27）都不是图
    files = [f for f in sorted(test_dir.glob("*.npz"))
             if not f.name.endswith((".label.npz", ".dsref.npz", ".opt.npz"))]
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no graphs in {test_dir}", file=sys.stderr)
        return 2

    graphs, refs, ref_methods = [], [], set()
    for f in files:
        graph = load_graph_npz(f)
        try:
            size, method, _optimal = load_reference(f, args.problem, graph.n)
        except FileNotFoundError:
            continue
        graphs.append(graph)
        refs.append(size)
        ref_methods.add(method)
    refs_a = np.array(refs, dtype=np.float64)

    print(
        f"{args.predictor} on {args.dataset}-test [{args.problem.upper()}]: {len(graphs)} graphs, "
        f"Ts={args.steps or spec['steps']}, sampling={args.sampling}, runs={args.runs}, "
        f"参照({sorted(ref_methods)}) mean {refs_a.mean():.2f}",
        flush=True,
    )

    predictor = build_predictor(args, spec)
    import torch

    run_means, run_drops, run_times = [], [], []
    for r in range(args.runs):
        t0 = time.perf_counter()
        sizes = []
        for idx, g in enumerate(graphs):
            best = None  # 按问题方向取最优：MIS 取最大、DS/VC 取最小（D-18 教训）
            for j in range(args.sampling):
                if args.seed_mode == "sequential":
                    predictor.seed = r * args.sampling + j
                elif args.seed_mode == "large":
                    predictor.seed = SEED_BASE + r * args.sampling + j
                else:
                    predictor.seed = derive_seed(r, idx, j)
                probs = raw_probabilities(predictor, g, torch)
                size_j = greedy_size(g, probs, args.problem)
                if best is None or (size_j > best if MAXIMIZE[args.problem] else size_j < best):
                    best = size_j
            sizes.append(int(best))
            if (idx + 1) % max(1, len(graphs) // 10) == 0:
                done = idx + 1
                el = time.perf_counter() - t0
                eta = el / done * (len(graphs) - done) / 60
                print(f"    run {r + 1}/{args.runs}: {done}/{len(graphs)} 已用 {el / 60:.1f} min, 预计还需 {eta:.1f} min",
                      flush=True)
        sizes_a = np.array(sizes, dtype=np.float64)
        run_means.append(float(sizes_a.mean()))
        # 口径与历史结果（D-14 基线行）一致：用"解规模均值之比"计算 drop，只把符号按
        # 问题方向校正（MIS 最大化 / DS、VC 最小化），避免 per-graph 平均带来的口径漂移。
        ref_mean = float(refs_a.mean())
        run_drops.append(float(drop_percent(float(sizes_a.mean()), ref_mean, MAXIMIZE[args.problem])))
        run_times.append(time.perf_counter() - t0)
        print(
            f"  run {r + 1}: mean |S| {sizes_a.mean():.2f}  drop {run_drops[-1]:.2f}%  "
            f"({run_times[-1] / 60:.2f} min)",
            flush=True,
        )

    means = np.array(run_means)
    drops = np.array(run_drops)
    result = {
        "predictor": args.predictor,
        "dataset": args.dataset,
        "problem": args.problem,
        "graphs": len(graphs),
        "caliber": {k: v for k, v in spec.items() if k != "ckpt"},
        "ckpt": str(spec["ckpt"]),
        "ts": args.steps or spec["steps"],
        "sampling": args.sampling,
        "runs": args.runs,
        "seed_mode": args.seed_mode,
        "drop_aggregation": "ratio_of_means",
        "ref_methods": sorted(ref_methods),
        "ref_mean": float(refs_a.mean()),
        "kamis_ref_mean": float(refs_a.mean()),  # 兼容旧结果字段名（mis 口径下即 KaMIS）
        "run_means": run_means,
        "run_drops": run_drops,
        "mean_size": float(means.mean()),
        "std_size": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
        "mean_drop": float(drops.mean()),
        "std_drop": float(drops.std(ddof=1)) if len(drops) > 1 else 0.0,
        "minutes_per_run": [t / 60 for t in run_times],
    }
    print(
        f"=== {args.predictor} {args.dataset} [{args.problem.upper()}] (Ts={result['ts']}, "
        f"sampling={args.sampling}, runs={args.runs}, seed_mode={args.seed_mode}) ===\n"
        f"  |S| = {result['mean_size']:.2f} ± {result['std_size']:.2f}    "
        f"drop = {result['mean_drop']:.2f}% ± {result['std_drop']:.2f}%    "
        f"参照 = {result['ref_mean']:.2f}",
        flush=True,
    )

    problem_infix = "" if args.problem == "mis" else f"{args.problem}-"
    tag = f"-{args.tag}" if args.tag else ""
    out = args.out or (
        args.data_root / "results" / "baselines"
        / f"{args.predictor}-{problem_infix}{args.dataset}-ts{result['ts']}-s{args.sampling}-{args.seed_mode}{tag}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  结果已写入 {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
