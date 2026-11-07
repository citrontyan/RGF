"""P2 主表：RGF 在 RB-[200-300] / ER-[700-800] 测试集上的论文口径评测。

口径（`docs/decisions.md` D-8 / D-11 / D-12）：

* α=50（合成图）；``--beta`` 为 Alg. 2 的邻域半径（默认 None = 完整预测）；
* **噪声口径**：每张图独立（由 (run, 图序) 派生的大整数种子播种一次），图内的多次预测调用
  共享同一条推进的随机流 —— 对齐官方实现，且避免"同一噪声向量套到所有图"的系统偏差（D-11 因素一）；
* **5 次独立运行**取平均并报标准差（D-11 口径要求）；
* 参照解：MIS / VC 用预标注的 KaMIS 标签（VC = 补集，D-6）；DS 用 ``rgf.baselines.ds_reference``
  （MILP 证优 / incumbent / 贪心+冗余删除，D-6），结果缓存到 ``<data-root>/results/ds_reference/``；
* **必须在忠实 WSL 环境里跑**（Windows venv 的 ``torch_sparse`` 桩不等价，D-11 因素二）：

    wsl -e bash scripts/run_wsl.sh scripts/run_rgf.py --problem mis --partitioner metis \\
        --predictor fastt2t --dataset rb --runs 5
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

from rgf.baselines.ds_reference import ds_reference  # noqa: E402
from rgf.datasets.io import load_graph_npz  # noqa: E402
from rgf.eval.metrics import drop_percent, solution_size  # noqa: E402
from rgf.framework import RGFSolver  # noqa: E402
from rgf.partition import MetisPartitioner, RandomBalancedPartitioner, SingleVertexPartitioner  # noqa: E402
from rgf.problems import DominatingSet, MaximumIndependentSet, VertexCover  # noqa: E402

DEFAULT_DATA_ROOT = Path(r"E:\data\rgf")
DATASETS = {"rb": "rb/rb200_300/test", "er": "er/er700_800/test"}
PROBLEMS = {"mis": MaximumIndependentSet, "vc": VertexCover, "ds": DominatingSet}

# 预测器口径（D-8）：DIFUSCO Ts=100；FastT2T RB (1,1)/ratio 0.3/hidden 256，ER (5,5)/ratio 0.2/hidden 128
PREDICTORS = {
    ("fastt2t", "rb"): dict(ckpt="checkpoints/fastt2t/mis_rb.ckpt", kind="fastt2t",
                            steps=1, rewrite_steps=1, rewrite_ratio=0.3, hidden_dim=256),
    ("fastt2t", "er"): dict(ckpt="checkpoints/fastt2t/mis_er.ckpt", kind="fastt2t",
                            steps=5, rewrite_steps=5, rewrite_ratio=0.2, hidden_dim=128),
    ("difusco", "rb"): dict(ckpt="checkpoints/difusco/mis_er_gaussian.ckpt", kind="difusco", steps=100),
    ("difusco", "er"): dict(ckpt="checkpoints/difusco/mis_er_gaussian.ckpt", kind="difusco", steps=100),
}

# 新增预测器（COExpander / GenSCO，camera-ready 补充基线）：具体权重与模型由
# rgf/predictor/{coexpander,gensco}.py 从 --data-root 下解析。
PREDICTORS.update({
    ("coexpander", "rb"): {"kind": "coexpander"},
    ("coexpander", "er"): {"kind": "coexpander"},
    ("gensco", "rb"): {"kind": "gensco"},
    ("gensco", "er"): {"kind": "gensco"},
})


def derive_seed(run: int, graph_idx: int, sample: int = 0) -> int:
    digest = hashlib.blake2b(f"rgf:{run}:{graph_idx}:{sample}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**63 - 1)


# --------------------------------------------------------------------- readout
def decode_mis(g, probs, ctx=None):
    """按概率降序的贪心极大独立集（与官方 ``mis_decode_np`` 等价）。

    ``ctx=(labels, dominated)``（D-20，覆盖感知解码）：已判 1 的顶点视为已选并
    封锁其邻居，已判 0 的顶点不可再选——递归深层解码只覆盖剩余未支配顶点，
    避免与高层决策重复选点（DS 上无此条件会使解规模随层数膨胀）。
    """
    chosen = np.zeros(g.n, dtype=bool)
    blocked = np.zeros(g.n, dtype=bool)
    labels = None
    if ctx is not None:
        labels, pre_dominated = ctx
        labels = np.asarray(labels)
        chosen |= labels == 1
        blocked |= chosen
        if pre_dominated is not None:
            blocked |= np.asarray(pre_dominated, dtype=bool)
    for v in np.argsort(-probs, kind="stable"):
        if blocked[v] or (labels is not None and labels[v] == 0):
            continue
        chosen[v] = True
        blocked[v] = True
        blocked[g.neighbors(v)] = True
    return chosen


def decode_vc(g, probs, ctx=None):
    """最小顶点覆盖 = 最大独立集的补集（D-6）。ctx 暂不感知（VC 行未重跑）。"""
    return ~decode_mis(g, probs)


def decode_ds(g, probs, ctx=None):
    """按概率降序的贪心支配集：取该顶点当且仅当它仍是未被支配的。

    ``ctx`` 见 :func:`decode_mis`：高层已选顶点的支配先验生效、已判 0 顶点不再
    参选——没有这一条件，每层解码都会重新支配"高层已支配"的顶点，跨层重复选点
    使 RGF-DS 的解规模随递归层数单调膨胀（D-20 实测：α=50/100/200 → 20.6/18.9/16.5）。
    """
    chosen = np.zeros(g.n, dtype=bool)
    dominated = np.zeros(g.n, dtype=bool)
    labels = None
    if ctx is not None:
        labels, pre_dominated = ctx
        labels = np.asarray(labels)
        chosen |= labels == 1
        dominated |= chosen
        for v in np.flatnonzero(chosen):
            dominated[g.neighbors(v)] = True
        if pre_dominated is not None:
            dominated |= np.asarray(pre_dominated, dtype=bool)
    for v in np.argsort(-probs, kind="stable"):
        if dominated[v] or (labels is not None and labels[v] == 0):
            continue
        chosen[v] = True
        dominated[v] = True
        dominated[g.neighbors(v)] = True
    return chosen


DECODERS = {"mis": decode_mis, "vc": decode_vc, "ds": decode_ds}


class DecodedCutPredictor:
    """把整图概率解码成一致的部分解，再取出 ``V_C`` 处的赋值作为 CDSP 决策。

    ``threshold`` 读数（直接对概率取 0.5）在 MIS 上会给出不独立的 ``S_C``，被
    ``repair_cdsp`` 用度数贪心重写后模型信息基本丢失（实测 RGF 掉到度启发式水平）；
    与直接基线一致的读法是"先解码、再取 $V_C$"（本类）。

    ``cdsp_sampling`` > 1 时，对同一决策做 K 次独立推理（噪声不同），再用 ``cdsp_rule``
    合成一个决策——这是把论文的 sampling decoding 用在 CDSP 决策上（而非只用在整解上）：

    * ``best``：取"在本次任务图上解码后解最大"的那次候选的 $V_C$ 决策；
    * ``vote``：对 $V_C$ 逐顶点多数投票（K/2 以上判 1），再交给 ``repair_cdsp`` 修正。
    """

    name = "decoded"

    def __init__(self, base, problem: str, readout: str = "decode",
                 cdsp_sampling: int = 1, cdsp_rule: str = "best"):
        self.base = base
        self.problem = problem
        self.readout = readout
        self.cdsp_sampling = max(1, int(cdsp_sampling))
        self.cdsp_rule = cdsp_rule
        self.seed = getattr(base, "seed", None)

    def _decode(self, g, probs, ctx=None):
        if self.readout == "decode":
            return DECODERS[self.problem](g, probs, ctx)
        return probs > 0.5

    def predict(self, batch: list[tuple]) -> list[np.ndarray]:
        out = []
        for item in batch:
            g, vc_ids = item[0], item[1]
            ctx = item[2] if len(item) > 2 else None  # (labels, dominated)，D-20 覆盖感知
            idx = None if vc_ids is None else np.asarray(vc_ids, dtype=np.int64)
            if self.cdsp_sampling == 1:
                labels = self._decode(g, self.base._predict_graph(g), ctx)
                out.append(np.asarray(labels if idx is None else labels[idx], dtype=bool))
                continue

            candidates = []
            for _ in range(self.cdsp_sampling):
                probs = self.base._predict_graph(g)
                labels = self._decode(g, probs, ctx)
                candidates.append(labels)
            maximize = self.problem == "mis"  # 取优方向按问题（D-18：DS/VC 取最小）
            if self.cdsp_rule == "vote" and idx is not None:
                votes = np.stack([c[idx] for c in candidates]).sum(axis=0)
                merged = votes * 2 > len(candidates)
            else:  # best：按整图解码解的"优"挑一次候选
                best = max(candidates, key=lambda lab: (int(lab.sum()) if maximize else -int(lab.sum())))
                merged = best if idx is None else best[idx]
            out.append(np.asarray(merged, dtype=bool))
        return out


def build_predictor(args, spec):
    if args.predictor == "degree":
        from rgf.predictor.base import GreedyDegreePredictor

        return GreedyDegreePredictor()

    if args.predictor == "coexpander":
        from rgf.predictor.coexpander import CoExpanderPredictor

        return CoExpanderPredictor(args.data_root, args.dataset)

    if args.predictor == "gensco":
        from rgf.predictor.gensco import GenSCOPredictor

        return GenSCOPredictor(args.data_root, args.dataset)

    spec = dict(spec or {})
    # 命令行覆盖（用于口径筛选：Ts/Tg/ratio/hidden 都是工程参数）
    if args.ts:
        spec["steps"] = args.ts
    if args.tg is not None:
        spec["rewrite_steps"] = args.tg
    if args.ratio is not None:
        spec["rewrite_ratio"] = args.ratio
    if args.hidden_dim:
        spec["hidden_dim"] = args.hidden_dim

    # --ckpt 覆盖（DS/VC 需要在对应问题的数据上重训预测器，权重不在 PREDICTORS 表里）
    ckpt = Path(spec["ckpt"])
    if args.ckpt:
        ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = args.data_root / ckpt

    if spec["kind"] == "difusco":
        from rgf.predictor.difusco import DifuscoPredictor

        return DifuscoPredictor(ckpt, inference_diffusion_steps=spec["steps"], seed=None)

    from rgf.predictor.fastt2t import FastT2TPredictor

    return FastT2TPredictor(
        ckpt,
        inference_diffusion_steps=spec["steps"],
        hidden_dim=spec["hidden_dim"],
        guided=True,
        rewrite_steps=spec["rewrite_steps"],
        rewrite_ratio=spec["rewrite_ratio"],
        c1=2.0,
        c2=2.0,
        seed=None,
    )


def build_partitioner(name: str, nparts: int = 2):
    if name == "metis":
        return MetisPartitioner(nparts=nparts, seed=0)
    if name == "random":
        return RandomBalancedPartitioner()
    return SingleVertexPartitioner()


def load_references(args, clips):
    """返回每个图的参照解（MIS 为标签，VC/DS 为对应最优/参考目标值）。"""
    refs = []
    ds_cache_path = args.data_root / "results" / "ds_reference" / f"{args.dataset}.json"
    ds_cache = {}
    if args.problem == "ds" and ds_cache_path.exists():
        ds_cache = json.loads(ds_cache_path.read_text(encoding="utf-8"))

    for clip in clips:
        labels_path = clip.with_suffix(".label.npz")
        with np.load(labels_path) as data:
            mis_size = int(data["labels"].sum())
        if args.problem == "mis":
            refs.append({"size": mis_size, "method": "kamis", "optimal": False})
        elif args.problem == "vc":
            g = load_graph_npz(clip)
            refs.append({"size": g.n - mis_size, "method": "kamis-complement", "optimal": False})
        else:
            # DS 参照解优先读与评测脚本同一份规范口径（D-6/D-18：MILP + 冗余删除 +
            # 与逆向贪心取更优）；旧缓存 JSON（results/ds_reference/*.json）是早期
            # 60 s 现算口径，仅在缺 .dsref.npz 时兜底，避免两个脚本参照不一致。
            ref_path = clip.with_suffix(".dsref.npz")
            if ref_path.exists():
                with np.load(ref_path) as data:
                    refs.append({"size": int(data["objective"]), "method": str(data["method"]),
                                 "optimal": bool(data["optimal"])})
                continue
            key = clip.stem
            if key in ds_cache:
                refs.append(ds_cache[key])
            else:
                ref = ds_reference(load_graph_npz(clip), time_limit_s=args.ds_ref_time)
                entry = {"size": int(ref.objective), "method": ref.method,
                         "optimal": bool(getattr(ref, "optimal", False))}
                ds_cache[key] = entry
                refs.append(entry)
                ds_cache_path.parent.mkdir(parents=True, exist_ok=True)
                ds_cache_path.write_text(json.dumps(ds_cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    ap.add_argument("--partitioner", choices=["metis", "random", "single"], default="metis")
    ap.add_argument("--predictor", choices=["difusco", "fastt2t", "degree", "coexpander", "gensco"], required=True)
    ap.add_argument("--alpha", type=int, default=50)
    ap.add_argument("--beta", type=int, default=None)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--sampling", type=int, default=1,
                    help="每图独立 RGF 求解次数，取最优（对应论文的 sampling decoding ×K）")
    ap.add_argument("--metis-nparts", type=int, default=2, help="METIS 每层分区数")
    ap.add_argument("--max-levels", type=int, default=256,
                    help="递归层数上限（RGFSolver 默认 256；single 划分等深递归可上调）")
    ap.add_argument("--ts", type=int, default=0, help="覆盖预测器推理步数 Ts")
    ap.add_argument("--tg", type=int, default=None, help="覆盖 FastT2T 引导重写步数 Tg")
    ap.add_argument("--ratio", type=float, default=None, help="覆盖 FastT2T rewrite_ratio")
    ap.add_argument("--hidden-dim", type=int, default=0, help="覆盖预测器隐藏维度")
    ap.add_argument("--ckpt", default=None,
                    help="覆盖预测器权重路径（DS/VC 重训权重用；相对路径按 --data-root 解析）")
    ap.add_argument("--tag", default="", help="结果文件名后缀（区分扫描实验）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张图（0=全部）")
    ap.add_argument("--ds-ref-time", type=float, default=60.0)
    ap.add_argument("--readout", choices=["decode", "threshold"], default="decode",
                    help="CDSP 读数：decode=先按概率做问题相关贪心解码再取 V_C（默认，D-13）；threshold=直接对概率取 0.5")
    ap.add_argument("--decode-scope", choices=["full", "component"], default="full",
                    help="预测器/解码看到的子实例：full=含切边（论文原样）；component=去切边（分解一致解码，D-20，DS 用）")
    ap.add_argument("--sub-overlap", choices=["auto", "on", "off"], default="auto",
                    help="叶子精确求解的外圈重叠（D-21）：auto=DS 自动开启；覆盖型问题需要外圈提供跨侧支配候选")
    ap.add_argument("--cdsp-sampling", type=int, default=1,
                    help="每个 CDSP 决策的独立推理次数（>1 时按 --cdsp-rule 合成决策）")
    ap.add_argument("--cdsp-rule", choices=["best", "vote"], default="best",
                    help="CDSP 多推理的合成规则：best=取解码解最大的候选；vote=对 V_C 逐顶点多数投票")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    spec = PREDICTORS.get((args.predictor, args.dataset))
    if args.predictor != "degree" and spec is None:
        print(f"no caliber for {args.predictor}/{args.dataset}", file=sys.stderr)
        return 2
    if args.ckpt and spec is not None:
        spec = {**spec, "ckpt": args.ckpt}  # 口径与结果 JSON 都反映真实权重

    test_dir = args.data_root / DATASETS[args.dataset]
    # 排除派生文件：*.label.npz（KaMIS 参照）、*.dsref.npz（DS 参照解）、
    # *.opt.npz（精确最优解，D-27）都不是图
    clips = [f for f in sorted(test_dir.glob("*.npz"))
             if not f.name.endswith((".label.npz", ".dsref.npz", ".opt.npz"))]
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        print(f"no graphs in {test_dir}", file=sys.stderr)
        return 2
    graphs = [load_graph_npz(c) for c in clips]
    refs = load_references(args, clips)

    problem = PROBLEMS[args.problem]()
    print(
        f"{args.problem.upper()} | {args.dataset} | partitioner={args.partitioner} | "
        f"predictor={args.predictor} | α={args.alpha} | β={args.beta} | runs={args.runs} | {len(graphs)} graphs",
        flush=True,
    )
    print(f"  参照解均值: {np.mean([r['size'] for r in refs]):.2f} "
          f"(methods: {sorted({r['method'] for r in refs})})", flush=True)

    predictor = build_predictor(args, spec)
    if args.predictor != "degree":
        predictor = DecodedCutPredictor(predictor, args.problem, args.readout,
                                        cdsp_sampling=args.cdsp_sampling, cdsp_rule=args.cdsp_rule)
    import torch

    run_means, run_drops, run_times = [], [], []
    first_run_detail = []
    for run in range(args.runs):
        t0 = time.perf_counter()
        sizes, drops, times, detail = [], [], [], []
        for idx, g in enumerate(graphs):
            # 每图独立播种；图内多次预测共享推进的随机流（D-11 因素一）。
            # --sampling>1 时做 K 次独立 RGF 求解并取最优（论文的 sampling decoding ×K）。
            best_size, best_feasible, best_stats, t_graph = None, False, None, 0.0
            for s in range(args.sampling):
                torch.manual_seed(derive_seed(run, idx, s))
                solver = RGFSolver(problem, build_partitioner(args.partitioner, args.metis_nparts), predictor,
                                   alpha=args.alpha, beta=args.beta, seed=run, decode_scope=args.decode_scope,
                                   max_levels=args.max_levels,
                                   overlap=(args.sub_overlap == "on"
                                            or (args.sub_overlap == "auto" and args.problem == "ds")))
                tg = time.perf_counter()
                labels, stats = solver.solve(g)
                t_graph += time.perf_counter() - tg
                size_s = solution_size(labels)
                better = best_size is None or (size_s > best_size if problem.maximize else size_s < best_size)
                if better:
                    best_size = size_s
                    best_feasible = bool(problem.feasible(g, labels))
                    best_stats = stats
            size, elapsed, feasible, stats = best_size, t_graph, best_feasible, best_stats
            drop = drop_percent(size, refs[idx]["size"], problem.maximize)
            sizes.append(size)
            drops.append(drop)
            times.append(elapsed)
            if run == 0:
                detail.append({"graph": clips[idx].stem, "n": g.n, "m": g.m, "size": size,
                               "ref": refs[idx]["size"], "drop": drop, "time_s": round(elapsed, 2),
                               "feasible": bool(feasible), "levels": stats.levels,
                               "nn_inferences": stats.nn_inferences, "exact_calls": stats.exact_calls})
            done = idx + 1
            if done % max(1, len(graphs) // 10) == 0:
                el = time.perf_counter() - t0
                print(f"    run {run + 1}/{args.runs}: {done}/{len(graphs)} 已用 {el / 60:.1f} min, "
                      f"预计还需 {el / done * (len(graphs) - done) / 60:.1f} min", flush=True)
        sizes_a, drops_a = np.array(sizes, dtype=float), np.array(drops, dtype=float)
        run_means.append(float(sizes_a.mean()))
        run_drops.append(float(drops_a.mean()))
        run_times.append(time.perf_counter() - t0)
        if run == 0:
            first_run_detail = detail
        print(f"  run {run + 1}: size {sizes_a.mean():.2f}  drop {drops_a.mean():.2f}%  "
              f"({run_times[-1] / 60:.2f} min, {np.mean(times):.2f} s/graph)", flush=True)

    means, drops = np.array(run_means), np.array(run_drops)
    result = {
        "problem": args.problem,
        "dataset": args.dataset,
        "partitioner": args.partitioner,
        "predictor": args.predictor,
        "readout": args.readout,
        "caliber": spec,
        "alpha": args.alpha,
        "beta": args.beta,
        "max_levels": args.max_levels,
        "runs": args.runs,
        "sampling": args.sampling,
        "cdsp_sampling": args.cdsp_sampling,
        "cdsp_rule": args.cdsp_rule,
        "metis_nparts": args.metis_nparts,
        "overrides": {"ts": args.ts or None, "tg": args.tg, "ratio": args.ratio,
                      "hidden_dim": args.hidden_dim or None},
        "tag": args.tag,
        "graphs": len(graphs),
        "ref_mean": float(np.mean([r["size"] for r in refs])),
        "ref_methods": sorted({r["method"] for r in refs}),
        "run_means": run_means,
        "run_drops": run_drops,
        "mean_size": float(means.mean()),
        "std_size": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
        "mean_drop": float(drops.mean()),
        "std_drop": float(drops.std(ddof=1)) if len(drops) > 1 else 0.0,
        "minutes_per_run": [t / 60 for t in run_times],
        "run1_detail": first_run_detail,
    }
    print(f"=== RGF/{args.predictor} {args.problem} {args.dataset} (α={args.alpha}, β={args.beta}, "
          f"{args.partitioner}, runs={args.runs}) ===\n"
          f"  size = {result['mean_size']:.2f} ± {result['std_size']:.2f}    "
          f"drop = {result['mean_drop']:.2f}% ± {result['std_drop']:.2f}%    "
          f"参照 = {result['ref_mean']:.2f}", flush=True)

    suffix = f"-{args.tag}" if args.tag else ""
    out = args.out or (
        args.data_root / "results" / "rgf"
        / f"rgf-{args.predictor}-{args.problem}-{args.dataset}-{args.partitioner}-a{args.alpha}{suffix}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  结果已写入 {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
