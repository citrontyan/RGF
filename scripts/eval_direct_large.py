"""Direct (non-recursive) Fast-T2T baseline on large sparse graphs.

RQ4 只把 RGF 与 KaMIS/随机游走比较，缺少"同一神经预测器不做递归"的直接基线。
本脚本补上这一行：把**整张图**一次性喂给 Fast-T2T（不切分、不递归），贪心解码成
MIS，再与缓存的 KaMIS 参照解比 drop。它回答的是"递归框架本身值多少"：
同样的预测器、同样的读出，直接推理在大图上的质量与显存门槛。

口径：
* 预测器 = 与 RGF 大图口径相同的官方权重（hidden 128 / mis_er.ckpt）或 hidden 256；
* 读出 = decode-then-read（与 RGF 的 DecodedCutPredictor 同一解码器）；
* 参照解 = `<data-dir>/<stem>.label.npz`（KaMIS，已缓存）；
* 显存门槛按实测 ~7.8 KB/边（hidden 128）/ 15.5 KB/边（hidden 256）+ 1.5 GB 常驻估算，
  超过 --mem-gb 的实例记为 `skipped_oom`，不做无谓尝试（也说明 RGF 能处理的规模更大）。

    wsl -e bash scripts/run_wsl.sh scripts/eval_direct_large.py \
        --instances luxembourg.osm,coAuthorsCiteseer --out-dir /mnt/e/data/rgf/results/direct_large

结果写入 `<out-dir>/<stem>[.s<K>].json`；已存在则跳过（--force 覆盖）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 必须在 import torch 前设置（与 run_large_table.py 相同的大图碎片缓解）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rgf.datasets.io import load_graph_npz  # noqa: E402
from rgf.predictor.fastt2t import FastT2TPredictor  # noqa: E402
from rgf.problems import MaximumIndependentSet  # noqa: E402
from run_rgf import DecodedCutPredictor  # noqa: E402

DATA = Path("/mnt/e/data/rgf")
CKPTS = {
    256: DATA / "checkpoints/fastt2t/mis_rb.ckpt",
    128: DATA / "checkpoints/fastt2t/mis_er.ckpt",
}
BYTES_PER_EDGE = {256: 15.5e3, 128: 7.8e3}
MODEL_RESIDENT_GB = 1.5

# 默认实例集：一张图整图前向能在 24GB 内完成的稀疏实例（KaMIS 参照均已缓存）
DEFAULT_INSTANCES = [
    "luxembourg.osm", "citationCiteseer", "coAuthorsCiteseer", "coAuthorsDBLP",
    "com-Amazon", "com-DBLP", "web-NotreDame", "web-Stanford",
    "roadNet-PA", "roadNet-TX", "belgium.osm",
]

DIRS = {
    "luxembourg.osm": "dimacs10", "citationCiteseer": "dimacs10",
    "coAuthorsCiteseer": "dimacs10", "coAuthorsDBLP": "dimacs10",
    "belgium.osm": "dimacs10",
    "com-Amazon": "external", "com-DBLP": "external", "web-NotreDame": "external",
    "web-Stanford": "external", "roadNet-PA": "external", "roadNet-TX": "external",
    "roadNet-CA": "external", "amazon0601": "external", "web-BerkStan": "external",
    "netherlands.osm": "dimacs10", "asia.osm": "dimacs10",
}


def estimate_gb(g, hidden: int) -> float:
    return g.m * BYTES_PER_EDGE.get(hidden, 15.5e3) / 1024 ** 3 + MODEL_RESIDENT_GB


def health(g, labels: np.ndarray) -> dict:
    sel = labels == 1
    if g.m:
        conflicts = int((sel[g.edges[:, 0]] & sel[g.edges[:, 1]]).sum())
    else:
        conflicts = 0
    return {"conflicts": conflicts, "size": int(sel.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default=",".join(DEFAULT_INSTANCES),
                    help="逗号分隔实例名，或 all（默认实例集）")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="实例 npz 与 <stem>.label.npz 目录；缺省按 DIRS 表选 dimacs10/external")
    ap.add_argument("--out-dir", type=Path, default=DATA / "results" / "direct_large")
    ap.add_argument("--hidden", type=int, default=128, choices=[128, 256])
    ap.add_argument("--steps", type=int, default=1, help="Fast-T2T 推理扩散步数 Ts")
    ap.add_argument("--guided", action="store_true", help="开启 Fast-T2T guided rewrite")
    ap.add_argument("--rewrite-steps", type=int, default=0, help="guided rewrite 轮数 Tg")
    ap.add_argument("--ratio", type=float, default=0.2, help="guided rewrite ratio")
    ap.add_argument("--samples", type=int, default=1, help="独立采样次数，取解码后最优（best-of-K）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-gb", type=float, default=20.0, help="整图前向的显存预算（估算门槛）")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的结果 JSON")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.instances == "all":
        names = list(DEFAULT_INSTANCES)
    else:
        names = [s.strip() for s in args.instances.split(",") if s.strip()]

    for name in names:
        stem = name[:-4] if name.endswith(".npz") else name
        if args.data_dir is not None:
            data_dir = args.data_dir
        else:
            data_dir = DATA / DIRS.get(stem, "dimacs10")
        graph_path = data_dir / f"{stem}.npz"
        if not graph_path.exists():
            print(f"=== {stem}: missing graph {graph_path} ===", flush=True)
            continue
        suffix = "".join(
            [f".h{args.hidden}", f".ts{args.steps}",
             (f".tg{args.rewrite_steps}" if args.guided else ""),
             (f".s{args.samples}" if args.samples > 1 else ""),
             ("" if args.seed == 0 else f".seed{args.seed}")]
        )
        out = out_dir / f"{stem}{suffix}.json"
        if out.exists() and not args.force:
            print(f"=== {stem}: 已有结果，跳过 ===", flush=True)
            continue

        g = load_graph_npz(graph_path)
        density = g.m / (g.n * (g.n - 1)) if g.n > 1 else 0.0
        avg_deg = 2 * g.m / g.n
        ref_path = data_dir / f"{stem}.label.npz"
        if not ref_path.exists():
            print(f"=== {stem}: missing KaMIS reference {ref_path} ===", flush=True)
            continue
        with np.load(ref_path) as z:
            ref = int(z["labels"].sum())
        est_gb = estimate_gb(g, args.hidden)
        print(f"=== {stem}: |V|={g.n:,} |E|={g.m:,} avg_deg={avg_deg:.2f} "
              f"density={density:.2e} KaMIS={ref:,} 估算显存={est_gb:.1f}GB ===", flush=True)
        if est_gb > args.mem_gb:
            rec = {
                "instance": stem, "n": g.n, "m": g.m, "avg_degree": avg_deg,
                "density": density, "hidden": args.hidden, "steps": args.steps,
                "guided": args.guided, "rewrite_steps": args.rewrite_steps,
                "ratio": args.ratio, "samples": args.samples, "seed": args.seed,
                "kamis_ref": ref, "status": "skipped_oom",
                "estimated_gb": est_gb, "mem_gb": args.mem_gb,
            }
            out.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  跳过：估算显存 {est_gb:.1f}GB > 预算 {args.mem_gb:.0f}GB", flush=True)
            continue

        try:
            import torch  # noqa: PLC0415

            predictor = FastT2TPredictor(
                CKPTS[args.hidden], inference_diffusion_steps=args.steps,
                hidden_dim=args.hidden, guided=args.guided,
                rewrite_steps=args.rewrite_steps, rewrite_ratio=args.ratio,
                seed=args.seed,
            )
            decoder = DecodedCutPredictor(predictor, "mis", cdsp_sampling=1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            best = None
            for k in range(max(1, args.samples)):
                predictor.seed = int(args.seed) * 100_000 + k
                labels = decoder.predict([(g, None)])[0]
                labels = np.asarray(labels, dtype=np.int8)
                if best is None or int(labels.sum()) > int(best.sum()):
                    best = labels
            elapsed = time.perf_counter() - t0
            if torch.cuda.is_available():
                peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
            else:
                peak_gb = None
        except RuntimeError as exc:  # CUDA OOM 等
            rec = {
                "instance": stem, "n": g.n, "m": g.m, "avg_degree": avg_deg,
                "density": density, "hidden": args.hidden, "steps": args.steps,
                "guided": args.guided, "rewrite_steps": args.rewrite_steps,
                "ratio": args.ratio, "samples": args.samples, "seed": args.seed,
                "kamis_ref": ref, "status": "error", "error": str(exc)[:2000],
                "estimated_gb": est_gb, "mem_gb": args.mem_gb,
            }
            out.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  失败：{exc}", flush=True)
            continue

        h = health(g, best)
        size = int(h["size"])
        drop = (ref - size) / ref * 100
        rec = {
            "instance": stem, "n": g.n, "m": g.m, "avg_degree": avg_deg,
            "density": density, "hidden": args.hidden, "steps": args.steps,
            "guided": args.guided, "rewrite_steps": args.rewrite_steps,
            "ratio": args.ratio, "samples": args.samples, "seed": args.seed,
            "kamis_ref": ref, "status": "ok",
            "direct_size": size, "direct_drop": drop,
            "direct_conflicts": h["conflicts"],
            "direct_time_s": elapsed, "peak_gb": peak_gb,
            "estimated_gb": est_gb, "mem_gb": args.mem_gb,
        }
        out.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  direct: |S|={size:,} drop={drop:.2f}% 用时 {elapsed / 60:.1f} min "
              f"峰值显存={peak_gb if peak_gb is None else round(peak_gb, 1)}GB "
              f"冲突边={h['conflicts']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
