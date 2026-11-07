"""R6b：同墙钟 Random Walk 基线——预算取配对 RGF run 的实测 rgf_time_s。

论文 Table 9 的 RW 列口径是 "same wall clock"：RW 与同一实例的 RGF 用相同时长。
本脚本不重跑 RGF，直接读 RGF 的 result JSON（含 rgf_time_s / rgf_size / kamis_ref），
用**当前代码**（rgf.baselines.random_walk.random_walk_mis）在该预算下跑 RW，
写出可追溯的 JSON：

    wsl -e bash scripts/run_wsl.sh scripts/run_rw_from_result.py \
        --result-json /mnt/e/data/rgf/results/rerun_2026-11/results/sparse/r1/a500/europe.osm.json \
        --data-dir /mnt/e/data/rgf/rerun_2026-11/data/dimacs10 \
        --out /mnt/e/data/rgf/results/rerun_2026-11/results/rw/europe.osm.a500.json --seed 0

输出字段：rw_size / rw_drop / rw_budget_s / rw_iterations + 配对的 rgf_size / rgf_time_s /
rgf_drop / kamis_ref，以及 seed 与 RNG 说明（供 run_record 归档）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rgf.baselines.random_walk import random_walk_mis  # noqa: E402
from rgf.datasets.io import load_graph_npz  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-json", type=Path, required=True, help="配对的 RGF 结果 JSON")
    ap.add_argument("--data-dir", type=Path, required=True, help="图 npz 所在目录")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-s", type=float, default=0.0,
                    help="0 = 用 result JSON 里的 rgf_time_s（论文口径）")
    args = ap.parse_args()

    rgf = json.loads(args.result_json.read_text(encoding="utf-8"))
    stem = rgf["instance"]
    graph_path = args.data_dir / f"{stem}.npz"
    if not graph_path.is_file():
        print(f"missing graph: {graph_path}", file=sys.stderr)
        return 1
    budget = args.budget_s if args.budget_s > 0 else float(rgf["rgf_time_s"])
    g = load_graph_npz(graph_path)
    rng = np.random.default_rng(args.seed)
    print(f"[rw] {stem}: |V|={g.n:,} |E|={g.m:,} budget={budget:.1f}s (RGF rgf_time_s) "
          f"seed={args.seed}", flush=True)
    t0 = time.perf_counter()
    labels, iterations = random_walk_mis(g, budget, rng=rng)
    elapsed = time.perf_counter() - t0
    size = int(np.asarray(labels).sum())
    ref = int(rgf["kamis_ref"])
    drop = (ref - size) / ref * 100.0
    rec = {
        "instance": stem,
        "n": int(g.n),
        "m": int(g.m),
        "seed": args.seed,
        "rw_size": size,
        "rw_drop": drop,
        "rw_budget_s": budget,
        "rw_iterations": int(iterations),
        "rw_elapsed_s": elapsed,
        "rgf_size": rgf.get("rgf_size"),
        "rgf_drop": rgf.get("rgf_drop"),
        "rgf_time_s": rgf.get("rgf_time_s"),
        "kamis_ref": ref,
        "budget_source": "rgf_time_s of the paired RGF run (paper: same wall clock)",
        "paired_result_json": str(args.result_json),
        "rng": "numpy.random.default_rng(seed)",
        "deterministic": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[rw] {stem}: |S|={size:,} drop={drop:.4f}% iterations={iterations} "
          f"elapsed={elapsed:.1f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
