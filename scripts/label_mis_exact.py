"""为图数据集并行求**精确最优** MIS 标签，缓存为 `<stem>.opt.npz`。

用途：CDSP 监督的目标是 `V_C ∩ S*`（论文 Def. 4 要求 $S^*$ 是**最优解**），
所以训练数据必须配精确标签，而不是 KaMIS 的近似标签（虽然实测 RB 上两者一致，
见 D-26）。

每个进程一个 MILP（HiGHS 单线程），并行度靠 `--workers`；
`--time-limit 0` 表示不给 MILP 设上限（求到最优），非 0 时记录是否证得最优。

用法：
    python scripts/label_mis_exact.py --dataset /mnt/e/data/rgf/rb/rb200_300/train --workers 24
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _solve_one(args: tuple[str, float, bool]) -> tuple[str, int, bool, float, str]:
    import numpy as np

    from rgf.datasets.io import load_graph_npz
    from rgf.problems.exact import solve_mis_timed

    path_str, time_limit, force = args
    path = Path(path_str)
    # 不能用 with_suffix("").with_suffix(".opt.npz")：名字里带点的数据集
    # （ER_700_800_0.15_0）会被当成"后缀"替换，拼出错误文件名。
    out = path.with_name(path.stem + ".opt.npz")
    if out.exists() and not force:
        with np.load(out) as z:
            return path.name, int(z["labels"].sum()), bool(z["optimal"]), 0.0, "cached"
    g = load_graph_npz(path)
    t0 = time.perf_counter()
    labels, optimal = solve_mis_timed(g, np.full(g.n, -1, dtype=np.int8),
                                      time_limit_s=(time_limit or None))
    elapsed = time.perf_counter() - t0
    np.savez_compressed(out, n=g.n, labels=labels.astype(np.int8),
                        objective=int(np.count_nonzero(labels == 1)),
                        optimal=bool(optimal), time_s=float(elapsed))
    return path.name, int(np.count_nonzero(labels == 1)), bool(optimal), elapsed, "solved"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="含 *.npz 图文件的目录")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0=全部）")
    ap.add_argument("--time-limit", type=float, default=0.0, help="MILP 墙钟上限（0=求到最优）")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(args.dataset)
    print(f"精确 MIS 标注：目录 {root}", flush=True)
    # 只做**一次**目录扫描：这个目录有 20 万个文件且挂在 9p 上，逐个 exists() 会卡住
    all_paths = list(root.glob("*.npz"))
    files = sorted(p for p in all_paths
                   if not p.name.endswith((".label.npz", ".dsref.npz", ".opt.npz")))
    have = {p.name for p in all_paths if p.name.endswith(".opt.npz")}
    if args.limit:
        files = files[: args.limit]
    todo = [f for f in files
            if args.force or f"{f.stem}.opt.npz" not in have]
    print(f"  图文件 {len(files)} 张，待处理 {len(todo)} 张，并行 {args.workers}，"
          f"时间上限 {'无（求最优）' if not args.time_limit else f'{args.time_limit:.0f}s'}", flush=True)
    if not todo:
        print("  无需处理", flush=True)
        return 0

    t_start = time.perf_counter()
    done = 0
    suboptimal = 0
    slowest = (0.0, "")
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(_solve_one, (str(f), args.time_limit, args.force)) for f in todo]
        for fut in as_completed(futures):
            name, size, optimal, elapsed, note = fut.result()
            done += 1
            if not optimal:
                suboptimal += 1
            if elapsed > slowest[0]:
                slowest = (elapsed, name)
            if done % 200 == 0 or done == len(todo):
                rate = (time.perf_counter() - t_start) / done
                eta = rate * (len(todo) - done) / 60
                print(f"  {done}/{len(todo)} 已用 {(time.perf_counter() - t_start) / 60:.1f} min，"
                      f"预计还需 {eta:.1f} min，最新 {name} |S|={size}（{note}）", flush=True)
    print(f"完成：{done} 张，未证最优 {suboptimal} 张，最慢 {slowest[0]:.1f}s（{slowest[1]}），"
          f"总用时 {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
