"""汇总稀疏图多种子主表：每实例 mean ± std drop、95% CI、min/max、同墙钟 RW 对比。

用法（WSL，经 run_wsl.sh 进 difusco 环境）：

    bash scripts/run_wsl.sh scripts/summarize_sparse_seeds.py \
        [--dir /mnt/e/data/rgf/results/sparse_seeds] [--out table.md]

读 `--dir/` 下 `run_large_table.py` 的两种命名：
  * `<stem>.json`         —— seed = 0（脚本对种子 0 不加后缀）
  * `<stem>.seed<N>.json` —— seed = N（N 非 0）

逐实例输出：
  * n、drop 均值 ± 样本标准差（n−1）、95% 置信区间半宽（t 分布，小样本修正）、min/max；
  * 同墙钟 RW 的 drop（`rw_drop`，通常只有 seed 0 那次跑；`--rw-dir` 可为缺 RW 的
    实例从历史同口径目录补一个参照值，并在表里用 † 标注来源不同）；
  * 墙钟用时 mean ± std（只计 `RGFSolver.solve()`，不含图加载/模型初始化）；
  * β 与 mem_gb（跨种子不一致时标「混合」——β 不同则 drop 不可跨实例直接比较）。

输出既有人读的明细，也有可直接粘进报告的 markdown 表（`--out` 另存一份）。
本脚本只读输入目录，不修改任何结果 JSON。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

DATA = Path("/mnt/e/data/rgf")
SEED_SUFFIX = re.compile(r"^(?P<stem>.+?)\.seed(?P<seed>\d+)$")

# 95% 双侧 t 临界值（df = n-1）；n=1 无 CI。df>12 用 1.96 的保守近似。
T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
}


def load(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, var ** 0.5


def ci95(xs: list[float]) -> float:
    """均值 95% 置信区间半宽；n<2 时返回 nan。"""
    n = len(xs)
    if n < 2:
        return float("nan")
    m, sd = mean_std(xs)
    t = T975.get(n - 1, 1.96)
    return t * sd / math.sqrt(n)


def collect(root: Path) -> tuple[dict[str, list[dict]], list[str]]:
    """按实例聚合结果；返回 (grouped, 被跳过的坏文件列表)。"""
    grouped: dict[str, list[dict]] = {}
    bad: list[str] = []
    for p in sorted(root.glob("*.json")):
        j = load(p)
        if not j or "rgf_drop" not in j:
            bad.append(p.name)
            continue
        m = SEED_SUFFIX.match(p.stem)
        if m:
            stem, seed_name = m.group("stem"), int(m.group("seed"))
        else:
            stem, seed_name = p.stem, 0
        j = dict(j)
        # 历史 JSON（2026-10-04 单次运行）没有 seed 字段，此时以文件名推断
        j["_seed"] = int(j.get("seed", seed_name))
        j["_file"] = p.name
        # 用 JSON 里的 instance 作为规范实例名（文件名解析只是兜底）
        grouped.setdefault(str(j.get("instance", stem)), []).append(j)
    for rows in grouped.values():
        rows.sort(key=lambda r: r["_seed"])
    return grouped, bad


def find_rw(rows: list[dict], stem: str, rw_dir: Path | None) -> tuple[float | None, str]:
    """实例的 RW drop：优先本目录 seed0 那次（同墙钟同批次），否则查历史目录。"""
    for r in rows:
        if r.get("rw_drop") is not None:
            return float(r["rw_drop"]), ""
    if rw_dir is not None:
        j = load(rw_dir / f"{stem}.json")
        if j and j.get("rw_drop") is not None:
            return float(j["rw_drop"]), "†"
    return None, ""


def fmt(x: float | None, spec: str = ".3f", dash: str = "—") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return dash
    return format(x, spec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=DATA / "results" / "sparse_seeds",
                    help="结果 JSON 目录（默认 /mnt/e/data/rgf/results/sparse_seeds）")
    ap.add_argument("--rw-dir", type=Path, default=DATA / "results" / "large",
                    help="为缺 rw_drop 的实例补历史同口径 RW 的目录（默认 results/large；"
                         "传空字符串关闭）")
    ap.add_argument("--out", type=Path, default=None,
                    help="可选：把 markdown 表另存到此文件")
    args = ap.parse_args()

    root: Path = args.dir
    rw_dir: Path | None = args.rw_dir
    if rw_dir is not None and str(rw_dir) == "":
        rw_dir = None
    if rw_dir is not None and not rw_dir.is_dir():
        rw_dir = None

    grouped, bad = collect(root)
    if not grouped:
        print(f"{root} 下没有可用结果（可能仍在跑）")
        return 0

    print(f"数据目录：{root}")
    print(f"实例数：{len(grouped)}；总运行数：{sum(len(v) for v in grouped.values())}")
    if bad:
        print(f"跳过 {len(bad)} 个不可解析文件：{', '.join(bad)}")
    if rw_dir is not None:
        print(f"历史 RW 补位目录：{rw_dir}（表内 † 表示 RW 来自该目录，非同批次运行）")
    print()

    head = (f"{'实例':<22}{'n':>2}  {'drop 均值±std (pp)':>24}  {'95%CI ±':>8}  "
            f"{'min':>7}  {'max':>7}  {'RW drop':>9}  {'Δ(RW−RGF)':>10}  "
            f"{'用时(min) 均值±std':>19}  {'β':>8}  {'|V|':>10}  {'|E|':>11}")
    print(head)
    print("-" * len(head))

    md: list[str] = []
    md.append("| 实例 | n | drop 均值 ± std (pp) | 95% CI ± | min | max | RW drop (pp) | Δ = RW − RGF (pp) | 用时 mean ± std (min) | β | \\|V\\| | \\|E\\| |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

    inst_stats: list[tuple[str, int, float, float]] = []  # (stem, n, mean, std) 汇总噪声用
    for stem in sorted(grouped):
        rows = grouped[stem]
        drops = [float(r["rgf_drop"]) for r in rows]
        mins = [float(r["rgf_time_s"]) / 60 for r in rows if r.get("rgf_time_s") is not None]
        dm, ds = mean_std(drops)
        tm, ts = mean_std(mins)
        half = ci95(drops)
        dmin, dmax = min(drops), max(drops)
        beta0 = rows[0].get("beta", "—")
        betas = {r.get("beta") for r in rows}
        beta_s = fmt(beta0, "d", "—") if betas == {beta0} or len(betas) == 1 else "混合"
        rw, rw_mark = find_rw(rows, stem, rw_dir)
        delta = None if rw is None else rw - dm
        n = rows[0].get("n", 0)
        m = rows[0].get("m", 0)

        print(f"{stem:<22}{len(rows):>2}  {dm:>11.3f} ± {ds:>7.3f}      {fmt(half, '.3f'):>8}  "
              f"{dmin:>7.3f}  {dmax:>7.3f}  {fmt(rw, '.3f'):>9}  "
              f"{('+' if delta is not None and delta >= 0 else '') + fmt(delta, '.3f') if delta is not None else '—':>10}  "
              f"{tm:>8.2f} ± {ts:>5.2f}    {fmt(beta0, 'd'):>8}  {n:>10,d}  {m:>11,d}")
        md.append(
            f"| {stem} | {len(rows)} | {dm:.3f} ± {ds:.3f} | {fmt(half, '.3f')} | "
            f"{dmin:.3f} | {dmax:.3f} | {fmt(rw, '.3f')}{rw_mark} | "
            f"{fmt(delta, '+.3f') if delta is not None else '—'} | "
            f"{tm:.2f} ± {ts:.2f} | {fmt(beta0, 'd')} | {n:,d} | {m:,d} |"
        )
        inst_stats.append((stem, len(rows), dm, ds))

        detail = ", ".join(
            f"s{r['_seed']}:{float(r['rgf_drop']):.3f}"
            + (f"/|S|={int(r['rgf_size'])}" if r.get("rgf_size") is not None else "")
            + (f"/{r['rgf_time_s'] / 60:.1f}min" if r.get("rgf_time_s") is not None else "")
            for r in rows
        )
        rw_note = ""
        if rw is not None:
            rw_note = f"；RW drop={rw:.3f}（Δ={delta:+.3f} pp）{rw_mark}"
        print(f"{'':<22}  逐种子 {detail}{rw_note}  [β={beta_s}]")

    # ---------------- 汇总判读 ----------------
    stds = [sd for _, n, _, sd in inst_stats if n >= 2]
    if stds:
        sm, ss = mean_std(stds)
        print()
        print(f"种内标准差：{sm:.4f} ± {ss:.4f} pp（{len(stds)} 个实例，n≥2）；"
              f"最大 {max(stds):.4f} pp")
    print()
    print("判读口径：")
    print("  1. std 是同一实例跨种子的样本标准差；n=5 时相对标准误约 32%，应读作噪声量级而非精确误差棒。")
    print("  2. 同种子重跑本身也不逐位复现（docs/RESULTS-seeds-2026-10-06.md §4.2），整表分辨率约 ±0.02 pp；")
    print("     实例间差异不显著时不要排序。")
    print("  3. 95% CI 用 t 分布小样本临界值（df=n−1），n=1 时为「—」。")
    print("  4. RW 是「与 RGF 同墙钟」的单次运行；Δ>0 表示同墙钟下 RW 的 drop 更高（即 RGF 更好）。")
    print("     带 † 的 RW 来自 --rw-dir 的历史运行，非同批次、可能口径（β/mem）不同。")
    print("  5. β 不同的实例之间不能直接比较 drop（D-25：β 是质量主控变量）。")

    if args.out is not None:
        md_text = "\n".join(md) + "\n"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md_text, encoding="utf-8")
        print()
        print(f"markdown 表已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
