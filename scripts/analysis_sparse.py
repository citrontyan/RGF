"""Aggregate and statistically analyse the sparse-graph RGF experiments.

Inputs (all are result JSONs written by scripts/run_large_table.py, scripts/eval_direct_large.py):
  * sparse_seeds/       -- alpha=500 main table, multi-seed (RGF + same-wallclock RW)
  * sparse_alpha/       -- alpha grid, alpha<A>/<instance>[.seedN].json
  * sparse_error/       -- per-level telemetry reruns (alpha500 / alpha2000)
  * reference_audit/    -- KaMIS runtime/seed-stability audit (reference_audit.jsonl)
  * direct_large/       -- direct (non-recursive) Fast-T2T baseline
  * archived mirrors can be added with --extra-dir (e.g. results_archive/large_seeds).

Outputs a markdown report to stdout (and optionally --out-md), with:
  1. per-instance seed mean/std of drop and wall-clock;
  2. alpha effect: paired (same instance, same seed) Delta-drop vs alpha=500 baseline;
  3. RGF vs direct neural baseline: per-instance paired comparison;
  4. RGF vs same-wallclock random walk: per-instance paired comparison;
  5. paired bootstrap CIs + Wilcoxon signed-rank tests (scipy);
  6. reference audit summary (KaMIS wall-clock and seed stability).

Usage (Windows venv has numpy/scipy; paths are Windows-side because this only reads JSON):
    python scripts/analysis_sparse.py --root E:\\data\\rgf\\results --out-md docs/analysis.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:  # scipy is available in both the Windows venv and the WSL env
    from scipy import stats
except Exception:  # noqa: BLE001
    stats = None


# ------------------------------------------------------------------ loading
def load_json_dir(path: Path, alpha: int | None = None) -> list[dict]:
    records = []
    if not path.exists():
        return records
    for f in sorted(path.glob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if "rgf_drop" not in rec and "direct_drop" not in rec:
            continue
        if alpha is not None and rec.get("alpha") not in (None, alpha):
            continue
        records.append(rec)
    return records


def load_alpha_grid(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    for f in sorted(path.glob("alpha*/*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if "rgf_drop" in rec and rec.get("alpha") is not None:
            rec["_alpha_dir"] = f.parent.name
            records.append(rec)
    return records


def load_direct(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    for f in sorted(path.glob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if rec.get("status") == "ok" and "direct_drop" in rec:
            records.append(rec)
    return records


# ------------------------------------------------------------------ stats
def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    ddof = 1 if arr.size > 1 else 0
    return float(arr.mean()), float(arr.std(ddof=ddof))


def bootstrap_ci(values: list[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def paired_report(pairs: list[tuple[str, float, float]], label: str) -> list[str]:
    """pairs = [(key, value_a, value_b)] -> markdown lines for a paired test."""
    lines = []
    if not pairs:
        lines.append(f"- {label}: no paired data.")
        return lines
    diffs = np.asarray([a - b for _, a, b in pairs], dtype=float)
    mean_d = float(diffs.mean())
    lo, hi = bootstrap_ci(list(diffs))
    p = float("nan")
    if stats is not None and diffs.size >= 5:
        try:
            p = float(stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
        except Exception:  # noqa: BLE001
            p = float("nan")
    lines.append(
        f"- {label}: n={len(pairs)}, mean delta={mean_d:+.4f} pp "
        f"[95% bootstrap CI {lo:+.4f}, {hi:+.4f}], Wilcoxon p={p:.3g}; "
        f"positive means first better."
    )
    return lines


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(r"E:\data\rgf\results"))
    ap.add_argument("--extra-dir", type=Path, action="append", default=[],
                    help="additional directory of alpha=500 RGF JSONs (e.g. an archive mirror)")
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args()

    root = args.root
    seeds = load_json_dir(root / "sparse_seeds")
    alpha_grid = load_alpha_grid(root / "sparse_alpha")
    direct = load_direct(root / "direct_large")
    ref_audit: list[dict] = []
    audit_path = root / "reference_audit" / "reference_audit.jsonl"
    if audit_path.exists():
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ref_audit.append(json.loads(line))

    lines = ["# Sparse-graph RGF experiment analysis", ""]

    # ---- 1. per-instance seed table (alpha=500)
    by_inst: dict[str, list[dict]] = defaultdict(list)
    for rec in seeds:
        by_inst[rec["instance"]].append(rec)
    lines += ["## 1. alpha=500 main table (per-instance, multi-seed)", "",
              "| instance | n | seeds | mean drop (pp) | std | min | max | mean time (min) | RW drop (pp) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for inst in sorted(by_inst):
        recs = by_inst[inst]
        drops = [r["rgf_drop"] for r in recs]
        m, s = mean_std(drops)
        times = [r.get("rgf_time_s", float("nan")) / 60 for r in recs]
        rw = [r["rw_drop"] for r in recs if r.get("rw_drop") is not None]
        rw_txt = f"{rw[0]:.3f}" if rw else "–"
        lines.append(
            f"| {inst} | {recs[0].get('n', '')} | {len(recs)} | {m:.4f} | {s:.4f} | "
            f"{min(drops):.4f} | {max(drops):.4f} | {np.nanmean(times):.2f} | {rw_txt} |"
        )
    lines.append("")

    # ---- 2. alpha effect (paired by instance+seed, baseline alpha=500)
    base = {}
    base_time = {}
    for rec in seeds:
        key = (rec["instance"], rec.get("seed", 0))
        base[key] = rec["rgf_drop"]
        base_time[key] = rec.get("rgf_time_s")
    if (root / "large_seeds").exists():
        for rec in load_json_dir(root / "large_seeds", alpha=500):
            key = (rec["instance"], rec.get("seed", 0))
            base.setdefault(key, rec["rgf_drop"])
            base_time.setdefault(key, rec.get("rgf_time_s"))
    for extra in args.extra_dir:
        for rec in load_json_dir(extra, alpha=500):
            key = (rec["instance"], rec.get("seed", 0))
            base.setdefault(key, rec["rgf_drop"])
            base_time.setdefault(key, rec.get("rgf_time_s"))

    alpha_vals = sorted({int(r["alpha"]) for r in alpha_grid})
    lines += ["## 2. alpha effect (paired: same instance, same seed; reference alpha=500)", ""]
    for a in alpha_vals:
        pairs = []
        for rec in alpha_grid:
            if int(rec["alpha"]) != a:
                continue
            key = (rec["instance"], rec.get("seed", 0))
            if key in base:
                pairs.append((f"{key[0]}#{key[1]}", base[key] - rec["rgf_drop"], 0.0))
        lines += paired_report([(k, a_, b_) for k, a_, b_ in pairs], f"alpha={a} vs alpha=500 (drop improvement, pp)")
        tr = []
        for r in alpha_grid:
            if int(r["alpha"]) != a:
                continue
            t0 = base_time.get((r["instance"], r.get("seed", 0)))
            if t0 and r.get("rgf_time_s"):
                tr.append(r["rgf_time_s"] / t0)
        tr = [x for x in tr if isinstance(x, float) and math.isfinite(x)]
        if tr:
            lines.append(f"  - wall-clock ratio alpha={a}/alpha=500: median {np.median(tr):.3f} "
                         f"(<1 means faster)")
    lines.append("")

    # ---- 2b. theory validation: loss vs |V_NN| = cut_vertices (telemetry rows only)
    tele_rows = []
    for d in [root / "sparse_error_lead", root / "sparse_error"]:
        if not d.exists():
            continue
        for rec in load_alpha_grid(d):
            if rec.get("cut_vertices") is None or not rec.get("rgf_drop"):
                continue
            tele_rows.append(rec)
    if tele_rows:
        lines += ["## 2b. theory validation: drop vs |V_NN| (= cut_vertices)", "",
                  "| instance | alpha | seed | drop (pp) | |V_NN| | |V_NN|/n | drop / |V_NN| (1e-4) |",
                  "|---|---|---|---|---|---|---|"]
        for rec in sorted(tele_rows, key=lambda r: (r["instance"], int(r["alpha"]), r.get("seed", 0))):
            vn = rec["cut_vertices"]
            ratio = rec["rgf_drop"] / vn * 1e4 if vn else float("nan")
            lines.append(
                f"| {rec['instance']} | {rec['alpha']} | {rec.get('seed', 0)} | {rec['rgf_drop']:.4f} | "
                f"{vn:,} | {vn / rec['n'] * 100:.3f}% | {ratio:.3f} |"
            )
        lines.append("")

    # ---- 3. RGF vs direct neural baseline (paired by instance)
    best_rgf: dict[str, list[float]] = {}
    for rec in seeds:
        best_rgf.setdefault(rec["instance"], []).append(rec["rgf_drop"])
    for rec in alpha_grid:
        best_rgf.setdefault(rec["instance"], []).append(rec["rgf_drop"])
    # 归档里的 alpha=500 RGF 行也要能跟"直接基线"配对（新目录可能还没有该实例）
    for rec in load_json_dir(root / "large", alpha=500) if (root / "large").exists() else []:
        best_rgf.setdefault(rec["instance"], []).append(rec["rgf_drop"])
    for rec in load_json_dir(root / "large_ext", alpha=500) if (root / "large_ext").exists() else []:
        best_rgf.setdefault(rec["instance"], []).append(rec["rgf_drop"])
    for extra in args.extra_dir:
        for rec in load_json_dir(extra, alpha=500):
            best_rgf.setdefault(rec["instance"], []).append(rec["rgf_drop"])
    lines += ["## 3. RGF (best alpha) vs direct non-recursive Fast-T2T", "",
              "| instance | n | RGF best drop | alpha* | direct drop | advantage (pp) |",
              "|---|---|---|---|---|---|"]
    pairs3 = []
    for rec in direct:
        inst = rec["instance"]
        if inst not in best_rgf:
            continue
        vals = best_rgf[inst]
        best = min(vals)
        # alpha of the best run is reported in section 2; find it
        alpha_of_best = next((r["alpha"] for r in seeds + alpha_grid
                              if r["instance"] == inst and float(r["rgf_drop"]) == best), "")
        adv = rec["direct_drop"] - best
        pairs3.append((inst, rec["direct_drop"], best))
        lines.append(f"| {inst} | {rec.get('n', '')} | {best:.4f} | {alpha_of_best} | "
                     f"{rec['direct_drop']:.4f} | {adv:+.4f} |")
    lines += paired_report(pairs3, "direct - RGF (positive = RGF better)")
    lines.append("")

    # ---- 4. RGF vs same-wallclock RW (paired by instance+seed)
    pairs4 = []
    for rec in seeds:
        if rec.get("rw_drop") is not None:
            pairs4.append((f"{rec['instance']}#{rec.get('seed', 0)}", rec["rw_drop"], rec["rgf_drop"]))
    lines += ["## 4. RGF vs same-wallclock random walk", ""]
    lines += paired_report(pairs4, "RW - RGF (positive = RGF better)")
    lines.append("")

    # ---- 5. reference audit
    if ref_audit:
        lines += ["## 5. KaMIS reference audit", "",
                  "| instance | n | seed | |S| | delta vs cached | wall (s) |",
                  "|---|---|---|---|---|---|"]
        for rec in ref_audit:
            lines.append(f"| {rec['instance']} | {rec['n']} | {rec['seed']} | {rec['kamis_size']:,} | "
                         f"{rec['delta_vs_cached']} | {rec['wall_s']:.2f} |")
        lines.append("")

    md = "\n".join(lines)
    print(md)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
        print(f"\nwritten to {args.out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
