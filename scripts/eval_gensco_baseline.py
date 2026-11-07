#!/usr/bin/env python3
"""Standalone GenSCO baseline on the RGF MIS test sets (RB / ER).

Protocol (camera-ready supplement):

* official GenSCO MIS inference, ``sampling_steps=3`` (``scripts/eval/mis_rb.sh``),
  ``decode_intermediate=True``, ``flip_rate=(0.25, 0.4)``;
* ``cycles=1000`` (the RGF paper's C=1000 convention);
* batch size 1, graphs run one at a time; the first 20 graphs of the test split;
* padding to the dataset maximum (300 for RB, 800 for ER), exactly what the
  official eval does when it batches the whole test split;
* metric: ``drop = (ref - size) / ref * 100`` where ``ref`` is the KaMIS
  reference stored in ``*.label.npz`` (same reference the RGF runs use);
* wall-clock: per-graph solve time, excluding the one-off model load/compile
  (a warmup graph is run before timing).

``--seeds 0-4`` runs the same protocol for five seeds (the adapter draws its
per-call seed from the torch RNG, which is seeded explicitly here) and reports
the 5-seed mean +/- std, matching the RGF 5-run aggregation.

Run inside the isolated ``gensco`` env::

    CUDA_VISIBLE_DEVICES=1 wsl -e bash scripts/run_wsl.sh \\
        scripts/eval_gensco_baseline.py --dataset rb --limit 20

(``scripts/run_wsl.sh`` defaults to ``ENV_NAME=difusco``; for this predictor use
``ENV_NAME=gensco``.)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rgf.datasets.io import load_graph_npz  # noqa: E402
from rgf.predictor.gensco import DEFAULT_CYCLES, GenSCOPredictor  # noqa: E402

DATASETS = {
    "rb": ("rb/rb200_300/test", 300),
    "er": ("er/er700_800/test", 800),
}
_REF_SUFFIXES = (".label.npz", ".dsref.npz", ".opt.npz")


def _commands() -> str:
    return " ".join(["python"] + sys.argv)


def _git_commit(path: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        return None


def load_test_graphs(data_root: Path, dataset: str, limit: int):
    rel, _pad = DATASETS[dataset]
    test_dir = data_root / rel
    clips = [
        clip
        for clip in sorted(test_dir.glob("*.npz"))
        if not clip.name.endswith(_REF_SUFFIXES)
    ]
    if limit:
        clips = clips[:limit]
    graphs, refs = [], []
    for clip in clips:
        graph = load_graph_npz(clip)
        with np.load(clip.with_suffix(".label.npz")) as data:
            ref = int(data["labels"].sum())
        graphs.append(graph)
        refs.append({"graph": clip.stem, "ref": ref, "method": "kamis"})
    return graphs, refs


def compare_with_official(rows, official_path: Path) -> dict:
    """Sanity check: our first-N graphs match the official MaskCO test npz order."""
    try:
        with np.load(official_path) as data:
            official_n = [int(x) for x in data["num_nodes"][: len(rows)]]
            official_m = [
                int(np.any(data["edges"][i] != 0, axis=-1).sum()) for i in range(len(rows))
            ]
    except Exception as exc:  # noqa: BLE001 - optional cross-check only
        return {"available": False, "error": repr(exc)}
    ours_n = [row["n"] for row in rows]
    ours_m = [row["m"] for row in rows]
    return {
        "available": True,
        "path": str(official_path),
        "n_match": ours_n == official_n,
        "m_match": ours_m == official_m,
        "ours_n_head": ours_n[:5],
        "official_n_head": official_n[:5],
    }


def parse_seeds(spec: str) -> list[int]:
    """Parse ``"0-4"`` / ``"0,1,2,3,4"`` into a list of ints."""
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            seeds.extend(range(int(low), int(high) + 1))
        else:
            seeds.append(int(part))
    return seeds


def seed_torch(seed: int) -> None:
    """Seed the RNG stream the adapter draws its per-call seeds from."""
    try:
        import torch  # noqa: PLC0415 - torch is present in the gensco env (CPU build)

        torch.manual_seed(int(seed))
    except Exception:  # noqa: BLE001 - adapter falls back to its own counter
        pass


def run_graphs(predictor, graphs, refs, cycles: int, verbose: bool = True) -> dict:
    """Solve every graph once; returns per-graph rows and per-seed aggregates."""
    rows = []
    started = time.perf_counter()
    for index, (graph, ref) in enumerate(zip(graphs, refs)):
        result = predictor.solve_standalone(graph, cycles=cycles)
        drop = (ref["ref"] - result["size"]) / ref["ref"] * 100.0
        rows.append(
            {
                "graph": ref["graph"],
                "n": int(graph.n),
                "m": int(graph.m),
                "ref": ref["ref"],
                "size": int(result["size"]),
                "drop": round(float(drop), 6),
                "time_s": round(float(result["seconds"]), 4),
                "cycles": int(result["cycles"]),
                "n_padded": int(result["n_padded"]),
            }
        )
        if verbose:
            print(
                f"  [{index + 1}/{len(graphs)}] {ref['graph']}: n={graph.n} "
                f"size={result['size']} ref={ref['ref']} drop={drop:.3f}% "
                f"time={result['seconds']:.2f}s",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    drops = np.array([row["drop"] for row in rows], dtype=float)
    times = np.array([row["time_s"] for row in rows], dtype=float)
    return {
        "rows": rows,
        "elapsed": float(elapsed),
        "mean_drop": float(drops.mean()),
        "std_drop": float(drops.std(ddof=1)) if len(drops) > 1 else 0.0,
        "mean_time_s": float(times.mean()),
        "std_time_s": float(times.std(ddof=1)) if len(times) > 1 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/e/data/rgf"))
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    parser.add_argument("--sampling-steps", type=int, default=3)
    parser.add_argument("--no-intermediate", action="store_true",
                        help="disable decode_intermediate (official default is on)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None,
                        help="multi-seed protocol, e.g. '0-4' or '0,1,2,3,4'; "
                             "writes standalone_<dataset>_seeds.json by default")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--official-npz", type=Path, default=None,
                        help="optional official MaskCO test npz for a graph-order cross-check")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rel, pad_to = DATASETS[args.dataset]
    data_root = args.data_root
    graphs, refs = load_test_graphs(data_root, args.dataset, args.limit)
    if not graphs:
        print(f"no graphs under {data_root / rel}", file=sys.stderr)
        return 2

    predictor = GenSCOPredictor(
        data_root,
        args.dataset,
        cycles=args.cycles,
        sampling_steps=args.sampling_steps,
        decode_intermediate=not args.no_intermediate,
        seed=args.seed,
        pad_to=pad_to,
        verbose=args.verbose,
    )
    mode = "standalone_seeds" if args.seeds else "standalone"
    print(
        f"GenSCO {mode} | {args.dataset} | {len(graphs)} graphs | "
        f"cycles={args.cycles} | steps={args.sampling_steps} | pad={pad_to} | "
        f"seeds={args.seeds or args.seed} | "
        f"ckpt={predictor.ckpt.name} sha256={predictor.ckpt_sha256[:16]}...",
        flush=True,
    )

    warm_started = time.perf_counter()
    warm = predictor.warmup([pad_to], cycles=2)
    t_warm = time.perf_counter() - warm_started
    print(f"  warmup: {warm} ({t_warm:.1f}s)", flush=True)

    import jax  # noqa: PLC0415

    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ckpt": str(predictor.ckpt),
        "ckpt_sha256": predictor.ckpt_sha256,
        "ckpt_size": predictor.ckpt.stat().st_size,
        "gensco_commit": _git_commit(_REPO_ROOT / "third_party" / "GenSCO"),
        "rgmis_commit": _git_commit(_REPO_ROOT),
    }
    common = {
        "predictor": "gensco",
        "dataset": args.dataset,
        "graphs": len(graphs),
        "cycles": args.cycles,
        "sampling_steps": args.sampling_steps,
        "decode_intermediate": not args.no_intermediate,
        "flip_rate": [0.25, 0.4],
        "batch_size": 1,
        "n_padded": pad_to,
        "ref_method": "kamis",
        "environment": environment,
        "warmup": warm,
        "warmup_seconds": round(float(t_warm), 2),
        "command": _commands(),
        "cwd": os.getcwd(),
        "wsl": os.environ.get("WSL_DISTRO_NAME"),
    }

    if args.seeds:
        seeds = parse_seeds(args.seeds)
        per_seed = []
        all_rows = []
        started = time.perf_counter()
        for seed in seeds:
            seed_torch(seed)  # deterministic per-call seed stream for this seed run
            run = run_graphs(predictor, graphs, refs, args.cycles, verbose=args.verbose)
            per_seed.append(
                {
                    "seed": seed,
                    "mean_drop": round(run["mean_drop"], 6),
                    "std_drop": round(run["std_drop"], 6),
                    "mean_time_s": round(run["mean_time_s"], 4),
                    "std_time_s": round(run["std_time_s"], 4),
                    "total_time_s": round(run["elapsed"], 2),
                    "rows": run["rows"],
                }
            )
            all_rows.extend({"seed": seed, **row} for row in run["rows"])
            print(
                f"  seed {seed}: drop {run['mean_drop']:.3f}% ± {run['std_drop']:.3f}% "
                f"| time {run['mean_time_s']:.2f}s/graph",
                flush=True,
            )
        elapsed = time.perf_counter() - started
        seed_drops = np.array([row["mean_drop"] for row in per_seed], dtype=float)
        seed_times = np.array([row["mean_time_s"] for row in per_seed], dtype=float)
        ddof = 1 if len(seed_drops) > 1 else 0
        summary = {
            **common,
            "mode": "standalone_seeds",
            "seeds": seeds,
            "mean_drop": round(float(seed_drops.mean()), 6),
            "std_drop": round(float(seed_drops.std(ddof=ddof)), 6),
            "mean_time_s": round(float(seed_times.mean()), 4),
            "std_time_s": round(float(seed_times.std(ddof=ddof)), 4),
            "total_time_s": round(float(elapsed), 2),
            "per_seed": per_seed,
            "rows": all_rows,
        }
        out = args.out or (
            _REPO_ROOT
            / "results_archive"
            / "camera_ready"
            / "recent_predictors"
            / "gensco"
            / f"standalone_{args.dataset}_seeds.json"
        )
    else:
        seed_torch(args.seed)
        run = run_graphs(predictor, graphs, refs, args.cycles, verbose=True)
        summary = {
            **common,
            "mode": "standalone",
            "mean_drop": round(run["mean_drop"], 6),
            "std_drop": round(run["std_drop"], 6),
            "mean_time_s": round(run["mean_time_s"], 4),
            "std_time_s": round(run["std_time_s"], 4),
            "total_time_s": round(float(run["elapsed"]), 2),
            "rows": run["rows"],
        }
        out = args.out or (
            _REPO_ROOT
            / "results_archive"
            / "camera_ready"
            / "recent_predictors"
            / "gensco"
            / f"standalone_{args.dataset}.json"
        )

    if args.official_npz is not None and args.official_npz.is_file():
        summary["official_cross_check"] = compare_with_official(summary["rows"], args.official_npz)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"=== GenSCO {summary['mode']} {args.dataset}: drop = {summary['mean_drop']:.3f}% "
        f"± {summary['std_drop']:.3f}% | time = {summary['mean_time_s']:.2f}s/graph "
        f"| total {summary['total_time_s']:.1f}s ===",
        flush=True,
    )
    print(f"  written {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
