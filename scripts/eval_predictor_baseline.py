"""Evaluate one predictor as a plain MIS baseline on RB-[200-300] (Table row).

Uses the precomputed KaMIS labels (``*.label.npz``) as the reference, exactly
like the drop metric of the paper: ``drop = (ref - ours) / ref``.

    set CUDA_VISIBLE_DEVICES=1
    .venv\\Scripts\\python.exe scripts\\eval_predictor_baseline.py --graphs 20 --predictor fastt2t
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgf.datasets.io import load_graph_npz  # noqa: E402

DEFAULT_DATA_ROOT = Path(r"E:\data\rgf")


def greedy_solution(g, probs):
    chosen = np.zeros(g.n, dtype=np.int8)
    blocked = np.zeros(g.n, dtype=bool)
    for v in np.argsort(-probs, kind="stable"):
        if blocked[v]:
            continue
        chosen[v] = 1
        blocked[v] = True
        blocked[g.neighbors(v)] = True
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--graphs", type=int, default=20)
    ap.add_argument("--predictor", choices=["fastt2t", "difusco"], default="fastt2t")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--no-guided", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.predictor == "fastt2t":
        from rgf.predictor.fastt2t import FastT2TPredictor

        predictor = FastT2TPredictor(
            args.data_root / "checkpoints" / "fastt2t" / "mis_rb.ckpt",
            inference_diffusion_steps=args.steps,
            hidden_dim=args.hidden_dim,
            guided=not args.no_guided,
            rewrite_ratio=0.3,
            c1=2.0,
            c2=2.0,
            seed=args.seed,
        )
    else:
        from rgf.predictor.difusco import DifuscoPredictor

        predictor = DifuscoPredictor(
            args.data_root / "checkpoints" / "difusco" / "mis_er_gaussian.ckpt",
            inference_diffusion_steps=args.steps,
            seed=args.seed,
        )

    test_dir = args.data_root / "rb" / "rb200_300" / "test"
    files = [f for f in sorted(test_dir.glob("*.npz"))
             if not f.name.endswith((".label.npz", ".dsref.npz", ".opt.npz"))][: args.graphs]
    sizes, refs, drops, times = [], [], [], []
    for f in files:
        g = load_graph_npz(f)
        label_file = f.with_suffix(".label.npz")
        if not label_file.exists():
            continue
        with np.load(label_file) as data:
            ref = int(data["labels"].sum())
        t0 = time.perf_counter()
        probs = predictor.predict([(g, None)])[0]
        elapsed = time.perf_counter() - t0
        size = int(greedy_solution(g, probs).sum())
        sizes.append(size)
        refs.append(ref)
        drops.append((ref - size) / ref * 100)
        times.append(elapsed)
    print(
        f"{args.predictor} (steps={args.steps}, guided={not args.no_guided}) on {len(sizes)} graphs: "
        f"mean size={np.mean(sizes):.2f}  mean KaMIS={np.mean(refs):.2f}  "
        f"mean drop={np.mean(drops):.2f}%  mean time={np.mean(times):.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
