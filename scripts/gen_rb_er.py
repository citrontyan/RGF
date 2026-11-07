"""Generate the synthetic benchmarks of the paper (RB-[200-300], ER-[700-800]).

Datasets are written under the data root (default ``E:\\data\\rgf``); nothing is
written inside the repository (``docs/decisions.md`` items D-1, D-2).

    .venv\\Scripts\\python.exe scripts\\gen_rb_er.py --rb-test 500 --er-test 128

The ER evaluation set used by DIFUSCO / Fast T2T is the DIMES ER-[700-800] set
shipped inside ``third_party/Fast-T2T/data/mis/er/er_700_800_test``; ``--import-er``
moves a copy of it under the data root instead of resampling.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgf.datasets.io import save_graph_npz  # noqa: E402
from rgf.datasets.synthetic import er_graph_in_range, rb_graph_in_range  # noqa: E402

DEFAULT_DATA_ROOT = Path(r"E:\data\rgf")
REPO_ROOT = Path(__file__).resolve().parents[1]


def write_dataset(name: str, out_dir: Path, count: int, sampler) -> dict:
    """Stream graphs to disk one by one (never holding the whole set in RAM)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"name": name, "count": 0, "graphs": []}
    ns_min, ns_max = None, None
    for i in range(count):
        g = sampler(i)
        path = out_dir / f"{name}_{i}.npz"
        save_graph_npz(g, path)
        n = int(g.n)
        manifest["graphs"].append({"file": path.name, "n": n, "m": int(g.m)})
        ns_min = n if ns_min is None else min(ns_min, n)
        ns_max = n if ns_max is None else max(ns_max, n)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{count} generated", flush=True)
    manifest["count"] = len(manifest["graphs"])
    manifest["n_min"] = ns_min
    manifest["n_max"] = ns_max
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rb-test", type=int, default=500, help="number of RB-[200-300] test graphs")
    ap.add_argument("--rb-train", type=int, default=0, help="number of RB-[200-300] training graphs")
    ap.add_argument("--er-test", type=int, default=0, help="number of ER-[700-800] graphs to sample")
    ap.add_argument("--er-train", type=int, default=0, help="number of ER-[700-800] training graphs")
    ap.add_argument("--import-er", action="store_true", help="copy the DIMES ER-[700-800] set from third_party")
    args = ap.parse_args()

    root: Path = args.data_root
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()

    if args.rb_test or args.rb_train:
        for split, count in (("test", args.rb_test), ("train", args.rb_train)):
            if not count:
                continue
            out = root / "rb" / "rb200_300" / split
            manifest = write_dataset(
                "RB_200_300", out, count, lambda _i: rb_graph_in_range(200, 300, rng)
            )
            print(f"[rb/{split}] {manifest['count']} graphs  |V| in [{manifest['n_min']}, {manifest['n_max']}]  -> {out}")

    if args.er_train:
        out = root / "er" / "er700_800" / "train"
        manifest = write_dataset(
            "ER_700_800_0.15", out, args.er_train,
            lambda _i: er_graph_in_range(700, 800, 0.15, rng),
        )
        print(f"[er/train] {manifest['count']} graphs  |V| in [{manifest['n_min']}, {manifest['n_max']}]  -> {out}")

    if args.er_test:
        out = root / "er" / "er700_800" / "test"
        manifest = write_dataset(
            "ER_700_800_0.15", out, args.er_test,
            lambda _i: er_graph_in_range(700, 800, 0.15, rng),
        )
        print(f"[er/test] {manifest['count']} graphs  |V| in [{manifest['n_min']}, {manifest['n_max']}]  -> {out}")

    if args.import_er:
        # The bundled DIMES files are copied out of the third_party clone into the
        # data root (datasets must not live inside the repository), so look in
        # both places.
        candidates = [
            args.data_root / "dimes" / "fastt2t_mis" / "er" / "er_700_800_test",
            REPO_ROOT / "third_party" / "Fast-T2T" / "data" / "mis" / "er" / "er_700_800_test",
        ]
        src = next((c for c in candidates if c.exists()), None)
        dst = root / "dimes" / "er_700_800_test"
        if src is None:
            print(f"[import-er] source missing; looked in {[str(c) for c in candidates]}", file=sys.stderr)
            return 2
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(src.glob("*.gpickle"))
        for f in files:
            shutil.copy2(f, dst / f.name)
        print(f"[import-er] copied {len(files)} gpickle files -> {dst}")

    print(f"done in {time.perf_counter() - t0:.1f}s; data root = {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
