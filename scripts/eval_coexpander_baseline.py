"""Standalone COExpander baseline on the RGF synthetic MIS test sets.

Runs the *official* COExpander MIS inference path for the paper's main MIS table
(``COExpander (S=1, Ds=20, Is=1)``, no RLSA; the configuration of
``ML4CO-Bench-101/test_scripts/co_expander/co_expander_mis.py``, the official
test script shipped with these weights) on the first N graphs of
RB-[200-300] or ER-[700-800], using the reference KaMIS labels stored next to
each graph (``*.label.npz``), and reports

    drop = (|KaMIS| - |S|) / |KaMIS| * 100      (mean over graphs)
    wall clock per graph (batch=1, includes official data processing +
    inference + decoding, CUDA synchronised)

The script must run in the ``coexpander`` conda env (torch >= 2.0 + the
torch-scatter/torch-sparse builds + ml4co-kit), NOT in ``difusco``:

    COEXPANDER_ROOT=/mnt/d/repos/COExpander \\
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/d/repos/rgmis/experiments \\
    /home/pengb/miniconda3/envs/coexpander/bin/python \\
        scripts/eval_coexpander_baseline.py --dataset rb --limit 20 --out ...
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/mnt/e/data/rgf") if Path("/mnt/e").exists() else Path(r"E:\data\rgf")
DATASETS = {"rb": "rb/rb200_300/test", "er": "er/er700_800/test"}
WEIGHTS = {
    "rb": "weights/coexpander_mis_rb-small_sparse.pt",
    "er": "weights/coexpander_mis_er-700-800_sparse.pt",
}


def _load_predictor_module():
    """Import ``rgf/predictor/coexpander.py`` by path (avoids rgf.predictor package)."""
    path = REPO_ROOT / "rgf" / "predictor" / "coexpander.py"
    spec = importlib.util.spec_from_file_location("_coexpander_adapter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_coexpander_adapter"] = module
    spec.loader.exec_module(module)
    return module


def _clips(test_dir: Path, limit: int) -> list[Path]:
    clips = [f for f in sorted(test_dir.glob("*.npz"))
             if not f.name.endswith((".label.npz", ".dsref.npz", ".opt.npz"))]
    return clips[:limit] if limit else clips


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--root", type=Path,
                    default=Path(os.environ.get("COEXPANDER_ROOT", "/mnt/d/repos/COExpander")))
    ap.add_argument("--seed", type=int, default=1234, help="official solver seed")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    import torch
    from co_expander import COExpanderMISSolver
    from ml4co_kit import MISGraphData

    adapter = _load_predictor_module()
    weight_path = args.root / WEIGHTS[args.dataset]
    if not weight_path.exists():
        raise FileNotFoundError(weight_path)

    test_dir = args.data_root / DATASETS[args.dataset]
    clips = _clips(test_dir, args.limit)
    if not clips:
        raise FileNotFoundError(f"no graphs under {test_dir}")

    print(f"standalone COExpander | dataset={args.dataset} | {len(clips)} graphs | "
          f"weights={weight_path}", flush=True)

    # ---- data (loading is NOT timed; mirrors from_txt in the official script) --
    graph_data = []
    refs = []
    records = []
    for clip in clips:
        with np.load(clip) as d:
            n = int(d["n"])
            edges = np.asarray(d["edges"], dtype=np.int64)
        with np.load(clip.with_suffix(".label.npz")) as d:
            ref = int(np.asarray(d["labels"]).sum())
        # official convention: both directions + one self-loop per vertex
        loops = np.arange(n, dtype=np.int64)
        both = np.concatenate(
            [edges.T, edges[:, ::-1].T, np.stack([loops, loops], axis=0)], axis=1
        )
        gd = MISGraphData()
        gd.nodes_num = n
        gd.edge_index = both
        graph_data.append(gd)
        refs.append(ref)
        records.append({"graph": clip.stem, "n": n, "m": int(edges.shape[0]), "ref": ref})

    # ---- official model + solver (identical config to test_cm_mis.py) ----------
    model = adapter._worker_build_model(str(weight_path), args.device)
    solver = COExpanderMISSolver(model=model, seed=args.seed)

    def solve_one(idx: int) -> int:
        solver.graph_data = [graph_data[idx]]
        solver.solve(batch_size=1, sampling_num=adapter._OFFICIAL["sampling_num"],
                     show_time=False)
        return int(np.asarray(solver.graph_data[0].nodes_label).sum())

    # ---- warmup (CUDA init + cuDNN autotune), excluded from the mean ----------
    for w in range(min(args.warmup, len(graph_data))):
        solve_one(w)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ---- timed pass -----------------------------------------------------------
    for i, rec in enumerate(records):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        size = solve_one(i)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rec["size"] = size
        rec["drop"] = (rec["ref"] - size) / rec["ref"] * 100.0
        rec["time_s"] = dt
        print(f"  [{i + 1}/{len(records)}] {rec['graph']}: n={rec['n']} m={rec['m']} "
              f"|S|={size} ref={rec['ref']} drop={rec['drop']:.2f}% t={dt * 1000:.1f} ms",
              flush=True)

    drops = np.array([r["drop"] for r in records], dtype=float)
    times = np.array([r["time_s"] for r in records], dtype=float)
    result = {
        "method": "COExpander (official code, official MIS weights)",
        "dataset": args.dataset,
        "source": {
            "code": "https://github.com/Thinklab-SJTU/COExpander",
            "weights": "https://huggingface.co/ML4CO/ML4CO-Bench-101 (co_expander/*.pt); "
                       "Google Drive link from the official README was unreachable from this host",
            "weight_file": str(weight_path),
            "weight_sha256": _sha256(weight_path),
        },
        "caliber": {
            "inference_steps": adapter._OFFICIAL["inference_steps"],
            "determinate_steps": adapter._OFFICIAL["determinate_steps"],
            "sampling_num": adapter._OFFICIAL["sampling_num"],
            "beam_size": adapter._OFFICIAL["beam_size"],
            "use_rlsa": False,
            "batch_size": 1,
            "seed": args.seed,
            "timing": "official data processing + inference + decode; per graph; "
                      "CUDA synchronised; model load and data loading excluded; "
                      f"{args.warmup} warmup graphs excluded",
        },
        "env": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "ml4co_kit": __import__("ml4co_kit").__version__
            if hasattr(__import__("ml4co_kit"), "__version__") else "unknown",
            "numpy": np.__version__,
        },
        "n_graphs": len(records),
        "ref_mean": float(np.mean([r["ref"] for r in records])),
        "size_mean": float(np.mean([r["size"] for r in records])),
        "drop_mean": float(drops.mean()),
        "drop_std": float(drops.std(ddof=1)) if len(drops) > 1 else 0.0,
        "time_mean_s": float(times.mean()),
        "time_median_s": float(np.median(times)),
        "time_std_s": float(times.std(ddof=1)) if len(times) > 1 else 0.0,
        "total_inference_time_s": float(times.sum()),
        "graphs": records,
        "command": " ".join([sys.executable] + sys.argv),
    }
    print(f"=== {args.dataset}: drop = {result['drop_mean']:.2f}% ± {result['drop_std']:.2f} | "
          f"size = {result['size_mean']:.2f} | ref = {result['ref_mean']:.2f} | "
          f"time = {result['time_mean_s'] * 1000:.1f} ms/graph ===", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  written {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
