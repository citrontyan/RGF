"""Batch KaMIS labeling for a dataset directory (parallel).

Phase 1 (Windows/Python): convert every ``*.npz`` graph of a dataset into a
METIS ``.graph`` file under ``<dataset>/_labels_work``.
Phase 2 (single WSL invocation): run ``redumis`` over all of them with
``xargs -P <workers>``.
Phase 3 (Windows/Python): parse each ``*.is`` output (a 0/1 indicator vector)
into ``<stem>.label.npz``.  Resumable: finished labels are skipped.

    .venv\\Scripts\\python.exe scripts\\label_dataset.py ^
        --dataset E:\\data\\rgf\\rb\\rb200_300\\train --time-limit 5 --workers 30
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgf.baselines.kamis import find_kamis, to_wsl_path  # noqa: E402
from rgf.datasets.io import load_graph_npz  # noqa: E402
from rgf.graph import Graph  # noqa: E402

_LABEL_SCRIPT = r"""#!/usr/bin/env bash
# usage: label.sh <redumis> <workdir> <time_limit> <seed> <workers>
#
# One worker = one graph = one thread: KaMIS (redumis) is OpenMP-linked and would
# otherwise size its thread pool to the whole machine (64 logical CPUs here), so
# the pool is pinned to a single thread and the parallelism comes from xargs.
set -u
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
EXE="$1"; DIR="$2"; TL="$3"; SEED="$4"; P="$5"
ls "$DIR"/*.graph | xargs -P "$P" -I{} sh -c '
  g="{}"; out="{}.is"
  if [ -s "$out" ]; then exit 0; fi
  '"$EXE"' "$g" --output="$out" --seed='"$SEED"' --time_limit='"$TL"' >/dev/null 2>&1 \
    || echo "FAIL $g"
'
"""


def write_metis_graph(g: Graph, path: Path) -> None:
    lines = [f"{g.n} {g.m}"]
    for v in range(g.n):
        nb = g.neighbors(v)
        lines.append(" ".join(str(int(u) + 1) for u in nb) if nb.size else "")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True, help="directory of *.npz graphs")
    ap.add_argument("--time-limit", type=float, default=5.0, help="KaMIS time limit per graph (s)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=30, help="parallel workers (= threads, see label.sh)")
    ap.add_argument("--limit", type=int, default=0, help="only label the first N files (0 = all)")
    ap.add_argument("--chunk", type=int, default=4000, help="graphs per batch (bounds disk use)")
    args = ap.parse_args()

    exe = find_kamis()
    if not exe:
        print("KaMIS binary not found", file=sys.stderr)
        return 2
    if exe.lower().endswith(".exe"):
        print("only the WSL KaMIS build is supported by this batch script", file=sys.stderr)
        return 2

    files = [f for f in sorted(args.dataset.glob("*.npz"))
             if not f.name.endswith((".label.npz", ".dsref.npz", ".opt.npz"))]
    if args.limit:
        files = files[: args.limit]
    work = args.dataset / "_labels_work"
    work.mkdir(exist_ok=True)

    pending = [f for f in files if not f.with_suffix(".label.npz").exists()]
    print(
        f"{len(files)} graphs, {len(pending)} to label "
        f"(workers={args.workers}, chunk={args.chunk}, tl={args.time_limit}s)",
        flush=True,
    )
    if not pending:
        return 0

    script = work / "label.sh"
    # write_bytes: this script must keep LF endings (write_text would emit CRLF
    # on Windows and bash would choke on the trailing \r)
    script.write_bytes(_LABEL_SCRIPT.encode("utf-8"))

    total_done = total_failed = 0
    for start in range(0, len(pending), args.chunk):
        chunk = pending[start : start + args.chunk]
        # ---- phase 1: METIS graphs for this chunk
        for f in chunk:
            meta = work / (f.stem + ".graph")
            if not meta.exists():
                write_metis_graph(load_graph_npz(f), meta)
        # ---- phase 2: one WSL call, xargs-parallel redumis
        cmd = [
            "wsl", "-e", "bash", to_wsl_path(script),
            to_wsl_path(exe), to_wsl_path(work), str(args.time_limit), str(args.seed), str(args.workers),
        ]
        subprocess.run(cmd, check=True)
        # ---- phase 3: parse indicators, then drop the intermediates
        done = failed = 0
        for f in chunk:
            out = work / (f.stem + ".graph.is")
            meta = work / (f.stem + ".graph")
            if not out.exists():
                failed += 1
                continue
            g = load_graph_npz(f)
            values = np.loadtxt(out, dtype=np.int64).reshape(-1)
            if values.size == g.n and set(np.unique(values)).issubset({0, 1}):
                labels = values.astype(np.int8)
            else:
                labels = np.zeros(g.n, dtype=np.int8)
                labels[np.clip(values, 0, g.n - 1)] = 1
            np.savez_compressed(f.with_suffix(".label.npz"), n=np.int64(g.n), labels=labels)
            out.unlink(missing_ok=True)
            meta.unlink(missing_ok=True)
            done += 1
        total_done += done
        total_failed += failed
        print(f"  chunk done: labeled {done}, failed {failed} | total {total_done}/{len(pending)}", flush=True)
    print(f"labeled {total_done}, failed {total_failed}")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
