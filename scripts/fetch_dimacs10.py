"""Download the 12 real-world DIMACS10 instances of the paper's Table 1.

Source (verified 2026-09-17): the 10th DIMACS Implementation Challenge archive,
``https://sites.cc.gatech.edu/dimacs10/downloads.shtml``, categories *Co-author and
Citation Networks* (``archive/coauthor.shtml``) and *Street Networks*
(``archive/streets.shtml``).  The ``n``/``m`` recorded below are the ones printed
on those pages and match the paper's Table 1 exactly.

Files are bzip2-compressed METIS-format graphs.  Each one is decompressed,
converted to the repository's ``.npz`` edge cache and checked against the
expected ``|V|`` and ``|E|``; nothing is written inside the repository.

    .venv\\Scripts\\python.exe scripts\\fetch_dimacs10.py --data-root E:\\data\\rgf
    .venv\\Scripts\\python.exe scripts\\fetch_dimacs10.py --only asia.osm
"""

from __future__ import annotations

import argparse
import bz2
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgf.datasets.io import save_graph_npz  # noqa: E402
from rgf.graph import Graph  # noqa: E402

BASE = "https://sites.cc.gatech.edu/dimacs10/archive/data"
DEFAULT_DATA_ROOT = Path(r"E:\data\rgf")

INSTANCES: dict[str, dict] = {
    # Co-author and citation networks
    "citationCiteseer": {"group": "coauthor", "n": 268_495, "m": 1_156_647},
    "coAuthorsCiteseer": {"group": "coauthor", "n": 227_320, "m": 814_134},
    "coAuthorsDBLP": {"group": "coauthor", "n": 299_067, "m": 977_676},
    "coPapersCiteseer": {"group": "coauthor", "n": 434_102, "m": 16_036_720},
    "coPapersDBLP": {"group": "coauthor", "n": 540_486, "m": 15_245_729},
    # Street networks
    "asia.osm": {"group": "streets", "n": 11_950_757, "m": 12_711_603},
    "belgium.osm": {"group": "streets", "n": 1_441_295, "m": 1_549_970},
    "germany.osm": {"group": "streets", "n": 11_548_845, "m": 12_369_181},
    "great-britain.osm": {"group": "streets", "n": 7_733_822, "m": 8_156_517},
    "italy.osm": {"group": "streets", "n": 6_686_493, "m": 7_013_978},
    "luxembourg.osm": {"group": "streets", "n": 114_599, "m": 119_666},
    "netherlands.osm": {"group": "streets", "n": 2_216_688, "m": 2_441_238},
    # 论文 Table 3 没用、但同源同格式的更大路网（2026-10-05 加入，见 D-29）：
    # 官方 streets 页给出 n=50,912,018 / m=54,054,660，是 asia 的 4.3 倍
    "europe.osm": {"group": "streets", "n": 50_912_018, "m": 54_054_660},
}


def url_for(name: str) -> str:
    return f"{BASE}/{INSTANCES[name]['group']}/{name}.graph.bz2"


def download(name: str, dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[{name}] already downloaded: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    url = url_for(name)
    print(f"[{name}] GET {url}")
    t0 = time.perf_counter()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as resp, open(tmp, "wb") as fh:
        total = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            total += len(chunk)
            if total % (64 << 20) < (1 << 20):
                print(f"[{name}]   {total/1e6:8.1f} MB", flush=True)
    tmp.replace(dest)
    print(f"[{name}] saved {dest} ({dest.stat().st_size/1e6:.1f} MB) in {time.perf_counter()-t0:.0f}s")
    return dest


def _next_data_line(fh) -> str | None:
    """Next line that is not a DIMACS10 comment (``%``); blank lines are data
    lines here, since a vertex with no neighbour is written as an empty line."""
    while True:
        line = fh.readline()
        if not line:
            return None
        if line.lstrip().startswith("%"):
            continue
        return line


def convert(name: str, src: Path, out_dir: Path, force: bool = False) -> dict:
    """Parse the METIS-format graph and cache it as an ``.npz`` edge list."""
    npz = out_dir / f"{name}.npz"
    expected = INSTANCES[name]
    if npz.exists() and not force:
        g = None
        print(f"[{name}] already converted: {npz}")
        return {"name": name, "npz": str(npz), "skipped": True, **expected}
    t0 = time.perf_counter()
    with bz2.open(src, "rt") as fh:
        header = _next_data_line(fh).split()
        n, m = int(header[0]), int(header[1])
        # The DIMACS extension sets the third field to 100 when the file lists
        # edge entries (self loops / multi edges); then m is the entry count and
        # the edge count has to be derived from the parsed data.
        # Allocate for the worst case: sum of degrees is 2|E| for simple graphs,
        # and the DIMACS extension (third field 100) records the entry count.
        entries = 2 * m
        edges = np.empty((entries, 2), dtype=np.int64)
        pos = 0
        for v in range(n):
            line = _next_data_line(fh)
            if line is None:
                break
            nbrs = line.split()
            if not nbrs:
                continue
            arr = np.array(nbrs, dtype=np.int64) - 1
            end = pos + arr.size
            if end > entries:  # grow if the header understated the entry count
                edges = np.vstack([edges, np.empty((end - entries, 2), dtype=np.int64)])
                entries = edges.shape[0]
            edges[pos:end, 0] = v
            edges[pos:end, 1] = arr
            pos = end
        edges = edges[:pos]
    g = Graph.from_edges(n, edges)
    save_graph_npz(g, npz)
    ok = g.n == expected["n"] and g.m == expected["m"]
    print(
        f"[{name}] converted {g.n} vertices / {g.m} edges in {time.perf_counter()-t0:.0f}s "
        f"(expected {expected['n']} / {expected['m']}) {'OK' if ok else 'MISMATCH'}"
    )
    return {
        "name": name,
        "npz": str(npz),
        "n": int(g.n),
        "m": int(g.m),
        "expected_n": expected["n"],
        "expected_m": expected["m"],
        "matches_paper": bool(ok),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--only", nargs="*", default=None, help="subset of instance names")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-convert", action="store_true")
    args = ap.parse_args()

    names = args.only or list(INSTANCES)
    unknown = [n for n in names if n not in INSTANCES]
    if unknown:
        print(f"unknown instances: {unknown}", file=sys.stderr)
        return 2

    raw_dir = args.data_root / "dimacs10" / "raw"
    npz_dir = args.data_root / "dimacs10"
    manifest = []
    for name in names:
        try:
            path = download(name, raw_dir / f"{name}.graph.bz2", force=args.force)
            if not args.skip_convert:
                manifest.append(convert(name, path, npz_dir, force=args.force))
        except Exception as exc:  # keep going: the street networks are large
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            manifest.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})

    out = npz_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"manifest -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
