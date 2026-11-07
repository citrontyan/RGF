"""汇总大图表（论文 Table 3 / tab:ex）：读 results/large/*.json 打印对照表。

    .venv\\Scripts\\python.exe scripts\\summarize_large_table.py [结果目录]
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

# 论文 Table 3 的原值（RGF drop, RW drop, time）
PAPER = {
    "citationCiteseer": (4.76, 6.78, "6m21s"),
    "coAuthorsCiteseer": (6.25, 7.17, "6m21s"),
    "coAuthorsDBLP": (4.39, 5.33, "6m19s"),
    "coPapersCiteseer": (5.11, 5.93, "7m3s"),
    "coPapersDBLP": (5.78, 8.06, "10m36s"),
    "asia.osm": (8.23, 9.15, "1h15m"),
    "belgium.osm": (7.75, 7.79, "46m52s"),
    "germany.osm": (7.39, 10.15, "1h20m"),
    "great-britain.osm": (6.98, 10.10, "1h5m"),
    "italy.osm": (7.06, 9.22, "1h6m"),
    "luxembourg.osm": (4.24, 6.90, "7m10s"),
    "netherlands.osm": (8.13, 11.13, "57m31s"),
}


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/e/data/rgf/results/large")
    rows = []
    for p in sorted(glob.glob(str(Path(root) / "*.json"))):
        rows.append(json.loads(Path(p).read_text(encoding="utf-8")))
    if not rows:
        print(f"{root} 下没有结果", file=sys.stderr)
        return 2
    print(f"{'实例':24s} {'|V|':>10s} {'|E|':>11s} {'β':>7s} | "
          f"{'RGF':>7s} {'RW':>7s} | {'论文RGF':>7s} {'论文RW':>7s} | {'我们耗时':>8s} {'论文耗时':>8s}")
    print("-" * 110)
    for j in rows:
        inst = j["instance"]
        pr = PAPER.get(inst, ("-", "-", "-"))
        print(f"{inst:24s} {j['n']:10,d} {j['m']:11,d} {j['beta']:7,d} | "
              f"{j['rgf_drop']:6.2f}% {j['rw_drop']:6.2f}% | "
              f"{pr[0] if isinstance(pr[0], str) else f'{pr[0]:.2f}%':>7s} "
              f"{pr[1] if isinstance(pr[1], str) else f'{pr[1]:.2f}%':>7s} | "
              f"{j['rgf_time_s']/60:6.1f}m {pr[2]:>8s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
