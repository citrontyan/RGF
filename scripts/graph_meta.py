"""读取一个图 npz 的 |V| / |E| / 平均度，输出 JSON（供 run_unit.py 记录到 run_record）。

用法（必须在忠实 WSL 环境里跑，因为 rgf.datasets.io 依赖 scipy）：

    wsl -e bash scripts/run_wsl.sh scripts/graph_meta.py /mnt/e/data/rgf/dimacs10/luxembourg.osm.npz

输出（stdout，单行 JSON）：

    {"path": "...", "n": 114599, "m": 119666, "avg_degree": 2.09}

只读，不写任何文件。文件 sha256 由 run_unit.py 用 stdlib 直接算，不在这里做。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgf.datasets.io import load_graph_npz  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("用法: graph_meta.py <graph.npz>", file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.is_file():
        print(json.dumps({"path": str(path), "error": "not_found"}, ensure_ascii=False))
        return 1
    g = load_graph_npz(path)
    print(
        json.dumps(
            {
                "path": str(path),
                "n": int(g.n),
                "m": int(g.m),
                "avg_degree": (2.0 * g.m / g.n) if g.n else None,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
