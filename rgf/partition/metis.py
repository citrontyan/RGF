"""METIS partitioning (paper §Experimental Settings).

METIS is a C program; no Windows wheel of ``pymetis`` exists on PyPI, so this
wrapper drives the ``gpmetis`` executable built under ``third_party/metis`` (or
found on ``PATH``) and parses its partition file.  See ``docs/decisions.md``
item D-7 for the build recipe and the balance parameters used.

METIS' *reference manual* defines the objective (edge cut minimisation subject
to a balance constraint); we keep METIS defaults and only expose the number of
parts (2) and the random seed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from rgf.graph import Graph
from rgf.partition.base import Cut, Partitioner
from rgf.wsl import wrap_command

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUILD_CANDIDATES = [
    _REPO_ROOT / "third_party" / "METIS" / "build-wsl" / "programs" / "gpmetis",
    _REPO_ROOT / "third_party" / "METIS" / "build-vs" / "programs" / "Release" / "gpmetis.exe",
    _REPO_ROOT / "third_party" / "METIS" / "build" / "programs" / "gpmetis.exe",
    _REPO_ROOT / "third_party" / "METIS" / "gpmetis.exe",
]


def find_gpmetis() -> str | None:
    env = os.environ.get("GPMETIS")
    if env and Path(env).exists():
        return env
    for cand in _BUILD_CANDIDATES:
        if cand.exists():
            return str(cand)
    return shutil.which("gpmetis")


class MetisPartitioner(Partitioner):
    name = "metis"

    def __init__(self, nparts: int = 2, seed: int | None = None, executable: str | None = None):
        self.nparts = nparts
        self.seed = seed
        self.executable = executable or find_gpmetis()

    def partition(self, g: Graph, rng: np.random.Generator) -> Cut:
        if g.n < 2:
            return self.empty_cut(g)
        if g.m == 0:
            # No edges: fall back to a balanced random split (there is no cut edge).
            side = np.zeros(g.n, dtype=np.int8)
            side[rng.permutation(g.n)[: g.n // 2]] = 1
            return self.cut_from_assignment(g, side)
        if not self.executable:
            raise FileNotFoundError(
                "gpmetis not found; build METIS under third_party/metis or set GPMETIS "
                "(see docs/decisions.md item D-7)"
            )
        with tempfile.TemporaryDirectory(prefix="rgf_metis_") as tmp:
            graph_file = os.path.join(tmp, "g.graph")
            _write_metis_graph(g, graph_file)
            args = [graph_file, str(self.nparts)]
            if self.seed is not None:
                args.append(f"-seed={self.seed}")
            cmd = wrap_command(self.executable, args, path_args={0})
            subprocess.run(cmd, check=True, capture_output=True)
            part_file = graph_file + f".part.{self.nparts}"
            side = np.loadtxt(part_file, dtype=np.int64)
        return self.cut_from_assignment(g, side.astype(np.int8))


def _write_metis_graph(g: Graph, path: str) -> None:
    """Write the METIS adjacency format (1-based ids, no self loops, no duplicates)."""
    deg = g.degrees()
    lines = [f"{g.n} {g.m}"]
    for v in range(g.n):
        nb = g.neighbors(v)
        if nb.size == 0:
            lines.append("")
        else:
            lines.append(" ".join(str(int(u) + 1) for u in nb))
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")
