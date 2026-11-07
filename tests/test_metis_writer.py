"""Guard: the METIS text writer must stay byte-identical to the naive version.

The vectorised writer in :mod:`rgf.partition.metis` is an optimisation only; if
its output ever drifts from the straightforward per-vertex join, METIS would be
fed a different graph and every downstream number would move.
"""

from __future__ import annotations

import numpy as np
import pytest

from rgf.graph import Graph
from rgf.partition.metis import _write_metis_graph


def _naive(g: Graph) -> bytes:
    lines = [f"{g.n} {g.m}"]
    for v in range(g.n):
        nb = g.neighbors(v)
        lines.append("" if nb.size == 0 else " ".join(str(int(u) + 1) for u in nb))
    return ("\n".join(lines) + "\n").encode("ascii")


def _rb_like(n: int, seed: int) -> Graph:
    rng = np.random.default_rng(seed)
    iu, ju = np.triu_indices(n, k=1)
    mask = rng.random(iu.size) < 0.06
    return Graph.from_edges(n, np.stack([iu[mask], ju[mask]], axis=1))


@pytest.mark.parametrize(
    "g",
    [
        Graph.from_edges(6, np.empty((0, 2), dtype=np.int64)),
        Graph.from_edges(7, [[0, 1], [1, 2], [2, 3], [3, 0]]),
        Graph.from_edges(4, [[0, 1]]),
        _rb_like(300, 4),
    ],
    ids=["all-isolated", "isolated-tail", "small", "rb-300"],
)
def test_writer_matches_naive_output(g: Graph, tmp_path) -> None:
    got = tmp_path / "got.graph"
    _write_metis_graph(g, str(got))
    assert got.read_bytes() == _naive(g)
