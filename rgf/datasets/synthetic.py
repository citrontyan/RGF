"""Synthetic graph families of the paper (RB and ER).

The paper (§Dataset) says:

    "we use RB graphs and ER graphs for our experiments.  RB graphs are
     uniformly sampled with 200--300 vertices.  ER graphs are generated with
     700--800 vertices and a connection probability of 0.15."

Neither generator is spelled out in the paper; both are pinned by the data
pipeline of the works it cites (``docs/decisions.md`` items D-1, D-2):

* **ER.** ``ErdosRenyi`` in DIMES' ``data_generation/random_graph.py`` (vendored
  in Fast T2T, ``third_party/Fast-T2T/data/mis-benchmark-framework/``):
  ``n ~ Uniform{min_n..max_n}`` then ``G(n, p)``.  The paper's settings are
  ``min_n=700``, ``max_n=800``, ``p=0.15``, which is exactly the ER-[700-800]
  benchmark used by DIFUSCO / Fast T2T.
* **RB.** ``rb_graph/rbgraph_generator.py`` + ``rb_graph/xu_util.py`` of the same
  framework ("From Erdos goes neural paper"), invoked for RB-[200-300] with
  ``--graph_type small``.  The construction builds ``n_a`` groups of ``k``
  vertices, makes each group a clique (the NAND clauses inside a group), and adds
  inter-group edges by repeatedly picking two groups and sampling ``s`` vertex
  pairs; isolated vertices are removed and the graph is kept only when its vertex
  count lands in ``[200, 300]``.

Both functions are re-implementations of those published generators (we do not
call the vendored scripts); the RNG stream differs, the distribution does not.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from rgf.graph import Graph


# --------------------------------------------------------------------- ER
def er_graph(n: int, p: float, rng: np.random.Generator) -> Graph:
    """Erdos-Renyi ``G(n, p)``."""
    if n < 2:
        return Graph.from_edges(n, np.empty((0, 2), dtype=np.int64))
    iu, ju = np.triu_indices(n, k=1)
    mask = rng.random(iu.size) < p
    edges = np.stack([iu[mask], ju[mask]], axis=1).astype(np.int64)
    return Graph.from_edges(n, edges)


def er_graph_in_range(n_low: int, n_high: int, p: float, rng: np.random.Generator) -> Graph:
    """ER with the vertex count drawn uniformly from ``[n_low, n_high]`` (paper: 700--800)."""
    n = int(rng.integers(n_low, n_high + 1))
    return er_graph(n, p, rng)


# --------------------------------------------------------------------- RB
@dataclass(frozen=True)
class RBParams:
    """Parameters of the RB draw, kept for provenance."""

    na: int
    k: int
    p: float


def _rb_raw(na: int, k: int, p: float, rng: np.random.Generator) -> Graph:
    """One RB draw before isolate removal, mirroring ``xu_util.generate_instance``."""
    n = na
    v = k * n
    a = math.log(k) / math.log(n)
    s = int(p * (n ** (2 * a)))
    re = -a / math.log(1 - p)
    iterations = int(re * n * math.log(n) - 1)

    parts = np.arange(v).reshape(n, k)
    edges: set[tuple[int, int]] = set()
    # NAND clauses inside every group -> cliques
    for part in parts:
        edges.update(itertools.combinations([int(x) for x in part], 2))
    # inter-group edges
    for _ in range(max(iterations, 0)):
        i, j = rng.choice(n, 2, replace=False)
        candidates = [pair for pair in itertools.product(parts[i], parts[j]) if pair not in edges]
        if not candidates:
            continue
        take = min(s, len(candidates))
        if take <= 0:
            continue
        idx = rng.choice(len(candidates), size=take, replace=False)
        edges.update(candidates[int(t)] for t in idx)
    arr = np.array(sorted(edges), dtype=np.int64).reshape(-1, 2)
    return Graph.from_edges(v, arr)


def _drop_isolates(g: Graph) -> Graph:
    deg = g.degrees()
    keep = np.flatnonzero(deg > 0)
    sub, _, _ = g.induced(keep)
    return sub


def rb_graph(
    rng: np.random.Generator,
    graph_type: str = "small",
    max_draws: int = 10_000,
) -> tuple[Graph, RBParams]:
    """RB graph with ``200--300`` vertices (``graph_type="small"``).

    Mirrors ``rbgraph_generator.py``: draw, drop isolates, accept only when the
    vertex count is inside the range.
    """
    if graph_type == "small":
        n_low, n_high = 200, 300
    elif graph_type == "large":
        n_low, n_high = 800, 1200
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown graph_type {graph_type!r}")

    for _ in range(max_draws):
        na = int(rng.integers(20, 25)) if graph_type == "small" else int(rng.integers(40, 55))
        k = int(rng.integers(5, 12)) if graph_type == "small" else int(rng.integers(20, 25))
        p = float(rng.uniform(0.3, 1.0))
        raw = _rb_raw(na, k, p, rng)
        g = _drop_isolates(raw)
        if n_low <= g.n <= n_high:
            return g, RBParams(na=na, k=k, p=p)
    raise RuntimeError(f"RB generator failed to land in [{n_low}, {n_high}] within {max_draws} draws")


def rb_graph_in_range(n_low: int, n_high: int, rng: np.random.Generator, max_draws: int = 10_000) -> Graph:
    """RB graph with the vertex count inside ``[n_low, n_high]`` (paper: 200--300)."""
    for _ in range(max_draws):
        na = int(rng.integers(20, 25))
        k = int(rng.integers(5, 12))
        p = float(rng.uniform(0.3, 1.0))
        g = _drop_isolates(_rb_raw(na, k, p, rng))
        if n_low <= g.n <= n_high:
            return g
    raise RuntimeError(f"RB generator failed to land in [{n_low}, {n_high}] within {max_draws} draws")
