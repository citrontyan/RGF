"""Graph container.

The framework needs exactly three operations on a graph:

* size of the vertex set,
* neighbour iteration over an induced subgraph,
* connected components after vertices / edges have been removed.

Vertices are dense integers ``0 .. n-1``.  Every edge is stored once with
``u < v``; a CSR adjacency structure built with NumPy gives O(deg) neighbour
access without materialising a dense matrix, which matters because the paper
evaluates instances with up to 11.9M vertices (Table 1, ``asia.osm``).

No problem-specific logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components as _cc


def canonical_edges(edges) -> np.ndarray:
    """Return ``(m, 2)`` int64 edges with ``u < v``, self-loops dropped, unique.

    The rows are lexicographically sorted so that two graphs built from the same
    edge multiset are bit-identical.
    """
    e = np.asarray(edges, dtype=np.int64)
    if e.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    e = e.reshape(-1, 2)
    lo = np.minimum(e[:, 0], e[:, 1])
    hi = np.maximum(e[:, 0], e[:, 1])
    e = np.stack([lo, hi], axis=1)
    e = e[lo != hi]
    if e.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    # unique rows, sorted lexicographically
    order = np.lexsort((e[:, 1], e[:, 0]))
    e = e[order]
    keep = np.ones(len(e), dtype=bool)
    keep[1:] = np.any(e[1:] != e[:-1], axis=1)
    return e[keep]


@dataclass(frozen=True)
class Graph:
    """Undirected simple graph with CSR adjacency."""

    n: int
    edges: np.ndarray  # (m, 2) int64, u < v, unique, lexicographically sorted
    indptr: np.ndarray  # (n + 1,) int64
    indices: np.ndarray  # (2m,) int64, neighbour ids grouped by vertex

    # ------------------------------------------------------------------ build
    @staticmethod
    def from_edges(n: int, edges) -> "Graph":
        e = canonical_edges(edges)
        if e.size:
            if e.min() < 0 or e.max() >= n:
                raise ValueError("edge endpoint outside 0..n-1")
        src = np.concatenate([e[:, 0], e[:, 1]]) if e.size else np.empty(0, np.int64)
        dst = np.concatenate([e[:, 1], e[:, 0]]) if e.size else np.empty(0, np.int64)
        if src.size:
            order = np.lexsort((dst, src))
            indices = dst[order]
            counts = np.bincount(src, minlength=n)
        else:
            indices = np.empty(0, np.int64)
            counts = np.zeros(n, dtype=np.int64)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        return Graph(int(n), e, indptr, indices)

    # ------------------------------------------------------------- accessors
    @property
    def m(self) -> int:
        return int(self.edges.shape[0])

    def neighbors(self, v: int) -> np.ndarray:
        return self.indices[self.indptr[v] : self.indptr[v + 1]]

    def degrees(self) -> np.ndarray:
        return np.diff(self.indptr)

    def edge_mask_within(self, mask: np.ndarray) -> np.ndarray:
        """Boolean mask over ``edges`` selecting edges whose both endpoints are in ``mask``."""
        return mask[self.edges[:, 0]] & mask[self.edges[:, 1]]

    # --------------------------------------------------------------- derived
    def induced(self, vertices, edge_mask: np.ndarray | None = None) -> tuple["Graph", np.ndarray, np.ndarray]:
        """Induced subgraph on ``vertices`` (original ids).

        ``edge_mask`` optionally restricts which parent edges are kept (used to
        drop the cut edges for the DS variant, where no vertex is removed).

        Returns ``(sub, ids, pos)`` where ``ids`` are the original vertex ids in
        the order used by ``sub`` (``sub`` vertex ``i`` is original ``ids[i]``)
        and ``pos`` maps original id -> local id (``-1`` when absent).
        """
        ids = np.asarray(vertices, dtype=np.int64)
        return _induced(self, ids, edge_mask)

    def components(self, mask: np.ndarray | None = None) -> list[np.ndarray]:
        """Connected components of the (optionally induced) graph.

        ``mask`` selects original vertices to keep.  Returns a list of arrays of
        original vertex ids, ordered by their smallest vertex id.
        """
        if mask is None:
            mask = np.ones(self.n, dtype=bool)
        keep = np.flatnonzero(mask)
        if keep.size == 0:
            return []
        sub, ids, _ = _induced(self, keep)
        if sub.n == 0:
            return []
        if sub.m == 0:
            return [np.array([v], dtype=np.int64) for v in ids]
        rows = sub.edges[:, 0]
        cols = sub.edges[:, 1]
        data = np.ones(len(rows), dtype=np.int8)
        adj = sp.csr_matrix((data, (rows, cols)), shape=(sub.n, sub.n))
        n_comp, labels = _cc(adj, directed=False)
        order = np.argsort(labels, kind="stable")
        sorted_labels = labels[order]
        bounds = np.flatnonzero(np.diff(sorted_labels)) + 1
        groups = np.split(order, bounds)
        comps = [np.sort(ids[g]) for g in groups]
        comps.sort(key=lambda a: int(a[0]))
        return comps

    def to_scipy(self) -> sp.csr_matrix:
        rows = np.concatenate([self.edges[:, 0], self.edges[:, 1]]) if self.m else np.empty(0, np.int64)
        cols = np.concatenate([self.edges[:, 1], self.edges[:, 0]]) if self.m else np.empty(0, np.int64)
        data = np.ones(rows.size, dtype=np.int8)
        return sp.csr_matrix((data, (rows, cols)), shape=(self.n, self.n))


def _induced(g: Graph, ids: np.ndarray, edge_mask: np.ndarray | None = None) -> tuple[Graph, np.ndarray, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int64)
    pos = np.full(g.n, -1, dtype=np.int64)
    pos[ids] = np.arange(ids.size, dtype=np.int64)
    if g.m:
        keep = pos[g.edges[:, 0]] >= 0
        keep &= pos[g.edges[:, 1]] >= 0
        if edge_mask is not None:
            keep &= edge_mask
        e = g.edges[keep]
        e = pos[e]
    else:
        e = np.empty((0, 2), dtype=np.int64)
    return Graph.from_edges(ids.size, e), ids, pos


def greedy_maximal_independent_set(g: Graph, rng: np.random.Generator | None = None) -> np.ndarray:
    """Greedy (randomised) maximal independent set on ``g``.

    Used by the Random-Walk baseline of Table 1 (greedy initialisation) and by
    tests.  Returns an int8 label vector of length ``g.n``.
    """
    labels = np.full(g.n, -1, dtype=np.int8)
    deg = g.degrees().astype(np.float64)
    order = np.arange(g.n, dtype=np.int64)
    if rng is None:
        order = order[np.argsort(-deg, kind="stable")]
    else:
        key = -deg + rng.random(g.n) * 1e-9
        order = order[np.argsort(key, kind="stable")]
    chosen = np.zeros(g.n, dtype=bool)
    blocked = np.zeros(g.n, dtype=bool)
    for v in order:
        if blocked[v]:
            continue
        chosen[v] = True
        blocked[v] = True
        blocked[g.neighbors(v)] = True
    labels[chosen] = 1
    labels[~chosen] = 0
    return labels
