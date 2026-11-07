"""Graph IO: NumPy cache, Matrix Market (DIMACS10), METIS adjacency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from rgf.graph import Graph


def save_graph_npz(g: Graph, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # int32 halves the footprint of the large training sets (vertex counts stay
    # far below 2^31)
    np.savez_compressed(path, n=np.int64(g.n), edges=g.edges.astype(np.int32))
    return path


def load_graph_npz(path: str | Path) -> Graph:
    with np.load(path) as data:
        n = int(data["n"])
        edges = data["edges"].astype(np.int64)
    return Graph.from_edges(n, edges)


def read_matrix_market(path: str | Path, *, symmetry: bool = True) -> Graph:
    """Read a Matrix Market file (``.mtx`` or ``.mtx.gz``) as an undirected graph.

    The DIMACS10 co-author / citation / street networks are distributed in this
    format (``https://sites.cc.gatech.edu/dimacs10/downloads.shtml``).  Vertex
    ids are renumbered to ``0..n-1``; self loops are dropped by
    :meth:`Graph.from_edges`.
    """
    m = sio.mmread(str(path))
    m = sp.csr_matrix(m)
    if m.shape[0] != m.shape[1]:
        n = max(m.shape)
        square = sp.lil_matrix((n, n), dtype=np.int8)
        square[: m.shape[0], : m.shape[1]] = m
        m = square.tocsr()
    if symmetry:
        m = m.maximum(m.T)
    coo = sp.triu(m, k=1).tocoo()
    return Graph.from_edges(m.shape[0], np.stack([coo.row, coo.col], axis=1))
