"""Partitioner interface.

A partitioner answers Algorithm 1, line 4: it returns the cut ``C`` and the
associated vertex set ``V_C`` (Def. 4).  The paper is explicit that an exact
minimum cut is *not* required, only "a cut that balances the size of the
resulting subgraphs while minimising the number of cut edges".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from rgf.graph import Graph


@dataclass(frozen=True)
class Cut:
    edges: np.ndarray  # (k, 2) int64 local vertex ids of the cut edges
    vc_ids: np.ndarray  # sorted unique vertices incident to those edges
    edge_mask: np.ndarray  # bool over the parent graph edges; True = this edge is cut

    @property
    def size(self) -> int:
        return int(self.edges.shape[0])


class Partitioner(ABC):
    name: str

    @abstractmethod
    def partition(self, g: Graph, rng: np.random.Generator) -> Cut:
        """Return a cut of ``g``; may return an empty cut for tiny/degenerate graphs."""

    @staticmethod
    def empty_cut(g: Graph) -> Cut:
        return Cut(
            edges=np.empty((0, 2), dtype=np.int64),
            vc_ids=np.empty(0, dtype=np.int64),
            edge_mask=np.zeros(g.m, dtype=bool),
        )

    @staticmethod
    def cut_from_assignment(g: Graph, side: np.ndarray) -> Cut:
        """Cut induced by a vertex -> part assignment (``side`` in {0, 1})."""
        if g.m == 0:
            return Partitioner.empty_cut(g)
        crossing = side[g.edges[:, 0]] != side[g.edges[:, 1]]
        edges = g.edges[crossing]
        if edges.size == 0:
            return Partitioner.empty_cut(g)
        return Cut(edges=edges, vc_ids=np.unique(edges.reshape(-1)), edge_mask=crossing)
