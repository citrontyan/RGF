"""Graph partitioners used to identify a cut (paper §Experimental Settings)."""

from __future__ import annotations

from rgf.partition.base import Cut, Partitioner
from rgf.partition.metis import MetisPartitioner
from rgf.partition.random_balanced import RandomBalancedPartitioner
from rgf.partition.single_vertex import SingleVertexPartitioner

__all__ = ["Cut", "Partitioner", "MetisPartitioner", "RandomBalancedPartitioner", "SingleVertexPartitioner"]

PARTITIONERS: dict[str, type[Partitioner]] = {
    "metis": MetisPartitioner,
    "random": RandomBalancedPartitioner,
    "single-vertex": SingleVertexPartitioner,
}
