"""CDSP supervision.

Definition 4 makes the CDSP a *prediction* problem: given ``(G, V_C)``, decide
the states of ``V_C`` as they appear in some optimal solution ``S*``.  Building
training targets therefore needs an optimal solution of the instance: for a
reference labelling ``labels_ref`` the target of ``V_C`` is simply
``V_C ∩ S*`` (``rgf/problems/exact.py`` supplies ``S*`` for the small graphs used
in training).

RQ2 of the paper also re-trains DIFUSCO with this partial-solution objective
instead of the complete-solution objective, so the sampler below mirrors the
*distribution of cuts seen at inference time*: it partitions the remaining
instance recursively with the same partitioner and the same threshold ``alpha``
used by ``RGFSolver``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rgf.graph import Graph
from rgf.partition.base import Cut, Partitioner


@dataclass
class CDSPSample:
    graph: Graph
    vc_ids: np.ndarray
    target: np.ndarray  # bool, aligned with vc_ids
    depth: int


def cdsp_target(labels_ref: np.ndarray, vc_ids: np.ndarray) -> np.ndarray:
    """``S_C = V_C ∩ S*`` (Def. 4)."""
    return labels_ref[vc_ids] == 1


def sample_cuts_recursive(
    g: Graph,
    labels_ref: np.ndarray,
    partitioner: Partitioner,
    alpha: int,
    rng: np.random.Generator,
    max_samples: int = 64,
    max_levels: int = 64,
) -> list[CDSPSample]:
    """Collect ``(G_i, V_C, S_C)`` triples along the recursion the solver would follow.

    ``labels_ref`` must be an optimal labelling of ``g`` (``solve_exact``).
    """
    samples: list[CDSPSample] = []
    frontier: list[tuple[Graph, np.ndarray, np.ndarray, int]] = [
        (g, np.arange(g.n, dtype=np.int64), labels_ref.astype(np.int8), 0)
    ]
    for _ in range(max_levels):
        if not frontier or len(samples) >= max_samples:
            break
        nxt: list[tuple[Graph, np.ndarray, np.ndarray, int]] = []
        for sub, ids, ref_local, depth in frontier:
            if sub.n <= alpha or sub.m == 0:
                continue
            cut: Cut = partitioner.partition(sub, rng)
            if cut.vc_ids.size == 0:
                continue
            samples.append(
                CDSPSample(graph=sub, vc_ids=cut.vc_ids, target=cdsp_target(ref_local, cut.vc_ids), depth=depth)
            )
            if len(samples) >= max_samples:
                break
            keep = np.ones(sub.n, dtype=bool)
            keep[cut.vc_ids] = False
            if not keep.any():
                continue
            remaining, rem_ids, _ = sub.induced(np.flatnonzero(keep), edge_mask=~cut.edge_mask)
            for comp in remaining.components():
                parent_local = rem_ids[comp]
                child, _, _ = remaining.induced(comp)
                nxt.append((child, ids[parent_local], ref_local[parent_local], depth + 1))
        frontier = nxt
    return samples


def sample_cuts_single_level(
    g: Graph,
    labels_ref: np.ndarray,
    partitioner: Partitioner,
    rng: np.random.Generator,
    n_cuts: int = 8,
) -> list[CDSPSample]:
    """Cheap alternative: ``n_cuts`` independent cuts on the full graph."""
    out: list[CDSPSample] = []
    for _ in range(n_cuts):
        cut = partitioner.partition(g, rng)
        if cut.vc_ids.size == 0:
            continue
        out.append(CDSPSample(graph=g, vc_ids=cut.vc_ids, target=cdsp_target(labels_ref, cut.vc_ids), depth=0))
    return out
