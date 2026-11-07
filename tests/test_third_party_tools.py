"""Integration checks for the third-party tools (METIS, KaMIS).

Both are skipped when the corresponding binary has not been built, so the suite
stays green on a fresh checkout while still covering the real code paths on this
workstation (``scripts/build_third_party_windows.cmd`` and
``scripts/build_kamis_wsl.sh``).
"""

from __future__ import annotations

import numpy as np
import pytest

from rgf.baselines.ds_reference import ds_reference, greedy_ds
from rgf.baselines.kamis import find_kamis, kamis_mis, vc_via_mis
from rgf.datasets.synthetic import er_graph
from rgf.graph import Graph
from rgf.partition.metis import MetisPartitioner, find_gpmetis, find_libmetis
from rgf.problems import DominatingSet, MaximumIndependentSet, VertexCover


def brute_force(problem, g: Graph) -> int:
    """Smallest/largest objective over all feasible labellings (tiny graphs only)."""
    import itertools

    best = None
    for bits in itertools.product([0, 1], repeat=g.n):
        labels = np.array(bits, dtype=np.int8)
        if not problem.feasible(g, labels):
            continue
        value = problem.objective_value(g, labels)
        if best is None or problem.better(value, best):
            best = value
    return best


def _rb_like_graph(n: int = 240, rng: np.random.Generator | None = None) -> Graph:
    rng = rng or np.random.default_rng(0)
    iu, ju = np.triu_indices(n, k=1)
    mask = rng.random(iu.size) < 0.06
    return Graph.from_edges(n, np.stack([iu[mask], ju[mask]], axis=1))


@pytest.mark.skipif(find_gpmetis() is None, reason="gpmetis not built")
def test_metis_partitioner_returns_balanced_cut() -> None:
    rng = np.random.default_rng(0)
    g = _rb_like_graph(200, rng)
    part = MetisPartitioner()
    cut = part.partition(g, rng)
    assert cut.vc_ids.size > 0
    assert cut.edge_mask.shape == (g.m,)
    # a METIS cut is a genuine cut: subgraphs are disconnected once cut edges go
    remaining = g.induced(np.arange(g.n), edge_mask=~cut.edge_mask)[0]
    assert len(remaining.components()) >= 2
    # the cut vertices are exactly the endpoints of the cut edges
    assert set(cut.vc_ids.tolist()) == set(np.unique(cut.edges.reshape(-1)).tolist())


@pytest.mark.skipif(find_kamis() is None, reason="KaMIS not built")
def test_kamis_matches_exact_on_small_graph() -> None:
    problem = MaximumIndependentSet()
    rng = np.random.default_rng(1)
    g = er_graph(40, 0.1, rng)
    labels = kamis_mis(g, time_limit_s=5)
    assert problem.feasible(g, labels)
    exact = problem.solve_exact(g, np.full(g.n, -1, dtype=np.int8), np.zeros(g.n, dtype=bool))
    assert problem.objective_value(g, labels) == problem.objective_value(g, exact)


@pytest.mark.skipif(find_kamis() is None, reason="KaMIS not built")
def test_vc_via_mis_complement_is_feasible() -> None:
    problem = VertexCover()
    rng = np.random.default_rng(2)
    g = er_graph(60, 0.08, rng)
    mis = kamis_mis(g, time_limit_s=5)
    vc = vc_via_mis(g, mis)
    assert problem.feasible(g, vc)


def test_ds_reference_is_feasible_and_optimal_when_small() -> None:
    """The DS reference (docs/decisions.md D-6) must be feasible, and optimal on
    instances small enough for the MILP to prove it."""
    problem = DominatingSet()
    rng = np.random.default_rng(5)
    for trial in range(5):
        g = er_graph(int(rng.integers(12, 20)), float(rng.uniform(0.1, 0.35)), rng)
        ref = ds_reference(g, time_limit_s=30)
        assert problem.feasible(g, ref.labels)
        # D-18 起参照解会做冗余删除并与逆向贪心取更优，方法名带 +redundancy 后缀
        assert ref.method.startswith("milp-exact") and ref.optimal
        assert ref.objective == brute_force(problem, g)


def test_greedy_ds_is_feasible_on_rb_like_graph() -> None:
    problem = DominatingSet()
    rng = np.random.default_rng(6)
    g = _rb_like_graph(240, rng)
    labels = greedy_ds(g)
    assert problem.feasible(g, labels)
    assert np.count_nonzero(labels == 1) <= g.n
