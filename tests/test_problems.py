"""Self-checks for the three problem definitions and their exact solvers.

Brute force over ``2^n`` labellings is the authority for tiny graphs.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from rgf.graph import Graph
from rgf.problems import PROBLEMS, DominatingSet, MaximumIndependentSet, VertexCover


def random_graph(n: int, p: float, rng: np.random.Generator) -> Graph:
    edges = [(u, v) for u, v in itertools.combinations(range(n), 2) if rng.random() < p]
    return Graph.from_edges(n, np.array(edges, dtype=np.int64).reshape(-1, 2))


def brute_force(problem, g: Graph) -> int:
    best = None
    for bits in itertools.product([0, 1], repeat=g.n):
        labels = np.array(bits, dtype=np.int8)
        if not problem.feasible(g, labels):
            continue
        val = problem.objective_value(g, labels)
        if best is None or problem.better(val, best):
            best = val
    return best


@pytest.mark.parametrize("name", list(PROBLEMS))
def test_exact_matches_brute_force(name: str) -> None:
    problem = PROBLEMS[name]()
    rng = np.random.default_rng(0)
    for trial in range(15):
        g = random_graph(n=int(rng.integers(4, 11)), p=float(rng.uniform(0.15, 0.6)), rng=rng)
        labels = np.full(g.n, -1, dtype=np.int8)
        dominated = np.zeros(g.n, dtype=bool)
        solved = problem.solve_exact(g, labels, dominated)
        assert problem.feasible(g, solved)
        assert problem.objective_value(g, solved) == brute_force(problem, g), (name, trial)


@pytest.mark.parametrize("name", list(PROBLEMS))
def test_exact_respects_fixed_labels(name: str) -> None:
    problem = PROBLEMS[name]()
    rng = np.random.default_rng(1)
    g = random_graph(n=8, p=0.35, rng=rng)
    labels = np.full(g.n, -1, dtype=np.int8)
    labels[0] = 1 - int(problem.maximize)  # a harmless fixed value for each problem
    dominated = np.zeros(g.n, dtype=bool)
    solved = problem.solve_exact(g, labels, dominated)
    assert solved[0] == labels[0]


def test_mis_feasibility_and_objective() -> None:
    problem = MaximumIndependentSet()
    g = Graph.from_edges(3, np.array([[0, 1], [1, 2]], dtype=np.int64))
    assert problem.feasible(g, np.array([1, 0, 1], dtype=np.int8))
    assert not problem.feasible(g, np.array([1, 1, 0], dtype=np.int8))
    assert problem.objective_value(g, np.array([1, 0, 1], dtype=np.int8)) == 2


def test_vc_feasibility() -> None:
    problem = VertexCover()
    g = Graph.from_edges(3, np.array([[0, 1], [1, 2]], dtype=np.int64))
    assert problem.feasible(g, np.array([1, 0, 1], dtype=np.int8))
    assert not problem.feasible(g, np.array([0, 0, 1], dtype=np.int8))


def test_ds_feasibility() -> None:
    problem = DominatingSet()
    g = Graph.from_edges(4, np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64))
    assert problem.feasible(g, np.array([0, 1, 1, 0], dtype=np.int8))
    assert not problem.feasible(g, np.array([1, 0, 0, 0], dtype=np.int8))


def test_mis_cut_removes_cut_vertices_and_neighbours_of_chosen() -> None:
    problem = MaximumIndependentSet()
    g = Graph.from_edges(5, np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64))
    outcome = problem.apply_cut(g, np.array([2], dtype=np.int64), np.array([True]))
    assert outcome.labels[2] == 1
    assert outcome.labels[1] == 0 and outcome.labels[3] == 0
    assert set(np.flatnonzero(outcome.removed)) == {1, 2, 3}


def test_vc_cut_forces_neighbours_of_unchosen() -> None:
    problem = VertexCover()
    g = Graph.from_edges(5, np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64))
    outcome = problem.apply_cut(g, np.array([2], dtype=np.int64), np.array([False]))
    assert outcome.labels[2] == 0
    assert outcome.labels[1] == 1 and outcome.labels[3] == 1
    assert set(np.flatnonzero(outcome.removed)) == {1, 2, 3}


def test_ds_cut_removes_no_vertex_but_marks_domination() -> None:
    problem = DominatingSet()
    g = Graph.from_edges(4, np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64))
    outcome = problem.apply_cut(g, np.array([1], dtype=np.int64), np.array([True]))
    assert outcome.labels[1] == 1
    assert not outcome.removed.any()
    assert outcome.dominated[0] and outcome.dominated[2]
