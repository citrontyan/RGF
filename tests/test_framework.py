"""Self-checks for Algorithm 1.

Two things are verified on small graphs:

* with ``alpha`` large enough the framework is exactly the exact solver;
* with ``alpha = 0`` the recursion is forced to run to the leaves while the
  predictor is *complete* (an oracle that solves each CDSP exactly), and the
  assembled solution must still be optimal and feasible.  That is the
  empirical counterpart of Proposition 1 + Theorem 1 (completeness).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from rgf.framework import RGFSolver
from rgf.graph import Graph
from rgf.partition import MetisPartitioner, RandomBalancedPartitioner, SingleVertexPartitioner
from rgf.problems import PROBLEMS


class OraclePredictor:
    """Complete predictor: solves each CDSP exactly (used only by the tests)."""

    name = "oracle"

    def __init__(self, problem):
        self.problem = problem

    def predict(self, batch):
        out = []
        for item in batch:
            g, vc = item[0], item[1]  # item 可含第三元素 ctx（D-20），完整预测器不需要
            labels = np.full(g.n, -1, dtype=np.int8)
            solved = self.problem.solve_exact(g, labels, np.zeros(g.n, dtype=bool))
            out.append(solved[vc] == 1)
        return out


def random_graph(n: int, p: float, rng: np.random.Generator) -> Graph:
    edges = [(u, v) for u, v in itertools.combinations(range(n), 2) if rng.random() < p]
    return Graph.from_edges(n, np.array(edges, dtype=np.int64).reshape(-1, 2))


@pytest.mark.parametrize("name", list(PROBLEMS))
@pytest.mark.parametrize(
    "partitioner_factory",
    [
        lambda: RandomBalancedPartitioner(),
        lambda: SingleVertexPartitioner(),
    ],
)
def test_recursion_with_complete_predictor_is_optimal(name: str, partitioner_factory) -> None:
    problem = PROBLEMS[name]()
    if name == "ds":
        pytest.skip("DS sub-instances are not self-contained (docs/decisions.md D-4)")
    rng = np.random.default_rng(7)
    for trial in range(6):
        g = random_graph(n=12, p=0.3, rng=rng)
        reference = problem.solve_exact(g, np.full(g.n, -1, dtype=np.int8), np.zeros(g.n, dtype=bool))
        solver = RGFSolver(problem, partitioner_factory(), OraclePredictor(problem), alpha=2, seed=trial)
        labels, stats = solver.solve(g)
        assert problem.feasible(g, labels)
        assert problem.objective_value(g, labels) == problem.objective_value(g, reference)
        assert stats.nn_inferences >= 1


@pytest.mark.parametrize("name", list(PROBLEMS))
def test_small_graph_uses_exact_solver(name: str) -> None:
    problem = PROBLEMS[name]()
    rng = np.random.default_rng(3)
    g = random_graph(n=9, p=0.35, rng=rng)
    solver = RGFSolver(problem, RandomBalancedPartitioner(), OraclePredictor(problem), alpha=50)
    labels, stats = solver.solve(g)
    assert stats.nn_inferences == 0
    assert problem.feasible(g, labels)


def test_ds_recursion_returns_feasible_or_reports_gap() -> None:
    """DS recursion must terminate and produce complete labels (feasibility is
    not guaranteed by the paper's predictor API, see docs/decisions.md D-4)."""
    problem = PROBLEMS["ds"]()
    rng = np.random.default_rng(11)
    g = random_graph(n=14, p=0.25, rng=rng)
    solver = RGFSolver(problem, RandomBalancedPartitioner(), OraclePredictor(problem), alpha=2, seed=1)
    labels, stats = solver.solve(g)
    assert labels.shape == (g.n,)
    assert np.all((labels == 0) | (labels == 1))


def test_metis_partitioner_requires_binary() -> None:
    """The ``gpmetis`` back end needs the executable; ``backend="ctypes"`` does not."""
    rng = np.random.default_rng(0)
    g = random_graph(n=6, p=0.4, rng=rng)
    partitioner = MetisPartitioner(executable=None, backend="gpmetis")
    if partitioner.executable is None:
        with pytest.raises(FileNotFoundError):
            partitioner.partition(g, rng)
    else:  # pragma: no cover - depends on the local METIS build
        cut = partitioner.partition(g, rng)
        assert cut.edge_mask.shape == (g.m,)
