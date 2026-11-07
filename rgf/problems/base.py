"""Common interface of the three ECSPs.

Terminology follows the paper:

``V_C``      vertices incident to the cut edges ``C`` (Def. 4);
``S_C``      the part of ``V_C`` the CDSP decides to include (Def. 4);
``removed``  vertices that leave the remaining instance after a cut is resolved
             (Eq. ``mis_r`` for MIS / Eq. ``vc_r`` for VC; DS removes none);
``dominated`` (DS only) vertices already dominated by a decided ``1`` vertex
             across a removed cut edge.

The last item is not spelled out in the paper: the appendix proof of
Proposition 1 argues for DS that "the domination constraint for ``v`` in ``V_i``
only depends on edges within ``E_i`` and the labels of vertices in ``V_i``", but
every cut edge has *both* endpoints in ``V_C``, so a component can be adjacent to
a decided cut vertex that ends up in a *different* component.  We therefore carry
an explicit ``dominated`` flag instead of assuming the constraint away.  See
``docs/decisions.md`` item D-4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from rgf.graph import Graph


@dataclass
class CutOutcome:
    """Result of resolving one cut on one sub-instance."""

    labels: np.ndarray  # int8, -1 unknown / 0 / 1, aligned with the sub-instance vertices
    removed: np.ndarray  # bool, vertices leaving the remaining instance
    dominated: np.ndarray  # bool, DS only (see module docstring)


class ECSP(ABC):
    """Abstract edge-cut separable problem."""

    name: ClassVar[str]
    maximize: ClassVar[bool]

    # ------------------------------------------------------------- semantics
    @abstractmethod
    def feasible(self, g: Graph, labels: np.ndarray) -> bool:
        """Whether ``labels`` (complete, values in {0, 1}) is a valid solution of ``g``."""

    def objective_value(self, g: Graph, labels: np.ndarray) -> int:
        """Solution size |S|; all three ECSPs optimise the cardinality of ``S``."""
        return int(np.count_nonzero(labels == 1))

    def better(self, a: int, b: int) -> bool:
        """Whether objective value ``a`` is better than ``b``."""
        return a > b if self.maximize else a < b

    def is_better_equal(self, a: int, b: int) -> bool:
        return a >= b if self.maximize else a <= b

    # ------------------------------------------------------------- cut logic
    @abstractmethod
    def apply_cut(self, g: Graph, vc_ids: np.ndarray, s_c: np.ndarray) -> CutOutcome:
        """Resolve the cut: fix ``V_C`` to ``S_C``, infer neighbours, return what leaves."""

    def repair_cdsp(self, g: Graph, cut, s_c: np.ndarray) -> np.ndarray:
        """Legality safeguard applied to the predictor's ``S_C`` before ``apply_cut``.

        The paper's completeness theorem (appendix, Theorem 1) assumes a
        *complete* predictor, i.e. one whose ``S_C`` is consistent with some
        optimal solution.  A learned predictor is only approximate, and an
        inconsistent ``S_C`` does not merely cost quality: for VC it can leave a
        cut edge uncovered, and for DS it can leave a decided-0 cut vertex
        undominated, which makes the recursive sub-instances infeasible.  This
        hook restores a *legal* (not necessarily optimal) decision; the default
        is the identity.  See ``docs/decisions.md`` item D-10.
        """
        return s_c

    def polish(self, g: Graph, labels: np.ndarray) -> np.ndarray:
        """Final cleanup after merging overlapping sub-instances (D-21).

        Default: identity.  Overlapping decompositions (DS) take the union of the
        sub-solutions, which may contain redundant selections; problems override
        this with their own cleanup (e.g. DS redundancy removal).
        """
        return labels

    # ------------------------------------------------------------ exact solve
    @abstractmethod
    def solve_exact(self, g: Graph, labels: np.ndarray, dominated: np.ndarray) -> np.ndarray:
        """Solve a sub-instance exactly (used when ``|V| <= alpha``).

        ``labels`` marks already decided vertices (``-1`` = free); ``dominated``
        marks vertices already dominated by a decided vertex outside the
        sub-instance (DS only).  Returns a complete label vector.
        """

    # ----------------------------------------------------------------- utils
    @staticmethod
    def _empty_dominated(g: Graph) -> np.ndarray:
        return np.zeros(g.n, dtype=bool)
