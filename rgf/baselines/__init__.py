"""Baselines used in the paper's tables."""

from __future__ import annotations

from rgf.baselines.ds_reference import ReferenceSolution, ds_reference, greedy_ds
from rgf.baselines.random_walk import random_walk_mis

__all__ = ["random_walk_mis", "ds_reference", "greedy_ds", "ReferenceSolution"]
