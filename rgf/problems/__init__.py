"""Edge-Cut Separable Problems (ECSPs): MIS, DS, VC.

对应论文 Def. 1–3 与 §The Cut-Determination Sub-problem 的分割规则。
"""

from __future__ import annotations

from rgf.problems.base import CutOutcome, ECSP
from rgf.problems.ds import DominatingSet
from rgf.problems.mis import MaximumIndependentSet
from rgf.problems.vc import VertexCover

__all__ = ["CutOutcome", "ECSP", "MaximumIndependentSet", "DominatingSet", "VertexCover", "PROBLEMS"]

PROBLEMS: dict[str, type[ECSP]] = {
    "mis": MaximumIndependentSet,
    "ds": DominatingSet,
    "vc": VertexCover,
}
