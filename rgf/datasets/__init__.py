"""Dataset construction and IO.

Data never lives in the repository: everything is written under ``E:\\data\\rgf``
(override with ``--data-root``).  Only scripts and configuration live here.
"""

from __future__ import annotations

from rgf.datasets.io import load_graph_npz, read_matrix_market, save_graph_npz
from rgf.datasets.synthetic import er_graph, er_graph_in_range, rb_graph, rb_graph_in_range

__all__ = [
    "er_graph",
    "er_graph_in_range",
    "rb_graph",
    "rb_graph_in_range",
    "save_graph_npz",
    "load_graph_npz",
    "read_matrix_market",
]
