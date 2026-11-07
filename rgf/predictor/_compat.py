"""Compatibility shims for importing the third-party predictors.

The DIFUSCO / Fast T2T code targets Python 3.7 + torch 1.11 + torch_sparse +
pickle5, none of which install on this Python 3.14 environment.  The two models
only use a small slice of ``torch_sparse`` (``SparseTensor(row, col, value,
sparse_sizes)`` plus ``sum``/``mean``/``max`` aggregations over dim=1), which is
reimplemented here with native torch ops, so no compiled extension is needed.

``install_shims()`` must be called before importing any third-party module.
"""

from __future__ import annotations

import importlib
import pickle
import sys
import types

import torch


def _version_tuple(version: str) -> tuple[int, ...]:
    core = version.split("+")[0]
    parts = []
    for chunk in core.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


# ``weights_only`` reached torch.load in 1.13; the paper-era environment (torch
# 1.11) forwards unknown keywords straight to Unpickler and dies with TypeError.
_TORCH_HAS_WEIGHTS_ONLY = _version_tuple(torch.__version__) >= (1, 13)


def torch_load(path, map_location: str = "cpu"):
    """``torch.load`` that also works on the paper-era torch (1.11)."""
    kwargs = {"map_location": map_location}
    if _TORCH_HAS_WEIGHTS_ONLY:
        # the official DIFUSCO / Fast T2T checkpoints are pickled namespaces, not
        # plain tensor dicts, so the safe loader has to be opted out of explicitly
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


class _SparseTensor:
    """Minimal stand-in for ``torch_sparse.SparseTensor``.

    Supports exactly what ``models/gnn_encoder.py`` and the two ``pl_mis_model``
    modules use: construction from (row, col, value, sparse_sizes), ``.to``,
    ``.size``, and module-level ``sum``/``mean``/``max`` aggregations of the
    stored per-edge values along a dimension.
    """

    def __init__(self, row=None, col=None, value=None, sparse_sizes=None, **kwargs):
        self.row = row
        self.col = col
        self.value = value
        self._sizes = tuple(sparse_sizes) if sparse_sizes is not None else None

    def to(self, device, *args, **kwargs):
        self.row = self.row.to(device) if self.row is not None else None
        self.col = self.col.to(device) if self.col is not None else None
        self.value = self.value.to(device) if self.value is not None else None
        return self

    def size(self, index=None):
        if index is None:
            return self._sizes
        return self._sizes[index]

    def to_dense(self) -> torch.Tensor:
        """Dense adjacency (scalar edge values only).

        Fast T2T's guided rewrite densifies the cut graph to build its
        conflict objective; the graphs it is used on here are small (the
        synthetic benchmarks and beta-bounded neighbourhoods).
        """
        value = self.value
        if value is None:
            value = torch.ones_like(self.row, dtype=torch.float)
        if value.dim() > 1:
            raise NotImplementedError("to_dense is only defined for scalar edge values")
        out = torch.zeros(self._sizes, dtype=value.dtype, device=value.device)
        out[self.row, self.col] = value
        return out


def _aggregate(st: "_SparseTensor", dim: int, reduce: str) -> torch.Tensor:
    """Aggregate ``st.value`` over ``dim`` grouped by the opposite index.

    ``dim=1`` aggregates incoming edges per target vertex (column), matching the
    ``sparse_mean(sparseVh, dim=1)`` calls in ``gnn_encoder.py``.
    """
    n = st._sizes[1] if dim == 1 else st._sizes[0]
    group = st.col if dim == 1 else st.row
    value = st.value
    if value is None:
        value = torch.ones_like(group, dtype=torch.float)
    if value.dim() == 1:
        out_shape = (n,)
    else:
        out_shape = (n,) + tuple(value.shape[1:])
    count = torch.zeros(n, dtype=value.dtype, device=value.device)
    count.index_add_(0, group, torch.ones_like(group, dtype=value.dtype))
    if reduce == "sum":
        out = torch.zeros(out_shape, dtype=value.dtype, device=value.device)
        out.index_add_(0, group, value)
        return out
    if reduce == "mean":
        out = torch.zeros(out_shape, dtype=value.dtype, device=value.device)
        out.index_add_(0, group, value)
        return out / count.clamp(min=1).view(-1, *([1] * (value.dim() - 1)))
    if reduce == "max":
        fill = torch.finfo(value.dtype).min
        out = torch.full(out_shape, fill, dtype=value.dtype, device=value.device)
        out = out.index_reduce(0, group, value, "amax", include_self=True)
        empty = count == 0
        if empty.any():
            out[empty] = 0
        return out
    raise ValueError(reduce)


def _make_module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def install_shims() -> None:
    """Register the compatibility modules (idempotent)."""
    # pickle5: a Python 3.7 backport of the pickle protocol; stdlib pickle reads
    # those payloads on 3.14.
    if "pickle5" not in sys.modules:
        # expose the stdlib functions at the module level (mis_dataset calls
        # pickle5.load directly)
        _make_module(
            "pickle5",
            load=pickle.load,
            loads=pickle.loads,
            dump=pickle.dump,
            dumps=pickle.dumps,
        )

    if "wandb" not in sys.modules:
        # train.py imports wandb at module level for its training loop; the
        # inference path never calls into it, so a stub is enough.
        wandb = _make_module(
            "wandb",
            init=lambda *a, **k: None,
            log=lambda *a, **k: None,
        )
        wandb.util = _make_module("wandb.util", generate_id=lambda: "shim")

    if "torch_sparse" not in sys.modules:
        # Prefer the real compiled extension whenever it is importable (the WSL
        # faithful environment has the cu113 wheels).  The Python stub below is
        # only a last resort for environments without it, and it is NOT
        # numerically equivalent to the real sparse kernels: measured on ER
        # test graphs the stub costs ~10 MIS points (see docs/decisions.md D-11).
        try:
            importlib.import_module("torch_sparse")
        except Exception:
            _make_module(
                "torch_sparse",
                SparseTensor=_SparseTensor,
                sum=lambda st, dim=None: _aggregate(st, dim if dim is not None else 1, "sum"),
                mean=lambda st, dim=None: _aggregate(st, dim if dim is not None else 1, "mean"),
                max=lambda st, dim=None: _aggregate(st, dim if dim is not None else 1, "max"),
            )

    if "utils.cython_merge.cython_merge" not in sys.modules:
        # DIFUSCO's TSP chain imports a compiled cython tour-merger at module
        # level; the MIS path never calls it.  Fast T2T needs the same stub
        # under its own "diffusion." namespace.
        stub = _make_module("utils.cython_merge.cython_merge")

        def _merge_cython(*_args, **_kwargs):
            raise NotImplementedError("cython tour merging is TSP-only")

        stub.merge_cython = _merge_cython
        parent = _make_module("utils.cython_merge", cython_merge=stub)
        diff_stub = _make_module("diffusion.utils.cython_merge.cython_merge")

        def _merge_cython2(*_args, **_kwargs):
            raise NotImplementedError("cython tour merging is TSP-only")

        diff_stub.merge_cython = _merge_cython2
        diff_parent = _make_module("diffusion.utils.cython_merge", cython_merge=diff_stub)

    # torch_geometric >= 2.4 moved DataLoader to torch_geometric.loader; the
    # third-party code imports it from torch_geometric.data.
    try:
        tg_data = importlib.import_module("torch_geometric.data")
        if not hasattr(tg_data, "DataLoader"):
            loader = importlib.import_module("torch_geometric.loader")
            tg_data.DataLoader = loader.DataLoader
    except ImportError:
        pass
