"""DIFUSCO predictor adapter.

Loads the official DIFUSCO MIS checkpoints (``difusco/pl_mis_model.py``) without
PyTorch-Lightning's ``load_from_checkpoint``: the checkpoints carry no
hyper-parameters, so the training-time argument set is reconstructed from
DIFUSCO's own ``reproducing_scripts.md`` (ER-[700-800] gaussian block and the
SATLIB categorical block; both share the argparse defaults ``n_layers=12``,
``hidden_dim=256``, ``aggregation=sum``, ``sparse_factor=-1``,
``diffusion_steps=1000``, ``diffusion_schedule=linear``) and the state dict is
loaded directly.

Inference mirrors ``MISModel.test_step`` verbatim (initial noise, the cosine
inference schedule with ``inference_diffusion_steps`` denoise steps) and then
returns the per-node membership probability: ``xt * 0.5 + 0.5`` for the gaussian
path, and the final ``x0`` posterior probability of class 1 for the categorical
path.  Per the paper, only the cut-vertex entries are read out.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch

from rgf.graph import Graph
from rgf.predictor._compat import install_shims, torch_load

_DEFAULTS = dict(
    task="mis",
    diffusion_schedule="linear",
    diffusion_steps=1000,
    sparse_factor=-1,
    n_layers=12,
    hidden_dim=256,
    aggregation="sum",
    use_activation_checkpoint=False,
    inference_schedule="cosine",
    inference_diffusion_steps=50,
    sequential_sampling=1,
    parallel_sampling=1,
)


def _load_mis_model_class(difusco_dir: Path):
    # DIFUSCO's modules import each other as top-level packages ("models",
    # "co_datasets", "utils", "pl_mis_model"), so the inner directory that
    # contains them must be on sys.path (this mirrors their PYTHONPATH="$PWD"
    # with the script living in difusco/).
    difusco_dir = str(difusco_dir)
    if difusco_dir not in sys.path:
        sys.path.insert(0, difusco_dir)
        # train.py also pulls in the TSP chain, whose tour-merging helper is a
        # compiled cython extension; the MIS path never calls it, so register a
        # stub instead of building the extension.
    stub = types.ModuleType("utils.cython_merge.cython_merge")

    def _merge_cython(*_args, **_kwargs):
        raise NotImplementedError("cython tour merging is TSP-only")

    stub.merge_cython = _merge_cython
    parent = types.ModuleType("utils.cython_merge")
    parent.cython_merge = stub
    sys.modules.setdefault("utils.cython_merge.cython_merge", stub)
    sys.modules.setdefault("utils.cython_merge", parent)
    from pl_mis_model import MISModel  # noqa: PLC0415 - third-party import

    return MISModel


class DifuscoPredictor:
    """Adapter satisfying the ``rgf.predictor.base.Predictor`` protocol."""

    def __init__(
        self,
        checkpoint: str | Path,
        diffusion_type: str | None = None,
        inference_diffusion_steps: int = 50,
        device: str = "cuda",
        seed: int | None = None,
        node_feature_only: bool = True,
    ):
        repo_root = Path(__file__).resolve().parents[2]
        difusco_dir = repo_root / "third_party" / "DIFUSCO"
        install_shims()
        checkpoint = Path(checkpoint)
        if diffusion_type is None:
            diffusion_type = "categorical" if "categorical" in checkpoint.stem else "gaussian"
        if diffusion_type not in ("gaussian", "categorical"):
            raise ValueError(diffusion_type)

        ckpt = torch_load(checkpoint, map_location="cpu")
        state_dict = ckpt["state_dict"]

        mis_model = _load_mis_model_class(difusco_dir / "difusco")

        # The checkpoints carry no hyper-parameters, so rebuild the full
        # argument set from DIFUSCO's own argparse (defaults + the overrides of
        # reproducing_scripts.md) instead of hand-picking fields.
        from train import arg_parser  # noqa: PLC0415 - third-party import

        # train.py's arg_parser() itself calls parse_args() on sys.argv and
        # returns the parsed namespace, so point sys.argv at the required flags.
        argv_backup = sys.argv
        sys.argv = [
            argv_backup[0],
            "--task", "mis",
            "--storage_path", str(repo_root / "third_party" / "_empty_storage"),
        ]
        try:
            args = arg_parser()
        finally:
            sys.argv = argv_backup
        for key, value in _DEFAULTS.items():
            setattr(args, key, value)
        args.diffusion_type = diffusion_type
        args.inference_diffusion_steps = int(inference_diffusion_steps)
        args.storage_path = str(repo_root / "third_party" / "_empty_storage")
        args.training_split = "train/*gpickle"
        args.validation_split = "val/*gpickle"
        args.test_split = "test/*gpickle"
        args.training_split_label_dir = None
        self._scratch = Path(args.storage_path)
        self._scratch.mkdir(parents=True, exist_ok=True)

        self.model = mis_model(param_args=args)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(f"missing state-dict keys: {missing[:5]}")
        if unexpected:
            raise RuntimeError(f"unexpected state-dict keys: {unexpected[:5]}")
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.diffusion_type = diffusion_type
        self.steps = int(inference_diffusion_steps)
        self.seed = seed

    # ------------------------------------------------------------------ predict
    def predict(self, batch: list[tuple[Graph, np.ndarray]]) -> list[np.ndarray]:
        """Return one boolean array per ``(graph, vc_ids)`` pair, aligned with ``vc_ids``.

        The network predicts a membership probability per node; Algorithm 1 only
        reads the cut vertices (``V_C``), so the probabilities are sliced to
        ``vc_ids`` and thresholded at 0.5 into the CDSP decision.
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
        out = []
        for item in batch:
            g, vc_ids = item[0], item[1]  # item 可能含第三元素 ctx（D-20），阈值读数不用
            probs = self._predict_graph(g)
            if vc_ids is None:
                out.append(np.asarray(probs > 0.5, dtype=bool))
            else:
                idx = np.asarray(vc_ids, dtype=np.int64)
                out.append(np.asarray(probs[idx] > 0.5, dtype=bool))
        return out

    def _edge_index(self, g: Graph, device: str) -> torch.Tensor:
        # Mirrors co_datasets/mis_dataset.py: both directions plus a self-loop
        # on every vertex.
        e = g.edges
        loops = np.arange(g.n).reshape(-1, 1).repeat(2, axis=1)
        both = np.concatenate([e, e[:, ::-1], loops], axis=0)
        ei = torch.as_tensor(both.T, dtype=torch.long)
        return ei.reshape(2, -1).to(device)

    def _predict_graph(self, g: Graph) -> np.ndarray:
        from utils.diffusion_schedulers import InferenceSchedule  # noqa: PLC0415 - third-party import

        n = g.n
        edge_index = self._edge_index(g, self.device)
        if self.diffusion_type == "gaussian":
            xt = torch.randn(n).reshape(-1)
            xt = xt.to(self.device)
        else:
            xt = (torch.randn(n) > 0).long().reshape(-1).to(self.device)

        schedule = InferenceSchedule(
            inference_schedule=self.model.args.inference_schedule,
            T=self.model.diffusion.T,
            inference_T=self.steps,
        )
        for i in range(self.steps):
            t1, t2 = schedule(i)
            t1 = np.array([t1]).astype(int)
            t2 = np.array([t2]).astype(int)
            if self.diffusion_type == "gaussian":
                xt = self.model.gaussian_denoise_step(xt, t1, self.device, edge_index, target_t=t2)
            else:
                xt = self.model.categorical_denoise_step(xt, t1, self.device, edge_index, target_t=t2)

        with torch.no_grad():
            if self.diffusion_type == "gaussian":
                probs = (xt.float().cpu().numpy() * 0.5 + 0.5).reshape(-1)
            else:
                t0 = torch.tensor([0.0], device=self.device)
                logits = self.model.forward(xt.float().to(self.device), t0, edge_index)
                prob = logits.reshape((1, n, -1, 2)).softmax(dim=-1)[0, :, :, 1]
                probs = prob.float().cpu().numpy().reshape(-1)
        return probs
