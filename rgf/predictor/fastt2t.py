"""Fast T2T predictor adapter.

Loads the official Fast T2T MIS checkpoints (consistency models: one network
evaluation maps a noisy state ``(xt, t)`` to the predicted ``x0``).

Inference follows ``MISConsistency.consistency_test_step`` verbatim:

1. **initial sampling** — ``inference_diffusion_steps`` consistency steps from
   Bernoulli noise (paper: Ts = 1 or 5);
2. **guided rewrite** — ``rewrite_steps`` rounds (paper: Tg = 0/1/5).  Each round
   re-noises the current best solution to ``steps_T = diffusion_steps *
   rewrite_ratio`` through ``Q_bar`` and applies
   ``MISConsistency.guided_denoise_step``, which runs an inner 10-step gradient
   guidance (``c1 * BCE + c2 * (f_mis + 0.5 * conflict)``) and returns two
   candidate probability vectors; both are decoded greedily and the best is kept.

Per-dataset settings from the official eval scripts: RB ratio 0.3 / c1=c2=2 /
hidden 256 / Ts=1, Tg=1; ER ratio 0.2 / c1=c2=2 / hidden 128 / Ts=5, Tg=5;
SATLIB ratio 0.5 / c1=1 / c2=20.

Only the cut-vertex entries of the resulting heatmap are read out inside the RGF
recursion.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
    inference_diffusion_steps=1,
    sequential_sampling=1,
    parallel_sampling=1,
    consistency=True,
    guided=True,
    rewrite=False,
    norm=False,
    boundary_func="truncate",
    c1=2.0,
    c2=2.0,
    rewrite_steps=1,
    rewrite_ratio=0.3,
)


def _load_train_module(train_py: Path):
    """Import Fast T2T's train.py under a private module name (DIFUSCO's adapter
    may already have claimed the plain ``train`` name)."""
    name = "_fastt2t_train"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, train_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FastT2TPredictor:
    """Adapter satisfying the ``rgf.predictor.base.Predictor`` protocol."""

    def __init__(
        self,
        checkpoint: str | Path,
        inference_diffusion_steps: int = 1,
        device: str = "cuda",
        seed: int | None = None,
        hidden_dim: int = 256,
        guided: bool = True,
        rewrite_steps: int = 1,
        rewrite_ratio: float = 0.3,
        c1: float = 2.0,
        c2: float = 2.0,
    ):
        repo_root = Path(__file__).resolve().parents[2]
        fastt2t_root = repo_root / "third_party" / "Fast-T2T"
        install_shims()
        if str(fastt2t_root) not in sys.path:
            sys.path.insert(0, str(fastt2t_root))

        train_module = _load_train_module(fastt2t_root / "train.py")

        argv_backup = sys.argv
        sys.argv = [
            argv_backup[0],
            "--task", "mis",
            "--storage_path", str(repo_root / "third_party" / "_empty_storage"),
        ]
        try:
            args = train_module.arg_parser()
        finally:
            sys.argv = argv_backup
        for key, value in _DEFAULTS.items():
            setattr(args, key, value)
        args.inference_diffusion_steps = int(inference_diffusion_steps)
        args.hidden_dim = int(hidden_dim)
        args.consistency = True
        args.guided = bool(guided)
        args.rewrite = bool(guided) and rewrite_steps > 0
        args.rewrite_steps = int(rewrite_steps)
        args.rewrite_ratio = float(rewrite_ratio)
        args.c1 = float(c1)
        args.c2 = float(c2)
        args.storage_path = str(repo_root / "third_party" / "_empty_storage")
        args.training_split = "train/*gpickle"
        args.validation_split = "val/*gpickle"
        args.test_split = "test/*gpickle"
        args.training_split_label_dir = None
        self._scratch = Path(args.storage_path)
        self._scratch.mkdir(parents=True, exist_ok=True)

        from diffusion.pl_mis_model import MISModel  # noqa: PLC0415 - third-party import

        self.model = MISModel(param_args=args)
        ckpt = torch_load(checkpoint, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt["state_dict"])
        # FastT2T's on_save_checkpoint strips the "model." prefix; add it back.
        state_dict = {
            (key if key.startswith("model.") else f"model.{key}"): value
            for key, value in state_dict.items()
        }
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(f"missing state-dict keys: {missing[:5]}")
        if unexpected:
            raise RuntimeError(f"unexpected state-dict keys: {unexpected[:5]}")
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.steps = int(inference_diffusion_steps)
        self.seed = seed

    # ------------------------------------------------------------------ utils
    def _edge_index(self, g: Graph, device: str) -> torch.Tensor:
        """Both directions plus a self-loop on every vertex (datasets/mis_dataset)."""
        e = g.edges
        loops = np.arange(g.n).reshape(-1, 1).repeat(2, axis=1)
        both = np.concatenate([e, e[:, ::-1], loops], axis=0)
        ei = torch.as_tensor(both.T, dtype=torch.long)
        return ei.reshape(2, -1).to(device)

    @staticmethod
    def _greedy_solution(g: Graph, probs: np.ndarray) -> np.ndarray:
        """Greedy MIS decoding (rank by heatmap, insert when it conflicts with
        nothing already chosen) — identical to ``mis_decode_np``."""
        chosen = np.zeros(g.n, dtype=np.int8)
        blocked = np.zeros(g.n, dtype=bool)
        for v in np.argsort(-probs, kind="stable"):
            if blocked[v]:
                continue
            chosen[v] = 1
            blocked[v] = True
            blocked[g.neighbors(v)] = True
        return chosen

    # ------------------------------------------------------------------ predict
    def predict(self, batch: list[tuple[Graph, np.ndarray]]) -> list[np.ndarray]:
        """Return one boolean array per ``(graph, vc_ids)`` pair, aligned with ``vc_ids``.

        Like the DIFUSCO adapter: probabilities are read off at the cut vertices
        and thresholded at 0.5 into the CDSP decision (see ``predictor/base.py``).
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

    def _initial_sample(self, g: Graph, edge_index: torch.Tensor) -> np.ndarray:
        from diffusion.utils.diffusion_schedulers import InferenceSchedule  # noqa: PLC0415

        n = g.n
        xt = (torch.randn(n) > 0).long().reshape(-1).to(self.device)
        schedule = InferenceSchedule(
            inference_schedule=self.model.args.inference_schedule,
            T=self.model.diffusion.T,
            inference_T=self.steps,
        )
        probs = None
        # 推理必须关闭梯度（上游官方路径 pl_mis_model.categorical_denoise_step 同样用
        # torch.no_grad 包住前向）。2026-10-04 修：此前第 208 行的前向在开启梯度的状态下
        # 执行，自动求导会保留每一层的中间张量，大邻域子图上内存放大约 10 倍以上——
        # citationCiteseer 的整图前向（268,495 顶点 / 1,156,647 条无向边）因此申请到
        # 122 GB 而 OOM（24 GB 卡），而同样规模在 no_grad 下实测约 9 GB。
        with torch.no_grad():
            for i in range(self.steps):
                t1, t2 = schedule(i)
                xt_scale = xt * 2 - 1
                xt_scale = xt_scale * (1.0 + 0.05 * torch.rand_like(xt.float(), device=self.device))
                x0_pred = self.model(
                    xt_scale.reshape(-1),
                    torch.tensor([t1], device=self.device).float(),
                    edge_index.long().to(self.device),
                )
                x0_pred_prob = x0_pred.reshape((1, n, -1, 2)).softmax(dim=-1)
                probs = x0_pred_prob[..., 1].detach().float().cpu().numpy().reshape(-1)
                if int(t2) != 0:
                    x0 = torch.bernoulli(x0_pred_prob[..., 1].clamp(0, 1))
                    x0_onehot = F.one_hot(x0.long(), num_classes=2).float()
                    q_bar = torch.from_numpy(self.model.diffusion.Q_bar[int(t2)]).float().to(x0_onehot.device)
                    xt_prob = torch.matmul(x0_onehot, q_bar)
                    xt = torch.bernoulli(xt_prob[..., 1].clamp(0, 1)).long().reshape(-1)
        return probs

    def _guided_rewrite(self, g: Graph, edge_index: torch.Tensor, seed_solution: np.ndarray) -> tuple[np.ndarray, int]:
        """The guided rewrite of ``consistency_test_step``, verbatim."""
        trainer = self.model.consistency_trainer
        device = self.device
        steps_t = int(self.model.args.diffusion_steps * self.model.args.rewrite_ratio)
        q_bar = torch.from_numpy(self.model.diffusion.Q_bar[steps_t]).float().to(device)
        t = torch.tensor([steps_t]).int()

        g_best = seed_solution.copy()
        g_best_cost = int(g_best.sum())
        best_probs = None
        for _round in range(int(self.model.args.rewrite_steps)):
            g_x0 = torch.from_numpy(g_best).unsqueeze(0).to(device)
            g_x0_onehot = F.one_hot(g_x0.long(), num_classes=2).float()
            g_xt_prob = torch.matmul(g_x0_onehot, q_bar)
            if self.model.args.guided:
                out = trainer.guided_denoise_step(self.model, g_xt_prob, g_x0.reshape(-1), t, device, edge_index)
                n_candidates = 2
            else:
                g_xt = torch.bernoulli(g_xt_prob[..., 1].clamp(0, 1)).to(device).reshape(-1)
                out = trainer.denoise_step(self.model, g_xt, t, device, edge_index)
                n_candidates = 1
            pred = out.detach().float().cpu().numpy().reshape(-1) + 1e-6
            for chunk in np.split(pred, n_candidates):
                solution = self._greedy_solution(g, chunk)
                cost = int(solution.sum())
                if cost > g_best_cost:
                    g_best_cost = cost
                    g_best = solution
                    best_probs = chunk
        return best_probs, g_best_cost

    def _predict_graph(self, g: Graph) -> np.ndarray:
        edge_index = self._edge_index(g, self.device)
        probs = self._initial_sample(g, edge_index)
        if not self.model.args.rewrite:
            return probs
        seed = self._greedy_solution(g, probs)
        guided_probs, _cost = self._guided_rewrite(g, edge_index, seed)
        return probs if guided_probs is None else guided_probs
