"""GenSCO predictor adapter for the RGF harness.

Official implementation (JAX/Flax): https://github.com/Thinklab-SJTU/GenSCO
(NeurIPS 2025, ``Generation as Search Operator for Test-Time Scaling of
Diffusion-Based Combinatorial Optimization``).  The upstream checkout lives
unmodified at ``third_party/GenSCO``; this module only drives it.

Why a dedicated conda env: GenSCO pins ``jax[cuda12]==0.5.0`` / ``flax 0.10.4``
/ ``triton 3.1.0``, which conflicts with the RGF harness env ``difusco``
(torch 1.11).  Both this predictor and ``scripts/run_rgf.py`` therefore run in
the isolated ``gensco`` env (``ENV_NAME=gensco scripts/run_wsl.sh ...``).

Inference algorithm (verbatim from ``decoding/mis.py::mis_flow_decode``):

1. encode random node features with the GNN encoder (once per graph);
2. start from a random greedy MIS ``S``;
3. each cycle: flip 1->0 with rate ~ U(0.25, 0.4), integrate the rectified-flow
   decoder with ``sampling_steps`` Euler steps to get the denoised solution,
   greedily insert nodes in descending order into ``S`` and keep the best size;
4. ``decode_intermediate`` (official eval default) also tries every intermediate
   Euler state as a candidate ordering.

The adapter returns the decoder output of the best cycle as the per-vertex
heatmap; the RGF readout then greedily decodes the cut vertices from it, exactly
like the other learned predictors.  For the standalone baseline the same code
path also reports GenSCO's own best solution size -- the number the official
eval script reports.

Cycle budget (camera-ready protocol):

* standalone baseline: ``C = 1000`` cycles;
* inside RGF: ``C'(n) = max(1, floor(1000 / log2(n)))`` cycles for a sub-instance
  with ``n`` vertices (``cycles=None`` in the constructor).

The RGF cycle rule can be overridden for the "full budget" ablation:
``RGF_GENSCO_FULL_BUDGET=1`` in the environment makes every RGF call spend the
full ``C = 1000`` cycles (``RGF_GENSCO_FULL_BUDGET_CYCLES`` overrides the
number).  The default behaviour stays ``max(1, floor(1000 / log2 n))``.

Padding: the official MIS eval builds the adjacency at the *dataset* maximum
(300 for RB, 800 for ER), so the standalone baseline replays that exactly
(``pad_to``).  Inside RGF the padded size is bucketed to a multiple of
``pad_bucket`` (32) -- a JAX trace is compiled per padded size, so bucketing
keeps the recursion from recompiling hundreds of distinct shapes while the
padding stays within 31 vertices.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENSCO_ROOT = _REPO_ROOT / "third_party" / "GenSCO"

DEFAULT_CYCLES = 1000


def auto_cycles(n: int, budget: int = DEFAULT_CYCLES) -> int:
    """``C'(n) = max(1, floor(budget / log2 n))`` (camera-ready RGF protocol)."""
    n = max(2, int(n))
    return max(1, int(math.floor(budget / math.log2(n))))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GenSCOPredictor:
    """Adapter satisfying the ``rgf.predictor.base.Predictor`` protocol."""

    name = "gensco"

    def __init__(
        self,
        data_root: str | Path,
        dataset: str,
        *,
        cycles: int | None = None,
        sampling_steps: int = 3,
        flip_rate: tuple[float, float] = (0.25, 0.4),
        decode_intermediate: bool = True,
        seed: int = 42,
        ckpt: str | Path | None = None,
        pad_to: int | None = None,
        pad_bucket: int = 32,
        verbose: bool = False,
    ):
        if dataset not in ("rb", "er"):
            raise ValueError(f"unsupported dataset for GenSCO MIS: {dataset!r}")
        self.data_root = Path(data_root)
        self.dataset = dataset
        self.fixed_cycles = None if cycles is None else int(cycles)
        # "Full budget" ablation: RGF_GENSCO_FULL_BUDGET=1 gives every call the
        # full C cycles instead of C'(n)=max(1, floor(1000/log2 n)); default off.
        self.full_budget = os.environ.get("RGF_GENSCO_FULL_BUDGET", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        try:
            self.full_budget_cycles = int(
                os.environ.get("RGF_GENSCO_FULL_BUDGET_CYCLES", DEFAULT_CYCLES)
            )
        except ValueError:
            self.full_budget_cycles = DEFAULT_CYCLES
        if self.fixed_cycles is not None:
            self.budget_mode = "fixed"
        elif self.full_budget:
            self.budget_mode = "full_env"
        else:
            self.budget_mode = "auto_log2"
        self.sampling_steps = int(sampling_steps)
        self.flip_rate = (float(flip_rate[0]), float(flip_rate[1]))
        self.decode_intermediate = bool(decode_intermediate)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.pad_to = None if pad_to is None else int(pad_to)
        self.pad_bucket = max(2, int(pad_bucket))

        if ckpt is None:
            candidate = self.data_root / "checkpoints" / "gensco" / f"mis_{dataset}.ckpt"
            env_override = os.environ.get(f"GENSCO_CKPT_{dataset.upper()}")
            if env_override:
                candidate = Path(env_override)
            ckpt = candidate
        self.ckpt = Path(ckpt)
        if not self.ckpt.is_file():
            raise FileNotFoundError(
                f"GenSCO MIS checkpoint not found: {self.ckpt} "
                "(download ckpts/mis_{rb,er}.ckpt from the official Google Drive folder)"
            )
        self.ckpt_sha256 = file_sha256(self.ckpt)

        self._loaded = False
        self._call_count = 0
        self._stats: list[dict] = []
        self._stats_jsonl = os.environ.get("GENSCO_STATS_JSONL")
        self.model = None
        self.model_config = None

    # --------------------------------------------------------------- loading
    def _load(self) -> None:
        if self._loaded:
            return
        if str(_GENSCO_ROOT) not in sys.path:
            sys.path.insert(0, str(_GENSCO_ROOT))
        import jax  # noqa: PLC0415 - heavy import, only in the gensco env
        import jax.numpy as jnp  # noqa: PLC0415
        from flax import nnx  # noqa: PLC0415
        from models import MISModel, MISModelConfig  # noqa: PLC0415
        from training import load_ckpt  # noqa: PLC0415

        params, _opt_state, _np_rd_state, model_config, _train_config, _step = load_ckpt(
            str(self.ckpt)
        )
        if not isinstance(model_config, MISModelConfig):
            raise TypeError(f"unexpected checkpoint model_config: {type(model_config)!r}")
        if not getattr(model_config, "sigmoid_head", True):
            raise RuntimeError("GenSCO MIS checkpoints must use the sigmoid head")
        model = model_config.construct_model()
        graphdef = nnx.graphdef(model)
        self.model = nnx.merge(graphdef, params)
        self.model_config = model_config

        from decoding.continuous_sampling import sample_euler  # noqa: PLC0415

        model_local = self.model
        steps = jnp.linspace(0.0, 1.0, self.sampling_steps + 1)
        steps = steps.astype(jnp.float32)
        decode_intermediate = self.decode_intermediate

        @jax.jit
        def encode(adjmat, key):
            return model_local.encode(adjmat, key)

        @jax.jit
        def decode(features, solution, adjmat, num_nodes):
            def denoised_fn(x, timestep):
                logits, _ = model_local.decode(features, x, timestep, adjmat, num_nodes)
                return jax.nn.sigmoid(logits)

            if not decode_intermediate:
                pred = sample_euler(denoised_fn, solution, steps)
                cand = jnp.argsort(pred, axis=-1, stable=False, descending=True)
                return pred, (), (cand, ())
            pred, history = sample_euler(denoised_fn, solution, steps, need_history=True)
            history = tuple(history)
            cand = jnp.argsort(pred, axis=-1, stable=False, descending=True)
            hist_cand = tuple(
                jnp.argsort(item, axis=-1, stable=False, descending=True) for item in history
            )
            return pred, history, (cand, hist_cand)

        self.jax = jax
        self.jnp = jnp
        self._encode = encode
        self._decode = decode
        self._loaded = True
        if self.verbose:
            print(
                f"[gensco] loaded {self.ckpt.name} (sha256 {self.ckpt_sha256[:12]}...) "
                f"layers={model_config.num_layers} embed={model_config.embed_dim} "
                f"devices={jax.devices()}",
                flush=True,
            )

    # -------------------------------------------------------------- helpers
    def _n_padded(self, n: int) -> int:
        n = int(n)
        if self.pad_to is not None:
            return max(self.pad_to, n)
        bucket = self.pad_bucket
        return max(bucket, int(math.ceil(n / bucket)) * bucket)

    def _next_seed(self) -> int:
        """Draw a seed from the harness RNG.

        ``scripts/run_rgf.py`` seeds torch per graph before solving; drawing from
        that stream keeps every inference call reproducible yet distinct, which
        is what the other learned predictors do (``torch.manual_seed`` in their
        sampling loops).
        """
        try:
            import torch  # noqa: PLC0415 - optional, only to follow the harness RNG

            return int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
        except Exception:  # noqa: BLE001 - no torch / CPU-only fallback
            self._call_count += 1
            return (self.seed + 888 * self._call_count) % (2**31 - 1)

    # ------------------------------------------------------------ inference
    def _infer(self, g, cycles: int, n_padded: int | None = None) -> dict:
        """Run the official GenSCO search for one graph.

        Returns a dict with ``probs`` (length ``g.n``, the decoder output of the
        best cycle), ``size`` (GenSCO's own best MIS size), ``cycles``,
        ``seconds`` and bookkeeping fields.
        """
        self._load()
        from helpers.mis_transform import edges2adj  # noqa: PLC0415
        from lib import mis_edges2neighbors, mis_greedy_insert  # noqa: PLC0415

        n = int(g.n)
        n_pad = self._n_padded(n) if n_padded is None else max(int(n_padded), n)
        edges = np.ascontiguousarray(np.asarray(g.edges, dtype=np.int32))
        num_nodes = np.array([n], dtype=np.int32)
        num_edges = np.array([edges.shape[0]], dtype=np.int32)
        adjmat = edges2adj(self.jnp.asarray(edges[None]), n_pad, dtype=self.jnp.float16)
        neighbors = mis_edges2neighbors(edges[None], num_nodes, num_edges, tolist=False)

        seed = self._next_seed()
        rng = np.random.default_rng(seed)
        key = rng.integers(0, np.iinfo(np.uint32).max, size=[2], dtype=np.uint32)

        started = time.perf_counter()
        features = self._encode(adjmat, key)

        order = np.broadcast_to(np.arange(n_pad, dtype=np.int32)[None], (1, n_pad))
        order = np.ascontiguousarray(rng.permuted(order, axis=-1))
        solutions, sizes = mis_greedy_insert(
            neighbors, num_nodes, n_pad, order, num_workers=1
        )
        best_size = int(sizes[0])
        best_pred: np.ndarray | None = None

        n_cycles = max(0, int(cycles))
        for _ in range(n_cycles):
            rate = rng.uniform(*self.flip_rate, size=[1, 1]).astype(np.float32)
            flip = rng.binomial(1, rate, size=[1, n_pad]).astype(bool)
            solutions = np.where(solutions, np.logical_xor(solutions, flip), False)

            pred, history, (cand, hist_cand) = self._decode(
                features, solutions, adjmat, num_nodes
            )
            pred_np = np.asarray(pred[0], dtype=np.float32)

            solutions, sizes = mis_greedy_insert(
                neighbors, num_nodes, n_pad, np.asarray(cand, dtype=np.int32), num_workers=1
            )
            size = int(sizes[0])
            if size > best_size:
                best_size, best_pred = size, pred_np
            elif best_pred is None:
                best_pred = pred_np

            for hist_pred, hist_nodes in zip(history, hist_cand):
                _, hist_sizes = mis_greedy_insert(
                    neighbors,
                    num_nodes,
                    n_pad,
                    np.asarray(hist_nodes, dtype=np.int32),
                    num_workers=1,
                )
                if int(hist_sizes[0]) > best_size:
                    best_size = int(hist_sizes[0])
                    best_pred = np.asarray(hist_pred[0], dtype=np.float32)

        seconds = time.perf_counter() - started
        probs = np.full(n, 0.5, dtype=np.float32)
        if best_pred is not None:
            probs[:] = best_pred[:n]
        return {
            "probs": probs,
            "size": best_size,
            "cycles": n_cycles,
            "seconds": seconds,
            "n": n,
            "n_padded": n_pad,
            "seed": seed,
        }

    # ------------------------------------------------------------- protocol
    def _predict_graph(self, g) -> np.ndarray:
        if self.fixed_cycles is not None:
            cycles = self.fixed_cycles
        elif self.full_budget:
            cycles = self.full_budget_cycles
        else:
            cycles = auto_cycles(int(g.n))
        result = self._infer(g, cycles)
        record = {
            "n": result["n"],
            "m": int(g.m),
            "cycles": result["cycles"],
            "n_padded": result["n_padded"],
            "seconds": round(result["seconds"], 4),
            "size": result["size"],
            "budget": self.budget_mode,
        }
        self._stats.append(record)
        if self._stats_jsonl:
            import json  # noqa: PLC0415 - only for the optional provenance log

            with open(self._stats_jsonl, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return result["probs"]

    def predict(self, batch: list[tuple]) -> list[np.ndarray]:
        """Return one boolean array per ``(graph, vc_ids)`` pair, aligned with V_C."""
        out: list[np.ndarray] = []
        for item in batch:
            g, vc_ids = item[0], item[1]
            probs = self._predict_graph(g)
            if vc_ids is None:
                out.append(np.asarray(probs > 0.5, dtype=bool))
            else:
                idx = np.asarray(vc_ids, dtype=np.int64)
                out.append(np.asarray(probs[idx] > 0.5, dtype=bool))
        return out

    # ------------------------------------------------------------ utilities
    def solve_standalone(self, g, cycles: int | None = None) -> dict:
        """One full-graph GenSCO solve (standalone baseline); returns size/time."""
        n_cycles = int(cycles if cycles is not None else (self.fixed_cycles or DEFAULT_CYCLES))
        result = self._infer(g, n_cycles)
        return result

    def warmup(self, graph_sizes: list[int], cycles: int = 2) -> dict:
        """Compile the jitted encoder/decoder for the padded sizes that will be used."""
        self._load()
        from rgf.graph import Graph  # noqa: PLC0415

        padded = sorted({self._n_padded(int(size)) for size in graph_sizes})
        compiled = []
        started = time.perf_counter()
        for n_pad in padded:
            n = int(n_pad)
            if n < 2:
                continue
            # ring graph: enough edges to exercise the encoder/decoder paths
            src = np.arange(n, dtype=np.int64)
            dst = (src + 1) % n
            edges = np.stack([src, dst], axis=1)
            edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
            g = Graph.from_edges(n, edges)
            self._infer(g, cycles, n_padded=n_pad)
            compiled.append(n_pad)
        return {"padded_sizes": compiled, "seconds": round(time.perf_counter() - started, 3)}

    def stats(self) -> dict:
        calls = self._stats
        return {
            "calls": len(calls),
            "budget_mode": self.budget_mode,
            "fixed_cycles": self.fixed_cycles,
            "full_budget_cycles": self.full_budget_cycles,
            "cycles_total": int(sum(call["cycles"] for call in calls)),
            "predict_seconds": round(float(sum(call["seconds"] for call in calls)), 3),
            "n_min": int(min((call["n"] for call in calls), default=0)),
            "n_max": int(max((call["n"] for call in calls), default=0)),
            "cycles_min": int(min((call["cycles"] for call in calls), default=0)),
            "cycles_max": int(max((call["cycles"] for call in calls), default=0)),
            "detail": calls,
        }
