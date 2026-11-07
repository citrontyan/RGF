"""COExpander predictor adapter (ICML 2025, Thinklab-SJTU).

Official code: https://github.com/Thinklab-SJTU/COExpander
Weights:      https://huggingface.co/ML4CO/ML4CO-Bench-101  (``co_expander/*.pt``)
              Google Drive (official README link) is unreachable from this host.

IMPORTANT — process isolation
-----------------------------
The RGF harness runs in the ``difusco`` conda env (torch 1.11), while COExpander
needs torch >= 2.0 + torch-scatter/torch-sparse built for it.  We therefore run
COExpander in a *persistent worker subprocess* of the ``coexpander`` conda env:
the parent (this module) writes the graph to a scratch ``.npz``, sends a JSON
line on the worker's stdin, the worker runs the official model + official
``mis_sparse_decode`` and writes the 0/1 solution to another ``.npz``.  The
model is loaded once per process; the harness still runs unchanged in
``difusco``.

The worker's inference is the official solve path for the paper's main MIS table
(Table 1, ``COExpander (S=1, Ds=20, Is=1)``, no RLSA) — the configuration used
by the official test scripts that ship with these weights
(``ML4CO-Bench-101/test_scripts/co_expander/co_expander_mis.py``):
``INFERENCE_STEP=1, DETERMINATE_STEP=20, SAMPLING_NUM=1, BEAM_SIZE=-1,
USE_RLSA=False``.  (The older COExpander repo's ``test_cm_mis.py`` shows
``DETERMINATE_STEP=1``, which reproduces the ablation row "S=1,Ds=1,Is=1"
instead of the main table.)

    data = env.data_processor.mis_batch_data_process([graph_data], sampling_num=1)
    vars = model.inference_node_sparse_process(*data)
    solution = model.decoder.sparse_decode(vars, *data)[0]

Randomness: the harness seeds torch *inside the difusco process*, which does not
propagate across process boundaries.  Each ``_predict_graph`` call therefore
passes an explicit seed derived from a monotonically increasing call counter
(blake2b of ``counter``), so the 5 RGF runs see independent stochastic samples.
This is documented in the archived ``summary.json``.

Environment variables (all optional):
    COEXPANDER_ROOT    checkout of the official repo   (default /mnt/d/repos/COExpander)
    COEXPANDER_PYTHON  python of the coexpander env     (default: miniconda env coexpander)
    COEXPANDER_DEVICE  torch device                     (default cuda:0)
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from rgf.graph import Graph

_DEFAULT_ROOT = "/mnt/d/repos/COExpander"
_DEFAULT_PY = "/home/pengb/miniconda3/envs/coexpander/bin/python"

# official test_scripts/test_cm_mis.py -> weights/cm_mis_{rb-small,er-700-800}_sparse.pt
_WEIGHTS = {
    "rb": "weights/coexpander_mis_rb-small_sparse.pt",
    "er": "weights/coexpander_mis_er-700-800_sparse.pt",
}
# Official settings for the paper's main MIS table (Table 1, COExpander (S=1,
# Ds=20, Is=1), no RLSA) and for the official test scripts shipped with these
# weights: ML4CO-Bench-101 ``test_scripts/co_expander/co_expander_mis.py`` uses
# INFERENCE_STEP=1, DETERMINATE_STEP=20, SAMPLING_NUM=1, BEAM_SIZE=-1,
# USE_RLSA=False.  (The older COExpander repo's ``test_cm_mis.py`` still says
# DETERMINATE_STEP=1, which reproduces the paper's ablation row
# "S=1,Ds=1,Is=1" — 20.6% drop on MIS-ER — not the main table.)
_OFFICIAL = dict(
    inference_steps=1,
    determinate_steps=20,
    beam_size=-1,
    use_rlsa=False,
    block_layers=[2, 4, 4, 2],
    hidden_dim=256,
    sampling_num=1,
)
_RLSA_KWARGS = {  # passed by the official test script; unused with use_rlsa=False
    "use_rlsa": False,
    "rlsa_tau": 0.01,
    "rlsa_d": 5,
    "rlsa_k": 1000,
    "rlsa_t": 1000,
    "rlsa_beta": 1.02,
    "rlsa_alpha": 0.3,
}


# --------------------------------------------------------------------------- #
# worker side (runs in the coexpander env; imports torch lazily)
# --------------------------------------------------------------------------- #
def _worker_build_model(weight_path: str, device: str):
    from co_expander import COExpanderCMModel, COExpanderDecoder, COExpanderEnv, GNNEncoder

    model = COExpanderCMModel(
        env=COExpanderEnv(task="MIS", mode="solve", sparse_factor=1, device=device),
        encoder=GNNEncoder(
            task="MIS", sparse=True, block_layers=_OFFICIAL["block_layers"],
            hidden_dim=_OFFICIAL["hidden_dim"],
        ),
        decoder=COExpanderDecoder(decode_kwargs=dict(_RLSA_KWARGS)),
        inference_steps=_OFFICIAL["inference_steps"],
        determinate_steps=_OFFICIAL["determinate_steps"],
        beam_size=_OFFICIAL["beam_size"],
        weight_path=weight_path,
    )
    model.eval()
    return model


def _worker_solve(model, n: int, edges: np.ndarray, seed: int) -> np.ndarray:
    """Official solve path for one graph; returns a 0/1 int8 array of length n."""
    import torch
    from ml4co_kit import MISGraphData

    torch.manual_seed(int(seed) % (2**63 - 1))
    n = int(n)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    # Official test files store ``edge_index`` as (2, 2m + n): every undirected
    # edge in both directions plus one self-loop per vertex (verified by parsing
    # ``mis_rb-small_kamis-60s_20.090.txt``: 6,494 pairs = 2*3,127 + 240 loops,
    # and 89,624 = 2*44,427 + 770 loops on ``mis_er-700-800``).  Reproduce that
    # convention exactly; ``mis_sparse_process`` does not add loops itself.
    loops = np.arange(n, dtype=np.int64)
    both = np.concatenate(
        [edges.T, edges[:, ::-1].T, np.stack([loops, loops], axis=0)], axis=1
    )
    graph_data = MISGraphData()
    graph_data.nodes_num = n
    graph_data.edge_index = both
    data = model.env.data_processor.mis_batch_data_process(
        graph_data=[graph_data], sampling_num=_OFFICIAL["sampling_num"],
    )
    with torch.no_grad():
        vars = model.inference_node_sparse_process(*data)
        solutions = model.decoder.sparse_decode(vars, *data)
    sol = np.asarray(solutions[0]).reshape(-1)
    return sol.astype(np.int8)


def _run_worker(root: str, dataset: str, device: str) -> int:
    """Persistent worker: one JSON request per line, one JSON reply per line."""
    root_path = Path(root)
    weight_path = str(root_path / _WEIGHTS[dataset])
    sys.path.insert(0, str(root_path))
    model = _worker_build_model(weight_path, device)
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("op") == "quit":
                out.write(json.dumps({"ok": True}) + "\n")
                out.flush()
                return 0
            with np.load(req["graph"]) as d:
                n = int(d["n"])
                edges = np.asarray(d["edges"])
            sol = _worker_solve(model, n, edges, int(req.get("seed", 1234)))
            np.savez_compressed(req["out"], solution=sol)
            out.write(json.dumps({"ok": True, "n": n, "size": int(sol.sum())}) + "\n")
        except Exception as exc:  # noqa: BLE001 - report to parent, keep serving
            out.write(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        out.flush()
    return 0


# --------------------------------------------------------------------------- #
# parent side
# --------------------------------------------------------------------------- #
class CoExpanderPredictor:
    """Adapter satisfying the ``rgf.predictor.base.Predictor`` protocol."""

    name = "coexpander"

    def __init__(self, data_root, dataset: str):
        if dataset not in _WEIGHTS:
            raise ValueError(f"unsupported dataset for COExpander: {dataset}")
        self.dataset = dataset
        self.root = Path(os.environ.get("COEXPANDER_ROOT", _DEFAULT_ROOT))
        weight = self.root / _WEIGHTS[dataset]
        if not weight.exists():
            raise FileNotFoundError(
                f"COExpander weight not found: {weight} (set COEXPANDER_ROOT / see summary.json)"
            )
        self.weight_path = weight
        self.python = os.environ.get("COEXPANDER_PYTHON", _DEFAULT_PY)
        if not Path(self.python).exists():
            raise FileNotFoundError(f"coexpander python not found: {self.python}")
        self.device = os.environ.get("COEXPANDER_DEVICE", "cuda:0")
        self.data_root = Path(data_root)
        self._scratch = self.data_root / "results" / "coexpander_cache" / dataset
        self._scratch.mkdir(parents=True, exist_ok=True)
        self._proc: subprocess.Popen | None = None
        self._counter = 0
        self._stderr_log = self._scratch / "worker.log"
        self._stderr_handle = None

    # ------------------------------------------------------------- lifecycle
    def _stderr_tail(self, n: int = 4000) -> str:
        try:
            return self._stderr_log.read_text(encoding="utf-8", errors="replace")[-n:]
        except Exception:  # noqa: BLE001
            return ""

    def _ensure_worker(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "8")
        self._stderr_log = self._scratch / "worker.log"
        self._stderr_handle = open(self._stderr_log, "a", encoding="utf-8")
        self._stderr_handle.write(f"\n===== worker start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        self._stderr_handle.flush()
        self._proc = subprocess.Popen(
            [self.python, os.path.abspath(__file__), "--serve",
             "--root", str(self.root), "--dataset", self.dataset,
             "--device", self.device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr_handle,
            text=True, bufsize=1, env=env,
        )
        # fail fast if the worker dies on start-up
        if self._proc.poll() is not None:
            raise RuntimeError(f"COExpander worker failed to start:\n{self._stderr_tail()}")

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                self._proc.kill()
        self._proc = None
        handle = getattr(self, "_stderr_handle", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass
            self._stderr_handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------- inference
    def _graph_key(self, g: Graph) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(np.int64(g.n).tobytes())
        h.update(np.asarray(g.edges, dtype=np.int64).tobytes())
        return h.hexdigest()

    def _predict_graph(self, g: Graph) -> np.ndarray:
        """Return the official COExpander 0/1 solution as a float heatmap.

        COExpander's solve path returns a determined solution (no heatmap), so
        the adapter maps it to a 0/1 float array; the RGF harness then decodes
        it with ``decode_mis`` exactly like the other predictors.

        No solution cache: COExpander's inference is stochastic (random initial
        state + Bernoulli consistency sampling), and the RGF protocol needs each
        of the ``--runs`` repetitions to be an independent sample.  Caching by
        graph hash would make runs 2..N identical (std = 0 by construction).
        Only the *graph serialisation* is reused (same subgraph -> same file).
        """
        self._ensure_worker()
        self._counter += 1
        seed = int.from_bytes(
            hashlib.blake2b(f"coexpander:{self._counter}".encode(), digest_size=8).digest(),
            "big",
        ) % (2**63 - 1)
        key = self._graph_key(g)
        graph_path = self._scratch / f"{key}.graph.npz"
        sol_path = self._scratch / "last_solution.npz"
        if not graph_path.exists():
            np.savez_compressed(graph_path, n=np.int64(g.n),
                                edges=np.asarray(g.edges, dtype=np.int64))
        req = json.dumps({"graph": str(graph_path), "out": str(sol_path), "seed": seed})
        self._proc.stdin.write(req + "\n")
        self._proc.stdin.flush()
        reply = self._proc.stdout.readline()
        if not reply:
            raise RuntimeError(f"COExpander worker died:\n{self._stderr_tail()}")
        msg = json.loads(reply)
        if not msg.get("ok"):
            raise RuntimeError(f"COExpander worker error: {msg.get('error')}")
        with np.load(sol_path) as d:
            sol = np.asarray(d["solution"]).astype(np.int8)
        if sol.size != g.n:
            raise RuntimeError(f"solution size {sol.size} != n {g.n}")
        return sol.astype(np.float64)

    def predict(self, batch: list[tuple]) -> list[np.ndarray]:
        """Official ``predict`` contract (see ``rgf/predictor/fastt2t.py``)."""
        out = []
        for item in batch:
            g, vc_ids = item[0], item[1]
            probs = self._predict_graph(g)
            if vc_ids is None:
                out.append(np.asarray(probs > 0.5, dtype=bool))
            else:
                idx = np.asarray(vc_ids, dtype=np.int64)
                out.append(np.asarray(probs[idx] > 0.5, dtype=bool))
        return out


# --------------------------------------------------------------------------- #
# entry point: worker mode
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--root", default=os.environ.get("COEXPANDER_ROOT", _DEFAULT_ROOT))
    ap.add_argument("--dataset", choices=sorted(_WEIGHTS), required=True)
    ap.add_argument("--device", default=os.environ.get("COEXPANDER_DEVICE", "cuda:0"))
    args = ap.parse_args(argv)
    if not args.serve:
        ap.error("only --serve mode is supported as a standalone entry point")
    return _run_worker(args.root, args.dataset, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
