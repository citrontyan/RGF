"""KaMIS baseline (paper Table 1).

The paper uses KaMIS for MIS directly, and for DS / VC after reducing those
instances to MIS.  The VC reduction is the standard complement relation
(minimum vertex cover = complement of a maximum independent set); the DS
reduction is not a textbook identity and the paper does not spell it out, so it
is deliberately left unimplemented here (``docs/decisions.md`` item D-6).

KaMIS is a Linux project and this workstation builds it inside WSL
(``scripts/build_kamis_wsl.sh``), so a non-``.exe`` executable is invoked
through ``wsl -e`` with its paths translated to ``/mnt/<drive>/...``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from rgf.graph import Graph
from rgf.wsl import is_native_exe, on_windows, to_wsl_path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANDIDATES = [
    _REPO_ROOT / "third_party" / "KaMIS" / "build-wsl" / "redumis",
    _REPO_ROOT / "third_party" / "KaMIS" / "build" / "redumis.exe",
    _REPO_ROOT / "third_party" / "KaMIS" / "redumis.exe",
]


def find_kamis() -> str | None:
    env = os.environ.get("KAMIS_BIN")
    if env and Path(env).exists():
        return env
    for cand in _CANDIDATES:
        if cand.exists():
            return str(cand)
    return shutil.which("redumis") or shutil.which("kamis")


def kamis_mis(
    g: Graph,
    time_limit_s: float | None = None,
    seed: int = 1,
    executable: str | None = None,
) -> np.ndarray:
    """Run KaMIS (``redumis``) on a MIS instance and return its labelling."""
    exe = executable or find_kamis()
    if not exe:
        raise FileNotFoundError("KaMIS executable not found; build it in WSL or set KAMIS_BIN")
    with tempfile.TemporaryDirectory(prefix="rgf_kamis_") as tmp:
        graph_file = os.path.join(tmp, "g.graph")
        out_file = os.path.join(tmp, "g.is")
        _write_metis_graph(g, graph_file)
        args = [graph_file, f"--output={out_file}", f"--seed={seed}"]
        if time_limit_s is not None:
            args.append(f"--time_limit={float(time_limit_s)}")
        if is_native_exe(exe) or not on_windows():
            # native Windows build, or we are already inside the WSL environment
            cmd = [exe] + args
        else:
            cmd = [
                "wsl",
                "-e",
                to_wsl_path(exe),
                to_wsl_path(graph_file),
                f"--output={to_wsl_path(out_file)}",
                f"--seed={seed}",
            ]
            if time_limit_s is not None:
                cmd.append(f"--time_limit={float(time_limit_s)}")
        subprocess.run(cmd, check=True, capture_output=True)
        values = np.loadtxt(out_file, dtype=np.int64).reshape(-1)
    labels = np.zeros(g.n, dtype=np.int8)
    # KaMIS' --output file is a 0/1 indicator vector over all vertices (verified
    # on the local build: 40 lines of 0/1 for a 40-vertex graph).  Older layouts
    # list the chosen vertex ids instead, so accept both.
    if values.size == g.n and set(np.unique(values)).issubset({0, 1}):
        labels = values.astype(np.int8)
    else:
        labels[values] = 1
    return labels


def _write_metis_graph(g: Graph, path: str) -> None:
    """METIS adjacency format (1-based ids, no self loops)."""
    lines = [f"{g.n} {g.m}"]
    for v in range(g.n):
        nb = g.neighbors(v)
        lines.append(" ".join(str(int(u) + 1) for u in nb) if nb.size else "")
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def vc_via_mis(g: Graph, mis_labels: np.ndarray) -> np.ndarray:
    """Complement relation: ``V \\ S`` is a vertex cover iff ``S`` is independent."""
    return (1 - mis_labels).astype(np.int8)


def ds_via_mis(*_args, **_kwargs):
    raise NotImplementedError(
        "the paper does not state the DS -> MIS reduction and there is no textbook identity; "
        "see docs/decisions.md item D-6"
    )
