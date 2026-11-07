"""Guard: the in-process METIS call must agree with the ``gpmetis`` executable.

``MetisPartitioner(backend="ctypes")`` calls ``METIS_PartGraphKway`` in-process
instead of writing a graph file and shelling out.  The acceleration is only
allowed to change the runtime, so the labels have to match exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from rgf.graph import Graph
from rgf.partition.metis import MetisPartitioner, find_gpmetis, find_libmetis

pytestmark = pytest.mark.skipif(
    find_gpmetis() is None or find_libmetis() is None,
    reason="needs both gpmetis and libmetis.so",
)


def _rb_like(n: int, seed: int) -> Graph:
    rng = np.random.default_rng(seed)
    iu, ju = np.triu_indices(n, k=1)
    mask = rng.random(iu.size) < 0.06
    return Graph.from_edges(n, np.stack([iu[mask], ju[mask]], axis=1))


def test_backends_produce_the_same_cut() -> None:
    g = _rb_like(240, 3)
    a = MetisPartitioner(seed=0, backend="gpmetis").partition(g, np.random.default_rng(0))
    b = MetisPartitioner(seed=0, backend="ctypes").partition(g, np.random.default_rng(0))
    assert a.size == b.size
    assert np.array_equal(a.edge_mask, b.edge_mask)
    assert np.array_equal(a.vc_ids, b.vc_ids)


@pytest.mark.parametrize("seed", [0, 7])
def test_backends_produce_the_same_labels(seed: int) -> None:
    import subprocess
    import tempfile
    from pathlib import Path

    from rgf.partition.metis import _partition_inproc, _write_metis_graph
    from rgf.wsl import wrap_command

    g = _rb_like(150, seed + 1)
    with tempfile.TemporaryDirectory(prefix="rgf_metis_test_") as tmp:
        graph_file = str(Path(tmp) / "g.graph")
        _write_metis_graph(g, graph_file)
        cmd = wrap_command(find_gpmetis(), [graph_file, "2", f"-seed={seed}"], path_args={0})
        subprocess.run(cmd, check=True, capture_output=True)
        from_file = np.loadtxt(graph_file + ".part.2", dtype=np.int64)
    in_proc = _partition_inproc(g, 2, seed).astype(np.int64)
    assert np.array_equal(from_file, in_proc)
