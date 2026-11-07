"""下载 SNAP 边表并转成本仓库的 **无向简单图** npz（MIS 口径）。

用法（WSL，CPU）：

    ~/miniconda3/envs/difusco/bin/python scripts/fetch_snap.py --only com-DBLP
    ~/miniconda3/envs/difusco/bin/python scripts/fetch_snap.py            # 全部

落盘位置（全部在数据盘，仓库里只放本脚本）：

    /mnt/e/data/rgf/external/raw/<file>.txt.gz     原始压缩边表
    /mnt/e/data/rgf/external/<name>.npz            rgf 图缓存（n + edges，见 rgf/datasets/io.py）
    /mnt/e/data/rgf/external/manifest.json         清单（含各项计数与对官方数字的比对）

口径（与 scripts/fetch_dimacs10.py 一致，但 SNAP 原始表更脏）：

* SNAP 一行 ``u v``（0 基）就是一条**有向弧**。无向图必须自己对称化；
  若原表已经把无向边写了两次，去重后仍应只剩一条，所以“对称化 + 去重合并”
  是同一步：无序对 (min, max) 去重。
* 自环丢弃（MIS 里没有意义，且 ``Graph.from_edges`` 也会丢）。
* ``n`` = **原始表里出现过的不同顶点 id 数**（不是 ``max_id + 1``）。SNAP 的
  ``Nodes`` 就是这个量：roadNet-CA 的 id 不连续（max id = 1,971,280），官方
  ``Nodes`` = 1,965,206 = 出现过的 id 个数，不是 max+1。顶点随即重编号成
  紧凑的 ``0..n-1``（保序），与 DIMACS10 那些实例的口径一致；原表 id 只用于
  解析，不落盘。

比对口径（关键，必须说清；下表每个实例都逐条按 SNAP 自己那张统计表核过）：

* **存哪一个变体：按官方数字描述的那个来**。
  - *无向* 数据集（roadNet-*、as-Skitter、com-*）：SNAP 页面的 ``Nodes``/``Edges``
    是**整表**数字（roadNet-CA 官方 1,965,206 / 2,766,607，而它自己那行
    "in largest WCC" 是 1,957,027 / 2,760,388），所以存整表，``n``/``m`` 才能与
    文献对上；
  - *有向* 数据集（cit-Patents / wiki-Talk / web-* / amazon0601 / soc-LiveJournal1）：
    官方 ``Nodes``/``Edges`` 是**最大弱连通分量**的口径，整表要大得多
    （cit-Patents 整表 6,009,555 点，最大 WCC 3,764,117），所以只存最大 WCC。
  `--whole` 可强制整表，`--wcc-only` 可强制所有实例只留最大 WCC。
  两个变体的数字都写进清单（``whole_*`` / ``wcc_*``）。
* ``undirected`` 类：官方 ``Edges`` 就是无向边数 → 要求 **等于**。
* ``directed`` 类：官方 ``Edges`` 数的是**有向弧**，而本仓库存的是无向简单图，
  去重后无向对数必然 **≤** 官方弧数。所以对 ``m`` 比对的是"不超过"，并在清单里
  同时记下原始弧数（``raw_edge_entries``）与最大 WCC 的无向对数（``wcc_m``）作参照。
  注意**不能**用"官方弧数 ÷ 2"当等式基准：双向都写的 2-环贡献 2 条弧却只有 1 个
  无序对，而单向弧只贡献 1 条弧，两类混在一起，÷2 没有意义（实测 wiki-Talk
  4,656,682 > 5,018,445/2）。
* ``n`` 在 ``directed`` 类上仍可能差一点：官方数字来自 SNAP 论文所用的那一版数据，
  与现在放出的 raw 文件不是逐字节一致（cit-Patents 官方 3,774,768，而 raw 文件的最大
  WCC 只有 3,764,117，差 10,651）。这种差额脚本据实报 MISMATCH，不做掩饰。
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components as _cc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgf.datasets.io import save_graph_npz  # noqa: E402
from rgf.graph import Graph  # noqa: E402

BASE = "https://snap.stanford.edu/data"
# Windows 侧数据根是 E:\data\rgf；WSL 侧必须写成 /mnt/e/data/rgf，否则
# pathlib 会把 "E:/..." 当成相对路径（本脚本主要在 WSL 里跑）。
DEFAULT_DATA_ROOT = Path("/mnt/e/data/rgf") if Path("/mnt/e").exists() else Path(r"E:\data\rgf")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

# kind: "undirected" -> 官方 Edges 即无向边数（要求相等）
#       "directed"   -> 官方 Edges 数有向弧（要求 n/m 按"最大 WCC"口径核对）
# url 是 SNAP data 目录下的相对路径（经 HTTP HEAD 逐个核对过）。
# n/m 取自各数据集自己的页面的 Dataset statistics 表（2026-10 抓的，见
# https://snap.stanford.edu/data/index.html 的逐页 _snap_published 记录）。
# wcc_* 是该页 "Nodes/Edges in largest WCC" 两行，用于按 WCC 口径比对。
# variant: 官方 n/m 描述的是哪一个图 —— "whole"（整表）或 "wcc"（最大弱连通分量）。
INSTANCES: dict[str, dict] = {
    "roadNet-CA": {"url": "roadNet-CA.txt.gz", "gzip": True, "n": 1_965_206, "m": 2_766_607,
                   "kind": "undirected", "variant": "whole", "wcc_n": 1_957_027, "wcc_m": 2_760_388},
    "roadNet-TX": {"url": "roadNet-TX.txt.gz", "gzip": True, "n": 1_379_917, "m": 1_921_660,
                   "kind": "undirected", "variant": "whole", "wcc_n": 1_351_137, "wcc_m": 1_879_201},
    "roadNet-PA": {"url": "roadNet-PA.txt.gz", "gzip": True, "n": 1_088_092, "m": 1_541_898,
                   "kind": "undirected", "variant": "whole", "wcc_n": 1_087_562, "wcc_m": 1_541_514},
    "cit-Patents": {"url": "cit-Patents.txt.gz", "gzip": True, "n": 3_774_768, "m": 16_518_948,
                    "kind": "directed", "variant": "wcc"},
    "wiki-Talk": {"url": "wiki-Talk.txt.gz", "gzip": True, "n": 2_394_385, "m": 5_021_410,
                  "kind": "directed", "variant": "wcc", "wcc_n": 2_388_953, "wcc_m": 5_018_445},
    "as-Skitter": {"url": "as-skitter.txt.gz", "gzip": True, "n": 1_696_415, "m": 11_095_298,
                   "kind": "undirected", "variant": "whole", "wcc_n": 1_694_616, "wcc_m": 11_094_209},
    "web-Google": {"url": "web-Google.txt.gz", "gzip": True, "n": 875_713, "m": 5_105_039,
                   "kind": "directed", "variant": "wcc", "wcc_n": 855_802, "wcc_m": 5_066_842},
    "web-BerkStan": {"url": "web-BerkStan.txt.gz", "gzip": True, "n": 685_230, "m": 7_600_595,
                     "kind": "directed", "variant": "wcc", "wcc_n": 654_782, "wcc_m": 7_499_425},
    "web-NotreDame": {"url": "web-NotreDame.txt.gz", "gzip": True, "n": 325_729, "m": 1_497_134,
                      "kind": "directed", "variant": "wcc", "wcc_n": 325_729, "wcc_m": 1_497_134},
    "web-Stanford": {"url": "web-Stanford.txt.gz", "gzip": True, "n": 281_903, "m": 2_312_497,
                     "kind": "directed", "variant": "wcc"},
    "amazon0601": {"url": "amazon0601.txt.gz", "gzip": True, "n": 403_394, "m": 3_387_388,
                   "kind": "directed", "variant": "wcc", "wcc_n": 403_364, "wcc_m": 3_387_224},
    "soc-LiveJournal1": {"url": "soc-LiveJournal1.txt.gz", "gzip": True, "n": 4_847_571, "m": 68_993_773,
                         "kind": "directed", "variant": "wcc", "wcc_n": 4_843_953, "wcc_m": 68_983_820},
    # 用户表里“文件名待查”的四个：真名在 bigdata/communities/ 下，且是 *.ungraph.txt.gz。
    # 它们本身就是无向图（每个无序对只写一次），官方 Edges 就是无向边数。
    "com-LiveJournal": {"url": "bigdata/communities/com-lj.ungraph.txt.gz", "gzip": True,
                        "n": 3_997_962, "m": 34_681_189, "kind": "undirected", "variant": "whole",
                        "wcc_n": 3_997_962, "wcc_m": 34_681_189},
    "com-DBLP": {"url": "bigdata/communities/com-dblp.ungraph.txt.gz", "gzip": True,
                 "n": 317_080, "m": 1_049_866, "kind": "undirected", "variant": "whole",
                 "wcc_n": 317_080, "wcc_m": 1_049_866},
    "com-Amazon": {"url": "bigdata/communities/com-amazon.ungraph.txt.gz", "gzip": True,
                   "n": 334_863, "m": 925_872, "kind": "undirected", "variant": "whole",
                   "wcc_n": 334_863, "wcc_m": 925_872},
    "com-Youtube": {"url": "bigdata/communities/com-youtube.ungraph.txt.gz", "gzip": True,
                    "n": 1_134_890, "m": 2_987_624, "kind": "undirected", "variant": "whole",
                    "wcc_n": 1_134_890, "wcc_m": 2_987_624},
}

_CHUNK = 1 << 25  # 32 MiB 解压后读一块


# --------------------------------------------------------------------- download
def _head_size(url: str) -> int | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def _fetch_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _download_parallel(url: str, tmp: Path, total: int, chunks: int) -> None:
    """Range 分块并行下载。

    SNAP 的服务器对单连接限速很不稳定（实测同一个文件 240 KB/s ~ 7 MB/s 都出现过），
    单连接下 web-BerkStan 只剩 30 KB/s、cit-Patents 要 320 s；分块并行 + 整体重试
    能绕开单条连接的慢通道。任何一块失败就整体重来一次（最多 5 轮，块数翻倍）。
    """
    step = (total + chunks - 1) // chunks
    bounds = [(i * step, min(total - 1, (i + 1) * step - 1)) for i in range(chunks) if i * step <= total - 1]
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            with ThreadPoolExecutor(max_workers=len(bounds)) as ex:
                parts = list(ex.map(lambda b: _fetch_range(url, b[0], b[1]), bounds))
            if sum(len(p) for p in parts) != total:
                raise IOError(f"分块字节数不符：{sum(len(p) for p in parts)} != {total}")
            with open(tmp, "wb") as fh:
                for p in parts:
                    fh.write(p)
            return
        except Exception as exc:  # 换更多块再试，慢通道会被拆散
            last_exc = exc
            print(f"    分块下载失败（第 {attempt + 1} 轮，{len(bounds)} 块）：{type(exc).__name__}: {exc}", flush=True)
            chunks = min(chunks * 2, 64)
            step = (total + chunks - 1) // chunks
            bounds = [(i * step, min(total - 1, (i + 1) * step - 1)) for i in range(chunks) if i * step <= total - 1]
    raise IOError(f"分块下载重试 5 轮仍失败：{last_exc}")


def download(url: str, dest: Path, force: bool = False, chunks: int = 16) -> Path:
    """下载到 ``dest``（先写 ``.part`` 再改名；已存在则跳过，除非 ``force``）。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"  已存在 {dest}（{dest.stat().st_size/1e6:.1f} MB），跳过下载", flush=True)
        return dest
    total = _head_size(url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.perf_counter()
    if total and total > (1 << 20) and chunks > 1:
        print(f"  GET {url}（{total/1e6:.1f} MB，{chunks} 个 Range 并行）", flush=True)
        _download_parallel(url, tmp, total, chunks)
    else:
        print(f"  GET {url}（单连接）", flush=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=600) as r, open(tmp, "wb") as fh:
            got = 0
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                fh.write(b)
                got += len(b)
                if got % (64 << 20) < (1 << 22):
                    print(f"    {got/1e6:.1f} MB", flush=True)
    got = tmp.stat().st_size
    if total and got != total:
        raise IOError(f"下载不完整：{got} != Content-Length {total}")
    tmp.replace(dest)
    print(f"  -> {dest}（{got/1e6:.1f} MB，{time.perf_counter()-t0:.0f}s）", flush=True)
    return dest


# ------------------------------------------------------------------------ parse
def parse_snap(src: Path, gzipped: bool = True) -> dict:
    """把 SNAP 边表读成 (raw_edges, raw_lines, comment_lines)。

    只保留每行前两个整数；SNAP 注释行以 ``#`` 开头（可能有前导空格）。
    """
    opener = gzip.open if gzipped else open
    parts: list[np.ndarray] = []
    n_lines = 0
    n_comment = 0
    n_short = 0
    carry = b""
    with opener(src, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            buf = carry + chunk
            cut = buf.rfind(b"\n")
            if cut < 0:
                carry = buf
                continue
            line_chunk, carry = buf[: cut + 1], buf[cut + 1 :]
            n_lines += line_chunk.count(b"\n")
            if b"#" in line_chunk:
                keep = []
                for ln in line_chunk.split(b"\n"):
                    if ln.lstrip().startswith(b"#"):
                        n_comment += 1
                    else:
                        keep.append(ln)
                line_chunk = b"\n".join(keep)
            a = np.fromstring(line_chunk, dtype=np.float64, sep=" ")
            if a.size % 2:
                a = a[:-1]  # 行尾多余 token（SNAP 有几张带权表），丢掉半个
                n_short += 1
            if a.size:
                parts.append(a.reshape(-1, 2).astype(np.int64))
    if carry.strip():
        n_lines += 1
        ln = carry
        if ln.lstrip().startswith(b"#"):
            n_comment += 1
        else:
            a = np.fromstring(ln, dtype=np.float64, sep=" ")
            if a.size % 2:
                a = a[:-1]
            if a.size:
                parts.append(a.reshape(-1, 2).astype(np.int64))
    edges = np.concatenate(parts) if parts else np.empty((0, 2), dtype=np.int64)
    return {"edges": edges, "raw_lines": n_lines, "comment_lines": n_comment, "odd_token_lines": n_short}


def wcc_stats(g: Graph) -> dict:
    """最大弱连通分量的 (顶点数, 内部边数, 分量数)。"""
    if g.m == 0:
        return {"wcc_n": g.n, "wcc_m": 0, "n_components": g.n}
    adj = sp.csr_matrix((np.ones(g.m, dtype=np.int8), (g.edges[:, 0], g.edges[:, 1])), shape=(g.n, g.n))
    ncomp, lab = _cc(adj, directed=False)
    sizes = np.bincount(lab)
    k = int(np.argmax(sizes))
    mask = lab == k
    m_in = int((mask[g.edges[:, 0]] & mask[g.edges[:, 1]]).sum())
    return {"wcc_n": int(sizes[k]), "wcc_m": m_in, "n_components": int(ncomp)}


def convert(name: str, src: Path, out_dir: Path, spec: dict, force: bool = False,
            largest_wcc: bool | None = None) -> dict:
    """解析 + 转换 + 与官方数字比对。

    ``largest_wcc=None`` 时按官方口径自动选：官方 ``n``/``m`` 描述整表的实例
    （``spec['variant'] == 'whole'``，即全部无向数据集）存整表，官方数字描述最大
    弱连通分量的实例（有向数据集）只存 WCC。显式传 True/False 可强制切换。
    """
    variant = spec.get("variant", "whole")
    if largest_wcc is None:
        largest_wcc = variant == "wcc"
    npz = out_dir / f"{name}.npz"
    pub_n, pub_m = spec["n"], spec["m"]
    if npz.exists() and not force:
        with np.load(npz) as z:
            g = Graph.from_edges(int(z["n"]), z["edges"].astype(np.int64))
        m_ok = g.m == pub_m if spec["kind"] == "undirected" else g.m <= pub_m
        n_ok = g.n == pub_n
        print(f"[{name}] 已转换，跳过：{npz}（现 |V|={g.n:,} |E|={g.m:,}；"
              f"n {'MATCH' if n_ok else 'MISMATCH'} m {'MATCH' if m_ok else 'MISMATCH'}）", flush=True)
        return {
            "name": name,
            "source_url": f"{BASE}/{spec['url']}",
            "npz": str(npz),
            "npz_bytes": int(npz.stat().st_size),
            "kind": spec["kind"],
            "variant": "largest_wcc" if largest_wcc else "whole_file",
            "published_variant": variant,
            "n": int(g.n),
            "m": int(g.m),
            "published_n": pub_n,
            "published_m": pub_m,
            "n_match": bool(n_ok),
            "m_match": bool(m_ok),
            "skipped": True,
        }

    t0 = time.perf_counter()
    raw = parse_snap(src, gzipped=spec.get("gzip", True))
    e = raw["edges"]
    if e.size == 0:
        raise ValueError("没有解析出任何边")

    n_raw_entries = int(e.shape[0])
    self_loops = int((e[:, 0] == e[:, 1]).sum())
    max_id = int(max(e[:, 0].max(), e[:, 1].max()))

    # 顶点重编号：SNAP 的 id 有空洞（com-DBLP 最大 id 425,956 而只有 317,080 个顶点），
    # 官方 Nodes 数的是“出现过的 id 个数”，所以这里映射成紧凑 0..n-1，
    # n 才与官方数字可比。保序映射（sorted unique），重编号是双射。
    uniq_ids, inverse = np.unique(e, return_inverse=True)
    n = int(uniq_ids.size)
    e = inverse.reshape(-1, 2).astype(np.int64)
    n_unused_ids = max_id + 1 - n

    g_full = Graph.from_edges(n, e)  # 去自环 + 无序对去重（对称化在本步完成）
    wcc = wcc_stats(g_full)
    g = g_full
    if largest_wcc:
        adj = sp.csr_matrix(
            (np.ones(g_full.m, dtype=np.int8), (g_full.edges[:, 0], g_full.edges[:, 1])), shape=(g_full.n, g_full.n)
        )
        _, lab = _cc(adj, directed=False)
        sizes = np.bincount(lab)
        keep = np.flatnonzero(lab == int(np.argmax(sizes)))
        g, _, _ = g_full.induced(keep)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_graph_npz(g, npz)
    elapsed = time.perf_counter() - t0

    n_ok = g.n == pub_n
    if spec["kind"] == "undirected":
        m_ok = g.m == pub_m
        cmp_note = "官方 Edges 即无向边数 → 要求等于"
    else:
        m_ok = g.m <= pub_m
        cmp_note = ("官方 Edges 是最大 WCC 的有向弧数；本仓库存无向简单图，去重后无向对数应 ≤ 官方弧数"
                    "（不是 官方/2：双向都写的 2-环贡献 2 条弧但只有 1 个无序对，而单向弧也要算一份上界）")
    print(
        f"[{name}] 原始行={raw['raw_lines']:,}（注释 {raw['comment_lines']:,}）"
        f" 原始弧条目={n_raw_entries:,} 自环={self_loops:,}"
        f" | 最大原始 id={max_id:,}（空洞 {n_unused_ids:,}）"
        f" | 整表 |V|={g_full.n:,} |E|={g_full.m:,}（分量数 {wcc['n_components']:,}）"
        f" | 最大 WCC |V|={wcc['wcc_n']:,} |E|={wcc['wcc_m']:,}（无向对，参照）"
        f" | 实存 {g.n:,} / {g.m:,}（{'最大 WCC' if largest_wcc else '整表'}）"
        f" |V| 对官方：{'MATCH' if n_ok else 'MISMATCH'}"
        f" | |E| 对官方：{'MATCH' if m_ok else 'MISMATCH'}"
        f"（官方 n={pub_n:,} m={pub_m:,}；{cmp_note}）用时 {elapsed:.0f}s",
        flush=True,
    )
    if not n_ok:
        print(f"[{name}] !!! MISMATCH n: 实存 {g.n:,} != 官方 {pub_n:,} !!!", flush=True)
    if not m_ok:
        print(f"[{name}] !!! MISMATCH m: 实存 {g.m:,} 与官方 {pub_m:,} 不符（{cmp_note}） !!!", flush=True)

    return {
        "name": name,
        "source_url": f"{BASE}/{spec['url']}",
        "raw_file": src.name,
        "raw_bytes": int(src.stat().st_size),
        "npz": str(npz),
        "npz_bytes": int(npz.stat().st_size),
        "kind": spec["kind"],
        "raw_lines": raw["raw_lines"],
        "comment_lines": raw["comment_lines"],
        "raw_edge_entries": n_raw_entries,
        "self_loops_dropped": self_loops,
        "max_raw_id": max_id,
        "unused_ids_below_max": n_unused_ids,
        "variant": "largest_wcc" if largest_wcc else "whole_file",
        "published_variant": variant,
        "whole_n": int(g_full.n),
        "whole_m": int(g_full.m),
        "wcc_n": wcc["wcc_n"],
        "wcc_m": wcc["wcc_m"],
        "n_components": wcc["n_components"],
        "unique_undirected_pairs": int(g.m),
        "n": int(g.n),
        "m": int(g.m),
        "published_n": pub_n,
        "published_m": pub_m,
        "published_wcc_n": spec.get("wcc_n"),
        "published_wcc_m": spec.get("wcc_m"),
        "n_match": bool(n_ok),
        "m_match": bool(m_ok),
        "comparison": cmp_note,
        "seconds": round(elapsed, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 SNAP 边表并转成无向简单图 npz")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--only", nargs="*", default=None, help="只做这些实例名（默认全部）")
    ap.add_argument("--force", action="store_true", help="覆盖已有 raw / npz")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--chunks", type=int, default=16, help="大文件 Range 并行块数")
    ap.add_argument("--whole", action="store_true",
                    help="强制整表（默认按官方口径：无向整表 / 有向最大 WCC）")
    ap.add_argument("--wcc-only", action="store_true",
                    help="强制所有实例只保留最大弱连通分量")
    ap.add_argument("--fresh", action="store_true", help="强制重新下载并重新转换")
    ap.add_argument("--reconvert", action="store_true",
                    help="不重新下载，只用已落盘的 raw 重新生成 npz + 清单")
    args = ap.parse_args()
    if args.fresh:
        args.force = True

    names = args.only or list(INSTANCES)
    unknown = [n for n in names if n not in INSTANCES]
    if unknown:
        print(f"未知实例：{unknown}", file=sys.stderr)
        return 2

    raw_dir = args.data_root / "external" / "raw"
    npz_dir = args.data_root / "external"
    manifest_path = npz_dir / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(old, list):
                manifest = {e["name"]: e for e in old if isinstance(e, dict) and "name" in e}
            elif isinstance(old, dict):
                manifest = old
        except Exception as exc:
            print(f"警告：旧 manifest 读不了（{exc}），重新生成", file=sys.stderr)

    for name in names:
        spec = INSTANCES[name]
        url = f"{BASE}/{spec['url']}"
        try:
            src = raw_dir / Path(spec["url"]).name
            if not args.skip_download:
                download(url, src, force=args.force, chunks=args.chunks)
            if not src.exists():
                raise FileNotFoundError(f"缺少原始文件 {src}")
            # None = 按官方口径自动选（无向整表 / 有向最大 WCC）；两个开关可强制覆盖
            forced = False if args.whole else (True if args.wcc_only else None)
            manifest[name] = convert(name, src, npz_dir, spec,
                                     force=args.force or args.reconvert,
                                     largest_wcc=forced)
        except Exception as exc:
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            manifest[name] = {"name": name, "error": f"{type(exc).__name__}: {exc}"}

    npz_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({k: manifest[k] for k in sorted(manifest)}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"manifest -> {manifest_path}", flush=True)
    bad = [k for k, v in manifest.items()
           if isinstance(v, dict) and (v.get("error") or v.get("n_match") is False or v.get("m_match") is False)]
    if bad:
        print(f"注意：以下实例有 MISMATCH 或失败：{bad}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
