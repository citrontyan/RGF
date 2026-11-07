"""下载 SuiteSparse 的 GAP-road，转成本仓库的无向简单图 npz（MIS 口径）。

用法（WSL，CPU；scipy.io.mmread 在 difusco 环境里有）：

    ~/miniconda3/envs/difusco/bin/python scripts/fetch_suitesparse.py

落盘位置：

    /mnt/e/data/rgf/external/raw/GAP-road.tar.gz    官方压缩包
    /mnt/e/data/rgf/external/raw/GAP-road/GAP-road.mtx  解出来的矩阵
    /mnt/e/data/rgf/external/GAP-road.npz           rgf 图缓存
    /mnt/e/data/rgf/external/manifest.json          与 fetch_snap.py 共用同一份清单

官方数字（用户给的；包内 ``GAP-road.mtx`` 的 MM 头与之逐项核对过。
注意 sparse.tamu.edu/MM/GAP/GAP-road.html 这个 html 路径实测 404，改用
https://sparse.tamu.edu/GAP/GAP-road 与包内头）：

    23,947,347 × 23,947,347，非零元 57,708,624，structurally symmetric，
    无向简单边数 = 28,854,312。

**实测到的口径差异（重要）**：包内 ``GAP-road.mtx`` 的 MM 头是

    %%MatrixMarket matrix coordinate integer symmetric
    23947347 23947347 28854312

即文件里只存了**下三角**，存储条目数 28,854,312 —— 正好是 57,708,624 的一半。
所以 57,708,624 是"展开成完整对称存储后的非零元数"，28,854,312 既是存储条目数、
也是本仓库要的无向简单边数。脚本因此**以 MM 头为准**做断言（``m == 头里的条目数``），
并把用户给的 57,708,624 作为 ``published_nnz`` 一并记进清单。

矩阵只存下三角时，取严格上三角（或严格下三角）就等于"对称化去重"，比
``m.maximum(m.T)`` 省一半内存和一次全量稀疏加法。
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgf.datasets.io import save_graph_npz  # noqa: E402
from rgf.graph import Graph  # noqa: E402

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
DEFAULT_DATA_ROOT = Path("/mnt/e/data/rgf") if Path("/mnt/e").exists() else Path(r"E:\data\rgf")

INSTANCES: dict[str, dict] = {
    # name -> SuiteSparse 路径（组/矩阵名）、官方尺寸、官方非零元数
    "GAP-road": {
        "group": "GAP",
        "n": 23_947_347,
        "nnz": 57_708_624,       # 官网/用户给的“完整对称存储”非零元数
        "nnz_stored": 28_854_312,  # 实测 MM 头里的存储条目数（只存下三角）
        "expected_m": 28_854_312,  # 无向简单边数（官方给的数字）
    },
}

# 只解数据文件；GAP-road 的包里还有 GAP-road_sources.mtx（图的一个小辅助矩阵），
# 名字里含下划线，必须排除掉，否则会解错矩阵。
MTX_STEM = "GAP-road"


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
    with urllib.request.urlopen(req, timeout=1800) as r:
        return r.read()


def download(url: str, dest: Path, force: bool = False, chunks: int = 8) -> Path:
    """下载（Range 并行，先写 ``.part``）；已存在则跳过。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"  已存在 {dest}（{dest.stat().st_size/1e6:.1f} MB），跳过下载", flush=True)
        return dest
    total = _head_size(url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.perf_counter()
    if total and total > (32 << 20) and chunks > 1:
        step = (total + chunks - 1) // chunks
        bounds = [(i * step, min(total - 1, (i + 1) * step - 1)) for i in range(chunks) if i * step <= total - 1]
        print(f"  GET {url}（{total/1e6:.1f} MB，{len(bounds)} 个 Range 并行）", flush=True)
        with ThreadPoolExecutor(max_workers=len(bounds)) as ex:
            parts = list(ex.map(lambda b: _fetch_range(url, b[0], b[1]), bounds))
        with open(tmp, "wb") as fh:
            for p in parts:
                fh.write(p)
    else:
        print(f"  GET {url}（单连接）", flush=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=1800) as r, open(tmp, "wb") as fh:
            got = 0
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                fh.write(b)
                got += len(b)
                if got % (128 << 20) < (1 << 22):
                    print(f"    {got/1e6:.1f} MB", flush=True)
    got = tmp.stat().st_size
    if total and got != total:
        raise IOError(f"下载不完整：{got} != Content-Length {total}")
    tmp.replace(dest)
    print(f"  -> {dest}（{got/1e6:.1f} MB，{time.perf_counter()-t0:.0f}s）", flush=True)
    return dest


def extract_mtx(tar_path: Path, out_dir: Path, force: bool = False) -> Path:
    """从 SuiteSparse 的 ``<name>.tar.gz`` 里解出主 ``.mtx``（可能是 .mtx 或 .mtx.gz）。

    包里常有同名辅助文件（``GAP-road_sources.mtx``），只取 stem 完全等于矩阵名的那个。
    """
    name = tar_path.name[: -len(".tar.gz")]
    want = out_dir / name
    if want.exists() and any(want.iterdir()) and not force:
        print(f"  已存在解压目录 {want}，跳过解压", flush=True)
    else:
        want.mkdir(parents=True, exist_ok=True)
        print(f"  解压 {tar_path.name} -> {want}", flush=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                base = Path(m.name).name
                stem = base
                for suf in (".mtx.gz", ".mtx"):
                    if stem.endswith(suf):
                        stem = stem[: -len(suf)]
                        break
                if stem != name:
                    continue  # 只解主矩阵，跳过 *_sources.mtx / readme 之类
                with tf.extractfile(m) as src, open(want / base, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 22)
    cands = sorted(list(want.glob("*.mtx")) + list(want.glob("*.mtx.gz")))
    if not cands:
        raise FileNotFoundError(f"{want} 里没有 .mtx：{[p.name for p in want.iterdir()]}")
    return cands[0]


def read_mtx_header(path: Path) -> dict:
    """读 Matrix Market 头（banner + 第一条 size 行；中间可能夹 ``%`` 注释）。"""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        banner = fh.readline().strip()
        size = ""
        for line in fh:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            size = s
            break
    if not size:
        raise ValueError(f"{path} 里找不到 size 行")
    parts = banner.lstrip("%").split()
    dims = size.split()
    return {
        "banner": banner,
        "object": parts[2] if len(parts) > 2 else "?",
        "format": parts[3] if len(parts) > 3 else "?",
        "field": parts[4] if len(parts) > 4 else "?",
        "symmetry": parts[5] if len(parts) > 5 else "?",
        "rows": int(dims[0]),
        "cols": int(dims[1]),
        "entries": int(dims[2]),
    }


def convert(name: str, mtx: Path, out_dir: Path, spec: dict, force: bool = False) -> dict:
    npz = out_dir / f"{name}.npz"
    pub_n, pub_nnz, pub_m = spec["n"], spec["nnz"], spec["expected_m"]
    if npz.exists() and not force:
        with np.load(npz) as z:
            g = Graph.from_edges(int(z["n"]), z["edges"].astype(np.int64))
        print(f"[{name}] 已转换，跳过：{npz}（现 |V|={g.n:,} |E|={g.m:,}）", flush=True)
        return {
            "name": name,
            "npz": str(npz),
            "npz_bytes": int(npz.stat().st_size),
            "n": int(g.n),
            "m": int(g.m),
            "published_n": pub_n,
            "published_nnz": pub_nnz,
            "published_m": pub_m,
            "n_match": bool(g.n == pub_n),
            "m_match": bool(g.m == pub_m),
            "skipped": True,
        }

    t0 = time.perf_counter()
    head = read_mtx_header(mtx)
    print(f"[{name}] MM 头：{head['banner']} | size 行 {head['rows']}x{head['cols']} nnz={head['entries']:,}", flush=True)
    # 断言基准：symmetric 声明时 MM 头的存储条目数就是“无向简单边数”（只存一个三角）
    expected_m = int(head["entries"])
    mm = sio.mmread(str(mtx))
    coo = sp.coo_matrix(mm)
    del mm
    t_read = time.perf_counter() - t0
    print(f"[{name}] mmread 完成：shape={coo.shape} nnz={coo.nnz:,} 用时 {t_read:.0f}s", flush=True)

    entries = int(coo.nnz)
    diag = int((coo.row == coo.col).sum())
    upper = int((coo.row < coo.col).sum())
    lower = int((coo.row > coo.col).sum())
    print(f"[{name}] 对角元={diag:,} 严格上三角={upper:,} 严格下三角={lower:,}", flush=True)

    # 严格上三角 = 无序对去重。symmetric 存储时上下三角只存了其中一个，
    # 所以这里既要“对称化”也要“去重”，取严格上三角一次完成；若原文件两向都存，
    # 取上三角同样得到唯一无序对。
    up = sp.triu(coo, k=1).tocoo()
    edges = np.stack([up.row.astype(np.int64), up.col.astype(np.int64)], axis=1)
    del up, coo

    n = int(pub_n)
    if head["rows"] == head["cols"]:
        n = int(head["rows"])
    g = Graph.from_edges(n, edges)
    save_graph_npz(g, npz)
    elapsed = time.perf_counter() - t0

    n_ok = g.n == pub_n
    m_ok = g.m == expected_m
    m_ok_pub = g.m == pub_m
    print(
        f"[{name}] 去重后 |V|={g.n:,}（官方 {pub_n:,}）{'MATCH' if n_ok else 'MISMATCH'} | "
        f"|E|={g.m:,}（MM 头条目 {expected_m:,} → {'MATCH' if m_ok else 'MISMATCH'}；"
        f"用户给的 57,708,624 是完整对称口径，本字段按头断言）用时 {elapsed:.0f}s",
        flush=True,
    )
    if not n_ok:
        print(f"[{name}] !!! MISMATCH n: 实测 {g.n:,} != 官方 {pub_n:,} !!!", flush=True)
    if not m_ok:
        print(f"[{name}] !!! MISMATCH m: 实测 {g.m:,} != MM 头条目 {expected_m:,} !!!", flush=True)
    if not m_ok_pub:
        print(f"[{name}] 说明：实测 |E|={g.m:,} 与用户给的完整对称口径 57,708,624 不同，"
              f"但等于其一半（与官方无向边数 {pub_m:,} 一致）", flush=True)

    return {
        "name": name,
        "source_url": f"https://sparse.tamu.edu/MM/{spec['group']}/{name}.tar.gz",
        "raw_file": str(mtx.relative_to(out_dir)),
        "raw_bytes": int(mtx.stat().st_size),
        "npz": str(npz),
        "npz_bytes": int(npz.stat().st_size),
        "mm_header": head,
        "mm_entries": entries,
        "diagonal_entries": diag,
        "strict_upper": upper,
        "strict_lower": lower,
        "unique_undirected_pairs": int(g.m),
        "n": int(g.n),
        "m": int(g.m),
        "published_n": pub_n,
        "published_nnz": pub_nnz,
        "published_nnz_stored": spec.get("nnz_stored"),
        "published_m": pub_m,
        "published_variant": "whole (structure-symmetric, one triangle stored)",
        "n_match": bool(n_ok),
        "m_match": bool(m_ok),
        "m_match_published_undirected": bool(m_ok_pub),
        "comparison": "symmetric 存储：MM 头条目数 = 无向简单边数；用户给的 57,708,624 是完整对称口径（=2x）",
        "seconds": round(elapsed, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 SuiteSparse GAP-road 并转成无向简单图 npz")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--chunks", type=int, default=8)
    args = ap.parse_args()

    names = args.only or list(INSTANCES)
    unknown = [n for n in names if n not in INSTANCES]
    if unknown:
        print(f"未知矩阵：{unknown}", file=sys.stderr)
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
        url = f"https://sparse.tamu.edu/MM/{spec['group']}/{name}.tar.gz"
        try:
            tar_path = raw_dir / f"{name}.tar.gz"
            if not args.skip_download:
                download(url, tar_path, force=args.force, chunks=args.chunks)
            if not tar_path.exists():
                raise FileNotFoundError(f"缺少 {tar_path}")
            mtx = extract_mtx(tar_path, raw_dir, force=args.force)
            manifest[name] = convert(name, mtx, npz_dir, spec, force=args.force)
        except Exception as exc:
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            manifest[name] = {"name": name, "error": f"{type(exc).__name__}: {exc}"}

    npz_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({k: manifest[k] for k in sorted(manifest)}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"manifest -> {manifest_path}", flush=True)
    bad = [k for k, v in manifest.items() if isinstance(v, dict) and (v.get("error") or v.get("m_match") is False or v.get("n_match") is False)]
    if bad:
        print(f"注意：以下矩阵有 MISMATCH 或失败：{bad}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
