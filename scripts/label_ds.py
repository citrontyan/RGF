"""批量生成 DS 参照解（训练监督 / 评测参照），口径同 D-6 与 D-14 决策 1A。

每张图写一个 ``<stem>.dsref.npz``（``n`` / ``labels`` / ``objective`` / ``method`` /
``optimal`` / ``time_s``），断点续跑：已存在的跳过。

    .venv\\Scripts\\python.exe scripts\\label_ds.py --dataset E:\\data\\rgf\\rb\\rb200_300\\train ^
        --limit 20000 --time-limit 30 --workers 24

并发纪律（《共享机器批处理纪律》）：每个 worker 单线程（HiGHS 会被 OMP_NUM_THREADS
限制），总线程数 = --workers。
"""

from __future__ import annotations

import os

# 必须在 import numpy/scipy 之前设置，worker 进程（spawn）重新导入时同样生效
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import concurrent.futures as cf  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgf.baselines.ds_reference import ds_reference  # noqa: E402
from rgf.datasets.io import load_graph_npz  # noqa: E402


def process(job: tuple[str, str, float]) -> tuple[str, str, int, str, bool, float]:
    clip_s, out_s, time_limit = job
    clip = Path(clip_s)
    out = Path(out_s)
    t0 = time.perf_counter()
    try:
        g = load_graph_npz(clip)
        ref = ds_reference(g, time_limit_s=time_limit)
        np.savez_compressed(
            out,
            n=np.int64(g.n),
            labels=ref.labels.astype(np.int8),
            objective=np.int64(ref.objective),
            method=ref.method,
            optimal=np.bool_(ref.optimal),
            time_s=np.float32(ref.time_s),
        )
        return (clip.stem, "ok", int(ref.objective), ref.method, bool(ref.optimal), float(ref.time_s))
    except MemoryError:
        # 并发下 HiGHS 偶发 bad_alloc：退化为贪心 + 冗余删除（D-6 兜底口径）
        try:
            from rgf.baselines.ds_reference import greedy_ds

            g = load_graph_npz(clip)
            labels = greedy_ds(g)
            np.savez_compressed(
                out,
                n=np.int64(g.n),
                labels=labels.astype(np.int8),
                objective=np.int64(int(labels.sum())),
                method="greedy+redundancy",
                optimal=np.bool_(False),
                time_s=np.float32(time.perf_counter() - t0),
            )
            return (clip.stem, "ok", int(labels.sum()), "greedy+redundancy", False, time.perf_counter() - t0)
        except Exception as exc2:  # noqa: BLE001
            return (clip.stem, f"fail:{type(exc2).__name__}:{exc2}", -1, "-", False, 0.0)
    except Exception as exc:  # noqa: BLE001
        return (clip.stem, f"fail:{type(exc).__name__}:{exc}", -1, "-", False, 0.0)


def free_gb() -> float:
    """可用物理内存（GB）。Windows 下用 ctypes，其它平台返回 +inf。"""
    if os.name != "nt":
        return float("inf")
    try:
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = _MemStatus()
        st.dwLength = ctypes.sizeof(_MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
        return st.ullAvailPhys / 1024 ** 3
    except Exception:  # noqa: BLE001
        return float("inf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0=全部）")
    ap.add_argument("--time-limit", type=float, default=30.0, help="每图 MILP 墙钟上限（秒）")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--min-free-gb", type=float, default=3.0, help="可用物理内存低于该值就等待")
    ap.add_argument("--max-attempts", type=int, default=50, help="进程池崩溃后的重试次数上限")
    ap.add_argument("--force", action="store_true",
                    help="覆盖已存在的 .dsref.npz（口径修正后重生成时用，见 D-18）")
    args = ap.parse_args()

    clips = [f for f in sorted(args.dataset.glob("*.npz"))
             if not f.name.endswith((".label.npz", ".dsref.npz", ".opt.npz"))]
    if args.limit:
        clips = clips[: args.limit]

    def dsref_path(clip: Path) -> Path:
        return clip.parent / (clip.stem + ".dsref.npz")

    def pending_now() -> list[Path]:
        """队列口径：--force 时全部重算；否则只补缺失。"""
        if args.force:
            return list(clips)
        return [c for c in clips if not dsref_path(c).exists()]

    def unwritten() -> list[Path]:
        """收口口径：**只看文件是否落盘**（2026-11-01 修 bug#5：原实现用 pending_now()
        统计 left，--force 时恒等于总数 → 全部成功也 exit 1）。"""
        return [c for c in clips if not dsref_path(c).exists()]

    pending = pending_now()
    cached_before = len(clips) - len(pending)
    print(f"requested={len(clips)} pending={len(pending)} skipped_cached={cached_before} "
          f"workers={args.workers} 每图上限={args.time_limit}s"
          + ("（--force：全部重算）" if args.force else ""), flush=True)
    if not pending:
        print(f"完成 requested={len(clips)} computed=0 failed=0 skipped_cached={cached_before} left=0",
              flush=True)
        return 0

    t0 = time.perf_counter()
    done = failed = 0
    n_optimal = n_incumbent = 0
    methods: dict[str, int] = {}
    objectives: list[int] = []
    workers = args.workers
    attempt = 0

    while attempt < args.max_attempts:
        # 内存门控：外部任务（如别的会话的批处理）可能把物理内存吃到接近 0，
        # 此时 worker 会被系统直接杀掉（BrokenProcessPool），白跑一轮。
        while free_gb() < args.min_free_gb:
            print(f"  等待内存：可用 {free_gb():.2f} GB < {args.min_free_gb} GB，60 s 后重试", flush=True)
            time.sleep(60)

        pending = pending_now()
        if not pending:
            break
        jobs = [(str(c), str(c.parent / (c.stem + ".dsref.npz")), args.time_limit) for c in pending]
        attempt += 1
        print(f"  第 {attempt} 轮：{len(jobs)} 图，workers={workers}，"
              f"可用内存 {free_gb():.1f} GB", flush=True)
        processed = 0
        try:
            with cf.ProcessPoolExecutor(max_workers=workers) as pool:
                for stem, status, obj, method, optimal, el in pool.map(process, jobs, chunksize=8):
                    processed += 1
                    if status == "ok":
                        done += 1
                        if optimal:
                            n_optimal += 1
                        else:
                            n_incumbent += 1
                        methods[method] = methods.get(method, 0) + 1
                        objectives.append(obj)
                    else:
                        failed += 1
                        if failed <= 5:
                            print(f"  {status}", flush=True)
                    if processed % 500 == 0:
                        el_s = time.perf_counter() - t0
                        left = len(jobs) - processed
                        eta = el_s / max(done, 1) * left / 60
                        print(f"  {processed}/{len(jobs)}  已用 {el_s/60:.1f} min, 预计还需 {eta:.1f} min, "
                              f"ok={done} fail={failed}", flush=True)
            break  # 正常跑完
        except (cf.process.BrokenProcessPool, BrokenPipeError, OSError) as exc:
            # worker 被 OOM killer 杀掉的典型表现；进度已落盘，下一轮从磁盘重建待办
            print(f"  进程池崩溃（{type(exc).__name__}），已写出 {done} 张，重建进程池继续", flush=True)
            workers = max(4, workers // 2)
            time.sleep(30)

    left = len(unwritten())
    print(f"完成 {done}，失败 {failed}，剩余未写 {left}，用时 {(time.perf_counter()-t0)/60:.1f} min")
    print(f"方法分布: {methods}")
    # 供 run_unit summary.regexes 提取：`optimal=(\d+)` / `incumbent=(\d+)`
    print(f"optimal={n_optimal} incumbent={n_incumbent} total={done} left={left}")
    # 口径分开打印：requested=全部候选；computed=本轮真正算成功；failed=本轮失败；
    # skipped_cached=本轮之前已有文件而跳过；left=跑完仍缺文件数。
    print(f"完成 requested={len(clips)} computed={done} failed={failed} "
          f"skipped_cached={cached_before} left={left}", flush=True)
    if objectives:
        print(f"DS 规模: 均值 {np.mean(objectives):.2f}  中位 {np.median(objectives):.0f}  "
              f"min {min(objectives)}  max {max(objectives)}")
    # 退出码只反映真实失败：所有图都有 dsref 落盘且本轮无失败 → 0（--force 不影响该判定）
    return 0 if (left == 0 and failed == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
