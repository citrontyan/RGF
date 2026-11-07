"""Helpers for driving Linux-only third-party binaries through WSL.

METIS and KaMIS are built in WSL on this workstation (see
``docs/decisions.md`` item D-7), so a non-``.exe`` executable is invoked as
``wsl -e <binary>`` with every path translated to its ``/mnt/<drive>/...`` view.
"""

from __future__ import annotations

import os
from pathlib import Path


def on_windows() -> bool:
    """True when this process runs on Windows (so Linux binaries need ``wsl -e``)."""
    return os.name == "nt"


def to_wsl_path(path: str | Path) -> str:
    """``D:\\a\\b`` -> ``/mnt/d/a/b``."""
    p = Path(path).resolve()
    drive = p.drive.rstrip(":").lower()
    rest = str(p)[len(p.drive) :].replace("\\", "/")
    return f"/mnt/{drive}{rest}"


def is_native_exe(executable: str | Path) -> bool:
    return str(executable).lower().endswith(".exe")


def wrap_command(executable: str | Path, args: list[str], path_args: set[int] | None = None) -> list[str]:
    """Build the argv for ``executable``.

    ``path_args`` lists the indices of ``args`` that hold filesystem paths; they
    are translated when the binary has to run inside WSL.

    When this process already runs *inside* Linux (the faithful WSL environment,
    see D-11/D-12), the binary is invoked directly and no path translation is
    needed — nesting ``wsl -e`` is not possible there.
    """
    exe = str(executable)
    if is_native_exe(exe):
        return [exe] + args
    if not on_windows():
        return [exe] + args
    path_args = path_args or set()
    translated = [to_wsl_path(a) if i in path_args else a for i, a in enumerate(args)]
    return ["wsl", "-e", to_wsl_path(exe)] + translated
