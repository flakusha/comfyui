#!/usr/bin/env python3
"""Shared helpers for the fork-local orchestration scripts.

scripts/update_local.py and scripts/start_local.py both operate on this repo
checkout and shell out to git / uv / the .venv; this module keeps the path
resolution and subprocess discipline in one place. Stdlib only; pathlib and
os names only, so behavior is identical on POSIX and Windows (no sh-isms).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = REPO_ROOT / ".venv"


def _exe(name: str) -> str:
    """Executable file name inside a venv, suffixed for Windows layouts."""
    return f"{name}.exe" if os.name == "nt" else name


def venv_bin_dir() -> Path:
    """Directory holding a venv's executables on this platform."""
    return VENV_DIR / ("Scripts" if os.name == "nt" else "bin")


def venv_python() -> Path:
    """Interpreter of the repo-managed .venv."""
    return venv_bin_dir() / _exe("python")


def venv_tool(name: str) -> Path:
    """Console script (ruff, ...) inside the repo-managed .venv."""
    return venv_bin_dir() / _exe(name)


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None) -> int:
    """Run a command with stdio inherited; return its exit status.

    A missing executable yields 127 — the `command not found` status the
    shell `|| exit` / `|| true` sites were written against.
    """
    try:
        return subprocess.run(cmd, env=env).returncode
    except OSError as exc:
        print(f"{cmd[0]}: {exc}", file=sys.stderr)
        return 127


def run_quietly(cmd: list[str]) -> None:
    """Run a command and ignore the outcome — the shell `|| true` sites."""
    run_cmd(cmd)


def fail(message: str) -> NoReturn:
    """Print a fatal message to stderr and exit 1 (shell `echo ... >&2; exit 1`)."""
    print(message, file=sys.stderr)
    sys.exit(1)
