#!/usr/bin/env python3
"""Shared helpers for the fork-local orchestration scripts.

scripts/update_local.py and scripts/start_local.py both operate on this repo
checkout and shell out to git / uv / the .venv; this module keeps the path
resolution, GPU detection, and subprocess discipline in one place. Stdlib only;
pathlib and os names only, so behavior is identical on POSIX and Windows (no sh-isms).
"""

from __future__ import annotations

import os
import re
import shutil
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


# ---------------------------------------------------------------------------
# GPU detection: pick the hardware + ROCm arch extras for this machine.
#
# Order: explicit override (COMFYUI_GPU_EXTRA / COMFYUI_ROCM_ARCH) > detected
# hardware. Detection is best-effort stdlib probing (PCI IDs via sysfs/lspci,
# vendor device nodes); unknown hardware falls back to None and the caller uses
# its default (rocm + gfx1100 on linux x86_64, matching this box).
# ---------------------------------------------------------------------------

#: gfx target -> arch extra (pyproject `rocm-arch-<target>`).
ROCM_ARCH_EXTRAS: dict[str, str] = {
    "gfx1100": "rocm-arch-gfx1100",
    "gfx1101": "rocm-arch-gfx1101",
    "gfx1151": "rocm-arch-gfx1151",
    "gfx1200": "rocm-arch-gfx1200",
    "gfx1201": "rocm-arch-gfx1201",
    "gfx942": "rocm-arch-gfx942",
    "gfx950": "rocm-arch-gfx950",
}

#: AMD PCI device IDs (high word of the sub-device) -> gfx target. Covers the
#: arch extras above; unknown AMD cards fall back to gfx1100 (this box).
_AMD_PCI_TO_GFX: dict[str, str] = {
    "744c": "gfx1100",  # RX 7900 XTX / XT (Navi 31)
    "7448": "gfx1100",  # RX 7900 GRE (Navi 31)
    "745e": "gfx1101",  # RX 7800 XT (Navi 32)
    "7460": "gfx1101",  # RX 7700 XT (Navi 32)
    "747e": "gfx1101",  # RX 7600 XT (Navi 33 — same family kernels)
    "7480": "gfx1151",  # Strix Halo (Navi 3.5 APU)
    "7590": "gfx1201",  # RX 9070 / XT (Navi 44)
    "75a0": "gfx1200",  # RX 9060 / XT (Navi 44)
    "7408": "gfx942",  # MI300
    "7410": "gfx950",  # MI325
}


def _read_sysfs_pci_ids() -> list[str]:
    """Lowercase AMD/Intel/NVIDIA PCI device-id high words from sysfs."""
    ids: list[str] = []
    by_path = Path("/sys/bus/pci/devices")
    if not by_path.is_dir():
        return ids
    try:
        entries = sorted(by_path.iterdir())
    except OSError:
        return ids
    for entry in entries:
        try:
            vendor = (entry / "vendor").read_text().strip().lower()
            device = (entry / "device").read_text().strip().lower()
        except OSError:
            continue
        if vendor in ("0x1002", "0x8086", "0x10de"):
            ids.append(f"{vendor}:{device.removeprefix('0x')}")
    return ids


def _lspci_ids() -> list[str]:
    """Fallback PCI probe when sysfs is unavailable (macOS, containers)."""
    lspci = shutil.which("lspci")
    if lspci is None:
        return []
    try:
        out = subprocess.run(
            [lspci, "-n"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except OSError:
        return []
    ids: list[str] = []
    for line in out.splitlines():
        m = re.search(r"(1002|8086|10de):([0-9a-f]{4})", line.lower())
        if m:
            ids.append(f"0x{m.group(1)}:{m.group(2)}")
    return ids


def detect_gpu() -> tuple[str | None, str | None]:
    """Return (hardware_extra, rocm_arch_extra) for this machine.

    hardware_extra is one of the rocm-arch-*, rocm-arch-all, cuda, xpu, or
    cpu extras (None = caller default). rocm_arch_extra echoes the arch
    extra on ROCm machines, else None. Mixed-gfx boxes (two or more distinct
    AMD targets) resolve to rocm-arch-all. Env overrides win: COMFYUI_GPU_EXTRA
    forces the hardware extra, COMFYUI_ROCM_ARCH forces the gfx target
    (e.g. gfx1201).
    """
    override = os.environ.get("COMFYUI_GPU_EXTRA", "").strip()
    if override:
        arch = os.environ.get("COMFYUI_ROCM_ARCH", "").strip().lower()
        arch_extra = ROCM_ARCH_EXTRAS.get(arch) if arch else None
        return override, arch_extra
    arch = os.environ.get("COMFYUI_ROCM_ARCH", "").strip().lower()
    if arch:
        return ROCM_ARCH_EXTRAS.get(arch), ROCM_ARCH_EXTRAS.get(arch)
    override = os.environ.get("COMFYUI_GPU_EXTRA", "").strip()
    if override:
        arch = os.environ.get("COMFYUI_ROCM_ARCH", "").strip().lower()
        arch_extra = ROCM_ARCH_EXTRAS.get(arch) if arch else None
        return override, arch_extra
    arch = os.environ.get("COMFYUI_ROCM_ARCH", "").strip().lower()
    if arch:
        return ROCM_ARCH_EXTRAS.get(arch), ROCM_ARCH_EXTRAS.get(arch)
    pci = _read_sysfs_pci_ids() or _lspci_ids()
    seen = {entry.split(":", 1)[0] for entry in pci}
    if "0x10de" in seen:
        return "cuda", None
    if "0x8086" in seen:
        # Intel discrete (Arc) or integrated with XPU runtime: the xpu extra
        # covers both; CPU fallback stays a manual --extra cpu choice.
        return "xpu", None
    amd_targets = [
        _AMD_PCI_TO_GFX.get(entry.split(":", 1)[1])
        for entry in pci
        if entry.startswith("0x1002:")
    ]
    amd_targets = [t for t in amd_targets if t]
    if amd_targets:
        if len(set(amd_targets)) > 1:
            # Mixed-gfx box (e.g. gfx11xx + gfx12xx): the union extra carries
            # every arch's device kernels, so both GPUs run under one venv.
            return "rocm-arch-all", "rocm-arch-all"
        target = amd_targets[0]
        return ROCM_ARCH_EXTRAS.get(target), ROCM_ARCH_EXTRAS.get(target)
    if Path("/dev/kfd").exists():
        return ROCM_ARCH_EXTRAS["gfx1100"], ROCM_ARCH_EXTRAS["gfx1100"]
    return None, None
