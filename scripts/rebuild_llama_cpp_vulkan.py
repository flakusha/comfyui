#!/usr/bin/env python3
"""Rebuild llama-cpp-python with a GPU backend for the GGUF node packs.

The lock carries the PyPI sdist (no Linux wheels), so every `uv sync`
compiles it with default flags: CPU-only, `llama_supports_gpu_offload()`
False, and WAS `LLM GGUF Loader` silently forces `gpu_layers=0`. A hand
rebuild does not survive the next sync.

The durable shape follows the repo's extra conventions: pyproject
`llama-*` extras (`llama-vulkan`, `llama-cuda`, `llama-rocm`, `llama-cpu`)
select the backend the same way `onnx-*` selects the ONNX build. They all
resolve the same locked version — the extra is a flavor selector, not a
different binary — and this script compiles the selected flavor after the
sync (`update_local.py` step 5b). Same version, so the `custom_nodes_deps`
audit (`>=0.3.26`) stays OK and no drift warning fires.

Backend matrix on this box (RX 7900 XTX, Vulkan headers, no HIP toolkit):
  vulkan  builds here (`-DGGML_VULKAN=on`). The working choice.
  cuda    needs `nvcc`; refused when absent.
  rocm    needs `hipcc` (a HIP source build, NOT the `torch/rocm` pip
          runtime — hence no resolver conflict with the `rocm` extra,
          just a toolchain gap here); refused when absent.
  cpu     reverts to the lock's plain CPU build.

runbook
routine: .venv/bin/python scripts/rebuild_llama_cpp_vulkan.py --flavor vulkan
check only: .venv/bin/python scripts/rebuild_llama_cpp_vulkan.py --check
force: .venv/bin/python scripts/rebuild_llama_cpp_vulkan.py --flavor vulkan --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

from local_common import REPO_ROOT, venv_python

PACKAGE = "llama-cpp-python"
MODULE = "llama_cpp"

#: flavor -> (CMAKE_ARGS, toolchain probe). Empty CMAKE_ARGS = plain CPU build.
FLAVORS: dict[str, tuple[str, str | None]] = {
    "vulkan": ("-DGGML_VULKAN=on", "glslc"),
    "cuda": ("-DGGML_CUDA=on", "nvcc"),
    "rocm": ("-DGGML_HIP=on", "hipcc"),
    "cpu": ("", None),
}
DEFAULT_FLAVOR = "vulkan"


def locked_version() -> str | None:
    try:
        return pkg_version(PACKAGE)
    except PackageNotFoundError:
        return None


def gpu_ready() -> bool:
    try:
        import llama_cpp
    except ImportError:
        return False
    try:
        if not bool(llama_cpp.llama_supports_gpu_offload()):
            return False
    except Exception:
        return False
    lib = Path(llama_cpp.__file__).parent / "lib"
    try:
        names = {p.name for p in lib.glob("*.so")}
    except OSError:
        return False
    return any("vulkan" in n or "cuda" in n or "hip" in n for n in names)


def check_toolchain(flavor: str) -> str | None:
    """Error text when `flavor` cannot build here, else None."""
    _, tool = FLAVORS[flavor]
    if tool is None:
        return None
    if shutil.which(tool) is None:
        if flavor == "rocm":
            return (
                "llama-rocm needs hipcc (a HIP source build), which this box "
                "lacks — the torch `rocm` extra ships only the pip runtime, "
                "not the compiler. Use --flavor vulkan on this machine."
            )
        return (
            f"llama-{flavor} needs `{tool}` on PATH, which is missing. "
            "Use --flavor vulkan on this machine."
        )
    return None


def rebuild(version: str | None, flavor: str) -> int:
    cmake_args, _ = FLAVORS[flavor]
    target = f"{PACKAGE}=={version}" if version else PACKAGE
    env = dict(os.environ)
    if cmake_args:
        env["CMAKE_ARGS"] = cmake_args
    else:
        env.pop("CMAKE_ARGS", None)
    cmd = [
        "uv", "pip", "install", "--python", str(venv_python()),
        "--force-reinstall", "--no-cache-dir", target,
    ]
    print(f"=== rebuilding {target} [{flavor}] ({cmake_args or 'plain CPU'}) ===")
    try:
        rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode
    except OSError as exc:
        print(f"{cmd[0]}: {exc}", file=sys.stderr)
        return 127
    if rc != 0:
        print(f"rebuild failed (exit {rc})", file=sys.stderr)
        return rc
    # The sdist compile takes minutes; a killed/timed-out caller must not
    # mistake "installer started" for "offload works". Verify the live
    # install before reporting success.
    if flavor != "cpu" and not gpu_ready():
        print(
            f"{PACKAGE} rebuilt but GPU offload still reports False — the "
            "install phase may not have completed. Rerun with --force.",
            file=sys.stderr,
        )
        return 1
    print(f"{PACKAGE} [{flavor}]: ready.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavor", choices=sorted(FLAVORS), default=DEFAULT_FLAVOR,
                        help="backend to build (default: %(default)s)")
    parser.add_argument("--check", action="store_true",
                        help="exit 0 when GPU offload works, 1 otherwise (no install)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even when GPU offload already works")
    opts = parser.parse_args(argv)
    os.chdir(REPO_ROOT)
    ready = gpu_ready()
    if opts.check:
        print(f"{PACKAGE}: {'GPU offload ready' if ready else 'CPU only'}")
        return 0 if ready else 1
    if opts.flavor == "cpu":
        return rebuild(locked_version(), "cpu")
    problem = check_toolchain(opts.flavor)
    if problem is not None:
        print(problem, file=sys.stderr)
        return 2
    if ready and not opts.force:
        print(f"{PACKAGE}: GPU offload already ready; nothing to do.")
        return 0
    return rebuild(locked_version(), opts.flavor)


if __name__ == "__main__":
    sys.exit(main())
