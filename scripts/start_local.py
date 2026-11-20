#!/usr/bin/env python3
"""Local launcher: uses the uv-managed .venv; model/input/output dirs come from
symlinks in the repo root or extra-user args — nothing is configured here.
dependency map: --enable-manager needs manager extra + user/__manager/config.ini use_uv=true
--use-ck-attention needs core comfy-kitchen; --use-sage-attention needs sage extra (available, not enabled)

runbook
routine: python scripts/start_local.py [extra main.py args]
Every argument passes straight through to main.py after the fixed block below —
this launcher has no flags of its own (unlike scripts/update_local.py).

Python port of start-local.sh. The shell `exec`ed the interpreter; here the
child runs via subprocess and this process exits with its status — the same
observable result, and the portable form (os.execv is unreliable on Windows).
"""

from __future__ import annotations

import os
import sys
from glob import glob
from pathlib import Path

from local_common import REPO_ROOT, detect_gpu, run_cmd, run_quietly, venv_python

# Fixed argument block, verbatim from start-local.sh (commented-out entries kept).
FIXED_ARGS: list[str] = [
    # --help
    "--cuda-device",
    "0",
    "--preview-method",
    "latent2rgb",
    # "--use-pytorch-cross-attention",
    "--use-ck-attention",
    "--enable-manager",
    # "--fast", "autotune",
]

# $HOME/git-ai/ai-models/Data/Images/Text2Img in the shell; Path.home() is the
# portable form.
OUTPUT_DIRECTORY = Path.home() / "git-ai" / "ai-models" / "Data" / "Images" / "Text2Img"
CUSTOM_NODES_DEPS = REPO_ROOT / "scripts" / "custom_nodes_deps.py"

# Linux hosts here have /tmp on zram; park ComfyUI scratch there instead of the
# repo temp dir. Other platforms keep the upstream defaults.
TEMP_ROOT = Path("/tmp/comfyui") if sys.platform == "linux" else None


def main(argv: list[str] | None = None) -> int:
    extra = list(sys.argv[1:] if argv is None else argv)
    if extra in (["-h"], ["--help"]):
        print(__doc__.strip())
        print("\nusage: start_local.py [-- extra main.py args]")
        print("\nFixed main.py flags (this launcher has no flags of its own):")
        print("  " + " ".join(a for a in FIXED_ARGS if not a.startswith("#")))
        print("  --output-directory " + str(OUTPUT_DIRECTORY))
        if TEMP_ROOT is not None:
            print("  --temp-directory " + str(TEMP_ROOT) + "  (TMPDIR=" + str(TEMP_ROOT) + ")")
        return 0
    os.chdir(REPO_ROOT)
    env = dict(os.environ)
    hardware, _arch = detect_gpu()
    if hardware is not None and hardware.startswith("rocm-arch") and hardware != "rocm-arch-all" and "ROCR_VISIBLE_DEVICES" not in env:
        # Prefer the dGPU: on mixed iGPU+dGPU boxes (e.g. gfx1036 + gfx1100)
        # the iGPU disables comfy-kitchen WMMA via capability intersection.
        # Explicit env wins; detection only supplies the default.
        render_nodes = sorted(glob("/dev/dri/renderD*"))
        if len(render_nodes) > 1:
            env["ROCR_VISIBLE_DEVICES"] = "0"
    # WAS Node Suite pip-installs missing deps at import, uninstalling opencv-python first —
    # which deletes files shared with opencv-python-headless. This venv has no pip anyway,
    # so let it skip rather than let a missing dep take the whole pack down.
    env["WAS_BLOCK_AUTO_INSTALL"] = "1"

    temp_args: list[str] = []
    if TEMP_ROOT is not None:
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        env.setdefault("TMPDIR", str(TEMP_ROOT))
        temp_args = ["--temp-directory", str(TEMP_ROOT)]

    # Pre-flight dependency audit: warn about missing custom node deps without blocking startup.
    print("=== Custom node dependency audit ===")
    run_quietly([str(venv_python()), str(CUSTOM_NODES_DEPS), "audit"])
    print()

    cmd = [
        str(venv_python()),
        "main.py",
        *FIXED_ARGS,
        *temp_args,
        "--output-directory",
        str(OUTPUT_DIRECTORY),
        *extra,
    ]
    return run_cmd(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
