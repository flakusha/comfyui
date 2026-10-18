#!/usr/bin/env python3
"""Local updater: fetch upstream master (FF-only) -> guarded mirror to origin/master -> rebase
the patch stack onto master -> uv sync (rocm + manager + dev) -> custom-node deps -> gates.

runbook
routine: python scripts/update_local.py
--mirror      auto-approve the origin/master mirror push (default asks [y/N]; it mutates a
              public remote). --no-mirror skips it (push manually: git push origin master:master).
--force       accept a rebase that truly lost commits (inspect the printed list first)
--skip-gates  skip post-bump ruff/import checks
--tests       also run the full upstream tests-unit suite (opt-in; the suite is
              an upstream concern, not this fork's gate)
pass optional extras through: --extra sage --extra onnx-rocm (rocm and cuda extras are mutually
exclusive — passing both exits code 2)
first run / after changing extras: uv lock --upgrade first, then this script
pyproject.toml conflict -> merge hunks BY HAND, never a wholesale side; runbook:
  .tmp/proposals/update-rebase-tooling.md §3b (merge [tool.uv] keys; then uv lock --check).
rollback: mid-rebase -> git rebase --abort ; after -> git reset --hard refs/pre-rebase-backup
  (on local; the ref is kept until gates pass, then deleted)
  env: git checkout uv.lock && uv sync --inexact --extra rocm --extra manager --extra dev --python .venv/bin/python
--inexact required: bare uv sync removes custom-node packages
out-of-band venv installs (uv pip install ...) are silently reverted by the sync; the
  updater prints a drift note before syncing (Python port only; no shell counterpart)
companion pin policy: the [project] dependencies floors in pyproject.toml are the
  spec; requirements.txt exact pins drive them — this script rewrites each managed
  floor to >=<requirements-exact>, so a fresh-clone bare `uv sync` reproduces the
  exact companion versions. Hardware extras (torch rocm/cuda pins) and custom_nodes
  are out of requirements' scope and live in pyproject extras /
  scripts/custom_nodes_deps.py respectively.
FF safety at both levels: `fetch upstream master:master` without + refuses a non-FF, and
`push origin master:master` is refused by the remote — a diverged mirror surfaces as an
error, never a clobber.

Python port of update-local.sh: same steps, same exit codes, same messages.
Windows-safe: every subprocess call is an argv list, the environment is a dict
copy, and all paths go through pathlib. Pure, git-free pieces (flag parse,
requirements pins, patch-id set diffing) are separate functions for testing.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal

from local_common import (
    REPO_ROOT,
    VENV_DIR,
    fail,
    run_cmd,
    run_quietly,
    venv_python,
    venv_tool,
)

BACKUP_REF = "refs/pre-rebase-backup"
BASE_EXTRAS: tuple[str, ...] = ("rocm", "manager", "dev")
# torch resolves through conflicting uv forks (pyproject [tool.uv].conflicts);
# both extras together can never sync.
HARDWARE_EXTRAS = frozenset({"rocm", "cuda"})
REQUIREMENTS_TXT = REPO_ROOT / "requirements.txt"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"
CUSTOM_NODES_DEPS = REPO_ROOT / "scripts" / "custom_nodes_deps.py"
_IMPORT_GATE = "import server, execution, nodes"
# grep -E '^[A-Za-z0-9._-]+==' : name is every leading package-name char up to the first ==
_PIN_LINE = re.compile(r"^([A-Za-z0-9._-]+)==(.*)$")

MirrorMode = Literal["ask", "yes", "no"]


@dataclass(frozen=True)
class Options:
    """Parsed CLI state, mirroring the shell script's variables."""

    mirror: MirrorMode = "ask"  # ask | yes | no
    force: bool = False
    gates: bool = True
    tests: bool = False
    extras: tuple[str, ...] = ()


def parse_args(argv: list[str] | None = None) -> Options:
    """Strict flag parse; only --extra pairs pass through to uv sync.

    argparse exits 2 on unknown args and on `--extra` missing its value,
    matching the shell's `unknown arg` / `--extra needs a value` paths.
    """
    parser = argparse.ArgumentParser(
        description="Update the fork: fetch upstream master, mirror origin/master, "
        "rebase the patch stack, sync, gate.",
    )
    parser.add_argument(
        "--mirror", action="store_true", help="auto-approve the origin/master mirror push"
    )
    parser.add_argument(
        "--no-mirror",
        dest="no_mirror",
        action="store_true",
        help="skip the origin/master mirror push",
    )
    parser.add_argument(
        "--force", action="store_true", help="accept a rebase that truly lost commits"
    )
    parser.add_argument(
        "--skip-gates",
        dest="skip_gates",
        action="store_true",
        help="skip post-bump ruff/import checks",
    )
    parser.add_argument(
        "--tests", action="store_true", help="also run the upstream tests-unit suite"
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="NAME",
        help="extra passthrough to uv sync (repeatable)",
    )
    args = parser.parse_args(argv)

    extras = tuple(args.extra)
    if HARDWARE_EXTRAS.issubset(extras):
        parser.error("--extra rocm and --extra cuda are mutually exclusive")
    return Options(
        mirror="yes" if args.mirror else "no" if args.no_mirror else "ask",
        force=args.force,
        gates=not args.skip_gates,
        tests=args.tests,
        extras=extras,
    )


# ---------------------------------------------------------------------------
# Pure, git-free helpers (unit-testable without running git)
# ---------------------------------------------------------------------------


def extract_pins(requirements: str) -> list[tuple[str, str]]:
    """Exact `name==version` pins from requirements.txt content.

    Mirrors `grep -E '^[A-Za-z0-9._-]+=='` plus ${pin%%==*}/${pin#*==}:
    the first == splits name from version. Unpinned lines (torch, >= floors)
    are skipped. The regex admits [A-Za-z0-9._-] names only, which keeps the
    later -c snippet interpolation injection-free.
    """
    pins: list[tuple[str, str]] = []
    for line in requirements.splitlines():
        match = _PIN_LINE.match(line)
        if match:
            pins.append((match.group(1), match.group(2)))
    return pins


# Managed floors live only in [project] dependencies: from the `dependencies = [`
# line to the closing `]` at column 0. Extras sections ([project.optional-dependencies],
# [build-system], ...) carry other pins (torch==...+rocm) that must never be touched.
_DEPS_OPEN = re.compile(r"^dependencies = \[\s*$")
_FLOOR_LINE = re.compile(r'^(\s*"[A-Za-z0-9._-]+>=)([^"]*)(".*)$')


def parse_pyproject_floors(pyproject: str) -> dict[str, str]:
    """`name -> floor version` for >=-floored entries in [project] dependencies ONLY.

    The array is delimited textually (open line to a column-0 `]`), so pins in
    extras sections are never seen. A line without `>=` (bare names like torch)
    contributes nothing; multiple specifiers keep everything after the first `>=`.
    """
    floors: dict[str, str] = {}
    inside = False
    for line in pyproject.splitlines():
        if not inside:
            if _DEPS_OPEN.match(line):
                inside = True
            continue
        if line.startswith("]"):
            break
        match = _FLOOR_LINE.match(line)
        if match:
            entry = match.group(1).lstrip()
            name = entry[1:].split(">=", 1)[0]
            floors[name] = match.group(2)
    return floors


def set_floor(pyproject: str, name: str, version: str) -> str:
    """Rewrite `name`'s floor to `>=version` in [project] dependencies, or insert it.

    Rewrites preserve the line's indentation, quotes, and trailing comma
    (everything outside the version text). A name not yet in the array is
    inserted directly after the last floored line (or the opening line when the
    array has no floors yet) with the array's prevailing indentation —
    deterministic placement, no reflow. Extras-section lines are out of scope.
    """
    lines = pyproject.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if _DEPS_OPEN.match(line))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("]"))
    prefix = f'"{name}>='
    for i in range(start + 1, end):
        match = _FLOOR_LINE.match(lines[i])
        if match and match.group(1).lstrip().startswith(prefix):
            lines[i] = _FLOOR_LINE.sub(rf"\g<1>{version}\g<3>", lines[i])
            return "".join(lines)
    insert_at = start + 1
    indent = "  "
    for i in range(end - 1, start, -1):
        if _FLOOR_LINE.match(lines[i]):
            insert_at = i + 1
            indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
            break
    lines.insert(insert_at, f'{indent}"{name}>={version}",\n')
    return "".join(lines)


def parse_patch_ids(output: str | bytes) -> list[tuple[str, str]]:
    """Parse `git patch-id --stable` output: one "<patch-id> <sha>" pair per line."""
    if isinstance(output, bytes):
        output = output.decode("ascii", errors="replace")
    pairs: list[tuple[str, str]] = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    return pairs


def missing_patch_ids(
    pre_pairs: list[tuple[str, str]], post_pairs: list[tuple[str, str]]
) -> list[str]:
    """Patch-ids present before the rebase but gone after (comm -23 of the id columns)."""
    pre = {pid for pid, _ in pre_pairs}
    post = {pid for pid, _ in post_pairs}
    return sorted(pre - post)


def lost_patch_ids(missing: list[str], master_pairs: list[tuple[str, str]]) -> list[str]:
    """Missing patch-ids no commit brought in by the fetch can account for.

    `master_pairs` are the old_master..master patch-ids: a pid there means the
    change was absorbed upstream (pid drift on replay makes this best-effort).
    """
    master = {pid for pid, _ in master_pairs}
    return sorted(pid for pid in missing if pid not in master)


_DELTA_LINE = re.compile(r"^ ([+-]) ([A-Za-z0-9._-]+)==(.+?)\s*$")


def _release_tuple(version: str) -> tuple[int, ...] | None:
    """Numeric release tuple of a version string ('1.2.3rc1'/'1.2.3+rocm' -> (1, 2, 3)).

    None when any dot-separated chunk of the pre-build part is not an int.
    """
    chunks = version.split("+", 1)[0].split(".")
    if not all(chunk.isdigit() for chunk in chunks):
        return None
    return tuple(int(chunk) for chunk in chunks)


def parse_sync_deltas(output: str) -> tuple[dict[str, str], dict[str, str]]:
    """`uv sync --dry-run` output -> (uninstalled, installed) name->version maps.

    Delta lines are ` - name==ver` / ` + name==ver`; everything else (Resolved,
    Checked, "Would make no changes", ...) is ignored.
    """
    uninstalled: dict[str, str] = {}
    installed: dict[str, str] = {}
    for line in output.splitlines():
        match = _DELTA_LINE.match(line)
        if not match:
            continue
        sign, name, version = match.groups()
        (uninstalled if sign == "-" else installed)[name] = version
    return uninstalled, installed


def drift_from_deltas(
    uninstalled: dict[str, str], installed: dict[str, str]
) -> list[tuple[str, str, str | None]]:
    """The revert signature: packages the sync would move BACKWARD or remove.

    Runs on the pre-sync dry-run, so intended 4b movement (installed -> lock
    target, including a bump's transitive upgrades) is a forward pair and is
    NOT drift; out-of-band installs are newer than the lock, so their pairs
    point backward — exactly the reversion. A fresh first run prints only `+`
    lines (nothing Uninstalled) and is silent. Lock target None means removal.
    Versions whose release part is not fully numeric are incomparable and are
    flagged (fallback), so weird-version manual drift still surfaces.
    """
    drift: list[tuple[str, str, str | None]] = []
    for name, current in sorted(uninstalled.items()):
        target = installed.get(name)
        if target is None:
            drift.append((name, current, None))
            continue
        cur_rel, tgt_rel = _release_tuple(current), _release_tuple(target)
        if cur_rel is None or tgt_rel is None or cur_rel > tgt_rel:
            drift.append((name, current, target))
    return drift


def sha_for_patch_id(pairs: list[tuple[str, str]], pid: str) -> str | None:
    """The pre-rebase commit sha a patch-id belonged to (grep '^<pid> ')."""
    for candidate, sha in pairs:
        if candidate == pid:
            return sha
    return None


# ---------------------------------------------------------------------------
# Subprocess pieces
# ---------------------------------------------------------------------------


def git_output(*args: str) -> str:
    """git stdout, or '' on failure (shell `2>/dev/null || echo ''`)."""
    proc = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else ""


def git_output_strict(*args: str) -> str:
    """git stdout; a nonzero exit is fatal (no `|| ` fallback in the shell)."""
    proc = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return proc.stdout


def collect_patch_ids(rev_range: str) -> list[tuple[str, str]]:
    """Patch-id pairs for a rev range, via the shell pipeline's exact shape:

    `for c in $(git rev-list R); do git show $c; done | git patch-id --stable`
    — every show is concatenated in rev-list order and fed to one patch-id
    run. Any failure degrades to a partial/empty set (`|| :` in the shell).
    """
    rev_list = subprocess.run(
        ["git", "rev-list", rev_range],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    show_blob = bytearray()
    for sha in rev_list.stdout.split():
        shown = subprocess.run(["git", "show", sha], capture_output=True, check=False)
        show_blob += shown.stdout
    patch_ids = subprocess.run(
        ["git", "patch-id", "--stable"],
        input=bytes(show_blob),
        capture_output=True,
        check=False,
    )
    return parse_patch_ids(patch_ids.stdout)


def uv_env() -> dict[str, str]:
    """os.environ + UV_PROJECT_ENVIRONMENT pinned to this repo's .venv."""
    env = dict(os.environ)
    env["UV_PROJECT_ENVIRONMENT"] = str(VENV_DIR)
    return env


def sync_cmd(extra_names: tuple[str, ...]) -> list[str]:
    """uv sync command: repo .venv, --inexact (keep custom-node packages),
    base extras plus any --extra passthrough."""
    cmd = ["uv", "sync", "--python", str(venv_python()), "--inexact"]
    for name in (*BASE_EXTRAS, *extra_names):
        cmd.extend(["--extra", name])
    return cmd


def run_sync(extra_names: tuple[str, ...]) -> None:
    """Step 4's sync; failure aborts with the rollback pointer."""
    if run_cmd(sync_cmd(extra_names), env=uv_env()) != 0:
        fail(
            "uv sync failed; aborting update "
            "(stack rollback: git reset --hard refs/pre-rebase-backup)."
        )


def warn_drift(extra_names: tuple[str, ...], skip: frozenset[str]) -> None:
    """Dry-run the sync before it runs and flag out-of-band venv changes it will revert.

    `skip` holds pending 4b bump names: a bump is a dry-run delta too, but it is
    intended, not drift. Warn-and-continue: the sync still runs, a bad update
    beats a failed one.
    """
    cmd = sync_cmd(extra_names)
    cmd.insert(2, "--dry-run")
    proc = subprocess.run(
        cmd,
        env=uv_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    # uv prints the transaction block on stderr when stdout is captured
    drift = [
        d
        for d in drift_from_deltas(*parse_sync_deltas(proc.stdout + proc.stderr))
        if d[0] not in skip
    ]
    if not drift:
        return
    print(
        "note: venv drift (uv pip install -U is not durable here; companion floors"
        " follow requirements.txt automatically — ask upstream to bump requirements.txt):"
    )
    for name, current, lock in drift:
        print(f"  - {name}: installed {current} -> lock {lock if lock else 'removed'}")


# ---------------------------------------------------------------------------
# Main flow — step numbers match update-local.sh's comments
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    opts = parse_args(argv)
    os.chdir(REPO_ROOT)

    current = git_output_strict("branch", "--show-current").strip()
    if not current:
        fail("detached HEAD; refusing to update.")
    if current == "master":
        fail("on master; run this from the patch-stack branch (local).")

    # 1. fetch upstream master (FF-only refspec)
    old_master = git_output("rev-parse", "master").strip()
    print("=== fetching upstream master ===")
    if run_cmd(["git", "fetch", "--quiet", "upstream", "master:master"]) != 0:
        fail("upstream fetch failed (non-FF or network); aborting before mirror/rebase.")

    # 2. mirror origin/master (remote mutation — guarded, see runbook header)
    new = git_output_strict("rev-parse", "--short", "master").strip()
    old = git_output("rev-parse", "--short", "origin/master").strip() or "none"
    if old == new:
        print(f"mirror: origin/master already at {new}.")
    elif opts.mirror == "no":
        print(f"mirror: --no-mirror; origin/master stays at {old} (push manually when ready).")
    else:
        if opts.mirror == "yes":
            answer = "y"
        else:
            # non-interactive runs (EOF) land here too -> skipped, the safe default
            try:
                answer = input(f"mirror: push origin/master {old} -> {new}? [y/N] ")
            except EOFError:
                answer = ""
        if answer[:1] in ("y", "Y"):
            if run_cmd(["git", "push", "--quiet", "origin", "master:master"]) != 0:
                print("mirror push failed (diverged remote?); continuing locally.", file=sys.stderr)
        else:
            print("mirror: skipped.")

    # 3. rebase local onto master
    status = git_output_strict("status", "--porcelain", "-uno")
    if status.strip():
        # parity with the shell: dirty tree only warns; the flow continues to sync
        print("tree is dirty; rebasing must be done manually.")
    else:
        pre_tip = git_output_strict("rev-parse", "local").strip()
        pre_count = int(git_output_strict("rev-list", "--count", "master..local").strip())

        # "<patch-id> <sha>" per commit — the drop-check primitive
        pre_pairs = collect_patch_ids("master..local")

        # named backup ref, not ORIG_HEAD (any ref-moving op clobbers ORIG_HEAD)
        run_cmd(["git", "update-ref", BACKUP_REF, pre_tip])

        if run_cmd(["git", "rebase", "master"]) != 0:
            print("rebase conflict; resolve then git rebase --continue, or git rebase --abort.", file=sys.stderr)
            print("pyproject.toml conflicts: merge hunks by hand — runbook:", file=sys.stderr)
            print("  .tmp/proposals/update-rebase-tooling.md §3b (then: uv lock --check)", file=sys.stderr)
            short_tip = git_output("rev-parse", "--short", pre_tip).strip() or pre_tip
            fail(f"pre-rebase stack tip: refs/pre-rebase-backup ({short_tip})")

        post_pairs = collect_patch_ids("master..local")
        missing = missing_patch_ids(pre_pairs, post_pairs)
        if missing:
            # classify against the patches this fetch actually brought in (master..upstream/master is
            # always empty here: step 1 already FF'd master to upstream/master). pid drift is inherent —
            # a replayed commit resolving with different context gets a new pid; durable skip ledger is
            # the follow-up (postponed-plan.md P-b), not implemented yet.
            master_pairs = collect_patch_ids(f"{old_master}..master") if old_master else []
            lost = lost_patch_ids(missing, master_pairs)
            for pid in missing:
                sha = sha_for_patch_id(pre_pairs, pid)
                if sha:
                    run_cmd(["git", "log", "-1", "--format=  absorbed: %h %s", sha])
            if lost and not opts.force:
                post_count = int(git_output_strict("rev-list", "--count", "master..local").strip())
                print(
                    f"rebase truly dropped commits (master..local: {pre_count} -> {post_count}); "
                    "refusing (--force to accept):",
                    file=sys.stderr,
                )
                for pid in lost:
                    sha = sha_for_patch_id(pre_pairs, pid)
                    if sha:
                        run_cmd(["git", "log", "-1", "--format=  LOST: %h %s", sha])
                sys.exit(1)
        if pre_count != 0:
            print(f"rebase: {pre_count} stack commit(s) intact on master ({new}).")

        # hard gate: a pyproject-touching rebase must still resolve a valid lock
        if git_output_strict("diff", "--name-only", pre_tip, "local", "--", "pyproject.toml").strip():
            print("=== uv lock --check (pyproject.toml changed in rebase) ===")
            if run_cmd(["uv", "lock", "--check"], env=uv_env()) != 0:
                print(
                    "uv lock --check FAILED after a pyproject.toml-touching rebase; fix the merge — runbook:",
                    file=sys.stderr,
                )
                print("  .tmp/proposals/update-rebase-tooling.md §3b", file=sys.stderr)
                sys.exit(1)

    # 4. sync env
    # --inexact: keep packages installed for custom_nodes that the lock does not own; a bare
    # `uv sync` prunes them and silently breaks nodes until they reinstall.
    # --python + UV_PROJECT_ENVIRONMENT: install into this repo's .venv, never ~/.venv.
    # --extra dev supplies pytest (and ruff) for the gates below.
    requirements = REQUIREMENTS_TXT.read_text(encoding="utf-8") if REQUIREMENTS_TXT.exists() else ""
    pyproject_text = PYPROJECT_TOML.read_text(encoding="utf-8")
    req_pins = extract_pins(requirements)
    req_map = dict(req_pins)
    proj_floors = parse_pyproject_floors(pyproject_text)
    # 4b pending floor changes computed before the sync so warn_drift can exclude
    # them: a pending floor bump shows up in the pre-sync dry-run as a delta, but
    # it is intended, not out-of-band drift. Anything else the dry-run would change
    # IS drift — warn now, before the sync (and the floor resync) silently revert it.
    pending = [(name, version) for name, version in req_pins if proj_floors.get(name) != version]
    for name in sorted(proj_floors):
        if name not in req_map:
            print(
                f"notice: {name} is no longer ==-pinned in requirements.txt; "
                f"leaving its pyproject floor (>= {proj_floors[name]}) as-is."
            )
    warn_drift(opts.extras, skip=frozenset(name for name, _ in pending))
    print("=== uv sync (rocm + manager + dev) ===")
    run_sync(opts.extras)

    # 4b. companion floors: pyproject [project] dependencies pins are the spec;
    # requirements.txt exact pins drive them. Rewrite each managed floor to
    # >=<requirements-exact> (fresh clones then reproduce exact companion
    # versions from a bare `uv sync`), re-resolve the lock (plain `uv lock` keeps
    # transitives via the locked strategy), and commit so the next run's
    # dirty-tree guard still rebases. torch etc. stay unpinned in
    # requirements.txt; hardware extras live in pyproject extras, untouched.
    if pending:
        print("=== companion floor update (requirements.txt -> pyproject) ===")
        text = pyproject_text
        for name, version in pending:
            print(f"bumping {name}: floor >= {proj_floors.get(name) or 'absent'} -> >= {version}")
            text = set_floor(text, name, version)
        PYPROJECT_TOML.write_text(text, encoding="utf-8")
        if run_cmd(["uv", "lock"], env=uv_env()) != 0:
            fail(
                "uv lock failed after rewriting floors; pyproject.toml is left in the tree "
                "(stack rollback: git reset --hard refs/pre-rebase-backup)."
            )
        run_sync(opts.extras)
        if run_cmd(["git", "add", "pyproject.toml"]) != 0 or run_cmd(
            ["git", "commit", "-m", "Update companion pins to requirements exacts"]
        ) != 0:
            fail(
                "git commit of the companion-floor update failed; the edit stays in the "
                "tree — commit pyproject.toml by hand, then rerun."
            )

    # 5. custom node deps the workspace/lock missed (install ends with its built-in post-install audit)
    print("=== custom node dependencies ===")
    run_quietly([str(venv_python()), str(CUSTOM_NODES_DEPS), "install"])

    # 6. post-bump gates
    if opts.gates:
        print("=== gates ===")
        failed = False
        pyfiles = [
            name
            for name in git_output_strict("diff", "--name-only", "master...local", "--", "*.py").splitlines()
            if name.strip()
        ]
        ruff = venv_tool("ruff")
        if ruff.exists() and pyfiles:
            # fork-touched files only; names from git, no spaces in this repo
            if run_cmd([str(ruff), "check", *pyfiles]) != 0:
                failed = True
        elif not ruff.exists():
            print("gate: ruff not in .venv — skipped (add to dev extra, or: uvx ruff check <files>)")
        if opts.tests:
            # no -x: list every failing test, not just the first
            if run_cmd([str(venv_python()), "-m", "pytest", "tests-unit", "-q"]) != 0:
                failed = True
        # main runs full app init at import; the rest covers the import graph silently.
        if run_cmd([str(venv_python()), "-c", _IMPORT_GATE]) != 0:
            failed = True
        if failed:
            fail("gates FAILED; stack rollback: git reset --hard refs/pre-rebase-backup (ref kept).")
    else:
        print("gates skipped (--skip-gates).")

    # everything passed — the backup ref has served its purpose
    run_cmd(["git", "update-ref", "-d", BACKUP_REF])

    print("update complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
