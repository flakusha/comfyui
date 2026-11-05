#!/usr/bin/env python3
"""Install and audit dependencies declared by custom_nodes/* packages.

Discovers pyproject.toml and requirements.txt under custom_nodes/, parses
PEP 508 dependency specifiers, checks them against the active venv, and
installs only genuinely missing packages via ``uv pip install``. A manifest
entry may declare ``extras``; each name expands from that node's
``[project.optional-dependencies]`` and joins the same audit/install path.

Safety invariants:
  - NEVER calls ``uv sync`` (would prune custom-node packages)
  - NEVER calls ``uv pip uninstall`` or ``--reinstall``
  - NEVER passes ``--upgrade`` (would upgrade existing packages)
  - Only installs packages that are genuinely missing from the venv
  - Version conflicts on already-installed packages are warnings, not auto-fixed
  - Requirements whose environment marker does not match this platform are ignored
    (e.g. ``amd-adlx ; platform_system == 'Windows'`` is not installable on Linux)

Subcommands:
  audit    report missing/conflicting/OK packages
  install  install missing packages (uv pip install)
  sync     clone/checkout custom_nodes/* per custom_nodes.manifest.toml (git
           only, no packages). sync owns ref pinning for manifest-managed
           nodes and replaces ComfyUI-Manager's install/update for them;
           once every node is manifest-managed, the "manager" extra in
           pyproject.toml can be dropped (deliberately left in place).
           sync --stamp-patches-base NAME... settles NAME at its pinned ref
           (patch stack applied) and records the resolved commit as the
           node's patches_base: the upstream commit the stack was last
           known-good on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import Version
    HAS_PACKAGING = True
except ImportError:
    HAS_PACKAGING = False

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DepEntry:
    """A single discovered dependency declaration."""

    package: str  # canonical (PEP 503) name
    specifier: str  # raw specifier string, e.g. ">=1.26.4" or ""
    source: str  # "pyproject.toml" or "requirements.txt"
    node: str  # custom_nodes subdirectory name
    raw: str  # original line from the file
    declared_by: list[str] = field(default_factory=list)  # declaring node or node[extra]
    extra: str = ""  # [project.optional-dependencies] group; "" = base deps


@dataclass
class AuditResult:
    """Audit outcome for one unique package."""

    package: str
    specifier: str
    installed_version: str | None
    status: str  # "ok" | "missing" | "conflict"
    declared_by: list[str] = field(default_factory=list)  # declaring node or node[extra]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

CUSTOM_NODES_DIR = Path(__file__).resolve().parent.parent / "custom_nodes"


def discover_dep_files() -> list[tuple[Path, str]]:
    """Return (path, source_type) for every dep file under custom_nodes/."""
    if not CUSTOM_NODES_DIR.is_dir():
        return []
    results: list[tuple[Path, str]] = []
    for entry in sorted(CUSTOM_NODES_DIR.iterdir()):
        if not entry.is_dir():
            continue
        pyproject = entry / "pyproject.toml"
        if pyproject.is_file():
            results.append((pyproject, "pyproject.toml"))
        req_txt = entry / "requirements.txt"
        if req_txt.is_file():
            results.append((req_txt, "requirements.txt"))
    return results


def marker_applies(req) -> bool:
    """True if the requirement's environment marker matches this machine.

    Marked requirements (e.g. ``amd-adlx ; platform_system == \"Windows\"``)
    are not installable elsewhere, so they must not be audited or installed
    on this platform.
    """
    marker = getattr(req, "marker", None)
    if marker is None:
        return True
    try:
        return bool(marker.evaluate())
    except Exception:
        return True


def req_to_entry(raw: str, node: str, source: str, extra: str = "") -> DepEntry | None:
    """DepEntry for one PEP 508 line, or None when it must be skipped:
    malformed, platform marker not matching this machine, or URL-based
    (e.g. sam2@git+https://...).
    """
    try:
        req = Requirement(raw)
    except Exception:
        return None
    # Skip requirements gated on a platform/env we are not on
    if not marker_applies(req):
        return None
    # Skip URL-based requirements (e.g. sam2@git+https://...)
    if "@" in raw and ("://" in raw or "git+" in raw):
        return None
    return DepEntry(
        package=canonicalize_name(req.name),
        specifier=str(req.specifier),
        source=source,
        node=node,
        raw=raw,
        extra=extra,
    )


def parse_pyproject(
    path: Path, extras: tuple[str, ...] = ()
) -> tuple[list[DepEntry], list[str]]:
    """Parse [project].dependencies plus any manifest-requested extras from
    [project.optional-dependencies].

    Returns (entries, problems). Problems name manifest extras the pyproject
    does not define; their packages cannot be expanded, so the caller treats
    any problem as fatal. Extra names match PEP 503-normalized (doc_export
    in the manifest honors a doc-export key upstream).
    """
    node = path.parent.name
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError):
        return [], []
    project = data.get("project")
    if not isinstance(project, dict):
        return [], []
    optional = project.get("optional-dependencies")
    optional = optional if isinstance(optional, dict) else {}
    defined = {
        canonicalize_name(name): name
        for name, reqs in optional.items()
        if isinstance(reqs, list)
    }
    problems = [
        f"{node}: manifest extra '{extra}' is not defined in "
        "[project.optional-dependencies]"
        for extra in extras
        if canonicalize_name(extra) not in defined
    ]

    base = project.get("dependencies", [])
    groups: list[tuple[str, list]] = [("", base)] if isinstance(base, list) else []
    for extra in extras:
        name = defined.get(canonicalize_name(extra))
        if name is not None:
            groups.append((extra, optional[name]))

    entries: list[DepEntry] = []
    for extra, reqs in groups:
        for raw in reqs:
            if not isinstance(raw, str):
                continue
            raw = raw.strip()
            if raw:
                entry = req_to_entry(raw, node, "pyproject.toml", extra)
                if entry:
                    entries.append(entry)
    return entries, problems


def parse_requirements_txt(path: Path) -> list[DepEntry]:
    """Parse a requirements.txt, skipping comments, options, and includes."""
    node = path.parent.name
    entries: list[DepEntry] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            # Skip options (--index-url, -r other.txt, -e ., etc.)
            continue
        # Strip inline comments
        if " #" in line:
            line = line.split(" #", 1)[0].strip()
        if not line:
            continue
        entry = req_to_entry(line, node, "requirements.txt")
        if entry:
            entries.append(entry)
    return entries


def discover_all_deps() -> tuple[list[DepEntry], list[str]]:
    """Walk custom_nodes/ and collect all dependency declarations.

    Manifest [nodes.*].extras expand from each node's pyproject
    [project.optional-dependencies]; extras declared for a node that only
    ships requirements.txt are a problem — there is no metadata to expand.
    """
    pins: dict[str, NodePin] = {}
    if MANIFEST_PATH.is_file():
        try:
            pins = parse_manifest(MANIFEST_PATH.read_text(encoding="utf-8"))
        except FileNotFoundError:
            pins = {}
    all_entries: list[DepEntry] = []
    problems: list[str] = []
    for path, source_type in discover_dep_files():
        pin = pins.get(path.parent.name)
        if source_type == "pyproject.toml":
            entries, node_problems = parse_pyproject(
                path, pin.extras if pin else ()
            )
            all_entries.extend(entries)
            problems.extend(node_problems)
        else:
            if pin and pin.extras and not (path.parent / "pyproject.toml").is_file():
                problems.append(
                    f"{path.parent.name}: manifest declares extras "
                    f"({', '.join(pin.extras)}) but the node has no "
                    "pyproject.toml to expand them from"
                )
            all_entries.extend(parse_requirements_txt(path))
    return all_entries, problems


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def deduplicate(entries: list[DepEntry]) -> dict[str, DepEntry]:
    """Merge declarations for the same package, keeping the most restrictive specifier.

    Returns a dict keyed by canonical package name.
    """
    merged: dict[str, DepEntry] = {}
    for entry in entries:
        key = entry.package
        source = f"{entry.node}[{entry.extra}]" if entry.extra else entry.node
        if key not in merged:
            entry.declared_by = [source]
            merged[key] = entry
        else:
            existing = merged[key]
            if source not in existing.declared_by:
                existing.declared_by.append(source)
            existing_spec = existing.specifier
            new_spec = entry.specifier
            if not existing_spec and new_spec:
                existing.specifier = new_spec
                existing.raw = entry.raw
            elif existing_spec and new_spec:
                # Both have specifiers — keep the more restrictive one
                # Use SpecifierSet to compare: prefer the one with more constraints
                try:
                    from packaging.specifiers import SpecifierSet
                    existing_ss = SpecifierSet(existing_spec)
                    new_ss = SpecifierSet(new_spec)
                    # Count constraints as a proxy for restrictiveness
                    existing_count = len(list(existing_ss))
                    new_count = len(list(new_ss))
                    if new_count > existing_count:
                        existing.specifier = new_spec
                        existing.raw = entry.raw
                except Exception:
                    # Fallback: keep existing
                    pass
    return merged


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def audit_package(package: str, specifier: str) -> AuditResult:
    """Check one package against the installed environment."""
    try:
        installed = pkg_version(package)
    except PackageNotFoundError:
        return AuditResult(
            package=package,
            specifier=specifier,
            installed_version=None,
            status="missing",
        )

    if not specifier:
        return AuditResult(
            package=package,
            specifier=specifier,
            installed_version=installed,
            status="ok",
        )

    try:
        from packaging.specifiers import SpecifierSet

        spec_set = SpecifierSet(specifier)
        if Version(installed) in spec_set:
            status = "ok"
        else:
            status = "conflict"
    except Exception:
        status = "ok"  # Can't parse specifier — assume OK

    return AuditResult(
        package=package,
        specifier=specifier,
        installed_version=installed,
        status=status,
    )


def run_audit(deps: dict[str, DepEntry]) -> list[AuditResult]:
    """Audit all deduplicated packages."""
    results: list[AuditResult] = []
    for package in sorted(deps):
        entry = deps[package]
        result = audit_package(package, entry.specifier)
        result.declared_by = entry.declared_by
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Persisted state map
# ---------------------------------------------------------------------------


DEPS_STATE_FILE = (
    Path(__file__).resolve().parent.parent / ".local-state" / "custom-nodes-deps.json"
)


def deps_fingerprint() -> str:
    """sha256 over every discovered dep file's bytes (path-delimited, discovery order).

    The manifest joins the digest: [nodes.*].extras change the parsed dep
    set without touching any dep file under custom_nodes/.
    """
    digest = hashlib.sha256()
    for path, _source_type in discover_dep_files():
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    if MANIFEST_PATH.is_file():
        digest.update(str(MANIFEST_PATH).encode("utf-8"))
        digest.update(b"\0")
        digest.update(MANIFEST_PATH.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def save_dep_state(results: list[AuditResult]) -> None:
    """Record installed versions after a fully clean audit (best-effort cache).

    The audit itself still runs every time — it is the only venv-side check for
    packages the lock does not own; the map records what it last verified.
    """
    try:
        state = {
            "schema": 1,
            "inputs_sha256": deps_fingerprint(),
            "versions": {
                r.package: r.installed_version for r in results if r.installed_version
            },
        }
        DEPS_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        DEPS_STATE_FILE.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"note: could not write {DEPS_STATE_FILE.name}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def install_missing(
    results: list[AuditResult], dry_run: bool = False
) -> list[AuditResult]:
    """Install missing packages via uv pip install. Returns post-install audit."""
    missing = [r for r in results if r.status == "missing"]
    if not missing:
        return results

    if dry_run:
        return results

    for r in missing:
        cmd = ["uv", "pip", "install", "--python", sys.executable]
        if r.specifier:
            cmd.append(f"{r.package}{r.specifier}")
        else:
            cmd.append(r.package)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  ERROR installing {r.package}: {e}", file=sys.stderr)
        except FileNotFoundError:
            print("  ERROR: 'uv' not found in PATH", file=sys.stderr)
            return results

    # Re-audit after install, preserving declared_by from original results
    declared_by_map = {r.package: r.declared_by for r in missing}
    post = run_audit({r.package: DepEntry(
        package=r.package,
        specifier=r.specifier,
        source="",
        node="",
        raw="",
    ) for r in missing})
    for r in post:
        r.declared_by = declared_by_map.get(r.package, [])
    return post


# ---------------------------------------------------------------------------
# Manifest-managed checkouts (custom_nodes.manifest.toml -> git refs)
# ---------------------------------------------------------------------------


class SyncError(Exception):
    """A manifest ref could not be resolved to a commit."""


@dataclass(frozen=True)
class NodePin:
    """One [nodes.<name>] manifest entry: clone URL plus pinned ref."""

    url: str
    ref: str
    extras: tuple[str, ...] = ()  # [project.optional-dependencies] groups to install
    patches: tuple[str, ...] = ()  # quilt-style local patches, applied in order
    # commit the patch stack was last known-good on; None = unrecorded
    # (legacy stacks that predate the field), "unknown" = explicitly unrecorded
    patches_base: str | None = None


MANIFEST_PATH = Path(__file__).resolve().parent.parent / "custom_nodes.manifest.toml"

_HASH_RE = re.compile(r"^[0-9a-f]{7,40}$")
_HEAD_OFFSET_RE = re.compile(r"^HEAD~(\d+)$")
# git refnames forbid these (approximate check, enough to reject typos like
# "HEAD~x" or "a:b" at parse time instead of at resolution time)
_BAD_REF_RE = re.compile(r"[~^:\s\[\]*?\\]|@{")


def classify_ref(ref: str) -> tuple[str, str]:
    """Classify a manifest ref as ("latest"|"hash"|"head_offset"|"branch", value).

    Semantics (full rationale in custom_nodes.manifest.toml):
      latest      HEAD of origin's default branch at fetch time
      hash        exact commit, 7-40 hex chars (resolved against the fetched
                  object store at sync time)
      head_offset HEAD~N relative to the remote default branch's HEAD at
                  fetch time, never local state
      branch      tag or branch name; tags win name collisions; a branch
                  tracks origin/<branch> at fetch time
    """
    if ref == "latest":
        return ("latest", "")
    if ref == "HEAD":
        # bare HEAD would silently mean origin's default-branch tip; git
        # forbids a branch named HEAD anyway, so it is never a valid pin
        raise ValueError('bare "HEAD" is ambiguous; pin "latest" or a branch/tag name')
    m = _HEAD_OFFSET_RE.match(ref)
    if m:
        return ("head_offset", m.group(1))
    if _HASH_RE.match(ref):
        return ("hash", ref)
    if not ref or ref.startswith("-") or _BAD_REF_RE.search(ref):
        raise ValueError(f"invalid ref: {ref!r}")
    return ("branch", ref)


def parse_manifest(text: str) -> dict[str, NodePin]:
    """Parse custom_nodes.manifest.toml content into name -> NodePin.

    Strict on purpose: unknown keys (typo insurance), non-plain directory
    names, malformed extras, and refs that do not classify cleanly are all
    errors.
    """
    nodes = tomllib.loads(text).get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        raise ValueError("manifest must define a non-empty [nodes] table")
    pins: dict[str, NodePin] = {}
    for name, table in nodes.items():
        if not isinstance(table, dict):
            raise ValueError(f"[nodes.{name}]: expected a table with url and ref")
        unknown = set(table) - {"url", "ref", "extras", "patches", "patches_base"}
        if unknown:
            raise ValueError(f"[nodes.{name}]: unknown key(s): {', '.join(sorted(unknown))}")
        url = table.get("url")
        ref = table.get("ref")
        if not isinstance(url, str) or not url:
            raise ValueError(f"[nodes.{name}]: url must be a non-empty string")
        if url.startswith("-"):
            # option-injection into `git clone <url>` argv
            raise ValueError(f"[nodes.{name}]: url must not start with '-'")
        if not isinstance(ref, str) or not ref:
            raise ValueError(f"[nodes.{name}]: ref must be a non-empty string")
        if name in (".", "..") or "/" in name or "\\" in name or not name.strip() or name != name.strip():
            raise ValueError(f"[nodes.{name}]: not a plain custom_nodes/ directory name")
        try:
            classify_ref(ref)
        except ValueError as e:
            raise ValueError(f"[nodes.{name}]: {e}") from None
        extras: tuple[str, ...] = ()
        extras_raw = table.get("extras", [])
        if extras_raw:
            if not isinstance(extras_raw, list) or not all(
                isinstance(e, str) and e.strip() for e in extras_raw
            ):
                raise ValueError(
                    f"[nodes.{name}]: extras must be a list of non-empty strings"
                )
            # canonicalize without 'packaging': sync must run without it
            norm = [re.sub(r"[-_.]+", "-", e).lower() for e in extras_raw]
            if len(set(norm)) != len(norm):
                raise ValueError(
                    f"[nodes.{name}]: duplicate extra(s): {', '.join(extras_raw)}"
                )
            extras = tuple(extras_raw)
        patches: tuple[str, ...] = ()
        patches_raw = table.get("patches", [])
        if patches_raw:
            if not isinstance(patches_raw, list) or not all(
                isinstance(p, str) and p.strip() for p in patches_raw
            ):
                raise ValueError(
                    f"[nodes.{name}]: patches must be a list of non-empty strings"
                )
            for p in patches_raw:
                if p != p.strip() or "/" in p or "\\" in p or p in (".", "..") or p.startswith("-"):
                    raise ValueError(
                        f"[nodes.{name}]: patch {p!r} is not a plain file name relative "
                        "to custom_nodes_patches/"
                    )
            if len(set(patches_raw)) != len(patches_raw):
                raise ValueError(
                    f"[nodes.{name}]: duplicate patch(es): {', '.join(patches_raw)}"
                )
            patches = tuple(patches_raw)
        patches_base = table.get("patches_base")
        if patches_base is not None:
            if not isinstance(patches_base, str):
                raise ValueError(f"[nodes.{name}]: patches_base must be a string")
            if patches_base != "unknown" and not _HASH_RE.match(patches_base):
                raise ValueError(
                    f"[nodes.{name}]: patches_base must be a 7-40 char lowercase hex "
                    'commit hash or "unknown"'
                )
            if not patches:
                raise ValueError(f"[nodes.{name}]: patches_base requires patches")
        pins[name] = NodePin(
            url=url, ref=ref, extras=extras, patches=patches, patches_base=patches_base
        )
    return pins


def parse_symref_head(ls_remote_output: str) -> str | None:
    """`git ls-remote --symref origin HEAD` output -> default branch name.

    The first line looks like "ref: refs/heads/master<TAB>HEAD"; anything
    else (network failure, empty output) yields None.
    """
    for line in ls_remote_output.splitlines():
        if line.startswith("ref: "):
            ref = line[5:].split("\t")[0].strip()
            if ref.startswith("refs/heads/"):
                return ref[len("refs/heads/"):]
    return None


@dataclass(frozen=True)
class RepoState:
    """Snapshot of one custom_nodes/<name> checkout for plan_sync."""

    exists: bool
    is_git: bool = False
    dirty: bool = False  # uncommitted changes to TRACKED files
    head: str | None = None  # full hash of HEAD; None = unborn
    branch: str | None = None  # checked-out branch; None = detached
    can_ff: bool = False  # branch pin: HEAD is an ancestor of the target


@dataclass(frozen=True)
class Plan:
    """Next mutation for one node: clone|ok|ff|checkout_branch|detach|refuse."""

    verb: str
    detail: str = ""


def plan_sync(state: RepoState, target: str, branch: str | None) -> Plan:
    """Decide the next mutation for one node, or 'ok'/'refuse'.

    An existing clone already at the target hash is left alone whatever kind
    of pin it is (idempotence). A dirty tracked-file worktree is never moved.
    Branch pins end on the named local branch fast-forwarded to
    origin/<branch>, or refuse when diverged; every other pin ends detached
    at the resolved hash.
    """
    if not state.exists:
        return Plan("clone")
    if not state.is_git:
        return Plan(
            "refuse",
            "exists but is not a git clone (Manager-installed copy?); move it aside "
            "by hand and re-run — the manifest clone replaces it",
        )
    if state.head == target:
        return Plan("ok")
    if state.dirty:
        return Plan(
            "refuse",
            "dirty worktree (uncommitted changes to tracked files); commit or "
            "stash before this ref move",
        )
    if branch is None:
        return Plan("detach")
    if state.branch == branch:
        if state.head is None:
            return Plan("refuse", f"unborn HEAD on {branch}")
        if state.can_ff:
            return Plan("ff")
        return Plan(
            "refuse",
            f"{branch} diverged from origin/{branch} (local commits?); refusing "
            "to discard them",
        )
    return Plan("checkout_branch")


def _git_out(repo: Path, *args: str) -> str:
    """git stdout inside a repo, or '' on failure (update_local.git_output style)."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )
    return proc.stdout if proc.returncode == 0 else ""


def _git_rc(repo: Path, *args: str) -> bool:
    """True when git exits 0 inside a repo. Existence probes run --quiet, so
    the exit code — not stdout — carries the answer."""
    return (
        subprocess.run(["git", "-C", str(repo), *args], capture_output=True).returncode
        == 0
    )


def _git_env_rc(repo: Path, env: dict[str, str], *args: str) -> bool:
    """_git_rc with extra environment (e.g. GIT_INDEX_FILE) on the git call."""
    return (
        subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, env=env
        ).returncode
        == 0
    )


def _git_env_out(repo: Path, env: dict[str, str], *args: str) -> str:
    """_git_out with extra environment (e.g. GIT_INDEX_FILE) on the git call."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, env=env
    )
    return proc.stdout if proc.returncode == 0 else ""


def _git_run(repo: Path | None, *args: str) -> int:
    """Run git with stdio inherited (clone/fetch/checkout progress is output)."""
    cmd = ["git"] if repo is None else ["git", "-C", str(repo)]
    try:
        return subprocess.run([*cmd, *args]).returncode
    except FileNotFoundError:
        print("ERROR: 'git' not found in PATH", file=sys.stderr)
        return 127


def is_dirty(porcelain: str) -> bool:
    """True when `git status --porcelain` shows tracked-file changes.

    Untracked entries ("?? ...") are not user changes a checkout would
    discard — git itself only refuses a checkout on a path collision.
    """
    return any(not line.startswith("?? ") for line in porcelain.splitlines())


PATCHES_DIR = Path(__file__).resolve().parent.parent / "custom_nodes_patches"


def _patch_files(name: str, pin: NodePin) -> list[Path]:
    """Absolute paths of a node's declared patch files, in manifest order.

    A declared file that does not exist is a hard error: a patch list is a
    promise about the checkout's content, never silently partially kept.
    """
    files = []
    for p in pin.patches:
        path = PATCHES_DIR / name / p
        if not path.is_file():
            raise SyncError(f"declared patch file not found: {path}")
        files.append(path)
    return files


def _patches_applied(repo: Path, files: list[Path]) -> bool:
    """True when the worktree diff is exactly the declared patch stack on top
    of HEAD, i.e. the whole stack unwinds back to the pinned ref's tree."""
    # Per-patch `apply --check --reverse` false-negatives on overlapping hunks
    # (a later patch mutates an earlier patch's context), and a single
    # --check invocation validates all diffs against one state without
    # simulating them sequentially — so unwind for real, onto a temp index
    # seeded with the worktree, and require the result to equal HEAD's tree.
    with tempfile.NamedTemporaryFile(suffix=".index") as idx:
        env = {**os.environ, "GIT_INDEX_FILE": idx.name}
        if not _git_env_rc(repo, env, "read-tree", "HEAD"):
            return False
        if not _git_env_rc(repo, env, "add", "-u"):
            return False
        for f in reversed(files):
            if not _git_env_rc(repo, env, "apply", "--cached", "--reverse", str(f)):
                return False
        tree = _git_env_out(repo, env, "write-tree")
        return bool(tree) and tree == _git_out(repo, "rev-parse", "HEAD^{tree}")


def apply_patches(repo: Path, name: str, files: list[Path]) -> str | None:
    """Converge the worktree onto ref + declared patches; an error message or None.

    Model: plain `git apply` (not `git am`) so patches stay simple diffs,
    trivially regenerated and never needing commit identity. On every sync
    the worktree must look like either the clean pinned ref or the ref with
    the full stack applied — anything else is a refusal, never discarded.
    """
    if _git_out(repo, "diff", "HEAD"):
        if _patches_applied(repo, files):
            print(f"{name}: patches already applied ({len(files)} patch(es))")
            return None
        return (
            "dirty worktree changes are not explained by the declared patches; "
            "commit/stash them, or reset to the pinned ref and re-run"
        )
    for f in files:
        proc = subprocess.run(
            ["git", "-C", str(repo), "apply", str(f)], capture_output=True, text=True
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()
            return f"patch failed to apply: {f.name}: {detail}"
    print(f"{name}: applied {len(files)} patch(es)")
    return None


def warn_base_drift(name: str, pin: NodePin, target: str) -> None:
    """Warn (never refuse) when the recorded patches_base differs from the
    resolved ref.

    Drift is expected — patches apply fine across nearby commits, and
    `git apply` validates every hunk against the tree it lands on — so the
    base is traceability ("which commit was this stack known-good on"),
    not a safety gate. Stacks without a usable base (field absent for
    legacy stacks, or the explicit "unknown") stay silent.
    """
    base = pin.patches_base
    if not base or base == "unknown" or base == target:
        return
    print(
        f"{name}: WARNING: patches_base {base} differs from resolved ref {target} "
        f"— patch stack last known-good on {base}; re-stamp with "
        f"--stamp-patches-base once verified on {target}",
        file=sys.stderr,
    )


def repo_state(
    repo: Path, target: str | None = None, branch: str | None = None
) -> RepoState:
    """Snapshot one checkout's state for plan_sync.

    can_ff (the merge-base check) is computed only when a branch pin sits on
    its own branch behind the target — the only shape where the answer
    changes the plan.
    """
    if not (repo / ".git").exists():
        return RepoState(exists=repo.is_dir(), is_git=False)
    head = _git_out(repo, "rev-parse", "HEAD").strip() or None
    checked_out = _git_out(repo, "branch", "--show-current").strip() or None
    porcelain = _git_out(repo, "status", "--porcelain")
    dirty = is_dirty(porcelain)
    can_ff = False
    if target and branch and checked_out == branch and head and head != target:
        proc = subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", head, target],
            capture_output=True,
        )
        can_ff = proc.returncode == 0
    return RepoState(
        exists=True, is_git=True, dirty=dirty, head=head, branch=checked_out, can_ff=can_ff
    )


def remote_default_branch(repo: Path) -> str | None:
    """Origin's default branch name.

    `ls-remote --symref` is authoritative at fetch time; the clone-time
    origin/HEAD symref is the offline fallback.
    """
    branch = parse_symref_head(_git_out(repo, "ls-remote", "--symref", "origin", "HEAD"))
    if branch:
        return branch
    short = _git_out(repo, "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
    return short.strip().removeprefix("origin/") or None


def resolve_pin(repo: Path, ref: str) -> tuple[str, str | None]:
    """(full commit hash, pinned branch or None) for a manifest ref, post-fetch.

    Deterministic given the fetched refs: latest/HEAD~N resolve against the
    remote default branch's origin ref (never local HEAD); a named ref that
    resolves as a tag detaches at the tagged commit (tags win name
    collisions), one that resolves as origin/<name> is a branch pin; a hash
    must exist in the object store.
    """
    kind, value = classify_ref(ref)
    if kind in ("latest", "head_offset"):
        default = remote_default_branch(repo)
        if not default:
            raise SyncError(f"cannot determine origin's default branch (needed for {ref!r})")
        suffix = f"~{value}" if kind == "head_offset" else ""
        candidates = [f"refs/remotes/origin/{default}{suffix}^{{commit}}"]
        branch = None
    elif kind == "hash":
        candidates = [f"{value}^{{commit}}"]
        branch = None
    else:
        tag = _git_out(
            repo, "rev-parse", "--verify", "--quiet", f"refs/tags/{value}^{{commit}}"
        ).strip()
        if tag:
            return tag, None
        candidates = [f"refs/remotes/origin/{value}^{{commit}}"]
        branch = value
    for candidate in candidates:
        out = _git_out(repo, "rev-parse", "--verify", "--quiet", candidate).strip()
        if out:
            return out, branch
    raise SyncError(
        f"ref {ref!r} did not resolve (tried: {', '.join(candidates)}); "
        "are the tags/branches fetched?"
    )


REFS_STATE_FILE = (
    Path(__file__).resolve().parent.parent / ".local-state" / "custom-nodes-refs.json"
)


def save_refs_state(pins: dict[str, NodePin], resolved: dict[str, str]) -> None:
    """Record the resolved hash per settled node (best-effort, cache only).

    Each clone's HEAD is the primary record; this map is the at-a-glance
    copy of the last fully settled sync, alongside custom-nodes-deps.json.
    """
    state = {
        "schema": 1,
        "refs": {
            name: {"url": pins[name].url, "ref": pins[name].ref, "hash": resolved[name]}
            for name in sorted(resolved)
        },
    }
    try:
        REFS_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        REFS_STATE_FILE.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"note: could not write {REFS_STATE_FILE.name}: {exc}", file=sys.stderr)


def sync_node(
    name: str,
    pin: NodePin,
    *,
    dry_run: bool = False,
    refusals: set[str] | None = None,
) -> str | None:
    """Converge one manifest node; returns its resolved hash or None.

    A missing repo is cloned (all branches + tags), an existing one fetched
    first so every pin resolves at fetch time; the resolved full hash is
    echoed either way. Refusals print loudly and never mutate anything.
    A node declaring patches must additionally end at ref + stack: the
    worktree has to match either the clean ref or the ref with every patch
    applied — anything else refuses; a matching stack resets before a ref
    move and is re-applied after convergence.
    """
    repo = CUSTOM_NODES_DIR / name

    if repo.is_dir() and not (repo / ".git").exists():
        # refuse BEFORE any git op: git in a bare directory walks up into
        # whatever repo contains it (this one) and fetches from its origin
        print(
            f"{name}: REFUSED: exists but is not a git clone (Manager-installed "
            "copy?); move it aside by hand and re-run — the manifest clone "
            "replaces it",
            file=sys.stderr,
        )
        if refusals is not None:
            refusals.add(name)
        return None

    did: list[str] = []
    if not repo.is_dir():
        if dry_run:
            print(f"{name}: would clone {pin.url} at ref {pin.ref}")
            return None
        print(f"{name}: cloning {pin.url}")
        if _git_run(None, "clone", pin.url, str(repo)) != 0:
            print(f"{name}: FAILED: git clone {pin.url}", file=sys.stderr)
            return None
        did.append("cloned")
    elif not dry_run:
        if _git_run(repo, "fetch", "--prune", "--tags", "origin") != 0:
            print(f"{name}: FAILED: git fetch origin", file=sys.stderr)
            return None

    try:
        target, branch = resolve_pin(repo, pin.ref)
    except SyncError as e:
        print(f"{name}: FAILED: {e}", file=sys.stderr)
        if refusals is not None:
            refusals.add(name)
        return None

    for _ in range(3):  # checkout_branch may cascade into one ff; never more
        state = repo_state(repo, target=target, branch=branch)
        plan = plan_sync(state, target, branch)
        if plan.verb == "ok":
            warn_base_drift(name, pin, target)
            if pin.patches:
                if dry_run:
                    print(f"{name}: would apply {len(pin.patches)} patch(es)")
                    return None
                try:
                    err = apply_patches(repo, name, _patch_files(name, pin))
                except SyncError as e:
                    err = str(e)
                if err:
                    print(f"{name}: REFUSED: {err} (ref {pin.ref})", file=sys.stderr)
                    if refusals is not None:
                        refusals.add(name)
                    return None
            if did:
                print(f"{name}: {' + '.join(did)} -> {target} (ref {pin.ref})")
            else:
                print(f"{name}: ok {target} (ref {pin.ref})")
            return target
        patch_files: list[Path] | None = None
        if plan.verb == "refuse" and plan.detail.startswith("dirty worktree") and pin.patches:
            # a patch-declared node's dirt is legitimate when it is exactly
            # the applied stack; reset it so the ref move can proceed (the
            # stack is re-applied right after convergence)
            try:
                patch_files = _patch_files(name, pin)
            except SyncError as e:
                print(f"{name}: FAILED: {e}", file=sys.stderr)
                if refusals is not None:
                    refusals.add(name)
                return None
            if _patches_applied(repo, patch_files):
                if dry_run:
                    print(f"{name}: would reset the applied patch stack and re-converge")
                    return None
                if _git_run(repo, "reset", "--hard", "HEAD") != 0:
                    print(f"{name}: FAILED: git reset --hard HEAD", file=sys.stderr)
                    return None
                did.append("reset patch stack")
                continue
        if plan.verb == "refuse":
            print(f"{name}: REFUSED: {plan.detail} (ref {pin.ref})", file=sys.stderr)
            if refusals is not None:
                refusals.add(name)
            return None
        if dry_run:
            print(f"{name}: would {plan.verb} -> {target} (ref {pin.ref})")
            return None
        if plan.verb == "detach":
            if _git_run(repo, "checkout", "--detach", target) != 0:
                print(f"{name}: FAILED: git checkout --detach {target[:12]}", file=sys.stderr)
                return None
            did.append("detached")
        elif plan.verb == "checkout_branch":
            if _git_rc(repo, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"):
                if not _git_rc(
                    repo, "merge-base", "--is-ancestor", f"refs/heads/{branch}", target
                ):
                    print(
                        f"{name}: REFUSED: local {branch} diverged from the pinned ref "
                        "(local commits?); refusing to discard them",
                        file=sys.stderr,
                    )
                    if refusals is not None:
                        refusals.add(name)
                    return None
                args = ("checkout", branch)
            else:
                args = ("checkout", "-b", branch, f"origin/{branch}")
            if _git_run(repo, *args) != 0:
                print(f"{name}: FAILED: git {' '.join(args)}", file=sys.stderr)
                return None
            did.append(f"checked out {branch}")
        elif plan.verb == "ff":
            old = state.head
            if _git_run(repo, "merge", "--ff-only", f"origin/{branch}") != 0:
                print(f"{name}: FAILED: fast-forward {branch}", file=sys.stderr)
                return None
            did.append(f"fast-forwarded {branch} {old[:12]}..")
        else:  # clone is handled before the loop; anything else is a bug
            print(f"{name}: FAILED: unexpected plan verb {plan.verb}", file=sys.stderr)
            return None
    print(f"{name}: FAILED: could not converge onto the pinned ref", file=sys.stderr)
    return None


def run_sync_manifest(manifest: Path, dry_run: bool = False) -> int:
    """Converge custom_nodes/ to the manifest: clone missing repos, move every
    checkout onto its pinned ref, echo + record the resolved full hash per
    node. Exits 1 if any node refused or failed (a dry run exits 1 only on
    refusals; planned clone/moves are a plan summary, not failures)."""
    try:
        text = manifest.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
        return 1
    try:
        pins = parse_manifest(text)
    except (tomllib.TOMLDecodeError, ValueError) as e:
        print(f"ERROR: {manifest}: {e}", file=sys.stderr)
        return 1

    resolved: dict[str, str] = {}
    refusals: set[str] = set()
    for name in sorted(pins):
        hash_ = sync_node(name, pins[name], dry_run=dry_run, refusals=refusals)
        if hash_:
            resolved[name] = hash_
    failed = sorted(refusals)

    if dry_run:
        would = sorted(set(pins) - set(resolved) - refusals)
        print(
            f"dry run: {len(resolved)} at pin, {len(would)} would "
            f"clone/move ({', '.join(would) or 'none'}), "
            f"{len(failed)} refused — nothing written."
        )
    else:
        if failed:
            print(
                f"{len(failed)} node(s) refused/failed: {', '.join(failed)}",
                file=sys.stderr,
            )
        else:
            # only a fully settled sync may overwrite the last-settled record
            if resolved:
                save_refs_state(pins, resolved)
            print(f"all {len(pins)} manifest node(s) settled at their pinned refs.")
    return 1 if failed else 0


def stamp_manifest_text(text: str, name: str, sha: str) -> tuple[str, str | None]:
    """Insert or update `patches_base = "<sha>"` inside [nodes.<name>]'s text
    block, leaving every other byte of the manifest untouched (comments and
    formatting survive; a toml serializer would not). Returns
    (new_text, error); error is None on success.
    """
    lines = text.splitlines(keepends=True)
    header = re.compile(rf"^\[nodes\.{re.escape(name)}\]\s*(?:#.*)?$")
    start = next((i for i, l in enumerate(lines) if header.match(l)), None)
    if start is None:
        return text, f"[nodes.{name}]: table not found"
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("[")),
        len(lines),
    )
    existing = re.compile(r'^(\s*patches_base\s*=\s*)"[^"]*"')
    for i in range(start + 1, end):
        if existing.match(lines[i]):
            # re-stamp in place; anything after the closing quote (a trailing
            # comment) survives
            lines[i] = existing.sub(lambda m: f'{m.group(1)}"{sha}"', lines[i], count=1)
            return "".join(lines), None
    patches_key = re.compile(r"^\s*patches\s*=")
    for i in range(start + 1, end):
        if not patches_key.match(lines[i]):
            continue
        close = i
        while close < end and not lines[close].rstrip().endswith("]"):
            close += 1
        if close >= end:
            return text, (
                f"[nodes.{name}]: cannot find the end of the patches array; "
                "close it with a ']' line and re-run"
            )
        indent = re.match(r"\s*", lines[i]).group(0)
        lines.insert(close + 1, f'{indent}patches_base = "{sha}"\n')
        return "".join(lines), None
    return text, f"[nodes.{name}]: declares no patches to stamp a base for"


def run_stamp_patches_base(manifest: Path, names: list[str], dry_run: bool = False) -> int:
    """Settle the named patch-declaring node(s) at their pinned refs (patches
    applied) and record each resolved hash as the node's patches_base in the
    manifest — the commit the stack was just verified against. This is a
    subset sync: other nodes are untouched and the last-settled refs record
    (custom-nodes-refs.json) is deliberately not overwritten.
    """
    try:
        text = manifest.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
        return 1
    try:
        pins = parse_manifest(text)
    except (tomllib.TOMLDecodeError, ValueError) as e:
        print(f"ERROR: {manifest}: {e}", file=sys.stderr)
        return 1

    failures: list[str] = []
    for name in names:
        pin = pins.get(name)
        if pin is None:
            print(f"{name}: ERROR: not in {manifest}", file=sys.stderr)
            failures.append(name)
        elif not pin.patches:
            print(
                f"{name}: ERROR: declares no patches; patches_base has nothing to record",
                file=sys.stderr,
            )
            failures.append(name)

    resolved: dict[str, str] = {}
    refusals: set[str] = set()
    for name in names:
        if name in failures:
            continue
        hash_ = sync_node(name, pins[name], dry_run=dry_run, refusals=refusals)
        if hash_:
            resolved[name] = hash_
        elif not (dry_run and name not in refusals):
            failures.append(name)  # sync_node printed the refusal
        # else: a dry-run plan (would clone/move) — the prediction below
        # reports the stamp that would follow; a plan is not a failure

    stamped = 0
    for name in names:
        sha = resolved.get(name)
        if sha is None and dry_run and name not in refusals and name in pins:
            # a planned (dry-run) settle has no hash yet; resolve_pin against
            # the already-fetched refs predicts where convergence will land
            try:
                sha, _ = resolve_pin(CUSTOM_NODES_DIR / name, pins[name].ref)
            except SyncError:
                print(f"{name}: would clone/settle, then stamp its resolved hash")
        if sha is None:
            continue
        if dry_run:
            print(f'{name}: would stamp patches_base = "{sha}" in {manifest.name}')
            continue
        new_text, err = stamp_manifest_text(text, name, sha)
        if err:
            print(f"{name}: ERROR: {err}", file=sys.stderr)
            failures.append(name)
            continue
        try:
            parse_manifest(new_text)  # never leave broken TOML behind
        except (tomllib.TOMLDecodeError, ValueError) as e:
            print(
                f"{name}: ERROR: stamping would break the manifest: {e}",
                file=sys.stderr,
            )
            failures.append(name)
            continue
        text = new_text
        stamped += 1
        print(f'{name}: stamped patches_base = "{sha}" in {manifest.name}')

    if stamped:
        try:
            manifest.write_text(text, encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: could not write {manifest}: {exc}", file=sys.stderr)
            return 1
    if failures:
        print(
            f"{len(failures)} node(s) not stamped: {', '.join(sorted(set(failures)))}",
            file=sys.stderr,
        )
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table(results: list[AuditResult]) -> None:
    """Print a human-readable audit table."""
    if not results:
        print("No custom node dependencies found.")
        return

    # Column widths
    pkg_w = max(len(r.package) for r in results)
    spec_w = max(len(r.specifier or "(any)") for r in results)
    ver_w = max(len(r.installed_version or "-") for r in results)

    header = f"{'Package':<{pkg_w}}  {'Specifier':<{spec_w}}  {'Installed':>{ver_w}}  Status"
    print(header)
    print("-" * len(header))

    for r in results:
        spec = r.specifier or "(any)"
        ver = r.installed_version or "-"
        status_icon = {"ok": "OK", "missing": "MISSING", "conflict": "CONFLICT"}[r.status]
        print(f"{r.package:<{pkg_w}}  {spec:<{spec_w}}  {ver:>{ver_w}}  {status_icon}")

    # Summary
    ok = sum(1 for r in results if r.status == "ok")
    missing = sum(1 for r in results if r.status == "missing")
    conflict = sum(1 for r in results if r.status == "conflict")
    print()
    print(f"Summary: {ok} OK, {missing} missing, {conflict} conflict(s)")


def print_json(results: list[AuditResult]) -> None:
    """Print audit results as JSON."""
    output = []
    for r in results:
        output.append(
            {
                "package": r.package,
                "specifier": r.specifier,
                "installed_version": r.installed_version,
                "status": r.status,
                "declared_by": r.declared_by,
            }
        )
    print(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit and install custom_nodes dependencies; sync their git checkouts",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # audit
    audit_parser = sub.add_parser("audit", help="Report missing/conflicting/OK packages")
    audit_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # install
    install_parser = sub.add_parser("install", help="Install missing packages")
    install_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be installed"
    )
    install_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # sync (needs no 'packaging': git-side work only, no venv audit)
    sync_parser = sub.add_parser(
        "sync",
        help="Clone/checkout custom_nodes repos pinned in custom_nodes.manifest.toml",
    )
    sync_parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Manifest path (default: %(default)s)",
    )
    sync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only: resolve pins against last-fetched refs, mutate nothing",
    )
    sync_parser.add_argument(
        "--stamp-patches-base",
        nargs="+",
        metavar="NODE",
        help="Sync only these patch-declaring node(s), then record each node's "
        "resolved commit hash as its patches_base in the manifest "
        "(inserted next to patches, or updated in place on re-stamp)",
    )

    args = parser.parse_args()

    if args.command == "sync":
        if args.stamp_patches_base:
            return run_stamp_patches_base(
                args.manifest, args.stamp_patches_base, args.dry_run
            )
        return run_sync_manifest(args.manifest, args.dry_run)

    if not HAS_PACKAGING:
        print(
            "ERROR: 'packaging' library is required but not installed.\n"
            "Install it with: uv pip install packaging",
            file=sys.stderr,
        )
        return 1

    # Discover + deduplicate
    try:
        entries, extra_problems = discover_all_deps()
    except (tomllib.TOMLDecodeError, ValueError) as e:
        print(f"ERROR: {MANIFEST_PATH.name}: {e}", file=sys.stderr)
        return 1
    if extra_problems:
        for problem in extra_problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        print(
            "Fix the extras in custom_nodes.manifest.toml; refusing to audit "
            "a partially-expanded dependency set.",
            file=sys.stderr,
        )
        return 1
    deduped = deduplicate(entries)

    if not deduped:
        print("No custom node dependencies found.")
        return 0

    # Audit
    results = run_audit(deduped)

    if args.command == "audit":
        if args.json:
            print_json(results)
        else:
            print_table(results)
        # Exit 1 if any missing or conflict
        has_issues = any(r.status in ("missing", "conflict") for r in results)
        return 1 if has_issues else 0

    # install
    missing = [r for r in results if r.status == "missing"]
    conflicts = [r for r in results if r.status == "conflict"]

    if args.dry_run:
        if missing:
            print("Would install:")
            for r in missing:
                spec = f"{r.specifier}" if r.specifier else ""
                print(f"  {r.package}{spec}")
        else:
            print("Nothing to install — all packages present.")
        if conflicts:
            print()
            print("Warnings (version conflicts, NOT auto-fixed):")
            for r in conflicts:
                print(
                    f"  {r.package}: installed {r.installed_version}, "
                    f"need {r.specifier}"
                )
        if args.json:
            print_json(results)
        return 0

    # Actual install
    if missing:
        print(f"Installing {len(missing)} missing package(s)...")
        post_results = install_missing(results, dry_run=False)
    else:
        print("Nothing to install — all packages present.")
        post_results = results

    if conflicts:
        print()
        print("Warnings (version conflicts, NOT auto-fixed):")
        for r in conflicts:
            print(
                f"  {r.package}: installed {r.installed_version}, "
                f"need {r.specifier}"
            )

    if args.json:
        print_json(post_results)
    else:
        print()
        print("Post-install audit:")
        print_table(post_results)

    # Exit 1 if still missing or conflict
    has_issues = any(r.status in ("missing", "conflict") for r in post_results)
    if not has_issues:
        save_dep_state(post_results)
    return 1 if has_issues else 0


if __name__ == "__main__":
    sys.exit(main())
