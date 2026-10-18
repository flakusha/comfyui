#!/usr/bin/env python3
"""Install and audit dependencies declared by custom_nodes/* packages.

Discovers pyproject.toml and requirements.txt under custom_nodes/, parses
PEP 508 dependency specifiers, checks them against the active venv, and
installs only genuinely missing packages via ``uv pip install``.

Safety invariants:
  - NEVER calls ``uv sync`` (would prune custom-node packages)
  - NEVER calls ``uv pip uninstall`` or ``--reinstall``
  - NEVER passes ``--upgrade`` (would upgrade existing packages)
  - Only installs packages that are genuinely missing from the venv
  - Version conflicts on already-installed packages are warnings, not auto-fixed
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
    declared_by: list[str] = field(default_factory=list)  # nodes declaring this package


@dataclass
class AuditResult:
    """Audit outcome for one unique package."""

    package: str
    specifier: str
    installed_version: str | None
    status: str  # "ok" | "missing" | "conflict"
    declared_by: list[str] = field(default_factory=list)  # node names


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


def parse_pyproject(path: Path) -> list[DepEntry]:
    """Parse [project].dependencies from a pyproject.toml."""
    node = path.parent.name
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError):
        return []
    project = data.get("project")
    if not isinstance(project, dict):
        return []
    deps = project.get("dependencies", [])
    if not isinstance(deps, list):
        return []
    entries: list[DepEntry] = []
    for raw in deps:
        if not isinstance(raw, str):
            continue
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = Requirement(raw)
        except Exception:
            continue
        # Skip URL-based requirements (e.g. sam2@git+https://...)
        if "@" in raw and ("://" in raw or "git+" in raw):
            continue
        entries.append(
            DepEntry(
                package=canonicalize_name(req.name),
                specifier=str(req.specifier),
                source="pyproject.toml",
                node=node,
                raw=raw,
            )
        )
    return entries


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
        try:
            req = Requirement(line)
        except Exception:
            continue
        # Skip URL-based requirements (e.g. sam2@git+https://...)
        if "@" in line and ("://" in line or "git+" in line):
            continue
        entries.append(
            DepEntry(
                package=canonicalize_name(req.name),
                specifier=str(req.specifier),
                source="requirements.txt",
                node=node,
                raw=line,
            )
        )
    return entries


def discover_all_deps() -> list[DepEntry]:
    """Walk custom_nodes/ and collect all dependency declarations."""
    all_entries: list[DepEntry] = []
    for path, source_type in discover_dep_files():
        if source_type == "pyproject.toml":
            all_entries.extend(parse_pyproject(path))
        else:
            all_entries.extend(parse_requirements_txt(path))
    return all_entries


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
        if key not in merged:
            entry.declared_by = [entry.node]
            merged[key] = entry
        else:
            existing = merged[key]
            if entry.node not in existing.declared_by:
                existing.declared_by.append(entry.node)
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
    if not HAS_PACKAGING:
        print(
            "ERROR: 'packaging' library is required but not installed.\n"
            "Install it with: uv pip install packaging",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(
        description="Audit and install custom_nodes dependencies",
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

    args = parser.parse_args()

    # Discover + deduplicate
    entries = discover_all_deps()
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
    return 1 if has_issues else 0


if __name__ == "__main__":
    sys.exit(main())
