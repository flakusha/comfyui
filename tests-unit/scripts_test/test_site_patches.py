"""Tests for custom_nodes_deps.apply_site_patches (pip-installed targets).

Resource contract (parallel-safe): every test owns its own `tmp_path`
subtree — repo/patches (fake patch dir) and repo/venv/fakepkg (fake
package) — and `git apply` runs with cwd inside that subtree. All shared
module state (PATCHES_DIR, SITE_PATCHES, importlib) is swapped per-test
via the monkeypatch fixture and auto-restored on failure, so tests pass
alone, in any order, and under xdist workers. No ports, no databases,
no files outside tmp_path.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import custom_nodes_deps as deps


PATCH = """\
diff --git a/core/mod.py b/core/mod.py
--- a/core/mod.py
+++ b/core/mod.py
@@ -1,4 +1,3 @@
 alpha
-beta
 gamma
 delta
"""

ORIG = "alpha\nbeta\ngamma\ndelta\n"
PATCHED = "alpha\ngamma\ndelta\n"


def _fake_package(tmp_path, inside=True, exists=True):
    """Lay out a fake repo root with a patched-file target; (pkg_dir, patch_path).

    apply_site_patches runs `git apply` with cwd=PATCHES_DIR.parent, so the
    layout mirrors the real one: patches/ inside the repo root, the package
    under it (or a sibling directory when inside=False).
    """
    repo = tmp_path / "repo"
    pkg = repo / "venv" / "fakepkg"
    if not inside:
        pkg = tmp_path / "outside" / "fakepkg"
    rel = "core/mod.py"
    (pkg / rel).parent.mkdir(parents=True)
    (pkg / rel).write_text(ORIG)
    (pkg / "__init__.py").write_text("")
    patch_dir = repo / "patches" / "fakepkg"
    patch_dir.mkdir(parents=True)
    patch = patch_dir / "0001-x.patch"
    if exists:
        patch.write_text(PATCH)
    return pkg, patch, rel


def _install(monkeypatch, tmp_path, pkg_dir):
    repo = tmp_path / "repo"
    monkeypatch.setattr(deps, "PATCHES_DIR", repo / "patches")
    monkeypatch.setattr(
        deps, "SITE_PATCHES", [("fakepkg", "core/mod.py", "0001-x.patch")]
    )
    spec = None
    if pkg_dir is not None:
        spec = SimpleNamespace(origin=str(pkg_dir / "__init__.py"))
    monkeypatch.setattr(
        deps,
        "importlib",
        SimpleNamespace(util=SimpleNamespace(find_spec=lambda name: spec)),
    )


def test_missing_patch_file_refuses(tmp_path, monkeypatch, capsys):
    pkg, _patch, _rel = _fake_package(tmp_path, exists=False)
    _install(monkeypatch, tmp_path, pkg)
    assert deps.apply_site_patches() is False
    assert "declared patch file not found" in capsys.readouterr().err


def test_unknown_package_is_skipped(tmp_path, monkeypatch, capsys):
    _fake_package(tmp_path)
    _install(monkeypatch, tmp_path, None)
    assert deps.apply_site_patches() is True
    assert "not installed" in capsys.readouterr().err


def test_outside_repo_target_refuses_without_mutation(tmp_path, monkeypatch, capsys):
    pkg, _patch, rel = _fake_package(tmp_path, inside=False)
    _install(monkeypatch, tmp_path, pkg)
    before = (pkg / rel).read_text()
    assert deps.apply_site_patches() is False
    assert "outside this repository" in capsys.readouterr().err
    assert (pkg / rel).read_text() == before


def test_dry_run_reports_without_touching(tmp_path, monkeypatch, capsys):
    pkg, _patch, rel = _fake_package(tmp_path)
    _install(monkeypatch, tmp_path, pkg)
    assert deps.apply_site_patches(dry_run=True) is True
    assert "would apply site patch" in capsys.readouterr().out
    assert (pkg / rel).read_text() == ORIG


def test_converge_applies_then_is_idempotent(tmp_path, monkeypatch, capsys):
    pkg, _patch, rel = _fake_package(tmp_path)
    _install(monkeypatch, tmp_path, pkg)
    assert deps.apply_site_patches() is True
    assert "applied site patch" in capsys.readouterr().out
    assert (pkg / rel).read_text() == PATCHED
    assert deps.apply_site_patches() is True
    assert "already applied" in capsys.readouterr().out
    assert (pkg / rel).read_text() == PATCHED


def test_target_in_neither_state_refuses_without_mutation(tmp_path, monkeypatch, capsys):
    pkg, _patch, rel = _fake_package(tmp_path)
    (pkg / rel).write_text("something else entirely\n")
    _install(monkeypatch, tmp_path, pkg)
    assert deps.apply_site_patches() is False
    assert "failed dry-run" in capsys.readouterr().err
    assert (pkg / rel).read_text() == "something else entirely\n"


def test_dry_run_refusal_exits_nonzero(tmp_path, monkeypatch, capsys):
    # the CLI folds apply_site_patches' return into install's exit code
    pkg, _patch, _rel = _fake_package(tmp_path)
    (pkg / "core/mod.py").write_text("junk\n")
    _install(monkeypatch, tmp_path, pkg)
    assert deps.apply_site_patches(dry_run=True) is False
