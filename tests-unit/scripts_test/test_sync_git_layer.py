"""Tests for custom_nodes_deps.py's git-executing layer (sync subcommand).

Fixture: a real bare origin + clone under tmp_path — deterministic, isolated,
no network (origin is a local path). These tests cover the layer where the
--quiet show-ref existence probe used to silently fail (always-empty stdout).
"""

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import custom_nodes_deps as deps

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")

# -c flags keep commits deterministic; global/system config is cut off so the
# runner's user identity or hooks can't leak in
_G = ["git", "-c", "user.email=test@example", "-c", "user.name=test"]


def _run(*args, cwd=None):
    subprocess.run([*_G, *args], cwd=cwd, check=True, capture_output=True)


def _rev(repo, ref="HEAD"):
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", ref],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _head_branch(repo):
    out = subprocess.run(
        ["git", "-C", str(repo), "symbolic-ref", "--quiet", "HEAD"],
        capture_output=True,
        text=True,
    )
    return None if out.returncode != 0 else out.stdout.strip()


def _make_origin(tmp_path):
    """Bare origin on main with one commit; returns (origin_path, c1)."""
    seed = tmp_path / "seed"
    seed.mkdir()
    _run("init", "-q", "-b", "main", cwd=seed)
    (seed / "a.txt").write_text("one\n", encoding="utf-8")
    _run("add", ".", cwd=seed)
    _run("commit", "-qm", "c1", cwd=seed)
    c1 = _rev(seed)
    origin = tmp_path / "origin.git"
    _run("clone", "--bare", "-q", str(seed), str(origin))
    return origin, c1


@pytest.fixture
def git_repos(tmp_path):
    """Bare origin (main: c1->c2->c3, tag v1 at c1) plus a clone of it."""
    seed = tmp_path / "seed"
    seed.mkdir()
    _run("init", "-q", "-b", "main", cwd=seed)
    (seed / "a.txt").write_text("one\n", encoding="utf-8")
    _run("add", ".", cwd=seed)
    _run("commit", "-qm", "c1", cwd=seed)
    c1 = _rev(seed)
    _run("tag", "v1", cwd=seed)
    (seed / "a.txt").write_text("two\n", encoding="utf-8")
    _run("commit", "-qam", "c2", cwd=seed)
    c2 = _rev(seed)
    (seed / "a.txt").write_text("three\n", encoding="utf-8")
    _run("commit", "-qam", "c3", cwd=seed)
    c3 = _rev(seed)
    origin = tmp_path / "origin.git"
    _run("clone", "--bare", "-q", str(seed), str(origin))
    work = tmp_path / "work"
    _run("clone", "-q", str(origin), str(work))
    return SimpleNamespace(origin=origin, work=work, c1=c1, c2=c2, c3=c3)


def _sync(git_repos, monkeypatch, ref, name="work"):
    monkeypatch.setattr(deps, "CUSTOM_NODES_DIR", git_repos.work.parent)
    pin = deps.NodePin(url=str(git_repos.origin), ref=ref)
    return deps.sync_node(name, pin)


def test_resolve_pin_kinds(git_repos):
    repo = git_repos.work
    assert deps.resolve_pin(repo, "latest") == (git_repos.c3, None)
    assert deps.resolve_pin(repo, "HEAD~1") == (git_repos.c2, None)
    assert deps.resolve_pin(repo, "v1") == (git_repos.c1, None)
    assert deps.resolve_pin(repo, git_repos.c2) == (git_repos.c2, None)
    assert deps.resolve_pin(repo, "main") == (git_repos.c3, "main")


def test_sync_node_branch_pin_with_existing_local_branch(git_repos, monkeypatch):
    # the blocker-1 scenario: clone sits detached at an older commit while the
    # pinned branch already exists locally; show-ref --quiet has EMPTY stdout,
    # so existence must come from the exit code, and `checkout -b` must not run
    w = git_repos.work
    _run("reset", "-q", "--hard", "main~1", cwd=w)  # local main behind origin/main
    _run("checkout", "-q", "--detach", git_repos.c2, cwd=w)
    assert _rev(w) == git_repos.c2
    assert _head_branch(w) is None
    assert _sync(git_repos, monkeypatch, "main") == git_repos.c3
    assert _head_branch(w) == "refs/heads/main"
    assert _rev(w) == git_repos.c3


def test_sync_node_branch_pin_diverged_refuses_before_moving(git_repos, monkeypatch):
    # local pinned branch carries a commit the pin can't fast-forward to:
    # refusal must happen BEFORE any checkout moves the worktree
    w = git_repos.work
    _run("checkout", "-q", "main", cwd=w)
    _run("commit", "--allow-empty", "-qm", "local work", cwd=w)  # main diverges
    diverged = _rev(w, "main")
    _run("checkout", "-q", "--detach", git_repos.c2, cwd=w)  # detached elsewhere
    assert _sync(git_repos, monkeypatch, "main") is None
    assert _rev(w) == git_repos.c2  # worktree untouched
    assert _rev(w, diverged) == diverged  # local commit still reachable


def test_sync_node_branch_pin_detached_ff(git_repos, monkeypatch):
    # local main behind origin/main, checked out: plain fast-forward, no refusal
    w = git_repos.work
    _run("reset", "-q", "--hard", "main~1", cwd=w)
    assert _sync(git_repos, monkeypatch, "main") == git_repos.c3
    assert _head_branch(w) == "refs/heads/main"
    assert _rev(w) == git_repos.c3


def test_sync_node_latest_detaches_at_default_tip(git_repos, monkeypatch):
    w = git_repos.work
    _run("checkout", "-q", "--detach", "main~2", cwd=w)
    assert _sync(git_repos, monkeypatch, "latest") == git_repos.c3
    assert _head_branch(w) is None


def test_sync_node_head_offset_and_tag_detaches(git_repos, monkeypatch):
    assert _sync(git_repos, monkeypatch, "HEAD~1") == git_repos.c2
    assert _head_branch(git_repos.work) is None
    assert _sync(git_repos, monkeypatch, "v1") == git_repos.c1


def test_run_sync_manifest_dry_run_would_clone_exits_zero(tmp_path, monkeypatch, capsys):
    # a missing repo in dry-run is a plan summary ("would clone"), not a failure
    origin, _c1 = _make_origin(tmp_path)
    monkeypatch.setattr(deps, "CUSTOM_NODES_DIR", tmp_path / "custom_nodes")
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        f'[nodes.n1]\nurl = "{origin}"\nref = "main"\n', encoding="utf-8"
    )
    assert deps.run_sync_manifest(manifest, dry_run=True) == 0
    assert "would clone" in capsys.readouterr().out


def test_run_sync_manifest_refusal_exits_one_and_skips_state(tmp_path, monkeypatch):
    # a real refusal still exits 1, and the last-settled state map is never
    # overwritten by a partial run
    origin, _c1 = _make_origin(tmp_path)
    nodes = tmp_path / "custom_nodes"
    work = nodes / "n1"
    _run("clone", "-q", str(origin), str(work))
    # a Manager-installed-looking copy (no .git): refused in dry and real runs
    stray = nodes / "n2"
    stray.mkdir()
    (stray / "blob.bin").write_bytes(b"")
    monkeypatch.setattr(deps, "CUSTOM_NODES_DIR", nodes)
    monkeypatch.setattr(deps, "REFS_STATE_FILE", tmp_path / "refs.json")
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        f'[nodes.n1]\nurl = "{origin}"\nref = "main"\n\n'
        f'[nodes.n2]\nurl = "{origin}"\nref = "main"\n', encoding="utf-8"
    )
    assert deps.run_sync_manifest(manifest, dry_run=True) == 1
    assert deps.run_sync_manifest(manifest) == 1
    assert not (tmp_path / "refs.json").exists()


def test_run_sync_manifest_clean_run_saves_state(tmp_path, monkeypatch):
    origin, _c1 = _make_origin(tmp_path)
    monkeypatch.setattr(deps, "CUSTOM_NODES_DIR", tmp_path / "custom_nodes")
    monkeypatch.setattr(deps, "REFS_STATE_FILE", tmp_path / "refs.json")
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        f'[nodes.n1]\nurl = "{origin}"\nref = "main"\n', encoding="utf-8"
    )
    assert deps.run_sync_manifest(manifest) == 0
    assert "n1" in (tmp_path / "refs.json").read_text(encoding="utf-8")


def test_parse_manifest_rejects_dash_url_and_blank_name():
    with pytest.raises(ValueError, match="'-'"):
        deps.parse_manifest('[nodes.a]\nurl = "-oStrictHostKeyChecking"\nref = "main"\n')
    with pytest.raises(ValueError, match="directory name"):
        deps.parse_manifest('[nodes."  "]\nurl = "u"\nref = "main"\n')


def test_classify_ref_rejects_bare_head():
    with pytest.raises(ValueError, match="HEAD"):
        deps.classify_ref("HEAD")
