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


def test_sync_node_dry_run_reports_patched_node_at_pin(tmp_path, monkeypatch, capsys):
    # a patch-declared node already at its pin is "at pin" in the dry run: it
    # is neither a planned clone/move nor a refusal, and the real run lands the
    # stack on top of the pinned ref without touching the ref itself.
    # Owns only its own tmp_path: origin.git, the clone and the patch stack;
    # CUSTOM_NODES_DIR/PATCHES_DIR are redirected per test and restored.
    origin, c1 = _make_origin(tmp_path)
    work = tmp_path / "work"
    _run("clone", "-q", str(origin), str(work))
    patches_dir = tmp_path / "patches"
    (patches_dir / "work").mkdir(parents=True)
    (patches_dir / "work" / "0001-a.txt.patch").write_text(
        "diff --git a/a.txt b/a.txt\n"
        "--- a/a.txt\n"
        "+++ b/a.txt\n"
        "@@ -1 +1 @@\n"
        "-one\n"
        "+patched\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(deps, "CUSTOM_NODES_DIR", tmp_path)
    monkeypatch.setattr(deps, "PATCHES_DIR", patches_dir)
    pin = deps.NodePin(
        url=str(origin), ref="main", patches=("0001-a.txt.patch",), patches_base=c1
    )
    assert deps.sync_node("work", pin, dry_run=True) == c1
    assert "would apply 1 patch(es)" in capsys.readouterr().out
    assert deps.sync_node("work", pin) == c1
    assert (work / "a.txt").read_text(encoding="utf-8") == "patched\n"
    assert deps.sync_node("work", pin) == c1  # idempotent: stack already applied


def _patched_clone(tmp_path, monkeypatch, *, a_content="one\n", patch_from="one\n"):
    """origin + clone at main, with a one-hunk patch stack declared for "work".

    The clone sits at the pinned ref, so the sync plan is "ok" and the patch
    stack decides whether the node settles or refuses. CUSTOM_NODES_DIR and
    PATCHES_DIR are redirected here and restored by monkeypatch.
    """
    seed = tmp_path / "seed"
    seed.mkdir()
    _run("init", "-q", "-b", "main", cwd=seed)
    (seed / "a.txt").write_text(a_content, encoding="utf-8")
    _run("add", ".", cwd=seed)
    _run("commit", "-qm", "c1", cwd=seed)
    c1 = _rev(seed)
    origin = tmp_path / "origin.git"
    _run("clone", "--bare", "-q", str(seed), str(origin))
    work = tmp_path / "work"
    _run("clone", "-q", str(origin), str(work))
    patches_dir = tmp_path / "patches"
    (patches_dir / "work").mkdir(parents=True)
    (patches_dir / "work" / "0001-a.txt.patch").write_text(
        "diff --git a/a.txt b/a.txt\n"
        "--- a/a.txt\n"
        "+++ b/a.txt\n"
        "@@ -1 +1 @@\n"
        f"-{patch_from}"
        "+patched\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(deps, "CUSTOM_NODES_DIR", tmp_path)
    monkeypatch.setattr(deps, "PATCHES_DIR", patches_dir)
    pin = deps.NodePin(
        url=str(origin), ref="main", patches=("0001-a.txt.patch",), patches_base=c1
    )
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        f'[nodes.work]\nurl = "{origin}"\nref = "main"\n'
        'patches = ["0001-a.txt.patch"]\n',
        encoding="utf-8",
    )
    return SimpleNamespace(origin=origin, work=work, c1=c1, pin=pin, manifest=manifest)


def test_sync_node_dry_run_refuses_patch_stack_that_cannot_apply(
    tmp_path, monkeypatch, capsys
):
    # the pinned ref moved so the stack no longer applies: the real sync
    # REFUSES this node, so the dry run must refuse it too — naming the node
    # and the failing patch — and must not count it as "at pin".
    node = _patched_clone(tmp_path, monkeypatch, a_content="two\n")
    assert deps.sync_node("work", node.pin, dry_run=True) is None
    err = capsys.readouterr().err
    assert "work: REFUSED" in err
    assert "0001-a.txt.patch" in err
    # the real run refuses the same node, with the same exit code the dry
    # run uses (run_sync_manifest returns 1 on any refusal)
    assert deps.run_sync_manifest(node.manifest, dry_run=True) == 1
    out = capsys.readouterr().out
    assert "0 at pin" in out
    assert "1 refused" in out
    assert deps.run_sync_manifest(node.manifest) == 1
    # nothing was written: the clone still sits at the pinned ref, untouched
    assert _rev(node.work) == node.c1
    assert (node.work / "a.txt").read_text(encoding="utf-8") == "two\n"


def test_sync_node_dry_run_refuses_unexplained_worktree_dirt(
    tmp_path, monkeypatch, capsys
):
    # the worktree holds changes the declared stack does not explain: the
    # real sync REFUSES this node, so the dry run must refuse it too and
    # must not count it as "at pin".
    node = _patched_clone(tmp_path, monkeypatch)
    (node.work / "a.txt").write_text("dirty\n", encoding="utf-8")
    assert deps.sync_node("work", node.pin, dry_run=True) is None
    err = capsys.readouterr().err
    assert "work: REFUSED" in err
    assert "not explained by the declared patches" in err
    assert deps.run_sync_manifest(node.manifest, dry_run=True) == 1
    out = capsys.readouterr().out
    assert "0 at pin" in out
    assert "1 refused" in out
    assert deps.run_sync_manifest(node.manifest) == 1
    # the dry run wrote nothing: the dirt is still exactly as it was
    assert (node.work / "a.txt").read_text(encoding="utf-8") == "dirty\n"


def _creating_patched_clone(tmp_path, monkeypatch):
    """origin + clone at main, with a stack that CREATES b.txt.

    The clone sits at the pinned ref, so the sync plan is "ok" and the patch
    stack decides whether the node settles or refuses. CUSTOM_NODES_DIR and
    PATCHES_DIR are redirected here and restored by monkeypatch.
    """
    seed = tmp_path / "seed"
    seed.mkdir()
    _run("init", "-q", "-b", "main", cwd=seed)
    (seed / "a.txt").write_text("one\n", encoding="utf-8")
    _run("add", ".", cwd=seed)
    _run("commit", "-qm", "c1", cwd=seed)
    c1 = _rev(seed)
    origin = tmp_path / "origin.git"
    _run("clone", "--bare", "-q", str(seed), str(origin))
    work = tmp_path / "work"
    _run("clone", "-q", str(origin), str(work))
    patches_dir = tmp_path / "patches"
    (patches_dir / "work").mkdir(parents=True)
    (patches_dir / "work" / "0001-b.txt.patch").write_text(
        "diff --git a/b.txt b/b.txt\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/b.txt\n"
        "@@ -0,0 +1 @@\n"
        "+created\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(deps, "CUSTOM_NODES_DIR", tmp_path)
    monkeypatch.setattr(deps, "PATCHES_DIR", patches_dir)
    pin = deps.NodePin(
        url=str(origin), ref="main", patches=("0001-b.txt.patch",), patches_base=c1
    )
    return SimpleNamespace(origin=origin, work=work, c1=c1, pin=pin)


def test_sync_node_stack_creating_a_file_converges_and_is_idempotent(
    tmp_path, monkeypatch, capsys
):
    # a stack that CREATES a file: the apply path must stage what it created,
    # because an untracked file is invisible to `git diff HEAD` — the next
    # sync would read the worktree as "stack not applied" and apply the
    # stack again, which fails with "already exists in working directory".
    # Owns only its own tmp_path: origin.git, the clone and the patch stack;
    # CUSTOM_NODES_DIR/PATCHES_DIR are redirected per test and restored.
    node = _creating_patched_clone(tmp_path, monkeypatch)
    assert deps.sync_node("work", node.pin) == node.c1
    assert (node.work / "b.txt").read_text(encoding="utf-8") == "created\n"
    # idempotent: the next syncs see the stack as applied and change nothing
    for _ in range(2):
        capsys.readouterr()
        assert deps.sync_node("work", node.pin) == node.c1
        assert "patches already applied (1 patch(es))" in capsys.readouterr().out
    assert _rev(node.work) == node.c1
    # the created file is staged: the clone's index records the stack
    assert "A  b.txt" in subprocess.run(
        ["git", "-C", str(node.work), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _one_node_manifest(tmp_path, origin, node="work"):
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        f'[nodes.{node}]\nurl = "{origin}"\nref = "main"\n'
        'patches = ["0001-b.txt.patch"]\n',
        encoding="utf-8",
    )
    return manifest


def test_sync_node_dry_run_and_real_refuse_a_created_file_the_clone_ignores(
    tmp_path, monkeypatch, capsys
):
    # a stack that creates a file the clone's .gitignore covers: `git add`
    # refuses it, so the real sync REFUSES the node — the dry run must
    # refuse it too (same exit code), and neither run may touch the clone.
    # Owns only its own tmp_path: origin.git, the clone and the patch stack;
    # CUSTOM_NODES_DIR/PATCHES_DIR are redirected per test and restored.
    node = _creating_patched_clone(tmp_path, monkeypatch)
    (node.work / ".gitignore").write_text("b.txt\n", encoding="utf-8")
    manifest = _one_node_manifest(tmp_path, node.origin)
    assert deps.run_sync_manifest(manifest, dry_run=True) == 1
    err = capsys.readouterr().err
    assert "work: REFUSED" in err
    assert "could not be staged" in err
    assert deps.run_sync_manifest(manifest) == 1
    err = capsys.readouterr().err
    assert "work: REFUSED" in err
    assert "could not be staged" in err
    # the clone sits at the pinned ref, untouched: nothing applied, nothing staged
    assert _rev(node.work) == node.c1
    assert not (node.work / "b.txt").exists()
    assert subprocess.run(
        ["git", "-C", str(node.work), "diff", "--cached", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout == ""


def test_sync_node_dry_run_and_real_refuse_a_created_file_that_already_exists(
    tmp_path, monkeypatch, capsys
):
    # a stack that creates a file the worktree already holds (untracked):
    # `git apply` refuses it, so the real sync REFUSES the node — the dry
    # run must refuse it too, and neither run may touch the clone.
    node = _creating_patched_clone(tmp_path, monkeypatch)
    (node.work / "b.txt").write_text("mine\n", encoding="utf-8")
    manifest = _one_node_manifest(tmp_path, node.origin)
    assert deps.run_sync_manifest(manifest, dry_run=True) == 1
    err = capsys.readouterr().err
    assert "work: REFUSED" in err
    assert "already exist in the working directory" in err
    assert deps.run_sync_manifest(manifest) == 1
    err = capsys.readouterr().err
    assert "work: REFUSED" in err
    assert "already exist in the working directory" in err
    # the untracked file is the user's, not the stack's: untouched, unstaged
    assert (node.work / "b.txt").read_text(encoding="utf-8") == "mine\n"
    assert _rev(node.work) == node.c1


def test_sync_node_dry_run_and_real_refuse_a_stack_that_does_not_compile(
    tmp_path, monkeypatch, capsys
):
    # a stack that lands a .py file Python cannot parse: the real sync
    # REFUSES it (the compile check), so the dry run must refuse it too.
    seed = tmp_path / "seed"
    seed.mkdir()
    _run("init", "-q", "-b", "main", cwd=seed)
    (seed / "a.py").write_text("ok = 1\n", encoding="utf-8")
    _run("add", ".", cwd=seed)
    _run("commit", "-qm", "c1", cwd=seed)
    c1 = _rev(seed)
    origin = tmp_path / "origin.git"
    _run("clone", "--bare", "-q", str(seed), str(origin))
    work = tmp_path / "work"
    _run("clone", "-q", str(origin), str(work))
    patches_dir = tmp_path / "patches"
    (patches_dir / "work").mkdir(parents=True)
    (patches_dir / "work" / "0001-a.py.patch").write_text(
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1 +1 @@\n"
        "-ok = 1\n"
        "+def broken(:\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(deps, "CUSTOM_NODES_DIR", tmp_path)
    monkeypatch.setattr(deps, "PATCHES_DIR", patches_dir)
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        f'[nodes.work]\nurl = "{origin}"\nref = "main"\n'
        'patches = ["0001-a.py.patch"]\n',
        encoding="utf-8",
    )
    assert deps.run_sync_manifest(manifest, dry_run=True) == 1
    err = capsys.readouterr().err
    assert "work: REFUSED" in err
    assert "failed to compile" in err
    assert deps.run_sync_manifest(manifest) == 1
    err = capsys.readouterr().err
    assert "work: REFUSED" in err
    assert "failed to compile" in err
    # the clone sits at the pinned ref, untouched
    assert _rev(work) == c1
    assert (work / "a.py").read_text(encoding="utf-8") == "ok = 1\n"


def test_sync_node_dry_run_refusal_reports_no_drift_warning(
    tmp_path, monkeypatch, capsys
):
    # a refusing node reports the refusal ALONE: the real run only warns
    # about a drifted patches_base on a successful apply, so a dry run that
    # warns first reports a state the real run never reaches.
    node = _patched_clone(tmp_path, monkeypatch)
    (node.work / "a.txt").write_text("dirty\n", encoding="utf-8")
    drifted = deps.NodePin(
        url=node.pin.url,
        ref=node.pin.ref,
        patches=node.pin.patches,
        patches_base="deadbeef",
    )
    assert deps.sync_node("work", drifted, dry_run=True) is None
    err = capsys.readouterr().err
    assert "REFUSED" in err
    assert "WARNING" not in err
    assert "patches_base" not in err
    # the real run refuses it too, and warns about nothing
    assert deps.sync_node("work", drifted) is None
    err = capsys.readouterr().err
    assert "REFUSED" in err
    assert "WARNING" not in err


def test_sync_node_stages_only_what_the_stack_created(tmp_path, monkeypatch, capsys):
    # a stack that creates a file AND edits a tracked one: only the created
    # file is staged (the tracked edit stays unstaged, as before), and a
    # created file in a subdirectory stages by its repo-relative path.
    # Owns only its own tmp_path: origin.git, the clone and the patch stack;
    # CUSTOM_NODES_DIR/PATCHES_DIR are redirected per test and restored.
    seed = tmp_path / "seed"
    seed.mkdir()
    _run("init", "-q", "-b", "main", cwd=seed)
    (seed / "a.txt").write_text("one\n", encoding="utf-8")
    _run("add", ".", cwd=seed)
    _run("commit", "-qm", "c1", cwd=seed)
    c1 = _rev(seed)
    origin = tmp_path / "origin.git"
    _run("clone", "--bare", "-q", str(seed), str(origin))
    work = tmp_path / "work"
    _run("clone", "-q", str(origin), str(work))
    patches_dir = tmp_path / "patches"
    (patches_dir / "work").mkdir(parents=True)
    (patches_dir / "work" / "0001-mixed.patch").write_text(
        "diff --git a/a.txt b/a.txt\n"
        "--- a/a.txt\n"
        "+++ b/a.txt\n"
        "@@ -1 +1 @@\n"
        "-one\n"
        "+ONE\n"
        "diff --git a/sub/dir/new.py b/sub/dir/new.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/sub/dir/new.py\n"
        "@@ -0,0 +1 @@\n"
        "+VALUE = 1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(deps, "CUSTOM_NODES_DIR", tmp_path)
    monkeypatch.setattr(deps, "PATCHES_DIR", patches_dir)
    pin = deps.NodePin(
        url=str(origin), ref="main", patches=("0001-mixed.patch",), patches_base=c1
    )
    assert deps.sync_node("work", pin) == c1
    porcelain = subprocess.run(
        ["git", "-C", str(work), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "A  sub/dir/new.py" in porcelain  # the created file is staged
    assert " M a.txt" in porcelain  # the tracked edit is NOT staged
    assert "A  a.txt" not in porcelain
    # and the stack still reads as applied on the next sync
    capsys.readouterr()
    assert deps.sync_node("work", pin) == c1
    assert "patches already applied (1 patch(es))" in capsys.readouterr().out


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
