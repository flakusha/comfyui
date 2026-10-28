import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import custom_nodes_deps as deps


@pytest.mark.skipif(not deps.HAS_PACKAGING, reason="packaging not installed")
def test_marker_gated_requirement_is_skipped_on_other_platform(tmp_path):
    req_txt = tmp_path / "requirements.txt"
    req_txt.write_text('amd-adlx; platform_system == "Windows"\n')
    entries = deps.parse_requirements_txt(req_txt)
    if sys.platform == "win32":
        assert [e.package for e in entries] == ["amd-adlx"]
    else:
        assert entries == []


@pytest.mark.skipif(not deps.HAS_PACKAGING, reason="packaging not installed")
def test_pyproject_marker_gated_dependency_is_skipped(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "node"\nversion = "0.1"\n'
        'dependencies = ["amd-adlx ; platform_system == \'Windows\'"]\n'
    )
    entries = deps.parse_pyproject(pyproject)
    if sys.platform == "win32":
        assert [e.package for e in entries] == ["amd-adlx"]
    else:
        assert entries == []


@pytest.mark.skipif(not deps.HAS_PACKAGING, reason="packaging not installed")
def test_unmarked_requirement_is_kept(tmp_path):
    req_txt = tmp_path / "requirements.txt"
    req_txt.write_text("numpy>=1.26.4\n")
    entries = deps.parse_requirements_txt(req_txt)
    assert [e.package for e in entries] == ["numpy"]
    assert entries[0].specifier == ">=1.26.4"


# --- manifest-managed checkouts (sync subcommand) ---------------------------

MANIFEST_SAMPLE = """\
[nodes.alpha]
url = "https://example.com/alpha.git"
ref = "main"

[nodes.beta]
url = "https://example.com/beta"
ref = "HEAD~2"

[nodes.gamma]
url = "https://example.com/gamma"
ref = "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b"

[nodes.delta]
url = "https://example.com/delta"
ref = "latest"
"""


def test_parse_manifest_reads_pins():
    pins = deps.parse_manifest(MANIFEST_SAMPLE)
    assert set(pins) == {"alpha", "beta", "gamma", "delta"}
    assert pins["alpha"] == deps.NodePin(url="https://example.com/alpha.git", ref="main")
    assert pins["delta"].ref == "latest"


def test_parse_manifest_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown key"):
        deps.parse_manifest('[nodes.a]\nurl = "u"\nref = "main"\nrfef = "x"\n')


def test_parse_manifest_rejects_missing_fields_and_bad_names():
    with pytest.raises(ValueError, match="ref"):
        deps.parse_manifest('[nodes.a]\nurl = "u"\n')
    with pytest.raises(ValueError, match="url"):
        deps.parse_manifest('[nodes.a]\nref = "main"\n')
    with pytest.raises(ValueError, match="directory name"):
        deps.parse_manifest('[nodes."sub/dir"]\nurl = "u"\nref = "main"\n')
    with pytest.raises(ValueError, match="directory name"):
        deps.parse_manifest('[nodes.".."]\nurl = "u"\nref = "main"\n')
    with pytest.raises(ValueError, match=r"non-empty \[nodes\]"):
        deps.parse_manifest("[other]\nx = 1\n")


def test_classify_ref_kinds():
    assert deps.classify_ref("latest") == ("latest", "")
    assert deps.classify_ref("HEAD~0") == ("head_offset", "0")
    assert deps.classify_ref("HEAD~12") == ("head_offset", "12")
    assert deps.classify_ref("a" * 40) == ("hash", "a" * 40)
    assert deps.classify_ref("deadbeef") == ("hash", "deadbeef")
    # 5 hex chars is below the hash floor, so it stays a name (tag/branch)
    assert deps.classify_ref("caffe") == ("branch", "caffe")
    assert deps.classify_ref("v1.2.3") == ("branch", "v1.2.3")
    with pytest.raises(ValueError):
        deps.classify_ref("HEAD~x")
    with pytest.raises(ValueError):
        deps.classify_ref("")
    with pytest.raises(ValueError):
        deps.classify_ref("-b")
    with pytest.raises(ValueError):
        deps.classify_ref("two words")


def test_parse_symref_head():
    assert deps.parse_symref_head("ref: refs/heads/master\tHEAD\ndeadbeef\tHEAD\n") == "master"
    assert deps.parse_symref_head("deadbeef\tHEAD\n") is None
    assert deps.parse_symref_head("ref: refs/pull/9/head\tHEAD\n") is None
    assert deps.parse_symref_head("") is None


def st(**kw):
    """A settled-on-old-commit RepoState, overridden per test."""
    base = dict(exists=True, is_git=True, dirty=False, head="h1", branch=None, can_ff=False)
    return deps.RepoState(**{**base, **kw})


def test_plan_clone_and_non_git_refusal():
    assert deps.plan_sync(deps.RepoState(exists=False), "t", None).verb == "clone"
    plan = deps.plan_sync(deps.RepoState(exists=True, is_git=False), "t", None)
    assert plan.verb == "refuse"
    assert "not a git clone" in plan.detail


def test_plan_ok_when_already_at_target_hash():
    # idempotence: at-target is left alone even if dirty, detached, or elsewhere
    assert deps.plan_sync(st(head="t"), "t", None).verb == "ok"
    assert deps.plan_sync(st(head="t", dirty=True), "t", None).verb == "ok"
    assert deps.plan_sync(st(head="t", branch="main"), "t", "main").verb == "ok"
    assert deps.plan_sync(st(head="t", branch="other"), "t", None).verb == "ok"


def test_plan_detach_moves_clean_worktree_only():
    assert deps.plan_sync(st(), "t", None).verb == "detach"
    plan = deps.plan_sync(st(dirty=True), "t", None)
    assert plan.verb == "refuse"
    assert "dirty worktree" in plan.detail


def test_plan_branch_pin_ff_diverged_and_dirty():
    assert deps.plan_sync(st(branch="main", can_ff=True), "t", "main").verb == "ff"
    plan = deps.plan_sync(st(branch="main", can_ff=False), "t", "main")
    assert plan.verb == "refuse"
    assert "diverged" in plan.detail
    plan = deps.plan_sync(st(branch="main", can_ff=True, dirty=True), "t", "main")
    assert plan.verb == "refuse"
    assert "dirty worktree" in plan.detail


def test_plan_branch_pin_switches_from_other_ref():
    assert deps.plan_sync(st(branch="master"), "t", "main").verb == "checkout_branch"
    assert deps.plan_sync(st(branch=None), "t", "main").verb == "checkout_branch"
    plan = deps.plan_sync(st(branch="master", dirty=True), "t", "main")
    assert plan.verb == "refuse"
    assert "dirty worktree" in plan.detail


def test_plan_unborn_head_on_pinned_branch_refuses():
    plan = deps.plan_sync(st(branch="main", head=None), "t", "main")
    assert plan.verb == "refuse"
    assert "unborn" in plan.detail


def test_is_dirty_tracks_only_tracked_file_changes():
    assert deps.is_dirty("") is False
    assert deps.is_dirty("?? untracked.txt\n") is False
    assert deps.is_dirty(" M edited.txt\n") is True
    assert deps.is_dirty("M  staged.txt\n?? untracked.txt\n") is True
    assert deps.is_dirty(" D deleted.txt\n") is True
    assert deps.is_dirty("R  old.txt -> new.txt\n") is True
