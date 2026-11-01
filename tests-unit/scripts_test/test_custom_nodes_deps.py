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
    entries, problems = deps.parse_pyproject(pyproject)
    assert problems == []
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


def test_parse_manifest_empty_extras_list_means_no_extras():
    pins = deps.parse_manifest('[nodes.a]\nurl = "u"\nref = "main"\nextras = []\n')
    assert pins["a"].extras == ()


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


# --- manifest extras -> pyproject [project.optional-dependencies] -----------

PYPROJECT_WITH_EXTRAS = """\
[project]
name = "node"
version = "0.1"
dependencies = ["base-pkg>=1.0"]

[project.optional-dependencies]
doc_export = [
    "python-docx>=1.0",
    "winonly ; platform_system == 'Windows'",
    "gitdep @ git+https://example.com/g.git",
]
vector = ["chromadb>=1.5"]
"""


@pytest.mark.skipif(not deps.HAS_PACKAGING, reason="packaging not installed")
def test_parse_pyproject_expands_requested_extra(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(PYPROJECT_WITH_EXTRAS)
    entries, problems = deps.parse_pyproject(pyproject, ("doc_export",))
    assert problems == []
    by_name = {e.package: e for e in entries}
    assert by_name["base-pkg"].extra == ""
    assert by_name["python-docx"].extra == "doc_export"
    assert by_name["python-docx"].specifier == ">=1.0"
    # URL requirements are skipped inside extras too
    assert "gitdep" not in by_name
    # platform-gated extras requirements follow the same marker rules
    if sys.platform == "win32":
        assert "winonly" in by_name
    else:
        assert "winonly" not in by_name


@pytest.mark.skipif(not deps.HAS_PACKAGING, reason="packaging not installed")
def test_parse_pyproject_without_extras_stays_base_only(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(PYPROJECT_WITH_EXTRAS)
    entries, problems = deps.parse_pyproject(pyproject)
    assert problems == []
    assert {e.package for e in entries} == {"base-pkg"}


@pytest.mark.skipif(not deps.HAS_PACKAGING, reason="packaging not installed")
def test_parse_pyproject_undefined_extra_is_a_problem(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(PYPROJECT_WITH_EXTRAS)
    entries, problems = deps.parse_pyproject(pyproject, ("doc_export", "nonexistent"))
    assert {e.package: e.extra for e in entries}["python-docx"] == "doc_export"
    assert problems == [
        f"{pyproject.parent.name}: manifest extra 'nonexistent' is not defined "
        "in [project.optional-dependencies]"
    ]


@pytest.mark.skipif(not deps.HAS_PACKAGING, reason="packaging not installed")
def test_parse_pyproject_extra_name_matches_normalized(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(PYPROJECT_WITH_EXTRAS)
    entries, problems = deps.parse_pyproject(pyproject, ("doc-export",))
    assert problems == []
    assert {e.package for e in entries} == {"base-pkg", "python-docx"}


@pytest.mark.skipif(not deps.HAS_PACKAGING, reason="packaging not installed")
def test_parse_pyproject_non_list_extra_table_is_a_problem(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "node"\nversion = "0.1"\n'
        'dependencies = []\n\n[project.optional-dependencies]\n'
        'broken = "not-a-list"\n'
    )
    entries, problems = deps.parse_pyproject(pyproject, ("broken",))
    assert entries == []
    assert problems == [
        f"{pyproject.parent.name}: manifest extra 'broken' is not defined "
        "in [project.optional-dependencies]"
    ]


@pytest.mark.skipif(not deps.HAS_PACKAGING, reason="packaging not installed")
def test_deduplicate_records_extra_provenance():
    entries = [
        deps.DepEntry(package="pkg", specifier="", source="pyproject.toml", node="a", raw="pkg"),
        deps.DepEntry(package="pkg", specifier=">=2", source="pyproject.toml", node="a", raw="pkg>=2", extra="doc"),
    ]
    merged = deps.deduplicate(entries)
    assert merged["pkg"].declared_by == ["a", "a[doc]"]


def test_parse_manifest_reads_and_validates_extras():
    pins = deps.parse_manifest('[nodes.a]\nurl = "u"\nref = "main"\nextras = ["doc_export"]\n')
    assert pins["a"].extras == ("doc_export",)
    with pytest.raises(ValueError, match="non-empty strings"):
        deps.parse_manifest('[nodes.a]\nurl = "u"\nref = "main"\nextras = "doc"\n')
    with pytest.raises(ValueError, match="non-empty strings"):
        deps.parse_manifest('[nodes.a]\nurl = "u"\nref = "main"\nextras = ["doc", ""]\n')
    with pytest.raises(ValueError, match="duplicate extra"):
        deps.parse_manifest('[nodes.a]\nurl = "u"\nref = "main"\nextras = ["doc_export", "doc-export"]\n')


@pytest.mark.skipif(not deps.HAS_PACKAGING, reason="packaging not installed")
def test_discover_all_deps_wires_manifest_extras(tmp_path, monkeypatch):
    nodes = tmp_path / "nodes"
    node_py = nodes / "with-pyproject"
    node_req = nodes / "req-only"
    node_py.mkdir(parents=True)
    node_req.mkdir(parents=True)
    (node_py / "pyproject.toml").write_text(PYPROJECT_WITH_EXTRAS)
    (node_req / "requirements.txt").write_text("other-pkg\n")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        '[nodes.with-pyproject]\nurl = "u"\nref = "main"\nextras = ["doc_export", "nope"]\n'
        '\n[nodes.req-only]\nurl = "u"\nref = "main"\nextras = ["doc_export"]\n'
    )
    monkeypatch.setattr(deps, "CUSTOM_NODES_DIR", nodes)
    monkeypatch.setattr(deps, "MANIFEST_PATH", manifest)
    entries, problems = deps.discover_all_deps()
    by = {(e.node, e.package): e.extra for e in entries}
    assert by[("with-pyproject", "python-docx")] == "doc_export"
    assert by[("req-only", "other-pkg")] == ""
    assert any("nope" in p for p in problems)
    assert any("req-only" in p and "no pyproject.toml" in p for p in problems)


def test_deps_fingerprint_covers_manifest_extras(tmp_path, monkeypatch):
    nodes = tmp_path / "nodes"
    (nodes / "a").mkdir(parents=True)
    (nodes / "a" / "requirements.txt").write_text("pkg\n")
    manifest = tmp_path / "manifest.toml"
    monkeypatch.setattr(deps, "CUSTOM_NODES_DIR", nodes)
    monkeypatch.setattr(deps, "MANIFEST_PATH", manifest)
    bare = deps.deps_fingerprint()  # manifest file absent
    manifest.write_text('[nodes.a]\nurl = "u"\nref = "main"\n')
    with_base = deps.deps_fingerprint()
    manifest.write_text('[nodes.a]\nurl = "u"\nref = "main"\nextras = ["doc"]\n')
    with_extra = deps.deps_fingerprint()
    assert bare != with_base != with_extra
