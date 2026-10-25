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
