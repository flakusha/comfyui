"""Unit tests for scripts/update_local.py's persisted-state helpers.

Resource contract (parallel-safe): every test owns only its runner-allocated
``tmp_path`` — no fixed paths, ports, databases, or shared mutable globals.
Module import is read-only (constants and function definitions), and the
repo's real .local-state/ is never touched: save_state/load_state are always
aimed at tmp_path, so the suite passes under any runner parallelism or order.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import update_local as ul


def test_inputs_fingerprint_is_content_and_order_sensitive():
    base = ["lock-a", "req", "py", "rocm,dev"]
    assert ul.inputs_fingerprint(base) == ul.inputs_fingerprint(list(base))
    changed_lock = ul.inputs_fingerprint(["lock-b", "req", "py", "rocm,dev"])
    assert changed_lock != ul.inputs_fingerprint(base)
    # extras order is part of the uv sync command line, so it must matter
    assert ul.inputs_fingerprint(["lock-a", "req", "py", "dev,rocm"]) != ul.inputs_fingerprint(base)


def test_load_state_missing_returns_none(tmp_path):
    assert ul.load_state(tmp_path / "absent.json") is None


def test_load_state_corrupt_returns_none(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    assert ul.load_state(path) is None


def test_load_state_non_object_returns_none(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("[1, 2]", encoding="utf-8")
    assert ul.load_state(path) is None


def test_load_state_wrong_schema_returns_none(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"schema": 99, "inputs_sha256": "x"}), encoding="utf-8")
    assert ul.load_state(path) is None


def test_load_state_bool_schema_returns_none(tmp_path):
    # True == 1 under ==; the schema check must not admit booleans
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"schema": True, "inputs_sha256": "x"}), encoding="utf-8")
    assert ul.load_state(path) is None


def test_state_roundtrip_and_currentity(tmp_path):
    path = tmp_path / "state.json"
    fingerprint = ul.inputs_fingerprint(["lock-a", "req", "py", "rocm,dev"])
    ul.save_state(
        path,
        {
            "schema": ul.STATE_SCHEMA,
            "inputs_sha256": fingerprint,
            "extras": ["rocm", "dev"],
        },
    )
    state = ul.load_state(path)
    assert state is not None
    assert ul.state_is_current(state, fingerprint, ("rocm", "dev"))
    assert not ul.state_is_current(state, "other-fingerprint", ("rocm", "dev"))
    assert not ul.state_is_current(state, fingerprint, ("dev", "rocm"))
    assert not ul.state_is_current(state, fingerprint, ("rocm", "dev", "sage"))
    assert not ul.state_is_current(None, fingerprint, ("rocm", "dev"))


def test_parse_sync_deltas_clean_run_has_no_deltas():
    # the exact shape a no-op dry-run prints; both maps empty is the skip condition
    output = "Using CPython 3.14.0\nResolved 223 packages in 4ms\nChecked 113 packages in 1ms\nWould make no changes\n"
    assert ul.parse_sync_deltas(output) == ({}, {})


def test_parse_sync_deltas_upgrade_and_removal_are_deltas():
    output = " - driftpkg==2.1.0\n + driftpkg==2.0.0\n - gonerpkg==1.0.0\n + newpkg==3.2.1\n"
    would_remove, would_install = ul.parse_sync_deltas(output)
    assert would_remove == {"driftpkg": "2.1.0", "gonerpkg": "1.0.0"}
    assert would_install == {"driftpkg": "2.0.0", "newpkg": "3.2.1"}


def test_sync_required_matrix():
    # clean dry-run is the only skip case; a failed dry-run proves nothing
    assert not ul.sync_required(False, 0, {}, {})
    assert ul.sync_required(True, 0, {}, {})
    assert ul.sync_required(False, 1, {}, {})
    assert ul.sync_required(False, 0, {}, {"newpkg": "3.2.1"})
    assert ul.sync_required(False, 0, {"gonerpkg": "1.0.0"}, {})
