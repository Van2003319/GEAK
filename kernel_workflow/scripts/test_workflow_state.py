#!/usr/bin/env python3
"""GPU-free tests for versioned continuation state and hash invalidation."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("workflow_state.py")
SPEC = importlib.util.spec_from_file_location("workflow_state_under_test", SCRIPT)
assert SPEC and SPEC.loader
WS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WS
SPEC.loader.exec_module(WS)


def payload() -> dict:
    return {
        "cumulative": 1.2,
        "cumulative_frame": "vs the original seed",
        "wave_local_cumulative": 1.05,
        "insights": ["x"],
        "ledger": [],
        "asm_ledger": [],
        "keypoint_citations": [],
        "best_per_case": [],
        "rounds": [],
        "bottleneck_now": "compute",
        "search_phase": "phase2_asm",
        "phase_stall": 3,
        "cycle": 2,
        "phase_log": [{"phase": "phase2_asm", "entered_round": 8, "cycle": 2}],
        "deep_attempts": [{"round": 7, "outcome": "valid_new_structure"}],
        "last_round": 8,
        "process_log": [{"event": "phase_transition", "round": 7}],
    }


def canonical(root: Path) -> Path:
    root.mkdir()
    (root / "kernel.py").write_text("print('source')\n", encoding="utf-8")
    (root / "kernel.s").write_text("s_endpgm\n", encoding="utf-8")
    (root / "kernel.co").write_bytes(b"code-object")
    (root / "build").mkdir()
    (root / "build" / "generated.so").write_bytes(b"ephemeral")
    return root


def test_round_trip_persists_all_control_fields_and_hashes(tmp_path: Path) -> None:
    tree = canonical(tmp_path / "canonical")
    state_dir = tmp_path / "state"
    receipt = WS.persist_state(state_dir, tree, payload(), refresh_best=True)
    assert receipt["ok"] is True
    state = json.loads((state_dir / "STATE.json").read_text(encoding="utf-8"))
    assert state["schema"] == "geak.workflow-state/v2"
    assert state["schema_version"] == 2
    for field in ("search_phase", "phase_stall", "cycle", "phase_log", "deep_attempts",
                  "last_round", "process_log"):
        assert state[field] == payload()[field]
    assert state["canonical_source_hash"] == state["canonical_hashes"]["source"]["sha256"]
    assert state["canonical_asm_hash"] == state["canonical_hashes"]["asm"]["sha256"]
    assert state["canonical_binary_hash"] == state["canonical_hashes"]["binary"]["sha256"]
    assert WS.validate_state(state, state_dir / "best")["status"] == "valid"
    assert not list(state_dir.glob(".*.tmp.*"))


@pytest.mark.parametrize(
    ("relative", "artifact"),
    [("kernel.py", "source"), ("kernel.s", "asm"), ("kernel.co", "binary")],
)
def test_each_canonical_hash_mismatch_explicitly_invalidates(
    tmp_path: Path, relative: str, artifact: str
) -> None:
    tree = canonical(tmp_path / "canonical")
    state_dir = tmp_path / "state"
    WS.persist_state(state_dir, tree, payload(), refresh_best=True)
    state = WS.load_state(state_dir / "STATE.json")
    (state_dir / "best" / relative).write_bytes(b"tampered")
    verdict = WS.validate_state(state, state_dir / "best")
    assert verdict["status"] == "invalidated"
    assert f"canonical_hash_mismatch({artifact})" in verdict["invalidate_reason"]
    assert [row["artifact"] for row in verdict["mismatches"]] == [artifact]


def test_legacy_or_partial_state_is_not_reinterpreted(tmp_path: Path) -> None:
    tree = canonical(tmp_path / "canonical")
    legacy = payload()
    verdict = WS.validate_state(legacy, tree)
    assert verdict["status"] == "invalidated"
    assert "schema_mismatch" in verdict["invalidate_reason"]
    assert "canonical_hashes_missing" in verdict["invalidate_reason"]


def test_required_control_fields_fail_closed(tmp_path: Path) -> None:
    tree = canonical(tmp_path / "canonical")
    broken = payload()
    del broken["phase_log"]
    with pytest.raises(WS.StateError, match="phase_log"):
        WS.persist_state(tmp_path / "state", tree, broken)


def test_generated_build_products_do_not_change_durable_binary_identity(tmp_path: Path) -> None:
    tree = canonical(tmp_path / "canonical")
    before = WS.canonical_hashes(tree)
    (tree / "build" / "generated.so").write_bytes(b"new build on another machine")
    (tree / "build" / "kernel.o").write_bytes(b"object")
    assert WS.canonical_hashes(tree) == before


def test_alias_or_hash_metadata_tampering_invalidates_even_when_tree_bytes_match(
    tmp_path: Path,
) -> None:
    tree = canonical(tmp_path / "canonical")
    state_dir = tmp_path / "state"
    WS.persist_state(state_dir, tree, payload(), refresh_best=True)
    state = WS.load_state(state_dir / "STATE.json")
    state["canonical_source_hash"] = "0" * 64
    verdict = WS.validate_state(state, state_dir / "best")
    assert verdict["status"] == "invalidated"
    assert "canonical_hash_alias_mismatch(source)" in verdict["invalidate_reason"]

    state = WS.load_state(state_dir / "STATE.json")
    state["canonical_hashes"]["asm"]["algorithm"] = "md5"
    verdict = WS.validate_state(state, state_dir / "best")
    assert verdict["status"] == "invalidated"
    assert "canonical_hash_record_invalid(asm)" in verdict["invalidate_reason"]


def test_validate_cli_writes_an_atomic_receipt_for_unreadable_state(tmp_path: Path) -> None:
    tree = canonical(tmp_path / "canonical")
    state_file = tmp_path / "STATE.json"
    state_file.write_text("{broken", encoding="utf-8")
    receipt = tmp_path / "validation.json"
    rc = WS.main([
        "validate", "--state-file", str(state_file), "--canonical", str(tree),
        "--receipt", str(receipt),
    ])
    assert rc == 2
    verdict = json.loads(receipt.read_text(encoding="utf-8"))
    assert verdict["status"] == "invalidated"
    assert verdict["valid"] is False
    assert "cannot read state" in verdict["invalidate_reason"]
    assert not list(tmp_path.glob(".validation.json.tmp.*"))


def test_source_symlink_hash_binds_external_target_bytes(tmp_path: Path) -> None:
    external = tmp_path / "external.py"
    external.write_text("x = 1\n", encoding="utf-8")
    tree = tmp_path / "canonical"
    tree.mkdir()
    (tree / "kernel.py").symlink_to(external)
    state_dir = tmp_path / "state"
    WS.persist_state(state_dir, tree, payload(), refresh_best=True)
    state = WS.load_state(state_dir / "STATE.json")
    assert WS.validate_state(state, state_dir / "best")["status"] == "valid"
    external.write_text("x = 2\n", encoding="utf-8")
    verdict = WS.validate_state(state, state_dir / "best")
    assert verdict["status"] == "invalidated"
    assert "canonical_hash_mismatch(source)" in verdict["invalidate_reason"]


def test_phase_log_tail_must_match_persisted_control_cursor(tmp_path: Path) -> None:
    tree = canonical(tmp_path / "canonical")
    broken = payload()
    broken["phase_log"][-1]["phase"] = "phase1_source"
    with pytest.raises(WS.StateError, match="phase_log tail"):
        WS.persist_state(tmp_path / "state", tree, broken)
