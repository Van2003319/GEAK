#!/usr/bin/env python3
"""GPU-free tests for atomic, registered-only interrupted-round recovery."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("round_journal.py")
sys.path.insert(0, str(SCRIPT.parent))
from test_paired_verify import task_receipt as paired_task_receipt  # noqa: E402

SPEC = importlib.util.spec_from_file_location("round_journal_under_test", SCRIPT)
assert SPEC and SPEC.loader
RJ = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RJ
SPEC.loader.exec_module(RJ)


def make_candidate(tmp_path: Path, *, artifact: bool = True):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / "kernel.py").write_text("x = 1\n", encoding="utf-8")
    old_eval = tmp_path / "wave1"
    out = old_eval / "round_4" / "engineer_0"
    out.mkdir(parents=True)
    patch = out / "best_patch.diff"
    if artifact:
        patch.write_text("diff --git a/kernel.py b/kernel.py\n", encoding="utf-8")
    receipt = out / "worker_result.json"
    receipt.write_text(json.dumps({
        "engineer_id": "r4_d0",
        "specialty": "deep_explore",
        "status": "success" if artifact else "no_improvement",
        "patch_file": str(patch) if artifact else "",
        "structure_fingerprint": "persistent-grid-v2",
        "config": {"waves": 2},
    }), encoding="utf-8")
    return canonical, old_eval, patch, receipt


def register(tmp_path: Path, *, artifact: bool = True):
    canonical, old_eval, patch, receipt = make_candidate(tmp_path, artifact=artifact)
    journal = tmp_path / "state" / "round_journal"
    result = RJ.register(
        journal,
        eval_dir=str(old_eval),
        candidate_id="r4:r4_d0",
        round_number=4,
        direction="r4_d0",
        phase="deep_explore",
        artifact=patch,
        worker_receipt=receipt,
        canonical=canonical,
        deadline_epoch=123,
        target_routes=["route-a", "route-b"],
    )
    assert result["registered"] is True
    return journal, canonical, old_eval, patch, receipt, result


def test_engineer_completion_is_atomically_registered_with_all_receipts(tmp_path: Path) -> None:
    journal, _, _, _, _, result = register(tmp_path)
    entry = result["entry"]
    assert entry["schema"] == "geak.round-journal/v1"
    assert len(entry["artifact"]["sha256"]) == 64
    assert len(entry["worker_receipt"]["sha256"]) == 64
    assert entry["target_routes"] == ["route-a", "route-b"]
    assert entry["integration"] == {"status": "pending", "receipt": None}
    assert set(entry["parent_canonical_hashes"]) == {"source", "asm", "binary"}
    assert not list((journal / "entries").glob(".*.tmp.*"))


def test_next_wave_discovers_only_registered_interrupted_artifacts(tmp_path: Path) -> None:
    journal, canonical, _, patch, _, _ = register(tmp_path)
    # A tempting patch beside the registered one has no receipt and must never be harvested.
    rogue = patch.parent.parent / "engineer_rogue" / "best_patch.diff"
    rogue.parent.mkdir()
    rogue.write_text("malicious or stale", encoding="utf-8")
    found = RJ.discover(journal, canonical=canonical, current_eval_dir=str(tmp_path / "wave2"))
    assert found["policy"].startswith("registered-only")
    assert [entry["candidate_id"] for entry in found["recoverable"]] == ["r4:r4_d0"]
    assert found["recoverable"][0]["worker_result"]["structure_fingerprint"] == "persistent-grid-v2"
    assert found["recoverable"][0]["target_routes"] == ["route-a", "route-b"]
    assert str(rogue) not in json.dumps(found)


def test_tampered_interrupted_artifact_is_explicitly_invalidated(tmp_path: Path) -> None:
    journal, canonical, _, patch, _, _ = register(tmp_path)
    patch.write_text("changed after registration", encoding="utf-8")
    found = RJ.discover(journal, canonical=canonical, current_eval_dir=str(tmp_path / "wave2"))
    assert found["recoverable"] == []
    assert "artifact_hash_mismatch" in found["invalidated"][0]["reason"]
    entry = json.loads(next((journal / "entries").glob("*.json")).read_text(encoding="utf-8"))
    assert entry["status"] == "invalidated"


def test_parent_hash_change_requires_revalidation_but_does_not_execute_or_discard(tmp_path: Path) -> None:
    journal, canonical, _, _, _, _ = register(tmp_path)
    (canonical / "kernel.py").write_text("x = 2\n", encoding="utf-8")
    found = RJ.discover(journal, canonical=canonical, current_eval_dir=str(tmp_path / "wave2"))
    assert found["recoverable"][0]["parent_hash_mismatches"] == ["source"]
    assert "independently reverify" in found["recoverable"][0]["recovery_reason"]


def test_verified_winner_records_integration_receipt(tmp_path: Path) -> None:
    journal, _, old_eval, _, _, _ = register(tmp_path)
    verify = old_eval / "round_4" / "engineer_0" / "verify" / "verification_result.json"
    verify.parent.mkdir()
    verify.write_text('{"status":"verified","correctness":"pass"}', encoding="utf-8")
    RJ.transition(journal, eval_dir=str(old_eval), candidate_id="r4:r4_d0", status="verified",
                  reason="verified", verification_receipt=str(verify))
    commit = old_eval / "round_4" / "commit_result.json"
    commit.write_text('{"committed":true,"head_sha_before":"a","head_sha_after":"b"}',
                      encoding="utf-8")
    RJ.settle_round(journal, eval_dir=str(old_eval), round_number=4,
                    winner_direction="r4_d0", committed=True,
                    integration_receipt=str(commit))
    entry = json.loads(next((journal / "entries").glob("*.json")).read_text(encoding="utf-8"))
    assert entry["status"] == "integrated"
    assert entry["verification"]["receipt"]["sha256"]
    assert entry["integration"]["receipt"]["sha256"]


def test_no_patch_completion_is_recorded_but_not_recovered(tmp_path: Path) -> None:
    journal, canonical, old_eval, _, _, result = register(tmp_path, artifact=False)
    assert result["entry"]["artifact"]["present"] is False
    RJ.settle_round(journal, eval_dir=str(old_eval), round_number=4,
                    winner_direction="", committed=False)
    found = RJ.discover(journal, canonical=canonical, current_eval_dir=str(tmp_path / "wave2"))
    assert found["recoverable"] == []


def test_restart_with_same_eval_dir_still_discovers_registered_artifact(tmp_path: Path) -> None:
    journal, canonical, old_eval, _, _, _ = register(tmp_path)
    found = RJ.discover(journal, canonical=canonical, current_eval_dir=str(old_eval))
    assert [entry["candidate_id"] for entry in found["recoverable"]] == ["r4:r4_d0"]


def test_tampered_verification_receipt_invalidates_verified_interruption(tmp_path: Path) -> None:
    journal, canonical, old_eval, _, _, _ = register(tmp_path)
    verify = old_eval / "round_4" / "engineer_0" / "verify" / "verification_result.json"
    verify.parent.mkdir()
    verify.write_text('{"status":"verified","correctness":"pass"}', encoding="utf-8")
    RJ.transition(journal, eval_dir=str(old_eval), candidate_id="r4:r4_d0", status="verified",
                  reason="verified", verification_receipt=str(verify))
    verify.write_text('{"status":"tampered"}', encoding="utf-8")
    found = RJ.discover(journal, canonical=canonical, current_eval_dir=str(tmp_path / "wave2"))
    assert found["recoverable"] == []
    assert "verification_receipt_hash_mismatch" in found["invalidated"][0]["reason"]


def test_settle_candidate_filter_does_not_discard_untouched_same_round_entry(
    tmp_path: Path,
) -> None:
    journal, canonical, old_eval, patch, receipt, _ = register(tmp_path)
    second = RJ.register(
        journal, eval_dir=str(old_eval), candidate_id="r4:r4_d1", round_number=4,
        direction="r4_d1", phase="phase1_source", artifact=patch,
        worker_receipt=receipt, canonical=canonical,
    )
    assert second["registered"] is True
    for candidate, suffix in (("r4:r4_d0", "0"), ("r4:r4_d1", "1")):
        verify = old_eval / "round_4" / f"verify_{suffix}.json"
        verify.write_text('{"status":"verified","correctness":"pass"}', encoding="utf-8")
        RJ.transition(journal, eval_dir=str(old_eval), candidate_id=candidate, status="verified",
                      reason="verified", verification_receipt=str(verify))
    commit = old_eval / "round_4" / "commit_result.json"
    commit.write_text(
        '{"committed":true,"head_sha_before":"a","head_sha_after":"b"}',
        encoding="utf-8",
    )
    RJ.settle_round(
        journal, eval_dir=str(old_eval), round_number=4, winner_direction="r4_d0",
        committed=True, integration_receipt=str(commit), candidate_ids=["r4:r4_d0"],
    )
    entries = {
        json.loads(path.read_text(encoding="utf-8"))["candidate_id"]:
        json.loads(path.read_text(encoding="utf-8"))["status"]
        for path in (journal / "entries").glob("*.json")
    }
    assert entries == {"r4:r4_d0": "integrated", "r4:r4_d1": "verified"}


def test_interrupted_no_patch_receipt_is_settled_without_becoming_recoverable(
    tmp_path: Path,
) -> None:
    journal, canonical, old_eval, _, _, _ = register(tmp_path, artifact=False)
    found = RJ.discover(journal, canonical=canonical, current_eval_dir=str(old_eval))
    assert found["recoverable"] == []
    assert found["invalidated"][0]["reason"] == "no_patch"
    entry = json.loads(next((journal / "entries").glob("*.json")).read_text(encoding="utf-8"))
    assert entry["status"] == "rejected"


def test_commit_receipt_cannot_promote_unverified_engineer_artifact(tmp_path: Path) -> None:
    journal, _, old_eval, _, _, _ = register(tmp_path)
    commit = old_eval / "round_4" / "commit_result.json"
    commit.write_text(
        '{"committed":true,"head_sha_before":"a","head_sha_after":"b"}',
        encoding="utf-8",
    )
    settled = RJ.settle_round(
        journal, eval_dir=str(old_eval), round_number=4, winner_direction="r4_d0",
        committed=True, integration_receipt=str(commit), candidate_ids=["r4:r4_d0"],
    )
    assert settled["settled"] == 0
    entry = json.loads(next((journal / "entries").glob("*.json")).read_text(encoding="utf-8"))
    assert entry["status"] == "engineer_complete"


def test_empty_patch_is_journaled_as_no_patch(tmp_path: Path) -> None:
    canonical, old_eval, patch, receipt = make_candidate(tmp_path)
    patch.write_bytes(b"")
    result = RJ.register(
        tmp_path / "journal", eval_dir=str(old_eval), candidate_id="r4:empty",
        round_number=4, direction="empty", phase="phase1_source", artifact=patch,
        worker_receipt=receipt, canonical=canonical,
    )
    assert result["registered"] is True
    assert result["entry"]["artifact"]["present"] is False


def test_gate_invocation_error_is_retryable_and_prioritized_without_commit(tmp_path: Path) -> None:
    journal, canonical, old_eval, patch, receipt, _ = register(tmp_path)
    verify = old_eval / "round_4" / "verify_0" / "verification_result.json"
    verify.parent.mkdir()
    verify.write_text('{"status":"verified","correctness":"pass"}', encoding="utf-8")
    RJ.transition(
        journal, eval_dir=str(old_eval), candidate_id="r4:r4_d0", status="verified",
        reason="independent verification completed", verification_receipt=str(verify),
    )
    settled = RJ.settle_round(
        journal, eval_dir=str(old_eval), round_number=4, winner_direction="r4_d0",
        committed=False, candidate_ids=["r4:r4_d0"], retryable_infrastructure=True,
    )
    entry = settled["entries"][0]
    assert entry["status"] == "verified"
    assert entry["retryable_infrastructure"] is True
    assert "retryable_infrastructure" in entry["history"][-1]["reason"]

    # A newer ordinary interruption must still wait behind the infrastructure retry.
    ordinary = RJ.register(
        journal, eval_dir=str(old_eval), candidate_id="r5:r5_d0", round_number=5,
        direction="r5_d0", phase="phase1_source", artifact=patch,
        worker_receipt=receipt, canonical=canonical, target_routes=["route-a"],
    )
    assert ordinary["registered"] is True
    found = RJ.discover(journal, canonical=canonical, current_eval_dir=str(tmp_path / "wave2"))
    assert [row["candidate_id"] for row in found["recoverable"]] == [
        "r4:r4_d0", "r5:r5_d0",
    ]
    retry = found["recoverable"][0]
    assert retry["recovery_classification"] == "retryable_infrastructure"
    assert "independently reverify without committing" in retry["recovery_reason"]
    assert (canonical / "kernel.py").read_text(encoding="utf-8") == "x = 1\n"


def test_resume_reopens_latest_legacy_invocation_error_only_after_full_validation(
    tmp_path: Path,
) -> None:
    journal, canonical, old_eval, _, _, _ = register(tmp_path)
    verify_dir = old_eval / "round_4" / "verify_0"
    verify_dir.mkdir()
    verify = verify_dir / "verification_result.json"
    verify.write_text('{"status":"verified","correctness":"pass"}', encoding="utf-8")
    RJ.transition(
        journal, eval_dir=str(old_eval), candidate_id="r4:r4_d0", status="verified",
        reason="independent verification completed", verification_receipt=str(verify),
    )
    # Simulate the pre-fix lane: an invocation error was incorrectly settled as rejection.
    RJ.settle_round(
        journal, eval_dir=str(old_eval), round_number=4, winner_direction="",
        committed=False, candidate_ids=["r4:r4_d0"],
    )
    task = paired_task_receipt(verify_dir)
    (verify_dir / "paired_task_receipt.json").write_text(
        json.dumps(task, sort_keys=True), encoding="utf-8")
    state = {
        "schema": RJ.workflow_state.SCHEMA,
        "last_round": 4,
        "canonical_hashes": RJ.workflow_state.canonical_hashes(canonical),
        "rounds": [{
            "round": 4,
            "winner": {"source": "engineer r4_d0", "geomean": 1.05},
            "gate_receipt": {"decision": "invocation_error"},
            "results": [{"id": "r4_d0", "status": "verified"}],
        }],
    }
    state_file = tmp_path / "STATE.json"
    state_file.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    state_before = state_file.read_bytes()
    canonical_before = (canonical / "kernel.py").read_bytes()

    reopened = RJ.reopen_latest_gate_invocation(
        journal, state_file=state_file, canonical=canonical,
        expected_routes=["target", "guard"],
    )
    assert reopened["reopened"] is True and reopened["idempotent"] is False
    assert reopened["entry"]["status"] == "verified"
    assert reopened["entry"]["retryable_infrastructure"] is True
    again = RJ.reopen_latest_gate_invocation(
        journal, state_file=state_file, canonical=canonical,
        expected_routes=["target", "guard"],
    )
    assert again["reopened"] is True and again["idempotent"] is True
    assert state_file.read_bytes() == state_before
    assert (canonical / "kernel.py").read_bytes() == canonical_before
    found = RJ.discover(journal, canonical=canonical, current_eval_dir=str(tmp_path / "wave2"))
    assert found["recoverable"][0]["candidate_id"] == "r4:r4_d0"
    assert found["recoverable"][0]["recovery_classification"] == "retryable_infrastructure"


def test_resume_does_not_reopen_tampered_or_statistically_refused_candidate(tmp_path: Path) -> None:
    journal, canonical, old_eval, _, _, _ = register(tmp_path)
    verify_dir = old_eval / "round_4" / "verify_0"
    verify_dir.mkdir()
    verify = verify_dir / "verification_result.json"
    verify.write_text('{"status":"verified","correctness":"pass"}', encoding="utf-8")
    RJ.transition(
        journal, eval_dir=str(old_eval), candidate_id="r4:r4_d0", status="verified",
        reason="independent verification completed", verification_receipt=str(verify),
    )
    RJ.settle_round(
        journal, eval_dir=str(old_eval), round_number=4, winner_direction="",
        committed=False, candidate_ids=["r4:r4_d0"],
    )
    task = paired_task_receipt(verify_dir)
    receipt_path = verify_dir / "paired_task_receipt.json"
    receipt_path.write_text(json.dumps(task, sort_keys=True), encoding="utf-8")
    state = {
        "schema": RJ.workflow_state.SCHEMA, "last_round": 4,
        "canonical_hashes": RJ.workflow_state.canonical_hashes(canonical),
        "rounds": [{"round": 4, "winner": {"source": "engineer r4_d0"},
                    "gate_receipt": {"decision": "refuse"},
                    "results": [{"id": "r4_d0", "status": "verified"}]}],
    }
    state_file = tmp_path / "STATE.json"
    state_file.write_text(json.dumps(state), encoding="utf-8")
    refused = RJ.reopen_latest_gate_invocation(
        journal, state_file=state_file, canonical=canonical,
        expected_routes=["target", "guard"],
    )
    assert refused["reopened"] is False

    state["rounds"][0]["gate_receipt"]["decision"] = "invocation_error"
    state_file.write_text(json.dumps(state), encoding="utf-8")
    receipt_path.write_text('{"schema":"tampered"}', encoding="utf-8")
    try:
        RJ.reopen_latest_gate_invocation(
            journal, state_file=state_file, canonical=canonical,
            expected_routes=["target", "guard"],
        )
    except RJ.JournalError as exc:
        assert "paired task receipt is invalid" in str(exc)
    else:
        raise AssertionError("tampered paired receipt was reopened")
    entry = json.loads(next((journal / "entries").glob("*.json")).read_text(encoding="utf-8"))
    assert entry["status"] == "rejected"


def test_resume_reopens_invocation_error_after_state_barrier_before_settlement(
    tmp_path: Path,
) -> None:
    journal, canonical, old_eval, _, _, _ = register(tmp_path)
    verify_dir = old_eval / "round_4" / "verify_0"
    verify_dir.mkdir()
    verify = verify_dir / "verification_result.json"
    verify.write_text('{"status":"verified","correctness":"pass"}', encoding="utf-8")
    RJ.transition(
        journal, eval_dir=str(old_eval), candidate_id="r4:r4_d0", status="verified",
        reason="independent verification completed", verification_receipt=str(verify),
    )
    task = paired_task_receipt(verify_dir)
    (verify_dir / "paired_task_receipt.json").write_text(
        json.dumps(task, sort_keys=True), encoding="utf-8")
    state = {
        "schema": RJ.workflow_state.SCHEMA,
        "last_round": 4,
        "canonical_hashes": RJ.workflow_state.canonical_hashes(canonical),
        "rounds": [{
            "round": 4,
            "winner": {"source": "engineer r4_d0", "geomean": 1.05},
            "gate_receipt": {"decision": "invocation_error"},
            "results": [{"id": "r4_d0", "status": "verified"}],
        }],
    }
    state_file = tmp_path / "STATE.json"
    state_file.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")

    reopened = RJ.reopen_latest_gate_invocation(
        journal, state_file=state_file, canonical=canonical,
        expected_routes=["target", "guard"],
    )
    assert reopened["reopened"] is True and reopened["idempotent"] is False
    assert reopened["entry"]["status"] == "verified"
    assert reopened["entry"]["retryable_infrastructure"] is True
    found = RJ.discover(journal, canonical=canonical, current_eval_dir=str(tmp_path / "wave2"))
    assert found["recoverable"][0]["recovery_classification"] == "retryable_infrastructure"
