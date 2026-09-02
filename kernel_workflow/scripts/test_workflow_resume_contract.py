#!/usr/bin/env python3
"""Static cross-file guards for durable resume/deep/journal/config contracts."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANE = (ROOT / "kernel_lane.js").read_text(encoding="utf-8")
DIRECTOR = (ROOT / "roles" / "director.md").read_text(encoding="utf-8")
TECH_LEAD = (ROOT / "roles" / "tech_lead.md").read_text(encoding="utf-8")
ASM = (ROOT / "roles" / "asm_engineer.md").read_text(encoding="utf-8")


def test_prior_state_schema_and_resume_cover_the_full_control_plane() -> None:
    fields = (
        "search_phase", "phase_stall", "cycle", "phase_log", "deep_attempts", "last_round",
        "canonical_hashes", "asm_ledger", "keypoint_citations", "process_log", "rounds",
    )
    for field in fields:
        assert f"{field}:" in LANE, f"{field} missing from lane schema/state"
        assert f"ps.{field}" in LANE or field in ("canonical_hashes",), f"{field} is persisted but not restored"
        assert f"`{field}`" in DIRECTOR, f"Director can silently drop prior_state.{field}"
    assert "schema: 'geak.workflow-state/v2'" not in LANE  # schema is helper-owned, not hand-built
    assert "WORKFLOW_STATE_HELPER" in LANE
    assert "resume_invalidated" in LANE and "resume_invalidation_reason" in LANE


def test_state_snapshot_occurs_after_every_round_barrier() -> None:
    transition_record = LANE.index("transition_only: true")
    transition_persist = LANE.index("persistWorkflowState(false, 'phase_transition')")
    assert transition_record < transition_persist
    measured_record = LANE.index("// Persist the effective record")
    measured_persist = LANE.index("persistWorkflowState(committedThisRound, 'round_barrier')")
    assert measured_record < measured_persist
    assert "cumulative: cumulativeVsSeed()" in LANE
    assert "canonical_source_hash" in (ROOT / "scripts" / "workflow_state.py").read_text()
    assert "canonical_asm_hash" in (ROOT / "scripts" / "workflow_state.py").read_text()
    assert "canonical_binary_hash" in (ROOT / "scripts" / "workflow_state.py").read_text()


def test_deep_outcomes_are_distinct_and_one_no_gain_cannot_stop() -> None:
    for outcome in (
        "api_failure", "no_patch", "compile_or_correctness_failure", "valid_no_gain",
        "valid_new_structure",
    ):
        assert f"outcome = '{outcome}'" in LANE
    assert "MIN_VALID_NOVEL_DEEP_ATTEMPTS = 2" in LANE
    assert "deepExhaustion(deepAttempts, cycle, MIN_VALID_NOVEL_DEEP_ATTEMPTS)" in LANE
    assert "stopReason = 'deep_explore_failed'" not in LANE
    assert "single no-gain attempt" in LANE


def test_engineer_artifact_is_registered_before_verification_and_recovery_is_registered_only() -> None:
    register = LANE.index("round_journal:register ${d.id}")
    verify = LANE.index("label: `verify ${d.id}")
    assert register < verify
    assert "if (!(journal && journal.registered && journal.entry))" in LANE
    assert "Only atomically receipted bytes may cross the execution boundary" in LANE
    discover = LANE.index("round_journal:discover")
    loop = LANE.index("while (!skipLoop && dispatched < BUDGET")
    assert discover < loop
    assert "they will be independently reverified before new work is planned" in LANE


def test_config_search_is_registered_without_a_phase_lock() -> None:
    assert "phase_independent: true" in LANE
    assert "AVAILABLE_CAPABILITIES: availableCapabilities()" in LANE
    assert "config_search may be" in LANE and "ANY search phase" in LANE
    assert "phase2 may SELECT an existing precompiled config" in TECH_LEAD
    assert "config_search create" in ASM
    assert "coarse phase lock" not in ASM.lower()


def test_direction_entropy_keeps_actual_focus_config_and_mechanism() -> None:
    helper = LANE[LANE.index("const effectiveDirectionRecord ="):]
    helper = helper[:2200]
    assert "worker.focus_files" in helper
    assert "worker.config" in helper
    assert "result.configSelection.config" in helper
    assert "artifact_sha256" in helper
    assert "structure_fingerprint" in helper
    history = LANE[LANE.index("directions: directions.map"):]
    assert "effectiveDirectionRecord(" in history[:300]
    assert "clean.find(" in history[:300]
    entropy = LANE[LANE.index("direction_entropy:"):]
    assert "dimensions: ['specialty', 'focus_files', 'mechanism', 'config']" in entropy[:1200]
    assert "stableValue(d.config || {})" in entropy[:1200]


def test_phase2_config_selection_is_runtime_enforced_not_only_prompted() -> None:
    assert "CONFIG_SELECT_SCHEMA" in LANE
    assert "config_search:select" in LANE
    assert "selected config has no config_search id" in LANE
    assert "configSelection.required && !configSelection.ok" in LANE
    assert "--status rejected" in LANE
    config_helper = (ROOT / "scripts" / "config_search.py").read_text(encoding="utf-8")
    assert "def select_config(" in config_helper
    assert 'selection_contract": "precompiled-and-hash-valid-only"' in config_helper


def test_recovery_has_no_unregistered_filesystem_harvest_path() -> None:
    unsafe = (
        "falls back to re-measuring whatever",
        "best_patch.diff on disk is unexamined",
        "verify a `best_patch.diff` left on disk",
        "recovery backstop the lane re-measures",
    )
    all_roles = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "roles").glob("*.md"))
    for phrase in unsafe:
        assert phrase not in LANE
        assert phrase not in all_roles
    assert "registered-only; unregistered filesystem artifacts are never harvested" in (
        ROOT / "scripts" / "round_journal.py"
    ).read_text(encoding="utf-8")
    assert "if (STATE_DIR && !setup.resume_invalidated)" in LANE
    assert "--candidate-id" in LANE


def test_commit_and_integrated_candidates_have_durable_receipt_barriers() -> None:
    assert "round_journal:verify integrated" in LANE
    assert "integratedDurablyVerified" in LANE
    assert "a merely registered integrated patch cannot cross" in LANE
    assert "/commit_result.json (unique temp, fsync, rename)" in LANE
    journal = (ROOT / "scripts" / "round_journal.py").read_text(encoding="utf-8")
    assert 'entry.get("status") != "verified"' in journal
    assert "integration_receipt_does_not_prove_head_moved" in journal


def test_recovery_preserves_planner_targets_but_remeasures_raw_pairs() -> None:
    assert 'target_routes: Array.isArray(entry.target_routes) ? entry.target_routes : []' in LANE
    assert ".map(route => `--target-route ${JSON.stringify(String(route))} `).join('')" in LANE
    assert 'TARGET_ROUTES: Array.isArray(d.target_routes) ? d.target_routes : []' in LANE
    assert 'paired_per_case: Array.isArray(r.ver.paired_per_case) ? r.ver.paired_per_case : null' in LANE
    assert 'JavaScript never substitutes bestPerCase from another session' in LANE
