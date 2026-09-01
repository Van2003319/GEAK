#!/usr/bin/env python3
"""The phase2 KNOWLEDGE and MEMORY contracts — the read path, the write path, and the
localiser-efficacy join.

`test_phase2_roles.py` pins what a phase2 round may DO. This file pins what it LEARNS, which is a
different failure mode and a quieter one: every assertion below guards a path that fails by silently
producing nothing rather than by breaking a run. A lane that stops reading its knowledge base, or
stops carrying its measurements into the next wave, keeps passing every other test in this tree while
the search re-buys the same experiments forever.

Three of these were real defects found by reading the accumulated evidence, and they are marked.
"""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROLES = ROOT / "roles"
LANE = ROOT / "kernel_lane.js"


def role(name: str) -> str:
    return (ROLES / name).read_text(encoding="utf-8")


class AsmEngineerReadsTheKnowledgeBaseTest(unittest.TestCase):
    """The read path. Built, deployed, and for a while unreachable: the role file carried ZERO
    references to `KERNEL_KNOWLEDGE_DIR` while the orchestrator appended an instruction to consult
    "your role's SOTA section" — a section that did not exist. The pages were written and no engineer
    could find them."""

    @classmethod
    def setUpClass(cls):
        cls.text = role("asm_engineer.md")

    def test_the_role_names_the_knowledge_dir_at_all(self):
        self.assertIn("KERNEL_KNOWLEDGE_DIR", self.text,
                      "the orchestrator passes it; a role that never names it cannot use it")

    def test_it_points_at_the_pages_written_for_editing_compiled_assembly(self):
        for page in ("editing_compiled_asm.md", "edit_patterns.md", "hazards_and_waitcnt.md"):
            self.assertIn(f"asm_mfma/{page}", self.text, f"{page} is unreachable from the role")

    def test_the_pre_flight_gates_are_pointed_at_before_the_edit_menu(self):
        """Gates first is the whole economy of the phase: they close windows on paper for zero GPU
        time, and the catalogue only matters for the windows that survive them."""
        gates = self.text.index("editing_compiled_asm.md")
        menu = self.text.index("edit_patterns.md")
        self.assertLess(gates, menu, "the catalogue is listed before the gates that bound it")

    def test_the_base_may_only_add_never_foreclose(self):
        """The same contract every other engineer gets. Without it a measured-loss entry reads as a
        prohibition, and the KB becomes a cage instead of an accelerant."""
        self.assertIn("ADD-only", self.text)
        self.assertRegex(self.text, r"never overrides|never narrows|can only help, never hurt")

    def test_the_measurement_is_still_the_judge(self):
        self.assertRegex(self.text, r"[Yy]our measurement is the judge|has to clear the floor")

    def test_the_edit_menu_defers_to_the_catalogue_rather_than_replacing_it(self):
        """A hardcoded list in a prompt cannot be corrected by evidence. The measured win rates put
        its original ordering upside down, so it may orient but must not be the authority."""
        self.assertRegex(self.text, r"[Tt]he catalogue is")
        self.assertRegex(self.text, r"not the authority|orientation list")


class AsmLedgerRowsCanBeCountedTest(unittest.TestCase):
    """DEFECT FOUND IN THE EVIDENCE: 63 measured rows carried a `fingerprint` — which dies at the
    next recompile — and no window SHAPE, so no rate could be computed across kernels at all. The
    ledger was a query log."""

    @classmethod
    def setUpClass(cls):
        cls.text = role("asm_engineer.md")

    def test_the_row_carries_the_window_evidence_shape(self):
        self.assertIn("window_evidence", self.text,
                      "without it a row can only ever be re-found, never aggregated")

    def test_the_row_can_say_that_two_sites_are_one_experiment(self):
        """A destination rotation and the `vmcnt` constants fitted to it are one pipeline even when
        the localiser names them as two sites. Counting them twice inflates every rate built on it."""
        self.assertIn("experiment_group", self.text)

    def test_the_shape_is_described_as_the_thing_that_survives_a_recompile(self):
        self.assertRegex(self.text, r"fingerprint .{0,80}worthless|dies at the next recompile|"
                                    r"survives is the window")


class UpdateMemoryMustAggregateNotOnlyFileTest(unittest.TestCase):
    """DEFECT FOUND IN THE EVIDENCE: the role said "file them verbatim" and "do not re-derive a
    row", and that was read as covering the whole duty — so 63 rows accumulated across two lanes and
    nothing ever counted them. The no-fabrication rule is right; it is not the whole job."""

    @classmethod
    def setUpClass(cls):
        cls.text = role("tech_lead.md")

    def test_the_no_fabrication_rule_survives(self):
        self.assertIn("Do not re-derive a row", self.text)

    def test_but_aggregation_is_an_explicit_separate_duty(self):
        self.assertRegex(self.text, r"DO aggregate over them|aggregate over them")
        self.assertRegex(self.text, r"query log|nobody ever counts")

    def test_the_aggregation_unit_is_edit_times_window_shape(self):
        """Per-`edit` alone is the one-dimensional collapse this role already, correctly, refuses:
        it pools the windows that discriminate a win from a null."""
        self.assertRegex(self.text, r"edit` id × window-evidence shape|window-evidence shape")

    def test_the_conclusion_is_promoted_and_the_rows_are_not(self):
        self.assertRegex(self.text, r"Promote the conclusion, not the rows")

    def test_a_rate_is_not_reported_on_too_small_a_sample(self):
        self.assertRegex(self.text, r"≥4 attempts|enough behind it")

    def test_a_vocabulary_split_is_flagged_because_it_voids_the_count(self):
        """`hoist_global_load` and `hoist_load_above_use` were the same edit under two ids in the
        real ledger, which silently split a 14-attempt cell into 13 and 1."""
        self.assertRegex(self.text, r"vocabulary split")


class KeypointEfficacyLoopTest(unittest.TestCase):
    """phase1 asks continually whether its knowledge base is lying to it — declared citations joined
    against an independent re-measurement, with losses that cost a card its standing. phase2 had no
    equivalent question about its LOCALISER, whose per-bottleneck-class weights are a hand-written
    table nothing had ever checked."""

    @classmethod
    def setUpClass(cls):
        cls.text = LANE.read_text(encoding="utf-8")

    def test_the_array_exists_beside_the_one_it_mirrors(self):
        self.assertIn("const keypointCitations = []", self.text)
        self.assertIn("const citations = []", self.text)

    def test_the_join_is_phase_gated(self):
        self.assertIn("roundPhase === PHASE_ASM && roundKeypoints.length", self.text)

    def test_the_keypoints_are_captured_at_localisation_not_read_back_later(self):
        """DEFECT AVOIDED: a committed round REPLACES `profileSummary` with a fresh reprofile, so a
        join that read the live variable would see zero keypoints on exactly the rounds that
        improved — blind to every success, and silently."""
        self.assertIn("let roundKeypoints = []", self.text)
        self.assertIn("roundKeypoints = found", self.text)
        capture = self.text.index("roundKeypoints = found")
        join = self.text.index("roundPhase === PHASE_ASM && roundKeypoints.length")
        self.assertLess(capture, join, "the join reads the variable before it is set")

    def test_the_join_runs_before_the_memory_step(self):
        """The memory step is the only writer of STATE.json, so a join placed after it cannot reach
        the next wave — and one wave rarely supplies enough windows to read a rate."""
        join = self.text.index("roundPhase === PHASE_ASM && roundKeypoints.length")
        mem = self.text.index("const mem = await agentT(")
        self.assertLess(join, mem)

    def test_an_untried_window_is_not_counted_as_a_miss(self):
        """The same three-state discipline `citations` uses: a loss and an absence of evidence must
        never land in the same bucket."""
        self.assertIn("worked: hit.length > 0", self.text)
        self.assertIn("any_confirmed:", self.text)

    def test_standing_requires_winner_and_improved_together(self):
        """A round winner that did not beat the incumbent advanced nothing; crediting the window it
        came from would let the scorer bank a confirmation for a flat round."""
        self.assertIn("became_winner: !!(winner && improved", self.text)

    def test_the_evidence_shape_is_carried_as_the_cross_kernel_join_key(self):
        self.assertIn("evidence: k.evidence || null", self.text)

    def test_both_rank_and_score_are_kept(self):
        """A scorer can be right about the order and wrong about the magnitude; those are separate
        defects with separate fixes."""
        self.assertIn("rank: i + 1, score: k.score", self.text)

    def test_a_weak_join_is_visible_rather_than_silent(self):
        self.assertIn("matched_by", self.text)
        self.assertIn("'fingerprint'", self.text)

    def test_the_sample_is_resumed_across_waves(self):
        self.assertIn("ps.keypoint_citations", self.text)

    def test_the_handoff_is_gated_on_non_empty_not_on_phase(self):
        """DEFECT FOUND AND FIXED IN REVIEW: gated on `roundPhase` the array reached the memory step
        only on assembly rounds, so the first source round afterwards wrote a STATE.json without it
        and silently discarded every window the assembly rounds had measured. `asm_ledger` escapes
        the same trap only because it rides inside `PRIOR_HISTORY`."""
        self.assertIn("...(keypointCitations.length ? { KEYPOINT_CITATIONS: keypointCitations } : {})",
                      self.text)

    def test_it_is_returned_to_the_caller(self):
        """One lane can rarely supply enough windows; the aggregation is a cross-lane job."""
        self.assertIn("keypoint_citations: keypointCitations", self.text)

    def test_the_lane_never_reaching_phase2_changes_no_prompt(self):
        """An empty array must add no key, so a source-only lane is byte-identical to before this
        existed — the same rule every other opt-in feature in this file follows."""
        self.assertRegex(self.text, r"byte-identical to before")


class TechLeadReadsTheEfficacyJoinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = role("tech_lead.md")

    def test_the_input_is_documented(self):
        self.assertIn("KEYPOINT_CITATIONS", self.text)

    def test_an_untried_window_is_not_read_as_evidence_against_the_scorer(self):
        self.assertRegex(self.text, r"`worked: false` is not evidence")

    def test_the_array_is_carried_into_state_verbatim_and_not_returned(self):
        self.assertIn("keypoint_citations", self.text)
        self.assertRegex(self.text, r"copied verbatim|carry the array into STATE\.json verbatim")
        self.assertRegex(self.text, r"do not return it")

    def test_a_small_sample_is_reported_as_small(self):
        self.assertRegex(self.text, r"4 worked windows|too small")


class UpdateExperienceDistilsAssemblyEvidenceTest(unittest.TestCase):
    """DEFECT FOUND IN THE EVIDENCE: the curation role never named `HISTORY.asm_ledger`, so whether
    a phase2 lane's 30 measured window edits became a card depended on the agent noticing them. The
    lane that used phase2 most heavily produced zero card proposals."""

    @classmethod
    def setUpClass(cls):
        cls.text = role("update_experience.md")

    def test_the_assembly_ledger_is_a_named_input(self):
        self.assertIn("HISTORY.asm_ledger", self.text)

    def test_walking_it_is_a_step_not_a_suggestion(self):
        self.assertRegex(self.text, r"walk every row")

    def test_there_is_a_rule_for_the_abstraction_level(self):
        """A card naming `.LBB34_90` is rejected by the card lint AND useless, because the next run's
        compiler will not emit that block. The distillation is a lift to the window SHAPE."""
        self.assertRegex(self.text, r"window INSTANCE to a window SHAPE|abstraction level")

    def test_the_site_identifying_fields_are_explicitly_dropped(self):
        self.assertRegex(self.text, r"ordinal_in_block.{0,120}nothing|do not survive a recompile")

    def test_a_broke_correctness_row_becomes_a_pitfall(self):
        self.assertRegex(self.text, r"broke-correctness.{0,200}pitfall")

    def test_an_explained_null_can_still_earn_a_card(self):
        """"Intra-wave schedule edits are null at >=4 waves/SIMD" is backed by several independent
        measurements and is reusable, even though every arm lost."""
        self.assertRegex(self.text, r"null CAN earn a card")


if __name__ == "__main__":
    unittest.main(verbosity=2)
