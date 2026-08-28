#!/usr/bin/env python3
"""The phase2 role contracts, pinned where they can actually be read.

`test_lane_gates.js` pins the phase2 WIRING, but its runner deliberately serves only the two JS
entry points, so it cannot open a role file. The contracts below are the half of phase2 that lives
in markdown, and they are the half most likely to be softened by accident: two of them are
prohibitions, and a prohibition that quietly disappears from a prompt fails silently and forever.
"""
from __future__ import annotations

import unittest
from pathlib import Path

ROLES = Path(__file__).resolve().parent.parent / "roles"


def read(name: str) -> str:
    return (ROLES / name).read_text(encoding="utf-8")


class AsmEngineerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = read("asm_engineer.md")

    def test_the_role_exists_and_says_what_phase_it_is(self):
        self.assertIn("phase2", self.text)
        self.assertIn("device assembly", self.text)

    def test_editing_source_is_prohibited_not_merely_discouraged(self):
        """The one rule that makes phase2 a different phase rather than phase1 relabelled. An
        assembly round that edits the source reports a source result under the assembly phase's
        name, and the ladder then concludes assembly optimisation works when it was never tried."""
        self.assertIn("Do not edit any source file", self.text)
        self.assertRegex(self.text, r"source is read-only|source files are read-only")

    def test_the_round_trip_gate_runs_before_any_edit(self):
        """An unedited round trip that already changes the binary means every later measurement
        attributes the round trip's own damage to the engineer."""
        self.assertIn("round_trip_identical: true", self.text)
        self.assertIn("verify", self.text)
        self.assertRegex(self.text, r"STOP and submit|status: \"blocked\"")

    def test_the_engineer_is_told_to_relocate_by_fingerprint_not_by_line(self):
        """Its own first edit moves every line below it."""
        self.assertIn("fingerprint", self.text)
        self.assertRegex(self.text, r"never by remembering a number|re-locate by fingerprint")

    def test_it_must_build_through_the_round_trip_rather_than_around_it(self):
        self.assertIn("asm_roundtrip.py rebuild", self.text)
        self.assertRegex(self.text, r"do not edit `build\.ninja`|do not edit build\.ninja")

    def test_resource_widening_is_called_out_as_a_way_to_fake_a_win(self):
        self.assertRegex(self.text, r"VGPR/AGPR/LDS/scratch|occupancy")

    def test_it_declares_mechanism_and_routes_like_every_other_engineer(self):
        self.assertIn("mechanism_claims", self.text)
        self.assertIn("target_routes", self.text)
        self.assertRegex(self.text, r"BEFORE you measure|Declare them BEFORE",
                         "a claim fitted to a result is not evidence of a mechanism")

    def test_an_empty_keypoint_list_is_not_permission_to_edit_anywhere(self):
        self.assertRegex(self.text, r"may be empty")
        self.assertIn("not keypoint-guided", self.text)


class VerifyEngineerAsmContractTest(unittest.TestCase):
    """The half of the assembly phase that decides whether its numbers mean anything.

    `build.ninja` has one `cuda_compile` edge per translation unit and its input is the `.hip`.
    NOTHING IN THE BUILD GRAPH COMPILES A `.s`. So a verifier that applies an assembly patch to a
    clean copy and builds it normally applies the assembly into a tree where nothing reads it,
    compiles the untouched source, and times THE PARENT -- while policy, twin-sync, the oracle
    digest and correctness all report a clean pass. The failure has no symptom, which is why the
    contract is pinned here rather than left to whoever reads the role next.
    """

    @classmethod
    def setUpClass(cls):
        cls.text = read("verify_engineer.md")

    def test_the_role_knows_the_build_graph_does_not_compile_assembly(self):
        """The premise. Without it every rule below reads as ceremony."""
        self.assertRegex(self.text,
                         r"[Nn]othing in the build graph compiles a `?\.s`?")

    def test_the_decision_is_made_from_the_patch_not_from_the_phase(self):
        """A verifier told "you are in phase2" by an input can be told wrong, and the warm-start
        arm is not told at all. The patch is the artefact that cannot be mistaken."""
        self.assertIn("git apply --numstat", self.text)
        self.assertRegex(self.text, r"[Dd]ecide from the PATCH")

    def test_it_routes_the_build_through_the_round_trip(self):
        self.assertIn("asm_roundtrip.py", self.text)
        self.assertIn("restore --source-root", self.text)

    def test_a_missing_helper_fails_closed_rather_than_measuring_anyway(self):
        """The one refusal that matters: without the helper the candidate's binary cannot be
        produced, and a number measured without it is not a weaker result but a confident
        measurement of a different program."""
        self.assertIn("asm_unbuilt", self.text)
        self.assertRegex(self.text, r"do not build, correctness-check or\s*\n?\s*benchmark")

    def test_a_restore_that_did_not_complete_is_not_benchmarked(self):
        """Exit 2 is the tool's HOLE and exit 3 is a failed rebuild. Both mean the edited assembly
        is absent from the binary, so there is nothing there to time."""
        self.assertRegex(self.text, r"`2` and `3` are both `status:\"asm_unbuilt\"`")

    def test_correctness_runs_twice_and_the_first_verdict_is_discarded(self):
        """The build is what creates `build.ninja`, and `restore` needs it -- so the first
        correctness run necessarily happens BEFORE the restore and validates the unpatched
        kernel. Reporting it would certify a binary that was never measured."""
        self.assertRegex(self.text, r"[Rr]un correctness AGAIN")
        self.assertRegex(self.text, r"first one (?:validated|measured) the unpatched kernel")

    def test_an_unedited_assembly_is_reported_rather_than_rounded(self):
        """`edited: false` means the engineer submitted a pristine `.s`. Recorded as "no effect"
        it closes a direction nothing tried -- finding (87)'s cost, in the assembly phase."""
        self.assertRegex(self.text, r"`edited:\s*\n?\s*false`")
        self.assertRegex(self.text, r"there was no\s*\n?\s*edit to measure")

    def test_unchanged_machine_code_is_decisive_for_an_assembly_candidate(self):
        """For a source candidate identical codegen is a remark. Here the patch IS the machine
        code, so identical means the restore never landed."""
        self.assertRegex(self.text, r"the patch IS the machine\s*\n?\s*code")
        self.assertIn("reached_binary", self.text)

    def test_the_receipt_is_part_of_the_return_and_not_a_boolean(self):
        for field in ('"asm_restore"', "assembly_found", "exit_code"):
            self.assertIn(field, self.text)
        self.assertIn("asm_unbuilt|correctness_failed", self.text,
                      "the status vocabulary must carry the new verdict, or the orchestrator "
                      "reads a fail-closed refusal as an unknown status")


class AsmEngineerKnowsHowItIsBuiltTest(unittest.TestCase):
    """The engineer's own copy of the same fact.

    This role used to be told verification rebuilds its patch "exactly as the lane does for a
    source patch". That sentence was the bug written down: for a source patch a rebuild recompiles
    the changed file, and for this one it recompiles the UNCHANGED one.
    """

    @classmethod
    def setUpClass(cls):
        cls.text = read("asm_engineer.md")

    def test_the_engineer_is_told_not_to_strip_the_provenance_stamp(self):
        """It is how the verifier learns which of this tree's builds the assembly belongs to --
        the tree holds the candidate AND the immutable oracle."""
        self.assertIn("geak.asm-provenance", self.text)
        self.assertRegex(self.text, r"[Kk]eep the first line")

    def test_it_is_told_how_its_patch_actually_reaches_a_number(self):
        self.assertRegex(self.text, r"no edge in `build\.ninja` compiles a\s*\n?\s*`?\.s`?",
                         "without this, a flat number reads to the engineer as a skipped build")
        self.assertIn("asm_roundtrip.py restore", self.text)


class AssemblyIsTheEvolvingArtifactTest(unittest.TestCase):
    """phase2 EVOLVES the assembly, and these are the contracts that let it compound.

    The role used to say "emit the assembly" unconditionally. `emit` regenerates it FROM SOURCE --
    the compiler's own output -- so every round threw away the edits the previous round banked and
    restarted. The search could re-discover one window's win forever and never stack two, while the
    log showed a busy phase. Nothing about that failure is visible in a number, which is why the
    contract is pinned rather than left to whoever reads the role next.
    """

    @classmethod
    def setUpClass(cls):
        cls.asm = read("asm_engineer.md")
        cls.lane = (ROLES.parent / "kernel_lane.js").read_text(encoding="utf-8")

    def test_the_role_says_the_assembly_is_the_artifact_under_optimisation(self):
        self.assertRegex(self.asm, r"THE `\.s` IS THE ARTIFACT UNDER OPTIMISATION")
        self.assertRegex(self.asm, r"[Tt]he `\.s` you find there is the incumbent")

    def test_continuation_is_checked_before_anything_is_emitted(self):
        self.assertIn("incumbent --source-root", self.asm)
        self.assertRegex(self.asm, r"CONTINUING or STARTING")
        self.assertRegex(self.asm, r"That file is your starting point\. Do NOT emit\.")

    def test_emit_is_named_as_the_cold_start_path_only(self):
        """Naming it is what stops the next reader restoring the unconditional emit as a tidy-up."""
        self.assertRegex(self.asm, r"`emit` is the cold-start path only|`emit` is the cold-start path")
        self.assertRegex(self.asm, r"regenerates the assembly \*\*from source\*\*|"
                                   r"regenerates the assembly from source")

    def test_supersession_is_the_one_case_a_fresh_emit_is_correct(self):
        self.assertRegex(self.asm, r"superseded")
        self.assertRegex(self.asm, r"program that no longer exists")

    def test_the_lane_prompt_tells_the_engineer_to_ask_first(self):
        """The role file is only read after the prompt, and the prompt used to carry the
        unconditional `emit it with asm_roundtrip.py`."""
        self.assertRegex(self.lane, r"START BY ASKING WHETHER YOU ARE CONTINUING")
        self.assertNotRegex(self.lane, r"ASSEMBLY, not the source: emit it with asm_roundtrip\.py",
                            "the unconditional emit instruction is what stopped phase2 compounding")

    def test_the_round_trip_gate_still_makes_sense_when_continuing(self):
        """It is a toolchain check, not a statement about the incumbent -- so it stays first, and
        the role has to say why it does not conflict with continuing."""
        self.assertRegex(self.asm, r"about the TOOLCHAIN")
        self.assertRegex(self.asm, r"does not overwrite it")


class AssemblyHasItsOwnMemoryTest(unittest.TestCase):
    """The assembly phase keeps its own ledger, and the separation is load-bearing in BOTH
    directions.

    Unit: a source row closes a MECHANISM, an assembly row closes a WINDOW plus an edit kind --
    `relax_waitcnt` on the MFMA drain is not the same experiment as `relax_waitcnt` on the epilogue
    store. Semantics: a source `dead-end` records that the compiler's own output was already the
    better program, which is the proposition phase2 exists to test, so handing it to an assembly
    engineer as a closure shuts the door the phase was opened to walk through.
    """

    @classmethod
    def setUpClass(cls):
        cls.asm = read("asm_engineer.md")
        cls.lead = read("tech_lead.md")
        cls.lane = (ROLES.parent / "kernel_lane.js").read_text(encoding="utf-8")

    def test_the_engineer_reads_the_assembly_ledger_and_not_the_source_one(self):
        self.assertIn("ASM_LEDGER", self.asm)
        self.assertRegex(self.asm, r"You are NOT given the source `LEDGER`")

    def test_the_lane_swaps_the_ledger_by_phase_rather_than_sending_both(self):
        self.assertRegex(
            self.lane,
            r"\.\.\.\(isAsm \? \{ ASM_LEDGER: history\.asm_ledger \} : \{ LEDGER: history\.ledger \}\)")

    def test_the_assembly_ledger_is_a_sibling_of_the_source_one_through_the_whole_lifeline(self):
        """Created, resumed, merged, and declared. A break at any one of the four loses the memory
        silently -- an undeclared field is stripped before the lane ever sees it."""
        for pattern, why in (
                (r"const history = \{ insights: \[\], ledger: \[\], asm_ledger: \[\]", "created"),
                (r"if \(Array\.isArray\(ps\.asm_ledger\)\) history\.asm_ledger = ps\.asm_ledger;",
                 "resumed"),
                (r"history\.asm_ledger = history\.asm_ledger\.concat\(mem\.asm_ledger\)", "merged"),
                (r"asm_ledger: \{ type: 'array'", "declared in MEMORY_SCHEMA"),
                (r"asm_ledger_rows: \{ type: 'array'", "declared in ENG_SCHEMA")):
            self.assertRegex(self.lane, pattern, f"the asm ledger is not {why}")

    def test_the_window_is_the_unit_and_it_is_relocated_by_fingerprint(self):
        self.assertRegex(self.asm, r"a window plus an edit kind|window plus an edit kind")
        self.assertRegex(self.asm, r"[Ll]ocate a row by `fingerprint`, never by line number")

    def test_the_verdicts_that_have_no_source_equivalent_are_defined(self):
        for verdict in ("broke-correctness", "not-realized", "confirmed", "dead-end"):
            self.assertIn(verdict, self.asm, f"{verdict} is not explained to the engineer")
        self.assertIn("broke-correctness", self.lead,
                      "the planner files the rows and must know the vocabulary it is filing")

    def test_the_planner_is_told_to_file_the_rows_verbatim(self):
        """It never saw the instruction stream, so a fingerprint it invented names a window the
        next round will fail to find."""
        self.assertIn("ASM_LEDGER_ROWS", self.lead)
        self.assertRegex(self.lead, r"[Dd]o not re-derive a row")
        self.assertRegex(self.lead, r"do not fold them into\s*\n?`?ledger`?|not to fold")

    def test_a_source_win_supersedes_the_assembly_rows_rather_than_deleting_them(self):
        self.assertRegex(self.lead, r'lifecycle: "superseded"')

    def test_continuation_is_reported_so_a_restart_is_not_read_as_a_weak_round(self):
        self.assertIn("continued_from", self.asm)
        self.assertRegex(self.lane, r"continued_from: \{ type: 'string', enum:")
        self.assertIn("ASM_CONTINUATION", self.lane)
        self.assertIn("ASM_CONTINUATION", self.lead)


class KeypointsFollowTheIncumbentTest(unittest.TestCase):
    """A keypoint has to name a window in the program that is ABOUT to be edited.

    The lane's comment used to say a fresh emit was "the same text". That holds only until the
    first phase2 round banks: after it, the canonical `.s` is the engineer's edited version and a
    fresh emit is the compiler's. Localising on the second yields windows that may not exist in the
    incumbent -- and the engineer, told to re-locate by fingerprint, lands on whichever site in its
    own assembly happens to match, which reads in the log exactly like a localised round.
    """

    @classmethod
    def setUpClass(cls):
        cls.prof = read("profile_engineer.md")
        cls.lane = (ROLES.parent / "kernel_lane.js").read_text(encoding="utf-8")

    def test_the_profiler_asks_for_the_incumbent_before_emitting(self):
        self.assertIn("incumbent --source-root", self.prof)
        self.assertRegex(self.prof, r"\*\*Do not emit when there is an incumbent\.\*\*")

    def test_the_same_text_claim_is_gone_from_the_lane(self):
        self.assertNotRegex(self.lane, r"localises on that instead, which is the same text",
                            "that claim is only true before the first phase2 round banks")

    def test_keypoints_are_scored_against_what_was_already_worked(self):
        """The tool ranks on static stall proxies and cannot know a window was already edited."""
        self.assertIn("ASM_LEDGER", self.prof)
        self.assertRegex(self.prof, r"already `confirmed`|already \`confirmed\`")


class IntegratorKnowsAssemblyTest(unittest.TestCase):
    """A `.s` is a whole-file alternative rendering of one translation unit's machine code, not a
    diff against the source, so `combine these patches` is not defined for two of them."""

    @classmethod
    def setUpClass(cls):
        cls.text = read("integrator.md")

    def test_it_detects_an_assembly_patch(self):
        self.assertIn("git apply --numstat", self.text)

    def test_two_assembly_patches_for_one_unit_are_not_combinable(self):
        self.assertRegex(self.text, r"SAME translation unit → not combinable")
        self.assertRegex(self.text, r"a schedule neither engineer measured")

    def test_a_source_patch_and_an_assembly_patch_for_one_unit_are_not_combinable(self):
        self.assertRegex(self.text, r"SOURCE patch touching the same translation unit")

    def test_different_translation_units_still_combine(self):
        self.assertRegex(self.text, r"[Dd]ifferent translation units → combinable")

    def test_a_stacked_assembly_arm_is_built_through_the_round_trip(self):
        self.assertIn("asm_roundtrip.py restore", self.text)
        self.assertRegex(self.text, r"No edge in `build\.ninja`\s*\n?\s*compiles a `\.s`")


class ProfilerKeypointExceptionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = read("profile_engineer.md")

    def test_the_scope_rule_survives_the_exception(self):
        """The rule exists because this role once did profiling AND attribution, which flattened the
        evidence ladder. phase2 needs it to read the assembly; that must read as a bounded exception
        rather than as the rule being withdrawn."""
        self.assertIn("you are L2, and only L2", self.text)
        self.assertIn("PHASE=keypoint", self.text)
        self.assertRegex(self.text, r"narrow, explicit\s*\n?\*{0,2}exception|exception to it")

    def test_the_exception_is_bounded_to_where_and_never_why(self):
        self.assertIn("You may not diagnose from it", self.text)
        self.assertRegex(self.text, r"no `attributed_pass`|no attributed_pass")
        self.assertRegex(self.text, r"that is a `request_l3`|request_l3",
                         "a lowering question found while localising must route to L3, not be "
                         "answered here")

    def test_keypoints_are_passed_through_verbatim(self):
        """The engineer relocates a window by fingerprint and ordinal; a hand-edited record breaks
        that, and a merged one describes a site the tool never scored."""
        self.assertRegex(self.text, r"verbatim")
        self.assertRegex(self.text, r"[Dd]o not rewrite the windows")

    def test_an_empty_result_is_a_legitimate_answer(self):
        self.assertRegex(self.text, r"EMPTY `keypoints` array is a legitimate answer")
        self.assertRegex(self.text, r"better than inventing a location")

    def test_the_bottleneck_is_passed_so_the_ranking_is_weighted_by_it(self):
        self.assertIn("--bottleneck", self.text)
        self.assertRegex(self.text, r"WEIGHTS the ranking|it WEIGHTS")

    def test_the_hot_kernel_is_passed_so_a_cold_one_cannot_outrank_it(self):
        self.assertIn("--kernel", self.text)
        self.assertRegex(self.text, r"cold kernel can\s*\n?\s*outrank")


class RoleFilesAreReachableTest(unittest.TestCase):
    def test_every_role_the_lane_dispatches_has_a_file(self):
        """A prompt that says `Read roles/X.md` for a file that does not exist degrades into an
        agent improvising the contract, which is indistinguishable in the log from following one."""
        lane = (ROLES.parent / "kernel_lane.js").read_text(encoding="utf-8")
        import re
        named = set(re.findall(r"roles/(\w+)\.md", lane))
        named |= set(re.findall(r"roleAgent\('(\w+)'", lane))
        missing = sorted(n for n in named if not (ROLES / f"{n}.md").is_file())
        self.assertEqual([], missing, f"dispatched but absent: {missing}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
