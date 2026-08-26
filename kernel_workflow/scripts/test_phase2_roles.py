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
