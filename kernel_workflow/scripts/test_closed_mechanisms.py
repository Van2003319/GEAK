"""Tests for the measured-closed mechanism registry.

Two failure modes are worth more than the rest. The registry can go quiet --
matching nothing a planner actually writes, which is finding (128) again. And it
can turn into a ban list -- entries with a reason instead of a number, or with no
way back. Most of what is below is aimed at those two.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import closed_mechanisms as CM  # noqa: E402


class EveryEntryCitesAMeasurementTest(unittest.TestCase):
    def test_every_entry_names_a_finding(self):
        for c in CM.CLOSED:
            self.assertTrue(c.finding.strip(), c.axis)

    def test_every_entry_names_what_was_built(self):
        for c in CM.CLOSED:
            self.assertTrue(c.built.strip(), c.axis)

    def test_every_entry_carries_a_number(self):
        # A closure without a number is a hypothesis, and hypotheses are what
        # the slots are for.
        for c in CM.CLOSED:
            self.assertRegex(c.measured, r"\d", c.axis)

    def test_every_entry_names_a_negative_control(self):
        for c in CM.CLOSED:
            self.assertTrue(c.control.strip(), c.axis)

    def test_every_entry_can_be_reopened(self):
        for c in CM.CLOSED:
            self.assertTrue(c.reopens_when.strip(),
                            f"{c.axis} is a ban, not a closure")

    def test_every_entry_names_the_epoch_that_bounded_it(self):
        # Finding (92): a closure is only as tight as the floor behind it, and a
        # floor belongs to a box. A closure may accumulate arms measured on
        # different boxes -- the PF entry's bk64 sweep was bounded on N and its
        # BK=32 arm on Q -- so the field is one epoch letter or a
        # comma-separated list of them, and never free prose that hides which
        # box did the bounding.
        for c in CM.CLOSED:
            self.assertRegex(c.epoch, r"^[A-Z](,[A-Z])*$", c.axis)

    def test_every_entry_states_its_bound(self):
        for c in CM.CLOSED:
            self.assertTrue(c.bound.strip(), c.axis)

    def test_axes_are_unique(self):
        axes = [c.axis for c in CM.CLOSED]
        self.assertEqual(len(axes), len(set(axes)))


class MatchingTest(unittest.TestCase):
    def test_it_matches_the_form_a_planner_actually_writes(self):
        hits = CM.matches("rasterization: grouped_m -> xcd_remapped_grouped")
        self.assertEqual([h.axis for h in hits], ["rasterization / L2 traffic reduction"])

    def test_underscores_and_hyphens_do_not_hide_a_match(self):
        for text in ("prefetch_depth", "prefetch-depth", "Prefetch Depth"):
            self.assertTrue(CM.matches(text), text)

    def test_output_path_in_either_direction_is_matched(self):
        for text in ("output_path -> direct_store", "atomic_fixup on the tail"):
            axes = [h.axis for h in CM.matches(text)]
            self.assertIn("output_path on split-K routes", axes)

    def test_an_open_idea_matches_nothing(self):
        self.assertEqual(CM.matches("widen the K-tile to 128 and re-tune slices"), [])

    def test_matching_is_case_insensitive(self):
        self.assertTrue(CM.matches("S_WAITCNT PLACEMENT"))

    def test_the_n_strip_closure_is_matched_by_its_mechanism(self):
        for text in ("try a narrower N strip so more CTAs launch",
                     "nwaves: adaptive strip width", "shrink the tile in N"):
            axes = [h.axis for h in CM.matches(text)]
            self.assertIn("shortening the N-strip to raise CU occupancy", axes, text)

    def test_the_fused_reduction_is_matched_by_the_words_a_rediscoverer_uses(self):
        # (142), and this one is a real catch rather than a constructed one.
        # The first string below is verbatim the proposal that was put to this
        # checker while planning a round on v98 -- reached honestly from the
        # reduce call site's own comment, with no knowledge that it was v78.
        # The registry HELD the measurement (-43.6% geomean) and still returned
        # exit 0, because every alias it carried was an implementation name.
        # A false clear is the expensive direction: it costs a build, a
        # correctness gate and a measured round to rediscover a number already
        # in the file.
        rediscovered = [
            "single-kernel split-K: last-CTA-does-reduction. The splitk reduce "
            "kernel is a measured ~4.7 us dispatch floor. Remove the second "
            "launch entirely: each slice atomically bumps a per-tile counter, "
            "and the last slice to arrive reduces that tile's partials in the "
            "same launch.",
            "stop launching a second kernel at all",
            "fuse the reduction into the GEMM with an arrival counter",
            "use a cooperative launch and grid sync instead of two kernels",
            "threadfence reduction in the last arriving CTA",
        ]
        for text in rediscovered:
            axes = [h.axis for h in CM.matches(text)]
            self.assertIn("output_path on split-K routes", axes, text)

    def test_the_new_aliases_did_not_swallow_an_unrelated_idea(self):
        # The other direction of the same trade-off. Widening an alias list is
        # only safe if it still discriminates; "reduction" and "kernel" are
        # ordinary words in this vocabulary.
        for text in ("raise the split-K slice count on decode routes",
                     "reduce the LDS stride to fit another CTA per CU",
                     "launch the kernel on a second stream"):
            axes = [h.axis for h in CM.matches(text)]
            self.assertNotIn("output_path on split-K routes", axes, text)

    # Entries whose `reopens_when` describes THE SAME mechanism under different
    # conditions ("the same idea, but on a route outside these two", "the same
    # idea, once a toolchain supports it"). For those, self-matching is correct:
    # the proposal really is the closed mechanism, and it really should have to
    # cite the condition. The check below applies to the other kind -- entries
    # whose reopen condition names a DIFFERENT mechanism, which must be
    # proposable without arguing your way past a measurement of something else.
    REOPENS_WITH_SAME_MECHANISM = {
        "rasterization / L2 traffic reduction",     # "a route outside these two"
        "prefetch depth / s_waitcnt placement (memory side of 7b)",
        "barrier count (barrier side of 7b)",
        "MFMA instruction shape (16x16x16 -> 32x32x8)",
        "active-CU fraction as the clock residual explanation",
        "output_path on split-K routes",
        # Its reopen condition is "the same mechanism on a route in a different
        # residency or intensity class", so it cannot be phrased without naming
        # PF. That is the honest shape of this closure -- it closes PF on five
        # named routes, not the axis -- and the exemption is what keeps it from
        # reading as a ban on prefetch everywhere.
        "register prefetch depth PF>1 on an already-resident tile",
    }

    def test_no_closure_flags_the_very_mechanism_it_says_would_reopen_it(self):
        # A closure whose alias list also covers its own `reopens_when` is a
        # ratchet wearing a measurement's clothes: the one proposal that is
        # SUPPOSED to get through is the one it stops. This bit for real --
        # the N-strip entry first shipped with "occupancy" and "idle cus" as
        # aliases, and those words describe split-K-with-reduction, which is
        # exactly what its reopen condition invites. Motivation words do not
        # belong in an alias list; mechanism words do.
        checked = 0
        for closure in CM.CLOSED:
            if closure.axis in self.REOPENS_WITH_SAME_MECHANISM:
                continue
            checked += 1
            hits = [h.axis for h in CM.matches(closure.reopens_when)]
            self.assertNotIn(
                closure.axis, hits,
                f"{closure.axis!r} flags its own reopen condition, so the "
                f"escape hatch it documents cannot be used without a "
                f"justification for a mechanism it never measured")
        self.assertGreater(checked, 0, "the exemption set swallowed every entry")

    def test_the_exemption_set_only_names_entries_that_exist(self):
        # Otherwise a renamed axis silently drops out of the check above and
        # takes its exemption with it -- the hole stops being labelled.
        axes = {c.axis for c in CM.CLOSED}
        self.assertEqual(self.REOPENS_WITH_SAME_MECHANISM - axes, set())


class RouteScopeTest(unittest.TestCase):
    def test_a_route_scoped_closure_does_not_bind_another_route(self):
        traffic = next(c for c in CM.CLOSED if c.axis.startswith("rasterization"))
        self.assertTrue(CM.relevant(traffic, "prefill_m1024_down"))
        self.assertFalse(CM.relevant(traffic, "decode_m2_square"))

    def test_a_suite_closure_binds_every_route(self):
        barrier = next(c for c in CM.CLOSED if c.axis.startswith("barrier"))
        self.assertTrue(CM.relevant(barrier, "decode_m2_square"))
        self.assertTrue(CM.relevant(barrier, None))

    def test_an_unnamed_route_is_treated_as_in_scope(self):
        # Fail closed: a proposal that does not say where it applies is checked
        # against everything rather than exempted from everything.
        traffic = next(c for c in CM.CLOSED if c.axis.startswith("rasterization"))
        self.assertTrue(CM.relevant(traffic, None))


class CheckTest(unittest.TestCase):
    def test_a_closed_proposal_is_flagged_with_its_citation(self):
        verdict = CM.check([{"axis": "rasterization", "route": "prefill_m1024_down"}])
        self.assertEqual(len(verdict["flagged"]), 1)
        record = verdict["flagged"][0]
        self.assertEqual(record["finding"], "(38), demoting item 7")
        self.assertIn("6.8%", record["measured"])
        self.assertTrue(record["reopens_when"])

    def test_the_same_proposal_on_an_out_of_scope_route_is_not_flagged(self):
        verdict = CM.check([{"axis": "rasterization", "route": "decode_m2_square"}])
        self.assertEqual(verdict["flagged"], [])

    def test_a_reopen_justification_moves_it_from_flagged_to_justified(self):
        verdict = CM.check([{"axis": "prefetch depth",
                             "reopen_justification": "exposure now measures 0.55"}])
        self.assertEqual(verdict["flagged"], [])
        self.assertEqual(len(verdict["justified"]), 1)

    def test_a_blank_justification_does_not_count(self):
        verdict = CM.check([{"axis": "prefetch depth", "reopen_justification": "   "}])
        self.assertEqual(len(verdict["flagged"]), 1)

    def test_it_reads_the_rationale_not_just_the_axis(self):
        # The axis field is often a tidy invented name; the mechanism shows up
        # in the prose underneath it.
        verdict = CM.check([{"axis": "epilogue_v2",
                             "rationale": "store straight out with direct_store"}])
        self.assertEqual(len(verdict["flagged"]), 1)

    def test_an_open_proposal_passes_clean(self):
        verdict = CM.check([{"axis": "k_tile", "rationale": "widen K to 128"}])
        self.assertEqual(verdict["flagged"], [])
        self.assertEqual(verdict["proposals_checked"], 1)

    def test_every_proposal_is_counted_even_when_none_are_flagged(self):
        verdict = CM.check([{"axis": "a"}, {"axis": "b"}, {"axis": "c"}])
        self.assertEqual(verdict["proposals_checked"], 3)


class CliTest(unittest.TestCase):
    def _run(self, argv, stdin=None):
        return subprocess.run([sys.executable, str(HERE / "closed_mechanisms.py")] + argv,
                              input=stdin, capture_output=True, text=True)

    def test_list_emits_the_registry_as_sorted_json(self):
        proc = self._run(["--list"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        entries = json.loads(proc.stdout)
        self.assertEqual(len(entries), len(CM.CLOSED))
        self.assertIn("reopens_when", entries[0])

    def test_a_flagged_proposal_exits_6_and_explains_on_stderr(self):
        proc = self._run(["--proposals", "-"],
                         stdin=json.dumps([{"axis": "rasterization",
                                            "route": "prefill_m1024_down"}]))
        self.assertEqual(proc.returncode, 6)
        self.assertIn("ALREADY MEASURED SHUT", proc.stderr)
        self.assertIn("Reopens when", proc.stderr)

    def test_a_justified_proposal_exits_0_but_is_still_announced(self):
        proc = self._run(["--proposals", "-"],
                         stdin=json.dumps([{"axis": "barrier",
                                            "reopen_justification": "four per stage now"}]))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("REOPENED", proc.stderr)

    def test_a_clean_proposal_list_exits_0_and_says_nothing_on_stderr(self):
        proc = self._run(["--proposals", "-"], stdin=json.dumps([{"axis": "k_tile"}]))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr.strip(), "")

    def test_it_accepts_a_directions_wrapper(self):
        proc = self._run(["--proposals", "-"],
                         stdin=json.dumps({"directions": [{"axis": "mfma shape"}]}))
        self.assertEqual(proc.returncode, 6)

    def test_neither_flag_is_an_error_not_a_silent_pass(self):
        proc = self._run([])
        self.assertNotEqual(proc.returncode, 0)


class WrapperKeyTest(unittest.TestCase):
    def test_both_techlead_phase_shapes_are_read(self):
        # plan_seed emits `candidate_directions`, plan_round emits `directions`.
        # Reading only one would make the checker silently pass an empty list on
        # the other phase -- the worst failure available to a tool whose job is
        # to notice things.
        for key in ("candidate_directions", "directions"):
            verdict = CM.check(json.loads(json.dumps({key: [{"title": "mfma shape"}]}))[key])
            self.assertEqual(len(verdict["flagged"]), 1, key)

    def test_the_techlead_field_names_are_scanned(self):
        # `why` and `prompt` are where the mechanism actually gets written.
        for field in ("why", "prompt", "title"):
            verdict = CM.check([{field: "swizzle the tiles for L2"}])
            self.assertEqual(len(verdict["flagged"]), 1, field)


class FalsePositivePostureTest(unittest.TestCase):
    """The matcher is deliberately loose, and that is a stated trade."""

    def test_an_incidental_mention_does_flag_and_that_is_the_intended_cost(self):
        # "remove the barrier" and "the barrier is not the problem" both trip the
        # barrier entry. A tighter matcher would miss the phrasings that matter,
        # and the cost of a false positive is one `reopen_justification` line
        # while the cost of a false negative is a whole slot.
        verdict = CM.check([{"why": "the barrier is not the problem here"}])
        self.assertEqual(len(verdict["flagged"]), 1)

    def test_the_escape_hatch_is_one_field(self):
        verdict = CM.check([{"why": "the barrier is not the problem here",
                             "reopen_justification": "not proposing a barrier change"}])
        self.assertEqual(verdict["flagged"], [])


class RolePromptTest(unittest.TestCase):
    PROMPT = (HERE.parent / "roles" / "tech_lead.md").read_text(encoding="utf-8")

    def test_the_planner_is_told_to_run_the_checker(self):
        self.assertIn("closed_mechanisms.py", self.PROMPT)

    def test_the_prompt_states_the_exit_code_the_script_actually_uses(self):
        self.assertIn("exits **6**", self.PROMPT)
        self.assertIn("return 6 if verdict[\"flagged\"] else 0",
                      (HERE / "closed_mechanisms.py").read_text(encoding="utf-8"))

    def test_the_prompt_names_the_escape_hatch_by_its_real_field_name(self):
        self.assertIn("reopen_justification", self.PROMPT)
        self.assertIn("reopens_when", self.PROMPT)

    def test_the_prompt_says_it_is_a_citation_not_a_veto(self):
        self.assertIn("not a veto", self.PROMPT)

    def test_every_registry_entry_is_reproduced_in_the_prompt(self):
        # Finding (131): provenance that requires a lookup is provenance that
        # gets skipped. The planner must see the closures without opening a
        # second file -- and the two copies must not drift, which is what this
        # test is really for.
        for c in CM.CLOSED:
            self.assertIn(c.finding.split(",")[0], self.PROMPT, c.axis)

    def test_the_prompt_does_not_read_as_the_routes_being_finished(self):
        # The registry closes MECHANISMS. prefill_m1024_down is still the
        # richest route in the suite, and a planner that reads this table as a
        # route blacklist would abandon the best target on the board.
        self.assertIn("does *not* say", self.PROMPT)
        self.assertIn("prefill_m1024_down", self.PROMPT)


if __name__ == "__main__":
    unittest.main()
