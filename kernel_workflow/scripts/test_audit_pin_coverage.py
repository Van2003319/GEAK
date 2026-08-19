#!/usr/bin/env python3
"""Tests for the clause extractor in `audit_pin_coverage.py`.

The auditor is deliberately not a test and its headline number is deliberately
not gated (see its docstring, and (55)). But its *parser* is now the thing that
decides whether a clause is audited at all, and the first version of it silently
skipped a third of the corpus by only reading the leading conjunct of each
assertion. A parser whose failure mode is "reports less work than exists" is
exactly the one that must not be checked by reading its output, because its
output looks better when it is more broken.

So: the count is ungated, the extraction is tested.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "audit_pin_coverage", HERE / "audit_pin_coverage.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


class ClauseExtractionTest(unittest.TestCase):
    def _pins(self, js: str):
        return audit._pins(js)

    def test_a_trailing_conjunct_is_extracted_not_just_the_leading_one(self):
        # The defect the second (84) pass found: 29 of 95 patterns lived here.
        pins = self._pins("ok(/alpha/.test(src) && /beta/.test(src), 'why');\n")
        self.assertEqual([p[2] for p in pins], ["alpha", "beta"])

    def test_every_clause_of_one_assertion_shares_an_owner(self):
        pins = self._pins("ok(/alpha/.test(src) && /beta/.test(src), 'why');\n")
        self.assertEqual(len({p[4] for p in pins}), 1)

    def test_clauses_of_different_assertions_get_different_owners(self):
        pins = self._pins("ok(/alpha/.test(src));\nok(/beta/.test(src));\n")
        self.assertEqual(len({p[4] for p in pins}), 2)

    def test_a_negated_clause_is_marked_negated(self):
        pins = self._pins("ok(/alpha/.test(src) && !/beta/.test(src));\n")
        self.assertEqual([(p[2], p[1]) for p in pins], [("alpha", False), ("beta", True)])

    def test_a_counting_clause_is_extracted_and_labelled(self):
        # `.match(/re/g)` -- invisible to a `.test`-only extractor, and it is the
        # strong "declared at every site" form, so missing it is the worst case.
        pins = self._pins("ok((src.match(/alpha/g) || []).length === 3);\n")
        self.assertEqual([(p[2], p[5]) for p in pins], [("alpha", "count")])

    def test_a_counting_clause_is_never_marked_negated(self):
        pins = self._pins("ok((src.match(/alpha/g) || []).length === 3);\n")
        self.assertFalse(pins[0][1])

    def test_test_and_count_clauses_coexist_in_one_assertion(self):
        pins = self._pins(
            "ok(/alpha/.test(src) && (src.match(/alpha/g) || []).length === 3);\n")
        self.assertEqual([p[5] for p in pins], ["test", "count"])
        self.assertEqual(len({p[4] for p in pins}), 1)

    def test_the_matched_file_is_recorded(self):
        pins = self._pins("ok(/alpha/.test(src) && /beta/.test(wfSrc));\n")
        self.assertEqual([p[3] for p in pins], ["src", "wfSrc"])

    def test_a_count_clause_against_the_workflow_file_keeps_its_file(self):
        pins = self._pins("ok((wfSrc.match(/alpha/g) || []).length === 2);\n")
        self.assertEqual((pins[0][3], pins[0][5]), ("wfSrc", "count"))

    def test_line_numbers_are_one_based_and_per_clause(self):
        pins = self._pins("ok(/alpha/.test(src));\n\nok(/beta/.test(src));\n")
        self.assertEqual([p[0] for p in pins], [1, 3])

    def test_a_regex_containing_an_escaped_slash_is_not_truncated(self):
        pins = self._pins(r"ok(/scripts\/sol_card\.py/.test(src));" + "\n")
        self.assertEqual([p[2] for p in pins], [r"scripts\/sol_card\.py"])

    def test_a_test_call_on_something_other_than_the_sources_is_ignored(self):
        # `/re/.test(someLocal)` is a fixture assertion, not a source pin, and
        # counting it would inflate the denominator with clauses that have no
        # source to mutate.
        pins = self._pins("ok(/alpha/.test(reasonString));\n")
        self.assertEqual(pins, [])


class SliceResolutionTest(unittest.TestCase):
    """(137). A pin against a `grab`bed slice is the STRONGER form -- a field can
    exist at the writer and be missing at the reader, and matching the whole file
    cannot tell those apart -- so an extractor that only understands `.test(src)`
    is blindest exactly where the suite is strongest. These check that the
    resolution which fixed that does not go the other way and score a pin against
    a haystack it was never written against, which would OVERSTATE coverage in
    the one script that exists to measure it."""

    def test_a_grabbed_slice_is_resolved_to_its_cutting_regex(self):
        js = "const push = grab(/foo[\\s\\S]*?bar/, 'push');\n"
        self.assertEqual(audit._slices(js), {"push": "foo[\\s\\S]*?bar"})

    def test_a_slice_cut_through_a_named_const_is_resolved(self):
        js = "const CUT_RE =\n  /alpha.*omega/;\nconst part = grabGroup(CUT_RE, 'x');\n"
        self.assertEqual(audit._slices(js), {"part": "alpha.*omega"})

    def test_a_match_zero_slice_is_resolved(self):
        js = "const cellProj = summary.match(/cells: [\\s\\S]*?\\}\\)\\),/)[0];\n"
        self.assertEqual(list(audit._slices(js)), ["cellProj"])

    def test_a_pin_against_a_slice_is_extracted_and_names_the_slice(self):
        js = ("const push = grab(/foo/, 'push');\n"
              "ok(/alpha/.test(push), 'why');\n")
        pins = audit._pins(js, audit._slices(js))
        self.assertEqual([(p[2], p[3]) for p in pins], [("alpha", "push")])

    def test_a_slice_is_recut_from_the_mutated_source(self):
        # The reason mutants are applied to the whole lane and the slice re-cut
        # rather than applied to the slice: a mutant landing OUTSIDE the slice
        # can still move the cut.
        self.assertEqual(audit._cut("A.*?Z", "xxAmidZyy"), "AmidZ")
        self.assertEqual(audit._cut("A.*?Z", "no cut here"), "")

    def test_an_uncuttable_slice_is_reported_rather_than_scored(self):
        # Falling back to the whole file would score the pin against a haystack
        # 300x too big, and every pin would look load-bearing.
        self.assertIsNone(audit._cut("(unclosed", "anything"))

    def test_a_scored_slice_leaves_the_unscorable_report(self):
        js = ("const push = grab(/foo/, 'push');\n"
              "ok(/alpha/.test(push), 'why');\n")
        self.assertEqual(audit._unseen(js, {"push"}), [])
        self.assertEqual([k for _, k in audit._unseen(js, set())], ["unreadable slice"])

    def test_a_runtime_built_pattern_stays_unscorable_even_on_a_scored_slice(self):
        # `new RegExp(`\\b${f}:`)` has no literal to compile, so resolving the
        # slice does not make it scorable and it must keep being reported.
        js = ("const push = grab(/foo/, 'push');\n"
              "ok(new RegExp(`\\\\b${f}:`).test(push), 'why');\n")
        self.assertEqual([k for _, k in audit._unseen(js, {"push"})], ["computed pattern"])


class RealSuiteTest(unittest.TestCase):
    """The extractor against the file it exists to read."""

    def setUp(self):
        self.js = (HERE / "test_lane_gates.js").read_text(encoding="utf-8")
        self.pins = audit._pins(self.js)

    def test_the_real_suites_slices_are_resolved_and_then_executed(self):
        """`_slices` must resolve the real suite's cuts, and the suite must spend
        them the strong way.

        The count of regex pins aimed AT a slice is deliberately not asserted.
        Matching a narrowed haystack is stronger than matching the whole file
        (see `_slices`), but weaker again than running the extracted code, and
        this suite does the latter for every slice it takes -- so a pin count
        here would read as coverage while measuring the absence of the better
        form. `SliceResolutionTest` above is what proves the extractor scores
        slice pins rather than skipping them, on input built to contain them.
        """
        slices = audit._slices(self.js)
        self.assertGreaterEqual(len(slices), 5, sorted(slices))
        for name in sorted(slices):
            with self.subTest(slice=name):
                self.assertRegex(
                    self.js, rf"new Function\((?:[^()]|\([^()]*\))*\$\{{{name}\}}",
                    f"`{name}` is cut out of the lane and never evaluated, so nothing "
                    "in this suite runs the code it narrowed to")

    def test_every_real_slice_cuts_something_out_of_the_lane(self):
        lane = (HERE.parent / "kernel_lane.js").read_text(encoding="utf-8")
        for name, pattern in sorted(audit._slices(self.js).items()):
            with self.subTest(slice=name):
                cut = audit._cut(pattern, lane)
                self.assertTrue(cut, f"`{name}` cuts nothing out of kernel_lane.js, so "
                                     "every positive pin against it is trivially false")
                self.assertLess(len(cut), len(lane) // 4,
                                f"`{name}` is most of the file; pinning against it is "
                                "not the narrowing it claims to be")

    def test_it_finds_substantially_more_clauses_than_assertions(self):
        # Not a pinned number -- the point is only that conjunct-bearing
        # assertions are being opened up rather than counted once each.
        assertions = len({p[4] for p in self.pins})
        self.assertGreater(len(self.pins), assertions)

    def test_every_extracted_pattern_compiles(self):
        import re
        for line, _neg, pattern, _which, _owner, _kind in self.pins:
            with self.subTest(line=line):
                re.compile(pattern)

    def test_no_clause_is_attributed_to_a_missing_assertion(self):
        self.assertTrue(all(p[4] >= 0 for p in self.pins))

    def test_both_clause_kinds_are_present_in_the_real_suite(self):
        self.assertEqual({p[5] for p in self.pins}, {"test", "count"})


if __name__ == "__main__":
    unittest.main()
