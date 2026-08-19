#!/usr/bin/env python3
"""Tests for the mutation generator in `mutate_python.py`.

The sweep's output is only meaningful if every mutant it counts as "killed" was
a real change and every mutant it counts as "survived" was too. Both halves fail
silently if the generator emits no-ops: a no-op mutant is always killed-looking
when the tests pass... no, worse -- it *survives*, and lands in the report as a
finding about the tests. So a broken generator manufactures false findings, and
the direction of the error is toward more work rather than less.

Hence: the score stays ungated (55), the transformation is tested.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("mutate_python", HERE / "mutate_python.py")
mut = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mut)


def variants(source: str) -> list[str]:
    """Every mutant of `source`, unparsed."""
    tree = ast.parse(source)
    return [ast.unparse(mut._apply(tree, index, kind))
            for index, _label, kind, _line in mut._mutations(tree)]


def only(source: str) -> str:
    out = variants(source)
    assert len(out) == 1, f"expected exactly one mutant, got {out}"
    return out[0]


class OperatorTest(unittest.TestCase):
    def test_less_than_becomes_less_or_equal(self):
        self.assertEqual(only("a < b"), "a <= b")

    def test_less_or_equal_becomes_less_than(self):
        self.assertEqual(only("a <= b"), "a < b")

    def test_greater_than_becomes_greater_or_equal(self):
        self.assertEqual(only("a > b"), "a >= b")

    def test_equality_becomes_inequality(self):
        self.assertEqual(only("a == b"), "a != b")

    def test_inequality_becomes_equality(self):
        self.assertEqual(only("a != b"), "a == b")

    def test_and_becomes_or(self):
        self.assertEqual(only("a and b"), "a or b")

    def test_or_becomes_and(self):
        self.assertEqual(only("a or b"), "a and b")

    def test_addition_becomes_subtraction(self):
        self.assertEqual(only("a + b"), "a - b")

    def test_multiplication_becomes_division(self):
        self.assertEqual(only("a * b"), "a / b")

    def test_not_is_dropped_leaving_the_operand(self):
        self.assertEqual(only("not a"), "a")

    def test_dropping_not_keeps_a_compound_operand_intact(self):
        # The in-place class/dict swap is the delicate part: the operand may be
        # a whole subtree, not a name.
        self.assertEqual(only("not f(a, b=c)"), "f(a, b=c)")

    def test_true_becomes_false(self):
        self.assertEqual(only("x = True"), "x = False")

    def test_false_becomes_true(self):
        self.assertEqual(only("x = False"), "x = True")

    def test_a_number_is_incremented(self):
        self.assertEqual(only("x = 7"), "x = 8")

    def test_a_float_is_incremented(self):
        self.assertEqual(only("x = 0.5"), "x = 1.5")

    def test_a_bool_is_not_also_treated_as_a_number(self):
        # `isinstance(True, int)` is true in Python; without the explicit guard
        # every boolean would yield both a flip and a `True -> 2`.
        self.assertEqual(len(variants("x = True")), 1)


class IsolationTest(unittest.TestCase):
    def test_each_mutant_differs_from_the_original(self):
        source = "def f(a, b):\n    return a < b and not b > 2\n"
        original = ast.unparse(ast.parse(source))
        for text in variants(source):
            self.assertNotEqual(text, original)

    def test_each_mutant_differs_from_every_other(self):
        source = "def f(a, b):\n    return a < b and not b > 2\n"
        out = variants(source)
        self.assertEqual(len(set(out)), len(out))

    def test_exactly_one_site_changes_per_mutant(self):
        source = "x = a < b\ny = c < d\n"
        out = variants(source)
        self.assertEqual(sorted(out), ["x = a < b\ny = c <= d", "x = a <= b\ny = c < d"])

    def test_applying_a_mutation_does_not_disturb_the_shared_tree(self):
        # `_apply` deep-copies; if it did not, the second mutant would be built
        # on top of the first and the sweep would compound changes.
        tree = ast.parse("x = a < b\ny = c < d\n")
        plans = list(mut._mutations(tree))
        for index, _label, kind, _line in plans:
            mut._apply(tree, index, kind)
        self.assertEqual(ast.unparse(tree), "x = a < b\ny = c < d")

    def test_a_string_constant_is_not_mutated(self):
        self.assertEqual(variants("x = 'hello'"), [])

    def test_none_is_not_mutated(self):
        self.assertEqual(variants("x = None"), [])

    def test_a_chained_comparison_is_left_alone(self):
        # `len(node.ops) == 1` guard: mutating one op of `a < b < c` would need
        # to say which, and the label would not identify it.
        self.assertEqual(variants("x = a < b < c"), [])

    def test_every_mutant_of_every_swept_module_reparses(self):
        # An unparseable mutant is reported as a survivor-shaped line, so a
        # systematic unparse failure would read as a wall of findings.
        for module in mut.MODULES:
            path = mut.SOURCE_ROOT / "scripts" / f"{module}.py"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            plans = list(mut._mutations(tree))
            # Sample: the full cross-product is the sweep's job, not this test's.
            for index, label, kind, line in plans[::37]:
                with self.subTest(module=module, line=line, label=label):
                    ast.parse(ast.unparse(mut._apply(tree, index, kind)))


class FailureParsingTest(unittest.TestCase):
    """The baseline subtraction, which is the half that was wrong.

    The sweep used to ask `returncode == 0`. The mirrored suite has a
    deterministic pre-existing failure, so that question had a constant answer
    and the confirmation pass reported "0 survived the whole suite" whether or
    not it worked. The replacement compares SETS, and a set comparison has its
    own way of being vacuous: if the ids never match between runs -- because the
    assertion message or the subtest parameters are left in them -- the
    subtraction removes nothing and the old bug is back wearing a diff.

    So the parse is pinned against output captured from this pytest, verbatim.
    """

    # Captured from `python3 -m pytest -q --no-header --tb=no -rf`, not
    # hand-written: the verdict token spelling is the thing being checked, and
    # a hand-written sample would only check that I typed it consistently.
    REAL = (
        "FF.                                                     [100%]\n"
        "=========================== short test summary info ============================\n"
        "FAILED test_v.py::T::test_err - RuntimeError: x\n"
        "FAILED test_v.py::T::test_plain - AssertionError: boom with spaces\n"
        "SUBFAILED(i=1) test_v.py::T::test_sub - AssertionError: sub boom\n"
        "SUBFAILED(i=2) test_v.py::T::test_sub - AssertionError: sub boom\n"
        "4 failed, 1 passed in 0.00s\n"
    )

    def test_a_green_run_reports_no_failures(self):
        self.assertEqual(set(), mut.parse_failures("12 passed in 0.4s\n", 0))

    def test_every_verdict_kind_is_recognised(self):
        got = mut.parse_failures(self.REAL, 1)
        self.assertEqual({
            "FAILED test_v.py::T::test_err",
            "FAILED test_v.py::T::test_plain",
            "SUBFAILED test_v.py::T::test_sub",
        }, got)

    def test_subtest_parameters_do_not_leak_into_the_id(self):
        # The two `SUBFAILED(i=N)` lines must collapse to one id, or a mutant
        # that flips which subtest fails would read as a brand-new failure.
        self.assertEqual(
            1, len([i for i in mut.parse_failures(self.REAL, 1)
                    if "test_sub" in i]))

    def test_the_assertion_message_does_not_leak_into_the_id(self):
        # Same output, different message: the ids must be identical, because
        # every mutant produces a different message for the same failure.
        other = self.REAL.replace("boom with spaces", "boom with 999")
        self.assertEqual(mut.parse_failures(self.REAL, 1),
                         mut.parse_failures(other, 1))

    def test_a_crash_with_no_reported_failure_is_named_not_swallowed(self):
        # Exit 2 is a collection error: no FAILED lines at all. Returning an
        # empty set here would classify the mutant as a survivor of a suite
        # that never ran.
        got = mut.parse_failures("INTERNALERROR\n", 2)
        self.assertEqual(1, len(got))
        self.assertIn("exited 2", next(iter(got)))

    def test_a_clean_exit_one_is_not_turned_into_a_synthetic_failure(self):
        # Exit 1 with parsed ids is the ordinary case; exit 1 with none would
        # be odd but is not a crash, and inventing an id would inflate the
        # baseline and defend every mutant.
        self.assertEqual(set(), mut.parse_failures("1 failed\n", 1))

    def test_a_baseline_failure_alone_does_not_count_as_a_kill(self):
        # The bug, stated as an equation. `-` is the whole fix.
        baseline = mut.parse_failures(self.REAL, 1)
        self.assertEqual(set(), mut.parse_failures(self.REAL, 1) - baseline)

    def test_a_new_failure_beside_a_baseline_one_does_count(self):
        baseline = mut.parse_failures(self.REAL, 1)
        mutant = self.REAL + "FAILED test_v.py::T::test_new - AssertionError: z\n"
        self.assertEqual({"FAILED test_v.py::T::test_new"},
                         mut.parse_failures(mutant, 1) - baseline)


class SkipParsingTest(unittest.TestCase):
    """A test that cannot run cannot kill, and the sweep did not say so.

    Six test files in this directory reach above `kernel_workflow` for the
    report or task file a transcribed constant came from. The mirror copied
    only `kernel_workflow`, so all seven of those provenance tests skipped, and
    every constant they are the sole pin for was unkillable for a whole sweep.
    The output did not read as a hole -- it read as `killed only by a test
    outside this module's own file`, a confident and false statement about the
    working tree. Anything pinned *only* by a provenance test would have been
    printed as SURVIVED.
    """

    # Verbatim from `pytest . -q -rs` on the pre-symlink mirror. pytest groups
    # identical reasons behind a `[n]` prefix, which is the part that makes a
    # naive `count SKIPPED lines` undercount.
    REAL = (
        "sssssss.............................                          [100%]\n"
        "=========================== short test summary info ==========\n"
        "SKIPPED [1] test_qd_route_priority.py:52: UNCHECKED: /tmp/pymut/examples/tasks/"
        "dense_bf16_gemm_fused/scripts/task_runner.py is absent, so the 11 harness "
        "shapes in SUITE_SHAPES are transcribed from nothing this run can read.\n"
        "SKIPPED [1] test_qd_robust_stats.py:187: UNCHECKED: run dir absent\n"
        "29 passed, 7 skipped, 70 subtests passed in 0.03s\n"
    )

    def test_a_real_rs_summary_yields_one_entry_per_skipped_test(self):
        self.assertEqual(["test_qd_route_priority.py:52", "test_qd_robust_stats.py:187"],
                         mut.parse_skips(self.REAL))

    def test_the_reason_text_does_not_leak_into_the_name(self):
        # The reasons here contain colons and absolute paths; a split that took
        # the whole tail would make every skip look distinct and the
        # deduplicated list useless.
        for name in mut.parse_skips(self.REAL):
            self.assertNotIn("UNCHECKED", name)
            self.assertNotIn("/", name)

    def test_a_green_run_reports_no_skips(self):
        self.assertEqual([], mut.parse_skips("781 passed in 9.25s\n"))

    def _siblings_the_suite_reaches_for(self):
        """Derived from the test sources, not from a list of six paths.

        A seventh test file that reaches for a new sibling next month gets
        covered without anyone remembering to extend a list (57).
        """
        wanted = set()
        for path in sorted(mut.SOURCE_ROOT.glob("scripts/test_*.py")):
            for rel in re.findall(r'parents\[2\]\s*\)?\s*/\s*\(?\s*"([^"]+)"',
                                  path.read_text(encoding="utf-8")):
                wanted.add(rel.split("/")[0])
        self.assertTrue(wanted, "no test reaches above kernel_workflow; this scan is "
                                "matching nothing and would pass on an empty mirror")
        return sorted(wanted)

    def test_the_mirror_is_declared_to_carry_the_evidence_the_suite_reaches_for(self):
        for name in self._siblings_the_suite_reaches_for():
            with self.subTest(sibling=name):
                self.assertIn(
                    name, mut.EVIDENCE_SIBLINGS,
                    f"tests read {name}/ from above kernel_workflow, but the mirror "
                    "does not carry it, so every test that reads it skips there and "
                    "kills no mutant")

    def test_mirroring_actually_puts_those_siblings_beside_the_mirror(self):
        """The declaration above is a list; this is the behaviour.

        Guarded like `ScratchTest`, and for a sharper reason than symmetry:
        inside the mirror `SOURCE_ROOT == MIRROR`, so calling `_mirror()` here
        copies the tree onto itself -- `shutil` refuses with `SameFileError`, but
        only after the sweep has already written a mutant into that tree. A test
        with a side effect on the thing being measured has no business running
        during the measurement.
        """
        if str(mut.SOURCE_ROOT.resolve()).startswith(str(mut.SCRATCH)):
            self.skipTest(
                "UNCHECKED here: this copy is itself the scratch mirror, so "
                "`_mirror()` would copy the tree onto itself. Run this suite from "
                "the working tree to check that mirroring links the siblings.")
        mut._mirror()
        for name in self._siblings_the_suite_reaches_for():
            if (mut.SOURCE_ROOT.parent / name).is_dir():
                with self.subTest(sibling=name):
                    self.assertTrue((mut.SCRATCH / name).is_dir(),
                                    f"{name}/ exists beside the working tree but not "
                                    "beside the mirror")


class NullMutantTest(unittest.TestCase):
    """The sweep must discount tests that a no-op rewrite already kills.

    Mutants here are whole-file `ast.unparse` output, so any test that greps the
    module's source instead of calling it objects to every mutant equally. Left
    unhandled, that test silently defends the entire file and the sweep reports
    it as coverage. This is the flattering direction, which is the one that does
    not get noticed: `qd_route_priority.py` scored 96/96 killed with 20 of those
    kills belonging to a single quote-style scan in `test_qd_lane_parity.py`.
    """

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="nullmut-"))
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.src = self.dir / "m.py"
        self.original = 'X = "double"   # a comment ast.unparse will drop\n'
        self.src.write_text(self.original, encoding="utf-8")

    def _with_failures(self, failures):
        seen = {}

        def fake(target, stop_early=False):
            seen["text"] = self.src.read_text(encoding="utf-8")
            return set(failures)
        return fake, seen

    def test_a_text_sensitive_test_is_reported_not_swallowed(self):
        fake, seen = self._with_failures({"t.py::greps_the_source"})
        with mock.patch.object(mut, "_failures", fake):
            blind = mut.null_mutant_failures(self.src, self.original, set())
        self.assertEqual({"t.py::greps_the_source"}, blind)
        # And it really did run against a rewrite, not against the original --
        # otherwise the control would report nothing and prove nothing.
        self.assertNotIn("# a comment", seen["text"])
        self.assertIn("'double'", seen["text"])

    def test_the_original_source_is_restored_afterwards(self):
        fake, _ = self._with_failures({"t.py::x"})
        with mock.patch.object(mut, "_failures", fake):
            mut.null_mutant_failures(self.src, self.original, set())
        self.assertEqual(self.original, self.src.read_text(encoding="utf-8"))

    def test_it_is_restored_even_when_the_run_blows_up(self):
        def boom(target, stop_early=False):
            raise RuntimeError("pytest died")
        with mock.patch.object(mut, "_failures", boom):
            with self.assertRaises(RuntimeError):
                mut.null_mutant_failures(self.src, self.original, set())
        self.assertEqual(self.original, self.src.read_text(encoding="utf-8"))

    def test_an_already_failing_test_is_not_blamed_on_the_null_mutant(self):
        # The pre-existing baseline is subtracted, so a suite that is red for
        # unrelated reasons does not turn every test into a text-sensitive one
        # and wipe out the module's real kills.
        fake, _ = self._with_failures({"t.py::already_red", "t.py::greps"})
        with mock.patch.object(mut, "_failures", fake):
            blind = mut.null_mutant_failures(self.src, self.original,
                                             {"t.py::already_red"})
        self.assertEqual({"t.py::greps"}, blind)

    def test_a_timeout_is_a_loud_unknown_rather_than_a_clean_bill(self):
        def slow(target, stop_early=False):
            raise subprocess.TimeoutExpired("pytest", 1)
        with mock.patch.object(mut, "_failures", slow):
            blind = mut.null_mutant_failures(self.src, self.original, set())
        self.assertTrue(blind, "a control that could not run must not return the "
                               "same empty set as a control that ran and found nothing")
        self.assertEqual(self.original, self.src.read_text(encoding="utf-8"))

    def test_the_sweep_folds_the_control_into_its_baseline(self):
        # (55): computing `blind` and then not subtracting it would leave the
        # bug in place behind a new print.
        source = inspect.getsource(mut.sweep)
        self.assertIn("blind = null_mutant_failures(src_path, original, baseline)", source)
        self.assertIn("baseline = baseline | blind", source)
        self.assertIn("return len(plans), confirmed, elsewhere, sorted(blind)", source,
                      "the control's findings must reach the caller, or `main` cannot "
                      "name them and the reader cannot tell a discounted kill from a real one")

    def test_main_prints_the_text_sensitive_tests_by_name(self):
        source = inspect.getsource(mut.main)
        self.assertIn("NULL-MUTANT", source)
        self.assertIn("text-sensitive", source)


class ScratchTest(unittest.TestCase):
    def test_the_sweep_targets_a_scratch_mirror_not_the_working_tree(self):
        # Inside the mirror this copy IS the mirror, so `SOURCE_ROOT == MIRROR`
        # and the property genuinely does not hold -- there is nothing here to
        # protect. Skipping loudly rather than failing matters more than it
        # looks: the sweep's confirmation pass runs this whole suite from the
        # mirror and reads a non-zero exit as "some test objected to the
        # mutant". A test that always fails there makes every mutant look
        # defended, which is a false all-clear -- the same shape as (55), and
        # in the direction that reports less work than exists.
        if str(mut.SOURCE_ROOT.resolve()).startswith(str(mut.SCRATCH)):
            self.skipTest(
                "UNCHECKED here: this copy of the tree is itself the scratch "
                "mirror, so the isolation property is about the original "
                "checkout and cannot be observed from inside. Run this suite "
                "from the working tree to check it.")
        self.assertNotEqual(mut.MIRROR.resolve(), mut.SOURCE_ROOT.resolve())
        self.assertFalse(str(mut.SOURCE_ROOT.resolve()).startswith(str(mut.SCRATCH)))

    def test_every_named_module_exists(self):
        for module in mut.MODULES:
            with self.subTest(module=module):
                self.assertTrue((mut.SOURCE_ROOT / "scripts" / f"{module}.py").exists())


if __name__ == "__main__":
    unittest.main()
