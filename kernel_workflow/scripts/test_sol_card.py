#!/usr/bin/env python3
"""GPU-free tests for sol_card.py."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("sol_card.py")
SPEC = importlib.util.spec_from_file_location("sol_card", SCRIPT)
assert SPEC and SPEC.loader
QSC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QSC
SPEC.loader.exec_module(QSC)


class PostSelectionContractTest(unittest.TestCase):
    def test_post_selection_false_is_refused(self):
        with self.assertRaises(QSC.SOLCardError):
            QSC.build_sol_card(post_selection=False, achieved_flops=1.0, achieved_bytes=1.0,
                                elapsed_s=1.0, dtype="bf16")

    def test_post_selection_omitted_is_refused(self):
        with self.assertRaises(TypeError):
            QSC.build_sol_card(achieved_flops=1.0, achieved_bytes=1.0, elapsed_s=1.0, dtype="bf16")

    def test_post_selection_truthy_non_true_is_refused(self):
        # Only the literal True is accepted -- "yes"/1 must not smuggle past the gate.
        with self.assertRaises(QSC.SOLCardError):
            QSC.build_sol_card(post_selection=1, achieved_flops=1.0, achieved_bytes=1.0,
                                elapsed_s=1.0, dtype="bf16")

    def test_no_batch_form_exists(self):
        # There is no code path that accepts more than one candidate's measurements at once:
        # only scalar keyword args, never a list/sequence of candidates to compare.
        import inspect
        sig = inspect.signature(QSC.build_sol_card)
        self.assertNotIn("candidates", sig.parameters)
        self.assertNotIn("results", sig.parameters)
        for name in ("achieved_flops", "achieved_bytes", "elapsed_s"):
            self.assertIn(name, sig.parameters)
            self.assertEqual(inspect.Parameter.KEYWORD_ONLY, sig.parameters[name].kind)


class FormulaTest(unittest.TestCase):
    def build(self, **overrides):
        kwargs = dict(post_selection=True, achieved_flops=1e12, achieved_bytes=1e9,
                      elapsed_s=0.01, dtype="bf16")
        kwargs.update(overrides)
        return QSC.build_sol_card(**kwargs)

    def test_schema_and_flags(self):
        card = self.build()
        self.assertEqual(QSC.SCHEMA, card["schema"])
        self.assertIs(True, card["post_selection"])
        self.assertEqual("gfx90a", card["arch"])

    def test_achieved_rates_are_flops_over_time(self):
        card = self.build(achieved_flops=2e12, elapsed_s=0.5)
        self.assertAlmostEqual(4e12, card["achieved_flop_rate"])

    def test_arithmetic_intensity_is_flops_over_bytes(self):
        card = self.build(achieved_flops=1e12, achieved_bytes=1e9)
        self.assertAlmostEqual(1000.0, card["arithmetic_intensity"])

    def test_zero_bytes_is_infinite_intensity_and_compute_bound(self):
        card = self.build(achieved_bytes=0.0)
        self.assertEqual(float("inf"), card["arithmetic_intensity"])
        self.assertEqual("compute_bound", card["regime"])
        self.assertAlmostEqual(card["peak_flops"], card["roofline_ceiling_flops"])

    def test_low_intensity_is_memory_bound(self):
        card = self.build(achieved_flops=1e9, achieved_bytes=1e9, elapsed_s=1.0)  # intensity == 1
        self.assertEqual("memory_bound", card["regime"])
        self.assertLess(card["roofline_ceiling_flops"], card["peak_flops"])

    def test_high_intensity_is_compute_bound_and_ceiling_is_peak(self):
        card = self.build(achieved_flops=1e12, achieved_bytes=1.0, elapsed_s=1e-3)
        self.assertEqual("compute_bound", card["regime"])
        self.assertAlmostEqual(card["peak_flops"], card["roofline_ceiling_flops"])

    def test_roofline_ceiling_never_exceeds_peak(self):
        for bytes_ in (1.0, 1e6, 1e9, 1e15):
            card = self.build(achieved_bytes=bytes_)
            self.assertLessEqual(card["roofline_ceiling_flops"], card["peak_flops"] * (1 + 1e-9))

    def test_pct_of_roofline_is_consistent(self):
        card = self.build()
        expected = card["achieved_flop_rate"] / card["roofline_ceiling_flops"]
        self.assertAlmostEqual(expected, card["pct_of_roofline"])

    def test_sol_floor_gap_and_headroom_use_time_domain_formulas(self):
        card = self.build()
        self.assertAlmostEqual(card["compute_floor_s"], 1e12 / card["peak_flops"])
        self.assertAlmostEqual(card["memory_floor_s"], 1e9 / card["peak_bandwidth_bytes_s"])
        self.assertAlmostEqual(card["sol_s"], max(card["compute_floor_s"], card["memory_floor_s"]))
        self.assertAlmostEqual(card["sol_gap"], card["elapsed_s"] / card["sol_s"])
        self.assertAlmostEqual(card["remaining_headroom"], 1.0 - 1.0 / card["sol_gap"])

    def test_measurement_above_modeled_roof_is_visible_not_clamped(self):
        # Noise or miscalibration can put a measurement above the modeled roof.
        # Keep that diagnostic signal: gap < 1 and negative modeled headroom.
        card = self.build(achieved_flops=1e15, achieved_bytes=0.0, elapsed_s=1e-6)
        self.assertGreater(card["sol_gap"], 0.0)
        self.assertLess(card["sol_gap"], 1.0)
        self.assertLess(card["remaining_headroom"], 0.0)

    def test_unsupported_arch_is_rejected(self):
        # gfx942 used to sit here; it is modeled as of (28), so this needs an
        # arch nobody measured to still be testing the refusal.
        with self.assertRaises(QSC.SOLCardError):
            self.build(arch="gfx1100")

    def test_unsupported_dtype_is_rejected(self):
        with self.assertRaises(QSC.SOLCardError):
            self.build(dtype="fp8")

    def test_negative_or_zero_work_inputs_are_rejected(self):
        with self.assertRaises(QSC.SOLCardError):
            self.build(achieved_flops=-1.0)
        with self.assertRaises(QSC.SOLCardError):
            self.build(elapsed_s=0.0)
        with self.assertRaises(QSC.SOLCardError):
            self.build(achieved_flops=0.0, achieved_bytes=0.0)

    def test_every_supported_dtype_builds_a_card(self):
        for dtype in QSC.SUPPORTED_DTYPES:
            card = self.build(dtype=dtype)
            self.assertEqual(dtype, card["dtype"])


class CalibrationTest(unittest.TestCase):
    def test_uncalibrated_uses_reference_numbers_and_labels_them_as_such(self):
        card = QSC.build_sol_card(post_selection=True, achieved_flops=1.0, achieved_bytes=1.0,
                                   elapsed_s=1.0, dtype="bf16")
        self.assertIn("reference", card["source"])

    def test_calibration_without_measured_flag_is_rejected(self):
        with self.assertRaises(QSC.SOLCardError):
            QSC.build_sol_card(post_selection=True, achieved_flops=1.0, achieved_bytes=1.0,
                                elapsed_s=1.0, dtype="bf16",
                                calibration={"peak_bandwidth_bytes_s": 1.0e12})

    def test_measured_calibration_overrides_peaks(self):
        card = QSC.build_sol_card(
            post_selection=True, achieved_flops=1.0, achieved_bytes=1.0, elapsed_s=1.0, dtype="bf16",
            calibration={"measured": True, "peak_flops": {"bf16": 100.0}, "peak_bandwidth_bytes_s": 10.0,
                         "source": "profiled on box rack-3"})
        self.assertEqual(100.0, card["peak_flops"])
        self.assertEqual(10.0, card["peak_bandwidth_bytes_s"])
        self.assertEqual("profiled on box rack-3", card["source"])

    def test_calibration_can_unlock_a_dtype_with_no_reference_peak(self):
        card = QSC.build_sol_card(
            post_selection=True, achieved_flops=1.0, achieved_bytes=1.0, elapsed_s=1.0, dtype="int8",
            calibration={"measured": True, "peak_flops": {"int8": 381.0e12}})
        self.assertEqual(381.0e12, card["peak_flops"])


class ValidateSolCardTest(unittest.TestCase):
    def valid_card(self):
        return QSC.build_sol_card(post_selection=True, achieved_flops=1e12, achieved_bytes=1e9,
                                   elapsed_s=0.01, dtype="bf16")

    def test_a_freshly_built_card_validates_clean(self):
        self.assertEqual([], QSC.validate_sol_card(self.valid_card()))

    def test_missing_post_selection_flag_is_flagged(self):
        card = dict(self.valid_card())
        card["post_selection"] = False
        problems = QSC.validate_sol_card(card)
        self.assertTrue(any("post_selection" in p for p in problems))

    def test_wrong_schema_is_flagged(self):
        card = dict(self.valid_card())
        card["schema"] = "not-the-schema"
        self.assertTrue(any("schema" in p for p in QSC.validate_sol_card(card)))

    def test_ceiling_above_peak_is_flagged(self):
        card = dict(self.valid_card())
        card["roofline_ceiling_flops"] = card["peak_flops"] * 2
        self.assertTrue(any("cannot exceed peak_flops" in p for p in QSC.validate_sol_card(card)))

    def test_inconsistent_pct_of_roofline_is_flagged(self):
        card = dict(self.valid_card())
        card["pct_of_roofline"] = card["pct_of_roofline"] + 1.0
        self.assertTrue(any("inconsistent" in p for p in QSC.validate_sol_card(card)))

    def test_bad_regime_is_flagged(self):
        card = dict(self.valid_card())
        card["regime"] = "somewhat_bound"
        self.assertTrue(any("regime" in p for p in QSC.validate_sol_card(card)))

    def test_never_raises_on_garbage_input(self):
        problems = QSC.validate_sol_card({})
        self.assertIsInstance(problems, list)
        self.assertGreater(len(problems), 0)

    def test_negative_sol_gap_is_flagged(self):
        card = dict(self.valid_card())
        card["sol_gap"] = -0.1
        self.assertTrue(any("sol_gap" in p for p in QSC.validate_sol_card(card)))

    def test_a_gap_below_one_is_flagged_even_though_the_card_is_self_consistent(self):
        """Finding (59). sol_s is a lower bound on time, so elapsed >= sol and
        sol_gap >= 1. A card below 1 says the kernel outran the roofline, which
        is a broken model -- most plausibly the peak of the wrong arch, which
        finding (55) showed the lane could select silently.

        The point of the test is the *self-consistency*: every consistency check
        in the validator passes on this card. sol_gap really does equal
        elapsed/sol, and remaining_headroom really does equal 1 - 1/gap. It is
        internally perfect and describes something impossible, so only an
        explicit physical bound catches it.
        """
        card = dict(self.valid_card())
        card["elapsed_s"] = card["sol_s"] / 2
        card["sol_gap"] = card["elapsed_s"] / card["sol_s"]
        card["remaining_headroom"] = 1.0 - 1.0 / card["sol_gap"]
        problems = QSC.validate_sol_card(card)
        self.assertTrue(any("below 1" in p for p in problems), problems)
        # ...and it is caught by the bound, not by a consistency complaint.
        self.assertFalse(any("inconsistent" in p for p in problems), problems)

    def test_a_kernel_exactly_at_the_speed_of_light_is_accepted(self):
        # The bound is >= 1, not > 1: gap == 1 is the ideal kernel, headroom 0.
        card = dict(self.valid_card())
        card["elapsed_s"] = card["sol_s"]
        card["sol_gap"] = 1.0
        card["remaining_headroom"] = 0.0
        self.assertEqual([], [p for p in QSC.validate_sol_card(card)
                              if "sol_gap" in p or "remaining_headroom" in p])

    def test_inconsistent_floor_gap_and_headroom_are_flagged(self):
        card = dict(self.valid_card())
        card["sol_s"] *= 2
        self.assertTrue(any("sol_s" in p for p in QSC.validate_sol_card(card)))
        card = dict(self.valid_card())
        card["sol_gap"] *= 2
        self.assertTrue(any("sol_gap" in p for p in QSC.validate_sol_card(card)))
        card = dict(self.valid_card())
        card["remaining_headroom"] = 0.123
        self.assertTrue(any("remaining_headroom" in p for p in QSC.validate_sol_card(card)))


class TheMeasuredGfx942CeilingIsDataAndTheScalarBandwidthModelCannotHoldIt(unittest.TestCase):
    """Finding (19)/(28): gfx942 is now modeled, and only through the table.

    The earlier version of this class pinned the opposite -- that recording the
    measured constant had NOT enabled gfx942 -- which was true for five stages
    and is the thing (28) closed. What must be pinned now is narrower and more
    useful: gfx942 resolves *without* a caller-supplied calibration, and it can
    only ever resolve through the footprint table, because the card carries no
    scalar to fall back to. A scalar appearing on this card later would silently
    reintroduce the 2.8x mis-ranking the table exists to prevent.
    """

    CEILINGS = QSC.MEASURED_GFX942_CEILINGS

    def test_gfx942_resolves_from_its_own_card_with_no_calibration(self):
        card = QSC.build_sol_card(post_selection=True, achieved_flops=1e12,
                                   achieved_bytes=128 << 20, elapsed_s=0.01,
                                   dtype="bf16", arch="gfx942")
        self.assertEqual([], QSC.validate_sol_card(card))
        self.assertEqual("footprint_table", card["bandwidth_ceiling_basis"])
        # 128 MB is a measured point, so it must come back exactly, not
        # interpolated off its neighbours.
        self.assertEqual(2.68e12, card["peak_bandwidth_bytes_s"])
        self.assertEqual(668e12, card["peak_flops"])
        self.assertEqual("gfx942", card["arch"])

    def test_the_gfx942_card_carries_no_scalar_ceiling_to_fall_back_to(self):
        self.assertNotIn("peak_bandwidth_bytes_s", self.CEILINGS)
        # ...so a caller who omits footprint_bytes is refused rather than
        # served one of the two wrong answers.
        with self.assertRaises(QSC.SOLCardError) as ctx:
            QSC.build_sol_card(post_selection=True, achieved_flops=1e12, achieved_bytes=0.0,
                                elapsed_s=0.01, dtype="bf16", arch="gfx942")
        self.assertIn("footprint", str(ctx.exception))

    def test_an_unmeasured_dtype_on_gfx942_is_refused_not_borrowed_from_gfx90a(self):
        # fp32 has a peak on the gfx90a reference card. Reading across is the
        # exact failure the per-arch table is meant to prevent.
        with self.assertRaises(QSC.SOLCardError) as ctx:
            QSC.build_sol_card(post_selection=True, achieved_flops=1e12,
                                achieved_bytes=128 << 20, elapsed_s=0.01,
                                dtype="fp32", arch="gfx942")
        self.assertIn("fp32", str(ctx.exception))
        self.assertIn("gfx942", str(ctx.exception))

    def test_an_unmodeled_arch_is_still_refused(self):
        with self.assertRaises(QSC.SOLCardError) as ctx:
            QSC.build_sol_card(post_selection=True, achieved_flops=1e12, achieved_bytes=1e9,
                                elapsed_s=0.01, dtype="bf16", arch="gfx1100")
        self.assertIn("gfx1100", str(ctx.exception))
        card = dict(QSC.build_sol_card(post_selection=True, achieved_flops=1e12,
                                        achieved_bytes=1e9, elapsed_s=0.01, dtype="bf16"))
        card["arch"] = "gfx1100"
        self.assertTrue(any("arch" in p for p in QSC.validate_sol_card(card)))

    def test_the_constant_attests_it_was_measured_and_says_where(self):
        # The whole point of the `measured` flag is that nobody may hand
        # `_resolve_peaks` a number they did not take off the box.
        self.assertIs(True, self.CEILINGS["measured"])
        self.assertIn("machine H", self.CEILINGS["source"])
        self.assertIn("bwceil.py", self.CEILINGS["source"])

    def test_every_measured_number_is_below_the_paper_peak_it_replaces(self):
        # 1307 TFLOP/s and 5.3 TB/s are the vendor figures for this part.
        self.assertLess(self.CEILINGS["peak_flops"]["bf16"], 1307e12)
        self.assertGreater(self.CEILINGS["peak_flops"]["bf16"], 0.0)
        for footprint, bw in self.CEILINGS["peak_bandwidth_bytes_s_by_footprint"].items():
            with self.subTest(footprint=footprint):
                self.assertGreater(bw, 0.0)
                self.assertLess(bw, 5.3e12)

    def test_bandwidth_rises_monotonically_with_footprint(self):
        items = sorted(self.CEILINGS["peak_bandwidth_bytes_s_by_footprint"].items())
        bws = [bw for _, bw in items]
        self.assertEqual(bws, sorted(bws))
        self.assertEqual(len(bws), len(set(bws)))

    def test_the_spread_is_wide_enough_that_no_scalar_can_stand_in(self):
        # This is the finding. A single `peak_bandwidth_bytes_s` has to be
        # wrong by this factor at one end of the suite or the other, which is
        # why the fix is a footprint-indexed ceiling and hence a schema change.
        bws = list(self.CEILINGS["peak_bandwidth_bytes_s_by_footprint"].values())
        self.assertGreater(max(bws) / min(bws), 2.5)

    def test_the_footprint_keys_cover_the_suites_actual_operand_sizes(self):
        # The eleven-shape BF16 suite's dominant operands: 4096x4096,
        # 11008x4096 and 8192x8192 bf16 -- 32, 86 and 128 MB.
        keys = set(self.CEILINGS["peak_bandwidth_bytes_s_by_footprint"])
        for nbytes in (4096 * 4096 * 2, 11008 * 4096 * 2, 8192 * 8192 * 2):
            with self.subTest(nbytes=nbytes):
                self.assertIn(nbytes, keys)

    def test_a_scalar_calibration_still_works_and_still_demands_measured(self):
        # The existing single-number path is untouched; only its adequacy on
        # this box is in question, not its behaviour.
        cal = {"measured": True, "peak_bandwidth_bytes_s": 2.30e12, "source": "unit test"}
        card = QSC.build_sol_card(post_selection=True, achieved_flops=1e12, achieved_bytes=1e9,
                                   elapsed_s=0.01, dtype="bf16", calibration=cal)
        self.assertEqual(2.30e12, card["peak_bandwidth_bytes_s"])
        self.assertEqual([], QSC.validate_sol_card(card))
        with self.assertRaises(QSC.SOLCardError):
            QSC.build_sol_card(post_selection=True, achieved_flops=1e12, achieved_bytes=1e9,
                                elapsed_s=0.01, dtype="bf16",
                                calibration={"peak_bandwidth_bytes_s": 2.30e12})

    def test_the_two_scalar_choices_disagree_about_whether_a_route_is_finished(self):
        # decode_m2_square: M=2, N=K=4096, 0.02412 ms measured on machine H. At
        # M=2 the A and C operands are ~2 KB each and the traffic is the 32 MB
        # B matrix alone. Score that against the 32 MB ceiling -- the footprint
        # it actually has -- and the route is done; against the 1 GB ceiling it
        # looks like it has two thirds of its performance still on the table.
        # Same kernel, same measurement, opposite verdict: a scalar cannot rank
        # a suite whose footprints span the curve.
        achieved_bytes = 4096.0 * 4096 * 2
        elapsed_s = 0.02412e-3
        gaps = {}
        for label, bw in (("32MB", 1.42e12), ("1GB", 3.94e12)):
            card = QSC.build_sol_card(
                post_selection=True, achieved_flops=2.0 * 2 * 4096 * 4096,
                achieved_bytes=achieved_bytes, elapsed_s=elapsed_s, dtype="bf16",
                calibration={"measured": True, "peak_bandwidth_bytes_s": bw,
                             "source": "finding (19) unit test"})
            self.assertEqual("memory_bound", card["regime"])
            gaps[label] = card["sol_gap"]
        self.assertLess(gaps["32MB"], 1.1)          # finished: 1.02x achievable
        self.assertGreater(gaps["1GB"], 2.5)        # apparently wide open
        self.assertGreater(gaps["1GB"] / gaps["32MB"], 2.5)


class CliTest(unittest.TestCase):
    def test_cli_is_deterministic(self):
        argv = [sys.executable, str(SCRIPT), "--achieved-flops", "1e12", "--achieved-bytes", "1e9",
                "--elapsed-s", "0.01", "--dtype", "bf16", "--arch", "gfx942"]
        run1 = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        run2 = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        self.assertEqual(run1.stdout, run2.stdout)
        card = json.loads(run1.stdout)
        self.assertEqual([], QSC.validate_sol_card(card))

    def test_cli_reports_errors_on_stderr_and_fails_closed(self):
        argv = [sys.executable, str(SCRIPT), "--achieved-flops", "1e12", "--achieved-bytes", "1e9",
                "--elapsed-s", "0.01", "--dtype", "fp8", "--arch", "gfx942"]
        run = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(2, run.returncode)
        self.assertEqual("", run.stdout)
        self.assertIn("fp8", run.stderr)


class FootprintIndexedCeilingTest(unittest.TestCase):
    """Schema v1 -> v2: the bandwidth ceiling moves 2.8x across one suite.

    The v1 hole was recorded in the module for several stages: a scalar
    `peak_bandwidth_bytes_s` cannot express a ceiling that runs 1.42 TB/s at
    32 MB and 3.94 at 1024 MB, so every calibration of it mis-ranked one end of
    the eleven-shape suite or the other.
    """

    TABLE = QSC.MEASURED_GFX942_CEILINGS["peak_bandwidth_bytes_s_by_footprint"]
    CAL = {"measured": True, "peak_flops": {"bf16": 668e12},
           "peak_bandwidth_bytes_s_by_footprint": TABLE,
           "source": "test"}

    def test_every_measured_point_comes_back_exactly(self):
        # Interpolation that does not reproduce its own data is a model, and
        # this table is too small to earn one.
        for size, rate in self.TABLE.items():
            got = QSC.bandwidth_ceiling(float(size), self.TABLE)
            self.assertAlmostEqual(rate, got["bytes_s"], delta=rate * 1e-9, msg=f"at {size}")

    def test_an_interior_footprint_stays_between_its_neighbours(self):
        got = QSC.bandwidth_ceiling(96 << 20, self.TABLE)
        self.assertLess(self.TABLE[86 << 20], got["bytes_s"])
        self.assertGreater(self.TABLE[128 << 20], got["bytes_s"])
        self.assertEqual("measured_interpolated", got["confidence"])
        self.assertFalse(got["extrapolated"])

    def test_a_chord_in_bytes_would_contradict_the_measurements_it_spans(self):
        # The actual reason for log-log, stated as the failure it avoids: a
        # straight line in bytes from 32 MB to 1024 MB passes *below* the
        # measured 256 MB point by a third, so it is not merely a different
        # interpolation -- it disagrees with data it is drawn through.
        lo_x, hi_x = 32 << 20, 1024 << 20
        lo_y, hi_y = self.TABLE[lo_x], self.TABLE[hi_x]
        at = float(256 << 20)
        chord = lo_y + (hi_y - lo_y) * (at - lo_x) / (hi_x - lo_x)
        measured = self.TABLE[256 << 20]
        self.assertLess(chord, measured * 0.75)
        # Log-log reproduces it, because 256 MB is a table point.
        self.assertAlmostEqual(measured, QSC.bandwidth_ceiling(at, self.TABLE)["bytes_s"],
                               delta=measured * 1e-9)

    def test_outside_the_measured_range_is_clamped_and_flagged_low(self):
        for footprint in (1 << 20, 4096 << 20):
            got = QSC.bandwidth_ceiling(footprint, self.TABLE)
            self.assertEqual("low", got["confidence"], msg=f"at {footprint}")
            self.assertTrue(got["extrapolated"], msg=f"at {footprint}")
        # And clamped, not extrapolated along the trend -- a tiny decode shape
        # gets the smallest measured rate, never something invented below it.
        self.assertEqual(self.TABLE[32 << 20],
                         QSC.bandwidth_ceiling(1 << 20, self.TABLE)["bytes_s"])

    def test_a_scalar_and_a_table_together_are_refused(self):
        # Two mutually exclusive claims about the same quantity; taking one
        # silently is how a stale number survives (finding 25e).
        with self.assertRaises(QSC.SOLCardError):
            QSC.build_sol_card(post_selection=True, achieved_flops=1e12, achieved_bytes=1e9,
                                elapsed_s=0.01, dtype="bf16",
                                calibration=dict(self.CAL, peak_bandwidth_bytes_s=3e12))

    def test_a_card_built_from_the_table_records_how_it_resolved(self):
        card = QSC.build_sol_card(post_selection=True, achieved_flops=1e12,
                                   achieved_bytes=float(96 << 20), elapsed_s=0.01,
                                   dtype="bf16", calibration=self.CAL)
        self.assertEqual(QSC.SCHEMA, card["schema"])
        self.assertEqual("footprint_table", card["bandwidth_ceiling_basis"])
        self.assertEqual(float(96 << 20), card["footprint_bytes"])
        self.assertEqual([float(86 << 20), float(128 << 20)],
                         card["bandwidth_ceiling_bracket"])
        self.assertEqual([], QSC.validate_sol_card(card))

    def test_footprint_defaults_to_the_bytes_actually_moved(self):
        # Right for a single-pass GEMM, which is the suite this was built for.
        card = QSC.build_sol_card(post_selection=True, achieved_flops=1e12,
                                   achieved_bytes=float(96 << 20), elapsed_s=0.01,
                                   dtype="bf16", calibration=self.CAL)
        explicit = QSC.build_sol_card(post_selection=True, achieved_flops=1e12,
                                       achieved_bytes=float(96 << 20), elapsed_s=0.01,
                                       dtype="bf16", calibration=self.CAL,
                                       footprint_bytes=float(96 << 20))
        self.assertEqual(explicit["peak_bandwidth_bytes_s"], card["peak_bandwidth_bytes_s"])

    def test_the_two_ends_of_the_suite_get_different_ceilings(self):
        # The whole point. A 32 MB decode shape and a 1 GB prefill shape must
        # not be scored against one number.
        small = QSC.bandwidth_ceiling(32 << 20, self.TABLE)["bytes_s"]
        large = QSC.bandwidth_ceiling(1024 << 20, self.TABLE)["bytes_s"]
        self.assertGreater(large / small, 2.5)

    def test_a_scalar_card_says_so_rather_than_claiming_a_footprint(self):
        card = QSC.build_sol_card(post_selection=True, achieved_flops=1e12,
                                   achieved_bytes=1e9, elapsed_s=0.01, dtype="bf16")
        self.assertEqual("scalar", card["bandwidth_ceiling_basis"])
        self.assertIsNone(card["footprint_bytes"])
        self.assertEqual([], QSC.validate_sol_card(card))


class CeilingTableProvenanceTest(unittest.TestCase):
    """(44)'s third corner for the bandwidth table -- as far as it goes.

    `MEASURED_GFX942_CEILINGS` names its source: `exp/opt_bf16_20260814/bwceil.py`
    and `shapeceil.py`, machine H, harness regime. Unlike the shipped-latency and
    noise-floor tables, **the numbers themselves have no recorded output on disk**
    -- only the scripts that produced them, and machine H is gone, so they cannot
    be re-derived here. That half stays a labelled hole rather than a checked
    claim, and it is stated here so a reader of a green suite does not conclude
    otherwise (54).

    What *is* checkable is the half that drifts silently: the six footprints the
    table is indexed by are a transcription of the six `bwceil.py` measured, and
    a key added or moved without a measurement behind it produces an
    "interpolated" answer that was never interpolated from anything.
    """

    MEASURER = (Path(__file__).resolve().parents[2]
                / "exp/opt_bf16_20260814/bwceil.py")

    def setUp(self):
        if not self.MEASURER.exists():
            self.skipTest(
                f"UNCHECKED: {self.MEASURER} is absent, so the six footprints "
                "indexing peak_bandwidth_bytes_s_by_footprint are transcribed "
                "from a script nothing in this run can read. Which points are "
                "measured and which are interpolated is unverified here.")
        import ast
        tree = ast.parse(self.MEASURER.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "FOOTPRINTS" for t in node.targets):
                # The entries are written as arithmetic (`4096 * 4096 * 2`,
                # `64 << 20`), which `literal_eval` refuses. Evaluated with no
                # builtins and no names, so nothing but arithmetic can run.
                self.measured = {int(eval(ast.unparse(elt.elts[1]),  # noqa: S307
                                          {"__builtins__": {}}, {}))
                                 for elt in node.value.elts}
                return
        raise AssertionError(f"{self.MEASURER} no longer defines FOOTPRINTS")

    def test_the_table_is_indexed_by_exactly_the_footprints_that_were_measured(self):
        table = QSC.MEASURED_GFX942_CEILINGS["peak_bandwidth_bytes_s_by_footprint"]
        self.assertEqual(sorted(self.measured), sorted(int(k) for k in table))

    def test_the_three_suite_operand_footprints_are_measured_points_not_brackets(self):
        # 4096x4096, 11008x4096 and 8192x8192 bf16 are the dominant operands of
        # the eleven routes. If one of these stopped being a table key it would
        # be answered by interpolation between two brackets, which is a weaker
        # claim than the docstring makes for it.
        table = QSC.MEASURED_GFX942_CEILINGS["peak_bandwidth_bytes_s_by_footprint"]
        for n, k in ((4096, 4096), (11008, 4096), (8192, 8192)):
            with self.subTest(operand=(n, k)):
                self.assertIn(n * k * 2, {int(key) for key in table})


class SchemaCompatibilityTest(unittest.TestCase):
    def base_card(self):
        return QSC.build_sol_card(post_selection=True, achieved_flops=1e12,
                                   achieved_bytes=1e9, elapsed_s=0.01, dtype="bf16")

    OLD_FIELDS = ("bandwidth_ceiling_basis", "bandwidth_ceiling_confidence",
                  "footprint_bytes", "bandwidth_ceiling_bracket",
                  "bandwidth_ceiling_extrapolated",
                  "compute_ceiling_witnessed", "compute_ceiling_attainment",
                  "compute_ceiling_witness")

    def test_an_old_v1_card_still_validates(self):
        # Archive elites written before the bump are old, not corrupt.
        card = dict(self.base_card(), schema=QSC.SCHEMA_V1)
        for key in self.OLD_FIELDS:
            card.pop(key)
        self.assertEqual([], QSC.validate_sol_card(card))

    def test_an_old_v2_card_still_validates(self):
        card = dict(self.base_card(), schema=QSC.SCHEMA_V2)
        for key in ("compute_ceiling_witnessed", "compute_ceiling_attainment",
                    "compute_ceiling_witness"):
            card.pop(key)
        self.assertEqual([], QSC.validate_sol_card(card))

    def test_a_v2_label_on_a_v3_card_is_refused(self):
        # The witness fields carry an attainability claim a v2 consumer does
        # not know to read -- (92): the version string is the only thing that
        # makes that consumer fail loudly instead of silently agreeing.
        card = dict(self.base_card(), schema=QSC.SCHEMA_V2)
        problems = QSC.validate_sol_card(card)
        self.assertTrue(any("compute_ceiling_witnessed" in p for p in problems), problems)

    def test_a_v3_card_missing_the_witness_fields_is_refused(self):
        card = self.base_card()
        card.pop("compute_ceiling_witnessed")
        self.assertTrue(any("compute_ceiling_witnessed" in p
                            for p in QSC.validate_sol_card(card)))

    def test_a_v1_label_on_a_v2_card_is_refused(self):
        # It would be read under the v1 assumption that the ceiling is scalar.
        card = dict(self.base_card(), schema=QSC.SCHEMA_V1)
        problems = QSC.validate_sol_card(card)
        self.assertTrue(problems)
        self.assertTrue(any("bandwidth_ceiling_basis" in p for p in problems), problems)

    def test_a_v2_card_missing_the_new_fields_is_refused(self):
        card = self.base_card()
        card.pop("bandwidth_ceiling_basis")
        self.assertTrue(any("bandwidth_ceiling_basis" in p
                            for p in QSC.validate_sol_card(card)))

    def test_a_footprint_table_card_must_carry_its_footprint(self):
        card = dict(self.base_card(), bandwidth_ceiling_basis="footprint_table")
        self.assertTrue(any("footprint_bytes" in p for p in QSC.validate_sol_card(card)))


class ComputeCeilingWitnessTest(unittest.TestCase):
    """(89) item 2: a ceiling used as a denominator must have an achiever.

    `measured=True` attests where a number came from. It does not attest that
    anything can reach it: `rocminfo` reports 1307 TFLOP/s bf16 on this box
    with impeccable provenance, and scored against that every route in the
    suite -- and rocBLAS with it -- sits 3-7x from SOL, which ranks nothing.
    The separator implemented here is a witness, not a threshold, because any
    threshold would itself be the unevidenced number this module refuses.
    """

    def card(self, **kw):
        args = dict(post_selection=True, achieved_flops=9.2e10, achieved_bytes=1.2e8,
                    elapsed_s=2.7e-4, dtype="bf16", arch="gfx942")
        args.update(kw)
        return QSC.build_sol_card(**args)

    def test_the_measured_gfx942_bf16_peak_carries_its_achiever(self):
        card = self.card()
        self.assertIs(True, card["compute_ceiling_witnessed"])
        self.assertAlmostEqual(1.0, card["compute_ceiling_attainment"])
        self.assertIn("shapeceil", card["compute_ceiling_witness"])
        self.assertEqual([], QSC.validate_sol_card(card))

    def test_the_gfx90a_reference_peaks_are_unwitnessed_and_say_so(self):
        # Datasheet peaks. Nothing has been observed to reach them on that
        # part, and the card must not imply otherwise by omission.
        card = self.card(arch="gfx90a", achieved_flops=1e12, achieved_bytes=1e9, elapsed_s=0.01)
        self.assertIs(False, card["compute_ceiling_witnessed"])
        self.assertIsNone(card["compute_ceiling_attainment"])
        self.assertEqual([], QSC.validate_sol_card(card))

    def test_a_dtype_the_witness_table_does_not_cover_is_unwitnessed(self):
        # The witness is per dtype, not per card: one measured rate does not
        # vouch for a peak nobody ran at that precision.
        cal = {"peak_flops": {"bf16": 5e14, "fp8": 1e15}, "peak_bandwidth_bytes_s": 3e12,
               "measured": True,
               "attainment": {"bf16": {"achieved_flops": 4e14, "by": "run 1670"}}}
        self.assertIs(True, self.card(calibration=cal)["compute_ceiling_witnessed"])
        card = self.card(dtype="fp8", achieved_flops=1e12, achieved_bytes=1e9,
                         elapsed_s=0.01, calibration=cal)
        self.assertIs(False, card["compute_ceiling_witnessed"])

    def test_a_calibration_replaces_the_witness_it_replaces_the_peak_of(self):
        # The dangerous combination is the card's witness beside a caller's
        # peak: the attainment ratio would be computed from two ceilings.
        card = self.card(calibration={"peak_flops": {"bf16": 5e14},
                                      "peak_bandwidth_bytes_s": 3e12, "measured": True})
        self.assertIs(False, card["compute_ceiling_witnessed"])
        self.assertIsNone(card["compute_ceiling_attainment"])

    def test_a_calibration_may_supply_its_own_witness(self):
        card = self.card(calibration={
            "peak_flops": {"bf16": 5e14}, "peak_bandwidth_bytes_s": 3e12, "measured": True,
            "attainment": {"bf16": {"achieved_flops": 4e14, "by": "run 1670, machine L"}}})
        self.assertIs(True, card["compute_ceiling_witnessed"])
        self.assertAlmostEqual(0.8, card["compute_ceiling_attainment"])
        self.assertEqual("run 1670, machine L", card["compute_ceiling_witness"])

    def test_an_unattributed_witness_is_refused(self):
        with self.assertRaises(QSC.SOLCardError) as cm:
            self.card(calibration={
                "peak_flops": {"bf16": 5e14}, "peak_bandwidth_bytes_s": 3e12, "measured": True,
                "attainment": {"bf16": {"achieved_flops": 4e14, "by": "   "}}})
        self.assertIn("by", str(cm.exception))

    def test_a_nonpositive_achieved_rate_is_refused(self):
        with self.assertRaises(QSC.SOLCardError):
            self.card(calibration={
                "peak_flops": {"bf16": 5e14}, "peak_bandwidth_bytes_s": 3e12, "measured": True,
                "attainment": {"bf16": {"achieved_flops": 0, "by": "nothing"}}})

    def test_a_witness_that_outran_the_ceiling_condemns_the_ceiling(self):
        # Every card built from such a peak reports sol_gap < 1 and is refused
        # one case at a time for a reason that points at the kernel. The defect
        # is the ceiling, and it is named here instead.
        with self.assertRaises(QSC.SOLCardError) as cm:
            self.card(calibration={
                "peak_flops": {"bf16": 4e14}, "peak_bandwidth_bytes_s": 3e12, "measured": True,
                "attainment": {"bf16": {"achieved_flops": 5e14, "by": "run 1670"}}})
        self.assertIn("not a ceiling", str(cm.exception))

    def test_a_ratio_without_a_witness_is_refused_by_the_validator(self):
        card = dict(self.card(arch="gfx90a", achieved_flops=1e12, achieved_bytes=1e9,
                              elapsed_s=0.01), compute_ceiling_attainment=0.5)
        self.assertTrue(any("compute_ceiling_attainment" in p
                            for p in QSC.validate_sol_card(card)))

    def test_a_witness_flag_without_a_name_is_refused_by_the_validator(self):
        card = dict(self.card(), compute_ceiling_witness="")
        self.assertTrue(any("witness" in p for p in QSC.validate_sol_card(card)))

    def test_an_attainment_above_one_is_refused_by_the_validator(self):
        card = dict(self.card(), compute_ceiling_attainment=1.4)
        self.assertTrue(any("compute_ceiling_attainment" in p
                            for p in QSC.validate_sol_card(card)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
