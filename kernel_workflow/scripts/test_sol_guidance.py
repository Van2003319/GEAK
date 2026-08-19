#!/usr/bin/env python3
"""Tests for sol_guidance.py.

Written from the module's own point of view: what it must refuse, and the two
places where an obvious implementation would be wrong in a way that costs the
lane real budget --

  * a bare `t < 0.9 * t_SOL` screen condemns a fast kernel when the bound is
    the stale thing (four routes on this suite), so the verdict split is tested
    against both directions;
  * a geometric aggregate attenuates a single-route win by the route count and
    zeroes on one missing route, so the arithmetic aggregate is pinned by a
    test that a geometric one fails.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sol_guidance as sg
from qd_sol_card import SOLCardError

# The lane's measured gfx942 table, as qd_sol_card records it.
TABLE = {32 << 20: 1.42e12, 64 << 20: 2.11e12, 86 << 20: 2.30e12,
         128 << 20: 2.68e12, 256 << 20: 2.92e12, 1024 << 20: 3.94e12}
PEAK_BF16 = 668e12
WITNESS = ("exp/opt_bf16_20260814/shapeceil.py, 8192^3 bf16 matmul, machine H")


def ceilings_for(m, n, k, **kw):
    footprint = sg.gemm_semantic_traffic_bytes(m, n, k)
    kw.setdefault("compute_witness_by", WITNESS)
    return sg.resolve_ceilings(footprint_bytes=footprint, peak_flops=PEAK_BF16,
                               bandwidth_table=TABLE, **kw)


class SemanticsTest(unittest.TestCase):
    def test_work_is_two_flops_per_multiply_add(self):
        self.assertEqual(sg.gemm_semantic_work_flops(2, 4096, 4096), 2 * 2 * 4096 * 4096)

    def test_traffic_counts_each_operand_once_and_the_output_once(self):
        m, n, k = 8, 11008, 4096
        self.assertEqual(sg.gemm_semantic_traffic_bytes(m, n, k),
                         2 * (m * k + k * n + m * n))

    def test_traffic_is_dtype_width_scaled(self):
        args = (16, 4096, 4096)
        self.assertEqual(sg.gemm_semantic_traffic_bytes(*args, dtype="fp32"),
                         2 * sg.gemm_semantic_traffic_bytes(*args, dtype="bf16"))

    def test_unknown_dtype_is_refused_rather_than_defaulted(self):
        with self.assertRaises(SOLCardError):
            sg.gemm_semantic_traffic_bytes(8, 8, 8, dtype="fp4_imaginary")

    def test_non_positive_shape_is_refused(self):
        for bad in ((0, 8, 8), (8, -1, 8), (8, 8, 0)):
            with self.assertRaises(SOLCardError):
                sg.gemm_semantic_work_flops(*bad)


class CeilingTest(unittest.TestCase):
    def test_exactly_one_bandwidth_shape_is_accepted(self):
        with self.assertRaises(SOLCardError):
            sg.resolve_ceilings(footprint_bytes=1 << 25, peak_flops=PEAK_BF16,
                                bandwidth_table=TABLE, bandwidth_scalar=3e12)
        with self.assertRaises(SOLCardError):
            sg.resolve_ceilings(footprint_bytes=1 << 25, peak_flops=PEAK_BF16)

    def test_a_witness_must_name_what_achieved_the_rate(self):
        with self.assertRaises(SOLCardError):
            sg.resolve_ceilings(footprint_bytes=1 << 25, peak_flops=PEAK_BF16,
                                bandwidth_table=TABLE,
                                bandwidth_witness_bytes_s=2.2e12)

    def test_witness_raises_the_effective_bound_but_never_lowers_it(self):
        c = ceilings_for(2, 4096, 4096, bandwidth_witness_bytes_s=2.15e12,
                         bandwidth_witness_by="r2_d1 NT B-stream, decode_m2_square")
        self.assertGreater(c.effective_bandwidth_bytes_s, c.bandwidth_bytes_s)
        self.assertEqual(c.effective_bandwidth_bytes_s, 2.15e12)
        low = ceilings_for(2, 4096, 4096, bandwidth_witness_bytes_s=0.5e12,
                           bandwidth_witness_by="a slower probe")
        self.assertEqual(low.effective_bandwidth_bytes_s, low.bandwidth_bytes_s)


class SolTest(unittest.TestCase):
    def test_sol_is_the_max_of_the_two_floors(self):
        r = sg.analyze_route(route="decode_m2_square", m=2, n=4096, k=4096,
                             ceilings=ceilings_for(2, 4096, 4096))
        self.assertAlmostEqual(r.sol_s, max(r.compute_floor_s, r.memory_floor_s))
        self.assertEqual(r.regime, "memory_bound")

    def test_a_large_square_shape_is_compute_bound(self):
        r = sg.analyze_route(route="prefill_m2048_square", m=2048, n=4096, k=4096,
                             ceilings=ceilings_for(2048, 4096, 4096))
        self.assertEqual(r.regime, "compute_bound")

    def test_the_bound_is_computable_with_no_measurement(self):
        r = sg.analyze_route(route="r", m=512, n=11008, k=4096,
                             ceilings=ceilings_for(512, 11008, 4096))
        self.assertGreater(r.sol_s, 0)
        self.assertIsNone(r.gap)
        self.assertEqual(r.verdict, "no_measurement")

    def test_gap_and_headroom_agree_with_each_other(self):
        r = sg.analyze_route(route="r", m=256, n=4096, k=11008,
                             ceilings=ceilings_for(256, 4096, 11008),
                             measured_s=123.8e-6)
        self.assertAlmostEqual(r.gap, r.measured_s / r.sol_s)
        self.assertAlmostEqual(r.remaining_headroom, 1 - 1 / r.gap)
        self.assertAlmostEqual(r.achievement, 1 / r.gap)


class VerdictTest(unittest.TestCase):
    """The split that a bare threshold gets wrong."""

    def test_slower_than_the_bound_is_ok(self):
        r = sg.analyze_route(route="r", m=256, n=4096, k=11008,
                             ceilings=ceilings_for(256, 4096, 11008),
                             measured_s=123.8e-6)
        self.assertEqual(r.verdict, "ok")
        self.assertGreater(r.gap, 1.0)

    def test_just_under_the_bound_is_a_modelling_residue_not_a_finding(self):
        c = ceilings_for(2, 4096, 4096)
        sol = sg.analyze_route(route="r", m=2, n=4096, k=4096, ceilings=c).sol_s
        r = sg.analyze_route(route="r", m=2, n=4096, k=4096, ceilings=c,
                             measured_s=sol * 0.95)
        self.assertEqual(r.verdict, "near_sol")

    def test_beating_an_unwitnessed_bandwidth_bound_blames_the_bound(self):
        # decode_m2_square as actually measured on this lane: 15.6 us against a
        # 23.6 us memory floor from a table taken before non-temporal loads.
        c = ceilings_for(2, 4096, 4096)
        r = sg.analyze_route(route="decode_m2_square", m=2, n=4096, k=4096,
                             ceilings=c, measured_s=15.6e-6)
        self.assertLess(r.gap, 1.0 - sg.GAMING_MARGIN)
        self.assertEqual(r.verdict, "ceiling_contradicted")

    def test_beating_a_witnessed_bandwidth_bound_blames_the_kernel(self):
        c = ceilings_for(2, 4096, 4096, bandwidth_witness_bytes_s=2.15e12,
                         bandwidth_witness_by="r2_d1 NT B-stream, decode_m2_square")
        # A time far below even the witnessed rate has no physical story left.
        r = sg.analyze_route(route="decode_m2_square", m=2, n=4096, k=4096,
                             ceilings=c, measured_s=1.0e-6)
        self.assertEqual(r.verdict, "gaming_suspected")

    def test_the_11_56x_aliasing_report_would_have_been_caught(self):
        # The quarantined run reported ~6-9 us on every route regardless of FLOPs.
        c = ceilings_for(2048, 4096, 4096, bandwidth_witness_bytes_s=3.0e12,
                         bandwidth_witness_by="cold streaming read, 64 MB footprint")
        r = sg.analyze_route(route="prefill_m2048_square", m=2048, n=4096, k=4096,
                             ceilings=c, measured_s=8.0e-6)
        self.assertEqual(r.verdict, "gaming_suspected")

    def test_a_compute_bound_route_is_judged_on_the_compute_witness(self):
        # No compute witness: beating the bound indicts the nameplate, not the kernel.
        c = sg.resolve_ceilings(
            footprint_bytes=sg.gemm_semantic_traffic_bytes(2048, 4096, 4096),
            peak_flops=1307e12, bandwidth_table=TABLE)
        r = sg.analyze_route(route="prefill_m2048_square", m=2048, n=4096, k=4096,
                             ceilings=c, measured_s=8.0e-6)
        self.assertEqual(r.regime, "compute_bound")
        self.assertEqual(r.verdict, "ceiling_contradicted")

    def test_an_unwitnessed_compute_ceiling_is_flagged_in_the_notes(self):
        c = sg.resolve_ceilings(
            footprint_bytes=sg.gemm_semantic_traffic_bytes(2048, 4096, 4096),
            peak_flops=1307e12, bandwidth_table=TABLE)
        r = sg.analyze_route(route="r", m=2048, n=4096, k=4096, ceilings=c)
        self.assertTrue(any("no witness" in n for n in r.notes))


class AggregateTest(unittest.TestCase):
    def _routes(self, achievements):
        out = []
        for i, a in enumerate(achievements):
            c = ceilings_for(2048, 4096, 4096)
            r = sg.analyze_route(route="r%d" % i, m=2048, n=4096, k=4096, ceilings=c)
            if a is not None:
                r = sg._place_measurement(r, r.sol_s / a)
            out.append(r)
        return out

    def test_a_single_route_win_is_not_divided_by_the_route_count(self):
        """The property a geometric aggregate does not have.

        Eleven routes; one improves its achievement by 10%. Arithmetic moves the
        aggregate by ~1/11 of 10% in absolute terms, and crucially the SAME
        amount whether the other ten sit at 0.9 or at 0.1 -- so a route win can
        never be attenuated into invisibility by where the rest happen to be.
        """
        base_hi = self._routes([0.9] * 11)
        bump_hi = self._routes([0.99] + [0.9] * 10)
        base_lo = self._routes([0.1] * 11)
        bump_lo = self._routes([0.11] + [0.1] * 10)
        d_hi = sg.weighted_achievement(bump_hi) - sg.weighted_achievement(base_hi)
        d_lo = sg.weighted_achievement(bump_lo) - sg.weighted_achievement(base_lo)
        self.assertAlmostEqual(d_hi, 0.09 / 11, places=6)
        self.assertAlmostEqual(d_lo, 0.01 / 11, places=6)

    def test_an_unmeasured_route_scores_zero_rather_than_vanishing(self):
        with_gap = sg.weighted_achievement(self._routes([0.9, 0.9, None]))
        without = sg.weighted_achievement(self._routes([0.9, 0.9]))
        self.assertLess(with_gap, without)
        self.assertAlmostEqual(with_gap, 1.8 / 3)

    def test_weights_are_honoured_and_must_sum_positive(self):
        routes = self._routes([0.2, 0.8])
        heavy_first = sg.weighted_achievement(routes, {"r0": 9.0, "r1": 1.0})
        heavy_last = sg.weighted_achievement(routes, {"r0": 1.0, "r1": 9.0})
        self.assertLess(heavy_first, heavy_last)
        with self.assertRaises(SOLCardError):
            sg.weighted_achievement(routes, {"r0": 0.0, "r1": 0.0})

    def test_median_ignores_unmeasured_routes(self):
        self.assertAlmostEqual(sg.route_achievement(self._routes([0.4, 0.6, None])), 0.5)


class TriageTest(unittest.TestCase):
    def test_the_exponent_only_engages_past_the_pivot(self):
        self.assertAlmostEqual(sg.gap_exponent(5.0), 1.0)
        self.assertAlmostEqual(sg.gap_exponent(1.0), 1.0)
        self.assertAlmostEqual(sg.gap_exponent(50.0), 2.0)

    def test_far_from_sol_promotes_the_ambitious_hypothesis(self):
        hyps = [{"id": "safe", "estimated_speedup": 1.05, "impl_risk": 1.0, "perf_risk": 1.0},
                {"id": "bold", "estimated_speedup": 2.0, "impl_risk": 3.0, "perf_risk": 2.0}]
        near = sg.triage(hyps, gap=1.2)
        far = sg.triage(hyps, gap=500.0)
        self.assertEqual(near[0]["id"], "safe")
        self.assertEqual(far[0]["id"], "bold")

    def test_risk_divides_and_must_be_a_multiplier(self):
        a = sg.hypothesis_roi(estimated_speedup=2.0, gap=5.0, impl_risk=1.0, perf_risk=1.0)
        b = sg.hypothesis_roi(estimated_speedup=2.0, gap=5.0, impl_risk=2.0, perf_risk=1.0)
        self.assertAlmostEqual(a / b, 2.0)
        with self.assertRaises(SOLCardError):
            sg.hypothesis_roi(estimated_speedup=2.0, gap=5.0, impl_risk=0.5, perf_risk=1.0)

    def test_ties_keep_input_order(self):
        hyps = [{"id": "first", "estimated_speedup": 1.5},
                {"id": "second", "estimated_speedup": 1.5}]
        self.assertEqual([h["id"] for h in sg.triage(hyps, gap=2.0)], ["first", "second"])


class EligibilityTest(unittest.TestCase):
    def test_near_sol_and_ahead_of_the_oracle_stops_spending(self):
        e = sg.route_eligibility(gap=1.2, ahead_of_reference=True,
                                 rounds_without_progress=0, sol_gap_epsilon=0.5)
        self.assertFalse(e.eligible)
        self.assertIn("ceiling", e.reason)

    def test_a_route_still_losing_to_the_oracle_is_never_retired(self):
        e = sg.route_eligibility(gap=1.01, ahead_of_reference=False,
                                 rounds_without_progress=99, sol_gap_epsilon=0.5)
        self.assertTrue(e.eligible)

    def test_no_progress_retires_a_route_that_is_already_ahead(self):
        e = sg.route_eligibility(gap=9.0, ahead_of_reference=True,
                                 rounds_without_progress=2, no_progress_window=2)
        self.assertFalse(e.eligible)
        self.assertIn("no progress", e.reason)

    def test_an_open_gap_with_recent_progress_stays_eligible(self):
        e = sg.route_eligibility(gap=3.0, ahead_of_reference=True,
                                 rounds_without_progress=1, sol_gap_epsilon=1.0,
                                 no_progress_window=2)
        self.assertTrue(e.eligible)

    def test_degenerate_knobs_are_refused(self):
        with self.assertRaises(SOLCardError):
            sg.route_eligibility(gap=2.0, ahead_of_reference=True,
                                 rounds_without_progress=0, sol_gap_epsilon=-0.1)
        with self.assertRaises(SOLCardError):
            sg.route_eligibility(gap=2.0, ahead_of_reference=True,
                                 rounds_without_progress=0, no_progress_window=0)


class ReportTest(unittest.TestCase):
    def test_the_screen_fails_only_on_gaming_never_on_a_stale_ceiling(self):
        stale = sg.analyze_route(route="decode_m2_square", m=2, n=4096, k=4096,
                                 ceilings=ceilings_for(2, 4096, 4096),
                                 measured_s=15.6e-6)
        rep = sg.report([stale])
        self.assertEqual(rep["ceiling_contradicted"], ["decode_m2_square"])
        self.assertTrue(rep["screen_passed"])

        c = ceilings_for(2, 4096, 4096, bandwidth_witness_bytes_s=2.15e12,
                         bandwidth_witness_by="r2_d1 NT B-stream")
        cheat = sg.analyze_route(route="decode_m2_square", m=2, n=4096, k=4096,
                                 ceilings=c, measured_s=1.0e-6)
        self.assertFalse(sg.report([cheat])["screen_passed"])

    def test_receipt_is_canonical_json(self):
        spec = {
            "peak_flops": PEAK_BF16,
            "bandwidth_table": {str(k): v for k, v in TABLE.items()},
            "compute_witness_by": WITNESS,
            "routes": [{"route": "decode_m32_down", "m": 32, "n": 4096, "k": 11008,
                        "measured_s": 38.4e-6}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "spec.json")
            with open(path, "w") as fh:
                json.dump(spec, fh, sort_keys=True)
            proc = subprocess.run(
                [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "sol_guidance.py"), path, "--json"],
                capture_output=True, text=True)
        self.assertIn(proc.returncode, (0, 3), proc.stderr)
        doc = json.loads(proc.stdout)
        self.assertEqual(doc["schema"], sg.SCHEMA)
        # Canonical: re-serialising the parsed document reproduces the bytes.
        self.assertEqual(json.dumps(doc, indent=1, sort_keys=True).strip(),
                         proc.stdout.strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
