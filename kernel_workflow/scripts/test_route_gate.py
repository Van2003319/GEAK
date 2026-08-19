#!/usr/bin/env python3
"""Tests for route_gate.py.

Two of these are the reason the module exists rather than a tightened threshold:

  * `test_a_real_route_mechanism_that_the_geomean_gate_rejects` builds the exact
    situation the old gate mishandled -- one route +7%, ten flat -- and pins
    that the geomean the old gate read is under its own 2% threshold while the
    per-route gate accepts.
  * `test_a_route_with_no_band_is_refused_not_defaulted` pins the refusal, since
    a default band would be wrong by up to ~7x across this suite's routes.

`NoDeviceScreenTest` pins an ABSENCE: there is no cross-device screen, on
purpose. Absences do not fail on their own, so without it a later reader of the
calibration would have no way to tell "removed by decision" from "never built".
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import route_gate as rg

# Per-route bands measured on this lane (24 repeats of the unchanged round-7 tree).
BANDS = {
    "decode_m2_square": 0.1034, "decode_m8_up": 0.0552, "decode_m16_square": 0.0807,
    "decode_m32_down": 0.0298, "decode_m64_square": 0.0427, "decode_m96_up": 0.0318,
    "prefill_m128_square": 0.0393, "prefill_m256_down": 0.1644,
    "prefill_m512_up": 0.0402, "prefill_m1024_down": 0.0395,
    "prefill_m2048_square": 0.0268,
}
ROUTES = sorted(BANDS)


def rows(times, speedups=None):
    out = []
    for i, route in enumerate(ROUTES):
        row = {"name": route, "optimized_ms": times[route]}
        if speedups:
            row["speedup"] = speedups[route]
        out.append(row)
        del i
    return out


def flat(base=0.100):
    return {route: base for route in ROUTES}


class RowReadingTest(unittest.TestCase):
    def test_candidate_ms_is_accepted_as_an_alias(self):
        d = rg.decide(candidate_per_case=[{"name": "a", "candidate_ms": 0.09}],
                      incumbent_per_case=[{"name": "a", "optimized_ms": 0.10}],
                      bands={"a": 0.01})
        self.assertTrue(d.accepted)

    def test_a_row_without_a_time_is_refused(self):
        with self.assertRaises(rg.RouteGateError):
            rg.decide(candidate_per_case=[{"name": "a"}],
                      incumbent_per_case=[{"name": "a", "optimized_ms": 0.1}],
                      bands={"a": 0.01})

    def test_a_duplicate_route_is_refused(self):
        with self.assertRaises(rg.RouteGateError):
            rg.decide(candidate_per_case=[{"name": "a", "optimized_ms": 0.1},
                                          {"name": "a", "optimized_ms": 0.2}],
                      incumbent_per_case=[{"name": "a", "optimized_ms": 0.1}],
                      bands={"a": 0.01})

    def test_an_empty_per_case_is_refused(self):
        with self.assertRaises(rg.RouteGateError):
            rg.decide(candidate_per_case=[], incumbent_per_case=[
                {"name": "a", "optimized_ms": 0.1}], bands={"a": 0.01})


class BandTest(unittest.TestCase):
    def test_a_route_with_no_band_is_refused_not_defaulted(self):
        inc = rows(flat())
        cand = rows(flat(0.09))
        partial = {r: BANDS[r] for r in ROUTES[:-1]}
        with self.assertRaises(rg.RouteGateError) as ctx:
            rg.decide(candidate_per_case=cand, incumbent_per_case=inc, bands=partial)
        self.assertIn(ROUTES[-1], str(ctx.exception))

    def test_bands_come_from_repeats_of_one_unchanged_tree(self):
        reps = [{"test_cases": [{"name": "a", "optimized_ms": ms}]}
                for ms in (0.100, 0.104, 0.096)]
        bands = rg.bands_from_repeats(reps)
        self.assertAlmostEqual(bands["a"], (0.104 - 0.096) / 0.100, places=6)

    def test_two_repeats_cannot_define_a_spread(self):
        reps = [{"test_cases": [{"name": "a", "optimized_ms": 0.1}]}] * 2
        with self.assertRaises(rg.RouteGateError):
            rg.bands_from_repeats(reps)

    def test_a_route_missing_from_some_repeats_is_refused(self):
        reps = [{"test_cases": [{"name": "a", "optimized_ms": 0.1},
                                {"name": "b", "optimized_ms": 0.2}]},
                {"test_cases": [{"name": "a", "optimized_ms": 0.1}]},
                {"test_cases": [{"name": "a", "optimized_ms": 0.1}]}]
        with self.assertRaises(rg.RouteGateError):
            rg.bands_from_repeats(reps)


class DecisionTest(unittest.TestCase):
    def test_all_flat_is_refused(self):
        d = rg.decide(candidate_per_case=rows(flat()), incumbent_per_case=rows(flat()),
                      bands=BANDS)
        self.assertFalse(d.accepted)
        self.assertIn("all flat", d.reason)

    def test_a_win_inside_the_band_is_not_a_win(self):
        t = flat()
        t["prefill_m256_down"] = 0.100 * (1 - 0.10)     # -10%, band is 16.44%
        d = rg.decide(candidate_per_case=rows(t), incumbent_per_case=rows(flat()),
                      bands=BANDS)
        self.assertFalse(d.accepted)
        self.assertEqual(d.improved, [])

    def test_the_same_win_on_a_quiet_route_is_a_win(self):
        t = flat()
        t["prefill_m2048_square"] = 0.100 * (1 - 0.10)   # -10%, band is 2.68%
        d = rg.decide(candidate_per_case=rows(t), incumbent_per_case=rows(flat()),
                      bands=BANDS)
        self.assertTrue(d.accepted)
        self.assertEqual(d.improved, ["prefill_m2048_square"])

    def test_one_regression_past_its_band_vetoes_any_number_of_wins(self):
        t = flat()
        t["prefill_m2048_square"] = 0.100 * (1 - 0.20)
        t["decode_m96_up"] = 0.100 * (1 + 0.10)          # band 3.18%
        d = rg.decide(candidate_per_case=rows(t), incumbent_per_case=rows(flat()),
                      bands=BANDS)
        self.assertFalse(d.accepted)
        self.assertEqual(d.regressed, ["decode_m96_up"])
        self.assertIn("regressed", d.reason)

    def test_an_unmeasured_incumbent_route_blocks_the_non_regression_claim(self):
        t = flat()
        t["prefill_m2048_square"] = 0.05
        cand = [r for r in rows(t) if r["name"] != "decode_m2_square"]
        d = rg.decide(candidate_per_case=cand, incumbent_per_case=rows(flat()),
                      bands=BANDS)
        self.assertFalse(d.accepted)
        self.assertIn("decode_m2_square", d.reason)

    def test_a_win_off_the_declared_target_is_not_the_claimed_mechanism(self):
        t = flat()
        t["decode_m32_down"] = 0.100 * (1 - 0.10)
        d = rg.decide(candidate_per_case=rows(t), incumbent_per_case=rows(flat()),
                      bands=BANDS, target_routes=["prefill_m1024_down"])
        self.assertFalse(d.accepted)
        self.assertIn("not the claimed mechanism", d.reason)

    def test_a_win_on_the_declared_target_is_accepted(self):
        t = flat()
        t["prefill_m1024_down"] = 0.100 * (1 - 0.10)
        d = rg.decide(candidate_per_case=rows(t), incumbent_per_case=rows(flat()),
                      bands=BANDS, target_routes=["prefill_m1024_down"])
        self.assertTrue(d.accepted)

    def test_an_empty_target_list_narrows_nothing(self):
        """`[]` means "this direction declared no target", not "narrow to nothing".

        This is the one input on which this function and its JS twin in
        kernel_lane.js disagreed. Under `target_routes is not None` an empty list
        narrowed to nothing and refused, while the twin -- the gate that actually
        decides whether a round commits -- treats the same value as "no narrowing"
        and accepts. A parity test between the two is what neither side had, and
        the shape is reachable: engineer.md only says "omit when the direction
        names none", and an agent filling a declared array field it has nothing to
        say about returns [] as readily as it omits the key.
        """
        t = flat()
        t["decode_m32_down"] = 0.100 * (1 - 0.10)
        for empty in ([], None):
            d = rg.decide(candidate_per_case=rows(t), incumbent_per_case=rows(flat()),
                          bands=BANDS, target_routes=empty)
            self.assertTrue(d.accepted, f"target_routes={empty!r} must not narrow")
            self.assertEqual(d.improved, ["decode_m32_down"])


class NoDeviceScreenTest(unittest.TestCase):
    """The gate takes no device argument, by decision of the run owner.

    Pinned rather than merely absent: the exposure it would have screened is
    real (the same unchanged tree measures 1.5-3% differently between
    invocations), so someone reading the calibration later could reasonably
    assume a screen exists. This test is where they find out it does not.
    """

    def test_decide_accepts_no_device_arguments(self):
        for kw in ({"device": "hip:5"}, {"incumbent_device": "hip:7"}):
            with self.assertRaises(TypeError):
                rg.decide(candidate_per_case=rows(flat()),
                          incumbent_per_case=rows(flat()), bands=BANDS, **kw)

    def test_the_receipt_carries_no_device_fields(self):
        t = flat()
        t["prefill_m2048_square"] = 0.100 * (1 - 0.10)
        rec = rg.receipt(rg.decide(candidate_per_case=rows(t),
                                   incumbent_per_case=rows(flat()), bands=BANDS))
        self.assertNotIn("device", rec)
        self.assertNotIn("incumbent_device", rec)


class GeomeanContrastTest(unittest.TestCase):
    def test_a_real_route_mechanism_that_the_geomean_gate_rejects(self):
        """One route +7%, ten flat: the case the old gate could not see.

        This is the tech-lead's own arithmetic ("an eleven-case geomean dilutes a
        single-route win about elevenfold") turned into a test.
        """
        t = flat()
        t["prefill_m1024_down"] = 0.100 / 1.07
        speedups = {r: 1.0 for r in ROUTES}
        speedups["prefill_m1024_down"] = 1.07
        cand = rows(t, speedups)

        geo = rg.suite_geomean(cand)
        self.assertLess(geo, 1.02)          # under the old MIN_IMPROVE default
        self.assertGreater(geo, 1.005)      # and it is a real, positive signal

        d = rg.decide(candidate_per_case=cand, incumbent_per_case=rows(flat()),
                      bands=BANDS, target_routes=["prefill_m1024_down"])
        self.assertTrue(d.accepted)
        self.assertAlmostEqual(d.suite_geomean_speedup, geo)

    def test_the_geomean_is_carried_but_never_decides(self):
        """A candidate with a high geomean and a banded regression is still refused."""
        t = flat()
        for route in ROUTES:
            t[route] = 0.100 / 1.30
        t["decode_m96_up"] = 0.100 * 1.10
        speedups = {r: 1.30 for r in ROUTES}
        speedups["decode_m96_up"] = 1 / 1.10
        cand = rows(t, speedups)
        d = rg.decide(candidate_per_case=cand, incumbent_per_case=rows(flat()), bands=BANDS)
        self.assertGreater(d.suite_geomean_speedup, 1.2)
        self.assertFalse(d.accepted)


class ReceiptTest(unittest.TestCase):
    def test_cli_receipt_is_canonical_and_exit_code_carries_the_verdict(self):
        t = flat()
        t["prefill_m2048_square"] = 0.100 * (1 - 0.10)
        spec = {"candidate_per_case": rows(t), "incumbent_per_case": rows(flat()),
                "bands": BANDS}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "spec.json")
            with open(path, "w") as fh:
                json.dump(spec, fh, sort_keys=True)
            here = os.path.dirname(os.path.abspath(__file__))
            proc = subprocess.run([sys.executable, os.path.join(here, "route_gate.py"),
                                   path, "--json"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        doc = json.loads(proc.stdout)
        self.assertTrue(doc["accepted"])
        self.assertEqual(doc["schema"], rg.SCHEMA)
        self.assertEqual(json.dumps(doc, indent=1, sort_keys=True).strip(),
                         proc.stdout.strip())

    def test_a_refusal_exits_nonzero(self):
        spec = {"candidate_per_case": rows(flat()), "incumbent_per_case": rows(flat()),
                "bands": BANDS}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "spec.json")
            with open(path, "w") as fh:
                json.dump(spec, fh, sort_keys=True)
            here = os.path.dirname(os.path.abspath(__file__))
            proc = subprocess.run([sys.executable, os.path.join(here, "route_gate.py"), path],
                                  capture_output=True, text=True)
        self.assertEqual(proc.returncode, 3)
        self.assertIn("REFUSE", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
