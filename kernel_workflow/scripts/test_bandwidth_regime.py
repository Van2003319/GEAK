#!/usr/bin/env python3
"""Tests for `bandwidth_regime.py`.

The load-bearing test is `test_it_rejects_the_impossible_column`. (142): a
scanner must be shown catching the real thing, in real data, before a clean
result from it means anything. The `seed_candidate_ms_median` column in the
tw035 run is genuinely impossible -- flat at ~0.0085 ms across shapes whose FLOP
counts differ 1376x -- so this checker cannot go blind without that test going
red. A synthetic fixture would keep passing after the arithmetic stopped
matching anything real data looks like.

The second load-bearing test is `test_the_equal_cu_control_resolves`. The
scaling experiment's headline number is only readable if the instrument can
resolve the case where the answer is known: two routes at the SAME CU count must
land on top of each other. If that control drifted, the interesting number would
be noise.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import bandwidth_regime as BR  # noqa: E402

LIVE = REPO_ROOT / "examples" / "tasks" / "dense_bf16_gemm_fused" / "src" / "custom_gemm.hip"
TW035 = REPO_ROOT / "exp" / "qd_v2_bf16_b4_20260816_tw035" / "baseline_timing.json"
EPOCH_O = REPO_ROOT / "exp" / "qd_v2_bf16_r16_20260816_epochO" / "baseline_timing.json"


def timing_file(values: dict[str, float], field: str = "ms") -> Path:
    path = Path(tempfile.mkdtemp(prefix="br_")) / "t.json"
    path.write_text(json.dumps(
        {"test_cases": [{"name": k, field: v} for k, v in values.items()]}),
        encoding="utf-8")
    return path


def all_cases(value_for) -> dict[str, float]:
    from kernel_launch_facts import CASES
    return {name: value_for(name, m, n, k) for name, m, n, k in CASES}


class LiveSightingTest(unittest.TestCase):
    def test_the_inputs_are_still_there(self):
        for p in (LIVE, TW035, EPOCH_O):
            with self.subTest(path=p):
                self.assertTrue(p.is_file(), f"{p} is gone; tests below are vacuous")

    def test_it_rejects_the_impossible_column(self):
        """(142). Real data, real defect, caught."""
        self.assertEqual(3, BR.main([str(LIVE), "--timing", str(TW035),
                                     "--field", "seed_candidate_ms_median"]))

    def test_the_rejection_names_both_kinds_of_impossibility(self):
        t = BR.load_timings(TW035, "seed_candidate_ms_median")
        bad = BR.check_physics(BR.analyse(LIVE, t))
        self.assertTrue(any("TFLOP/s" in b for b in bad), bad)
        self.assertTrue(any("TB/s" in b for b in bad), bad)

    def test_the_flatness_signature_fires_on_it(self):
        t = BR.load_timings(TW035, "seed_candidate_ms_median")
        self.assertIsNotNone(BR.flatness(BR.analyse(LIVE, t)))

    def test_a_real_column_survives_and_is_not_called_flat(self):
        t = BR.load_timings(TW035, "latency_ms")
        table = BR.analyse(LIVE, t)
        self.assertEqual([], BR.check_physics(table))
        self.assertIsNone(BR.flatness(table))
        self.assertEqual(0, BR.main([str(LIVE), "--timing", str(TW035),
                                     "--field", "latency_ms"]))


class PhysicsTest(unittest.TestCase):
    def test_a_latency_just_over_the_flop_peak_is_caught(self):
        """2*M*N*K FLOPs in less than peak allows is not a fast kernel."""
        def over(name, m, n, k):
            return (2 * m * n * k / (BR.BF16_PEAK_TFLOPS * 1e12) / 1.05) * 1e3
        path = timing_file(all_cases(over))
        bad = BR.check_physics(BR.analyse(LIVE, BR.load_timings(path, "ms")))
        self.assertTrue(any("TFLOP/s" in b for b in bad), bad)

    def test_a_generous_but_possible_latency_is_not_caught(self):
        """The gate must not fire on merely-fast. It is an impossibility bound,
        not a plausibility opinion."""
        def under(name, m, n, k):
            return (2 * m * n * k / (BR.BF16_PEAK_TFLOPS * 1e12) * 1.5) * 1e3 + 0.05
        path = timing_file(all_cases(under))
        self.assertEqual([], BR.check_physics(BR.analyse(LIVE, BR.load_timings(path, "ms"))))

    def test_a_nonpositive_latency_is_caught_not_divided_by(self):
        path = timing_file(all_cases(lambda n_, m, n, k: 0.0))
        bad = BR.check_physics(BR.analyse(LIVE, BR.load_timings(path, "ms")))
        self.assertTrue(all("not positive" in b for b in bad), bad)
        self.assertEqual(11, len(bad))


class ProvenanceTest(unittest.TestCase):
    """(144). The earlier draft hardcoded a table and married it to the wrong
    kernel. Provenance is now required and echoed."""

    def test_timing_and_field_are_required(self):
        with self.assertRaises(SystemExit):
            BR.main([str(LIVE)])

    def test_no_latency_table_is_hardcoded_in_the_module(self):
        text = (HERE / "bandwidth_regime.py").read_text(encoding="utf-8")
        self.assertNotIn("STORED_MS", text,
                         "a hardcoded latency table came back; that is the bug "
                         "this tool was rewritten to stop committing")

    def test_a_missing_field_is_refused_not_defaulted(self):
        with self.assertRaisesRegex(BR.Implausible, "has no"):
            BR.load_timings(TW035, "no_such_field")

    def test_it_reads_the_key_the_performance_runner_actually_emits(self):
        # This tool's whole purpose is to consume `task_runner.py performance`
        # output, and that emits `test_case_id`. The reader only knew `name`
        # (the key the CORRECTNESS path uses), so feeding it its own intended
        # input died on a bare KeyError. A tool that cannot parse the one file
        # it exists to read is not a tool with a rough edge; nothing had ever
        # run it end to end.
        import json, tempfile
        from pathlib import Path as _P
        p = _P(tempfile.mkdtemp()) / "perf.json"
        p.write_text(json.dumps({"test_cases": [
            {"test_case_id": "decode_m2_square", "candidate_ms": 0.072}]}))
        self.assertEqual(BR.load_timings(p, "candidate_ms"),
                         {"decode_m2_square": 0.072})

    def test_the_correctness_spelling_still_works(self):
        import json, tempfile
        from pathlib import Path as _P
        p = _P(tempfile.mkdtemp()) / "corr.json"
        p.write_text(json.dumps({"test_cases": [
            {"name": "decode_m2_square", "ms": 0.072}]}))
        self.assertEqual(BR.load_timings(p, "ms"), {"decode_m2_square": 0.072})

    def test_a_case_with_no_route_key_at_all_names_what_it_did_have(self):
        import json, tempfile
        from pathlib import Path as _P
        p = _P(tempfile.mkdtemp()) / "odd.json"
        p.write_text(json.dumps({"test_cases": [{"route": "x", "ms": 0.1}]}))
        with self.assertRaisesRegex(BR.Implausible, "neither .* nor"):
            BR.load_timings(p, "ms")

    def test_a_partial_suite_is_refused(self):
        path = timing_file({"decode_m2_square": 0.026})
        with self.assertRaisesRegex(BR.Implausible, "no latency for"):
            BR.analyse(LIVE, BR.load_timings(path, "ms"))


class ByteModelTest(unittest.TestCase):
    def test_compulsory_counts_each_operand_once(self):
        r = {"M": 2, "N": 4096, "K": 4096, "grid": "64x1"}
        t = BR.traffic(r)
        self.assertEqual(2 * 4096 * 2 + 4096 * 4096 * 2 + 2 * 4096 * 2, t["compulsory"])
        self.assertEqual(2 * 2 * 4096 * 4096, t["flop"])

    def test_requested_counts_the_grids_re_reads(self):
        r = {"M": 2048, "N": 4096, "K": 4096, "grid": "64x64"}
        t = BR.traffic(r)
        self.assertGreater(t["requested"], 40 * t["compulsory"],
                           "a 64-row grid re-reads B 64 times; the model must say so")

    def test_only_single_tile_row_routes_are_marked_exact(self):
        table = BR.analyse(LIVE, BR.load_timings(TW035, "latency_ms"))
        for r in table:
            with self.subTest(case=r["case"]):
                gy = int(r["grid"].split("x")[1])
                if gy > 1:
                    self.assertFalse(r["exact"])
                    self.assertIn("unusable", r["regime"])

    def test_a_route_touching_every_cu_is_marked_closed_by_23(self):
        """Not "open". Whatever is left on those routes, it is not a CU that
        has no work, and (23) already measured the stacking version."""
        table = BR.analyse(LIVE, BR.load_timings(TW035, "latency_ms"))
        for r in table:
            if r["exact"] and r["cu_touched_pct"] >= 99.0:
                with self.subTest(case=r["case"]):
                    self.assertIn("closed-by-23", r["regime"])


class ScalingExperimentTest(unittest.TestCase):
    def test_the_equal_cu_control_resolves(self):
        """Two routes at the same CU count must agree. If this control drifts,
        the headline number is noise and means nothing."""
        for src, field in ((TW035, "latency_ms"), (EPOCH_O, "latency_ms"),
                           (EPOCH_O, "baseline_ms")):
            with self.subTest(src=src.parent.name, field=field):
                table = BR.analyse(LIVE, BR.load_timings(src, field))
                same = {r["case"]: r["compulsory_tbs"] for r in table
                        if r["exact"] and abs(r["cu_touched_pct"] - 21.1) < 0.5}
                self.assertEqual(2, len(same), same)
                lo, hi = min(same.values()), max(same.values())
                self.assertLess(hi / lo, 1.05,
                                f"equal-CU routes disagree by {hi / lo:.2f}x: {same}")

    def test_bandwidth_does_not_scale_with_cus_across_three_sources(self):
        """The result D3 turns on, held to agree across two epochs and two
        independent implementations (a lane candidate and the rocBLAS oracle)."""
        for src, field in ((TW035, "latency_ms"), (EPOCH_O, "latency_ms"),
                           (EPOCH_O, "baseline_ms")):
            with self.subTest(src=src.parent.name, field=field):
                table = BR.analyse(LIVE, BR.load_timings(src, field))
                usable = [r for r in table if r["exact"]]
                base = min(usable, key=lambda r: r["cu_touched_pct"])
                far = max(usable, key=lambda r: r["cu_touched_pct"])
                cux = far["cu_touched_pct"] / base["cu_touched_pct"]
                ratio = far["compulsory_tbs"] / (base["compulsory_tbs"] * cux)
                self.assertGreater(cux, 2.5, "the CU contrast vanished")
                self.assertLess(ratio, 0.6,
                                "bandwidth now tracks CU count; the premise D3 was "
                                "bounded on has changed and the bound must be redone")
                self.assertGreater(ratio, 0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
