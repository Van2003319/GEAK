#!/usr/bin/env python3
"""Tests for the MFMA feed model.

The model's job is to make a refutable prediction before round 8 is timed, so
what needs pinning is the arithmetic it rests on -- if a constant drifts, the
prediction silently becomes a different prediction.
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mfma_feed_model as M  # noqa: E402


class ArithmeticTest(unittest.TestCase):
    def test_peak_flop_per_cycle_per_cu_is_2048(self):
        # 1307.4 TF / (304 CU * 2.1 GHz). If this is not ~2048 the published
        # figures have been mixed from different parts.
        self.assertAlmostEqual(M.flop_per_cycle_per_cu(), 2048, delta=2)

    def test_one_mfma_16x16x16_occupies_16_simd_cycles(self):
        # 2*16^3 = 8192 flop, 4 SIMDs per CU -> 512 flop/cycle/SIMD.
        per_simd = M.flop_per_cycle_per_cu() / 4
        self.assertAlmostEqual(8192 / per_simd, 16, delta=0.05)

    def test_the_stage_matches_the_isa_mfma_count(self):
        # 32 v_mfma per wave per stage was COUNTED in the ISA. Deriving it
        # independently from geometry is the cross-check that the model is
        # describing the same loop the probe compiled.
        s = M.stage()
        waves = M.THREADS // M.WAVE
        self.assertEqual(s["flop"] / waves / 8192, 32)

    def test_cta_level_ai_is_64_not_the_suite_figure(self):
        # The point the model exists to make: the loop asks for 64 flop/byte,
        # not the 762/1024 the suite table reports. Those describe DRAM traffic
        # after reuse.
        self.assertEqual(M.stage()["cta_ai"], 64)

    def test_demand_exceeds_hbm_by_about_4x(self):
        d = M.demand()
        self.assertAlmostEqual(d["bytes_per_cycle"], 32.0, delta=0.1)
        self.assertGreater(d["bytes_per_cycle"] / M.hbm_bytes_per_cycle_per_cu(), 3.5)

    def test_pf1_inflight_equals_one_whole_stage_of_traffic(self):
        # Self-consistency: at PF=1 the wave issues the entire stage's loads
        # before waiting, so in-flight bytes per CU must equal bytes per stage
        # per CU. If these diverge, either the ISA count or the geometry is
        # being misread.
        self.assertEqual(M.inflight_bytes_per_cu(4), M.demand()["cu_bytes_per_stage"])

    def test_pf2_doubles_inflight(self):
        self.assertEqual(M.inflight_bytes_per_cu(8),
                         2 * M.inflight_bytes_per_cu(4))

    def test_pf1_already_covers_littles_law_at_both_latency_ends(self):
        # This is the prediction. If it ever flips, the model now says PF=2 has
        # something real to relieve, and the recorded prediction must change
        # with it rather than being quoted from the log.
        d = M.demand()["bytes_per_cycle"]
        have = M.inflight_bytes_per_cu(4)
        for lat in (M.LAT_L2_CYCLES, M.LAT_HBM_CYCLES):
            self.assertGreaterEqual(have, M.required_inflight(d, lat), lat)

    def test_the_model_reports_the_hit_rate_it_depends_on(self):
        # 1 - 8.3/32 ~ 74%. The model is only as good as this assumption, so it
        # must be stated, not buried.
        d = M.demand()["bytes_per_cycle"]
        need = 1 - M.hbm_bytes_per_cycle_per_cu() / d
        self.assertAlmostEqual(need, 0.74, delta=0.02)


class GeometryTest(unittest.TestCase):
    def test_cta_per_cu_matches_the_lds_budget(self):
        lds = (M.BM + M.BN) * 36 * 2   # kMacroLdsStride = 36 at BK=32
        self.assertEqual(lds, 18432)
        self.assertEqual(65536 // lds, M.CTA_PER_CU)


if __name__ == "__main__":
    unittest.main()
