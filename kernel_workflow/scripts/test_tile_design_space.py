#!/usr/bin/env python3
"""Tests for the round-9 macro-tile enumerator.

Two of the three quantities here are exact arithmetic (LDS bytes, AGPR per
lane) and one is an estimate (VGPR). The tests pin the exact ones hard, pin the
incumbent as a control -- if the model cannot reproduce the tile we have
actually measured, it is not describing our kernel -- and pin the indivisible-
CTA rule, which is the bug this file was written after finding.
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tile_design_space as T  # noqa: E402

VGPR = 86   # measured, PF=2 128x128 -- the pessimistic real datapoint


class ExactArithmeticTest(unittest.TestCase):
    def test_lds_rounds_up_to_the_512b_allocation_granularity(self):
        self.assertEqual(T.ceil512(1), 512)
        self.assertEqual(T.ceil512(512), 512)
        self.assertEqual(T.ceil512(513), 1024)

    def test_cta_ai_is_the_harmonic_shape_not_the_area(self):
        # BM*BN/(BM+BN). Squarer is better at equal perimeter, and the figure
        # is independent of BK -- the whole reason BK cannot be the lever.
        self.assertEqual(T.cta_ai(128, 128), 64)
        self.assertEqual(T.cta_ai(256, 256), 128)
        self.assertAlmostEqual(T.cta_ai(128, 256), 85.333, places=3)
        for bk in (16, 32, 64, 128):
            self.assertEqual(T.lds_bytes(128, 128, bk) > 0, True)
        self.assertEqual(T.cta_ai(128, 128), T.cta_ai(128, 128))

    def test_agpr_per_lane_is_exact_output_elements_over_lanes(self):
        self.assertEqual(T.Tile(128, 128, 32, 256, VGPR).agpr, 64)
        self.assertEqual(T.Tile(256, 256, 32, 512, VGPR).agpr, 128)
        self.assertEqual(T.Tile(256, 256, 32, 1024, VGPR).agpr, 64)

    def test_demand_is_peak_over_ai_and_falls_as_the_tile_grows(self):
        self.assertAlmostEqual(T.demand_bytes_per_cycle(128, 128), 32.0, places=3)
        self.assertAlmostEqual(T.demand_bytes_per_cycle(256, 256), 16.0, places=3)


class IncumbentControlTest(unittest.TestCase):
    """The model must reproduce the tile we have already measured."""

    def setUp(self):
        self.inc = T.Tile(128, 128, 32, 256, VGPR)

    def test_lds_matches_the_kMacroLdsStride_36_figure(self):
        # (BM+BN) * (BK+4) * 2 = 256*36*2, already 512-aligned.
        self.assertEqual(self.inc.lds, 18432)

    def test_three_ctas_per_cu(self):
        self.assertEqual(self.inc.cta_per_cu_lds, 3)
        self.assertEqual(self.inc.cta_per_cu, 3)
        self.assertEqual(self.inc.waves_per_cu, 12)
        self.assertEqual(self.inc.waves_per_simd, 3.0)

    def test_agpr_matches_the_isa_probe(self):
        # The ISA probe measured AGPR 64 at every PF depth. If this diverges,
        # the geometry in the model is not the geometry in the source.
        self.assertEqual(self.inc.agpr, 64)

    def test_hit_rate_matches_the_feed_model(self):
        self.assertAlmostEqual(self.inc.hit_rate_needed, 0.74, delta=0.01)


class IndivisibleCtaTest(unittest.TestCase):
    """A block is resident or it is not. There is no partial residency.

    This is the bug the first version of the enumerator had: it reported
    occupancy as min(lds_waves, register_waves), which for a 1024-thread
    256x256 CTA printed a plausible 3 waves/SIMD for a launch that cannot
    happen at all.
    """

    def test_a_1024_thread_256x256_cta_cannot_be_resident(self):
        t = T.Tile(256, 256, 32, 1024, VGPR)
        self.assertEqual(t.waves_per_cta, 16)
        # AGPR 64 + VGPR 86 = 150 -> 3 waves/SIMD -> 12 waves/CU < 16.
        self.assertEqual(t.waves_per_simd_regs, 3)
        self.assertEqual(t.cta_per_cu, 0)
        self.assertFalse(t.feasible)

    def test_the_naive_min_would_have_reported_it_as_fine(self):
        # Guards the fix itself: assert the OLD formula and the new one
        # disagree here, so this test cannot quietly go vacuous.
        t = T.Tile(256, 256, 32, 1024, VGPR)
        naive = min(t.cta_per_cu_lds * t.waves_per_cta, T.WAVE_SLOTS_PER_CU,
                    t.waves_per_simd_regs * 4)
        self.assertEqual(naive, 12)
        self.assertEqual(t.waves_per_cu, 0)

    def test_the_512_thread_version_fits_exactly(self):
        t = T.Tile(256, 256, 32, 512, VGPR)
        self.assertEqual(t.waves_per_cta, 8)
        self.assertEqual(t.agpr, 128)
        # 128 + 86 = 214 -> floor(512/214) = 2 waves/SIMD = 8 waves/CU = 1 CTA.
        self.assertEqual(t.waves_per_simd_regs, 2)
        self.assertEqual(t.cta_per_cu, 1)
        self.assertTrue(t.feasible)

    def test_registers_can_bind_tighter_than_lds(self):
        # 224x224 BK=32: LDS allows 2 CTAs, registers allow 1. The report shows
        # this as "1/2" and the effective figure must be the smaller.
        t = T.Tile(224, 224, 32, 512, VGPR)
        self.assertEqual(t.cta_per_cu_lds, 2)
        self.assertEqual(t.cta_per_cu, 1)


class HeadroomTest(unittest.TestCase):
    def test_headroom_is_the_distance_to_the_launch_cliff(self):
        t = T.Tile(256, 256, 32, 512, VGPR)
        # 1 CTA of 8 waves needs 2 waves/SIMD, so 256 regs/lane, minus 128 AGPR
        # leaves 128 for VGPR -- 42 above the 86 estimate.
        self.assertEqual(t.vgpr_headroom, 128 - VGPR)

    def test_a_tile_at_its_headroom_still_fits_and_one_over_does_not(self):
        at = T.Tile(256, 256, 32, 512, VGPR + T.Tile(256, 256, 32, 512, VGPR).vgpr_headroom)
        self.assertEqual(at.cta_per_cu, 1)
        over = T.Tile(256, 256, 32, 512, at.vgpr + 1)
        self.assertEqual(over.cta_per_cu, 0)
        self.assertFalse(over.feasible)


class EnumerationTest(unittest.TestCase):
    def setUp(self):
        self.tiles = T.enumerate_tiles(VGPR)

    def test_every_enumerated_tile_is_actually_launchable(self):
        for t in self.tiles:
            self.assertGreaterEqual(t.cta_per_cu, 1, t)
            self.assertLessEqual(t.lds, T.LDS_BYTES_PER_CU, t)
            self.assertLessEqual(t.agpr + t.vgpr, T.REGS_PER_LANE, t)

    def test_the_incumbent_shape_is_in_the_space(self):
        # If the enumerator excludes the tile we ship, its filters are wrong.
        self.assertTrue(any(t.bm == 128 and t.bn == 128 and t.bk == 32
                            for t in self.tiles))

    def test_nothing_reaches_the_ridge(self):
        # The honest headline: even 256x256 gets to AI 128, half the ~247 ridge.
        # No tile in the LDS budget makes these routes compute-bound.
        self.assertLess(max(t.ai for t in self.tiles), T.RIDGE_AI)

    def test_the_top_usable_candidate_is_256x256_at_512_threads(self):
        # "Usable" = the >=2 waves/SIMD bar the shortlist applies. Below that
        # there is a single wave per SIMD and nothing at all to hide latency
        # behind, which is the opposite of what a bigger tile is being bought
        # for.
        usable = [t for t in self.tiles if t.waves_per_simd >= 2]
        best = max(usable, key=lambda t: t.ai)
        self.assertEqual((best.bm, best.bn, best.threads), (256, 256, 512))
        self.assertEqual(best.ai, 128.0)

    def test_the_256_thread_256x256_variant_is_legal_but_too_thin(self):
        # It sits exactly on the 256-AGPR ceiling and gets 1 wave/SIMD. The
        # enumerator must not silently prefer it just because the AI ties.
        t = T.Tile(256, 256, 32, 256, VGPR)
        self.assertEqual(t.agpr, T.MAX_AGPR_PER_LANE)
        self.assertEqual(t.cta_per_cu, 1)
        self.assertEqual(t.waves_per_simd, 1.0)
        self.assertTrue(t.feasible)
        self.assertEqual(t.ai, T.Tile(256, 256, 32, 512, VGPR).ai)

    def test_the_agpr_ceiling_is_enforced_independently_of_the_512_budget(self):
        # 256x256 at 128 threads wants 512 accumulators per lane: inside the
        # unified budget only if VGPR were zero, and over the AGPR half's own
        # 256 limit regardless.
        t = T.Tile(256, 256, 32, 128, 0)
        self.assertEqual(t.agpr, 512)
        self.assertLessEqual(t.agpr + t.vgpr, T.REGS_PER_LANE)
        self.assertFalse(t.feasible)

    def test_best_per_shape_keeps_one_row_per_geometry(self):
        rows = T.best_per_shape(self.tiles)
        keys = [(t.bm, t.bn, t.bk) for t in rows]
        self.assertEqual(len(keys), len(set(keys)))


class MeasuredProbeTest(unittest.TestCase):
    """The offline hipcc numbers, and the control that makes them admissible."""

    def test_the_control_reproduces_the_shipped_kernel(self):
        # Not decoration. If the control instantiation does not match the
        # shipped macro_gemm_128x128_kernel's measured 74/64/0/3/18432, the
        # probe compiled something other than our kernel and EVERY other row
        # in MEASURED is void.
        self.assertEqual(T.MEASURED[(128, 128, 32, 256)], (74, 64, 0, 3, 18432))

    def test_no_candidate_spilled_or_used_scratch(self):
        for key, (_v, _a, scratch, _occ, _lds) in T.MEASURED.items():
            self.assertEqual(scratch, 0, key)

    def test_the_estimate_was_close_on_the_register_total(self):
        # Predicted 128 AGPR + 86 VGPR = 214 for 256x256; measured 212. The
        # model may not claim the AGPR/VGPR split, only the sum.
        vgpr, agpr, *_ = T.MEASURED[(256, 256, 32, 512)]
        self.assertEqual(vgpr + agpr, 212)
        self.assertLess(abs((vgpr + agpr) - (128 + VGPR)), 8)

    def test_accumulators_landed_in_vgprs_not_agprs_at_512_threads(self):
        # Documents the one place the analytic model's vocabulary was wrong.
        for key, (_v, agpr, *_rest) in T.MEASURED.items():
            if key[3] == 512:
                self.assertEqual(agpr, 0, key)

    def test_256x256_launches_after_all(self):
        # The analytic pass flagged a launch cliff at 128 VGPR. Measured total
        # is 212, and 2 waves/SIMD allows 256, so the cliff did not bind.
        e = T.effective_from_measured(256, 256, 32, 512)
        self.assertEqual(e["cta_per_cu"], 1)
        self.assertEqual(e["waves_per_simd"], 2.0)
        self.assertEqual(e["ai"], 128.0)

    def test_the_compiler_occupancy_column_can_overstate(self):
        # 192x192 t512: compiler says 3 waves/SIMD, but the block is 8 waves so
        # only one fits -> 2. This is the indivisible-CTA rule again, now
        # catching a number that came from the toolchain rather than from us.
        e = T.effective_from_measured(192, 192, 32, 512)
        self.assertEqual(e["compiler_occ"], 3)
        self.assertEqual(e["cta_per_cu"], 1)
        self.assertEqual(e["waves_per_simd"], 2.0)

    def test_128x256_beats_the_incumbent_on_both_axes(self):
        # The finding the AI-only ranking hid: more AI *and* more occupancy.
        inc = T.effective_from_measured(128, 128, 32, 256)
        cand = T.effective_from_measured(128, 256, 32, 512)
        self.assertGreater(cand["ai"], inc["ai"])
        self.assertGreater(cand["waves_per_simd"], inc["waves_per_simd"])
        self.assertEqual(cand["cta_per_cu"], 2)

    def test_measured_rederivation_uses_the_total_not_the_split(self):
        # Feeding the same total split differently must not change residency.
        e = T.effective_from_measured(128, 128, 32, 256)
        self.assertEqual(e["total_regs"], 138)
        self.assertEqual(e["cta_per_cu"], 3)


if __name__ == "__main__":
    unittest.main()
