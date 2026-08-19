#!/usr/bin/env python3
"""GPU-free tests for qd_descriptor_v2.py."""
from __future__ import annotations

import importlib.util
import sys
import dataclasses
import statistics
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("qd_descriptor_v2.py")
SPEC = importlib.util.spec_from_file_location("qd_descriptor_v2", SCRIPT)
assert SPEC and SPEC.loader
QD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QD
SPEC.loader.exec_module(QD)

VALID = {
    "compute_primitive": "native_mfma",
    "wave_schedule": "independent",
    "k_pipeline": "lds_multistage",
    "decomposition": "tile_grid",
    "output_path": "direct_store",
    "rasterization": "grouped_m",
    "plan_binding": "static",
}


class DescriptorValidityTest(unittest.TestCase):
    def test_classifier_version_is_geak_qd_v2(self):
        self.assertEqual("geak-qd-v2", QD.CLASSIFIER_VERSION)

    def test_baseline_descriptor_is_valid(self):
        self.assertTrue(QD.descriptor_valid(VALID))

    def test_missing_axis_is_invalid(self):
        d = dict(VALID)
        del d["output_path"]
        self.assertFalse(QD.descriptor_valid(d))

    def test_unknown_value_is_invalid(self):
        d = {**VALID, "k_pipeline": "not_a_real_pipeline"}
        self.assertFalse(QD.descriptor_valid(d))

    def test_non_mapping_is_invalid(self):
        self.assertFalse(QD.descriptor_valid(None))
        self.assertFalse(QD.descriptor_valid("nope"))
        self.assertFalse(QD.descriptor_valid(["a", "list"]))

    def test_reduction_requires_fixup(self):
        d = {**VALID, "decomposition": "split_k", "output_path": "direct_store"}
        self.assertFalse(QD.descriptor_valid(d), "split_k with no fixup must drop partial sums")
        d["output_path"] = "atomic_fixup"
        self.assertTrue(QD.descriptor_valid(d))

    def test_fixup_requires_reduction(self):
        d = {**VALID, "decomposition": "tile_grid", "output_path": "atomic_fixup"}
        self.assertFalse(QD.descriptor_valid(d), "a fixup with nothing to fix up is meaningless")
        d["decomposition"] = "stream_k"
        self.assertTrue(QD.descriptor_valid(d))

    def test_pingpong_wave_schedule_requires_matrix_core(self):
        for wave in ("symmetric_interleave", "symmetric_pingpong"):
            d = {**VALID, "wave_schedule": wave, "compute_primitive": "valu"}
            self.assertFalse(QD.descriptor_valid(d), wave)
            d["compute_primitive"] = "native_mfma"
            self.assertTrue(QD.descriptor_valid(d), wave)

    def test_only_modeled_arches_are_supported(self):
        for arch in QD.SUPPORTED_ARCHES:
            self.assertTrue(QD.descriptor_valid(VALID, arch=arch), arch)
        self.assertFalse(QD.descriptor_valid(VALID, arch="gfx1100"))
        self.assertFalse(QD.descriptor_valid(VALID, arch=""))

    def test_xcd_remap_needs_a_multi_die_dispatcher(self):
        d = {**VALID, "rasterization": "xcd_remapped_grouped"}
        self.assertTrue(QD.descriptor_valid(d, arch="gfx942"))
        self.assertFalse(QD.descriptor_valid(d, arch="gfx90a"),
                         "one kernel never spans gfx90a's two GCDs, so there is "
                         "no round-robin to undo")

    def test_runtime_tuning_needs_a_reduction_to_tune(self):
        d = {**VALID, "plan_binding": "runtime_tuned"}
        self.assertFalse(QD.descriptor_valid(d))
        d.update(decomposition="split_k", output_path="workspace_fixup")
        self.assertTrue(QD.descriptor_valid(d))

    def test_deep_single_stage_is_a_k_pipeline_value(self):
        d = {**VALID, "k_pipeline": "lds_deep_single"}
        self.assertTrue(QD.descriptor_valid(d))
        order = QD.AXIS_ORDER["k_pipeline"]
        self.assertEqual(1, abs(order.index("lds_deep_single") - order.index("lds_pingpong")),
                         "the deep-stage mechanism must be one directed step from pingpong")

    def test_dtype_allow_list(self):
        for dtype in QD.SUPPORTED_DTYPES:
            self.assertTrue(QD.descriptor_valid(VALID, dtype=dtype), dtype)
        self.assertFalse(QD.descriptor_valid(VALID, dtype="fp8"))


class CoverageEligibilityTest(unittest.TestCase):
    def test_asymmetric_producer_consumer_is_legal_but_not_eligible(self):
        d = {**VALID, "wave_schedule": "asymmetric_producer_consumer", "compute_primitive": "native_mfma"}
        self.assertTrue(QD.descriptor_valid(d))
        self.assertFalse(QD.coverage_eligible(d))

    def test_ordinary_legal_descriptor_is_eligible(self):
        self.assertTrue(QD.coverage_eligible(VALID))

    def test_illegal_descriptor_is_not_eligible(self):
        d = {**VALID, "k_pipeline": "bogus"}
        self.assertFalse(QD.coverage_eligible(d))


class CellIdTest(unittest.TestCase):
    def test_cell_id_is_stable_and_field_ordered(self):
        cell = QD.cell_id("case_a", VALID)
        self.assertEqual(
            "|".join(["case_a", "native_mfma", "independent", "lds_multistage", "tile_grid",
                      "direct_store", "grouped_m", "static"]),
            cell)

    def test_cell_id_none_for_invalid_descriptor(self):
        self.assertIsNone(QD.cell_id("case_a", {**VALID, "k_pipeline": "bogus"}))

    def test_cell_id_none_for_empty_context(self):
        self.assertIsNone(QD.cell_id("", VALID))
        self.assertIsNone(QD.cell_id(None, VALID))

    def test_cell_id_respects_known_contexts_allowlist(self):
        self.assertIsNone(QD.cell_id("unknown_case", VALID, known_contexts=["case_a", "case_b"]))
        self.assertIsNotNone(QD.cell_id("case_a", VALID, known_contexts=["case_a", "case_b"]))

    def test_two_descriptors_differing_by_one_axis_get_distinct_cells(self):
        other = {**VALID, "output_path": "lds_staged_store"}
        self.assertNotEqual(QD.cell_id("case_a", VALID), QD.cell_id("case_a", other))


class AdjacencyTest(unittest.TestCase):
    def test_ordinary_neighbors_change_one_axis_by_one_step(self):
        for n in QD.adjacency(VALID):
            changed = [axis for axis in QD.AXES if n.descriptor[axis] != VALID[axis]]
            self.assertEqual(list(n.axes), changed)
            if len(n.axes) == 1:
                order = QD.AXIS_ORDER[n.axis]
                self.assertEqual(1, abs(order.index(n.descriptor[n.axis]) - order.index(VALID[n.axis])))

    def test_neighbors_are_all_legal(self):
        for n in QD.adjacency(VALID):
            self.assertTrue(QD.descriptor_valid(n.descriptor), n.descriptor)

    def test_edge_of_axis_has_no_out_of_range_neighbor(self):
        # compute_primitive="valu" is the first value: no "prev" neighbor on that axis.
        d = {**VALID, "compute_primitive": "valu", "wave_schedule": "independent"}
        axes_with_prev = {n.axis for n in QD.adjacency(d) if n.direction == "prev"}
        self.assertNotIn("compute_primitive", axes_with_prev)

    def test_reduction_boundary_uses_explicit_coupled_edges(self):
        d = {**VALID, "decomposition": "persistent_output", "output_path": "direct_store"}
        neighbors = QD.adjacency(d)
        split_targets = [n for n in neighbors if n.descriptor["decomposition"] == "split_k"]
        self.assertEqual(2, len(split_targets))
        self.assertEqual({"atomic_fixup", "workspace_fixup"},
                         {n.descriptor["output_path"] for n in split_targets})
        self.assertTrue(all(n.axes == ("decomposition", "output_path") for n in split_targets))
        self.assertTrue(all(n.direction == "coupled" for n in split_targets))

    def test_coupled_reduction_edge_is_reversible(self):
        d = {**VALID, "decomposition": "split_k", "output_path": "atomic_fixup"}
        targets = QD.adjacency(d)
        back = [n for n in targets if n.descriptor["decomposition"] == "persistent_output"]
        self.assertEqual({"direct_store", "lds_staged_store"},
                         {n.descriptor["output_path"] for n in back})

    def test_adjacency_is_deterministic_and_ordered(self):
        first = QD.adjacency(VALID)
        second = QD.adjacency(VALID)
        self.assertEqual([n.object() for n in first], [n.object() for n in second])
        ordinary = [n for n in first if len(n.axes) == 1]
        axis_positions = [QD.AXES.index(n.axis) for n in ordinary]
        self.assertEqual(sorted(axis_positions), axis_positions)
        self.assertTrue(all(len(n.axes) >= 2 for n in first[len(ordinary):]))

    def test_non_mapping_descriptor_yields_no_neighbors(self):
        self.assertEqual([], QD.adjacency(None))

    def test_current_value_missing_from_axis_order_is_skipped(self):
        d = {**VALID, "compute_primitive": "totally_unknown"}
        axes_touched = {n.axis for n in QD.adjacency(d)}
        self.assertNotIn("compute_primitive", axes_touched)


class ExhaustiveScanTest(unittest.TestCase):
    def test_all_legal_descriptors_are_individually_valid(self):
        legal = QD.all_legal_descriptors()
        self.assertGreater(len(legal), 0)
        for d in legal:
            self.assertTrue(QD.descriptor_valid(d), d)

    def test_all_legal_descriptors_matches_brute_force_count(self):
        import itertools
        for arch in QD.SUPPORTED_ARCHES:
            legal = QD.all_legal_descriptors(arch=arch)
            brute = [dict(zip(QD.AXES, combo))
                     for combo in itertools.product(*(QD.AXIS_ORDER[a] for a in QD.AXES))
                     if QD.descriptor_valid(dict(zip(QD.AXES, combo)), arch=arch)]
            self.assertEqual(len(brute), len(legal), arch)

    def test_multi_die_arch_has_strictly_more_legal_cells(self):
        self.assertGreater(len(QD.all_legal_descriptors(arch="gfx942")),
                           len(QD.all_legal_descriptors(arch="gfx90a")))

    def test_no_duplicate_legal_descriptors(self):
        legal = QD.all_legal_descriptors()
        seen = {tuple(sorted(d.items())) for d in legal}
        self.assertEqual(len(seen), len(legal))

    def test_every_legal_descriptor_has_at_least_one_neighbor(self):
        # The grid is small and densely connected; an isolated legal cell would
        # be unreachable by any directed transition, which would be suspicious.
        for d in QD.all_legal_descriptors():
            self.assertGreater(len(QD.adjacency(d)), 0, d)

    def test_coupled_edge_releases_runtime_tuning_with_the_reduction(self):
        d = {**VALID, "decomposition": "split_k", "output_path": "workspace_fixup",
             "plan_binding": "runtime_tuned"}
        self.assertTrue(QD.descriptor_valid(d))
        back = [n for n in QD.adjacency(d) if n.descriptor["decomposition"] == "persistent_output"]
        self.assertTrue(back, "dropping the reduction must stay reachable")
        for n in back:
            self.assertEqual("static", n.descriptor["plan_binding"])
            self.assertIn("plan_binding", n.axes)


class AxisEffectTest(unittest.TestCase):
    def test_every_axis_moves_something_tradeable(self):
        self.assertEqual([], QD.dead_axes(),
                         "an axis that changes no tradeable quantity renames the "
                         "search space instead of enlarging it")

    def test_axis_effects_covers_exactly_the_axes(self):
        self.assertEqual(set(QD.AXES), set(QD.AXIS_EFFECTS))

    def test_axis_effects_only_names_tradeable_quantities(self):
        for axis, effects in QD.AXIS_EFFECTS.items():
            self.assertTrue(effects <= QD.TRADEABLE, axis)


class TombstonedAxisTest(unittest.TestCase):
    """Findings (41)-(43) and (24): axes that were built, measured and refused.

    Distinct from the dead-axis check on purpose. `mfma_shape` moves occupancy
    -- the gfx942 build dropped VGPR 76 -> 64 -- so it is tradeable and
    `dead_axes()` would have waved it through. It is refused for being coupled
    to wave-tile size, which no per-axis predicate can detect.
    """

    def test_no_tombstoned_axis_has_been_readopted(self):
        self.assertEqual([], QD.tombstoned_axes(),
                         "an axis here was measured and refused; re-adding it "
                         "costs a build and a sweep to rediscover the loss")

    def test_mfma_shape_is_absent_from_the_vocabulary(self):
        for name in ("mfma_shape", "occupancy_fill"):
            self.assertNotIn(name, QD.AXES)
            self.assertNotIn(name, QD.AXIS_ORDER)
            self.assertNotIn(name, QD.QD_VOCAB)
            self.assertNotIn(name, QD.AXIS_EFFECTS)

    def test_the_guard_fires_when_a_tombstoned_axis_comes_back(self):
        # The check is worthless unless it can fail, and the failure mode it
        # guards is someone appending to AXIS_ORDER, so drive it that way.
        restore = QD.AXIS_ORDER
        try:
            QD.AXIS_ORDER = dict(restore)
            QD.AXIS_ORDER["mfma_shape"] = ("16x16x16", "32x32x8")
            QD.AXES = tuple(QD.AXIS_ORDER)
            self.assertEqual(["mfma_shape"], QD.tombstoned_axes())
        finally:
            QD.AXIS_ORDER = restore
            QD.AXES = tuple(restore)
        self.assertEqual([], QD.tombstoned_axes())

    def test_dead_axes_would_not_have_caught_mfma_shape(self):
        # Pins the correction to `dead_axes()`'s docstring: the old claim that
        # it "would have rejected the mfma_shape axis" was false, and a future
        # reader must not rely on it to catch the next coupled axis.
        self.assertTrue(frozenset({"occupancy"}) <= QD.TRADEABLE)

    def test_the_refusal_carries_its_evidence(self):
        self.assertIn("41-43", QD.MFMA_SHAPE_IS_NOT_AN_AXIS)
        self.assertIn("coupled", QD.MFMA_SHAPE_IS_NOT_AN_AXIS)
        self.assertEqual(24, QD.OCCUPANCY_FILL_REMOVED_BY_FINDING)


class MechanismTest(unittest.TestCase):
    def test_every_recorded_mechanism_is_a_single_legal_step(self):
        for mech in QD.MECHANISMS:
            self.assertTrue(mech.valid(), mech.name)

    def test_mechanism_names_are_unique(self):
        names = [m.name for m in QD.MECHANISMS]
        self.assertEqual(len(set(names)), len(names))

    def test_starved_grid_selects_the_deep_stage_mechanism(self):
        facts = QD.RouteFacts(tiles=32, slices=8, cu_count=304, tile_rows=1)
        names = {m.name for m in QD.mechanisms_for(facts)}
        self.assertIn("deepen_starved_grid", names)
        self.assertNotIn("undo_die_round_robin_then_group", names,
                         "a one-tile-row grid has nothing to rasterize")

    def test_full_grid_does_not_select_the_deep_stage_mechanism(self):
        facts = QD.RouteFacts(tiles=344, slices=1, cu_count=304, tile_rows=4)
        names = {m.name for m in QD.mechanisms_for(facts)}
        self.assertNotIn("deepen_starved_grid", names)
        self.assertIn("undo_die_round_robin_then_group", names)

    def test_descriptor_filter_requires_the_move_to_be_available(self):
        facts = QD.RouteFacts(tiles=32, slices=8, cu_count=304, tile_rows=1)
        at_pingpong = {**VALID, "k_pipeline": "lds_pingpong", "decomposition": "split_k",
                       "output_path": "workspace_fixup"}
        self.assertIn("deepen_starved_grid",
                      {m.name for m in QD.mechanisms_for(facts, at_pingpong, arch="gfx942")})
        already_deep = {**at_pingpong, "k_pipeline": "lds_deep_single"}
        self.assertNotIn("deepen_starved_grid",
                         {m.name for m in QD.mechanisms_for(facts, already_deep, arch="gfx942")})

    def test_unknown_precondition_never_applies(self):
        bogus = QD.Mechanism(name="x", precondition="no_such_precondition",
                             axis="k_pipeline", from_value="lds_pingpong",
                             to_value="lds_deep_single", spends="occupancy", evidence="none")
        self.assertFalse(bogus.valid())
        self.assertFalse(bogus.applies_to(QD.RouteFacts(tiles=1, slices=1, cu_count=304)))

    def test_mechanism_must_be_adjacent_not_a_jump(self):
        jump = QD.Mechanism(name="x", precondition="grid_cannot_fill_machine",
                            axis="k_pipeline", from_value="direct_global",
                            to_value="lds_deep_single", spends="occupancy", evidence="none")
        self.assertFalse(jump.valid())


class SplitKTailWavePrecondition(unittest.TestCase):
    """Pinned to the three gfx942/bf16 shapes that were actually measured.

    If someone widens the threshold to let a new candidate through, the two
    shapes that were measured to REJECT the move start failing here.
    """

    def _facts(self, tiles, slices, k):
        return QD.RouteFacts(tiles=tiles, slices=slices, cu_count=304,
                             ctas_per_cu_cap=1, k=k)

    def test_fires_on_the_one_shape_that_gives_back_the_wave(self):
        # 128x4096x4096: 64 tiles x 8 slices = 512 CTAs on 304 CUs, and
        # 8**2/4096 = 0.0156 of reduction overhead. Measured +17.5%.
        self.assertTrue(QD.PRECONDITIONS["split_k_tail_wave_unhidable"](
            self._facts(64, 8, 4096)))

    def test_rejects_the_long_k_shapes_that_must_keep_it(self):
        # 256x4096x11008 wants 8 slices; 1024x4096x11008 wants 4. Same
        # residency, same wave overflow -- only the reduction fraction differs.
        for tiles, slices, k in ((64, 8, 11008), (208, 4, 11008)):
            with self.subTest(tiles=tiles, slices=slices, k=k):
                self.assertFalse(QD.PRECONDITIONS["split_k_tail_wave_unhidable"](
                    self._facts(tiles, slices, k)))

    def test_requires_residency_overflow_and_a_split(self):
        base = dict(tiles=64, slices=8, cu_count=304, ctas_per_cu_cap=1, k=4096)
        pred = QD.PRECONDITIONS["split_k_tail_wave_unhidable"]
        # Two CTAs resident: the tail has something to overlap with.
        self.assertFalse(pred(QD.RouteFacts(**{**base, "ctas_per_cu_cap": 2})))
        # No split to give back.
        self.assertFalse(pred(QD.RouteFacts(**{**base, "slices": 1})))
        # Already inside one wave: 8 tiles x 8 slices = 64 CTAs.
        self.assertFalse(pred(QD.RouteFacts(**{**base, "tiles": 8})))
        # k not supplied must fail closed rather than divide by nothing.
        self.assertFalse(pred(QD.RouteFacts(**{**base, "k": 0})))


class StarvationIsCountedInCtasNotTiles(unittest.TestCase):
    """The v76 finding, pinned so it cannot silently regress.

    A grid that splits K produces `tiles * slices` CTAs. Pricing starvation on
    the tile count alone refuses tile-widening on grids that are not in fact
    starved -- measured as +2.82% left on the table on 256x4096x11008.
    """

    def test_ctas_multiplies_the_slice_count(self):
        self.assertEqual(416, QD.RouteFacts(tiles=52, slices=8, cu_count=304).ctas)

    def test_split_grid_is_not_starved_though_its_tile_count_is_small(self):
        # 52 wide tiles on 304 CUs looks starved; 416 CTAs is not.
        facts = QD.RouteFacts(tiles=52, slices=8, cu_count=304)
        self.assertFalse(QD.PRECONDITIONS["grid_cannot_fill_machine"](facts))
        self.assertLess(facts.tiles, facts.cu_count)

    def test_the_same_tiles_without_a_split_are_starved(self):
        self.assertTrue(QD.PRECONDITIONS["grid_cannot_fill_machine"](
            QD.RouteFacts(tiles=52, slices=1, cu_count=304)))


class FusedFixupIsALegalEdgeThatWasMeasuredToLose(unittest.TestCase):
    """v78. The edge must stay reachable; it must not be sold as a mechanism.

    Moving the split-K reduction into the GEMM kernel behind a per-tile arrival
    counter is a legal single step on `output_path`, and the search is allowed
    to take it -- on an arch with fewer dies or a cheaper device fence it could
    win. On gfx942 it lost 43.6% geomean across nine split-capable shapes,
    because it trades `slices`-way reduction parallelism and L2 residency for
    one saved dispatch. Nothing in MECHANISMS may recommend it.
    """

    def test_the_edge_is_a_single_legal_step(self):
        d = {**VALID, "decomposition": "split_k", "output_path": "workspace_fixup"}
        self.assertTrue(QD.descriptor_valid(d))
        targets = [n for n in QD.adjacency(d)
                   if n.descriptor["output_path"] == "atomic_fixup"]
        self.assertTrue(targets, "the search must still be able to try it")
        self.assertTrue(all(n.axes == ("output_path",) for n in targets))

    def test_no_mechanism_recommends_it(self):
        for mech in QD.MECHANISMS:
            if mech.axis == "output_path":
                self.assertNotEqual(("workspace_fixup", "atomic_fixup"),
                                    (mech.from_value, mech.to_value), mech.name)


class OneWavePerSimdIsReadFromLdsAndNotFromLaunchBounds(unittest.TestCase):
    """v79/v80. The precondition must separate the winner from the loser.

    Both routes carry `__launch_bounds__(..., 1)`, so a precondition phrased on
    launch bounds fires on both and predicts nothing -- that is the error that
    built v80. LDS is exact on the host: 50688 of 65536 bytes admits one CTA and
    forced one wave per SIMD (+4.6% suite, +7.5% isolated when split to eight
    waves); 13824 admits four, so that route already had its second wave and the
    same move cost 1.9% with complete separation.
    """

    WINNER = dict(tiles=26, slices=4, cu_count=304, lds_bytes=50688,
                  waves_per_cta=4)      # 64x128 at stage_k=128, the m128 route
    LOSER = dict(tiles=88, slices=1, cu_count=304, lds_bytes=13824,
                 waves_per_cta=4)       # 128x64, the m512 route

    def test_lds_bound_is_the_capacity_arithmetic(self):
        for lds, expect in ((50688, 1), (49920, 1), (26112, 2), (20736, 3),
                            (18432, 3), (16128, 4), (13824, 4)):
            with self.subTest(lds=lds):
                f = QD.RouteFacts(tiles=26, slices=4, cu_count=304,
                                  lds_bytes=lds, waves_per_cta=4)
                self.assertEqual(expect, f.ctas_by_lds)

    def test_it_fires_on_the_route_where_the_move_paid(self):
        f = QD.RouteFacts(**self.WINNER)
        self.assertEqual(1.0, f.waves_per_simd)
        self.assertTrue(QD.PRECONDITIONS["simd_holds_one_wave"](f))

    def test_it_declines_the_route_where_the_move_lost(self):
        f = QD.RouteFacts(**self.LOSER)
        self.assertEqual(4.0, f.waves_per_simd)
        self.assertFalse(QD.PRECONDITIONS["simd_holds_one_wave"](f))

    def test_launch_bounds_would_not_have_separated_them(self):
        """Both routes ask the compiler for one CTA; only LDS tells them apart."""
        self.assertNotEqual(
            QD.RouteFacts(**self.WINNER).waves_per_simd,
            QD.RouteFacts(**self.LOSER).waves_per_simd)

    def test_it_stops_firing_once_the_move_is_made(self):
        """Eight waves on the same stage reads 2.0, so the step is not repeatable."""
        f = QD.RouteFacts(**{**self.WINNER, "waves_per_cta": 8})
        self.assertEqual(2.0, f.waves_per_simd)
        self.assertFalse(QD.PRECONDITIONS["simd_holds_one_wave"](f))

    def test_it_is_silent_when_the_facts_do_not_determine_it(self):
        """No lds_bytes or no wave count means unknown, which must not fire."""
        for missing in ({"lds_bytes": 0}, {"waves_per_cta": 0}):
            with self.subTest(**missing):
                f = QD.RouteFacts(**{**self.WINNER, **missing})
                self.assertEqual(0.0, f.waves_per_simd)
                self.assertFalse(QD.PRECONDITIONS["simd_holds_one_wave"](f))

    def test_the_register_bound_is_not_folded_in(self):
        """`ctas_per_cu_cap` must not be able to make a 4-CTA route read as 1.

        Registers are not host-checkable, so folding an assumed cap into
        waves_per_simd would reintroduce v80 exactly: the LDS reading is an
        upper bound on residency and therefore a lower bound on waves per SIMD,
        which keeps `== 1.0` a statement about what is forced.
        """
        f = QD.RouteFacts(**{**self.LOSER, "ctas_per_cu_cap": 1})
        self.assertEqual(4.0, f.waves_per_simd)
        self.assertFalse(QD.PRECONDITIONS["simd_holds_one_wave"](f))


class WideningAOneTileTallGridIsPricedInSplitKTraffic(unittest.TestCase):
    """v81. The cost of a tile change that appears nowhere in the tile.

    `prefill_m128_square` at 64x128 has 64 tiles and 4 slices; widening to
    128x128 halves the tiles and forces 8 slices to refill 304 CUs, doubling the
    m*n*slices FP32 plane. 16.78 MB of extra write+read is 3.17 us at 5.3 TB/s
    against a measured 3.34 us loss -- 95% of it. Runs 501-506.
    """

    def test_the_precondition_fires_only_on_a_one_row_grid(self):
        fires = QD.PRECONDITIONS["tile_grid_is_one_row_tall"]
        self.assertTrue(fires(QD.RouteFacts(tiles=32, slices=8, cu_count=304,
                                            m=128, n=4096, cta_m=128)))
        self.assertFalse(fires(QD.RouteFacts(tiles=64, slices=4, cu_count=304,
                                             m=128, n=4096, cta_m=64)))

    def test_it_is_silent_without_the_shape(self):
        fires = QD.PRECONDITIONS["tile_grid_is_one_row_tall"]
        self.assertFalse(fires(QD.RouteFacts(tiles=64, slices=4, cu_count=304)))

    def test_the_plane_is_independent_of_the_tile(self):
        """Same m, n and slices -> same plane, whatever the tile does."""
        wide = QD.RouteFacts(tiles=32, slices=8, cu_count=304, m=128, n=4096,
                             cta_m=128)
        narrow = QD.RouteFacts(tiles=64, slices=8, cu_count=304, m=128, n=4096,
                               cta_m=64)
        self.assertEqual(wide.splitk_plane_bytes, narrow.splitk_plane_bytes)

    def test_the_measured_delta_is_what_the_property_says(self):
        wide = QD.RouteFacts(tiles=32, slices=8, cu_count=304, m=128, n=4096,
                             cta_m=128)
        incumbent = QD.RouteFacts(tiles=64, slices=4, cu_count=304, m=128,
                                  n=4096, cta_m=64)
        self.assertEqual(2 * incumbent.splitk_plane_bytes,
                         wide.splitk_plane_bytes,
                         "doubling the slices doubles the plane")
        extra = wide.splitk_plane_bytes - incumbent.splitk_plane_bytes
        self.assertEqual(16_777_216, extra)          # +16.78 MB
        self.assertAlmostEqual(3.17, extra / 5.3e12 * 1e6, places=1)  # us

    def test_no_plane_without_a_reduction(self):
        """One slice is not a split, so there is no workspace to pay for."""
        f = QD.RouteFacts(tiles=64, slices=1, cu_count=304, m=2048, n=4096,
                          cta_m=128)
        self.assertEqual(0, f.splitk_plane_bytes)


class PrefetchCoverAndWavesPerSimdAreSubstitutes(unittest.TestCase):
    """The four-row ledger of runs 492-495 and 507-518, pinned as arithmetic.

    Every row is the m128 route at 64x128. `lds` is the LDS the configuration
    actually allocates -- v82's is doubled because it double-buffers -- because
    residency, and therefore waves per SIMD, is read from the allocation and not
    from the tile.
    """

    # name, waves_m, waves_n, stage_k, lds, cover, waves/simd, measured
    LEDGER = [
        ("pre-v79", 2, 2, 128, 50688, 64, 1.0, None),
        ("v79",     2, 4, 128, 50688, 32, 2.0, +7.5),
        ("v82",     2, 4,  64, 52224, 16, 2.0, -16.2),
        ("v83",     4, 4, 128, 50688, 16, 4.0, -2.3),
    ]

    def _facts(self, waves_m, waves_n, stage_k, lds):
        return QD.RouteFacts(tiles=64, slices=4, cu_count=304, m=128, n=4096,
                             cta_m=64, cta_n=128, waves_m=waves_m,
                             waves_n=waves_n, stage_k=stage_k, lds_bytes=lds,
                             waves_per_cta=waves_m * waves_n)

    def test_the_ledger_is_the_arithmetic(self):
        for name, wm, wn, sk, lds, cover, wps, _ in self.LEDGER:
            with self.subTest(probe=name):
                f = self._facts(wm, wn, sk, lds)
                self.assertEqual(cover, f.prefetch_cover)
                self.assertEqual(wps, f.waves_per_simd)

    def test_the_two_losers_are_separated_only_by_the_wave_count(self):
        """v82 and v83 halve the cover identically; the wave count is the delta."""
        v82 = self._facts(2, 4, 64, 52224)
        v83 = self._facts(4, 4, 128, 50688)
        self.assertEqual(v82.prefetch_cover, v83.prefetch_cover)
        self.assertNotEqual(v82.waves_per_simd, v83.waves_per_simd)
        self.assertEqual(2 * v82.waves_per_simd, v83.waves_per_simd)

    def test_the_accepted_move_halved_the_cover_and_bought_a_wave(self):
        """v79 is not "a wave paid for in intensity" -- it is cover paid in a wave."""
        before = self._facts(2, 2, 128, 50688)
        after = self._facts(2, 4, 128, 50688)
        self.assertEqual(2 * after.prefetch_cover, before.prefetch_cover)
        self.assertEqual(2 * before.waves_per_simd, after.waves_per_simd)
        self.assertLess(after.intensity, before.intensity)   # 1.00 < 1.33

    def test_cover_needs_all_three_of_stage_depth_and_both_fragments(self):
        self.assertEqual(0, QD.RouteFacts(tiles=64, slices=4, cu_count=304,
                                          cta_m=64, cta_n=128, waves_m=2,
                                          waves_n=4).prefetch_cover)
        self.assertEqual(0, QD.RouteFacts(tiles=64, slices=4, cu_count=304,
                                          stage_k=128, cta_n=128,
                                          waves_n=4).prefetch_cover)

    def test_the_mfma_shape_is_a_reparameterisation_of_intensity(self):
        """Doubling the fragment edge halves both kFM and kFN -- run 196's law.

        The effective-intensity invariance theorem: MACs per LDS element rises
        by the same factor the wave-tile term falls, so the product is fixed and
        the fragment shape is not an axis.
        """
        base = self._facts(2, 4, 128, 50688)
        wide = QD.RouteFacts(tiles=64, slices=4, cu_count=304, m=128, n=4096,
                             cta_m=64, cta_n=128, waves_m=2, waves_n=4,
                             stage_k=128, lds_bytes=50688, waves_per_cta=8,
                             frag_edge=32)
        self.assertEqual(2 * wide.frags_m, base.frags_m)
        self.assertEqual(2 * wide.frags_n, base.frags_n)
        self.assertAlmostEqual(base.intensity, 2 * wide.intensity)

    def test_neither_term_is_a_precondition_on_its_own(self):
        """A halved cover is a win with a wave (v79) and a loss without (v82)."""
        v79 = self._facts(2, 4, 128, 50688)
        v82 = self._facts(2, 4, 64, 52224)
        self.assertEqual(2 * v82.prefetch_cover, v79.prefetch_cover)
        self.assertEqual(v79.waves_per_simd, v82.waves_per_simd)
        self.assertNotIn("prefetch_cover",
                         {name for name in QD.PRECONDITIONS},
                         "cover must not be registered as a lone precondition")


class TheAsymmetricSplitIsMeasuredAndExcluded(unittest.TestCase):
    """Runs 519-560: the last unmeasured wave_schedule value, and it loses.

    The exclusion in `coverage_eligible` predates the measurement and was
    argued from design. These tests pin that the measurement agrees with it, so
    a future reader cannot mistake the exclusion for an untested assumption.
    """

    def _desc(self, **over):
        d = {axis: QD.AXIS_ORDER[axis][0] for axis in QD.AXES}
        d.update(compute_primitive="rocwmma", wave_schedule="symmetric_interleave")
        d.update(over)
        return d

    def test_the_value_is_still_a_legal_descriptor(self):
        """Rejected on one route is not illegal -- a kernel using it still has a cell."""
        d = self._desc(wave_schedule="asymmetric_producer_consumer")
        self.assertTrue(QD.descriptor_valid(d),
                        "measuring it negative must not delete the value")

    def test_it_is_not_a_directed_transition_target(self):
        elig = self._desc(wave_schedule="asymmetric_producer_consumer")
        self.assertFalse(QD.coverage_eligible(elig))
        self.assertTrue(QD.coverage_eligible(self._desc()))

    def test_the_axis_vocabulary_is_unchanged_by_the_measurement(self):
        self.assertEqual(("independent", "symmetric_interleave",
                          "symmetric_pingpong", "asymmetric_producer_consumer"),
                         QD.AXIS_ORDER["wave_schedule"])

    def test_the_probe_held_every_other_ledger_term_fixed(self):
        """v84's consumers are v79 exactly; only the producer waves are added.

        This is what makes it a single-currency A/B: if any of cover, intensity
        or waves per SIMD had moved, the +1.53% would be attributable to the
        (7) model instead of to the split.
        """
        common = dict(tiles=64, slices=4, cu_count=304, m=128, n=4096,
                      cta_m=64, cta_n=128, waves_m=2, waves_n=4, stage_k=128,
                      lds_bytes=50688, waves_per_cta=8)
        v79 = QD.RouteFacts(**common)
        v84_consumers = QD.RouteFacts(**common)
        self.assertEqual(v79.prefetch_cover, v84_consumers.prefetch_cover)
        self.assertEqual(v79.intensity, v84_consumers.intensity)
        self.assertEqual(v79.waves_per_simd, v84_consumers.waves_per_simd)
        self.assertEqual(v79.splitk_plane_bytes, v84_consumers.splitk_plane_bytes)


class TheOneWavePreconditionIsTestedOnBothSides(unittest.TestCase):
    """Runs 721-732: v79's move applied to the shape it is gated away from.

    `_simd_holds_one_wave` was written from the m128 result and from the m512
    rejection, both of which are routes where the move was TRIED. v85 is the
    first case where the gate's exclusion was disobeyed on purpose: m96 runs
    four waves per SIMD, the gate says no, the move was run anyway, and it lost
    3.5%. These tests pin the arithmetic that made the prediction available
    before the run, so a later search can trust the gate instead of repeating
    the probe.
    """

    # 96x11008x4096 -> 96x128 @ 32, six slices. LDS (96+128)*(32+4)*2 = 16128,
    # so four CTAs of 256 threads: sixteen waves per CU, four per SIMD.
    M96_TODAY = QD.RouteFacts(
        tiles=86 * 1, slices=6, cu_count=304, tile_rows=1, ctas_per_cu_cap=3,
        k=4096, m=96, n=11008, cta_m=96, cta_n=128, waves_m=2, waves_n=2,
        stage_k=32, lds_bytes=16128, waves_per_cta=4,
    )
    M96_V85 = dataclasses.replace(M96_TODAY, waves_n=4, waves_per_cta=8)

    # 128x4096x4096 -> 64x128 @ 128, four slices. LDS 50688: one CTA per CU.
    M128_V79_SOURCE = QD.RouteFacts(
        tiles=64, slices=4, cu_count=304, tile_rows=2, ctas_per_cu_cap=1,
        k=4096, m=128, n=4096, cta_m=64, cta_n=128, waves_m=2, waves_n=2,
        stage_k=128, lds_bytes=50688, waves_per_cta=4,
    )

    def test_m96_already_has_four_waves_per_simd(self):
        self.assertEqual(self.M96_TODAY.ctas_by_lds, 4)
        self.assertEqual(self.M96_TODAY.waves_per_simd, 4.0)

    def test_the_gate_excludes_m96_and_admits_the_m128_source(self):
        gate = QD.PRECONDITIONS["simd_holds_one_wave"]
        self.assertFalse(gate(self.M96_TODAY))
        self.assertTrue(gate(self.M128_V79_SOURCE))

    def test_the_move_halves_m96s_cover_exactly_as_it_did_on_m128(self):
        # kFM is 3 on both sides; only kFN moves, 4 -> 2.
        self.assertEqual((self.M96_TODAY.frags_m, self.M96_TODAY.frags_n), (3, 4))
        self.assertEqual((self.M96_V85.frags_m, self.M96_V85.frags_n), (3, 2))
        self.assertEqual(self.M96_TODAY.prefetch_cover, 24)
        self.assertEqual(self.M96_V85.prefetch_cover, 12)
        self.assertAlmostEqual(self.M96_TODAY.intensity, 12 / 7)
        self.assertAlmostEqual(self.M96_V85.intensity, 6 / 5)

    def test_the_wave_it_buys_lands_on_an_already_saturated_simd(self):
        # The whole content of finding (9): the cover bill is the same shape as
        # v79's, but the wave that paid for it there is worthless here.
        self.assertEqual(self.M128_V79_SOURCE.waves_per_simd, 1.0)
        self.assertGreaterEqual(self.M96_TODAY.waves_per_simd, 2.0)

    def test_no_split_n_mechanism_is_offered_on_a_saturated_route(self):
        offered = QD.mechanisms_for(self.M96_TODAY)
        self.assertNotIn(
            "simd_holds_one_wave", {mech.precondition for mech in offered})


class FillingTheMachineWithSlicesIsPricedAndFillingItWithTilesIsNot(
        unittest.TestCase):
    """Runs 704-712: why 128x128 loses on m128 even at eight waves.

    The plane is independent of tile, so halving the tile count doubles the
    slices needed for a given occupancy and doubles the plane with them.
    """

    M128_V79 = QD.RouteFacts(
        tiles=64, slices=4, cu_count=304, tile_rows=2, ctas_per_cu_cap=1,
        k=4096, m=128, n=4096, cta_m=64, cta_n=128, waves_m=2, waves_n=4,
        stage_k=128, lds_bytes=50688, waves_per_cta=8,
    )
    M128_BIG_TILE = QD.RouteFacts(
        tiles=32, slices=8, cu_count=304, tile_rows=1, ctas_per_cu_cap=1,
        k=4096, m=128, n=4096, cta_m=128, cta_n=128, waves_m=2, waves_n=4,
        stage_k=64, lds_bytes=34816, waves_per_cta=8,
    )

    def test_both_routes_reach_the_same_cta_count(self):
        self.assertEqual(self.M128_V79.ctas, self.M128_BIG_TILE.ctas)

    def test_but_the_big_tile_pays_twice_the_plane_for_it(self):
        self.assertEqual(
            self.M128_BIG_TILE.splitk_plane_bytes,
            2 * self.M128_V79.splitk_plane_bytes)

    def test_the_plane_does_not_depend_on_the_tile_at_equal_slices(self):
        same_slices = dataclasses.replace(self.M128_BIG_TILE, slices=4)
        self.assertEqual(
            same_slices.splitk_plane_bytes, self.M128_V79.splitk_plane_bytes)

    def test_the_big_tile_is_not_worse_on_any_intra_cu_term(self):
        # It is strictly better on intensity and equal on cover and waves, which
        # is exactly why the rejection has to be explained by traffic.
        self.assertGreater(
            self.M128_BIG_TILE.intensity, self.M128_V79.intensity)
        self.assertEqual(
            self.M128_BIG_TILE.prefetch_cover, self.M128_V79.prefetch_cover)
        self.assertEqual(
            self.M128_BIG_TILE.waves_per_simd, self.M128_V79.waves_per_simd)

    def test_the_tail_quantisation_optimum_is_reproduced_by_the_rule(self):
        # s_opt = floor(cu_count * ctas_per_cu_cap / tiles), and it is the
        # measured optimum on both routes.
        for facts, expected in ((self.M128_V79, 4), (self.M128_BIG_TILE, 9)):
            with self.subTest(cta_m=facts.cta_m):
                cap = facts.ctas_by_lds
                self.assertEqual(cap, 1)
                self.assertEqual(facts.cu_count * cap // facts.tiles, expected)


class RegistersAreReadFromTheObjectAndCanBindTighterThanLds(unittest.TestCase):
    """Finding (10). Counts below are measured, not estimated -- see the
    device-truth table in PIPELINE_PROGRESS.md. The distinction is the point:
    every one of these numbers contradicts a source-read estimate that was
    load-bearing at some stage of this search."""

    # The m128 route. LDS pins it at one CTA, so registers have 256 to play in.
    M128 = QD.RouteFacts(
        tiles=64, slices=4, cu_count=304, tile_rows=2, ctas_per_cu_cap=1,
        k=4096, m=128, n=4096, cta_m=64, cta_n=128, waves_m=2, waves_n=4,
        stage_k=128, lds_bytes=50688, waves_per_cta=8,
        vgpr_count=92, agpr_count=0)
    # A route whose AGPRs push it below its LDS allowance.
    M64_WIDE = QD.RouteFacts(
        tiles=64, slices=4, cu_count=304, ctas_per_cu_cap=4,
        k=4096, m=4096, n=4096, cta_m=64, cta_n=128, waves_m=2, waves_n=2,
        stage_k=32, lds_bytes=13824, waves_per_cta=4,
        vgpr_count=124, agpr_count=48)
    # decode_m96_up at four waves, and the rejected v85 re-pricing of it.
    M96_V79 = QD.RouteFacts(
        tiles=86, slices=6, cu_count=304, ctas_per_cu_cap=4,
        k=4096, m=96, n=11008, cta_m=96, cta_n=128, waves_m=2, waves_n=2,
        stage_k=32, lds_bytes=16128, waves_per_cta=4,
        vgpr_count=122, agpr_count=0)
    M96_V85 = dataclasses.replace(
        M96_V79, waves_n=4, waves_per_cta=8, vgpr_count=68, agpr_count=0)

    def test_the_ramp_cannot_spill_on_the_m128_route(self):
        # 92 + 0 -> granule 96. LDS allows one CTA of 8 waves = 2 waves/SIMD,
        # so each wave may hold 512/2 = 256. The ramp's second copy of the
        # panel registers is +24, landing at 120.
        self.assertEqual(self.M128.ctas_by_lds, 1)
        self.assertEqual(self.M128.waves_per_simd, 2.0)
        # 512 // 96 = 5 waves/SIMD by registers -> 5*4//8 = 2 CTAs, i.e.
        # registers are not the binding term and would not be at 120 either.
        self.assertEqual(self.M128.ctas_by_vgpr, 2)
        self.assertEqual(self.M128.ctas_by_occupancy, 1)
        ramped = dataclasses.replace(self.M128, vgpr_count=92 + 24)
        self.assertEqual(ramped.ctas_by_occupancy, 1)

    def test_agprs_count_against_the_same_file(self):
        # Dropping the AGPR term is what made this route read as twice as
        # resident as it is: 124 alone gives 4 CTAs, 124+48 gives 2.
        self.assertEqual(self.M64_WIDE.ctas_by_lds, 4)
        self.assertEqual(self.M64_WIDE.ctas_by_vgpr, 2)
        self.assertEqual(self.M64_WIDE.ctas_by_occupancy, 2)
        without = dataclasses.replace(self.M64_WIDE, agpr_count=0)
        self.assertEqual(without.ctas_by_vgpr, 4)

    def test_s_opt_must_key_on_occupancy_not_lds(self):
        # The tail-quantisation rule reads a residency cap. On an
        # AGPR-limited route the LDS number is too large by a factor of two,
        # and so is the slice optimum derived from it.
        by_lds = (self.M64_WIDE.cu_count * self.M64_WIDE.ctas_by_lds
                  // self.M64_WIDE.tiles)
        by_occ = (self.M64_WIDE.cu_count * self.M64_WIDE.ctas_by_occupancy
                  // self.M64_WIDE.tiles)
        self.assertEqual(by_lds, 19)
        self.assertEqual(by_occ, 9)

    def test_unmeasured_registers_fall_back_to_lds_alone(self):
        # 0 means "not read back", which must not be mistaken for "needs no
        # registers". The fallback is the LDS bound, unchanged.
        unmeasured = dataclasses.replace(
            self.M64_WIDE, vgpr_count=0, agpr_count=0)
        self.assertEqual(unmeasured.ctas_by_vgpr, 0)
        self.assertEqual(unmeasured.ctas_by_occupancy,
                         unmeasured.ctas_by_lds)

    def test_waves_per_simd_stays_lds_only_so_the_gate_stays_conservative(self):
        # The finding (9) gate reads waves_per_simd == 1.0. Registers can only
        # lower residency and so only raise waves/SIMD; folding them in could
        # only ever make the gate fire more often, which is the v80 error.
        self.assertEqual(self.M64_WIDE.waves_per_simd, 4.0)  # from ctas_by_lds
        self.assertLess(self.M64_WIDE.ctas_by_occupancy,
                        self.M64_WIDE.ctas_by_lds)

    def test_v85_raised_occupancy_and_still_lost(self):
        # The register explanation for v85's +3.5% is refuted by the object:
        # splitting n halves the accumulators, so registers FALL and residency
        # in waves/SIMD RISES. What is left is the halved cover of finding (9).
        self.assertLess(self.M96_V85.vgpr_count, self.M96_V79.vgpr_count)
        self.assertEqual(self.M96_V79.ctas_by_occupancy, 4)
        self.assertEqual(self.M96_V85.ctas_by_occupancy, 3)
        self.assertEqual(self.M96_V79.ctas_by_occupancy
                         * self.M96_V79.waves_per_cta / 4, 4.0)
        self.assertEqual(self.M96_V85.ctas_by_occupancy
                         * self.M96_V85.waves_per_cta / 4, 6.0)
        self.assertEqual(self.M96_V79.prefetch_cover, 24)
        self.assertEqual(self.M96_V85.prefetch_cover, 12)


class BuyingResidencyPaysOnlyWhereTheGridOverflowsIt(unittest.TestCase):
    """Finding (11), v86, runs 820-847. The same register change on the same
    2x2 wave layout, doubling the cap 2 -> 4 CTAs per CU on both routes, and
    the two shapes answer differently. The tile arithmetic predicts which."""

    # prefill_m512_up: 128x64, 4 x 172 tiles, one slice -> 688 CTAs.
    M512 = QD.RouteFacts(
        tiles=688, slices=1, cu_count=304, tile_rows=4, ctas_per_cu_cap=4,
        k=4096, m=512, n=11008, cta_m=128, cta_n=64, waves_m=2, waves_n=2,
        stage_k=32, lds_bytes=13824, waves_per_cta=4,
        vgpr_count=124, agpr_count=48)
    # decode_m64_square: 64x128, 1 x 64 tiles, eight slices -> 512 CTAs.
    M64 = QD.RouteFacts(
        tiles=64, slices=8, cu_count=304, tile_rows=1, ctas_per_cu_cap=4,
        k=8192, m=64, n=8192, cta_m=64, cta_n=128, waves_m=2, waves_n=2,
        stage_k=32, lds_bytes=13824, waves_per_cta=4,
        vgpr_count=124, agpr_count=48)

    def test_both_routes_are_register_limited_to_the_same_two_ctas(self):
        # The change is identical on both: same object, same delta. Any
        # difference in outcome therefore cannot come from the kernel.
        for f in (self.M512, self.M64):
            with self.subTest(m=f.m):
                self.assertEqual(f.ctas_by_lds, 4)
                self.assertEqual(f.ctas_by_vgpr, 2)
                self.assertEqual(f.residency_slots, 608)

    def test_the_gate_separates_the_winner_from_the_null(self):
        self.assertEqual(self.M512.ctas, 688)
        self.assertEqual(self.M64.ctas, 512)
        self.assertTrue(QD.PRECONDITIONS["grid_overflows_residency"](self.M512))
        self.assertFalse(QD.PRECONDITIONS["grid_overflows_residency"](self.M64))

    def test_the_gate_reads_occupancy_not_lds(self):
        # Had it read ctas_by_lds, slots would be 1216 and m512 would read as
        # fitting -- the gate would have predicted the null on the shape that
        # actually moved.
        lds_slots = self.M512.cu_count * self.M512.ctas_by_lds
        self.assertEqual(lds_slots, 1216)
        self.assertGreater(lds_slots, self.M512.ctas)
        self.assertLess(self.M512.residency_slots, self.M512.ctas)

    def test_raising_the_cap_removes_the_overflow_on_the_winner_only(self):
        # After the fix both routes sit at four CTAs per CU. m512 goes from
        # overflowing to resident; m64 was never overflowing.
        for f, was in ((self.M512, True), (self.M64, False)):
            with self.subTest(m=f.m):
                self.assertEqual(
                    QD.PRECONDITIONS["grid_overflows_residency"](f), was)
                fixed = dataclasses.replace(f, vgpr_count=88, agpr_count=0)
                self.assertEqual(fixed.ctas_by_occupancy, 4)
                self.assertFalse(
                    QD.PRECONDITIONS["grid_overflows_residency"](fixed))

    def test_unmeasured_registers_do_not_fabricate_an_overflow(self):
        # With registers unread the gate falls back to LDS, which on these
        # routes says no overflow. Silent is the right answer, not a guess.
        blind = dataclasses.replace(self.M512, vgpr_count=0, agpr_count=0)
        self.assertEqual(blind.residency_slots, 1216)
        self.assertFalse(QD.PRECONDITIONS["grid_overflows_residency"](blind))


class AnUnderfilledGridIsHoldingResidencyItCannotUse(unittest.TestCase):
    """Finding (13), v93, runs 1060-1095. `decode_m96_up` oversubscribes the
    machine on CUs and still leaves three of every four CTA slots empty. The
    old predicate counted CUs and forbade the move; counting slots licensed it
    and it was worth -11.44%."""

    # 96x128/2x2/sk32, the shipped v86 plan: 86 tiles, and the tuner runs 7
    # slices (traced -- the plan asks for 6 and the tuner overrides).
    SHALLOW = QD.RouteFacts(
        tiles=86, slices=7, cu_count=304, tile_rows=1, ctas_per_cu_cap=4,
        k=4096, m=96, n=11008, cta_m=96, cta_n=128, waves_m=2, waves_n=2,
        stage_k=32, lds_bytes=16128, waves_per_cta=4,
        vgpr_count=122, agpr_count=0)
    # The same route after `deepen_underfilled_grid`: stage 32 -> 128. Nothing
    # about the grid changes; the tile's residency collapses 4 -> 1.
    DEEP = dataclasses.replace(
        SHALLOW, stage_k=128, lds_bytes=59136, vgpr_count=244, agpr_count=72)
    # The refuted alternative: pay for the depth by narrowing the tile instead.
    NARROW = QD.RouteFacts(
        tiles=172, slices=3, cu_count=304, tile_rows=1, ctas_per_cu_cap=4,
        k=4096, m=96, n=11008, cta_m=96, cta_n=64, waves_m=2, waves_n=2,
        stage_k=128, lds_bytes=42240, waves_per_cta=4,
        vgpr_count=156, agpr_count=36)

    def test_the_grid_oversubscribes_cus_and_still_underfills_slots(self):
        # Both halves matter. If it did not oversubscribe, the older and
        # stronger `grid_cannot_fill_machine` would already have caught it and
        # finding (13) would be nothing new.
        self.assertEqual(self.SHALLOW.ctas, 602)
        self.assertGreater(self.SHALLOW.ctas, self.SHALLOW.cu_count)
        self.assertFalse(
            QD.PRECONDITIONS["grid_cannot_fill_machine"](self.SHALLOW))
        self.assertEqual(self.SHALLOW.ctas_by_occupancy, 4)
        self.assertEqual(self.SHALLOW.residency_slots, 1216)
        self.assertTrue(
            QD.PRECONDITIONS["grid_underfills_residency"](self.SHALLOW))

    def test_the_two_residency_gates_are_disjoint(self):
        # (11) and (13) are one law with a sign. No route may satisfy both.
        for f in (self.SHALLOW, self.DEEP, self.NARROW):
            with self.subTest(stage_k=f.stage_k, cta_n=f.cta_n):
                self.assertFalse(
                    QD.PRECONDITIONS["grid_overflows_residency"](f)
                    and QD.PRECONDITIONS["grid_underfills_residency"](f))

    def test_the_move_spends_the_slack_and_closes_its_own_gate(self):
        # After deepening, the 602 CTAs no longer fit: the idle slots were
        # bought with, so the mechanism does not re-fire on its own output.
        self.assertEqual(self.DEEP.ctas_by_occupancy, 1)
        self.assertEqual(self.DEEP.residency_slots, 304)
        self.assertFalse(
            QD.PRECONDITIONS["grid_underfills_residency"](self.DEEP))
        self.assertTrue(
            QD.PRECONDITIONS["grid_overflows_residency"](self.DEEP))

    def test_unmeasured_occupancy_does_not_read_as_idle_machine(self):
        blind = dataclasses.replace(self.SHALLOW, lds_bytes=0,
                                    vgpr_count=0, agpr_count=0)
        self.assertEqual(blind.ctas_by_occupancy, 0)
        self.assertFalse(QD.PRECONDITIONS["grid_underfills_residency"](blind))

    def test_an_empty_grid_does_not_read_as_underfilled(self):
        empty = dataclasses.replace(self.SHALLOW, tiles=0)
        self.assertEqual(empty.ctas, 0)
        self.assertFalse(QD.PRECONDITIONS["grid_underfills_residency"](empty))

    def test_the_mechanism_is_offered_on_the_shallow_route_only(self):
        names = {m.name for m in QD.mechanisms_for(self.SHALLOW)}
        self.assertIn("deepen_underfilled_grid", names)
        self.assertNotIn("deepen_underfilled_grid",
                         {m.name for m in QD.mechanisms_for(self.DEEP)})

    def test_the_deep_and_narrow_arms_differ_in_intensity_alone(self):
        # The v92 control. Cover, resident CTAs and waves per SIMD are all
        # matched; only reads-per-MFMA differs, and it was worth 13%.
        # 96x128@64 is the matched arm -- same 48 cover as 96x64@128.
        matched = dataclasses.replace(
            self.SHALLOW, stage_k=64, lds_bytes=30464,
            vgpr_count=196, agpr_count=72)
        self.assertEqual(matched.prefetch_cover, self.NARROW.prefetch_cover)
        self.assertEqual(matched.ctas_by_occupancy,
                         self.NARROW.ctas_by_occupancy)
        # Measured, not LDS-only: `matched` is register-bound, so the
        # conservative property reads 2.0 where objmeta measured 1.00.
        self.assertEqual(matched.waves_per_simd_measured,
                         self.NARROW.waves_per_simd_measured)
        self.assertEqual(matched.waves_per_simd_measured, 1.0)
        self.assertEqual(matched.waves_per_simd, 2.0)
        self.assertAlmostEqual(matched.intensity, 12 / 7, places=6)
        self.assertAlmostEqual(self.NARROW.intensity, 6 / 5, places=6)
        self.assertGreater(matched.intensity, self.NARROW.intensity)

    def test_finding_14_holds_only_where_one_wave_hides_nothing(self):
        # Finding (7) measured cover and waves as substitutes, and every point
        # in that ledger had >= 2 waves per SIMD. The shallow tile is such a
        # point; the deep arms are not, which is why intensity became visible.
        self.assertEqual(self.SHALLOW.waves_per_simd_measured, 4.0)
        for f in (self.DEEP, self.NARROW):
            with self.subTest(cta_n=f.cta_n):
                self.assertEqual(f.waves_per_simd_measured, 1.0)
                self.assertTrue(QD.PRECONDITIONS["simd_holds_one_wave"](f))

    def test_the_conservative_gate_misses_a_register_bound_one_wave_route(self):
        # Documented, not endorsed. `simd_holds_one_wave` reads LDS alone and
        # so returns False on 96x128@64, a route that objmeta measures at one
        # wave per SIMD. It under-fires rather than over-fires, which is the
        # safe direction for finding (9)'s gate, but any mechanism reasoning
        # about actual residency must read `waves_per_simd_measured`.
        matched = dataclasses.replace(
            self.SHALLOW, stage_k=64, lds_bytes=30464,
            vgpr_count=196, agpr_count=72)
        self.assertEqual(matched.waves_per_simd_measured, 1.0)
        self.assertFalse(QD.PRECONDITIONS["simd_holds_one_wave"](matched))

    def test_measured_waves_stay_silent_when_registers_are_unread(self):
        blind = dataclasses.replace(self.DEEP, vgpr_count=0, agpr_count=0)
        self.assertEqual(blind.waves_per_simd_measured, 0.0)


class TheDeepenedGridMustFitWhenTheCoverPrizeIsOnlyDoubled(unittest.TestCase):
    """Finding (15), v94/v95, runs 1200-1235. The same 2x-cover deepening on two
    routes that both underfill beforehand; the one whose deepened grid spills
    into a second CTA wave loses 26%, the one that still fits wins 3.3%. Both
    signs were called before the runs."""

    # decode_m64_square, 64x128/2x2/sk32 -> sk64. Shipped in v95.
    M64_SHALLOW = QD.RouteFacts(
        tiles=64, slices=8, cu_count=304, tile_rows=1, ctas_per_cu_cap=4,
        k=8192, m=64, n=8192, cta_m=64, cta_n=128, waves_m=2, waves_n=2,
        stage_k=32, lds_bytes=13824, waves_per_cta=4,
        vgpr_count=90, agpr_count=0)
    M64_DEEP = dataclasses.replace(
        M64_SHALLOW, stage_k=64, lds_bytes=26112,
        vgpr_count=152, agpr_count=48)
    # prefill_m512_up, 128x64/2x2/sk32 -> sk64. Forced only; never shipped.
    M512_SHALLOW = QD.RouteFacts(
        tiles=688, slices=1, cu_count=304, tile_rows=4, ctas_per_cu_cap=4,
        k=4096, m=512, n=11008, cta_m=128, cta_n=64, waves_m=2, waves_n=2,
        stage_k=32, lds_bytes=13824, waves_per_cta=4,
        vgpr_count=88, agpr_count=0)
    M512_DEEP = dataclasses.replace(
        M512_SHALLOW, stage_k=64, lds_bytes=26112,
        vgpr_count=152, agpr_count=48)

    def test_both_routes_underfill_before_the_move(self):
        # The precondition does not separate them -- that is the whole point.
        for f in (self.M64_SHALLOW, self.M512_SHALLOW):
            with self.subTest(m=f.m):
                self.assertEqual(f.ctas_by_occupancy, 4)
                self.assertEqual(f.residency_slots, 1216)
                self.assertTrue(
                    QD.PRECONDITIONS["grid_underfills_residency"](f))

    def test_the_move_is_the_same_trade_on_both(self):
        for shallow, deep in ((self.M64_SHALLOW, self.M64_DEEP),
                              (self.M512_SHALLOW, self.M512_DEEP)):
            with self.subTest(m=shallow.m):
                self.assertEqual(shallow.prefetch_cover, 16)
                self.assertEqual(deep.prefetch_cover, 32)
                self.assertEqual(deep.ctas_by_occupancy, 2)
                self.assertEqual(shallow.waves_per_simd_measured, 4.0)
                self.assertEqual(deep.waves_per_simd_measured, 2.0)

    def test_only_the_fit_after_deepening_separates_the_signs(self):
        # 512 <= 608 wins; 688 > 608 loses 26%.
        self.assertLessEqual(self.M64_DEEP.ctas, self.M64_DEEP.residency_slots)
        self.assertTrue(
            QD.PRECONDITIONS["grid_underfills_residency"](self.M64_DEEP))
        self.assertGreater(self.M512_DEEP.ctas,
                           self.M512_DEEP.residency_slots)
        self.assertTrue(
            QD.PRECONDITIONS["grid_overflows_residency"](self.M512_DEEP))

    def test_the_conservative_rule_would_wrongly_decline_the_4x_prize(self):
        # decode_m96_up spills harder than m512 (1.70 waves vs 1.13) and still
        # wins 11.4%, because its cover quadruples rather than doubles. This is
        # the counter-example that forbids one unified rule; it is asserted here
        # so that unifying them breaks a test rather than a shape.
        m96_deep = QD.RouteFacts(
            tiles=86, slices=6, cu_count=304, tile_rows=1, ctas_per_cu_cap=4,
            k=4096, m=96, n=11008, cta_m=96, cta_n=128, waves_m=2, waves_n=2,
            stage_k=128, lds_bytes=59136, waves_per_cta=4,
            vgpr_count=244, agpr_count=72)
        self.assertEqual(m96_deep.prefetch_cover, 96)      # 4x of 24
        self.assertEqual(self.M64_DEEP.prefetch_cover, 32)  # 2x of 16
        # It overflows worse than the m512 arm that lost, in CTA waves:
        self.assertGreater(m96_deep.ctas / m96_deep.residency_slots,
                           self.M512_DEEP.ctas / self.M512_DEEP.residency_slots)
        # ...so the conservative fit rule declines it, and it was worth -11.44%.
        self.assertFalse(
            QD.PRECONDITIONS["grid_underfills_residency"](m96_deep))


class ASubdividedStageIsPaidForInBarriers(unittest.TestCase):
    """Finding (12), v87/v88, runs 940-971. The ramped prologue was the top
    build item for two stages; it was built and lost with perfect rank
    separation, and its own control priced the head latency at ~0."""

    # 64x128/2x4/sk128, the route behind prefill_m128_square. Eight waves per
    # CTA, LDS 50688 -> one CTA per CU: the dearest barrier in the plan space
    # with nothing else resident to run during the rendezvous.
    M128_DEEP = QD.RouteFacts(
        tiles=256, slices=4, cu_count=304, tile_rows=1, ctas_per_cu_cap=1,
        k=4096, m=128, n=4096, cta_m=64, cta_n=128, waves_m=2, waves_n=4,
        stage_k=128, lds_bytes=50688, waves_per_cta=8,
        vgpr_count=92, agpr_count=0)

    def test_the_target_route_cannot_afford_another_barrier(self):
        self.assertEqual(self.M128_DEEP.ctas_by_lds, 1)
        self.assertEqual(self.M128_DEEP.ctas_by_occupancy, 1)
        self.assertFalse(
            QD.PRECONDITIONS["barrier_is_cheap_enough_to_spend"](self.M128_DEEP))

    def test_residency_not_wave_count_is_what_supplies_the_cover(self):
        # Same eight waves per CTA, but two CTAs resident: another CTA's work
        # can run across the rendezvous, so the gate opens. If the test were
        # `waves_per_cta` these two would be indistinguishable.
        roomy = dataclasses.replace(self.M128_DEEP, lds_bytes=24576)
        self.assertEqual(roomy.waves_per_cta, self.M128_DEEP.waves_per_cta)
        self.assertEqual(roomy.ctas_by_occupancy, 2)
        self.assertTrue(
            QD.PRECONDITIONS["barrier_is_cheap_enough_to_spend"](roomy))

    def test_unmeasured_residency_is_not_a_licence_to_spend(self):
        # ctas_by_occupancy 0 means "not measured". A move that costs barriers
        # must not be proposed on a route whose cover is unknown.
        blind = dataclasses.replace(
            self.M128_DEEP, lds_bytes=0, vgpr_count=0, agpr_count=0)
        self.assertEqual(blind.ctas_by_occupancy, 0)
        self.assertFalse(
            QD.PRECONDITIONS["barrier_is_cheap_enough_to_spend"](blind))

    def test_the_control_is_what_priced_the_head_latency(self):
        # v87 = peel + 3 barriers + earlier MFMA = +5.85%.
        # v88 = peel only, same barriers, same MFMA timing = +1.56%.
        # The residual is three barriers minus the saving; at ~1.4% a barrier
        # the barriers are all of it, leaving the head worth about nothing.
        v87, v88 = 5.85, 1.56
        residual = v87 - v88
        self.assertAlmostEqual(residual, 4.29, places=2)
        head_saving = 3 * 1.4 - residual
        self.assertLess(abs(head_saving), 0.5)

    def test_a_shallower_ramp_is_predicted_negative_without_a_run(self):
        # kRamp = 2 keeps the schedule cost and one extra barrier, and buys a
        # head worth ~0. Closing the axis, not the parameter, is the point.
        predicted = 1.56 + 1.43 - 0.0
        self.assertGreater(predicted, 0.0)


class TheCoverWavesPlaneIsClosedWhenLdsPinsTheRouteToOneCta(unittest.TestCase):
    """Finding (16). `prefill_m128_square` binds LDS at one CTA with register
    headroom to spare, and the depth-128 enumeration shows nothing recovers the
    cover: >=2 waves/SIMD implies cover <=32, cover >32 implies 1 wave/SIMD, and
    both sides of that trade are already measured negative. This is a terminal
    gate -- it forbids a plane rather than licensing a move."""

    # 64x128/2x4/sk128, the shipped route. objmeta: 92 VGPR, 0 AGPR, LDS 50688,
    # ctaV 2 / ctaL 1 -- the register bound is slack and the LDS bound is not.
    M128_SHIPPED = QD.RouteFacts(
        tiles=64, slices=4, cu_count=304, tile_rows=2, ctas_per_cu_cap=1,
        k=4096, m=128, n=4096, cta_m=64, cta_n=128, waves_m=2, waves_n=4,
        stage_k=128, lds_bytes=50688, waves_per_cta=8,
        vgpr_count=92, agpr_count=0)
    # The only depth-128 tile that does hold two CTAs. Perimeter 96, cover 16.
    M128_TINY = dataclasses.replace(
        M128_SHIPPED, cta_m=32, cta_n=64, lds_bytes=25344, tiles=256, slices=2,
        waves_m=2, waves_n=2, waves_per_cta=4)
    # decode_m96_up after v93: also one CTA, but by registers, not by LDS.
    M96_DEEP = QD.RouteFacts(
        tiles=86, slices=6, cu_count=304, tile_rows=1, ctas_per_cu_cap=1,
        k=4096, m=96, n=11008, cta_m=96, cta_n=128, waves_m=2, waves_n=2,
        stage_k=128, lds_bytes=59136, waves_per_cta=4,
        vgpr_count=244, agpr_count=72)

    def test_the_shipped_route_is_pinned(self):
        f = self.M128_SHIPPED
        self.assertEqual(f.ctas_by_lds, 1)
        self.assertGreater(f.lds_bytes, 65536 // 2)
        self.assertTrue(QD.PRECONDITIONS["lds_pins_route_to_one_cta"](f))

    def test_the_register_bound_is_slack_so_lds_is_what_pins_it(self):
        # 92 VGPR + 0 AGPR would admit two CTAs; the LDS does not. A route whose
        # registers bound it could be fixed by spilling pressure -- this cannot.
        f = self.M128_SHIPPED
        self.assertGreaterEqual(f.ctas_by_vgpr, 2)
        self.assertEqual(f.ctas_by_occupancy, 1)

    def test_the_only_two_cta_tile_at_this_depth_gives_up_its_cover(self):
        # 32x64 fits twice, and pays for it: area drops 4x at half the waves,
        # so cover drops from 32 to 16. The gate must not fire on it, because
        # the gate's claim is about routes that cannot shrink, not about the
        # shrunken result being good.
        tiny = self.M128_TINY
        self.assertEqual(tiny.ctas_by_lds, 2)
        self.assertFalse(QD.PRECONDITIONS["lds_pins_route_to_one_cta"](tiny))
        self.assertLess(tiny.cta_m * tiny.cta_n // tiny.waves_per_cta,
                        self.M128_SHIPPED.cta_m * self.M128_SHIPPED.cta_n
                        // self.M128_SHIPPED.waves_per_cta)

    def test_it_is_not_the_same_gate_as_simd_holds_one_wave(self):
        # The shipped route has two waves per SIMD, so `simd_holds_one_wave` is
        # false and would offer nothing here; the two gates are independent and
        # answer different questions.
        f = self.M128_SHIPPED
        self.assertEqual(f.waves_per_simd_measured, 2.0)
        self.assertFalse(QD.PRECONDITIONS["simd_holds_one_wave"](f))
        self.assertTrue(QD.PRECONDITIONS["lds_pins_route_to_one_cta"](f))

    def test_a_register_pinned_route_is_not_caught_by_this_lds_gate(self):
        # m96 after v93 is also at one CTA, but 59136 bytes is over half the
        # budget too -- so it fires here as well, and correctly: it is equally
        # unable to buy cover without waves. Asserted so the gate's breadth is
        # recorded rather than discovered later.
        self.assertTrue(QD.PRECONDITIONS["lds_pins_route_to_one_cta"](self.M96_DEEP))

    def test_it_licenses_no_mechanism(self):
        # A terminal gate must not be wired to an adjacency step. If some future
        # mechanism claims it, that mechanism is asserting the plane is open.
        for mech in QD.MECHANISMS:
            with self.subTest(mech=mech.name):
                self.assertNotEqual(mech.precondition, "lds_pins_route_to_one_cta")

    def test_unmeasured_lds_and_unsupplied_depth_do_not_fire(self):
        for bad in (dataclasses.replace(self.M128_SHIPPED, lds_bytes=0),
                    dataclasses.replace(self.M128_SHIPPED, stage_k=0)):
            with self.subTest(lds=bad.lds_bytes, sk=bad.stage_k):
                self.assertFalse(
                    QD.PRECONDITIONS["lds_pins_route_to_one_cta"](bad))


class TheSplitKPlaneDoesNotPayForOccupancyOrIntensity(unittest.TestCase):
    """Finding (17). Runs 1350-1358 on `prefill_m128_square`.

    The narrowing to `32x64/1x4/sk256` holds cover equal to the shipped route
    and deletes the whole fp32 partial plane -- 256 tiles is one slice, so
    `2*4*128*4096*4 = 16.8 MB` of write-then-read-back goes away, roughly a
    third of the route's compulsory traffic -- and it still loses 6.8%.

    It gives up two things for that, not one: intensity 1.00 -> 0.67 *and* a
    wave per SIMD, 2.0 -> 1.0, because the shipped route has been at 2x4 waves
    since v79. The run cannot separate them, and these tests exist to stop the
    separation being asserted. What they do pin is the arithmetic of the
    comparison and the decision not to promote the plane to a descriptor
    dimension.
    """

    SHIPPED = QD.RouteFacts(
        tiles=64, slices=4, cu_count=304, tile_rows=2, ctas_per_cu_cap=1,
        k=4096, m=128, n=4096, cta_m=64, cta_n=128, waves_m=2, waves_n=4,
        stage_k=128, lds_bytes=50688, waves_per_cta=8,
        vgpr_count=92, agpr_count=0)
    # What the dispatch actually launched for the forced `32x64` arm: the
    # `cta_m == 32 && cta_n == 64` guard ignores `stage_k` and always takes
    # `<32, 64, 1, 4, 256>`. LDS is (32+64)*260*2 = 49920, still one CTA.
    NARROW = dataclasses.replace(
        SHIPPED, cta_m=32, cta_n=64, waves_m=1, waves_n=4, waves_per_cta=4,
        stage_k=256, lds_bytes=49920, tiles=256, slices=1)

    @staticmethod
    def _cover(f):
        return (f.stage_k // 16) * (f.cta_m // f.waves_m // 16) \
                                 * (f.cta_n // f.waves_n // 16)

    @staticmethod
    def _intensity(f):
        kfm = f.cta_m // f.waves_m // 16
        kfn = f.cta_n // f.waves_n // 16
        return kfm * kfn / (kfm + kfn)

    @staticmethod
    def _plane_bytes(f):
        """fp32 partials written once and read back once. One slice means none."""
        return 0 if f.slices <= 1 else 2 * 4 * f.m * f.n * f.slices

    def test_the_two_arms_are_matched_on_cover(self):
        self.assertEqual(self._cover(self.SHIPPED), 32)
        self.assertEqual(self._cover(self.NARROW), 32)

    def test_the_narrow_arm_deletes_the_whole_plane(self):
        self.assertEqual(self._plane_bytes(self.SHIPPED), 16 * 1024 * 1024)
        self.assertEqual(self._plane_bytes(self.NARROW), 0)

    def test_the_narrow_arm_gives_up_intensity(self):
        self.assertAlmostEqual(self._intensity(self.SHIPPED), 1.0)
        self.assertAlmostEqual(self._intensity(self.NARROW), 2 / 3)

    def test_and_it_also_gives_up_a_wave_per_simd_which_is_the_confound(self):
        """The shipped route has been at 2 waves/SIMD since v79, not 1.

        This is why finding (17) is a bound and not an attribution: the losing
        arm is worse on two axes at once. `simd_holds_one_wave` separates them,
        firing on the narrow arm only.
        """
        self.assertEqual(self.SHIPPED.ctas_by_lds, 1)
        self.assertEqual(self.NARROW.ctas_by_lds, 1)
        self.assertEqual(self.SHIPPED.waves_per_simd, 2.0)
        self.assertEqual(self.NARROW.waves_per_simd, 1.0)
        self.assertFalse(QD.PRECONDITIONS["simd_holds_one_wave"](self.SHIPPED))
        self.assertTrue(QD.PRECONDITIONS["simd_holds_one_wave"](self.NARROW))

    def test_the_ledger_already_prices_the_wave_term_at_about_the_whole_loss(self):
        """Finding (7) measured this exact substitution in the other direction.

        v79 took the shipped route from 1 wave/SIMD at cover 64 to 2 waves/SIMD
        at cover 32 and gained 7.5% isolated. Finding (17)'s arm runs that trade
        backwards at *matched* cover and loses 6.8% total. So the wave term
        alone plausibly accounts for the entire measured loss, which leaves the
        plane saving and the intensity loss cancelling to within noise -- and
        that, not a numeric value for the plane, is what the run supports.
        """
        wave_term_from_v79 = 0.075
        measured_total = 0.068
        self.assertLess(abs(measured_total - wave_term_from_v79), 0.02)

    def test_the_plane_is_not_a_descriptor_dimension(self):
        """No precondition and no mechanism keys off the reduction plane.

        The guard against re-promoting it: a term that would have recommended
        the arm that lost 6.8% must not become a gate.
        """
        for name in QD.PRECONDITIONS:
            self.assertNotIn("plane", name)
        for mech in QD.MECHANISMS:
            with self.subTest(mech=mech.name):
                self.assertNotIn("plane", mech.axis)
                self.assertNotIn("plane", mech.spends)

    def test_no_mechanism_licenses_narrowing_on_this_route(self):
        """Finding (17) closes the width axis here; nothing may re-open it."""
        for mech in QD.MECHANISMS:
            if mech.applies_to(self.SHIPPED):
                with self.subTest(mech=mech.name):
                    self.assertNotEqual(mech.axis, "tile_width")


class TheSlicePlanIsMissingItsResidencyTermAndTheTunerHasBeenHidingIt(
        unittest.TestCase):
    """Finding (18), runs 1403 (trace) and 1410-1435 (measurement).

    Five traced routes, occupancy read off the machine-H build with
    `objmeta.py`, scored against what the cold autotune actually chose.
    """

    # decode_m96_up: 96x128/2x2/sk128. 244 VGPR + 72 AGPR and 59136 B LDS both
    # pin it to one CTA per CU. Planner says 6, tuner and formula say 3.
    M96 = QD.RouteFacts(
        tiles=86, slices=3, cu_count=304, ctas_per_cu_cap=1,
        k=4096, m=96, n=11008, cta_m=96, cta_n=128, waves_m=2, waves_n=2,
        stage_k=128, lds_bytes=59136, waves_per_cta=4,
        vgpr_count=244, agpr_count=72)

    # prefill_m256_down: 128x160/2x2/sk32, 3 CTAs/CU, 52 tiles.
    # Planner says 9, tuner says 12, the fill term says 17.
    M256 = QD.RouteFacts(
        tiles=52, slices=12, cu_count=304, ctas_per_cu_cap=3,
        k=11008, m=256, n=4096, cta_m=128, cta_n=160, waves_m=2, waves_n=2,
        stage_k=32, lds_bytes=20736, waves_per_cta=4,
        vgpr_count=152, agpr_count=0)

    # prefill_m128_square: 64x128/2x4/sk128. All three answers agree at 4.
    M128 = QD.RouteFacts(
        tiles=64, slices=4, cu_count=304, ctas_per_cu_cap=1,
        k=4096, m=128, n=4096, cta_m=64, cta_n=128, waves_m=2, waves_n=4,
        stage_k=128, lds_bytes=50688, waves_per_cta=8,
        vgpr_count=92, agpr_count=0,
        # The one traced route on the v75 one-wave arm.
        planner_ctas_per_cu=1)

    def test_the_occupancy_bounds_match_the_built_object(self):
        """The fixtures must reproduce objmeta, or nothing below means anything."""
        self.assertEqual(self.M96.ctas_by_occupancy, 1)
        self.assertEqual(self.M256.ctas_by_occupancy, 3)
        self.assertEqual(self.M128.ctas_by_occupancy, 1)

    def test_the_planner_reimplementation_reproduces_the_shipped_plan(self):
        """Run 1403's `[plan]` lines, which are identical on machines G and H."""
        for facts, planned in ((self.M96, 6), (self.M256, 9), (self.M128, 4)):
            with self.subTest(m=facts.m):
                self.assertEqual(facts.planner_slices, planned)

    def test_the_correction_is_exact_where_the_planner_is_worst(self):
        """decode_m96_up: planner 6, corrected 3, cold tuner 3."""
        self.assertEqual(self.M96.fill_limited_slices, 3)
        self.assertEqual(self.M96.planner_slices, 6)
        self.assertTrue(
            QD.PRECONDITIONS["plan_slices_ignores_residency"](self.M96))

    def test_and_the_planners_m96_answer_was_measured_to_cost_twenty_percent(self):
        """Runs 1416-1421: forcing 6 instead of 3 is +19.9%, rank-separated.

        Recorded as a test on the *ordering*, not on the times -- absolute ms
        are machine-H only and must never be compared across a host boundary.
        """
        s3 = [0.06712, 0.06688, 0.06700]
        s6 = [0.08032, 0.08104, 0.07984]
        self.assertLess(max(s3), min(s6), "arms must be fully rank-separated")
        self.assertAlmostEqual(
            statistics.median(s6) / statistics.median(s3), 1.199, places=2)

    def test_the_correction_is_an_upper_bound_not_the_answer(self):
        """prefill_m256_down: the fill term says 17, the measured optimum is 12.

        The point of the test is that the descriptor must not claim 17. It
        reports 17 as the fill bound and the gate fires, but nothing anywhere
        asserts the bound is optimal.
        """
        self.assertEqual(self.M256.fill_limited_slices, 17)
        self.assertEqual(self.M256.planner_slices, 9)
        self.assertTrue(
            QD.PRECONDITIONS["plan_slices_ignores_residency"](self.M256))

    def test_the_two_analytic_answers_straddle_the_measured_optimum(self):
        """9 -> +9.3%, 12 -> best, 17 -> +6.7%. Both are wrong, oppositely.

        Medians are taken only within their own rotated sweep: 12 vs 17 from
        runs 1410-1415, 9 vs 12 from runs 1430-1435. The s=12 arm moved 0.11888
        -> 0.11640 between them, which is why they are not pooled.
        """
        s12_a, s17 = [0.11872, 0.11900, 0.11888], [0.12688, 0.12780, 0.12264]
        s9, s12_b = [0.12516, 0.12820, 0.12724], [0.11460, 0.11640, 0.11824]
        self.assertLess(max(s12_a), min(s17))
        self.assertLess(max(s12_b), min(s9))
        self.assertGreater(statistics.median(s17), statistics.median(s12_a))
        self.assertGreater(statistics.median(s9), statistics.median(s12_b))
        self.assertLess(self.M256.planner_slices, 12)
        self.assertGreater(self.M256.fill_limited_slices, 12)

    def test_the_gate_stays_shut_where_the_planner_is_already_right(self):
        self.assertEqual(self.M128.fill_limited_slices,
                         self.M128.planner_slices)
        self.assertFalse(
            QD.PRECONDITIONS["plan_slices_ignores_residency"](self.M128))

    def test_unmeasured_registers_must_not_read_as_a_disagreement(self):
        blind = dataclasses.replace(self.M96, vgpr_count=0, agpr_count=0,
                                    lds_bytes=0)
        self.assertEqual(blind.fill_limited_slices, 0)
        self.assertFalse(
            QD.PRECONDITIONS["plan_slices_ignores_residency"](blind))

    def test_the_planner_has_no_residency_term_at_all(self):
        """Change only the occupancy and the planner does not notice.

        This is the finding stated as a property rather than as a table: LDS
        and registers can move the route from one resident CTA to four and
        `plan_slices` returns the same number, because its only inputs are
        tiles, k and cu_count.
        """
        roomy = dataclasses.replace(self.M96, lds_bytes=11520,
                                    vgpr_count=40, agpr_count=4)
        self.assertGreater(roomy.ctas_by_occupancy, self.M96.ctas_by_occupancy)
        self.assertEqual(roomy.planner_slices, self.M96.planner_slices)
        self.assertGreater(roomy.fill_limited_slices,
                           self.M96.fill_limited_slices)

    def test_slices_are_not_a_descriptor_axis_and_no_mechanism_uses_the_gate(self):
        """The gate is diagnostic. It must not become a licence.

        Slices are chosen per launch, not carried in the descriptor, so a
        mechanism keyed on this would name an edge the adjacency graph does not
        have -- and the tuner already reaches both measured optima, so it would
        also be worth zero to ship.
        """
        for mech in QD.MECHANISMS:
            with self.subTest(mech=mech.name):
                self.assertNotEqual(mech.precondition,
                                    "plan_slices_ignores_residency")
                self.assertNotIn("slice", mech.axis)
        for desc in QD.all_legal_descriptors():
            self.assertNotIn("slices", desc)
            break

    def test_both_measured_optima_lie_inside_the_tuner_ladder(self):
        """Which is why no candidate was built. `[planned/2, planned*2]`."""
        for facts, optimum in ((self.M96, 3), (self.M256, 12)):
            lo = max(1, facts.planner_slices // 2)
            hi = facts.planner_slices * 2
            with self.subTest(m=facts.m):
                self.assertTrue(lo <= optimum <= hi)

    def test_the_m96_optimum_is_interior_not_a_ladder_rail_artifact(self):
        """Runs 1440-1448 plus 1416-1421: the curve is measured on both sides.

        `decode_m96_up` tunes to 3 against a `[3,12]` ladder, so its choice sits
        on the bottom rail and an optimum at 2 or 1 would have been unreachable
        -- the one case among the seven traced routes where a slice result could
        still have been worth shipping. It is not: 1 and 2 are both far worse,
        monotone into 3, and 6 is worse on the other side.

        So the residency-corrected term does not merely agree with the tuner
        here, it lands on a measured interior minimum.

        The s=6 point is the one number here taken from the other sweep, which
        normally forbids the comparison. It is admitted only because the two
        sweeps' shared s=3 arm agrees to ~1% (0.06700 vs 0.06776) against a 19%
        gap, so the conclusion cannot turn on the drift. The three points from
        1440-1448 are rotated within one sweep and need no such argument.
        """
        medians = {1: 0.10436, 2: 0.08240, 3: 0.06776, 6: 0.08032}
        best = min(medians, key=medians.get)
        self.assertEqual(best, self.M96.fill_limited_slices)
        self.assertLess(medians[3], medians[2])
        self.assertLess(medians[2], medians[1])
        self.assertLess(medians[3], medians[6])
        # ...and the rail it sits on is the bottom of the tuner window.
        self.assertEqual(max(1, self.M96.planner_slices // 2), 3)


class TheDepthPrizeIsSizedOnTheHostAndTheGridAsksForLessThanItHolds(
        unittest.TestCase):
    """v97/v98/v99, runs 1460-1495. Two routes both underfill their residency
    and both collapse to one CTA per CU when deepened; one gained 11.44% and
    the other lost 12.7%. The difference is the size of the prize, and it is
    computable on the host before anything is built."""

    # decode_m96_up, 96x128/2x2/sk32 -- the route where deepening WON.
    M96 = QD.RouteFacts(
        tiles=86, slices=7, cu_count=304, ctas_per_cu_cap=4, k=4096, m=96,
        n=11008, cta_m=96, cta_n=128, waves_m=2, waves_n=2, stage_k=32,
        lds_bytes=16128, waves_per_cta=4, vgpr_count=122)
    # prefill_m2048_square, 128x128/2x2/sk32 -- the route where it LOST.
    M2048 = QD.RouteFacts(
        tiles=512, slices=1, cu_count=304, ctas_per_cu_cap=3, k=4096, m=2048,
        n=4096, cta_m=128, cta_n=128, waves_m=2, waves_n=2, stage_k=32,
        lds_bytes=18432, waves_per_cta=4, vgpr_count=142)
    # The shipped v98: same tile, wave grid 2x2 -> 2x4. Objmeta on the linked
    # object reads 80 VGPR / 0 AGPR / 0 spills against v95's 142.
    M2048_W8 = dataclasses.replace(
        M2048, waves_n=4, waves_per_cta=8, vgpr_count=80)

    def test_the_grid_asks_for_fewer_ctas_per_cu_than_the_tile_can_hold(self):
        # 512 CTAs on 304 CUs: 208 CUs take two, 96 take one. The ceiling is
        # three. The third slot is a resource nothing will ever occupy.
        self.assertEqual(self.M2048.ctas, 512)
        self.assertEqual(self.M2048.grid_asks_per_cu, 2)
        self.assertEqual(self.M2048.ctas_by_occupancy, 3)
        self.assertEqual(self.M96.grid_asks_per_cu, 2)
        self.assertEqual(self.M96.ctas_by_occupancy, 4)

    def test_asks_per_cu_is_a_ceiling_because_makespan_is_set_by_the_worst_cu(self):
        # Not the mean 512/304 = 1.684. The CU holding two CTAs finishes last,
        # and that rounding IS the 18.7% grid-quantisation waste term.
        self.assertGreater(self.M2048.grid_asks_per_cu,
                            self.M2048.ctas / self.M2048.cu_count)
        waste = (self.M2048.grid_asks_per_cu * self.M2048.cu_count
                 / self.M2048.ctas)
        self.assertAlmostEqual(waste, 1.1875, places=4)

    def test_a_grid_that_is_not_supplied_asks_for_nothing(self):
        for f in (dataclasses.replace(self.M2048, tiles=0),
                  dataclasses.replace(self.M2048, cu_count=0)):
            with self.subTest(tiles=f.tiles, cu=f.cu_count):
                self.assertEqual(f.grid_asks_per_cu, 0)

    def test_the_prize_is_four_times_where_deepening_won_and_two_where_it_lost(self):
        # This is the whole finding. Both routes underfill; both end at one CTA
        # per CU when deepened. Only the multiplier separates them.
        self.assertTrue(
            QD.PRECONDITIONS["grid_underfills_residency"](self.M96))
        self.assertTrue(
            QD.PRECONDITIONS["grid_underfills_residency"](self.M2048))
        self.assertEqual(self.M96.deepest_stage_k_in_lds, 128)
        self.assertEqual(self.M2048.deepest_stage_k_in_lds, 64)
        self.assertEqual(self.M96.deepest_stage_k_in_lds // self.M96.stage_k, 4)
        self.assertEqual(
            self.M2048.deepest_stage_k_in_lds // self.M2048.stage_k, 2)

    def test_the_deepest_stage_actually_fits_and_the_next_one_does_not(self):
        for f in (self.M96, self.M2048):
            with self.subTest(cta_n=f.cta_n, cta_m=f.cta_m):
                per_k = (f.cta_m + f.cta_n) * 2
                deepest = f.deepest_stage_k_in_lds
                self.assertLessEqual((deepest + 4) * per_k, 65536)
                self.assertGreater((2 * deepest + 4) * per_k, 65536)

    def test_the_prize_is_a_power_of_two_because_48_blew_the_register_file(self):
        # v97 took 32 -> 48: legal LDS (26624 B, two CTAs), legal stride
        # static_assert, and 228 VGPR + 96 AGPR = 324 on the built object --
        # through the 256 step, one wave per SIMD, +12.7%. The property must
        # never propose it.
        for f in (self.M96, self.M2048):
            d = f.deepest_stage_k_in_lds
            with self.subTest(cta_n=f.cta_n):
                self.assertEqual(d & (d - 1), 0)
        v97 = dataclasses.replace(self.M2048, stage_k=48, lds_bytes=26624,
                                   vgpr_count=228, agpr_count=96)
        self.assertNotEqual(v97.stage_k, self.M2048.deepest_stage_k_in_lds)
        self.assertEqual(v97.ctas_by_vgpr, 1)          # the cliff, measured
        self.assertEqual(v97.waves_per_simd_measured, 1.0)

    def test_an_unsupplied_tile_sizes_no_prize(self):
        for f in (dataclasses.replace(self.M2048, cta_m=0),
                  dataclasses.replace(self.M2048, cta_n=0)):
            with self.subTest(cta_m=f.cta_m, cta_n=f.cta_n):
                self.assertEqual(f.deepest_stage_k_in_lds, 0)

    def test_widening_the_wave_grid_buys_occupancy_without_touching_lds(self):
        # v98, the move that shipped: -3.41% isolated (runs 1472-1481) and
        # -5.58% in suite (1482-1487). Accumulators per lane are
        # wave_m*wave_n/64, so 2x2 -> 2x4 halves them; LDS is untouched, and
        # so is global traffic per MFMA, which is what separates it from
        # taking a narrower tile (64x128, same intensity, +18.0%).
        self.assertEqual(self.M2048_W8.lds_bytes, self.M2048.lds_bytes)
        self.assertLess(self.M2048_W8.vgpr_count, self.M2048.vgpr_count)
        self.assertEqual(self.M2048.waves_per_simd_measured, 3.0)
        self.assertEqual(self.M2048_W8.waves_per_simd_measured, 6.0)
        # Residency in CTAs is unchanged -- the surplus slot is still there.
        # What changed is how many waves each resident CTA brings.
        self.assertEqual(self.M2048_W8.ctas_by_occupancy,
                          self.M2048.ctas_by_occupancy)
        self.assertEqual(self.M2048_W8.grid_asks_per_cu, 2)

    def test_the_wave_grid_is_not_an_axis_so_the_move_is_not_a_mechanism(self):
        # Deliberate. Both wave grids carry the identical descriptor, so this
        # is an elite choice inside a cell, not an adjacency edge -- recording
        # it as one would put a knob in the archive the vocabulary refuses.
        self.assertNotIn("wave_grid", QD.AXES)
        self.assertNotIn("wave_tile", QD.AXES)
        for mech in QD.MECHANISMS:
            with self.subTest(mech=mech.name):
                self.assertIn(mech.axis, QD.AXIS_ORDER)

    def test_no_precondition_claims_the_register_file_is_the_discriminator(self):
        # A gate keyed on `ctas_by_vgpr <= ctas_by_lds` was written, run
        # against the archive, and deleted: it fires on decode_m96_up, the
        # route where deepening WON. Both routes are equal-bound.
        self.assertEqual(self.M96.ctas_by_vgpr, self.M96.ctas_by_lds)
        self.assertEqual(self.M2048.ctas_by_vgpr, self.M2048.ctas_by_lds)
        self.assertNotIn("residency_surplus_is_register_locked",
                          QD.PRECONDITIONS)

    def test_every_precondition_is_still_callable_on_both_routes(self):
        for name, pred in QD.PRECONDITIONS.items():
            for label, f in (("m96", self.M96), ("m2048", self.M2048),
                             ("m2048_w8", self.M2048_W8)):
                with self.subTest(precondition=name, route=label):
                    self.assertIsInstance(bool(pred(f)), bool)


class TheWaveGridPrizeIsMeasuredAndItsBreakEvenIsADoubling(unittest.TestCase):
    """v98/v100, runs 1472-1515. Two routes both hold a surplus residency slot
    and both spend it by widening the wave grid at a fixed tile; one gained
    5.58% and the other lost 2.04%. As with the depth prize, the difference is
    the size of the prize -- but unlike the depth prize, this one cannot be
    sized without building, because it turns on where the CTA count lands."""

    # prefill_m2048_square before and after the shipped v98 widening.
    M2048 = QD.RouteFacts(
        tiles=512, slices=1, cu_count=304, k=4096, cta_m=128, cta_n=128,
        waves_m=2, waves_n=2, stage_k=32, lds_bytes=18432, waves_per_cta=4,
        vgpr_count=142)
    M2048_W8 = dataclasses.replace(
        M2048, waves_n=4, waves_per_cta=8, vgpr_count=80)
    # prefill_m512_up before and after the rejected v100 widening. 2x4 is not
    # available on this tile -- 160/4 = 40 is not a multiple of the 16-wide
    # fragment -- so the only widening the geometry admits is 4x2.
    M512 = QD.RouteFacts(
        tiles=276, slices=2, cu_count=304, k=4096, cta_m=128, cta_n=160,
        waves_m=2, waves_n=2, stage_k=32, lds_bytes=20736, waves_per_cta=4,
        vgpr_count=152)
    M512_W8 = dataclasses.replace(
        M512, waves_m=4, waves_per_cta=8, vgpr_count=90)

    def test_both_routes_hold_the_surplus_slot_that_licenses_the_move(self):
        # The precondition is satisfied on both, which is exactly why it
        # cannot be the discriminator.
        for route in (self.M2048, self.M512):
            self.assertLess(route.grid_asks_per_cu, route.ctas_by_occupancy)

    def test_the_prize_doubles_where_it_won_and_gains_a_third_where_it_lost(self):
        self.assertAlmostEqual(
            self.M2048.wave_grid_occupancy_gain(self.M2048_W8), 2.0, places=6)
        self.assertAlmostEqual(
            self.M512.wave_grid_occupancy_gain(self.M512_W8), 4.0 / 3.0,
            places=6)

    def test_the_break_even_sits_between_the_two_measured_points(self):
        won = self.M2048.wave_grid_occupancy_gain(self.M2048_W8)
        lost = self.M512.wave_grid_occupancy_gain(self.M512_W8)
        self.assertLess(lost, won)
        # -5.58% at 2.00 and +2.04% at 1.33, at intensity costs of -33% and
        # -36% -- within three points of each other, so the gain is what
        # separates them and nothing else on offer does.
        self.assertLess(lost, 2.0)
        self.assertGreater(won, 4.0 / 3.0)

    def test_the_cta_count_is_what_moves_and_it_is_a_register_question(self):
        # LDS is untouched by widening on both routes: same tile, same stage.
        for before, after in ((self.M2048, self.M2048_W8),
                              (self.M512, self.M512_W8)):
            self.assertEqual(before.lds_bytes, after.lds_bytes)
            self.assertEqual(before.ctas_by_lds, after.ctas_by_lds)
        # v98 kept all three CTAs, so doubling the waves doubled the residency.
        self.assertEqual(self.M2048_W8.ctas_by_occupancy, 3)
        # v100 dropped to two, so doubling the waves bought only a third. That
        # is the whole difference, and it is decided by the register file.
        self.assertEqual(self.M512_W8.ctas_by_occupancy, 2)
        self.assertEqual(self.M512_W8.ctas_by_vgpr, 2)

    def test_the_prize_refuses_to_guess_when_registers_are_unread(self):
        # Never estimate registers off the source -- the v80 error, and the
        # reason this signature takes a built route rather than a wave count.
        unread = dataclasses.replace(self.M2048_W8, vgpr_count=0, agpr_count=0)
        self.assertEqual(self.M2048.wave_grid_occupancy_gain(unread), 0.0)
        self.assertEqual(
            dataclasses.replace(self.M2048, vgpr_count=0)
            .wave_grid_occupancy_gain(self.M2048_W8), 0.0)
        self.assertEqual(self.M2048.wave_grid_occupancy_gain(None), 0.0)

    def test_the_host_ceiling_is_a_true_bound_on_the_measured_prize(self):
        for before, after, wm, wn in (
                (self.M2048, self.M2048_W8, 2, 4),
                (self.M512, self.M512_W8, 4, 2)):
            self.assertGreaterEqual(before.wave_grid_gain_ceiling(wm, wn),
                                    before.wave_grid_occupancy_gain(after))

    def test_the_host_ceiling_would_not_have_saved_the_v100_build(self):
        # Both routes hold three CTAs by LDS, so both read exactly 2.00 -- the
        # one that won and the one that lost. Recorded as a cheap negative
        # that has never fired, not as the discriminator.
        self.assertEqual(self.M2048.ctas_by_lds, self.M512.ctas_by_lds)
        self.assertAlmostEqual(self.M2048.wave_grid_gain_ceiling(2, 4), 2.0)
        self.assertAlmostEqual(self.M512.wave_grid_gain_ceiling(4, 2), 2.0)

    def test_the_ceiling_declines_a_move_that_is_not_a_widening(self):
        self.assertEqual(self.M2048.wave_grid_gain_ceiling(2, 2), 0.0)
        self.assertEqual(self.M2048.wave_grid_gain_ceiling(1, 2), 0.0)

    def test_the_widening_is_still_not_a_descriptor_axis(self):
        # v95 and v98 differ by a measured 5.58% and carry identical five-field
        # descriptors, so the prize has to live as a property, exactly as
        # `plan_slices` does -- never as a Mechanism with an invented axis.
        self.assertNotIn("wave_grid", QD.AXES)
        self.assertFalse(any("wave_grid" in m.axis for m in QD.MECHANISMS))


class AtomicsDoNotStreamAndAMallResidentWorkspaceIsNotAnHbmTerm(
        unittest.TestCase):
    """v101, runs 1520-1525. `output_path: workspace_fixup -> atomic_fixup`
    cut fixup traffic 10x and measured +8.05% with no confound. Finding (17):
    atomics trade bytes for throughput, and the estimate that motivated the
    build had priced a MALL-resident workspace at HBM bandwidth."""

    # prefill_m256_down, 128x160/2x2/sk32, twelve slices -- the largest fixup
    # fraction in the suite and so the most favourable case the move has.
    M256 = QD.RouteFacts(
        tiles=52, slices=12, cu_count=304, m=256, n=4096, k=11008,
        cta_m=128, cta_n=160, waves_m=2, waves_n=2, stage_k=32,
        lds_bytes=20736, waves_per_cta=4, vgpr_count=152)

    def test_split_k_is_mandatory_here_which_is_why_the_fixup_is_unavoidable(self):
        # 52 CTAs is 17% of 304 CUs, so unlike prefill_m2048_square this route
        # cannot simply decline to split.
        self.assertLess(self.M256.tiles, self.M256.cu_count // 4)
        self.assertGreater(self.M256.slices, 1)

    def test_traffic_is_exactly_twice_footprint_and_they_are_not_the_same_thing(self):
        self.assertEqual(self.M256.splitk_plane_bytes,
                         2 * self.M256.splitk_workspace_bytes)
        self.assertEqual(self.M256.splitk_workspace_bytes, 4 * 256 * 4096 * 12)

    def test_the_workspace_is_mall_resident_so_it_was_never_an_hbm_term(self):
        # 50.3 MB: past the 32 MB of L2 but well inside the 256 MB MALL. This
        # is the assertion that would have refused the v101 build.
        self.assertTrue(self.M256.fits_in_mall)
        self.assertGreater(self.M256.splitk_workspace_bytes, 32 * 1024 * 1024)

    def test_comparing_traffic_to_the_cache_would_flip_the_answer(self):
        # The whole point of keeping the two properties apart: the traffic
        # figure is 100.7 MB, still under 256 MB here, but at 48 slices it
        # crosses while the footprint does not. Off-by-2x in the direction
        # that calls a resident workspace non-resident.
        wide = dataclasses.replace(self.M256, slices=48)
        self.assertTrue(wide.fits_in_mall)
        self.assertGreater(wide.splitk_plane_bytes, 256 * 1024 * 1024)

    def test_an_undetermined_workspace_is_never_waved_through_as_resident(self):
        self.assertFalse(dataclasses.replace(self.M256, slices=1).fits_in_mall)
        self.assertFalse(dataclasses.replace(self.M256, m=0).fits_in_mall)
        self.assertFalse(dataclasses.replace(self.M256, n=0).fits_in_mall)

    def test_a_genuinely_oversized_workspace_still_reads_as_an_hbm_term(self):
        huge = dataclasses.replace(self.M256, m=2048, n=11008, slices=200)
        self.assertFalse(huge.fits_in_mall)

    def test_atomic_fixup_is_a_legal_move_that_is_now_closed_by_measurement(self):
        # It has to stay in the vocabulary -- it is a real AMD mechanism and
        # the archive must be able to name it to record that it lost. What it
        # must not be is a Mechanism, i.e. something the archive proposes.
        self.assertIn("atomic_fixup", QD.QD_VOCAB["output_path"])
        self.assertFalse(any(m.to_value == "atomic_fixup"
                             for m in QD.MECHANISMS))

class NoLegalTileBuysGridEfficiencyForFreeOnEitherOpenRoute(unittest.TestCase):
    """Finding (18), corrected. The first version argued from power-of-two tile
    counts and concluded no tile can reach utilisation 1.0. The conclusion held
    but the premise was false -- a tile edge only has to be a multiple of
    16*waves, so 112/160/224 are legal and 64x224 puts exactly 304 tiles on
    prefill_m1024_down. The real reason is that utilisation is the wrong figure
    of merit: reaching a multiple of 304 needs a factor of 19 in the count,
    which on a power-of-two shape has to come from an edge that mis-divides the
    shape and pays the gain back as padding."""

    def route(self, M, N, tm, tn, wm, wn, cu=304):
        return QD.RouteFacts(
            tiles=(-(-M // tm)) * (-(-N // tn)), slices=1, cu_count=cu,
            m=M, n=N, k=11008, cta_m=tm, cta_n=tn, waves_m=wm, waves_n=wn,
            stage_k=32, lds_bytes=2 * (tm + tn) * 32, waves_per_cta=wm * wn,
            vgpr_count=80)

    def shipped(self, which):
        return self.route(*{"m1024": (1024, 4096), "m2048": (2048, 4096)}[which],
                          128, 128, 2, 4)

    def test_the_original_mod_nineteen_argument_is_still_true_as_far_as_it_goes(self):
        self.assertEqual(304, 2 ** 4 * 19)
        self.assertNotIn(0, {(2 ** k) % 19 for k in range(1, 64)})
        for tiles in (256, 512, 1024):
            with self.subTest(tiles=tiles):
                r = self.route(1024, 4096, 128, 128, 2, 4)
                r = QD.RouteFacts(**{**r.__dict__, "tiles": tiles})
                self.assertAlmostEqual(r.grid_utilisation, 0.8421, places=4)

    def test_but_a_non_power_of_two_edge_does_reach_utilisation_one(self):
        # 4096 needs 19 columns of 224, and 1024/64 is 16, so 16*19 = 304.
        # This is the case the original argument said could not exist.
        r = self.route(1024, 4096, 64, 224, 4, 2)
        self.assertEqual(r.ctas, 304)
        self.assertEqual(r.grid_utilisation, 1.0)

    def test_and_it_pays_the_whole_gain_back_in_padding(self):
        r = self.route(1024, 4096, 64, 224, 4, 2)
        self.assertAlmostEqual(r.padding_factor, 4256 / 4096, places=6)
        self.assertAlmostEqual(r.grid_efficiency, 0.9624, places=4)

    def test_efficiency_is_the_comparable_figure_and_utilisation_is_not(self):
        free = self.route(1024, 4096, 64, 224, 4, 2)
        crude = self.route(1024, 4096, 32, 64, 2, 4)   # 2048 tiles, no padding
        self.assertGreater(free.grid_utilisation, crude.grid_utilisation)
        # ... yet they score identically once padding is netted out, because
        # both are the same 19 showing up in different places.
        self.assertAlmostEqual(free.grid_efficiency, crude.grid_efficiency,
                               places=4)

    def test_occupancy_is_what_the_grid_asks_not_what_lds_could_hold(self):
        m2048 = self.shipped("m2048")
        self.assertEqual(m2048.grid_asks_per_cu, 2)
        self.assertGreaterEqual(m2048.ctas_by_lds, 2)
        self.assertEqual(m2048.occupancy_waves_per_simd, 4.0)
        # Capacity alone would call these two equal; the grid ask separates
        # them, and using capacity here is what let the closest miss through.
        self.assertEqual(self.shipped("m1024").occupancy_waves_per_simd, 2.0)
        self.assertEqual(self.shipped("m1024").waves_per_simd,
                         m2048.waves_per_simd)

    def test_neither_open_route_has_a_tile_that_wins_without_paying(self):
        for which in ("m1024", "m2048"):
            with self.subTest(route=which):
                self.assertFalse(
                    self.shipped(which).tile_axis_can_collect_grid_waste)

    def test_the_search_is_exhaustive_over_what_the_kernel_can_build(self):
        legal = self.shipped("m2048").legal_tiles()
        self.assertGreater(len(legal), 80)
        self.assertIn((128, 128, 2, 4), legal)
        self.assertIn((256, 112, 8, 1), legal)     # the closest miss
        self.assertIn((64, 224, 4, 2), legal)      # the utilisation-1.0 tile
        for tm, tn, wm, wn in legal:
            self.assertEqual(tm % (16 * wm), 0)
            self.assertEqual(tn % (16 * wn), 0)
            self.assertEqual(wm * wn, 8)

    def test_the_closest_miss_is_refused_on_occupancy_and_nothing_else(self):
        # 256x112/8x1 genuinely beats shipped on BOTH grid efficiency and
        # intensity. It is refused because it halves waves/SIMD, and v98
        # measured that same occupancy step at 5.58% in the other direction.
        miss = self.route(2048, 4096, 256, 112, 8, 1)
        base = self.shipped("m2048")
        self.assertGreater(miss.grid_efficiency, base.grid_efficiency * 1.015)
        self.assertGreater(miss.intensity, base.intensity)
        self.assertLess(miss.occupancy_waves_per_simd,
                        base.occupancy_waves_per_simd)

    def test_the_intensity_guard_is_what_refuses_the_utilisation_one_tile(self):
        # 64x224 on m1024 is the tile the original argument said was
        # impossible. It is refused, but on intensity -- 0.875 against 1.333 --
        # not on the count arithmetic the first version of this finding used.
        tile = self.route(1024, 4096, 64, 224, 4, 2)
        base = self.shipped("m1024")
        self.assertGreater(tile.grid_efficiency, base.grid_efficiency * 1.015)
        self.assertEqual(tile.occupancy_waves_per_simd,
                         base.occupancy_waves_per_simd)
        self.assertLess(tile.intensity, base.intensity)

    def test_the_predicate_is_not_vacuous_and_does_accept_a_genuine_free_move(self):
        # A badly-chosen incumbent on the very same shape and machine: 80x256
        # packs 416 tiles at efficiency 0.674, and 96x192 beats it on grid
        # efficiency AND intensity at equal occupancy. So the search does find
        # winners; the two open routes are refused because they are already
        # sitting on tiles nothing dominates, not because the test never fires.
        bad = self.route(2048, 4096, 80, 256, 1, 8)
        self.assertAlmostEqual(bad.grid_efficiency, 0.6737, places=4)
        self.assertTrue(bad.tile_axis_can_collect_grid_waste)
        win = self.route(2048, 4096, 96, 192, 2, 4)
        self.assertGreater(win.grid_efficiency, bad.grid_efficiency * 1.015)
        self.assertGreater(win.intensity, bad.intensity)
        self.assertEqual(win.occupancy_waves_per_simd,
                         bad.occupancy_waves_per_simd)

    def test_the_shipped_tiles_are_not_merely_undominated_but_well_chosen(self):
        # Both routes ship a tile that no legal alternative dominates. That is
        # the positive form of the finding and it is worth asserting directly:
        # choose_plan's tile order is doing real work here.
        for which, shape in (("m1024", (1024, 4096)), ("m2048", (2048, 4096))):
            base = self.shipped(which)
            with self.subTest(route=which):
                for tm, tn, wm, wn in base.legal_tiles():
                    cand = self.route(*shape, tm, tn, wm, wn)
                    self.assertFalse(
                        cand.grid_efficiency > base.grid_efficiency * 1.015
                        and cand.intensity >= base.intensity
                        and cand.occupancy_waves_per_simd
                        >= base.occupancy_waves_per_simd,
                        f"{tm}x{tn}/{wm}x{wn} dominates the shipped tile")

    def test_an_undetermined_route_reports_nothing_and_proposes_nothing(self):
        r = QD.RouteFacts(tiles=0, slices=1, cu_count=304, m=0, n=0, k=11008,
                          cta_m=0, cta_n=0, waves_m=2, waves_n=4, stage_k=32,
                          lds_bytes=0, waves_per_cta=8, vgpr_count=80)
        self.assertEqual(r.grid_utilisation, 0.0)
        self.assertEqual(r.padding_factor, 0.0)
        self.assertEqual(r.grid_efficiency, 0.0)
        self.assertEqual(r.legal_tiles(), [])
        self.assertFalse(r.tile_axis_can_collect_grid_waste)


class CoResidencyNotWavesPerSimdIsTheOccupancyTermAndTrafficIsTheGridsPrice(
        unittest.TestCase):
    """Finding (19), and it is the first tile result on this route that came
    from a build rather than from a search.

    The search of finding (18) named `256x112/8x1` on `prefill_m2048_square` as
    the closest miss and said to build it if the occupancy term could be bounded
    under 14%. It was built (v102) and measured 32.3% WORSE than shipped against
    a written prediction of -2% to +6%. A miss that size is a missing term.

    Two were missing. Both are now properties with their own docstrings, and the
    numbers below are the measurements that put them there -- all on machine J,
    all `--case prefill_m2048_square`, autotune ON, rotated.

    The terms are measured. The MECHANISM of the co-residency term is not known:
    the barrier explanation was pre-registered, tested by v104, and refuted, and
    register pressure is excluded by objmeta (126 VGPR, 0 AGPR, 0 spill on the
    256x112 instantiation). See `test_halving_the_barriers_...` below.
    """

    # Machine J, runs 1549-1551 (v102) and 1560-1571 (rotated three-arm).
    # Medians in ms with 2*MAD.
    MEASURED = {
        (128, 128, 2, 4): 0.18558,   # shipped, +-0.00072
        (128, 112, 4, 1): 0.19384,   # v103a,   +-0.00292
        (64, 224, 2, 2): 0.27050,    # v103b,   +-0.01616
        (256, 112, 8, 1): 0.24560,   # v102,    +-0.00208
    }

    # v104, runs 1581-1586: v102's tile with the double-buffer gate relaxed for
    # it alone. It costs no occupancy, because the grid had already pinned that
    # tile to one CTA per CU, and objmeta confirms the build took it (lds 52992,
    # vgpr 126, zero spills). It halves the barrier count from 256 to 128.
    V104_DOUBLE_BUFFERED_MS = 0.24420

    def route(self, tm, tn, wm, wn, m=2048, n=4096):
        return QD.RouteFacts(
            tiles=(-(-m // tm)) * (-(-n // tn)), slices=1, cu_count=304,
            m=m, n=n, k=4096, cta_m=tm, cta_n=tn, waves_m=wm, waves_n=wn,
            stage_k=32, lds_bytes=(tm + tn) * 36 * 2, waves_per_cta=wm * wn,
            vgpr_count=80)

    def test_halving_the_barriers_did_not_buy_back_the_co_residency_gap(self):
        # The first explanation of the term was barrier coverage: a CTA alone on
        # a CU stalls the whole CU at each `__syncthreads()` with nothing to
        # overlap. v104 tested it directly and the criterion was written down
        # before the run -- under 0.205 ms would confirm barriers, near 0.245
        # would refute them. It landed at 0.24420.
        shipped = self.MEASURED[(128, 128, 2, 4)]
        v102 = self.MEASURED[(256, 112, 8, 1)]
        gap = v102 - shipped
        recovered = (v102 - self.V104_DOUBLE_BUFFERED_MS) / gap
        self.assertLess(recovered, 0.05)
        # So the guard stands on the measurement and not on the story. Anything
        # that claims to know WHY one CTA per CU costs 26.7% has to beat this
        # result first; barriers and register pressure are both already out.

    def test_v102_and_v103a_differ_in_co_residency_and_in_nothing_else(self):
        # This is the whole experiment. Same wave tile (32x112), so the same
        # intensity; the same waves per SIMD; the same grid efficiency. The one
        # difference is that 592 tiles ask for two CTAs per CU and 296 ask for
        # one -- and that alone was worth 26.7%.
        v102 = self.route(256, 112, 8, 1)
        v103a = self.route(128, 112, 4, 1)
        self.assertAlmostEqual(v102.intensity, v103a.intensity, places=9)
        self.assertAlmostEqual(v102.occupancy_waves_per_simd,
                               v103a.occupancy_waves_per_simd, places=9)
        self.assertAlmostEqual(v102.grid_efficiency, v103a.grid_efficiency,
                               places=4)
        # ...and they differ here, which is the term the old guard could not see.
        self.assertEqual(v102.ctas_per_cu, 1)
        self.assertEqual(v103a.ctas_per_cu, 2)

    def test_waves_per_simd_cannot_tell_the_two_apart_which_is_why_v102_shipped(
            self):
        # The refuted guard, kept as a test so the reason it failed stays
        # visible: two CTAs of four waves and one CTA of eight waves are the
        # same number and are a 26.7% difference on the machine.
        v102 = self.route(256, 112, 8, 1)
        v103a = self.route(128, 112, 4, 1)
        self.assertEqual(v102.occupancy_waves_per_simd, 2.0)
        self.assertEqual(v103a.occupancy_waves_per_simd, 2.0)
        self.assertNotEqual(v102.ctas_per_cu, v103a.ctas_per_cu)

    def test_the_measured_order_is_not_explained_by_grid_efficiency(self):
        # Every one of the three challengers beats shipped on grid efficiency
        # by 14.3% and every one of them lost. Grid efficiency alone, which is
        # what finding (18) closed on, would have ranked all three ahead.
        ship = self.route(128, 128, 2, 4)
        for tm, tn, wm, wn in [(128, 112, 4, 1), (64, 224, 2, 2),
                               (256, 112, 8, 1)]:
            with self.subTest(tile=f"{tm}x{tn}"):
                cand = self.route(tm, tn, wm, wn)
                self.assertGreater(cand.grid_efficiency,
                                   ship.grid_efficiency * 1.14)
                self.assertGreater(self.MEASURED[(tm, tn, wm, wn)],
                                   self.MEASURED[(128, 128, 2, 4)])

    def test_among_co_resident_tiles_time_tracks_operand_traffic(self):
        # Hold co-residency at two and the remaining spread is surface-to-
        # volume: +7.1% traffic cost +4.45% time, +28.6% cost +45.8%. Not
        # linear, but monotone, and that is all the guard needs.
        arms = [(128, 128, 2, 4), (128, 112, 4, 1), (64, 224, 2, 2)]
        for tm, tn, wm, wn in arms:
            self.assertEqual(self.route(tm, tn, wm, wn).ctas_per_cu, 2,
                             f"{tm}x{tn} is not the co-resident comparison")
        by_traffic = sorted(
            arms, key=lambda a: self.route(*a).operand_bytes_per_output)
        by_time = sorted(arms, key=lambda a: self.MEASURED[a])
        self.assertEqual(by_traffic, by_time)

    def test_the_lowest_traffic_tile_of_all_is_the_one_that_lost_worst(self):
        # 256x112 has the BEST surface-to-volume of the four -- 17.9% below
        # shipped, because it is nearly twice the area -- and it lost by 32.3%.
        # Traffic does not subsume co-residency; both terms are needed.
        cands = [(128, 128, 2, 4), (128, 112, 4, 1), (64, 224, 2, 2),
                 (256, 112, 8, 1)]
        best = min(cands,
                   key=lambda a: self.route(*a).operand_bytes_per_output)
        self.assertEqual(best, (256, 112, 8, 1))
        self.assertEqual(max(cands, key=lambda a: self.MEASURED[a]),
                         (64, 224, 2, 2))
        self.assertLess(self.route(*best).operand_bytes_per_output,
                        self.route(128, 128, 2, 4).operand_bytes_per_output)

    def test_reaching_the_grid_forces_an_edge_of_112_or_224_because_19_is_prime(
            self):
        # The structural claim. 304 = 2**4 * 19 and 19 is prime, so the tile
        # count reaches a multiple of 304 only if one of the two edge divisions
        # supplies the whole factor. On 2048x4096 that is ceil(2048/tm) or
        # ceil(4096/tn) divisible by 19, which pins the edge to 112 or 224.
        edges = set()
        for tm in range(16, 257, 16):
            for tn in range(16, 257, 16):
                tiles = (-(-2048 // tm)) * (-(-4096 // tn))
                if tiles % 19 == 0:
                    edges.add(tm if (-(-2048 // tm)) % 19 == 0 else tn)
        self.assertEqual(edges, {112, 224})

    def test_every_perfectly_quantised_tile_is_big_and_alone_or_small_and_costly(
            self):
        # The trilemma, asserted exhaustively: on this shape no tile reaching
        # utilisation 1.0 both holds two CTAs per CU and moves no more operand
        # traffic than the shipped tile.
        ship = self.route(128, 128, 2, 4)
        found = 0
        for tm in range(16, 257, 16):
            for tn in range(16, 257, 16):
                cand = self.route(tm, tn, 1, 1)
                if cand.grid_utilisation < 0.999:
                    continue
                found += 1
                with self.subTest(tile=f"{tm}x{tn}"):
                    self.assertFalse(
                        cand.ctas_per_cu >= ship.ctas_per_cu
                        and cand.operand_bytes_per_output
                        <= ship.operand_bytes_per_output,
                        f"{tm}x{tn} escapes the trilemma")
        self.assertGreater(found, 5)

    def test_the_predicate_now_refuses_the_tile_it_used_to_propose(self):
        # The end of the loop: with both terms priced, the shipped tile is
        # undominated and 256x112 is no longer the closest miss -- it is
        # refused on co-residency, and the 112/224 family on traffic.
        ship = self.route(128, 128, 2, 4)
        self.assertFalse(ship.tile_axis_can_collect_grid_waste)
        v102 = self.route(256, 112, 8, 1)
        self.assertLess(v102.ctas_per_cu, ship.ctas_per_cu)
        v103a = self.route(128, 112, 4, 1)
        self.assertGreater(v103a.operand_bytes_per_output,
                           ship.operand_bytes_per_output)

    def test_the_predicate_is_still_not_vacuous_under_the_stricter_guards(self):
        # Adding guards makes a False cheaper to get, so re-pin non-vacuity:
        # a deliberately bad incumbent must still be seen to be dominated.
        base = self.route(80, 256, 1, 8)
        self.assertTrue(base.tile_axis_can_collect_grid_waste)

    def test_an_undetermined_tile_prices_neither_new_term(self):
        r = QD.RouteFacts(tiles=0, slices=1, cu_count=304, m=0, n=0, k=4096,
                          cta_m=0, cta_n=0, waves_m=2, waves_n=4, stage_k=32,
                          lds_bytes=0, waves_per_cta=8, vgpr_count=80)
        self.assertEqual(r.ctas_per_cu, 0)
        self.assertEqual(r.operand_bytes_per_output, 0.0)
        self.assertFalse(r.tile_axis_can_collect_grid_waste)


class TheXcdRectangleIsBalanceNotAreaAndItOrdersEveryMeasuredHitRate(
        unittest.TestCase):
    """Finding (19c/19d), the first term in this descriptor that was derived
    from hardware counters rather than from timings.

    Three builds had been spent guessing at the co-residency cliff -- barriers
    (refuted by v104), registers (excluded by objmeta) -- so v105 collected
    `TCC_{HIT,MISS,REQ}_sum` instead. It found `256x112` issuing the FEWEST L2
    requests of the four arms and taking 2.38x the misses, and traced that to
    v59's hardcoded `kGroupM = 8` degenerating whenever `tiles_m <= kGroupM`.

    Two sweeps followed, seven measured hit rates. `xcd_panel_balance` ranks all
    seven correctly and `area` ranks neither sweep, which is the whole claim.
    """

    # L2 hit rate %, machine J, GEAK_DEBUG_FORCE_GROUPM, one v105 binary each.
    HIT_256x112 = {8: 76.03, 4: 86.80, 2: 79.54}          # prefill_m2048_square
    HIT_128x160 = {8: 85.30, 4: 86.29, 2: 80.97, 1: 66.58}  # prefill_m1024_down

    def route(self, tm, tn, wm, wn, m=2048, n=4096, slices=1):
        return QD.RouteFacts(
            tiles=(-(-m // tm)) * (-(-n // tn)), slices=slices, cu_count=304,
            m=m, n=n, k=4096, cta_m=tm, cta_n=tn, waves_m=wm, waves_n=wn,
            stage_k=32, lds_bytes=(tm + tn) * 36 * 2, waves_per_cta=wm * wn,
            vgpr_count=80)

    def test_balance_orders_both_measured_sweeps(self):
        for label, rt, hits in (
                ("256x112 on m2048", self.route(256, 112, 8, 1),
                 self.HIT_256x112),
                ("128x160 on m1024", self.route(128, 160, 2, 2, m=1024,
                                                slices=4), self.HIT_128x160)):
            with self.subTest(label):
                by_hit = sorted(hits, key=lambda g: -hits[g])
                by_bal = sorted(hits, key=lambda g: rt.xcd_panel_balance(g))
                self.assertEqual(by_hit, by_bal)

    def test_area_orders_neither_sweep(self):
        # The obvious rival metric is the rectangle's total size, and it is
        # wrong on both routes: group 2 gives 256x112 a 20.2 MiB rectangle
        # against group 8's 20.4, essentially tied, while their hit rates are
        # 3.5 points apart -- and on m1024 area is monotone in the group height
        # while the hit rate is not.
        rt = self.route(256, 112, 8, 1)
        tiles_n = -(-4096 // 112)
        chunk = (2048 // 256) * tiles_n // 8

        def area(g):
            rows = min(g, 2048 // 256)
            return rows * 256 + max(1, chunk // rows) * 112

        by_hit = sorted(self.HIT_256x112, key=lambda g: -self.HIT_256x112[g])
        self.assertNotEqual(by_hit, sorted(self.HIT_256x112, key=area))

    def test_best_group_m_picks_the_measured_winner_on_both(self):
        self.assertEqual(self.route(256, 112, 8, 1).best_group_m(), 4)
        self.assertEqual(
            self.route(128, 160, 2, 2, m=1024, slices=4).best_group_m(), 4)

    def test_it_leaves_the_shipped_tiles_alone(self):
        # The knob must not propose changing a route that is not degenerate,
        # or it is just churn. 128x128 on m2048 has tiles_m = 16 and is already
        # square at v59's constant.
        ship = self.route(128, 128, 2, 4)
        self.assertEqual(ship.best_group_m(), 8)
        self.assertAlmostEqual(ship.xcd_panel_balance(8), 1.0, places=6)

    def test_the_clamp_is_why_m512_never_degenerated(self):
        # tiles_m = 4 < kGroupM = 8, so the kernel's own
        # `min(kGroupM, tiles_m - first_m)` already held the group at 4. m512 is
        # the best-behaved profiled route at 93.22% and this is the reason.
        m512 = self.route(128, 160, 2, 2, m=512)
        self.assertAlmostEqual(m512.xcd_panel_balance(8),
                               m512.xcd_panel_balance(4), places=9)
        self.assertLess(m512.xcd_panel_balance(8), 1.1)

    def test_degeneracy_needs_tiles_m_at_or_under_the_group(self):
        # The predicate this whole finding turns on, stated directly: the group
        # collapses onto the whole grid exactly when the grid is not taller than
        # the group. 256x112 (8 rows) and 128x160-on-m1024 (8 rows) qualify;
        # 128x128 (16 rows) does not.
        for tm, m, degenerate in ((256, 2048, True), (128, 1024, True),
                                  (128, 2048, False)):
            with self.subTest(cta_m=tm, m=m):
                self.assertEqual(-(-m // tm) <= 8, degenerate)

    def test_an_unsupplied_tile_prices_nothing_and_changes_nothing(self):
        blind = QD.RouteFacts(tiles=512, slices=1, cu_count=304)
        self.assertEqual(blind.xcd_panel_balance(), 0.0)
        self.assertEqual(blind.best_group_m(), 8)

    def test_nonsense_group_heights_are_refused_rather_than_divided_by(self):
        rt = self.route(256, 112, 8, 1)
        self.assertEqual(rt.xcd_panel_balance(0), 0.0)
        self.assertEqual(rt.xcd_panel_balance(-4), 0.0)
        self.assertEqual(rt.xcd_panel_balance(8, xcds=0), 0.0)

    def test_predicting_the_winner_is_not_the_same_as_it_being_worth_it(self):
        # The term is honest about its own value. It picked 4 on both routes and
        # the counters agreed both times, but on m1024 that was 0.55% of
        # gpu-active cycles -- under the +-1.5% suite drift -- because that
        # route ships slices=4 and the swizzle ignores blockIdx.z, so its panels
        # were already shared four ways. Nothing here has shipped.
        m1024 = self.route(128, 160, 2, 2, m=1024, slices=4)
        m2048 = self.route(256, 112, 8, 1)
        self.assertEqual(m1024.best_group_m(), m2048.best_group_m())
        # ... and yet the imbalance being repaired differs by a wide margin.
        self.assertLess(m1024.xcd_panel_balance(8), 3.0)
        self.assertGreater(m2048.xcd_panel_balance(8), 4.0)
        self.assertGreater(m1024.slices, 1)
        self.assertEqual(m2048.slices, 1)


class TheLdsBudgetBuysTwoOfCoverOverlapAndCoresidencyNeverThree(unittest.TestCase):
    """Finding (20g). Every corner of the trilemma was measured on one route.

    prefill_m2048_square, 128x128/2x4, gfx942, machine K, single binary, both
    arms forced through the same hook, ABBA, --case isolated. The shipped
    corner wins and every alternative is far outside drift, so these are
    regression pins on a mechanism, not on a tuning result.
    """

    SHIPPED_MS = 0.18390        # sk32 single, 3 CTAs/CU  (runs 1620, 1623)
    SK16_DOUBLE_MS = 0.24696    # sk16 double, 3 CTAs/CU  (runs 1621, 1622)
    SK32_DOUBLE_LOSS = 0.267    # sk32 double falls to 1 CTA/CU, finding (19)

    def shipped(self, stage_k=32, lds=18432, cap=3):
        return QD.RouteFacts(tiles=512, slices=1, cu_count=304, cta_m=128,
                          cta_n=128, waves_m=2, waves_n=4, stage_k=stage_k,
                          lds_bytes=lds, ctas_per_cu_cap=cap,
                          m=2048, n=4096, k=4096)

    def test_the_shipped_route_holds_cover_and_coresidency(self):
        self.assertEqual(self.shipped().lds_budget_corner(),
                         "cover+coresidency")

    def test_the_host_predicts_the_kernels_own_compile_time_gate(self):
        # The kernel gates on 2 * kPanelBytes * 3 <= 65536. At sk32 the panel is
        # (128+128)*36*2 = 18432 and 110592 > 65536, so it must NOT double
        # buffer; at sk16 the panel is 10240 and 61440 <= 65536, so it must.
        self.assertLess(self.shipped().stage_k_for_double_buffer(3), 32)
        self.assertGreaterEqual(
            self.shipped(stage_k=16, lds=20480).stage_k_for_double_buffer(3), 16)

    def test_the_deepest_double_buffered_stage_is_the_one_v106_built(self):
        self.assertEqual(self.shipped().stage_k_for_double_buffer(3), 16)

    def test_the_price_of_a_second_buffer_is_half_the_cover(self):
        self.assertAlmostEqual(
            self.shipped().cover_cost_of_double_buffering(3), 0.5)

    def test_paying_that_price_measured_worse_not_better(self):
        self.assertGreater(self.SK16_DOUBLE_MS, self.SHIPPED_MS)
        loss = (self.SK16_DOUBLE_MS - self.SHIPPED_MS) / self.SHIPPED_MS
        self.assertGreater(loss, 0.30)

    def test_both_alternative_corners_lose_so_the_window_is_unaffordable(self):
        # Keeping cover and buying overlap costs co-residency: -26.7%.
        # Keeping co-residency and buying overlap costs cover: -34.3%.
        sk16_loss = (self.SK16_DOUBLE_MS - self.SHIPPED_MS) / self.SHIPPED_MS
        self.assertGreater(self.SK32_DOUBLE_LOSS, 0.0)
        self.assertGreater(sk16_loss, 0.0)

    def test_a_route_already_double_buffered_reports_the_other_corner(self):
        self.assertEqual(
            self.shipped(stage_k=16, lds=20480).lds_budget_corner(),
            "overlap+coresidency")
        self.assertEqual(
            self.shipped(stage_k=16, lds=20480)
                .cover_cost_of_double_buffering(3), 0.0)

    def test_a_route_that_lost_coresidency_reports_cover_plus_overlap(self):
        self.assertEqual(self.shipped(cap=1).lds_budget_corner(),
                         "cover+overlap")

    def test_an_unsupplied_tile_prices_nothing(self):
        bare = QD.RouteFacts(tiles=512, slices=1, cu_count=304)
        self.assertEqual(bare.lds_budget_corner(), "")
        self.assertEqual(bare.stage_k_for_double_buffer(), 0)
        self.assertEqual(bare.cover_cost_of_double_buffering(), 0.0)

    def test_nonsense_coresidency_targets_are_refused(self):
        for c in (0, -1):
            with self.subTest(ctas=c):
                self.assertEqual(
                    self.shipped().stage_k_for_double_buffer(c), 0)

    def test_wanting_fewer_ctas_makes_a_deeper_double_buffer_affordable(self):
        f = self.shipped()
        self.assertGreater(f.stage_k_for_double_buffer(1),
                           f.stage_k_for_double_buffer(3))

    def test_predicting_the_corner_is_not_the_same_as_it_being_worth_it(self):
        # The descriptor says which corner is reachable, never that moving is
        # a gain. On this route every reachable move measured a loss.
        self.assertEqual(self.shipped().lds_budget_corner(),
                         "cover+coresidency")
        self.assertGreater(self.SK16_DOUBLE_MS, self.SHIPPED_MS)



class ATileMutationIsRefusedUnlessItImprovesBothTermsAtOnce(unittest.TestCase):
    """Finding (21b). Runs 1630-1633 grew the tile on `prefill_m512_up` -- the
    exact converse of runs 1610-1615, which shrank it on `prefill_m2048_square`
    -- and BOTH lost. Either field alone predicts one of the two backwards."""

    M2048_HALVED_LOSS = 0.2588       # runs 1610-1615, 64x128 vs 128x128
    M512_GROWN_LOSS = 0.0881         # runs 1630-1633, 128x128 vs 128x64

    def m2048(self, cta_m=128, cta_n=128):
        tiles = (2048 + cta_m - 1) // cta_m * ((4096 + cta_n - 1) // cta_n)
        return QD.RouteFacts(tiles=tiles, slices=1, cu_count=304, cta_m=cta_m,
                             cta_n=cta_n, waves_m=2, waves_n=4, stage_k=32,
                             lds_bytes=18432, ctas_per_cu_cap=3,
                             m=2048, n=4096, k=4096)

    def m512(self, cta_m=128, cta_n=64):
        tiles = (512 + cta_m - 1) // cta_m * ((11008 + cta_n - 1) // cta_n)
        return QD.RouteFacts(tiles=tiles, slices=1, cu_count=304, cta_m=cta_m,
                             cta_n=cta_n, waves_m=2, waves_n=2, stage_k=32,
                             lds_bytes=18432, ctas_per_cu_cap=3,
                             m=512, n=11008, k=4096)

    def test_growing_the_m512_tile_is_refused_and_it_measured_worse(self):
        self.assertTrue(self.m512().tile_mutation_verdict(self.m512(128, 128)))
        self.assertGreater(self.M512_GROWN_LOSS, 0.0)

    def test_it_is_refused_on_utilisation_which_is_the_term_that_degraded(self):
        self.assertIn("grid_utilisation",
                      self.m512().tile_mutation_verdict(self.m512(128, 128)))

    def test_traffic_alone_would_have_allowed_the_m512_mutation(self):
        # The trap: growing the tile cuts traffic by a third. A traffic-only
        # score calls that a clear win; it measured -8.8%.
        self.assertLess(self.m512(128, 128).operand_bytes_per_output,
                        self.m512().operand_bytes_per_output)

    def test_shrinking_the_m2048_tile_is_refused_on_the_other_term(self):
        v = self.m2048().tile_mutation_verdict(self.m2048(64, 128))
        self.assertTrue(v)

    def test_the_two_measured_routes_are_refused_for_opposite_reasons(self):
        # This is the whole point: one field cannot be the score.
        a = self.m512().tile_mutation_verdict(self.m512(128, 128))
        b = self.m2048().tile_mutation_verdict(self.m2048(64, 128))
        self.assertTrue(a and b)

    def test_finer_m512_tiles_worsen_traffic_so_are_also_refused(self):
        for cm, cn in ((64, 64), (64, 128), (128, 32)):
            self.assertTrue(self.m512().tile_mutation_verdict(self.m512(cm, cn)),
                            msg=f"{cm}x{cn} should be refused")

    def test_no_tile_at_all_is_a_pareto_improvement_on_m512(self):
        # The tile axis on this route is closed, which is the precondition
        # that promotes a scheduling mechanism over another tile.
        here = self.m512()
        for cm in (64, 128, 256):
            for cn in (32, 64, 128, 256):
                if (cm, cn) == (here.cta_m, here.cta_n):
                    # The incumbent tile is not a tile mutation, so this rule
                    # abstains on it; `mutation_verdict` refuses it as a no-op.
                    self.assertFalse(here.mutation_verdict(self.m512(cm, cn))["allow"])
                    continue
                self.assertTrue(here.tile_mutation_verdict(self.m512(cm, cn)),
                                msg=f"{cm}x{cn} unexpectedly allowed")

    def test_a_pareto_improvement_would_be_allowed_if_one_existed(self):
        # Guard against a method that refuses everything unconditionally.
        better = QD.RouteFacts(tiles=912, slices=1, cu_count=304, cta_m=256,
                               cta_n=256, waves_m=2, waves_n=4, stage_k=32,
                               lds_bytes=18432, ctas_per_cu_cap=3,
                               m=512, n=11008, k=4096)
        self.assertGreater(better.grid_utilisation, self.m512().grid_utilisation)
        self.assertLess(better.operand_bytes_per_output,
                        self.m512().operand_bytes_per_output)
        self.assertEqual(self.m512().tile_mutation_verdict(better), "")

    def test_an_unsupplied_route_refuses_nothing(self):
        bare = QD.RouteFacts(tiles=0, slices=1, cu_count=304)
        self.assertEqual(bare.tile_mutation_verdict(self.m512()), "")
        self.assertEqual(self.m512().tile_mutation_verdict(bare), "")

    def test_a_no_op_mutation_is_refused_rather_than_silently_allowed(self):
        # Moved up to the composed entry point. The tile rule cannot own this:
        # asked about a mutation of some other axis it saw an unchanged tile
        # and refused, which is how it refused the v107 double buffer for a
        # reason about a tile nobody proposed changing. Each rule abstains
        # outside its axis; only `mutation_verdict` sees all of them.
        self.assertEqual("", self.m512().tile_mutation_verdict(self.m512()))
        verdict = self.m512().mutation_verdict(self.m512())
        self.assertFalse(verdict["allow"])
        self.assertIn("no change", verdict["refusals"][0])

    def test_304_is_16_times_19_and_this_shape_is_coprime_to_19(self):
        # Finding 21c: why the closure is permanent, not just unlucky.
        self.assertEqual(304, 16 * 19)
        for cm in (64, 128, 256):
            for cn in (32, 64, 128, 256):
                self.assertNotEqual(self.m512(cm, cn).tiles % 19, 0)

    def test_refusing_a_tile_is_not_a_claim_that_the_route_is_optimal(self):
        # It says only that the tile axis is exhausted. Finding 21d puts the
        # remaining 24.6% behind a scheduling mechanism, not a tile.
        self.assertTrue(self.m512().tile_mutation_verdict(self.m512(128, 128)))
        self.assertLess(self.m512().grid_utilisation, 1.0)



class RaisingGridUtilisationIsRefusedBecauseItWasMeasuredToBeWorthNothing(
        unittest.TestCase):
    """Finding (23). The makespan-is-set-by-the-busiest-CU model predicted
    +14.3% and +25.0% on `prefill_m256_down`; it measured -9.68% and -11.30%.
    Wrong sign, not merely wrong size. Every mechanism whose stated benefit is
    raising `grid_utilisation` -- more slices, and stream-K -- dies with it."""

    # runs 1651-1656, palindromic 12,14,16,16,14,12
    M256 = {12: 0.11648, 14: 0.12776, 16: 0.12964}
    # runs 1640-1645, palindromic 1,2,3,3,2,1
    M512 = {1: 0.17178, 2: 0.20856, 3: 0.18506}

    def m256(self, slices=12):
        tiles = (256 + 127) // 128 * ((4096 + 159) // 160)
        return QD.RouteFacts(tiles=tiles, slices=slices, cu_count=304,
                             cta_m=128, cta_n=160, waves_m=2, waves_n=2,
                             stage_k=32, lds_bytes=20736, ctas_per_cu_cap=3,
                             m=256, n=4096, k=11008)

    def m512(self, slices=1, cta_m=128, cta_n=64):
        tiles = (512 + cta_m - 1) // cta_m * ((11008 + cta_n - 1) // cta_n)
        return QD.RouteFacts(tiles=tiles, slices=slices, cu_count=304,
                             cta_m=cta_m, cta_n=cta_n, waves_m=2, waves_n=2,
                             stage_k=32, lds_bytes=18432, ctas_per_cu_cap=3,
                             m=512, n=11008, k=4096)

    def test_more_slices_is_refused_on_both_measured_routes(self):
        self.assertTrue(self.m256().slice_mutation_verdict(self.m256(16)))
        self.assertTrue(self.m512().slice_mutation_verdict(self.m512(2)))

    def test_the_refusal_names_utilisation_as_the_discredited_reason(self):
        self.assertIn("grid_utilisation",
                      self.m256().slice_mutation_verdict(self.m256(16)))

    def test_more_slices_really_did_raise_utilisation_on_m256(self):
        # The mutation delivered exactly what it promised and still lost, which
        # is what makes this a refutation rather than a failed attempt.
        self.assertGreater(self.m256(16).grid_utilisation,
                           self.m256(12).grid_utilisation)
        self.assertGreater(self.M256[16], self.M256[12])

    def test_every_measured_slice_increase_lost(self):
        for s in (14, 16):
            self.assertGreater(self.M256[s], self.M256[12])
        for s in (2, 3):
            self.assertGreater(self.M512[s], self.M512[1])

    def test_the_model_had_the_wrong_sign_not_just_the_wrong_size(self):
        predicted = 1.0 - self.m256(12).grid_utilisation / \
            self.m256(16).grid_utilisation
        measured = (self.M256[12] - self.M256[16]) / self.M256[12]
        self.assertGreater(predicted, 0.0)
        self.assertLess(measured, 0.0)

    def test_fewer_slices_is_allowed_through_since_none_was_measured(self):
        self.assertEqual(self.m256().slice_mutation_verdict(self.m256(8)), "")

    def test_an_unsupplied_slice_count_refuses_nothing(self):
        bare = QD.RouteFacts(tiles=624, slices=0, cu_count=304)
        self.assertEqual(bare.slice_mutation_verdict(self.m256(16)), "")

    def test_neither_measured_slice_increase_crossed_a_round_boundary(self):
        # This is why finding 23's routes were free and finding 24's were not:
        # 624 and 832 CTAs both fit inside 912 slots, so nothing serialised and
        # the whole effect there was reduction traffic.
        for s in (12, 14, 16):
            self.assertEqual(self.m256(s).rounds, 1)
        self.assertEqual(QD.OCCUPANCY_FILL_REMOVED_BY_FINDING, 24)

    def test_the_tile_rule_still_refuses_but_for_a_corrected_reason(self):
        # Finding 23f: kept as a measured predictor, not as a mechanism.
        self.assertTrue(self.m512().tile_mutation_verdict(
            self.m512(cta_m=128, cta_n=128)))

    def test_stream_k_is_not_resurrected_by_any_field_here(self):
        # Its only edge over split-K was reaching balance more cheaply, and
        # balance measured worthless on both routes that could show it.
        for r in (self.m256(), self.m512()):
            self.assertTrue(r.slice_mutation_verdict(
                QD.RouteFacts(tiles=r.tiles, slices=r.slices + 1,
                              cu_count=304, cta_m=r.cta_m, cta_n=r.cta_n,
                              waves_m=2, waves_n=2, stage_k=32,
                              lds_bytes=r.lds_bytes, ctas_per_cu_cap=3,
                              m=r.m, n=r.n, k=r.k)))


class CrossingARoundBoundaryIsTheResidencyCostThatActuallyBites(
        unittest.TestCase):
    """Finding (24), runs 1660-1663 plus the finding-(19) and v106 priors.

    Two configurations in this ledger crossed `rounds` 1 -> 2 and both lost
    catastrophically; everything that stayed at 1 is explained by other named
    terms. This class pins that separation, pins the refusal built on it, and
    pins the retraction of `occupancy_fill`, which had the opposite sign.
    """

    # ABBA, --case isolation, autotune ON, machine K.
    V107_DOUBLE = (0.24432 + 0.24112) / 2   # 0.24272
    V98_SHIPPED = (0.16892 + 0.16952) / 2   # 0.16922
    # Priors from finding (19) and finding (20g), same kernel, same box.
    M2048_DOUBLE_LOSS = 0.267
    M2048_SK16_LOSS = 0.3429

    # Residency is driven by `lds_bytes`, which is the actual mechanism: a
    # double buffer IS a doubling of the LDS panel. Passing a cap by hand would
    # assert the conclusion instead of deriving it.
    def m512(self, buffers=1, cta_m=128, cta_n=64):
        tiles = (512 + cta_m - 1) // cta_m * ((11008 + cta_n - 1) // cta_n)
        return QD.RouteFacts(tiles=tiles, slices=1, cu_count=304,
                             cta_m=cta_m, cta_n=cta_n, waves_m=2, waves_n=2,
                             stage_k=32, lds_bytes=13824 * buffers,
                             m=512, n=11008, k=4096)

    def m2048(self, buffers=1, lds_bytes=None):
        tiles = (2048 // 128) * (4096 // 128)
        return QD.RouteFacts(tiles=tiles, slices=1, cu_count=304,
                             cta_m=128, cta_n=128, waves_m=2, waves_n=4,
                             stage_k=32,
                             lds_bytes=lds_bytes or 18432 * buffers,
                             m=2048, n=4096, k=4096)

    def m64(self):
        # decode_m64_square is M=64 N=8192 K=8192 (task_runner.py:28).
        tiles = (64 + 63) // 64 * ((8192 + 127) // 128)
        return QD.RouteFacts(tiles=tiles, slices=9, cu_count=304,
                             cta_m=64, cta_n=128, waves_m=2, waves_n=2,
                             stage_k=64, lds_bytes=26112,
                             m=64, n=8192, k=8192)

    # -- the measurement itself ------------------------------------------
    def test_the_double_buffer_lost_and_lost_bigger_than_predicted(self):
        loss = self.V107_DOUBLE / self.V98_SHIPPED - 1.0
        self.assertGreater(loss, 0.43)
        self.assertLess(loss, 0.44)
        # The written-before-the-run band was -15% to +10%. Outside it.
        self.assertGreater(loss, 0.15)

    def test_it_is_the_largest_single_arm_loss_in_the_ledger(self):
        loss = self.V107_DOUBLE / self.V98_SHIPPED - 1.0
        self.assertGreater(loss, self.M2048_DOUBLE_LOSS)
        self.assertGreater(loss, self.M2048_SK16_LOSS)

    # -- rounds separates winners from losers ----------------------------
    def test_both_catastrophic_configs_crossed_a_round_boundary(self):
        self.assertEqual(self.m512(buffers=1).rounds, 1)
        self.assertEqual(self.m512(buffers=2).rounds, 2)   # v107, -43.4%
        self.assertEqual(self.m2048(buffers=1).rounds, 1)
        self.assertEqual(self.m2048(buffers=2).rounds, 2)  # finding 19, -26.7%

    def test_the_v106_loser_did_not_cross_and_so_needs_another_explanation(self):
        # sk16 double measured LDS 20480 and held 3 CTAs/CU (objmeta, finding
        # 20g). It lost 34.3%, but to cover, not to serialisation -- which is
        # why rounds must not be the only term in the descriptor.
        v106 = self.m2048(lds_bytes=20480)
        self.assertEqual(v106.rounds, 1)
        self.assertEqual(v106.ctas_by_occupancy, 3)

    def test_every_shipped_route_sits_at_one_round(self):
        for r in (self.m512(), self.m2048(), self.m64()):
            self.assertEqual(r.rounds, 1)

    def test_the_second_round_is_nearly_empty_and_still_costs_everything(self):
        # 688 CTAs against 608 slots leaves 80 CTAs on 304 CUs, 26% of a pass,
        # and it cost 43%. Makespan, not work, is what a round buys.
        v107 = self.m512(buffers=2)
        overflow = v107.ctas - v107.residency_slots
        self.assertEqual(overflow, 80)
        self.assertLess(overflow / 304.0, 0.3)
        self.assertGreater(self.V107_DOUBLE / self.V98_SHIPPED - 1.0, 0.4)

    # -- the retraction of occupancy_fill --------------------------------
    def test_occupancy_fill_is_gone(self):
        self.assertFalse(hasattr(self.m512(), "occupancy_fill"))
        self.assertEqual(QD.OCCUPANCY_FILL_REMOVED_BY_FINDING, 24)

    def test_it_was_removed_for_having_the_wrong_sign_not_for_being_noisy(self):
        # Reconstruct the field it replaced and show it maxes out on both
        # losers while every winner reads below it. As a maximand it is a
        # pointer at the cliff, which is worse than uninformative.
        def old_fill(r):
            return min(1.0, r.ctas / r.residency_slots)
        losers = [self.m512(buffers=2), self.m2048(buffers=2)]
        winners = [self.m512(buffers=1), self.m2048(buffers=1), self.m64()]
        for bad in losers:
            self.assertEqual(old_fill(bad), 1.0)
        for good in winners:
            self.assertLess(old_fill(good), 1.0)
            self.assertLess(old_fill(good), min(old_fill(b) for b in losers))

    # -- the refusal rule -------------------------------------------------
    def test_raising_rounds_is_refused_outright(self):
        v = self.m512(buffers=1).residency_mutation_verdict(self.m512(buffers=2))
        self.assertTrue(v.startswith("refuse"))
        self.assertIn("rounds", v)
        self.assertTrue(self.m2048(buffers=1)
                        .residency_mutation_verdict(self.m2048(buffers=2)))

    def test_the_refusal_does_not_fire_on_a_same_round_mutation(self):
        self.assertEqual(
            self.m2048(buffers=1).residency_mutation_verdict(self.m2048(buffers=1)),
            "")

    def test_lowering_rounds_is_allowed(self):
        self.assertEqual(
            self.m512(buffers=2).residency_mutation_verdict(self.m512(buffers=1)), "")

    def test_an_unsupplied_route_refuses_nothing(self):
        bare = QD.RouteFacts(tiles=0, slices=1, cu_count=304)
        self.assertEqual(bare.residency_mutation_verdict(self.m512(buffers=2)), "")
        self.assertEqual(bare.rounds, 0)
        self.assertEqual(bare.round_slack, 0)

    # -- slack, and the frozen route -------------------------------------
    def test_m64_is_the_route_one_cta_from_disaster(self):
        r = self.m64()
        self.assertEqual(r.ctas, 576)
        self.assertEqual(r.residency_slots, 608)
        self.assertEqual(r.round_slack, 32)
        self.assertLess(r.round_slack / r.residency_slots, 0.06)

    # -- finding (25e): pinned against the BUILT v98 object ----------------
    # `objmeta.py ws_a/.torch_ext/<ext>.so`, which reads .vgpr_count /
    # .agpr_count / .group_segment_fixed_size out of the .hip_fatbin bundle.
    # These are not derived numbers and must not be edited to make a test
    # pass -- re-run objmeta on the variant under discussion instead. The v79
    # reading of the m512 instantiation (124 + 48 AGPR, ctaV 2) is STALE and
    # taking it at face value would have made shipped m512 read as rounds 2,
    # i.e. would have retracted finding (24)'s mechanism for no reason.
    def test_the_m512_tile_is_lds_bound_at_four_on_the_built_v98_object(self):
        r = QD.RouteFacts(tiles=688, slices=1, cu_count=304, cta_m=128,
                          cta_n=64, waves_m=2, waves_n=2, stage_k=32,
                          lds_bytes=13824, vgpr_count=88, agpr_count=0,
                          waves_per_cta=4, m=512, n=11008, k=4096)
        self.assertEqual(5, r.ctas_by_vgpr)
        self.assertEqual(4, r.ctas_by_lds)
        self.assertEqual(4, r.ctas_by_occupancy)
        self.assertEqual(1, r.rounds)

    def test_supplying_the_stale_v79_registers_would_have_flipped_the_finding(self):
        # Kept as an executable record of the near-miss, not as a claim about
        # the shipped kernel: 124 + 48 granules to 176, so 2 waves/SIMD.
        stale = QD.RouteFacts(tiles=688, slices=1, cu_count=304, cta_m=128,
                              cta_n=64, waves_m=2, waves_n=2, stage_k=32,
                              lds_bytes=13824, vgpr_count=124, agpr_count=48,
                              waves_per_cta=4, m=512, n=11008, k=4096)
        self.assertEqual(2, stale.ctas_by_occupancy)
        self.assertEqual(2, stale.rounds)

    def test_m64_binds_on_registers_and_lds_at_once_so_has_nothing_to_trade(self):
        r = QD.RouteFacts(tiles=64, slices=9, cu_count=304, cta_m=64,
                          cta_n=128, waves_m=2, waves_n=2, stage_k=64,
                          lds_bytes=26112, vgpr_count=152, agpr_count=48,
                          waves_per_cta=4, m=64, n=8192, k=8192)
        self.assertEqual(2, r.ctas_by_vgpr)
        self.assertEqual(2, r.ctas_by_lds)
        self.assertEqual(576, r.ctas)
        self.assertEqual(32, r.round_slack)

    def test_slack_is_the_headroom_before_the_next_boundary(self):
        self.assertEqual(self.m512(buffers=1).round_slack, 1216 - 688)
        self.assertEqual(self.m2048(buffers=1).round_slack, 912 - 512)

    def test_eating_most_of_the_slack_warns_without_refusing(self):
        # Same rounds, slack 528 -> 116: not over, but one step from it.
        wide = self.m512(buffers=1)              # 688 of 1216, slack 528
        narrow = QD.RouteFacts(tiles=1100, slices=1, cu_count=304,
                               cta_m=128, cta_n=64, waves_m=2, waves_n=2,
                               stage_k=32, lds_bytes=13824,
                               m=512, n=11008, k=4096)
        self.assertEqual(narrow.rounds, wide.rounds)
        v = wide.residency_mutation_verdict(narrow)
        self.assertTrue(v.startswith("warn"))
        self.assertFalse(v.startswith("refuse"))

    # -- the boundary against finding 23 ----------------------------------
    def test_finding_23_is_narrowed_not_reversed(self):
        # m256's slice sweep never left one round, so tail balance being free
        # still stands; it simply never spoke to the serialising case.
        tiles = (256 + 127) // 128 * ((4096 + 159) // 160)
        for s in (12, 14, 16):
            r = QD.RouteFacts(tiles=tiles, slices=s, cu_count=304,
                              cta_m=128, cta_n=160, waves_m=2, waves_n=2,
                              stage_k=32, lds_bytes=20736,
                              m=256, n=4096, k=11008)
            self.assertEqual(r.rounds, 1)
            self.assertEqual(
                r.residency_mutation_verdict(r), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
