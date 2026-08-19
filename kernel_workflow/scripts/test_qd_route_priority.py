#!/usr/bin/env python3
"""Finding (33): rank by what is left, not by how the vendor is doing."""
from __future__ import annotations

import ast
import unittest
import unittest.mock
from pathlib import Path

import qd_robust_stats as robust
import qd_route_priority as priority
import qd_sol_card as sol

# The read-only task whose harness defines the 11 cases. `SUITE_SHAPES` is a
# hand transcription of the `CASES` list there, and until this file grew the
# test below, nothing checked the transcription: a mutation sweep flipped every
# one of the 33 M/N/K values with the whole suite still green. A wrong shape is
# not a cosmetic error -- `shape_workload` turns it into FLOPs and bytes, which
# become the SOL floors, which become the headroom the planner ranks routes by.
TASK_RUNNER = (Path(__file__).resolve().parents[2]
               / "examples/tasks/dense_bf16_gemm_fused/scripts/task_runner.py")


def _harness_cases():
    """The authoritative shapes, parsed out of the task without importing it.

    Importing would pull in torch and the harness; `ast` reads the literal. The
    task is read-only and is never written by this test.
    """
    tree = ast.parse(TASK_RUNNER.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "CASES" for t in node.targets):
            cases = ast.literal_eval(node.value)
            return {c["id"]: (c["M"], c["N"], c["K"]) for c in cases}
    raise AssertionError(f"{TASK_RUNNER} no longer defines a CASES list")


class HarnessTranscriptionTest(unittest.TestCase):
    """(44)'s third corner for the shape table: the source it was copied from."""

    def setUp(self):
        if not TASK_RUNNER.exists():
            self.skipTest(
                f"UNCHECKED: {TASK_RUNNER} is absent, so the 11 harness shapes in "
                "SUITE_SHAPES are transcribed from nothing this run can read. Every "
                "SOL floor derived from them is unverified in this environment.")

    def test_the_case_ids_are_exactly_the_harness_case_ids(self):
        self.assertEqual(sorted(priority.SUITE_SHAPES), sorted(_harness_cases()))

    def test_every_shape_matches_the_harness_value_for_value(self):
        harness = _harness_cases()
        for name, shape in sorted(harness.items()):
            with self.subTest(case=name):
                self.assertEqual(priority.SUITE_SHAPES[name], shape)

    def test_the_transcription_has_no_extra_cases(self):
        # The sanity case (3x5x7) is deliberately not a suite route; a copy that
        # swept it in would add a twelfth entry to every geomean.
        self.assertNotIn("layout_sanity", priority.SUITE_SHAPES)


class ShipPointProvenanceTest(unittest.TestCase):
    """(44)'s third corner for the two latency tables.

    `SHIPPED_ELAPSED_MS` and this file's `VENDOR_MS` are both hand transcriptions
    of one report -- machine-L run 1670, ship point v98 -- and the mutation sweep
    found that nothing checked either of them: every value could be perturbed
    with the suite still green. The module docstring already records the
    provenance in prose; this asserts it against the recorded run, which is the
    difference between a documented number and a checked one (89).

    The report lives under `exp/`, which is gitignored, so on a machine without
    it this skips LOUDLY rather than passing. A silent skip here would be the
    same defect one level up: the reader would see green and conclude the
    numbers were confirmed.
    """

    REPORT = (Path(__file__).resolve().parents[2]
              / "exp/opt_bf16_20260814/perf_g6_1670_mLbase.json")

    def setUp(self):
        if not self.REPORT.exists():
            self.skipTest(
                f"UNCHECKED: {self.REPORT} is absent, so SHIPPED_ELAPSED_MS and "
                "VENDOR_MS are transcribed from a report nothing in this run can "
                "read. Every route ranking derived from them is unverified in "
                "this environment.")
        import json
        payload = json.loads(self.REPORT.read_text(encoding="utf-8"))
        self.cases = {c["test_case_id"]: c for c in payload["test_cases"]}

    def test_the_report_covers_exactly_the_routes_the_tables_claim(self):
        self.assertEqual(sorted(self.cases), sorted(priority.SHIPPED_ELAPSED_MS))

    def test_every_shipped_elapsed_matches_the_recorded_candidate_time(self):
        for context, claimed in sorted(priority.SHIPPED_ELAPSED_MS.items()):
            with self.subTest(case=context):
                self.assertAlmostEqual(
                    self.cases[context]["candidate_ms"], claimed, places=5)

    def test_every_vendor_time_matches_the_recorded_baseline(self):
        # The same report's other column. This one is load-bearing for
        # `TheTwoRankingsDisagreeTest`, whose whole claim is a comparison
        # between the two -- so a mistyped vendor number would not fail, it
        # would quietly weaken or strengthen the finding.
        for context, claimed in sorted(TheTwoRankingsDisagreeTest.VENDOR_MS.items()):
            with self.subTest(case=context):
                self.assertAlmostEqual(
                    self.cases[context]["baseline_ms"], claimed, places=5)

    def test_the_report_measured_the_shapes_the_table_claims(self):
        # A second, independent witness to `SUITE_SHAPES` -- the harness defines
        # the shapes, this is a run that actually executed them.
        for context, shape in sorted(priority.SUITE_SHAPES.items()):
            with self.subTest(case=context):
                row = self.cases[context]
                self.assertEqual((row["M"], row["N"], row["K"]), shape)


class NoRouteIsScoredAgainstAnExtrapolatedCeilingTest(unittest.TestCase):
    """The bandwidth table's own warning, checked against the eleven routes.

    `bandwidth_ceiling` clamps below 32 MB and flags the answer `low` /
    `extrapolated`, and its docstring is explicit about the direction of the
    error: below the measured range the working set is latency- and launch-bound,
    so clamping to 1.42 TB/s **overstates** the achievable rate and a route scored
    against it looks like it has headroom it does not have. That is the paper
    roofline's mistake, one order of magnitude smaller.

    Nothing asserted that no suite route was in that region -- and the margin is
    thinner than anyone would guess: `decode_m2_square` moves 32.03 MiB against a
    lowest measured point of 32 MiB, clearing it by one part in a thousand. A
    shape or a traffic-accounting change that shaved 0.1% off it would move the
    smallest, noisiest route onto an overstated ceiling silently.
    """

    TABLE = sol.MEASURED_GFX942_CEILINGS["peak_bandwidth_bytes_s_by_footprint"]

    def _footprint(self, context):
        return priority.shape_workload(*priority.SUITE_SHAPES[context])[1]

    def test_every_route_lands_on_a_measured_or_interpolated_ceiling(self):
        for context in sorted(priority.SUITE_SHAPES):
            with self.subTest(case=context):
                got = sol.bandwidth_ceiling(self._footprint(context), self.TABLE)
                self.assertFalse(got["extrapolated"])
                self.assertNotEqual("low", got["confidence"])

    def test_the_smallest_route_clears_the_lowest_measured_point(self):
        margin = self._footprint("decode_m2_square") / min(float(k) for k in self.TABLE)
        self.assertGreater(margin, 1.0)
        # Recorded, not required to stay: if a future change moves this the test
        # above is the one that must still hold.
        self.assertLess(margin, 1.01, "the knife-edge in the docstring has moved")

    def test_the_largest_route_is_well_inside_the_upper_end(self):
        largest = max(self._footprint(c) for c in priority.SUITE_SHAPES)
        self.assertLess(largest, max(float(k) for k in self.TABLE))


class ShapeTableTest(unittest.TestCase):
    def test_every_shape_has_a_measured_noise_floor_and_a_ship_point_latency(self):
        # The verdict is a ratio of the two, so a route present in one table
        # and absent from the other would silently take the fail-closed
        # default and read as more closed than it is.
        self.assertEqual(sorted(priority.SUITE_SHAPES), sorted(priority.SHIPPED_ELAPSED_MS))
        self.assertEqual(sorted(priority.SUITE_SHAPES), sorted(robust.MEASURED_NOISE_FLOOR))

    def test_the_workload_counts_each_operand_and_the_output_once(self):
        flops, byte_count = priority.shape_workload(128, 4096, 4096)
        self.assertEqual(2 * 128 * 4096 * 4096, flops)
        self.assertEqual(2 * (128 * 4096 + 4096 * 4096 + 128 * 4096), byte_count)

    def test_an_unknown_case_is_refused_rather_than_defaulted(self):
        with self.assertRaises(priority.RoutePriorityError):
            priority.route_priority("prefill_m4096_square")

    def test_a_non_positive_elapsed_is_refused(self):
        for bad in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(bad=bad):
                with self.assertRaises(priority.RoutePriorityError):
                    priority.route_priority("decode_m8_up", bad)


class VerdictTest(unittest.TestCase):
    def test_decode_m2_is_the_tightest_route_and_its_closure_follows_the_floor(self):
        # This used to assert `closed` outright. It was the route the ledger
        # kept as its example of a closure that cost no GPU time: 2.90% of
        # headroom against machine O's 3.78% floor, slack_to_floor 0.77. The
        # test itself said "this is the route to re-check first if the floors
        # move again", and the floors moved: P's floor here is 0.0224, so the
        # SAME 2.90% headroom now clears it at 1.29 and the route reads
        # `marginal`. Nothing about the kernel or the shape changed.
        #
        # So the durable claim is not the verdict, it is the mechanism: this
        # route is the one with the least headroom relative to its own floor,
        # and its verdict is whatever that ratio says on the current machine.
        # Asserting the verdict directly made a correct table look like a bug.
        #
        # Machine Q sharpened the same point once more. Its table is
        # PROVISIONAL -- flat at the fail-closed default on all eleven routes --
        # so `slack_to_floor` degenerates into plain headroom and the tightest
        # route is decode_m16_square, not decode_m2_square. Which route is
        # tightest is therefore a reading of a machine too (finding 125), and
        # only the mechanism below is asserted unconditionally.
        rows = priority.rank_routes()
        tightest = min(rows, key=lambda r: r["slack_to_floor"])
        if robust.CURRENT_MACHINE in robust.PROVISIONAL_MACHINES:
            floors = {r["context"]: r["noise_floor"] for r in rows}
            self.assertEqual(1, len(set(floors.values())),
                             "a provisional epoch's floors must be flat at the "
                             "default; if they are not, this branch is wrong")
            self.assertEqual(min(rows, key=lambda r: r["remaining_headroom"])["context"],
                             tightest["context"],
                             "with a flat floor the tightest route is just the one "
                             "with the least headroom")
        else:
            # Epoch R made the sentence above literally true for the first time
            # on a MEASURED table: tw008's floors put decode_m16_square (floor
            # 0.0110) tighter than decode_m2_square (0.0093 floor, but far more
            # headroom). The old hardcoded identity was the one thing this
            # comment already said not to assert, so the identity is now a
            # per-epoch RECORD rather than a constant -- still a real regression
            # test, and a new epoch has to state its answer instead of
            # inheriting one.
            tightest_by_epoch = {
                "N": "decode_m2_square",
                "O": "decode_m2_square",
                "P": "decode_m2_square",
                "R": "decode_m16_square",
                # tw003, measured 2026-08-18. Q's floors are the loosest of any
                # measured epoch on decode_m16_square (0.0305 against R's
                # 0.0110), and that route has only 0.0092 of headroom left, so
                # slack_to_floor is 0.30 -- the tightest reading any epoch has
                # produced. decode_m2_square, the historical answer, sits at
                # 1.53 here.
                "Q": "decode_m16_square",
                # tw046, measured 2026-08-18. Third measured epoch in a row to
                # answer decode_m16_square, and the tightest reading yet:
                # slack_to_floor 0.197. Its floor here (0.0470) is the loosest
                # any epoch has recorded on that route -- 4.3x tw008's 0.0110 --
                # against the same 0.0092 of headroom. Worth stating plainly:
                # decode_m2_square, the ORIGINAL hardcoded answer, has now been
                # wrong on three consecutive boxes, so the identity that once
                # looked like a property of the suite was a property of tw035
                # and tw054.
                "T": "decode_m16_square",
                # tw049, measured 2026-08-18. Fourth in a row, and the reading
                # that explains the other three: slack_to_floor 0.66, floor
                # 0.0140 -- a THIRD of T's 0.0470 on the same route, on a box
                # measured hours apart with the same tool.
                #
                # The floors do not sort by box, they sort by TREE. Every table
                # here carries the source_hash it was measured against, and they
                # fall into exactly two groups: R and U on f3da61b3 read 0.0110
                # and 0.0140, while Q and T on 943b1583 read 0.0305 and 0.0470.
                # Same statistic, same 8 repeats, same routes -- the split is
                # which candidate tree happened to be sitting in the task dir.
                # So retract the sentence above about T's floor being "the
                # loosest any epoch has recorded ON THAT ROUTE, 4.3x tw008's":
                # that comparison crosses a tree boundary and prices the tree,
                # not the box. Floors are comparable within a source_hash group
                # and nowhere else. What survives unharmed is the per-epoch USE
                # of a floor -- each table was measured on the tree it governs.
                "U": "decode_m16_square",
                # tw051, measured 2026-08-18. Fifth in a row, and a THIRD tree
                # group for the (49) grouping: floor 0.0087 on source_hash
                # c4b6dba0 (the post-fab4813 canonical), slack_to_floor 1.06
                # against the same 0.0092 of headroom. It lands with R/U's
                # f3da61b3 group (0.0110 / 0.0140) and nowhere near Q/T's
                # 943b1583 group (0.0305 / 0.0470), which is the grouping
                # holding on a tree it was not derived from -- a prediction, not
                # a refit. Note tw051 itself is not new to the lane: wave 1 ran
                # on it while it belonged to no epoch, and it gets the NEW
                # letter V rather than a backfilled old one, per (126).
                "V": "decode_m16_square",
                # tw042, measured 2026-08-18. Sixth in a row, floor 0.0101 on
                # source_hash f87a1ccd (the post-921836d canonical) -- a FOURTH
                # tree, landing in the tight group again alongside R/U/V rather
                # than with Q/T's 943b1583. (49) is now 2-for-2 as a forecast.
                "W": "decode_m16_square",
            }
            expected_route = tightest_by_epoch.get(robust.CURRENT_MACHINE)
            self.assertIsNotNone(
                expected_route,
                f"epoch {robust.CURRENT_MACHINE} has a measured floor table but no "
                f"recorded tightest route. Read it off `rank_routes()` on this box "
                f"and add it here; do not delete this check, it is what notices a "
                f"floor table shifting the suite's binding route.")
            self.assertEqual(expected_route, tightest["context"])
        row = priority.route_priority(tightest["context"])
        # On the SHIPPED kernel's elapsed, which is not a verdict about any
        # candidate (92) -- so the reading is reported as conditional.
        self.assertEqual("needs_fresh_elapsed", row["verdict"])
        self.assertTrue(row["elapsed_is_default"])
        self.assertLess(row["slack_to_floor"], priority.MARGINAL_RATIO,
                        "the tightest route in the suite must not read `open`")
        expected = ("closed" if row["slack_to_floor"] < priority.CLOSED_RATIO
                    else "marginal")
        self.assertEqual(expected, row["verdict_if_elapsed_confirmed"])
        # And the ratio is exactly the headroom-vs-floor comparison, whichever
        # side of 1.0 it lands on for this machine.
        self.assertEqual(row["noise_floor"] > row["remaining_headroom"],
                         row["verdict_if_elapsed_confirmed"] == "closed")

    def test_the_route_the_banner_chased_is_open_but_near_the_bottom(self):
        # `prefill_m128_square` is the only route losing to the vendor, and it
        # is 9th of eleven by slack. If this ever inverts, the finding-(33)
        # redirection needs re-deriving rather than quietly inheriting.
        #
        # The `open` half was epoch-keyed and is now conditional: on a
        # provisional epoch every floor is the fail-closed 0.072, so this
        # route's 12.6% headroom reads `marginal` (ratio 1.75) rather than
        # `open`. That is the floor talking, not the route -- the ORDER, which
        # is what the finding-(33) redirection actually rests on, is unchanged
        # because a flat floor scales every ratio by the same constant.
        rows = priority.rank_routes()
        order = [r["context"] for r in rows]
        row = rows[order.index("prefill_m128_square")]
        if robust.CURRENT_MACHINE in robust.PROVISIONAL_MACHINES:
            self.assertIn(row["verdict_if_elapsed_confirmed"], ("marginal", "open"),
                          "even at the widest floor in the tree this route must not "
                          "read closed; it is the one route losing to the vendor")
        else:
            self.assertEqual("open", row["verdict_if_elapsed_confirmed"])
        self.assertGreater(order.index("prefill_m128_square"),
                           order.index("prefill_m256_down"))

    def test_the_richest_route_is_a_prefill_the_ledger_never_targeted(self):
        rows = priority.rank_routes()
        self.assertEqual("prefill_m1024_down", rows[0]["context"])
        # The most *readable* pool is m512_up. It was m256_down on compulsory
        # traffic; measuring the traffic (finding 35) cut m256_down's slack by
        # more than half while leaving m512_up's nearly untouched, because
        # m256_down re-reads 2.5x and m512_up only 1.3x.
        #
        # Which route wins on READABILITY is machine-keyed, because the floor
        # is: on O it was m512_up, on P it is m2048_square (m512_up's floor
        # went 0.0045 -> 0.0218, 4.8x wider, while m2048_square's moved 0.0060
        # -> 0.0109). The claim that survives an epoch change is that the most
        # readable pool is a PREFILL and not the route the banner chased.
        best_ratio = max(rows, key=lambda r: r["slack_to_floor"])
        self.assertTrue(best_ratio["context"].startswith("prefill_"),
                        f"most readable pool was {best_ratio['context']}")
        self.assertNotEqual("prefill_m128_square", best_ratio["context"])

    def test_closed_routes_sort_last_even_when_their_slack_is_not_the_smallest(self):
        # Sorting purely by slack would let an unreachable route head a
        # priority list, which is the one thing this ranking exists to prevent.
        # Every route is given its own measured elapsed, because only a route
        # closed on its OWN measurement may be deranked (92).
        measured = {name: priority.route_priority(name)["elapsed_ms"]
                    for name in priority.SUITE_SHAPES}
        rows = priority.rank_routes(elapsed_ms_by_context=measured)
        self.assertNotIn("needs_fresh_elapsed", {r["verdict"] for r in rows})
        closed = [i for i, r in enumerate(rows) if r["verdict"] == "closed"]
        if closed:
            self.assertTrue(all(r["verdict"] == "closed" for r in rows[closed[0]:]))

        # On machine P no route in the suite is closed, so the live table alone
        # proves nothing here -- the assertion above is vacuous and the old
        # `next(...)` raised StopIteration on a perfectly correct ranking.
        # Force one: an elapsed just above the SOL floor leaves headroom
        # ~1e-4, under every floor in the table. The route chosen is the one
        # with the LARGEST slack, so sorting on slack alone would put it first
        # and only the deranking can move it last.
        richest = rows[0]["context"]
        sol_ms = priority.route_priority(richest)["sol_ms"]
        forced = dict(measured)
        forced[richest] = sol_ms * 1.0001
        ranked = priority.rank_routes(elapsed_ms_by_context=forced)
        self.assertEqual("closed",
                         next(r for r in ranked if r["context"] == richest)["verdict"])
        self.assertEqual(richest, ranked[-1]["context"],
                         "a closed route must sort last however much raw slack "
                         "it would otherwise have ranked on")

    def test_an_unmeasured_route_does_not_sort_as_a_closed_one(self):
        """(92). A route nobody measured and a route with nothing left are
        opposite states. Collapsing them buries exactly the routes that most
        need a measurement -- permanently, because a buried route is never
        dispatched and so never measured. The mistake removes its own evidence.
        """
        # `decode_m2_square` is the one that reads closed on the shipped
        # elapsed, and it is the reason this matters: it would sort last
        # forever on a number belonging to a different kernel.
        rows = priority.rank_routes()
        self.assertTrue(all(r["verdict"] == "needs_fresh_elapsed" for r in rows),
                        "a bare rank_routes() has measured nothing and must say so")
        by_slack = sorted(rows, key=lambda r: -float(r["slack_ms"]))
        self.assertEqual([r["context"] for r in by_slack], [r["context"] for r in rows],
                         "unmeasured routes must rank on slack alone, with no "
                         "conditional verdict allowed to derank them")

    def test_a_conditional_verdict_is_still_reported(self):
        # Withholding the verdict is the fix; withholding the READING would
        # just make the caller re-derive it from a worse position.
        row = priority.route_priority("decode_m2_square")
        # Machine-keyed: `closed` on L/N/O, `marginal` on P. What must hold on
        # every machine is that the conditional reading is a real verdict and
        # that confirming the elapsed reproduces it exactly.
        self.assertIn(row["verdict_if_elapsed_confirmed"],
                      ("closed", "marginal", "open"))
        fresh = priority.route_priority("decode_m2_square", row["elapsed_ms"])
        self.assertNotEqual("needs_fresh_elapsed", fresh["verdict"])
        self.assertFalse(fresh["elapsed_is_default"])
        self.assertEqual(fresh["verdict"], row["verdict_if_elapsed_confirmed"],
                         "the conditional verdict must be the one a confirming "
                         "measurement would produce, or it is not a preview of "
                         "anything")

    def test_a_slower_elapsed_opens_more_headroom_and_never_less(self):
        base = priority.route_priority("prefill_m256_down")
        slower = priority.route_priority("prefill_m256_down", base["elapsed_ms"] * 2)
        self.assertGreater(slower["slack_ms"], base["slack_ms"])
        self.assertGreater(slower["sol_gap"], base["sol_gap"])
        self.assertEqual("caller-supplied", slower["elapsed_provenance"])

    def test_a_kernel_at_the_sol_floor_reads_as_closed_on_every_route(self):
        # The reductio: perfect kernels have nothing left, whatever the floor.
        for name in priority.SUITE_SHAPES:
            with self.subTest(name=name):
                row = priority.route_priority(name)
                at_floor = priority.route_priority(name, row["sol_ms"])
                self.assertAlmostEqual(1.0, at_floor["sol_gap"], places=6)
                self.assertEqual("closed", at_floor["verdict"])

    def test_an_elapsed_override_naming_a_non_route_is_refused(self):
        with self.assertRaises(priority.RoutePriorityError):
            priority.rank_routes(None, {"not_a_case": 1.0})


class RegimeTest(unittest.TestCase):
    def test_only_three_routes_clear_the_ridge_once_traffic_is_measured(self):
        # Every prefill above 256 rows is compute-bound; every decode route and
        # the two small prefills are memory-bound. That split is the shape of
        # the suite, and a change in it means either the kernel or the traffic
        # table moved -- both worth failing over.
        rows = priority.rank_routes()
        self.assertEqual({"prefill_m512_up", "prefill_m1024_down", "prefill_m2048_square"},
                         {r["context"] for r in rows if r["regime"] == "compute_bound"})
        self.assertEqual(8, sum(r["regime"] == "memory_bound" for r in rows))
        # And the regime is INVARIANT to how much the route re-reads, which is
        # not obvious and is worth pinning: measuring traffic divides the
        # arithmetic intensity, but under finding (35) it also raises the
        # ceiling that sets the ridge, and on this suite the two move together
        # closely enough that no route crosses. So measured traffic corrects
        # the magnitude of a memory-bound route's floor, not which side of the
        # ridge it sits on.
        for name in priority.SUITE_SHAPES:
            with self.subTest(name=name):
                row = priority.route_priority(name)
                compulsory = priority.route_priority(
                    name, traffic_bytes=row["compulsory_bytes"])
                self.assertEqual(row["regime"], compulsory["regime"])

    def test_the_ridge_point_moves_across_the_suite_because_the_ceiling_does(self):
        # Finding (28)/(29): a footprint-indexed ceiling makes the ridge a
        # function of the shape. A scalar ceiling would make every ridge equal
        # here, and this assertion is what would catch a regression to one.
        ridges = {r["ridge_point"] for r in priority.rank_routes()}
        self.assertGreater(len(ridges), 1)
        self.assertGreater(max(ridges) / min(ridges), 1.5)

    def test_every_route_resolves_its_ceiling_from_measured_data(self):
        # If any route fell outside the measured footprint table it would come
        # back extrapolated, and its slack would be a guess.
        for r in priority.rank_routes():
            with self.subTest(context=r["context"]):
                self.assertEqual("measured_interpolated", r["bandwidth_ceiling_confidence"])


class MeasuredTrafficTest(unittest.TestCase):
    """Finding (34): the compulsory minimum is a lower bound, not the traffic."""

    def test_every_route_now_has_counters_and_says_so(self):
        rows = priority.rank_routes()
        self.assertEqual(11, sum(r["traffic_basis"] == "measured" for r in rows))
        for r in rows:
            with self.subTest(context=r["context"]):
                self.assertIsNotNone(r["traffic_provenance"])

    def test_a_route_without_counters_falls_back_and_admits_it(self):
        # The fallback path is no longer exercised by any real route, so it is
        # exercised here instead: silently ranking a lower bound alongside a
        # measurement is exactly what finding (34) cost.
        with unittest.mock.patch.object(priority, "MEASURED_TRAFFIC_BYTES", {}):
            row = priority.route_priority("prefill_m1024_down")
        self.assertEqual("compulsory", row["traffic_basis"])
        self.assertIsNone(row["traffic_provenance"])
        self.assertEqual(1.0, row["traffic_amplification"])

    def test_measured_traffic_shrinks_the_slack_it_corrects(self):
        # More traffic means a higher memory floor means LESS slack. The
        # ceiling rises with traffic too (finding 35), but sublinearly, so the
        # sign is unambiguous. If this ever came out the other way the sign of
        # the correction is wrong and every ranking built on it is too.
        for name in priority.SUITE_SHAPES:
            with self.subTest(name=name):
                measured = priority.route_priority(name)
                compulsory = priority.route_priority(
                    name, traffic_bytes=measured["compulsory_bytes"])
                self.assertLessEqual(measured["slack_ms"], compulsory["slack_ms"])
                self.assertLessEqual(measured["sol_gap"], compulsory["sol_gap"])
                if measured["regime"] == "compute_bound":
                    # Traffic cannot touch a route whose floor is MFMA issue:
                    # equal, not merely smaller, and that is the correct answer.
                    self.assertAlmostEqual(measured["sol_gap"], compulsory["sol_gap"])
                elif measured["traffic_amplification"] > 1.2:
                    self.assertLess(measured["sol_gap"], compulsory["sol_gap"])

    def test_indexing_the_ceiling_by_the_working_set_would_beat_the_hardware(self):
        """Finding (35), stated as the falsification that forced it.

        Indexing the bandwidth ceiling by distinct working set rather than by
        bytes actually streamed puts two routes BELOW the hardware floor --
        sol_gap under 1.0, i.e. the kernel finishing faster than physics. That
        is not a small mis-estimate, it is a refuted model, and this test is
        here so nobody quietly reverts to the intuitive reading.
        """
        impossible = []
        for name in priority.SUITE_SHAPES:
            row = priority.route_priority(name)
            m, n, k = priority.SUITE_SHAPES[name]
            flops, compulsory = priority.shape_workload(m, n, k)
            card = sol.build_sol_card(
                post_selection=True, achieved_flops=flops,
                achieved_bytes=row["traffic_bytes"],
                elapsed_s=row["elapsed_ms"] / 1000.0, dtype=priority.DTYPE,
                arch=priority.ARCH, footprint_bytes=compulsory)
            self.assertGreaterEqual(row["sol_gap"], 1.0)   # the shipped reading
            if card["sol_gap"] < 1.0:
                impossible.append(name)
        self.assertEqual(["decode_m16_square", "prefill_m128_square"], sorted(impossible))

    def test_traffic_below_compulsory_is_refused_as_a_broken_measurement(self):
        row = priority.route_priority("decode_m8_up")
        with self.assertRaises(priority.RoutePriorityError):
            priority.route_priority("decode_m8_up", traffic_bytes=row["compulsory_bytes"] - 1)

    def test_a_traffic_override_naming_a_non_route_is_refused(self):
        with self.assertRaises(priority.RoutePriorityError):
            priority.rank_routes(None, None, {"not_a_case": 1e9})

    def test_the_l2_request_volume_the_counters_saw_matches_the_tiling(self):
        # The load-bearing consistency check of finding (34): 512 CTAs each
        # pulling a full 128-row A panel and 128-column B panel over K=4096 is
        # 1024 MB, and `TCP_TCC_READ_REQ_sum` counted 16777216 x 64 B = 1024 MB
        # exactly. Nothing else in the chain has to be assumed.
        ctas, cta_m, cta_n, k = 512, 128, 128, 4096
        self.assertEqual(16777216 * 64, ctas * (cta_m + cta_n) * k * 2)
        # ... and the recorded DRAM traffic is the size-weighted EA figure,
        # which agrees with the independent `TCC_MISS x 128` reading (153.0 MB)
        # to under 1%. Two derivations from different counters landing on the
        # same number is what makes the table quotable at all.
        recorded = priority.MEASURED_TRAFFIC_BYTES["prefill_m2048_square"]
        self.assertLess(abs(recorded - 1195333 * 128.0) / recorded, 0.01)


class TheTwoRankingsDisagreeTest(unittest.TestCase):
    """The claim that motivated the whole finding, asserted rather than recalled."""

    #: Machine-L run 1670 `baseline_ms` — the vendor column of the same report.
    VENDOR_MS = {
        "decode_m2_square": 0.02752, "decode_m8_up": 0.05996,
        "decode_m16_square": 0.02788, "decode_m32_down": 0.07264,
        "decode_m64_square": 0.10784, "decode_m96_up": 0.07164,
        "prefill_m128_square": 0.03480, "prefill_m256_down": 0.13416,
        "prefill_m512_up": 0.17320, "prefill_m1024_down": 0.29228,
        "prefill_m2048_square": 0.29292,
    }

    def spearman(self, a: list[str], b: list[str]) -> float:
        n = len(a)
        ra = {v: i for i, v in enumerate(a)}
        rb = {v: i for i, v in enumerate(b)}
        d2 = sum((ra[v] - rb[v]) ** 2 for v in a)
        return 1 - 6 * d2 / (n * (n * n - 1))

    def test_worst_speedup_does_not_predict_worst_sol_gap(self):
        rows = priority.rank_routes()
        speedups = {r["context"]: self.VENDOR_MS[r["context"]] / r["elapsed_ms"] for r in rows}
        by_speedup = sorted(speedups, key=lambda c: speedups[c])
        by_gap = [r["context"] for r in sorted(rows, key=lambda r: -r["sol_gap"])]
        self.assertLess(abs(self.spearman(by_speedup, by_gap)), 0.25)

    def test_the_shipped_kernel_is_closer_to_the_floor_than_the_vendor_in_geomean(self):
        import math
        rows = priority.rank_routes()
        cand = math.exp(sum(math.log(r["sol_gap"]) for r in rows) / len(rows))
        vendor = math.exp(sum(
            math.log(priority.route_priority(r["context"],
                                             self.VENDOR_MS[r["context"]])["sol_gap"])
            for r in rows) / len(rows))
        self.assertLess(cand, vendor)


class ScalarCeilingWouldMisrankTest(unittest.TestCase):
    """Finding (29) refused to pick a scalar ceiling. This is why it was right."""

    def regime_under(self, bw: float, context: str) -> str:
        m, n, k = priority.SUITE_SHAPES[context]
        flops, byte_count = priority.shape_workload(m, n, k)
        ridge = sol.MEASURED_GFX942_CEILINGS["peak_flops"]["bf16"] / bw
        return "compute_bound" if flops / byte_count >= ridge else "memory_bound"

    def test_two_routes_change_regime_depending_on_which_scalar_is_picked(self):
        table = sol.MEASURED_GFX942_CEILINGS["peak_bandwidth_bytes_s_by_footprint"]
        low, high = min(table.values()), max(table.values())
        flipped = [r["context"] for r in priority.rank_routes()
                   if len({r["regime"], self.regime_under(low, r["context"]),
                           self.regime_under(high, r["context"])}) > 1]
        self.assertEqual(["prefill_m512_up", "prefill_m256_down"],
                         sorted(flipped, key=lambda c: -priority.SUITE_SHAPES[c][0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
