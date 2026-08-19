#!/usr/bin/env python3
"""GPU-free tests for qd_robust_stats.py."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("qd_robust_stats.py")
SPEC = importlib.util.spec_from_file_location("qd_robust_stats", SCRIPT)
assert SPEC and SPEC.loader
QRS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QRS
SPEC.loader.exec_module(QRS)


def _measured_tables() -> dict[str, dict[str, float]]:
    """The epochs whose floors were actually measured on that box.

    A provisional epoch carries a structurally normal table filled with the
    fail-closed default so that every code path behaves, but its numbers are a
    placeholder. Tests that assert something about a MEASUREMENT -- spread,
    ordering, one route being quieter than another -- must ask this, not
    `MEASURED_NOISE_FLOOR_BY_MACHINE`, or they assert that a placeholder is data.
    """
    return {m: dict(t) for m, t in QRS.MEASURED_NOISE_FLOOR_BY_MACHINE.items()
            if m not in QRS.PROVISIONAL_MACHINES}


class EpochIdentityTest(unittest.TestCase):
    """Finding (126). `CURRENT_MACHINE` was a remembered constant with the
    hostname only in a comment. The container was restored from a docker
    snapshot onto a third box in one day and the letter went on reading 'P'
    (tw008) while `hostname` returned tw003 -- so every floor in use was a live
    false-provenance claim, and nothing in the tree could notice.

    A hostname cannot be asserted equal to a constant (CI runs elsewhere, and
    the next restore lands somewhere new), so the check is conditional in the
    only direction that is always true: if this box is a REGISTERED epoch, the
    letter must be that epoch; if it is not registered, the letter must be an
    explicitly provisional one. Both branches are red for the failure that
    actually happened -- a registered letter pinned to the wrong registered box.
    """

    def test_the_epoch_letter_matches_the_host(self):
        import socket
        host = socket.gethostname()
        letter = QRS.machine_for_host(host)
        if letter is not None:
            self.assertEqual(
                letter, QRS.CURRENT_MACHINE,
                f"this process is running on {host!r}, which is registered as "
                f"epoch {letter!r}, but CURRENT_MACHINE says "
                f"{QRS.CURRENT_MACHINE!r}; the floors in use belong to another box")
        else:
            self.assertIn(
                QRS.CURRENT_MACHINE, QRS.PROVISIONAL_MACHINES,
                f"this process is running on {host!r}, which is not registered in "
                f"MACHINE_HOSTNAME, yet CURRENT_MACHINE claims the measured epoch "
                f"{QRS.CURRENT_MACHINE!r}. Register the host, or pin a provisional "
                f"epoch until its floors are measured here")

    def test_machine_for_host_does_not_match_an_unregistered_name(self):
        self.assertIsNone(QRS.machine_for_host("no-such-box-12345"))
        # The pre-convention epochs record None rather than a name. None must
        # never match, or every unregistered box would resolve to machine L.
        self.assertIn(None, QRS.MACHINE_HOSTNAME.values())
        self.assertIsNone(QRS.machine_for_host("None"))

    def test_every_registered_host_resolves_to_its_newest_letter(self):
        """A host maps to ONE epoch, and it is the most recent one.

        This used to assert a bijection, which held only while no box was ever
        re-used. tw008 carried P, then carried R after a second restore, so the
        mapping is many-to-one and the question became *which* letter a re-used
        host resolves to. It must be the newest: resolving tw008 to P would
        reinstate floors measured in the previous container, and the measured
        R table is up to 4.2x wider than P's on two routes -- so the stale
        answer is wrong in the admitting direction, which is (126)'s direction.
        """
        newest: dict[str, str] = {}
        for letter, host in QRS.MACHINE_HOSTNAME.items():
            if host is not None:
                newest[host] = letter          # later entries win, by construction
        for host, letter in newest.items():
            with self.subTest(host=host):
                self.assertEqual(letter, QRS.machine_for_host(host))

    def test_a_retired_letter_is_not_what_its_re_used_host_resolves_to(self):
        """The specific regression the change above introduced the risk of."""
        shared = [l for l, h in QRS.MACHINE_HOSTNAME.items() if h == "tw008"]
        self.assertGreater(len(shared), 1,
                           "tw008 no longer carries more than one epoch; this test "
                           "is vacuous and the many-to-one path is untested")
        self.assertEqual(shared[-1], QRS.machine_for_host("tw008"))
        for retired in shared[:-1]:
            with self.subTest(retired=retired):
                self.assertNotEqual(retired, QRS.machine_for_host("tw008"))

    def test_every_epoch_in_the_floor_table_has_a_hostname_entry(self):
        # One-directional on purpose: an epoch can be known without ever having
        # been measured (M is a gfx90a box with no table), but a table whose
        # epoch has no entry is a set of floors with no provenance at all.
        self.assertLessEqual(set(QRS.MEASURED_NOISE_FLOOR_BY_MACHINE),
                             set(QRS.MACHINE_HOSTNAME),
                             "a floor table exists for an epoch that MACHINE_HOSTNAME "
                             "has never heard of, so nothing can check which box it "
                             "came from")

    def test_a_provisional_epoch_does_not_claim_to_be_measured(self):
        for machine in QRS.PROVISIONAL_MACHINES:
            with self.subTest(machine=machine):
                table = QRS.MEASURED_NOISE_FLOOR_BY_MACHINE[machine]
                self.assertTrue(table, "a provisional epoch still needs a full table "
                                       "so that every code path behaves")
                for context in table:
                    self.assertFalse(QRS.floor_is_measured(context, machine),
                                     f"{context} on provisional {machine} reports as "
                                     f"measured")
                    self.assertEqual(QRS.DEFAULT_NOISE_FLOOR, table[context],
                                     "a provisional floor must be the fail-closed "
                                     "default, not a guess")

    def test_a_measured_epoch_reports_its_routes_as_measured(self):
        for machine, table in _measured_tables().items():
            with self.subTest(machine=machine):
                for context in table:
                    self.assertTrue(QRS.floor_is_measured(context, machine))
                self.assertFalse(QRS.floor_is_measured("decode_m4096_sideways", machine))


class RobustStatsTest(unittest.TestCase):
    def test_empty_samples(self):
        stats = QRS.robust_stats([])
        self.assertEqual({"n": 0, "median": 0.0, "mad": 0.0, "bound_radius": 0.0,
                          "lower": 0.0, "upper": 0.0}, stats)

    def test_single_sample_carries_the_noise_floor_not_a_point(self):
        # Retracted expectation: this asserted lower == upper == median, i.e.
        # the narrowest possible interval for the sample that knows the least.
        # Finding (26) replaced it with the measured floor.
        stats = QRS.robust_stats([2.5])
        self.assertEqual(1, stats["n"])
        self.assertEqual(2.5, stats["median"])
        self.assertEqual(0.0, stats["mad"])
        self.assertLess(stats["lower"], 2.5)
        self.assertGreater(stats["upper"], 2.5)
        self.assertAlmostEqual(2.5 * QRS.DEFAULT_NOISE_FLOOR, stats["bound_radius"])

    def test_identical_samples_do_not_get_a_zero_width_interval(self):
        # The hole finding (26) closed. Three equal samples put MAD at 0, and
        # the interval then admitted a challenger better by one nanosecond.
        # On decode_m2_square 3.2% of n=3 draws from a same-variant pool land
        # here, so this was reachable, not theoretical.
        stats = QRS.robust_stats([3.0, 3.0, 3.0])
        self.assertEqual(3.0, stats["median"])
        self.assertEqual(0.0, stats["mad"])
        self.assertLess(stats["lower"], 3.0)
        self.assertGreater(stats["upper"], 3.0)

    def test_the_floor_is_per_route_and_the_spread_is_machine_specific(self):
        # The claim this test used to make -- "spans an order of magnitude" --
        # was a machine-L fact (0.072 / 0.005 = 14x) and it does NOT survive the
        # move to machine N, where the same eleven routes span only 3.5x
        # (0.0378 / 0.0108). Most of L's spread was the one-time priming penalty
        # (105) landing on the short routes; debiasing the harness compressed it.
        # The load-bearing part is unchanged and is what is asserted: the floor
        # is per route, and the quietest and loudest differ by enough that a
        # single repeat-count rule cannot serve both ends of the suite.
        #
        # A PROVISIONAL epoch is exempt and must be: its table is flat at the
        # fail-closed default by construction, not by measurement, so asserting
        # a spread on it would be asserting that a placeholder is data.
        measured = _measured_tables()
        self.assertTrue(measured, "no epoch has a measured table left to check")
        for machine, table in sorted(measured.items()):
            with self.subTest(machine=machine):
                self.assertGreater(max(table.values()), min(table.values()) * 3.0)
        # ... and the ordering reaches noise_floor for every measured epoch,
        # named explicitly, not only for whichever one happens to be current.
        for machine, table in sorted(measured.items()):
            with self.subTest(machine=machine, path="noise_floor"):
                quiet_id = min(table, key=table.get)
                loud_id = max(table, key=table.get)
                self.assertGreater(QRS.noise_floor(loud_id, machine=machine),
                                   QRS.noise_floor(quiet_id, machine=machine) * 3.0)
        # ... and on a measured current machine it reaches robust_stats too.
        if QRS.CURRENT_MACHINE in measured:
            table = measured[QRS.CURRENT_MACHINE]
            quiet = QRS.robust_stats([1.0, 1.0, 1.0], context=min(table, key=table.get))
            loud = QRS.robust_stats([1.0, 1.0, 1.0], context=max(table, key=table.get))
            self.assertGreater(loud["bound_radius"], quiet["bound_radius"] * 3.0)

    def test_an_unknown_route_gets_the_widest_floor_not_the_narrowest(self):
        unknown = QRS.robust_stats([1.0, 1.0, 1.0], context="decode_m4096_sideways")
        # Widest measured ANYWHERE, not the widest on the current machine: an
        # unmeasured route has no epoch, so it cannot borrow this epoch's spread.
        loudest = max(v for t in QRS.MEASURED_NOISE_FLOOR_BY_MACHINE.values()
                      for v in t.values())
        self.assertAlmostEqual(loudest, unknown["bound_radius"])
        self.assertAlmostEqual(QRS.DEFAULT_NOISE_FLOOR, unknown["bound_radius"])
        # The control: "widest anywhere" and "widest here" must be different
        # rules for some epoch, or the fallback is untested. Asking it of the
        # CURRENT epoch made it a fact about which host we are on -- false on a
        # provisional epoch (flat at the default) and on whichever epoch
        # supplies the max. Ask it structurally instead, and only ask the live
        # question where the current epoch can answer it.
        narrower = [m for m, t in QRS.MEASURED_NOISE_FLOOR_BY_MACHINE.items()
                    if max(t.values()) < loudest]
        self.assertTrue(
            narrower,
            "every epoch's widest floor equals the default, so an unknown route "
            "and the loudest known route are indistinguishable and this test "
            "cannot tell the two fallback rules apart")
        if QRS.CURRENT_MACHINE in narrower:
            self.assertGreater(
                loudest,
                max(QRS.MEASURED_NOISE_FLOOR_BY_MACHINE[QRS.CURRENT_MACHINE].values()))

    def test_a_real_mad_still_wins_when_it_is_wider_than_the_floor(self):
        # The floor is a floor, not a replacement: a route that genuinely
        # scatters must not have its interval narrowed to the table value.
        stats = QRS.robust_stats([1.0, 2.0, 3.0], context="prefill_m256_down")
        self.assertAlmostEqual(2.0, stats["bound_radius"])

    def test_context_rows_pick_up_their_own_floor_without_being_told(self):
        # Which route is loud and which is quiet is a property of the CURRENT
        # machine's table, not a constant: prefill_m256_down was the quieter of
        # this pair on machine O (0.0097 vs 0.0416) and is the louder on P
        # (0.0430 vs 0.0224). Hard-coding the pair made this test fail on a
        # correct table, which is the wrong way round -- it must fail when a
        # row stops using its OWN floor, so read the two ends off the table.
        #
        # On a PROVISIONAL epoch the table is flat at the fail-closed default,
        # so no pair of routes can be ordered and the ordering half of this
        # test has nothing to say. What still has to hold there is that every
        # row went through the per-route lookup and landed on that route's own
        # (equal) floor -- a row that skipped the lookup entirely would be
        # caught by the same assertion. The ordering mechanism is then checked
        # against every measured epoch by name.
        table = QRS.MEASURED_NOISE_FLOOR_BY_MACHINE[QRS.CURRENT_MACHINE]
        quiet = min(table, key=table.get)
        loud = max(table, key=table.get)
        rows = QRS.context_robust_stats({quiet: [1.0, 1.0, 1.0],
                                         loud: [1.0, 1.0, 1.0]})
        by = {r["name"]: r for r in rows}
        for name in (quiet, loud):
            self.assertAlmostEqual(by[name]["upper"] - by[name]["median"],
                                   QRS.noise_floor(name),
                                   msg=f"{name}'s row did not use {name}'s own floor")
        if table[quiet] < table[loud]:
            self.assertLess(by[quiet]["upper"], by[loud]["upper"])
        else:
            self.assertTrue(
                QRS.CURRENT_MACHINE in QRS.PROVISIONAL_MACHINES,
                "a MEASURED machine's table is flat; either the measurement "
                "collapsed or a placeholder was left in the measured set")
            for machine, other in sorted(_measured_tables().items()):
                with self.subTest(machine=machine):
                    q = min(other, key=other.get)
                    ld = max(other, key=other.get)
                    self.assertLess(QRS.noise_floor(q, machine=machine),
                                    QRS.noise_floor(ld, machine=machine))

    def test_median_and_bounds_ordering(self):
        stats = QRS.robust_stats([1.0, 2.0, 3.0])
        self.assertEqual(2.0, stats["median"])
        self.assertLessEqual(stats["lower"], stats["median"])
        self.assertLessEqual(stats["median"], stats["upper"])

    def test_outlier_widens_bounds_instead_of_narrowing(self):
        tight = QRS.robust_stats([10.0, 10.1, 9.9])
        with_outlier = QRS.robust_stats([10.0, 10.1, 9.9, 1000.0])
        self.assertGreaterEqual(with_outlier["upper"] - with_outlier["lower"],
                                 tight["upper"] - tight["lower"])

    def test_median_resists_a_single_extreme_outlier(self):
        stats = QRS.robust_stats([10.0, 10.1, 9.9, 10000.0])
        self.assertLess(stats["median"], 20.0, "median must not be dragged up by one spike")

    def test_lower_bound_never_goes_non_positive(self):
        stats = QRS.robust_stats([0.001, 0.0011, 5.0])
        self.assertGreater(stats["lower"], 0.0)

    def test_nan_samples_are_dropped(self):
        stats_with_nan = QRS.robust_stats([1.0, 2.0, 3.0, float("nan")])
        stats_without = QRS.robust_stats([1.0, 2.0, 3.0])
        self.assertEqual(stats_without, stats_with_nan)

    def test_bounds_use_two_raw_mads(self):
        stats = QRS.robust_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(2.0, QRS.MAD_BOUND_MULTIPLIER)
        self.assertAlmostEqual(stats["mad"] * 2.0, stats["bound_radius"])
        self.assertAlmostEqual(stats["median"] - stats["bound_radius"], stats["lower"])
        self.assertAlmostEqual(stats["median"] + stats["bound_radius"], stats["upper"])


class ContextRobustStatsTest(unittest.TestCase):
    def test_shape_matches_case_samples_schema(self):
        rows = QRS.context_robust_stats({"case_b": [1.0, 2.0], "case_a": [3.0, 3.0, 3.0]})
        self.assertEqual(["case_a", "case_b"], [r["name"] for r in rows])
        for row in rows:
            self.assertEqual({"name", "samples", "median", "mad", "lower", "upper"}, set(row))

    def test_samples_are_preserved_as_floats(self):
        rows = QRS.context_robust_stats({"c": [1, 2, 3]})
        self.assertEqual([1.0, 2.0, 3.0], rows[0]["samples"])

    def test_empty_mapping_yields_empty_list(self):
        self.assertEqual([], QRS.context_robust_stats({}))

    def test_deterministic_ordering_regardless_of_input_order(self):
        first = QRS.context_robust_stats({"z": [1.0], "a": [2.0], "m": [3.0]})
        second = QRS.context_robust_stats({"a": [2.0], "m": [3.0], "z": [1.0]})
        self.assertEqual(first, second)


class CombineContextsTest(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual({"score": 0.0, "median": 0.0, "mad": 0.0, "lower": 0.0, "upper": 0.0},
                          QRS.combine_contexts([]))

    def test_single_context_passes_through(self):
        rows = QRS.context_robust_stats({"only": [2.0, 2.0, 2.0]})
        combined = QRS.combine_contexts(rows)
        self.assertEqual(2.0, combined["median"])
        self.assertEqual(2.0, combined["score"])

    def test_combination_is_the_mean_across_contexts(self):
        rows = [
            {"name": "a", "median": 2.0, "mad": 0.0, "lower": 2.0, "upper": 2.0},
            {"name": "b", "median": 4.0, "mad": 0.0, "lower": 4.0, "upper": 4.0},
        ]
        combined = QRS.combine_contexts(rows)
        self.assertEqual(3.0, combined["median"])
        self.assertEqual(3.0, combined["lower"])
        self.assertEqual(3.0, combined["upper"])

    def test_bounds_are_sound_under_linearity(self):
        # For any per-context [lower_i, upper_i], the true combined value's bound
        # is [mean(lower_i), mean(upper_i)] regardless of covariance -- so the
        # combined lower must never exceed the combined median, and vice versa.
        rows = [
            {"name": "a", "median": 5.0, "mad": 1.0, "lower": 3.0, "upper": 7.0},
            {"name": "b", "median": 1.0, "mad": 0.5, "lower": 0.2, "upper": 1.8},
            {"name": "c", "median": 9.0, "mad": 2.0, "lower": 5.0, "upper": 13.0},
        ]
        combined = QRS.combine_contexts(rows)
        self.assertLessEqual(combined["lower"], combined["median"])
        self.assertLessEqual(combined["median"], combined["upper"])

    def test_missing_mad_defaults_to_zero(self):
        rows = [{"name": "a", "median": 1.0, "lower": 1.0, "upper": 1.0}]
        combined = QRS.combine_contexts(rows)
        self.assertEqual(0.0, combined["mad"])


class NoiseFloorProvenanceTest(unittest.TestCase):
    """(44)'s third corner for `MEASURED_NOISE_FLOOR`.

    The module docstring states exactly how the eleven floors were obtained --
    max relative deviation from the median over four same-variant full-suite
    runs (1672/1673/1676/1677, machine L, v98), and over twenty isolated runs
    for `decode_m2_square` (1687-1721, spanning GPU 2 and GPU 3). Nothing
    recomputed them. A mutation sweep moved every one of the eleven values with
    the suite still green, which is the same shape as the shape-table finding:
    a number whose derivation is written in prose is documented, not checked
    (89).

    These floors are not decoration. `robust_stats` widens every interval to at
    least `|median| * noise_floor(context)`, and admission requires the
    challenger's lower bound to clear the incumbent's upper. A floor that is too
    small admits noise as a win on exactly the routes where noise is largest.

    The reports live under `exp/`, which is gitignored, so this skips LOUDLY
    where they are absent rather than passing.
    """

    RUN_DIR = Path(__file__).resolve().parents[2] / "exp/opt_bf16_20260814"
    SUITE_RUNS = (1672, 1673, 1676, 1677)
    ISOLATED_RANGE = (1687, 1721)

    @staticmethod
    def _max_rel_dev(xs):
        import statistics
        median = statistics.median(xs)
        return max(abs(x - median) / median for x in xs)

    def _candidate_ms(self, path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {c["test_case_id"]: c["candidate_ms"] for c in payload["test_cases"]}

    def setUp(self):
        import re
        self.suite = [self.RUN_DIR / f"perf_g6_{r}_v98.json" for r in self.SUITE_RUNS]
        low, high = self.ISOLATED_RANGE
        self.isolated = sorted(
            p for p in self.RUN_DIR.glob("perf_g6_1*_v98*.json")
            if low <= int(re.search(r"perf_g6_(\d+)_", p.name).group(1)) <= high)
        missing = [p.name for p in self.suite if not p.exists()]
        if missing or len(self.isolated) != 20:
            self.skipTest(
                f"UNCHECKED: {self.RUN_DIR} does not hold the runs the floors were "
                f"derived from (missing full-suite runs {missing}; "
                f"{len(self.isolated)} of 20 isolated runs). Every value in "
                "MEASURED_NOISE_FLOOR is unverified in this environment, and so is "
                "every admission interval widened by one.")

    @staticmethod
    def _agrees(computed, claimed):
        # Compared at the precision the table is written to, which is the only
        # precision it claims. `0.0047` is `0.005` there, and demanding more
        # would fail on the rounding rather than on the number.
        decimals = len(str(claimed).split(".")[1])
        return round(computed, decimals) == claimed

    def test_the_ten_suite_floors_recompute_from_the_recorded_runs(self):
        per_case = {}
        for path in self.suite:
            for case, ms in self._candidate_ms(path).items():
                per_case.setdefault(case, []).append(ms)
        for case, claimed in sorted(
                QRS.MEASURED_NOISE_FLOOR_BY_MACHINE["L"].items()):
            if case == "decode_m2_square":
                continue                      # its own pool; see below
            with self.subTest(case=case):
                self.assertEqual(4, len(per_case[case]))
                computed = self._max_rel_dev(per_case[case])
                self.assertTrue(
                    self._agrees(computed, claimed),
                    f"{case}: recorded runs give {computed:.5f}, table says {claimed}")

    def test_the_smallest_route_floor_recomputes_from_its_isolated_pool(self):
        xs = [self._candidate_ms(p)["decode_m2_square"] for p in self.isolated]
        computed = self._max_rel_dev(xs)
        claimed = QRS.MEASURED_NOISE_FLOOR_BY_MACHINE["L"]["decode_m2_square"]
        self.assertTrue(
            self._agrees(computed, claimed),
            f"isolated pool gives {computed:.5f}, table says {claimed}")

    def test_the_two_gpus_in_that_pool_agree_so_the_spread_is_not_silicon(self):
        # The docstring's reason for calling this within-GPU process variance
        # rather than a difference between devices. If the two disagreed, the
        # floor would be measuring the wrong thing and would not transfer.
        import statistics
        by_gpu = {"g2": [], "g3": []}
        for path in self.isolated:
            for tag in by_gpu:
                if tag in path.name:
                    by_gpu[tag].append(self._candidate_ms(path)["decode_m2_square"])
        self.assertTrue(all(len(v) == 6 for v in by_gpu.values()), by_gpu)
        medians = [statistics.median(v) for v in by_gpu.values()]
        self.assertLess(abs(medians[0] - medians[1]) / medians[1], 0.001)

    def test_every_floor_is_at_least_the_smallest_representable_positive(self):
        # A zero floor would restore the pre-(26) zero-width interval for that
        # route, which is the defect the table exists to prevent. Every machine's
        # table, not just the current one -- CURRENT_MACHINE is a one-line edit.
        for machine, table in sorted(QRS.MEASURED_NOISE_FLOOR_BY_MACHINE.items()):
            for case, value in sorted(table.items()):
                with self.subTest(machine=machine, case=case):
                    self.assertGreater(value, 0.0)


class MachineNNoiseFloorProvenanceTest(unittest.TestCase):
    """(44)'s third corner for the machine-N table.

    Machine L's floors were correct for machine L and wrong for this box in both
    directions -- 2x too wide on `decode_m2_square`, 3.3x too NARROW on
    `prefill_m2048_square`, the route carrying the largest claimed win in the
    ledger. The narrow direction admits noise as an elite, so these eleven
    numbers are load-bearing and are recomputed here from the runs they came
    from rather than trusted to the comment above them.

    Same derivation as machine L's: max relative deviation of `candidate_ms`
    from the median over same-variant full-suite runs, n=8 rather than n=4
    because n=4 could not see the tail on the short routes.

    Skips LOUDLY when the runs are absent -- `exp/` is gitignored.
    """

    RUN_DIR = (Path(__file__).resolve().parents[2]
               / "exp/opt_bf16_20260814/noisefloor_tw035_20260816")
    N_RUNS = 8

    def setUp(self):
        self.runs = [self.RUN_DIR / f"nf_{i}.json" for i in range(1, self.N_RUNS + 1)]
        missing = [p.name for p in self.runs if not p.exists()]
        if missing:
            self.skipTest(
                f"UNCHECKED: {self.RUN_DIR} does not hold the {self.N_RUNS} "
                f"same-variant runs the machine-N floors were derived from "
                f"(missing {missing}). Every value in "
                "MEASURED_NOISE_FLOOR_BY_MACHINE['N'] is unverified in this "
                "environment, and so is every admission interval widened by one.")

    def test_the_eleven_machine_n_floors_recompute_from_the_recorded_runs(self):
        import statistics
        per_case = {}
        for path in self.runs:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for c in payload["test_cases"]:
                per_case.setdefault(c["test_case_id"], []).append(c["candidate_ms"])
        table = QRS.MEASURED_NOISE_FLOOR_BY_MACHINE["N"]
        self.assertEqual(sorted(table), sorted(per_case),
                         "the table and the runs disagree about which routes exist")
        for case, claimed in sorted(table.items()):
            with self.subTest(case=case):
                xs = per_case[case]
                self.assertEqual(self.N_RUNS, len(xs))
                median = statistics.median(xs)
                computed = max(abs(x - median) / median for x in xs)
                self.assertEqual(
                    round(computed, 4), claimed,
                    f"{case}: recorded runs give {computed:.5f}, table says {claimed}")

    def test_the_narrow_direction_that_made_this_table_necessary(self):
        # Pins the specific defect: on these four routes machine L's floor is
        # smaller than machine N's own measured spread, so importing L onto N
        # would let same-variant noise clear the admission gate. If a future
        # edit makes N's floors quietly track L's again, this fails.
        L = QRS.MEASURED_NOISE_FLOOR_BY_MACHINE["L"]
        N = QRS.MEASURED_NOISE_FLOOR_BY_MACHINE["N"]
        for case in ("prefill_m2048_square", "decode_m64_square",
                     "prefill_m256_down", "prefill_m512_up"):
            with self.subTest(case=case):
                self.assertLess(L[case], N[case],
                                f"{case}: machine L's floor is no longer the "
                                "narrower one, so this control proves nothing")

    def test_an_unknown_machine_gets_the_widest_floor_anywhere_not_another_machines(self):
        # Fail-closed: floors do not pool across a machine boundary. The widest
        # value in any table is the only safe answer for an epoch nobody measured.
        widest = max(v for t in QRS.MEASURED_NOISE_FLOOR_BY_MACHINE.values()
                     for v in t.values())
        self.assertEqual(widest, QRS.noise_floor("decode_m2_square", machine="Z"))
        self.assertEqual(widest, QRS.noise_floor("prefill_m256_down", machine="Z"))
        # ... and a route missing from a table it *does* have gets the same.
        self.assertEqual(widest, QRS.noise_floor("no_such_route", machine="N"))

    def test_the_current_machine_is_the_default_and_selects_its_own_table(self):
        expected = QRS.MEASURED_NOISE_FLOOR_BY_MACHINE[QRS.CURRENT_MACHINE]
        for case, value in sorted(expected.items()):
            with self.subTest(case=case):
                self.assertEqual(value, QRS.noise_floor(case))
                self.assertEqual(value, QRS.noise_floor(case,
                                                        machine=QRS.CURRENT_MACHINE))


class CliTest(unittest.TestCase):
    def test_cli_is_deterministic_and_matches_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.json"
            path.write_text(json.dumps({"case_a": [1.0, 2.0, 3.0], "case_b": [4.0, 4.0]}), encoding="utf-8")
            run1 = subprocess.run([sys.executable, str(SCRIPT), str(path)], text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            run2 = subprocess.run([sys.executable, str(SCRIPT), str(path)], text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            self.assertEqual(run1.stdout, run2.stdout)
            payload = json.loads(run1.stdout)
            self.assertEqual("geak.qd-robust-stats/v1", payload["schema"])
            self.assertEqual(2, len(payload["per_context"]))
            self.assertIn("combined", payload)

    def test_the_same_data_in_a_different_key_order_emits_the_same_bytes(self):
        """`context_robust_stats`'s documented sort, carried through to bytes.

        The library-level version of this is already covered; what was not is
        that the CLI preserves it, which is the form a receipt is actually
        compared in. Removing the `sorted()` fails this alongside the two
        library tests, so it is not a restatement that always passes.
        """
        data = {"case_a": [1.0, 2.0, 3.0], "case_b": [4.0, 4.0], "case_c": [9.0]}
        outs = []
        for order in (["case_a", "case_b", "case_c"], ["case_c", "case_a", "case_b"]):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "samples.json"
                path.write_text(json.dumps({k: data[k] for k in order}), encoding="utf-8")
                outs.append(subprocess.run(
                    [sys.executable, str(SCRIPT), str(path)], text=True,
                    stdout=subprocess.PIPE, check=True).stdout)
        self.assertEqual(outs[0], outs[1],
                         "the same data in a different order produced different bytes, so "
                         "two identical receipts will not compare equal")

    def test_the_output_is_written_in_compact_canonical_form(self):
        """The survivor the mutation sweep found: `sort_keys=True -> False`.

        `test_cli_is_deterministic_and_matches_library` re-runs one input and
        compares, which passes whether or not the output is canonical -- CPython
        dicts preserve insertion order deterministically, so a non-canonical
        writer is exactly as repeatable. Six CLIs in this directory emit
        `sort_keys=True, separators=(",", ":")` so their receipts can be diffed
        and digested against each other; nothing was checking that any of them
        still did.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.json"
            path.write_text(json.dumps({"b": [1.0, 2.0], "a": [3.0]}), encoding="utf-8")
            out = subprocess.run([sys.executable, str(SCRIPT), str(path)], text=True,
                                 stdout=subprocess.PIPE, check=True).stdout
        payload = json.loads(out)
        self.assertEqual(out, json.dumps(payload, sort_keys=True,
                                         separators=(",", ":")) + "\n",
                         "the receipt is not in the canonical form the other CLIs in this "
                         "directory emit, so digests of it are not comparable across tools")


if __name__ == "__main__":
    unittest.main(verbosity=2)
