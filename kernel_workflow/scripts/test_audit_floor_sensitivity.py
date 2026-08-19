#!/usr/bin/env python3
"""Tests for the floor-sensitivity audit.

Every test here is written against a defect the audit could plausibly have and
still look like it worked, because that is this tool's specific hazard: it
reports a count, and a count of zero reads as "nothing is wrong" whether the
answer is "nothing is wrong" or "I could not open the file".
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_floor_sensitivity as A  # noqa: E402
AFS = A
import qd_robust_stats as QRS  # noqa: E402

SCRIPT = Path(__file__).with_name("audit_floor_sensitivity.py")
ARCHIVE = (Path(__file__).resolve().parents[2] / "exp"
           / "qd_v2_bf16_smoke_20260816b_tw054" / "qd_archive" / "manifest.json")


def cell(elite_id: str, context: str, baseline_ms: float, samples: list[float]) -> dict:
    return {"elite_id": elite_id, "per_case": [
        {"name": context, "baseline_ms": baseline_ms, "verify_samples_ms": samples}]}


class ReadingTest(unittest.TestCase):
    def test_it_reads_the_per_case_shape(self):
        m = {"cells": {"c1": cell("e1", "decode_m2_square", 1.0, [0.5, 0.5, 0.5])}}
        self.assertEqual(1, len(list(A.iter_cases(m))))

    def test_it_reads_the_nested_elite_shape_too(self):
        # The format changed mid-project; an audit that knows one shape reports a
        # clean bill of health for archives written in the other.
        m = {"cells": {"c1": {"elite": cell("e1", "decode_m2_square", 1.0, [0.5])}}}
        rows = list(A.iter_cases(m))
        self.assertEqual(["e1"], [r["elite_id"] for r in rows])

    def test_a_shape_this_does_not_know_is_reported_as_unread_not_as_clean(self):
        # THE failure mode. Zero flips out of zero cases is not evidence.
        report = A.audit({"cells": {"c1": {"some_future_shape": []}}},
                         {"N": QRS.MEASURED_NOISE_FLOOR_BY_MACHINE["N"]}, "N")
        self.assertEqual(0, report["cases_read"])
        self.assertEqual([], report["flips"])

    def test_a_case_with_no_baseline_is_counted_not_dropped(self):
        # Speedups need a baseline. Silently skipping such cases would shrink the
        # denominator and make the archive look better audited than it was.
        m = {"cells": {"c1": {"elite_id": "e1", "per_case": [
            {"name": "decode_m2_square", "verify_samples_ms": [0.5, 0.5, 0.5]}]}}}
        report = A.audit(m, {"N": QRS.MEASURED_NOISE_FLOOR_BY_MACHINE["N"]}, "N")
        self.assertEqual(0, report["cases_read"])
        self.assertEqual(1, report["cases_without_baseline"])


class ArithmeticTest(unittest.TestCase):
    def test_the_spread_survives_the_conversion_to_speedups(self):
        # Dividing the two MEDIANS instead of each sample would collapse the
        # spread to zero and make every interval clear.
        sp = A.speedups({"baseline_ms": 1.0, "samples": [0.5, 0.4, 0.6]})
        self.assertEqual([2.0, 2.5, 1.0 / 0.6], sp)
        self.assertGreater(A.clears(sp, 0.0)["mad"], 0.0)

    def test_a_wider_floor_can_only_take_a_clear_away_never_grant_one(self):
        sp = [1.05, 1.06, 1.04]
        self.assertTrue(A.clears(sp, 0.01)["clears"])
        self.assertFalse(A.clears(sp, 0.10)["clears"])

    def test_the_radius_is_the_same_rule_admission_uses(self):
        # If this drifts from qd_robust_stats the audit is measuring its own
        # invention rather than the gate that actually ran. Checked against the
        # real function on every route -- restating the formula here would pass
        # no matter what either side does.
        for context in QRS.MEASURED_NOISE_FLOOR:
            for sp in ([1.05, 1.06, 1.04],      # floor-dominated
                       [1.05, 1.40, 0.70],      # MAD-dominated
                       [1.05, 1.05, 1.05]):     # zero MAD
                with self.subTest(context=context, samples=sp):
                    theirs = QRS.robust_stats(sp, context=context)
                    mine = A.clears(sp, QRS.noise_floor(context))
                    self.assertAlmostEqual(theirs["bound_radius"], mine["radius"])
                    self.assertAlmostEqual(theirs["median"], mine["median"])
                    self.assertAlmostEqual(theirs["mad"], mine["mad"])
                    self.assertAlmostEqual(theirs["lower"], mine["lower"])

    def test_an_unmeasured_route_falls_back_to_the_widest_floor(self):
        m = {"cells": {"c1": cell("e1", "no_such_route", 1.0, [0.5, 0.5, 0.5])}}
        report = A.audit(m, {"N": QRS.MEASURED_NOISE_FLOOR_BY_MACHINE["N"]}, "N")
        self.assertEqual(QRS.DEFAULT_NOISE_FLOOR,
                         report["rows"][0]["verdicts"]["N"]["floor"])


class SensitivityTest(unittest.TestCase):
    def test_a_flip_is_reported_with_both_verdicts_visible(self):
        # 1.02 median, zero MAD: clears a 1% floor, fails a 5% one.
        m = {"cells": {"c1": cell("e1", "decode_m2_square", 1.02, [1.0, 1.0, 1.0])}}
        tables = {"a": {"decode_m2_square": 0.01}, "b": {"decode_m2_square": 0.05}}
        report = A.audit(m, tables, "a")
        self.assertEqual(1, len(report["flips"]))
        self.assertEqual({"a": 1, "b": 0}, report["clears_by_table"])
        self.assertTrue(report["flips"][0]["verdicts"]["a"]["clears"])
        self.assertFalse(report["flips"][0]["verdicts"]["b"]["clears"])

    def test_agreement_is_not_a_flip(self):
        m = {"cells": {"c1": cell("e1", "decode_m2_square", 2.0, [1.0, 1.0, 1.0])}}
        tables = {"a": {"decode_m2_square": 0.01}, "b": {"decode_m2_square": 0.05}}
        self.assertEqual([], A.audit(m, tables, "a")["flips"])

    def test_one_measurement_in_two_cells_counts_once(self):
        # Same route, same baseline, same samples, two elite ids: one variant
        # that won two cells, not two pieces of evidence.
        m = {"cells": {
            "c1": cell("e1", "decode_m2_square", 1.02, [1.0, 1.0, 1.0]),
            "c2": cell("e2", "decode_m2_square", 1.02, [1.0, 1.0, 1.0])}}
        tables = {"a": {"decode_m2_square": 0.01}, "b": {"decode_m2_square": 0.05}}
        report = A.audit(m, tables, "a")
        self.assertEqual(2, report["cases_read"])
        self.assertEqual(1, report["distinct_measurements"])
        self.assertEqual({"a": 1, "b": 0}, report["clears_by_table"])
        self.assertEqual({"a": 2, "b": 0}, report["clears_by_table_cases"])
        self.assertEqual(1, len(report["flips"]))
        self.assertEqual(["e1", "e2"], report["flips"][0]["elite_ids"])

    def test_differing_samples_are_two_measurements_not_one(self):
        # The dedup must not collapse genuinely distinct evidence: if it keyed
        # on route alone, every archive would shrink to eleven measurements.
        m = {"cells": {
            "c1": cell("e1", "decode_m2_square", 1.02, [1.0, 1.0, 1.0]),
            "c2": cell("e2", "decode_m2_square", 1.02, [1.0, 1.0, 1.1])}}
        report = A.audit(m, {"a": {"decode_m2_square": 0.01}}, "a")
        self.assertEqual(2, report["distinct_measurements"])

    def test_scaling_a_table_scales_every_route(self):
        scaled = A.scaled(QRS.MEASURED_NOISE_FLOOR_BY_MACHINE["N"], 3.0)
        self.assertEqual(sorted(scaled), sorted(QRS.MEASURED_NOISE_FLOOR_BY_MACHINE["N"]))
        for k, v in QRS.MEASURED_NOISE_FLOOR_BY_MACHINE["N"].items():
            self.assertAlmostEqual(3.0 * v, scaled[k])


class RealArchiveTest(unittest.TestCase):
    """The finding this tool was written to record, pinned to the archive itself.

    `exp/qd_v2_bf16_smoke_20260816b_tw054` is the only archive in the tree with
    admitted mutations carrying raw samples; the other two are seed-only (all
    eleven cells at 1.0), which is why the sensitivity result rests on one run.
    """

    def setUp(self):
        if not ARCHIVE.is_file():
            self.skipTest(f"archive absent: {ARCHIVE} -- the sensitivity result in "
                          "PIPELINE_PROGRESS.md cannot be recomputed without it")
        self.manifest = json.loads(ARCHIVE.read_text(encoding="utf-8"))

    def test_the_132_cases_are_only_22_measurements(self):
        # The correction that matters more than the sensitivity result itself.
        # Two variants x eleven routes, each variant's suite copied into every
        # cell it won: 4 cells for one, 8 for the other. Quoting 132 would
        # inflate the evidence base 6x.
        report = A.audit(
            self.manifest, {"L": QRS.MEASURED_NOISE_FLOOR_BY_MACHINE["L"]}, "L")
        self.assertEqual(132, report["cases_read"])
        self.assertEqual(22, report["distinct_measurements"])
        self.assertEqual(sorted([4, 8] * 11),
                         sorted(r["replicated_across_cells"] for r in report["rows"]))

    def test_one_of_22_measurements_depended_on_the_floor_being_right(self):
        # The blast radius of (107) on real content. A 3x floor error is the
        # size of the error actually found on `prefill_m2048_square`.
        report = A.audit(self.manifest, {
            "L": QRS.MEASURED_NOISE_FLOOR_BY_MACHINE["L"],
            "Lx3": A.scaled(QRS.MEASURED_NOISE_FLOOR_BY_MACHINE["L"], 3.0)}, "L")
        self.assertEqual(1, len(report["flips"]))
        self.assertEqual({"L": 12, "Lx3": 11}, report["clears_by_table"])
        # The case-level view is kept, and is exactly the inflated number.
        self.assertEqual({"L": 80, "Lx3": 76}, report["clears_by_table_cases"])

    def test_the_one_flip_is_the_known_bad_number_on_one_route(self):
        # It is r1_s1's 1.0091 on `decode_m64_square` -- the exact admission
        # finding (95) had already identified as the archive's worst, reached
        # from a completely independent direction. The floor correction and the
        # seed-interval fix converge on one entry, which is why the corrected
        # floor changes no other conclusion in this ledger. It occupies four
        # cells, so a case-level count reports it four times.
        report = A.audit(self.manifest, {
            "L": QRS.MEASURED_NOISE_FLOOR_BY_MACHINE["L"],
            "N": QRS.MEASURED_NOISE_FLOOR_BY_MACHINE["N"]}, "L")
        self.assertEqual(1, len(report["flips"]))
        flip = report["flips"][0]
        self.assertEqual("decode_m64_square", flip["context"])
        self.assertEqual(4, flip["replicated_across_cells"])
        self.assertAlmostEqual(1.0091, flip["verdicts"]["L"]["median"], places=3)
        self.assertTrue(flip["verdicts"]["L"]["clears"])
        self.assertFalse(flip["verdicts"]["N"]["clears"])


class CliTest(unittest.TestCase):
    def run_cli(self, *args: str, check: bool = True):
        return subprocess.run([sys.executable, str(SCRIPT), *args], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)

    def test_an_unknown_machine_is_an_error_not_a_silent_default(self):
        with self.subTest("exit code"):
            run = self.run_cli(str(ARCHIVE), "--machine", "Z", check=False)
            self.assertEqual(2, run.returncode)
            self.assertIn("unknown machine", run.stderr)

    def test_the_report_is_canonical_json(self):
        if not ARCHIVE.is_file():
            self.skipTest("archive absent")
        out = self.run_cli(str(ARCHIVE), "--machine", "L", "--scale", "3").stdout
        payload = json.loads(out)
        self.assertEqual(A.SCHEMA, payload["schema"])
        self.assertNotIn("rows", payload)
        self.assertEqual(json.dumps(payload, sort_keys=True, separators=(",", ":")),
                         out.strip())


if __name__ == "__main__":
    unittest.main()


class V2ManifestTest(unittest.TestCase):
    """Three defects found by pointing this audit at round 16's real archive.

    All three produced a confident-looking report. That is the pattern worth
    testing for: none of them raised, none of them printed a warning, and two of
    them returned a specific number that happened to be fabricated.
    """

    FLOOR = {"O": QRS.MEASURED_NOISE_FLOOR_BY_MACHINE["O"]}

    def test_the_v2_samples_ms_key_is_read(self):
        # The v2 manifest renamed verify_samples_ms -> samples_ms. The audit read
        # zero cases out of an 11-cell archive and printed `"cases_read":0,
        # "flips":[]`, which is the exact JSON it prints for an archive where
        # nothing depended on the floor.
        m = {"cells": {"c1": {"elite_id": "e1", "per_case": [
            {"name": "decode_m2_square", "baseline_ms": 1.0, "optimized_ms": 0.5,
             "samples_ms": [0.5, 0.5, 0.5]}]}}}
        self.assertEqual(1, len(list(A.iter_cases(m))))

    def test_paired_speedup_samples_are_preferred_over_reconstruction(self):
        # The pipeline already pairs rep i's baseline with rep i's candidate.
        # Reconstructing the ratio from a single scalar baseline throws that
        # pairing away even when the scalar is correct.
        case = {"samples": [0.5, 0.4], "baseline_ms": 1.0,
                "speedup_samples": [1.9, 2.6]}
        self.assertEqual([1.9, 2.6], A.speedups(case))

    def test_a_baseline_equal_to_the_candidate_is_refused_not_divided_by(self):
        # Round 16's manifest fills baseline_ms with the CANDIDATE latency --
        # identical to optimized_ms and latency_ms. Dividing by it gives a
        # per-sample ratio of ~1.0 for every case in the archive, so every
        # interval straddles 1.0 and the audit reports that zero admissions clear
        # under every floor. Uniform, confident, and entirely an artifact of the
        # wrong divisor.
        case = {"samples": [0.026, 0.0262], "baseline_ms": 0.02598,
                "optimized_ms": 0.02598}
        self.assertIsNone(A.speedups(case))

    def test_that_wrong_baseline_is_counted_as_unusable(self):
        m = {"cells": {"c1": {"elite_id": "e1", "per_case": [
            {"name": "decode_m2_square", "baseline_ms": 0.026,
             "optimized_ms": 0.026, "samples_ms": [0.026, 0.0262]}]}}}
        report = A.audit(m, self.FLOOR, "O")
        self.assertEqual(0, report["cases_read"])
        self.assertEqual(1, report["cases_without_baseline"])

    def test_a_populated_archive_it_cannot_read_exits_nonzero(self):
        # The library reports cases_read: 0 and the CLI has to turn that into a
        # refusal. A count a reader must notice is not a defence -- this is the
        # check that would have caught the samples_ms rename at the moment it
        # mattered instead of one round later.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "manifest.json"
            p.write_text(json.dumps({"cells": {"c1": {"some_future_shape": []}}}))
            proc = subprocess.run([sys.executable, str(SCRIPT), "--machine", "O", str(p)],
                                  capture_output=True, text=True, timeout=120)
        self.assertEqual(4, proc.returncode, proc.stdout + proc.stderr)
        self.assertIn("UNREAD", proc.stderr)
        self.assertIn("nothing was examined", proc.stderr)

    def test_an_empty_archive_is_not_an_error(self):
        # No cells at all is a real, readable state -- a fresh archive. Only
        # cells-but-unreadable is the fault.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "manifest.json"
            p.write_text(json.dumps({"cells": {}}))
            proc = subprocess.run([sys.executable, str(SCRIPT), "--machine", "O", str(p)],
                                  capture_output=True, text=True, timeout=120)
        self.assertEqual(0, proc.returncode, proc.stderr)


class Round16ArchiveTest(unittest.TestCase):
    """Against round 16's archive on disk, if it is still there."""

    MANIFEST = (Path(__file__).resolve().parents[2] / "exp"
                / "qd_v2_bf16_r16_20260816_epochO" / "qd_archive" / "manifest.json")

    def setUp(self):
        if not self.MANIFEST.is_file():
            self.skipTest("round 16 archive not present")
        self.report = A.audit(json.loads(self.MANIFEST.read_text()),
                              {"O": QRS.MEASURED_NOISE_FLOOR_BY_MACHINE["O"]}, "O")

    def test_eleven_cells_of_one_seed_are_eleven_measurements_not_121(self):
        # One seed elite won all 11 cells, so its 11-case suite is filed 11
        # times. Counting the copies would inflate the evidence base 11x.
        self.assertEqual(121, self.report["cases_read"])
        self.assertEqual(11, self.report["distinct_measurements"])

    def test_the_seed_suite_is_readable_at_all(self):
        self.assertEqual(0, self.report["cases_without_baseline"])
        self.assertGreater(self.report["clears_by_table"]["O"], 0,
                           "no case clears 1.0 under its own floor, which is what "
                           "the wrong-divisor bug looked like")


class SpeedupProvenanceTest(unittest.TestCase):
    """"8 of 11 clear" is unreadable until you know what the ratio divided by."""

    FLOOR = {"O": QRS.MEASURED_NOISE_FLOOR_BY_MACHINE["O"]}

    def test_the_report_says_where_each_ratio_came_from(self):
        m = {"cells": {"c1": {"elite_id": "e1", "per_case": [
            {"name": "decode_m2_square", "baseline_ms": 0.027, "optimized_ms": 0.026,
             "samples_ms": [0.026, 0.0262], "speedup_samples": [1.04, 1.03]},
            {"name": "decode_m8_up", "baseline_ms": 0.06, "optimized_ms": 0.055,
             "samples_ms": [0.055, 0.056]}]}}}
        report = A.audit(m, self.FLOOR, "O")
        self.assertEqual({"paired_speedup_samples": 1,
                          "reconstructed_from_baseline_ms": 1},
                         report["speedup_sources"])

    def test_the_round_16_seed_suite_is_entirely_oracle_ratios(self):
        # Not the parent comparison. Every seed row carries speedup: 1 by
        # definition -- the seed is its own parent -- while speedup_samples holds
        # the ratio against the frozen rocBLAS oracle. The audit reads the
        # latter, so its verdict answers "does the seed beat the oracle" and NOT
        # "did this admission depend on the floor". At generation 0 the second
        # question has no content: seeds are admitted by definition, at
        # robust.median 1, and no floor changes that.
        manifest = (Path(__file__).resolve().parents[2] / "exp"
                    / "qd_v2_bf16_r16_20260816_epochO" / "qd_archive" / "manifest.json")
        if not manifest.is_file():
            self.skipTest("round 16 archive not present")
        report = A.audit(json.loads(manifest.read_text()), self.FLOOR, "O")
        self.assertEqual({"paired_speedup_samples": 121}, report["speedup_sources"])


class SyntheticAggregateTest(unittest.TestCase):
    """A pseudo-case whose "samples" are already an aggregate must not be
    counted as a route.

    Round 1's engineer emitted `runs/speedup_samples.json` with twelve keys for
    eleven routes: the extra one, `__suite_geomean__`, is the per-rep geomean of
    the other eleven. It never reached the manifest -- but if it had, it would
    have been counted as a twelfth measurement whose spread is the spread of an
    average, i.e. much tighter than any route, and the headline would have read
    "12 of 12 clear" off eleven routes' worth of evidence.
    """

    FLOOR = {"O": {"decode_m2_square": 0.02, "prefill_m2048_square": 0.02}}

    def _manifest(self, names):
        return {"cells": {f"c{i}": {"elite_id": f"e{i}", "per_case": [
            {"name": n, "samples_ms": [0.026, 0.0261, 0.0259],
             "speedup_samples": [1.30, 1.31, 1.29]} for n in names]}
            for i in range(1)}}

    def test_a_dunder_aggregate_is_not_counted_as_a_route(self):
        m = self._manifest(["decode_m2_square", "prefill_m2048_square",
                            "__suite_geomean__"])
        report = A.audit(m, self.FLOOR, "O")
        self.assertEqual(2, report["cases_read"])
        self.assertEqual({"decode_m2_square", "prefill_m2048_square"},
                         {r["context"] for r in report["rows"]})

    def test_the_real_routes_are_still_read(self):
        """The skip must be keyed on the name, not on anything the aggregate
        happens to share with a route -- otherwise it silences real cases too."""
        m = self._manifest(["decode_m2_square", "prefill_m2048_square"])
        self.assertEqual(2, A.audit(m, self.FLOOR, "O")["cases_read"])

    def test_an_archive_of_nothing_but_aggregates_is_reported_unread(self):
        """Exit 4, not a clean bill of health: the manifest has a cell and the
        audit examined none of it."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "manifest.json"
            p.write_text(json.dumps(self._manifest(["__suite_geomean__"])))
            proc = subprocess.run(
                [sys.executable, str(A.__file__), str(p), "--machine", "O"],
                capture_output=True, text=True, timeout=120)
        self.assertEqual(4, proc.returncode, proc.stdout + proc.stderr)
        self.assertIn("UNREAD", proc.stderr)


class DeclaredUnitTest(unittest.TestCase):
    """A samples array in the wrong unit does not produce a wrong-looking answer.

    Read as ms, seconds give `baseline_ms / 0.000027` -- a ~1000x speedup, an
    interval far above 1.0, and a confident "every admission survives every
    floor". The fabrication is indistinguishable from a triumph, which is the
    property that makes it worth a refusal rather than a warning. It is also the
    only finding in this module that is about the input file rather than about
    history, and therefore the only one that may set a non-zero exit.
    """

    @staticmethod
    def _manifest(case):
        return {"cells": {"c0": {"elite_id": "e0", "per_case": [
            {"name": "decode_m2_square", **case}]}}}

    def test_a_key_suffix_declares_the_unit(self):
        self.assertEqual("ms", AFS.declared_unit({}, "verify_samples_ms"))
        self.assertEqual("ms", AFS.declared_unit({}, "samples_ms"))
        self.assertIsNone(AFS.declared_unit({}, "samples"),
                          "a bare `samples` declares nothing; silence is not a claim of ms")

    def test_an_explicit_field_outranks_the_suffix(self):
        self.assertEqual("s", AFS.declared_unit({"samples_unit": " S "}, "samples_ms"))

    def test_an_unrecognised_unit_is_not_a_declaration(self):
        """`None` and not a refusal. "I have never heard of this unit" is a
        different state from "these numbers cannot be that unit", and only the
        second one is evidence of anything."""
        self.assertIsNone(AFS.declared_unit({"samples_unit": "fortnights"}, "samples"))

    def test_consistent_values_are_not_refused(self):
        self.assertIsNone(AFS.unit_disagrees([0.027, 0.0271, 0.0269], "ms", 0.027))

    def test_seconds_read_as_ms_are_refused_and_the_real_unit_named(self):
        reason = AFS.unit_disagrees([0.000027, 0.000027, 0.0000271], "ms", 0.027)
        self.assertIsNotNone(reason)
        self.assertIn("'s'", reason, "the refusal should say what the values COULD be")

    def test_a_correctly_declared_second_is_converted_not_refused(self):
        self.assertIsNone(AFS.unit_disagrees([0.000027, 0.000027], "s", 0.027))
        cases = list(AFS.iter_cases(self._manifest(
            {"samples": [0.000027, 0.000027, 0.0000271], "samples_unit": "s",
             "optimized_ms": 0.027, "baseline_ms": 0.054})))
        self.assertEqual(1, len(cases))
        self.assertAlmostEqual(0.027, cases[0]["samples"][0], places=9,
                               msg="a declared, consistent unit must be converted to ms, "
                                   "or every statistic downstream is off by that factor")

    def test_no_second_number_means_no_verdict(self):
        """Refusing on an unverifiable claim would throw out every legacy case
        rather than the broken ones."""
        self.assertIsNone(AFS.unit_disagrees([0.027, 0.0271], "ms", None))
        self.assertIsNone(AFS.unit_disagrees([0.027, 0.0271], "ms", 0))

    def test_the_tolerance_admits_an_ordinary_spread(self):
        """`optimized_ms` is one summary of a run and the samples are its
        repeats; a factor of two apart is ordinary and must not be a refusal."""
        self.assertIsNone(AFS.unit_disagrees([0.05, 0.05], "ms", 0.027))
        self.assertIsNotNone(AFS.unit_disagrees([2.7, 2.7], "ms", 0.027))

    def test_a_bad_unit_case_is_excluded_from_every_count(self):
        report = AFS.audit(self._manifest(
            {"samples_ms": [0.000027, 0.000027, 0.0000271],
             "optimized_ms": 0.027, "baseline_ms": 0.054}),
            {"P": AFS.robust.MEASURED_NOISE_FLOOR_BY_MACHINE["P"]}, "P")
        self.assertEqual(1, report["cases_rejected_bad_unit"])
        self.assertEqual(0, report["cases_read"],
                         "a rejected case must not also be counted as read; the two "
                         "numbers together are what tell a reader the file is broken")
        self.assertEqual(0, report["clears_by_table"]["P"],
                         "the ~1000x speedup must never reach a verdict")

    def test_the_cli_exits_five_and_says_why(self):
        import io
        import contextlib
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(self._manifest(
                {"samples_ms": [0.000027, 0.000027, 0.0000271],
                 "optimized_ms": 0.027, "baseline_ms": 0.054})), encoding="utf-8")
            err, out = io.StringIO(), io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
                code = AFS.main([str(path), "--machine", "P"])
        self.assertEqual(5, code)
        self.assertIn("BAD UNIT", err.getvalue())
        self.assertIn("decode_m2_square", err.getvalue())
        self.assertNotIn("UNREAD", err.getvalue(),
                         "a manifest refused on units must not ALSO be reported as unread; "
                         "that is true and buries the actionable half")
        self.assertEqual(1, json.loads(out.getvalue())["cases_rejected_bad_unit"])
