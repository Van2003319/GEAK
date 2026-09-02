#!/usr/bin/env python3
"""Behavioural tests for the canonical paired statistical route gate."""
from __future__ import annotations

import copy
import json
import math
import os
import random
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import route_gate as rg


def make_spec(gains, targets, *, n=12, session="session-1", arm_log_noise=None):
    """Build complete paired blocks; gain > 0 means lower candidate latency."""
    pair_ids = [f"pair-{i:02d}" for i in range(n)]
    blocks = []
    for route_index, (route, gain) in enumerate(gains.items()):
        control = []
        candidate = []
        for i in range(n):
            common = 1.0 + route_index * 0.2 + 0.003 * math.sin(i + route_index)
            noise = arm_log_noise(route, i) if arm_log_noise else 0.0
            control.append(common)
            candidate.append(common * (1.0 - gain) * math.exp(noise))
        blocks.append({
            "route": route,
            "candidate": {"session_id": session, "pair_ids": list(pair_ids),
                          "samples_ms": candidate},
            "control": {"session_id": session, "pair_ids": list(pair_ids),
                        "samples_ms": control},
        })
    return {
        "schema": rg.INPUT_SCHEMA,
        "session_id": session,
        "expected_routes": list(gains),
        "expected_route_count": len(gains),
        "target_routes": list(targets),
        "paired_routes": blocks,
    }


def invalid(spec):
    receipt = rg.decide_fail_closed(spec)
    if receipt["decision"] != "invalid_evidence":
        raise AssertionError(receipt)
    return receipt


class StatisticalDecisionTest(unittest.TestCase):
    def test_aa_false_positive_rate_is_bounded(self):
        """A/A at MDE=0 exercises the nominal 5% bound instead of hiding behind the 2% MDE."""
        trials = 80
        accepted = 0
        for trial in range(trials):
            rnd = random.Random(10_000 + trial)
            spec = make_spec({"target": 0.0}, ["target"], n=16, session=f"aa-{trial}")
            spec["practical_mde"] = 0.0
            block = spec["paired_routes"][0]
            base = block["control"]["samples_ms"]
            # Independent arm jitter around one common-session level. The gate sees raw pairs,
            # not session medians; nonzero sigma makes this a real false-positive calibration.
            block["control"]["samples_ms"] = [
                v * math.exp(rnd.gauss(0, 0.01)) for v in base]
            block["candidate"]["samples_ms"] = [
                v * math.exp(rnd.gauss(0, 0.01)) for v in base]
            accepted += int(rg.decide(spec)["accepted"])
        self.assertLessEqual(accepted, 8, f"A/A false accepts: {accepted}/{trials}")

    def test_route_estimator_uses_paired_raw_log_ratios_and_trims_each_tail(self):
        logs = [-0.40, -0.02, 0.020, 0.021, 0.022, 0.023,
                0.024, 0.025, 0.026, 0.027, 0.08, 0.50]
        spec = make_spec({"target": 0.0}, ["target"], n=len(logs))
        block = spec["paired_routes"][0]
        block["control"]["samples_ms"] = [1.0] * len(logs)
        block["candidate"]["samples_ms"] = [math.exp(-value) for value in logs]
        rec = rg.decide(spec)
        row = rec["routes"][0]
        self.assertAlmostEqual(row["estimate_log_ratio"], rg._trimmed_mean(logs), places=14)
        self.assertNotAlmostEqual(row["estimate_log_ratio"], sum(logs) / len(logs), places=3)

    def test_suite_estimator_preserves_complete_pair_blocks_and_route_weights(self):
        spec = make_spec({"target": 0.0, "guard": 0.0}, ["target"], n=12)
        target_logs = [0.025 + 0.001 * i for i in range(12)]
        guard_logs = [-0.006 - 0.0002 * i for i in range(12)]
        for block, logs in zip(spec["paired_routes"], (target_logs, guard_logs)):
            block["control"]["samples_ms"] = [1.0] * 12
            block["candidate"]["samples_ms"] = [math.exp(-value) for value in logs]
        spec["route_weights"] = {"target": 1.0, "guard": 3.0}
        rec = rg.decide(spec)
        complete_blocks = [0.25 * target + 0.75 * guard
                           for target, guard in zip(target_logs, guard_logs)]
        self.assertAlmostEqual(rec["suite"]["estimate_log_ratio"],
                               rg._trimmed_mean(complete_blocks), places=14)

    def test_three_percent_injection_passes(self):
        noise = lambda _route, i: ((i % 5) - 2) * 0.0003
        rec = rg.decide(make_spec({"target": 0.03, "guard": 0.0}, ["target"],
                                  arm_log_noise=noise))
        self.assertTrue(rec["accepted"], rec["reason"])
        target = next(r for r in rec["routes"] if r["route"] == "target")
        self.assertGreater(target["ci95_lower_log_ratio"], target["threshold_log_ratio"])

    def test_seven_percent_injection_passes(self):
        rec = rg.decide(make_spec({"target": 0.07, "guard": 0.0}, ["target"]))
        self.assertTrue(rec["accepted"], rec["reason"])
        self.assertEqual(rec["target_passed"], ["target"])

    def test_target_below_default_mde_is_refused(self):
        rec = rg.decide(make_spec({"target": 0.015, "guard": 0.0}, ["target"]))
        self.assertFalse(rec["accepted"])
        self.assertIn("target MDE not proven", rec["reason"])

    def test_configured_mde_is_applied_per_target(self):
        spec = make_spec({"a": 0.03, "b": 0.07, "guard": 0.0}, ["a", "b"])
        spec["practical_mde"] = {"a": 0.04, "b": 0.06}
        rec = rg.decide(spec)
        self.assertFalse(rec["accepted"])
        self.assertEqual(rec["target_passed"], ["b"])

    def test_complete_suite_margin_maps_work_for_different_candidate_targets(self):
        spec = make_spec({"a": 0.03, "b": 0.0, "c": -0.005}, ["a"])
        spec["practical_mde"] = {"a": 0.02, "b": 0.04, "c": 0.06}
        spec["non_regression_margin"] = {"a": 0.01, "b": 0.01, "c": 0.01}
        rec = rg.decide(spec)
        self.assertTrue(rec["accepted"], rec["reason"])
        self.assertEqual(rec["effective_margins"]["practical_mde"], {"a": 0.02})
        self.assertEqual(rec["effective_margins"]["non_regression_margin"],
                         {"b": 0.01, "c": 0.01})

    def test_every_declared_target_must_pass(self):
        rec = rg.decide(make_spec({"a": 0.03, "b": 0.01, "guard": 0.0}, ["a", "b"]))
        self.assertFalse(rec["accepted"])
        self.assertIn("b", rec["reason"])
        rec = rg.decide(make_spec({"a": 0.03, "b": 0.04, "guard": 0.0}, ["a", "b"]))
        self.assertTrue(rec["accepted"], rec["reason"])

    def test_non_target_regression_past_margin_is_refused(self):
        rec = rg.decide(make_spec({"target": 0.07, "guard": -0.03}, ["target"]))
        self.assertFalse(rec["accepted"])
        self.assertEqual(rec["regressed"], ["guard"])
        self.assertIn("non-target non-regression", rec["reason"])

    def test_predeclared_route_specific_non_regression_margins(self):
        spec = make_spec({"target": 0.07, "tight": -0.015, "wide": -0.025}, ["target"])
        spec["non_regression_margin"] = {"tight": 0.01, "wide": 0.03}
        rec = rg.decide(spec)
        self.assertFalse(rec["accepted"])
        self.assertEqual(rec["regressed"], ["tight"])

    def test_suite_is_non_inferiority_not_raw_ratio_greater_than_one(self):
        # The suite estimate is slightly below 1.0, but its lower bound stays above the declared
        # -1% margin. That is a non-inferiority decision, not `suiteRatio > 1`.
        gains = {"target": 0.03, "g1": -0.012, "g2": -0.012, "g3": -0.012}
        rec = rg.decide(make_spec(gains, ["target"]))
        self.assertLess(rec["suite"]["estimate_ratio"], 1.0)
        self.assertTrue(rec["suite"]["passed"])
        self.assertTrue(rec["accepted"], rec["reason"])

    def test_suite_non_inferiority_can_veto_routewise_passes(self):
        gains = {"target": 0.03}
        gains.update({f"guard-{i}": -0.019 for i in range(9)})
        rec = rg.decide(make_spec(gains, ["target"]))
        self.assertTrue(all(r["passed"] for r in rec["routes"]), rec["routes"])
        self.assertFalse(rec["suite"]["passed"])
        self.assertFalse(rec["accepted"])
        self.assertIn("suite non-inferiority", rec["reason"])

    def test_catastrophic_fence_survives_a_large_target_win(self):
        rec = rg.decide(make_spec({"target": 0.40, "guard": -0.12}, ["target"]))
        self.assertFalse(rec["accepted"])
        self.assertEqual(rec["catastrophic"], ["guard"])
        self.assertIn("catastrophic", rec["reason"])

    def test_fixed_seed_and_route_order_make_receipt_deterministic(self):
        spec = make_spec({"b": 0.0, "target": 0.03, "a": 0.0}, ["target"],
                         arm_log_noise=lambda route, i: ((i + len(route)) % 5 - 2) * 0.0005)
        first = rg.decide(spec)
        second = rg.decide(copy.deepcopy(spec))
        shuffled = copy.deepcopy(spec)
        shuffled["expected_routes"].reverse()
        shuffled["paired_routes"].reverse()
        third = rg.decide(shuffled)
        self.assertEqual(first, second)
        # Input digest records representation order, while every decision/statistical field is stable.
        first.pop("input_sha256")
        third.pop("input_sha256")
        self.assertEqual(first, third)
        self.assertEqual(first["algorithm"]["seed"], rg.BOOTSTRAP_SEED)


class FailClosedEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.spec = make_spec({"target": 0.07, "guard": 0.0}, ["target"])

    def test_missing_top_level_session_is_invalid(self):
        self.spec.pop("session_id")
        self.assertIn("session_id is missing", invalid(self.spec)["reason"])

    def test_cross_session_arm_is_invalid(self):
        self.spec["paired_routes"][0]["control"]["session_id"] = "other-session"
        self.assertIn("does not match gate session", invalid(self.spec)["reason"])

    def test_missing_control_raw_block_is_invalid(self):
        self.spec["paired_routes"][0].pop("control")
        self.assertIn("control raw block is missing", invalid(self.spec)["reason"])

    def test_missing_candidate_samples_is_invalid(self):
        self.spec["paired_routes"][0]["candidate"].pop("samples_ms")
        self.assertIn("samples_ms", invalid(self.spec)["reason"])

    def test_too_few_pairs_is_invalid(self):
        for block in self.spec["paired_routes"]:
            for arm in (block["candidate"], block["control"]):
                arm["pair_ids"] = arm["pair_ids"][:rg.MIN_PAIRS - 1]
                arm["samples_ms"] = arm["samples_ms"][:rg.MIN_PAIRS - 1]
        self.assertIn("need at least", invalid(self.spec)["reason"])

    def test_mismatched_pair_ids_are_invalid(self):
        self.spec["paired_routes"][0]["candidate"]["pair_ids"][0] = "unpaired"
        self.assertIn("pair ids differ", invalid(self.spec)["reason"])

    def test_routes_must_share_complete_pair_blocks(self):
        block = self.spec["paired_routes"][1]
        for arm in (block["candidate"], block["control"]):
            arm["pair_ids"][0] = "different-block"
        self.assertIn("same paired blocks", invalid(self.spec)["reason"])

    def test_partial_single_route_table_cannot_pass_a_full_suite(self):
        self.spec["expected_routes"] = ["target", "guard", "third"]
        self.spec["expected_route_count"] = 3
        rec = invalid(self.spec)
        self.assertIn("complete declared suite", rec["reason"])
        self.assertIn("third", rec["reason"])

    def test_extra_route_is_invalid_too(self):
        self.spec["expected_routes"] = ["target"]
        self.spec["expected_route_count"] = 1
        self.assertIn("extra", invalid(self.spec)["reason"])

    def test_declared_suite_count_rejects_a_partial_baseline_manifest(self):
        self.spec["expected_routes"] = ["target"]
        self.spec["expected_route_count"] = 2
        rec = invalid(self.spec)
        self.assertIn("partial baseline table", rec["reason"])

    def test_missing_or_empty_targets_are_invalid(self):
        for value in (None, []):
            spec = copy.deepcopy(self.spec)
            if value is None:
                spec.pop("target_routes")
            else:
                spec["target_routes"] = value
            self.assertIn("target_routes", invalid(spec)["reason"])

    def test_extreme_finite_samples_fail_closed_without_overflowing_json(self):
        self.spec["paired_routes"][0]["candidate"]["samples_ms"] = [5e-324] * 12
        self.spec["paired_routes"][0]["control"]["samples_ms"] = [1e308] * 12
        rec = rg.decide_fail_closed(self.spec)
        self.assertEqual(rec["decision"], "invalid_evidence")
        self.assertIn("finite receipt range", rec["reason"])
        json.dumps(rec, allow_nan=False)

    def test_identifiers_with_surrounding_whitespace_are_invalid_not_normalized(self):
        mutations = (
            lambda spec: spec.update(session_id=" session-1"),
            lambda spec: spec["expected_routes"].__setitem__(0, " target"),
            lambda spec: spec["paired_routes"][0].__setitem__("route", "target "),
            lambda spec: spec["paired_routes"][0]["candidate"]["pair_ids"].__setitem__(0, " pair-00"),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                spec = copy.deepcopy(self.spec)
                mutate(spec)
                self.assertIn("whitespace", invalid(spec)["reason"])

    def test_complete_margin_map_validates_inactive_entries_too(self):
        self.spec["practical_mde"] = {"target": 0.02, "guard": "bad"}
        self.assertIn("practical_mde", invalid(self.spec)["reason"])

    def test_route_weight_total_must_remain_finite(self):
        self.spec["route_weights"] = {"target": 1e308, "guard": 1e308}
        self.assertIn("total must be finite", invalid(self.spec)["reason"])

    def test_nonfinite_or_nonpositive_samples_are_invalid(self):
        for bad in (0.0, -1.0, float("inf"), float("nan"), True):
            with self.subTest(bad=bad):
                spec = copy.deepcopy(self.spec)
                spec["paired_routes"][0]["candidate"]["samples_ms"][0] = bad
                self.assertEqual(rg.decide_fail_closed(spec)["decision"], "invalid_evidence")

    def test_v1_summary_shape_is_explicitly_not_live_evidence(self):
        legacy = {"candidate_per_case": [{"name": "target", "optimized_ms": 0.9}],
                  "incumbent_per_case": [{"name": "target", "optimized_ms": 1.0}],
                  "bands": {"target": 0.01}}
        rec = rg.decide_fail_closed(legacy)
        self.assertEqual(rec["decision"], "invalid_evidence")
        self.assertIn(rg.INPUT_SCHEMA, rec["reason"])


class ReceiptAndCliTest(unittest.TestCase):
    def test_receipt_is_versioned_and_describes_the_actual_test(self):
        rec = rg.decide(make_spec({"target": 0.03, "guard": 0.0}, ["target"]))
        self.assertEqual(rec["schema"], rg.RECEIPT_SCHEMA)
        self.assertEqual(rec["input_schema"], rg.INPUT_SCHEMA)
        self.assertEqual(rec["algorithm"]["name"], rg.ALGORITHM)
        self.assertEqual(rec["algorithm"]["confidence"], 0.95)
        self.assertEqual(rec["algorithm"]["seed"], 20260902)
        self.assertEqual(rec["algorithm"]["resamples"], 4096)
        self.assertIn("paired", rec["algorithm"]["interval"])
        self.assertIn("type 7", rec["algorithm"]["quantile"])
        self.assertIn("SplitMix64", rec["algorithm"]["rng"])
        self.assertEqual(rec["pair_ids"], [f"pair-{i:02d}" for i in range(12)])
        self.assertEqual(rec["suite"]["bootstrap_stream_seed"], "0x9c4e910031dad061")
        target = next(row for row in rec["routes"] if row["route"] == "target")
        self.assertEqual(target["bootstrap_stream_seed"], "0xf3d1182ae70f9801")
        self.assertTrue(all(r["bootstrap_stream_seed"].startswith("0x")
                            for r in rec["routes"]))
        self.assertEqual(rec["effective_margins"]["practical_mde"], {"target": 0.02})
        self.assertEqual(rec["route_weights"], {"guard": 0.5, "target": 0.5})

    def test_cli_reads_stdin_and_exit_code_matches_acceptance(self):
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "route_gate.py")
        spec = make_spec({"target": 0.03, "guard": 0.0}, ["target"])
        proc = subprocess.run([sys.executable, script, "-", "--json"],
                              input=json.dumps(spec), text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rec = json.loads(proc.stdout)
        self.assertTrue(rec["accepted"])
        self.assertEqual(json.dumps(rec, indent=1, sort_keys=True), proc.stdout.strip())

    def test_cli_emits_invalid_receipt_and_refusal_exit(self):
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "route_gate.py")
        proc = subprocess.run([sys.executable, script, "-", "--json"], input="{}",
                              text=True, capture_output=True)
        self.assertEqual(proc.returncode, 3)
        rec = json.loads(proc.stdout)
        self.assertEqual(rec["schema"], rg.RECEIPT_SCHEMA)
        self.assertEqual(rec["decision"], "invalid_evidence")

    def test_cli_malformed_json_still_emits_a_versioned_invalid_receipt(self):
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "route_gate.py")
        proc = subprocess.run([sys.executable, script, "-", "--json"], input="{not-json",
                              text=True, capture_output=True)
        self.assertEqual(proc.returncode, 3)
        rec = json.loads(proc.stdout)
        self.assertEqual(rec["schema"], rg.RECEIPT_SCHEMA)
        self.assertEqual(rec["decision"], "invalid_evidence")

    def test_legacy_spread_helper_is_labelled_and_isolated(self):
        self.assertAlmostEqual(rg.legacy_minmax_median_band([0.9, 1.0, 1.1]), 0.2)
        self.assertIs(rg.bands_from_samples, rg.legacy_bands_from_samples)
        self.assertIs(rg.bands_from_repeats, rg.legacy_bands_from_repeats)
        rec = rg.decide(make_spec({"target": 0.03, "guard": 0.0}, ["target"]))
        self.assertNotIn("band", json.dumps(rec).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
