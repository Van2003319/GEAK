"""Tests for the epoch floor sweep.

The script's whole value is that it refuses. A sweep that emits a table when it
should not is worse than no script at all: the table it emits looks exactly like
a measured one, and every admission for the rest of the epoch inherits it. So
most of what is tested here is the refusals, not the arithmetic.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import measure_noise_floor as M  # noqa: E402
import qd_robust_stats as QRS  # noqa: E402


def rows(speedups: dict[str, float]) -> list[dict[str, object]]:
    return [{"test_case_id": name, "speedup": s} for name, s in speedups.items()]


def clean_repeat(value: float = 1.0) -> list[dict[str, object]]:
    return rows({route: value for route in M.reference_routes()})


class FloorStatisticTest(unittest.TestCase):
    def test_the_statistic_is_two_mad_over_median(self):
        samples = [1.00, 1.02, 0.98, 1.04, 0.96, 1.00, 1.00, 1.00]
        got = M.floor_from_speedups(samples)
        expected = 2.0 * M.mad(samples) / 1.0
        self.assertAlmostEqual(got["floor_raw"], expected)

    def test_it_is_relative_so_it_survives_the_epoch_boundary(self):
        # The same relative spread on a route ten times slower must give the
        # same floor -- that is the whole reason the floor is a ratio.
        slow = M.floor_from_speedups([10.0, 10.2, 9.8, 10.4, 9.6])
        fast = M.floor_from_speedups([1.0, 1.02, 0.98, 1.04, 0.96])
        self.assertAlmostEqual(slow["floor"], fast["floor"], places=9)

    def test_a_zero_mad_is_clamped_up_and_says_so(self):
        got = M.floor_from_speedups([1.0] * 8)
        self.assertEqual(got["floor"], M.MIN_FLOOR)
        self.assertTrue(got["clamped_to_min"])
        self.assertEqual(got["floor_raw"], 0.0)

    def test_a_real_floor_is_not_clamped(self):
        got = M.floor_from_speedups([1.0, 1.05, 0.95, 1.05, 0.95])
        self.assertFalse(got["clamped_to_min"])
        self.assertGreater(got["floor"], M.MIN_FLOOR)

    def test_clamping_can_only_widen(self):
        for samples in ([1.0] * 5, [1.0, 1.0001, 0.9999], [1.0, 1.2, 0.8]):
            got = M.floor_from_speedups(samples)
            self.assertGreaterEqual(got["floor"], got["floor_raw"])


class RouteSetTest(unittest.TestCase):
    def test_the_reference_route_set_is_the_eleven_case_suite(self):
        self.assertEqual(len(M.reference_routes()), 11)

    def test_it_is_read_from_the_reference_epoch_not_hardcoded(self):
        self.assertEqual(M.reference_routes(),
                         frozenset(QRS.MEASURED_NOISE_FLOOR_BY_MACHINE[M.REFERENCE_EPOCH]))

    def test_the_reference_epoch_is_one_that_actually_measured(self):
        self.assertNotIn(M.REFERENCE_EPOCH, QRS.PROVISIONAL_MACHINES)

    def test_a_missing_route_is_a_problem_not_a_hole_in_the_table(self):
        short = [r for r in clean_repeat() if r["test_case_id"] != "decode_m2_square"]
        _, problems = M.collect([short, clean_repeat(), clean_repeat()])
        self.assertTrue(any("missing routes" in p and "decode_m2_square" in p
                            for p in problems))

    def test_an_unknown_route_is_also_refused(self):
        extra = clean_repeat() + [{"test_case_id": "decode_m4096_sideways", "speedup": 1.0}]
        _, problems = M.collect([extra])
        self.assertTrue(any("unknown routes" in p for p in problems))


class UnitErrorTest(unittest.TestCase):
    def test_a_thousandfold_speedup_is_refused_as_a_unit_error(self):
        bad = rows({route: (1000.0 if route == "decode_m2_square" else 1.0)
                    for route in M.reference_routes()})
        by_route, problems = M.collect([bad])
        self.assertTrue(any("unit error" in p for p in problems))
        # and the poisoned sample is not silently kept
        self.assertNotIn("decode_m2_square", by_route)

    def test_a_plausible_speedup_is_kept(self):
        by_route, problems = M.collect([clean_repeat(1.4)])
        self.assertEqual(problems, [])
        self.assertEqual(by_route["decode_m2_square"], [1.4])

    def test_a_nonpositive_speedup_is_refused(self):
        bad = rows({route: (0.0 if route == "decode_m8_up" else 1.0)
                    for route in M.reference_routes()})
        _, problems = M.collect([bad])
        self.assertTrue(any("no positive speedup" in p for p in problems))


class SweepRefusalTest(unittest.TestCase):
    """The sweep's exit code is a claim about whether a table may be written."""

    def _fake(self, monkey: dict) -> dict:
        saved = {k: getattr(M, k) for k in monkey}
        for k, v in monkey.items():
            setattr(M, k, v)
        try:
            return M.sweep(Path("/nonexistent/task"), repeats=4)
        finally:
            for k, v in saved.items():
                setattr(M, k, v)

    def test_a_clean_sweep_is_ok(self):
        verdict = self._fake({
            "run_correctness": lambda *a, **k: True,
            "run_suite": lambda *a, **k: clean_repeat(1.0 + 0.01 * len(a)),
            "QSH": type("H", (), {"tree_hash": staticmethod(lambda p: "same")}),
        })
        self.assertTrue(verdict["ok"], verdict.get("problems"))
        self.assertEqual(verdict["repeats_complete"], 4)
        self.assertEqual(set(verdict["routes"]), set(M.reference_routes()))

    def test_failed_correctness_stops_before_any_timing(self):
        seen = []
        verdict = self._fake({
            "run_correctness": lambda *a, **k: False,
            "run_suite": lambda *a, **k: seen.append(1) or clean_repeat(),
            "QSH": type("H", (), {"tree_hash": staticmethod(lambda p: "same")}),
        })
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["stage"], "correctness")
        self.assertEqual(seen, [], "timed a variant whose correctness failed")

    def test_a_moving_source_hash_voids_the_sweep(self):
        hashes = iter(["before", "after"])
        verdict = self._fake({
            "run_correctness": lambda *a, **k: True,
            "run_suite": lambda *a, **k: clean_repeat(),
            "QSH": type("H", (), {"tree_hash": staticmethod(lambda p: next(hashes))}),
        })
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["stage"], "identity")
        self.assertTrue(any("not same-variant" in p for p in verdict["problems"]))

    def test_too_few_complete_repeats_refuses(self):
        calls = {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] > 2:
                raise RuntimeError("device fell over")
            return clean_repeat()

        verdict = self._fake({
            "run_correctness": lambda *a, **k: True,
            "run_suite": flaky,
            "QSH": type("H", (), {"tree_hash": staticmethod(lambda p: "same")}),
        })
        self.assertFalse(verdict["ok"])
        self.assertTrue(any("complete repeats" in p for p in verdict["problems"]))
        self.assertTrue(any("device fell over" in p for p in verdict["problems"]),
                        "a lost repeat must be reported, not just counted")


class RenderTest(unittest.TestCase):
    TABLE = {"decode_m2_square": {"floor": 0.0721, "clamped_to_min": False},
             "prefill_m256_down": {"floor": 0.002, "clamped_to_min": True}}

    def test_python_render_is_parseable_and_names_the_machine(self):
        text = M.render_python("Q", self.TABLE)
        self.assertIn('MEASURED_NOISE_FLOOR_BY_MACHINE["Q"]', text)
        ns: dict = {"MEASURED_NOISE_FLOOR_BY_MACHINE": {}}
        exec(compile(text, "<render>", "exec"), ns)
        self.assertAlmostEqual(
            ns["MEASURED_NOISE_FLOOR_BY_MACHINE"]["Q"]["decode_m2_square"], 0.0721)

    def test_a_clamped_floor_is_marked_in_the_emitted_source(self):
        text = M.render_python("Q", self.TABLE)
        clamped_line = [l for l in text.splitlines() if "prefill_m256_down" in l][0]
        self.assertIn("clamped", clamped_line)

    def test_js_render_carries_the_same_numbers(self):
        text = M.render_js("Q", self.TABLE)
        self.assertIn("['Q', new Map([", text)
        self.assertIn("['decode_m2_square', 0.0721]", text)

    def test_both_engines_are_emitted_together(self):
        # (58): the table lives in two files and a table in one of them is a
        # table that will drift. The CLI must print both fragments.
        out = self._cli()
        self.assertIn("qd_robust_stats.py", out)
        self.assertIn("kernel_lane.js", out)
        self.assertIn("PROVISIONAL_MACHINES", out)

    def _cli(self) -> str:
        # Stamped with the box the default --machine (CURRENT_MACHINE) names,
        # because an unattributable verdict is now refused before rendering.
        verdict = {
            "ok": True, "stage": "sweep", "source_hash": "abc", "repeats_complete": 8,
            "host": QRS.MACHINE_HOSTNAME[QRS.CURRENT_MACHINE],
            "host_machine": QRS.CURRENT_MACHINE,
            "routes": self.TABLE, "problems": [],
        }
        path = HERE / "_tmp_floor_verdict.json"
        path.write_text(json.dumps(verdict))
        try:
            proc = subprocess.run(
                [sys.executable, str(HERE / "measure_noise_floor.py"),
                 "--task", ".", "--from-json", str(path)],
                capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            return proc.stdout
        finally:
            path.unlink(missing_ok=True)


class AttributionTest(unittest.TestCase):
    """A verdict is a reading of one box. The letter must not come from argv.

    Written after re-rendering epoch R's saved verdict (/tmp/floor_R.json,
    taken on tw008) as an `epoch S` table with one command, on a box where S
    had no measured floors at all. Nothing objected: the numbers printed under
    an S heading, carrying R's source_hash, ready to paste. That is finding
    (107) -- floors carried across a machine boundary on an argument -- with
    the boundary crossed by a CLI flag instead of a copy-paste.
    """

    def stamped(self, host: str) -> dict:
        return {"ok": True, "stage": "sweep", "host": host}

    def test_a_verdict_from_another_epochs_box_is_refused(self):
        other = "tw008" if QRS.MACHINE_HOSTNAME[QRS.CURRENT_MACHINE] != "tw008" else "tw003"
        problems = M.attribution_problems(self.stamped(other),
                                          QRS.CURRENT_MACHINE)
        self.assertTrue(problems)
        self.assertIn(other, problems[0])

    def test_an_unstamped_verdict_is_refused_rather_than_assumed_local(self):
        # Every verdict written before the stamp existed lands here. It cannot
        # be attributed after the fact, and guessing "probably this box" is the
        # assumption the stamp exists to remove.
        problems = M.attribution_problems({"ok": True}, QRS.CURRENT_MACHINE)
        self.assertTrue(problems)
        self.assertIn("no host", problems[0])

    def test_a_verdict_from_an_unregistered_box_is_refused(self):
        # A host in no epoch at all. This used to be tw051 (wave 1 ran on it
        # unregistered), but tw051 became epoch V on 2026-08-18 -- so the case
        # now uses a name that cannot be registered out from under it.
        problems = M.attribution_problems(self.stamped("tw000-unregistered"),
                                          QRS.CURRENT_MACHINE)
        self.assertTrue(problems)
        self.assertIn("no epoch", problems[0])

    def test_the_matching_box_passes(self):
        host = QRS.MACHINE_HOSTNAME[QRS.CURRENT_MACHINE]
        self.assertEqual([], M.attribution_problems(self.stamped(host),
                                                    QRS.CURRENT_MACHINE))

    def test_the_cli_refuses_a_foreign_verdict_and_prints_no_table(self):
        other = "tw008" if QRS.MACHINE_HOSTNAME[QRS.CURRENT_MACHINE] != "tw008" else "tw003"
        path = HERE / "_tmp_floor_foreign.json"
        path.write_text(json.dumps({
            "ok": True, "stage": "sweep", "source_hash": "abc",
            "repeats_complete": 8, "host": other, "routes": {}, "problems": [],
        }))
        try:
            proc = subprocess.run(
                [sys.executable, str(HERE / "measure_noise_floor.py"),
                 "--task", ".", "--from-json", str(path)],
                capture_output=True, text=True)
        finally:
            path.unlink(missing_ok=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("REFUSED", proc.stderr)
        # The dangerous outcome is not a bad exit code nobody reads -- it is a
        # pasteable table. There must not be one.
        self.assertNotIn("MEASURED_NOISE_FLOOR_BY_MACHINE", proc.stdout)

    def test_the_sweep_stamps_the_host_it_ran_on(self):
        # Guards the other end: a stamp the producer stops writing would make
        # every verdict unattributable and every check above vacuous.
        src = (HERE / "measure_noise_floor.py").read_text()
        self.assertIn('"host": host', src)


class ExitCodeTest(unittest.TestCase):
    def _run(self, verdict: dict) -> subprocess.CompletedProcess:
        path = HERE / "_tmp_floor_verdict_exit.json"
        path.write_text(json.dumps(verdict))
        try:
            return subprocess.run(
                [sys.executable, str(HERE / "measure_noise_floor.py"),
                 "--task", ".", "--from-json", str(path)],
                capture_output=True, text=True)
        finally:
            path.unlink(missing_ok=True)

    def test_correctness_and_identity_exit_3(self):
        for stage in ("correctness", "identity"):
            proc = self._run({"ok": False, "stage": stage, "problems": ["nope"]})
            self.assertEqual(proc.returncode, 3, stage)
            self.assertIn("REFUSED", proc.stderr)

    def test_a_bad_measurement_exits_4(self):
        proc = self._run({"ok": False, "stage": "sweep", "problems": ["too few"]})
        self.assertEqual(proc.returncode, 4)

    def test_a_refusal_emits_no_table(self):
        proc = self._run({"ok": False, "stage": "sweep", "problems": ["too few"]})
        self.assertNotIn("MEASURED_NOISE_FLOOR_BY_MACHINE", proc.stdout)


if __name__ == "__main__":
    unittest.main()
