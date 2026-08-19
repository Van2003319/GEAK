#!/usr/bin/env python3
"""Tests for the measurement-frame preflight.

The point of the preflight is to catch a frame that no test was watching. So
its own logic is tested through the pure `classify`, not by running the script
on whichever box happens to be underneath -- otherwise only the one branch that
matches today's machine is ever exercised, which is the same blind spot one
level up.
"""

import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_measurement_frame as CMF  # noqa: E402
import noise_floor_stats as QRS  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "check_measurement_frame.py"
PROV = {"Q", "S"}


def cls(host, resolved, current):
    return CMF.classify(host, resolved, current, PROV)[0]


class ClassifyTest(unittest.TestCase):
    def test_measured_epoch_matching_host_is_cleared(self):
        self.assertEqual(cls("tw008", "R", "R"), 0)

    def test_provisional_epoch_matching_host_is_3_not_0(self):
        # Fail-closed, not a fault: the box is registered, the floors are just
        # the 0.072 default until someone measures them.
        self.assertEqual(cls("tw054", "S", "S"), 3)

    def test_host_resolving_to_a_different_epoch_is_4(self):
        # The exact 2026-08-17 bug: rounds ran on tw003 (epoch Q) while
        # CURRENT_MACHINE had been left at another letter.
        self.assertEqual(cls("tw003", "Q", "S"), 4)

    def test_unregistered_host_is_4(self):
        # wave 1 ran on tw051, which is in no epoch at all. This branch is the
        # one that would have stopped it.
        # (tw051 itself became epoch V on 2026-08-18, so it is no longer an
        # example of an unregistered box; the branch is what is under test, and
        # it needs a host that resolves to nothing. Do NOT re-point this at a
        # real hostname -- the next restore would silently disarm it again.)
        self.assertEqual(cls("tw000-unregistered", None, "S"), 4)


class StateContinuityTest(unittest.TestCase):
    """Host continuity is recorded, and deliberately does NOT gate the lane.

    Every admission is a paired same-session ratio, and ratios compose across
    a restore, so a machine change is not a fault -- turning it into a hard
    stop would be a false alarm on every legitimate restore. What it IS is the
    fact that took a whole session to recover by hand from rocprofv3 directory
    names, so it gets written down.
    """

    def test_a_fresh_state_records_the_first_host(self):
        line, frame = CMF.state_continuity({}, "tw054", "S")
        self.assertEqual(frame["first_host"], "tw054")
        self.assertEqual(frame["hosts_seen"], ["tw054"])
        self.assertIn("all recorded rounds on tw054", line)

    def test_a_second_box_is_appended_and_the_first_is_kept(self):
        _, frame = CMF.state_continuity({}, "tw051", None)
        line, frame = CMF.state_continuity({"measurement_frame": frame},
                                           "tw054", "S")
        self.assertEqual(frame["first_host"], "tw051")
        self.assertEqual(frame["hosts_seen"], ["tw051", "tw054"])
        self.assertEqual(frame["last_host"], "tw054")
        self.assertIn("2 boxes", line)

    def test_returning_to_a_known_box_does_not_duplicate_it(self):
        frame = {"hosts_seen": ["tw051", "tw054"], "first_host": "tw051"}
        _, frame = CMF.state_continuity({"measurement_frame": frame},
                                        "tw051", None)
        self.assertEqual(frame["hosts_seen"], ["tw051", "tw054"])
        self.assertEqual(frame["last_host"], "tw051")

    def test_continuity_never_changes_the_exit_code(self):
        # The whole point: a multi-box lane is reported, not blocked.
        before = CMF.classify("tw054", "S", "S", PROV)[0]
        CMF.state_continuity({"measurement_frame": {"hosts_seen": ["tw003"]}},
                             "tw054", "S")
        self.assertEqual(CMF.classify("tw054", "S", "S", PROV)[0], before)

    def test_stamp_round_trips_through_the_cli(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "STATE.json"
            p.write_text(json.dumps({"cumulative": 1.39884}))
            r = subprocess.run(
                [sys.executable, str(SCRIPT), "--host", "tw054",
                 "--state", str(p), "--stamp"],
                capture_output=True, text=True)
            self.assertIn("stamped into", r.stdout)
            after = json.loads(p.read_text())
            # The stamp must not eat the state it was handed.
            self.assertEqual(after["cumulative"], 1.39884)
            self.assertEqual(after["measurement_frame"]["last_host"], "tw054")

    def test_without_stamp_the_state_file_is_untouched(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "STATE.json"
            original = json.dumps({"cumulative": 1.39884})
            p.write_text(original)
            subprocess.run([sys.executable, str(SCRIPT), "--host", "tw054",
                            "--state", str(p)], capture_output=True, text=True)
            self.assertEqual(p.read_text(), original)


class EndToEndTest(unittest.TestCase):
    def _run(self, *extra):
        return subprocess.run([sys.executable, str(SCRIPT), *extra],
                              capture_output=True, text=True)

    def test_exit_code_matches_classify_on_this_box(self):
        import socket
        host = socket.gethostname()
        # Must use the LIVE provisional set, not the module-level PROV
        # fixture. PROV is frozen at {"Q","S"} so the pure-function tests above
        # stay readable, but this one is about the real frame -- when an epoch
        # gets deprovisionalised the fixture goes stale and the assertion
        # starts comparing the script against a set nobody uses.
        expected = CMF.classify(host, QRS.machine_for_host(host),
                                QRS.CURRENT_MACHINE,
                                QRS.PROVISIONAL_MACHINES)[0]
        self.assertEqual(self._run().returncode, expected)

    def test_host_override_reproduces_the_historical_failure(self):
        # Finding (126): the letter said one box, the host was another. The
        # host that reproduces it depends on which letter is CURRENT, so pick
        # one that actually differs instead of hardcoding tw003 -- when the
        # lane moves onto tw003 that hardcoded name stops being a mismatch and
        # the test starts asserting the wrong thing.
        other = next(h for h in QRS.MACHINE_HOSTNAME.values()
                     if QRS.machine_for_host(h) != QRS.CURRENT_MACHINE)
        r = self._run("--host", other)
        self.assertEqual(r.returncode, 4, other)
        self.assertIn("another machine's ruler", r.stdout)

    def test_unregistered_host_names_itself_in_the_message(self):
        # This used to pass a real hostname (tw051), which stopped being
        # unregistered the moment the lane was restored onto it and it became
        # epoch V -- the test then asserted 4 and got 0. Same failure mode the
        # comment on the test above warns about, so take the same cure: a name
        # that is structurally never in MACHINE_HOSTNAME, asserted to be so.
        unknown = "tw000-unregistered"
        self.assertNotIn(unknown, set(QRS.MACHINE_HOSTNAME.values()))
        r = self._run("--host", unknown)
        self.assertEqual(r.returncode, 4)
        self.assertIn(unknown, r.stdout)


if __name__ == "__main__":
    unittest.main()
