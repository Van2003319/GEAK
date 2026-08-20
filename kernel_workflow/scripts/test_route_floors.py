#!/usr/bin/env python3
"""Tests for `route_floors.py`.

The failure this file is really about is not a wrong number, it is a table that
reaches the gate while describing a different box, or a fail-closed default
reaching it dressed as a measurement. Both look fine in the output and turn the
commit gate off for a whole wave.

Epoch letters are derived, never named: they are consumed roughly once per
machine change here, and a test that hardcoded one would go red on the next
restore reporting a bug where there was only a fixture that had aged out.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import noise_floor_stats as QRS  # noqa: E402

SCRIPT = HERE / "route_floors.py"
LIVE = QRS.CURRENT_MACHINE
LIVE_HOST = QRS.MACHINE_HOSTNAME[LIVE]
MEASURED = sorted(set(QRS.MEASURED_NOISE_FLOOR_BY_MACHINE) - QRS.PROVISIONAL_MACHINES)
# The epoch these tests exercise the EMITTER against. Deliberately not the live one: for a window
# after every machine change -- and this container changes machine every few hours -- the live epoch
# is registered but not yet swept, so `resolve()` refuses it, correctly. Tests that went red during
# that window would be red on schedule for reasons that are not defects, which is how a suite trains
# people to ignore it. The live epoch gets its own tests below, and they skip when it is provisional.
SAMPLE = MEASURED[-1] if MEASURED else None
LIVE_IS_MEASURED = LIVE not in QRS.PROVISIONAL_MACHINES
PROVISIONAL = sorted(QRS.PROVISIONAL_MACHINES & set(QRS.MEASURED_NOISE_FLOOR_BY_MACHINE))
UNREGISTERED = next(c for c in "BCDEFGHIJK"
                    if c not in QRS.MACHINE_HOSTNAME
                    and c not in QRS.MEASURED_NOISE_FLOOR_BY_MACHINE)


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True)


class RouteFloorsTest(unittest.TestCase):

    def test_a_measured_epoch_emits_its_table(self):
        proc = run("--machine", SAMPLE)
        self.assertEqual(0, proc.returncode, proc.stderr)
        got = json.loads(proc.stdout)
        self.assertEqual(SAMPLE, got["machine"])
        self.assertEqual(QRS.MACHINE_HOSTNAME[SAMPLE], got["host"])
        self.assertTrue(got["measured"])
        self.assertEqual(QRS.MEASURED_NOISE_FLOOR_BY_MACHINE[SAMPLE].keys(), got["floors"].keys())

    def test_the_numbers_are_the_table_and_not_a_rounding_of_it(self):
        """A floor is compared against a per-route delta, so a lossy round trip
        moves the bar. Six decimals is four more than any floor carries."""
        got = json.loads(run("--machine", SAMPLE).stdout)["floors"]
        for route, want in QRS.MEASURED_NOISE_FLOOR_BY_MACHINE[SAMPLE].items():
            with self.subTest(route=route):
                self.assertAlmostEqual(want, got[route], places=6)

    def test_a_provisional_epoch_is_refused_rather_than_emitted(self):
        """The one that matters. A provisional table is shaped exactly like a
        measured one -- that is what keeps every caller simple -- so nothing
        downstream can tell them apart. Handed to the gate it would raise every
        route's bar to the fail-closed default (~7%), which is far above any
        real win this lane produces, and the wave would run with the gate
        silently unable to accept anything."""
        if not PROVISIONAL:
            self.skipTest("no provisional epoch registered to test against")
        # Prefer the LIVE one when it is provisional: that is the case that actually occurs, in the
        # window between registering a new box and sweeping its floors.
        target = LIVE if not LIVE_IS_MEASURED else PROVISIONAL[0]
        proc = run("--machine", target)
        self.assertEqual(3, proc.returncode)
        self.assertIn("PROVISIONAL", proc.stderr)
        self.assertIn("measure_noise_floor.py", proc.stderr,
                      "a refusal that does not say how to clear itself is a dead end")

    def test_a_provisional_epoch_can_still_be_forced_and_says_so(self):
        if not PROVISIONAL:
            self.skipTest("no provisional epoch registered to test against")
        target = LIVE if not LIVE_IS_MEASURED else PROVISIONAL[0]
        proc = run("--machine", target, "--allow-provisional")
        self.assertEqual(0, proc.returncode, proc.stderr)
        floors = json.loads(proc.stdout)["floors"]
        self.assertEqual({QRS.DEFAULT_NOISE_FLOOR}, set(floors.values()))

    def test_an_unregistered_epoch_is_refused(self):
        proc = run("--machine", UNREGISTERED)
        self.assertEqual(4, proc.returncode)
        self.assertIn("no table", proc.stderr)

    def test_the_args_fragment_parses_as_json_after_the_key(self):
        """`--arg` exists so the table is pasted rather than retyped, and the
        thing it is pasted into is validated by lane_args.py. A fragment that
        does not parse would be found at launch, which is the one moment nobody
        has time to debug it."""
        proc = run("--machine", SAMPLE, "--arg")
        self.assertEqual(0, proc.returncode, proc.stderr)
        obj = json.loads("{" + proc.stdout.rstrip().rstrip(",") + "}")
        self.assertEqual(QRS.MEASURED_NOISE_FLOOR_BY_MACHINE[SAMPLE].keys(),
                         obj["route_bands"].keys())
        self.assertIn(SAMPLE, obj["_route_bands_provenance"])
        self.assertIn("measured", obj["_route_bands_provenance"],
                      "the fragment must carry which box and which epoch it describes; a floor "
                      "table with no provenance is how one went six epochs stale unnoticed")

    def test_the_default_machine_is_the_live_one(self):
        """Called with no --machine on the box the epoch describes, it must emit
        that epoch. On any other box it must refuse -- see the next test."""
        import socket
        if QRS.machine_for_host(socket.gethostname()) != LIVE:
            self.skipTest("this host is not the live epoch's host")
        if not LIVE_IS_MEASURED:
            self.skipTest(f"epoch {LIVE} is registered but not yet swept -- the correct state for a "
                          "box the container has just landed on, and `resolve()` refusing it is the "
                          "behaviour tested above, not a defect here")
        self.assertEqual(json.loads(run().stdout)["machine"], LIVE)

    def test_it_refuses_on_a_host_the_live_epoch_does_not_describe(self):
        """The whole point of the frame fence, one level up. gpu_lock refuses to
        TIME on the wrong box; this refuses to hand the gate the wrong box's
        floors, which fails later and much more quietly."""
        import socket
        if QRS.machine_for_host(socket.gethostname()) == LIVE:
            self.skipTest("this host IS the live epoch's host, so there is nothing to refuse")
        proc = run()
        self.assertEqual(4, proc.returncode)
        self.assertIn("check_measurement_frame.py", proc.stderr)

    def test_every_measured_epoch_round_trips(self):
        """Not vacuous: if the emitter only worked for the live epoch this would
        pass on one letter and hide the rest."""
        self.assertGreaterEqual(len(MEASURED), 3, MEASURED)
        for m in MEASURED:
            with self.subTest(machine=m):
                proc = run("--machine", m)
                self.assertEqual(0, proc.returncode, proc.stderr)
                self.assertEqual(QRS.MEASURED_NOISE_FLOOR_BY_MACHINE[m],
                                 {k: v for k, v in json.loads(proc.stdout)["floors"].items()})


if __name__ == "__main__":
    unittest.main(verbosity=2)
