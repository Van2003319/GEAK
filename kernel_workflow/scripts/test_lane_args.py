#!/usr/bin/env python3
"""Tests for lane_args.py.

The load-bearing one is `test_the_wave8_omission_is_refused`: it rebuilds the
exact file that cost this project its largest result -- the greedy lane's launch
arguments with `min_improve` dropped -- and pins that the checker refuses it. The
rest of the module is only worth having if that case fails loudly.

`test_the_accepted_set_is_extracted_not_restated` pins the other half: the
accepted-argument list must come out of the JS entry points' own source. A
restated copy would be a third list to keep in sync, and this module exists
because a hand-transcribed list was wrong.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lane_args as la

SCRIPTS = Path(__file__).resolve().parent
LANE_FILE = SCRIPTS.parent / "lanes" / "greedy_bf16_gemm.json"

GOOD = {
    "kernel_path": "/abs/task",
    "workflow_dir": "/abs/kernel_workflow",
    "mode": "optimize",
    "budget": 12,
    "min_improve": 0.005,
    "gpu_ids": "2,3",
    "state_dir": "/abs/state",
}


def write(obj) -> Path:
    fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(obj, fh)
    fh.close()
    return Path(fh.name)


class AcceptedSetTest(unittest.TestCase):
    def test_the_accepted_set_is_extracted_not_restated(self):
        lane = la.known_args(la.LANE_JS)
        disp = la.known_args(la.DISPATCH_JS)
        # Sanity: these are real sets read from real files, not empty defaults.
        self.assertIn("min_improve", lane)
        self.assertIn("route_bands", lane)
        self.assertIn("backends", disp)
        # The dispatcher forwards everything the lane reads, so it must accept at least as much.
        self.assertEqual(set(), lane - disp, "the dispatcher must accept every lane argument")

    def test_an_entry_point_that_lost_its_check_is_refused_not_defaulted(self):
        """If KNOWN_ARGS is ever deleted from an entry point, that is a silent-ignore
        regression -- so extraction must fail loudly rather than return an empty set,
        which would make every argument look unknown OR (worse) every one look fine."""
        p = write({})
        p.write_text("const KNOWN_ARGS = nothing;")
        with self.assertRaises(la.LaneArgsError) as ctx:
            la.known_args(p)
        self.assertIn("KNOWN_ARGS", str(ctx.exception))

    def test_search_strategy_is_not_accepted_by_either_entry_point(self):
        """The dead argument that motivated the whole check."""
        both = la.known_args(la.LANE_JS) | la.known_args(la.DISPATCH_JS)
        self.assertNotIn("search_strategy", both)


class CheckTest(unittest.TestCase):
    def test_a_good_file_has_no_problems(self):
        self.assertEqual([], la.check(write(GOOD)))

    def test_the_wave8_omission_is_refused(self):
        """The exact failure: the protocol pins min_improve=0.005, the file was retyped
        without it, the wave ran at the 0.02 default and refused a verified +1.58%.

        No check inside a running workflow can catch this -- an absent argument and a
        default one are the same thing by the time the run starts -- so it has to be
        caught here, against a written-down protocol, before launch."""
        d = dict(GOOD, _require={"min_improve": 0.005})
        del d["min_improve"]
        problems = la.check(write(d))
        self.assertEqual(1, len(problems), problems)
        self.assertIn("min_improve", problems[0])
        self.assertIn("ABSENT", problems[0])

    def test_an_omission_reads_differently_from_a_drift(self):
        """Two different fixes, so they must not produce the same message."""
        absent = la.check(write({k: v for k, v in
                                 dict(GOOD, _require={"min_improve": 0.005}).items()
                                 if k != "min_improve"}))[0]
        drifted = la.check(write(dict(GOOD, min_improve=0.02,
                                      _require={"min_improve": 0.005})))[0]
        self.assertIn("ABSENT", absent)
        self.assertNotIn("ABSENT", drifted)
        self.assertIn("0.02", drifted)

    def test_a_misspelled_knob_is_refused(self):
        problems = la.check(write(dict(GOOD, min_improv=0.005)))
        self.assertTrue(any("min_improv" in p and "silently ignored" in p for p in problems),
                        problems)

    def test_a_dead_argument_is_refused(self):
        problems = la.check(write(dict(GOOD, search_strategy="greedy")))
        self.assertTrue(any("search_strategy" in p for p in problems), problems)

    def test_a_missing_required_argument_is_refused(self):
        for key in la.REQUIRED_KEYS:
            with self.subTest(key=key):
                problems = la.check(write({k: v for k, v in GOOD.items() if k != key}))
                self.assertTrue(any(key in p and "required" in p for p in problems), problems)

    def test_a_relative_path_is_refused(self):
        problems = la.check(write(dict(GOOD, state_dir="exp/state")))
        self.assertTrue(any("absolute" in p for p in problems), problems)

    def test_cli_require_overrides_and_adds(self):
        problems = la.check(write(GOOD), {"budget": 6})
        self.assertTrue(any("budget" in p and "12" in p for p in problems), problems)

    def test_underscore_keys_are_comments_and_never_forwarded(self):
        args, require = la.load(write(dict(GOOD, _note="anything", _require={"budget": 12})))
        self.assertNotIn("_note", args)
        self.assertNotIn("_require", args)
        self.assertEqual({"budget": 12}, require)

    def test_a_non_object_file_is_refused(self):
        with self.assertRaises(la.LaneArgsError):
            la.check(write([1, 2, 3]))


class RenderTest(unittest.TestCase):
    def test_the_rendered_call_carries_every_argument_and_no_comments(self):
        out = la.render(write(dict(GOOD, _note="not an argument")))
        self.assertIn('scriptPath: "/abs/kernel_workflow/kernel_workflow.js"', out)
        for key in GOOD:
            self.assertIn(f"{key}:", out)
        self.assertNotIn("_note", out)

    def test_the_rendered_args_body_is_a_JS_OBJECT_not_a_json_string(self):
        """The README's standing warning: `args` must arrive as a real object, because a
        JSON-encoded STRING there makes the workflow unable to read args.workflow_dir and
        abort immediately. So the rendered form is a JS object literal -- bare keys, no
        quotes around the braces -- and each VALUE is JSON-encoded so it pastes correctly."""
        out = la.render(write(GOOD))
        self.assertRegex(out, r"args: \{")
        self.assertNotRegex(out, r'args: "')          # not a stringified object
        self.assertNotRegex(out, r'args: \'')
        for key, val in GOOD.items():
            # Bare key, JSON-encoded value: `budget: 12`, `mode: "optimize"`.
            self.assertIn(f"{key}: {json.dumps(val)}", out)
        # A string value keeps its quotes and a number does not gain any -- the distinction
        # that makes `budget` an int rather than the string "12" on the other side.
        self.assertIn('mode: "optimize"', out)
        self.assertIn("budget: 12", out)


class CommittedLaneFileTest(unittest.TestCase):
    """The greedy lane's own file, checked in CI rather than by whoever launches next."""

    def test_the_committed_greedy_lane_file_is_launchable(self):
        self.assertTrue(LANE_FILE.exists(), f"{LANE_FILE} is missing")
        self.assertEqual([], la.check(LANE_FILE))

    def test_it_pins_the_values_the_protocol_calls_load_bearing(self):
        args, require = la.load(LANE_FILE)
        self.assertEqual(0.005, require.get("min_improve"))
        self.assertEqual("optimize", require.get("mode"))
        self.assertIn("state_dir", require,
                      "state_dir's presence is what makes a wave a continuation rather than a "
                      "cold reseed; it has to be pinned, not merely present")

    def test_it_does_not_pin_route_bands(self):
        """Passing a band table would tie the gate to whichever epoch produced it -- how the
        one file on disk went six epochs stale. The lane derives its own from this wave's
        baseline repeats instead."""
        args, _ = la.load(LANE_FILE)
        self.assertNotIn("route_bands", args)

    def test_it_does_not_enable_the_isa_gate(self):
        """`gate` mode reads a patch that ADDS a kernel symbol as mechanism_verdict=refuted by
        construction, and adding a specialised instantiation is this lane's most common winning
        shape. Until that pairing artefact is fixed, gate mode would refuse real winners."""
        args, _ = la.load(LANE_FILE)
        self.assertNotEqual("gate", args.get("isa_evidence"))


class CliTest(unittest.TestCase):
    def run_cli(self, *argv):
        return subprocess.run([sys.executable, str(SCRIPTS / "lane_args.py"), *argv],
                              capture_output=True, text=True)

    def test_check_exits_zero_on_the_committed_file(self):
        r = self.run_cli("--check", str(LANE_FILE))
        self.assertEqual(0, r.returncode, r.stderr)
        self.assertIn("OK:", r.stdout)

    def test_check_exits_three_and_names_every_problem(self):
        p = write(dict(GOOD, bogus_key=1, _require={"budget": 6}))
        r = self.run_cli("--check", str(p))
        self.assertEqual(3, r.returncode)
        self.assertIn("bogus_key", r.stderr)
        self.assertIn("budget", r.stderr)

    def test_an_unreadable_file_exits_two_not_three(self):
        """A file that cannot be parsed is a different failure from a file that parses
        and is wrong, and the exit codes have to distinguish them for a launch script."""
        r = self.run_cli("--check", "/nonexistent/lane.json")
        self.assertEqual(2, r.returncode)


class ResolvedAtLaunchTest(unittest.TestCase):
    """`route_bands: "@current_epoch"` -- the one value that must not be written down.

    The per-route floors describe the box, this container changes box every few
    hours, and the only floor table ever committed went six epochs stale while
    still reading like a current measurement. Writing today's numbers into a lane
    file reproduces that exactly. So the file names the SOURCE and `--check`
    resolves it, which also means an epoch with no measured floors is a refusal
    before the wave starts rather than a table of fail-closed defaults that
    quietly holds every route to ~7% for the whole run.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="laneargs_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def write(self, **extra) -> Path:
        p = self.tmp / "lane.json"
        p.write_text(json.dumps({
            "kernel_path": "/abs/k", "workflow_dir": "/abs/wf", **extra}), encoding="utf-8")
        return p

    def test_the_directive_resolves_to_the_live_epochs_measured_table(self):
        import noise_floor_stats as QRS
        import route_floors
        try:
            want = route_floors.resolve()
        except route_floors.FloorsUnavailable as exc:
            self.skipTest(f"this epoch cannot supply a table: {exc}")
        args, _ = la.load(self.write(route_bands="@current_epoch"))
        self.assertEqual(want, args["route_bands"])
        # ...and it is the LIVE epoch's table, not some other epoch's that happens to have the same
        # route names. That distinction is the whole point: every epoch's table has these same
        # eleven keys, so a resolver pointed at the wrong one looks perfectly healthy.
        self.assertEqual(set(QRS.MEASURED_NOISE_FLOOR_BY_MACHINE[QRS.CURRENT_MACHINE]),
                         set(args["route_bands"]))
        for route, floor in QRS.MEASURED_NOISE_FLOOR_BY_MACHINE[QRS.CURRENT_MACHINE].items():
            self.assertAlmostEqual(floor, args["route_bands"][route], places=6)

    def test_an_explicit_table_is_left_exactly_as_written(self):
        """The directive is opt-in. A caller that genuinely has a better table --
        a fresh 8-repeat sweep it just took -- must still be able to pass it."""
        table = {"decode_m2_square": 0.05}
        args, _ = la.load(self.write(route_bands=table))
        self.assertEqual(table, args["route_bands"])

    def test_an_unresolvable_epoch_refuses_the_file_rather_than_the_wave(self):
        """The failure this is really for. A provisional epoch's table is shaped
        exactly like a measured one, so nothing downstream can tell them apart;
        handed to the gate it raises every route's bar to the fail-closed default
        and the wave runs unable to accept anything, with no line saying so."""
        import route_floors
        real = route_floors.resolve

        def boom(*a, **k):
            raise route_floors.FloorsUnavailable("epoch Q is PROVISIONAL -- not a measurement")
        route_floors.resolve = boom
        try:
            with self.assertRaises(la.LaneArgsError) as caught:
                la.load(self.write(route_bands="@current_epoch"))
        finally:
            route_floors.resolve = real
        self.assertIn("PROVISIONAL", str(caught.exception))
        self.assertIn("route_bands", str(caught.exception))

    def test_check_reports_that_it_resolved_rather_than_reading_a_literal(self):
        import route_floors
        try:
            route_floors.resolve()
        except route_floors.FloorsUnavailable as exc:
            self.skipTest(f"this epoch cannot supply a table: {exc}")
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "lane_args.py"), "--check", str(self.write(route_bands="@current_epoch"))],
            capture_output=True, text=True)
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("resolved at launch", proc.stdout,
                      "a resolved value that reports like a literal is one nobody checks the age of")

    def test_every_resolver_names_a_key_the_entry_points_accept(self):
        """(62) as a completeness claim: a directive on a key no worker reads
        would resolve beautifully and be dropped."""
        accepted = la.known_args(la.LANE_JS) | la.known_args(la.DISPATCH_JS)
        for key in la.RESOLVERS:
            with self.subTest(key=key):
                self.assertIn(key, accepted)


if __name__ == "__main__":
    unittest.main()
