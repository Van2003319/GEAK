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
    """`@token` values, which this module fills in at launch instead of writing down.

    RESOLVERS is currently empty -- the per-machine noise-floor resolver that
    motivated the mechanism is deleted -- so what is pinned here is the mechanism
    itself: an explicit value still passes through untouched, and a directive no
    resolver claims is DROPPED rather than handed downstream as a literal string.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="laneargs_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def write(self, **extra) -> Path:
        p = self.tmp / "lane.json"
        p.write_text(json.dumps({
            "kernel_path": "/abs/k", "workflow_dir": "/abs/wf", **extra}), encoding="utf-8")
        return p

    def test_an_explicit_table_is_left_exactly_as_written(self):
        """A caller that genuinely has a measured table must still be able to pass it."""
        table = {"decode_m2_square": 0.05}
        args, _ = la.load(self.write(route_bands=table))
        self.assertEqual(table, args["route_bands"])

    def test_an_unclaimed_directive_is_dropped_rather_than_passed_through(self):
        """Lane files and the code that reads them are versioned separately, so a file
        naming a directive this build no longer implements is reachable. Passed through,
        `"@current_epoch"` reaches kernel_lane.js's ROUTE_BANDS parser, which JSON.parses
        it and throws -- turning a stale key into a dead lane instead of an ignored one."""
        args, _ = la.load(self.write(route_bands="@current_epoch"))
        self.assertNotIn("route_bands", args)

    def test_dropping_a_directive_is_announced(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "lane_args.py"), "--check",
             str(self.write(route_bands="@current_epoch"))],
            capture_output=True, text=True)
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("dropping unresolved directive", proc.stderr,
                      "a key that silently vanishes is one nobody notices going stale")

    def test_every_resolver_names_a_key_the_entry_points_accept(self):
        """(62) as a completeness claim: a directive on a key no worker reads
        would resolve beautifully and be dropped."""
        accepted = la.known_args(la.LANE_JS) | la.known_args(la.DISPATCH_JS)
        for key in la.RESOLVERS:
            with self.subTest(key=key):
                self.assertIn(key, accepted)


if __name__ == "__main__":
    unittest.main()
