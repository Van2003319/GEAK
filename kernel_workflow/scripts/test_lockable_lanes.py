#!/usr/bin/env python3
"""Tests for `lockable_lanes.py`.

The whole point of the module is to be the thing that does NOT get forgotten at
3am, so the properties worth pinning are the ones whose failure is silent: an
empty result that reads as "all GPUs allowed", an optimistic answer for a lane
that cannot actually be opened, and a parser that understands only whichever
input shape its author happened to type first.
"""
from __future__ import annotations

import io
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lockable_lanes as LL  # noqa: E402


class ParseTest(unittest.TestCase):
    def test_every_input_shape_yields_the_same_ids(self):
        for text in ("0 4 5", "0,4,5", "poll 61 idle=[0 4 5]",
                     "GPU_IDLE_CONFIRMED: 0 4 5 ", "[0, 4, 5]"):
            with self.subTest(text=text):
                self.assertEqual([0, 4, 5], LL.parse_ids(text))

    def test_duplicates_collapse_and_order_is_kept(self):
        self.assertEqual([3, 1, 2], LL.parse_ids("3 1 3 2 1"))

    def test_no_digits_is_no_ids_not_a_crash(self):
        self.assertEqual([], LL.parse_ids("idle=[]"))

    def test_a_timestamped_line_without_a_marker_is_refused_not_mined(self):
        """Pulling every integer out of the real watcher line yields
        `2026,8,16,22,45,34,61,...` -- a timestamp and a poll counter promoted
        to GPU ids. Refusing is the generous behaviour here."""
        with self.assertRaises(ValueError):
            LL.parse_ids("2026-08-16T22:45:34+00:00 poll 61")

    def test_the_marker_wins_over_everything_around_it(self):
        line = "2026-08-16T22:45:34+00:00 poll 61 idle=[2 3]"
        self.assertEqual([2, 3], LL.parse_ids(line))

    def test_the_last_line_of_a_tail_is_the_current_one(self):
        tail = "poll 60 idle=[]\npoll 61 idle=[4 5]\n"
        self.assertEqual([4, 5], LL.parse_ids(tail))


class LockableTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="lanes_"))

    def test_a_writable_directory_admits_a_new_lane(self):
        self.assertTrue(LL.is_lockable(9, self.dir))
        self.assertTrue((self.dir / "gpu_9.lock").exists())

    def test_an_unwritable_directory_refuses_a_new_lane(self):
        """The live failure: the lane file does not exist and cannot be
        created, so `gpu_lock.sh` dies on the redirect."""
        os.chmod(self.dir, stat.S_IRUSR | stat.S_IXUSR)
        self.addCleanup(os.chmod, self.dir, 0o755)
        self.assertFalse(LL.is_lockable(9, self.dir))

    def test_an_existing_but_unwritable_lane_is_refused(self):
        """The other half of the live failure: the file is there, which is
        exactly why a check for existence would pass it."""
        path = self.dir / "gpu_3.lock"
        path.touch()
        os.chmod(path, stat.S_IRUSR)
        self.addCleanup(os.chmod, path, 0o644)
        self.assertFalse(LL.is_lockable(3, self.dir))
        self.assertTrue(path.exists(), "existence is not the question being asked")

    def test_a_missing_directory_is_refused_rather_than_created(self):
        self.assertFalse(LL.is_lockable(1, self.dir / "nope"))


class CliTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="lanes_"))
        for gpu in (2, 3):
            (self.dir / f"gpu_{gpu}.lock").touch()
        for gpu in (0, 1):
            path = self.dir / f"gpu_{gpu}.lock"
            path.touch()
            os.chmod(path, stat.S_IRUSR)
            self.addCleanup(os.chmod, path, 0o644)

    def run_cli(self, *argv: str, stdin: str = ""):
        out, err = io.StringIO(), io.StringIO()
        old, sys.stdin = sys.stdin, io.StringIO(stdin)
        try:
            with redirect_stdout(out), redirect_stderr(err):
                rc = LL.main([*argv, "--lock-dir", str(self.dir)])
        finally:
            sys.stdin = old
        return rc, out.getvalue().strip(), err.getvalue()

    def test_unusable_lanes_are_dropped_and_named(self):
        rc, out, err = self.run_cli("0", "1", "2", "3")
        self.assertEqual(0, rc)
        self.assertEqual("2,3", out)
        self.assertIn("0,1", err, "a silent drop is how a 2-GPU run becomes a 1-GPU run")

    def test_nothing_usable_exits_nonzero_with_empty_stdout(self):
        """`GEAK_GPU_ALLOWED=$(lockable_lanes.py ...)` must not be able to become
        the empty string: gpu_lock.sh reads an empty allocation as no fence."""
        rc, out, err = self.run_cli("0", "1")
        self.assertEqual(1, rc)
        self.assertEqual("", out)
        self.assertIn("no candidate lane is lockable", err)

    def test_no_candidates_is_its_own_exit_code(self):
        """Distinct from "candidates, none usable" -- the first is a caller bug,
        the second is the host defect, and treating them alike would hide
        whichever one is happening."""
        rc, out, err = self.run_cli("--from-watch-line", "poll 61 idle=[]")
        self.assertEqual(2, rc)
        self.assertEqual("", out)

    def test_a_watcher_line_is_accepted_verbatim(self):
        rc, out, _ = self.run_cli("--from-watch-line",
                                  "2026-08-16T22:45:34+00:00 poll 61 idle=[0 2 3]")
        self.assertEqual(0, rc)
        self.assertEqual("2,3", out)

    def test_stdin_is_accepted(self):
        rc, out, _ = self.run_cli("--stdin", stdin="GPU_IDLE_CONFIRMED: 1 2\n")
        self.assertEqual(0, rc)
        self.assertEqual("2", out)

    def test_the_separator_is_the_one_gpu_lock_expects(self):
        rc, out, _ = self.run_cli("2", "3")
        self.assertEqual("2,3", out,
                         "gpu_lock.sh splits its pool on commas; a space-separated "
                         "list would be read as one bogus id")


class EndToEndTest(unittest.TestCase):
    def test_the_script_runs_as_a_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "gpu_5.lock").touch()
            proc = subprocess.run(
                [sys.executable, str(HERE / "lockable_lanes.py"), "5", "6",
                 "--lock-dir", tmp],
                capture_output=True, text=True, timeout=60)
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertEqual("5,6", proc.stdout.strip(),
                         "a writable temp dir can create gpu_6.lock too")


if __name__ == "__main__":
    unittest.main(verbosity=2)
