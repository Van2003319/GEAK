#!/usr/bin/env python3
"""Tests for `register_epoch.py`.

Like `test_deprovisionalize_epoch.py`, everything here runs against a COPY of
the real `noise_floor_stats.py`, taken fresh from disk on every run rather than
frozen into a fixture. Both tools exist to edit that one load-bearing source,
and the failure they are built to prevent is an anchor that stopped matching it.
A fixture would keep passing straight through exactly that.

The epoch letter is DERIVED, never named. Consuming letters is the routine
event here -- nine restores in two and a half days -- so a test that said "Y"
would go red on the tenth for no reason but its own age, reporting a tool bug
where there is only a stale fixture. That is the mistake this suite's sibling
already made twice and fixed twice.

The registration is driven through the CLI in a subprocess. It is not a
convenience: the tool finishes by re-importing `noise_floor_stats` to check its
own edit took, and doing that in-process would leave the test session holding a
module object built from a temporary copy, which every other test in this
directory imports for real.
"""
from __future__ import annotations

import json
import shutil
import string
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import deprovisionalize_epoch as DE  # noqa: E402
import noise_floor_stats as QRS  # noqa: E402

SCRIPT = HERE / "register_epoch.py"
STATS = HERE / "noise_floor_stats.py"

# A letter no epoch has used yet. Taken from the end of the alphabet because
# entries are appended in epoch order and the file is read top to bottom by
# anyone debugging it, so a new epoch reading "Z" beside a retired "L" is less
# confusing than the reverse.
UNUSED = next(c for c in reversed(string.ascii_uppercase)
              if c not in QRS.MACHINE_HOSTNAME
              and c not in QRS.MEASURED_NOISE_FLOOR_BY_MACHINE)
# A host that is ALREADY registered, so the append-order property has something
# to be true about. `machine_for_host` resolves to the NEWEST match, and a box
# carrying two epochs is not hypothetical: tw008 carried both P and R.
REUSED_HOST = next(h for h in QRS.MACHINE_HOSTNAME.values() if h)
NEW_HOST = "tw999"


def facts(stats_path: Path, host: str) -> dict:
    """What the edited module actually says, read by importing it.

    In a subprocess, and reported as JSON, because the question being asked is
    "what would a fresh process see", and this test process has the real module
    imported already.
    """
    prog = (
        "import json,sys;"
        f"sys.path.insert(0,{str(stats_path.parent)!r});"
        "import noise_floor_stats as Q;"
        "print(json.dumps({"
        "'current': Q.CURRENT_MACHINE,"
        f"'resolves': Q.machine_for_host({host!r}),"
        "'provisional': sorted(Q.PROVISIONAL_MACHINES),"
        "'hostnames': Q.MACHINE_HOSTNAME,"
        "'tables': {k: sorted(v) for k, v in Q.MEASURED_NOISE_FLOOR_BY_MACHINE.items()},"
        "'floors': Q.MEASURED_NOISE_FLOOR_BY_MACHINE[Q.CURRENT_MACHINE],"
        "'default': Q.DEFAULT_NOISE_FLOOR,"
        "'measured': Q.floor_is_measured(sorted(Q.MEASURED_NOISE_FLOOR)[0]),"
        "}, sort_keys=True))")
    proc = subprocess.run([sys.executable, "-c", prog],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"the edited module does not import:\n{proc.stderr}")
    return json.loads(proc.stdout)


class RegisterEpochTest(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="regepoch_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.stats = self.tmp / "noise_floor_stats.py"
        shutil.copyfile(STATS, self.stats)
        self.before = self.stats.read_text(encoding="utf-8")

    def run_tool(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--file", str(self.stats), *args],
            capture_output=True, text=True)

    def register(self, letter: str = UNUSED, host: str = NEW_HOST, *extra: str):
        proc = self.run_tool("--letter", letter, "--host", host, *extra)
        self.assertEqual(0, proc.returncode,
                         f"registration failed:\n{proc.stdout}\n{proc.stderr}")
        return proc

    # --- the anchors, which is the whole point -----------------------------

    def test_it_registers_against_the_live_file_as_it_stands_today(self):
        """The one test that would catch `noise_floor_stats.py` being
        restructured. Every other assertion here is about behaviour on a file
        this tool already understands; this one is about whether it still
        understands the real one. It runs on a copy taken from disk this run,
        so it fails the day an anchor stops matching rather than the day a
        machine changes at 3am with nobody watching."""
        self.register()
        got = facts(self.stats, NEW_HOST)
        self.assertEqual(UNUSED, got["current"])
        self.assertEqual(UNUSED, got["resolves"])
        self.assertEqual(NEW_HOST, got["hostnames"][UNUSED])

    def test_the_new_epoch_is_provisional_and_reads_as_unmeasured(self):
        """A provisional table is shaped exactly like a measured one -- that is
        what keeps every caller simple, and exactly why the epoch must also be
        in PROVISIONAL_MACHINES. Shape alone cannot distinguish them, so a
        registration that produced the table without the flag would present
        fail-closed defaults as if a GPU had been measured."""
        self.register()
        got = facts(self.stats, NEW_HOST)
        self.assertIn(UNUSED, got["provisional"])
        self.assertFalse(got["measured"])
        self.assertEqual({got["default"]}, set(got["floors"].values()))

    def test_the_table_carries_the_reference_epochs_routes_exactly(self):
        """`check_measurement_frame` and every gate index this table by route
        name. A registration that invented, dropped or renamed one would hand
        back DEFAULT for a route that silently has no entry."""
        before = facts(STATS, NEW_HOST)
        self.register()
        after = facts(self.stats, NEW_HOST)
        self.assertEqual(after["tables"][before["current"]],
                         after["tables"][UNUSED])

    def test_registering_a_box_that_already_has_an_epoch_supersedes_it(self):
        """Finding (126). A re-used box gets a NEW letter, and `machine_for_host`
        must resolve it to that one -- resolving to the retired letter would
        reinstate floors measured in a different container. The property is
        carried by APPEND ORDER, which no assertion about the file's text would
        notice breaking."""
        was = QRS.machine_for_host(REUSED_HOST)
        self.assertIsNotNone(was)
        self.register(host=REUSED_HOST)
        got = facts(self.stats, REUSED_HOST)
        self.assertEqual(UNUSED, got["resolves"])
        self.assertNotEqual(was, got["resolves"])
        self.assertEqual(REUSED_HOST, got["hostnames"][was],
                         "the retired letter must keep its host: it is how a "
                         "past round's floors stay attributable")

    def test_other_provisional_epochs_survive(self):
        """Rewriting a set is the easy way to drop its other members, and the
        loss would read as those epochs having been measured."""
        self.register()
        got = facts(self.stats, NEW_HOST)
        for letter in sorted(QRS.PROVISIONAL_MACHINES):
            self.assertIn(letter, got["provisional"])

    # --- deprovisionalize_epoch.py has to be able to undo this -------------

    def test_what_it_writes_is_what_the_retiring_tool_anchors_on(self):
        """These two tools are a pair: this one writes the provisional block,
        `deprovisionalize_epoch.py --apply` replaces the comment AND the table
        in one edit once floors exist. If the block it writes were shaped even
        slightly differently -- a blank line between comment and assignment,
        say -- the retiring tool would insert its provenance above a comment
        still claiming the epoch is unmeasured, and both would be in the file
        at once."""
        self.register()
        text = self.stats.read_text(encoding="utf-8")
        blocks = {m.group("m"): m for m in DE.PY_TABLE.finditer(text)}
        self.assertIn(UNUSED, blocks, "the retiring tool cannot find this table")
        comment = blocks[UNUSED].group("comment")
        self.assertIn("PROVISIONAL", comment,
                      "the comment above the table must say what the table is, "
                      "or retiring the epoch leaves the claim behind")

    def test_the_provisional_claim_is_visible_to_the_stale_prose_backstop(self):
        """`stale_prose` is what stops a retirement from silently leaving prose
        that still calls the epoch unmeasured. It only sees a comment block
        containing BOTH the word PROVISIONAL and the letter, so a block that
        named the host but not the letter would be invisible to it -- which is
        a real distinction the file makes elsewhere and gets right for a
        different reason."""
        self.register()
        real = DE.STATS
        DE.STATS = self.stats
        try:
            self.assertTrue(DE.stale_prose(UNUSED),
                            "the block this tool wrote is not recognised as a "
                            "provisional claim about this epoch")
        finally:
            DE.STATS = real

    # --- refusals ----------------------------------------------------------

    def test_it_refuses_a_letter_that_is_already_in_use(self):
        """Overwriting a letter is how one box's floors end up judging
        another's timings, and a tool that is run after every restore will be
        run twice after some of them."""
        taken = sorted(QRS.MACHINE_HOSTNAME)[0]
        proc = self.run_tool("--letter", taken, "--host", NEW_HOST)
        self.assertNotEqual(0, proc.returncode)
        self.assertIn("already appears", proc.stderr)
        self.assertEqual(self.before, self.stats.read_text(encoding="utf-8"))

    def test_registering_twice_is_refused_rather_than_duplicated(self):
        self.register()
        after_first = self.stats.read_text(encoding="utf-8")
        proc = self.run_tool("--letter", UNUSED, "--host", NEW_HOST)
        self.assertNotEqual(0, proc.returncode)
        self.assertEqual(after_first, self.stats.read_text(encoding="utf-8"))

    def test_it_refuses_something_that_is_not_an_epoch_letter(self):
        for bad in ("YY", "3", "", "y1"):
            with self.subTest(letter=bad):
                proc = self.run_tool("--letter", bad, "--host", NEW_HOST)
                self.assertNotEqual(0, proc.returncode)
                self.assertEqual(self.before,
                                 self.stats.read_text(encoding="utf-8"))

    def test_a_lowercase_letter_is_accepted_and_normalised(self):
        """The letter is typed by hand at whatever hour a machine changed."""
        self.register(letter=UNUSED.lower())
        self.assertEqual(UNUSED, facts(self.stats, NEW_HOST)["current"])

    def test_it_refuses_a_file_that_is_not_there(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--file", str(self.tmp / "nope.py"),
             "--letter", UNUSED, "--host", NEW_HOST],
            capture_output=True, text=True)
        self.assertNotEqual(0, proc.returncode)
        self.assertIn("does not exist", proc.stderr)

    def test_a_missing_anchor_is_a_refusal_and_not_a_partial_edit(self):
        """The failure mode this tool was written for: an edit that does not
        happen. If one anchor has moved, the run must stop with the file as it
        was -- three of four edits applied is worse than none, because the
        module still imports and the frame check still answers."""
        self.stats.write_text(
            self.before.replace("PROVISIONAL_MACHINES = ", "PROVISIONAL_MACHINES_RENAMED = "),
            encoding="utf-8")
        mutated = self.stats.read_text(encoding="utf-8")
        proc = self.run_tool("--letter", UNUSED, "--host", NEW_HOST)
        self.assertNotEqual(0, proc.returncode)
        self.assertIn("PROVISIONAL_MACHINES", proc.stderr)
        self.assertEqual(mutated, self.stats.read_text(encoding="utf-8"))

    def test_an_edit_that_does_not_take_is_reverted(self):
        """Text changing is not the same as the module saying something new.
        Here a second assignment further down wins at import time, so all four
        edits apply cleanly and the registration is still false. The tool has to
        find that by asking the module, which is the entire reason it re-imports
        rather than trusting its own replace count."""
        # Any registered letter EXCEPT the live one. Naming the live one appends a line identical
        # to the existing assignment, and the anchor-count guard refuses first -- a correct
        # refusal, but not the one this test is about. This read `sorted(...)[0]` until the
        # alphabet wrapped past Z and epoch A became both the first letter and the current one.
        stale = sorted(set(QRS.MACHINE_HOSTNAME) - {QRS.CURRENT_MACHINE})[0]
        self.stats.write_text(self.before + f'\nCURRENT_MACHINE = "{stale}"\n',
                              encoding="utf-8")
        mutated = self.stats.read_text(encoding="utf-8")
        proc = self.run_tool("--letter", UNUSED, "--host", NEW_HOST)
        self.assertNotEqual(0, proc.returncode)
        self.assertIn("REVERTED", proc.stderr)
        self.assertEqual(mutated, self.stats.read_text(encoding="utf-8"),
                         "a refused registration must leave the file untouched")

    # --- dry run -----------------------------------------------------------

    def test_dry_run_shows_the_diff_and_writes_nothing(self):
        proc = self.run_tool("--letter", UNUSED, "--host", NEW_HOST, "--dry-run")
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn(f'"{UNUSED}": "{NEW_HOST}"', proc.stdout)
        self.assertIn(f'+CURRENT_MACHINE = "{UNUSED}"', proc.stdout)
        self.assertEqual(self.before, self.stats.read_text(encoding="utf-8"))

    def test_the_note_reaches_the_file_so_an_epoch_says_why_it_exists(self):
        """Eight of the thirteen entries in the host table carry a clause
        saying which restore they came from. That is the only record tying a
        letter to an event, and it is written once, here."""
        self.register(UNUSED, NEW_HOST, "--note", "wave 42 restore")
        line = next(l for l in self.stats.read_text(encoding="utf-8").splitlines()
                    if f'"{UNUSED}": "{NEW_HOST}"' in l)
        self.assertIn("wave 42 restore", line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
