#!/usr/bin/env python3
"""Tests for `deprovisionalize_epoch.py`.

Everything here runs against a COPY of the real file, with the module's `STATS`
pointer redirected. The tool's whole purpose is to edit a load-bearing source,
so a test suite that edited the original to prove it works would be the least
trustworthy possible way to prove it.

The copy is read from the real file each run rather than reproduced as a
fixture: an anchor that stops matching the live source is the failure mode this
file exists to catch, and a fixture would keep passing right through it.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import deprovisionalize_epoch as DE  # noqa: E402
import measure_noise_floor as MNF  # noqa: E402
import noise_floor_stats as QRS  # noqa: E402
import noise_floor_stats as QRS  # noqa: E402

# The epoch under test must be one that is still PROVISIONAL and already has a
# table to overwrite. Derived, not hardcoded: this read "Q" until Q itself was
# deprovisionalised, at which point three tests went red reporting a tool bug
# where there was only a fixture that had aged out. Retiring an epoch is the
# routine event here, so nothing may name one by letter.
MACHINE = sorted(QRS.PROVISIONAL_MACHINES
                 & set(QRS.MEASURED_NOISE_FLOOR_BY_MACHINE))[0]
ROUTES = sorted(QRS.MEASURED_NOISE_FLOOR_BY_MACHINE[MACHINE])
# The epochs that stay provisional after MACHINE is retired. Derived, not the
# literal `[]` this file used to assert: that was only true while Q was the sole
# provisional epoch, and it went red the moment a second box was registered --
# reporting a tool bug where there was only a stale fixture. Retiring one epoch
# must leave the others exactly as they were, which is the real invariant.
REMAINING_PROVISIONAL = sorted(QRS.PROVISIONAL_MACHINES - {MACHINE})
# Deliberately varied, and deliberately not all equal: a table of one repeated
# number is what the provisional table already is, so a rewrite to it would be
# invisible in every assertion below.
FLOORS = [0.0043, 0.0091, 0.0125, 0.0160, 0.0208, 0.0240,
          0.0271, 0.0310, 0.0355, 0.0402, 0.0451]


def row(floor: float) -> dict:
    """A verdict row with a chosen floor and the producer's own key set.

    Built from `floor_from_speedups` rather than typed out, so a key added on
    the producing side appears here too. The first version of this file listed
    four keys from memory and was missing two.
    """
    return {**MNF.floor_from_speedups([1.0, 1.01, 0.99]),
            "n": 8, "floor": floor, "floor_raw": floor, "clamped_to_min": False}


def verdict(routes=None, **over) -> dict:
    routes = routes if routes is not None else {
        route: row(floor) for route, floor in zip(ROUTES, FLOORS)
    }
    out = {
        "ok": True,
        "stage": "sweep",
        # The box the sweep ran on. A verdict with no host, or one taken on
        # another epoch's box, is refused before any floor is looked at.
        "host": QRS.MACHINE_HOSTNAME[MACHINE],
        "host_machine": MACHINE,
        "source_hash": "bc7ea649e9ea3b7e",
        "repeats_requested": 8,
        "repeats_complete": 8,
        "routes": routes,
        "problems": [],
    }
    out.update(over)
    return out


class Harness(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="deprov_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.stats = self.tmp / "noise_floor_stats.py"
        shutil.copyfile(DE.STATS, self.stats)
        self.real = DE.STATS
        DE.STATS = self.stats
        self.addCleanup(self.restore)

    def restore(self):
        DE.STATS = self.real

    def apply(self, v=None, machine=MACHINE, mode="--apply") -> int:
        path = self.tmp / "verdict.json"
        path.write_text(json.dumps(v if v is not None else verdict()), encoding="utf-8")
        return DE.main(["--verdict", str(path), "--machine", machine, mode])


class RefusalTest(Harness):
    """A floor decides what the archive accepts from here on, so the bar to
    install one is the sweep's own bar and not a softer one."""

    def refuses(self, **over):
        before = self.stats.read_bytes()
        self.assertEqual(2, self.apply(verdict(**over)))
        self.assertEqual(before, self.stats.read_bytes(),
                         "a refused verdict still edited the file")

    def test_a_verdict_that_is_not_ok_is_refused(self):
        self.refuses(ok=False, problems=["correctness failed"])

    def test_too_few_repeats_is_refused(self):
        self.refuses(repeats_complete=2)

    def test_a_missing_route_is_refused(self):
        """The dangerous one. A route absent from the table does not error, it
        falls back to DEFAULT_NOISE_FLOOR -- so a ten-route table on an
        eleven-route epoch reads as a quiet box with one stubborn shape."""
        rows = verdict()["routes"]
        rows.pop(ROUTES[0])
        self.refuses(routes=rows)

    def test_an_unknown_route_is_refused(self):
        rows = verdict()["routes"]
        rows["decode_m4096_imaginary"] = {"n": 8, "floor": 0.01}
        self.refuses(routes=rows)

    def test_an_absurd_floor_is_refused(self):
        rows = verdict()["routes"]
        rows[ROUTES[0]] = {"n": 8, "floor": 0.9}
        self.refuses(routes=rows)

    def test_a_floor_below_the_sampler_resolution_is_refused(self):
        rows = verdict()["routes"]
        rows[ROUTES[0]] = {"n": 8, "floor": 0.0}
        self.refuses(routes=rows)

    def test_a_good_verdict_is_not_refused(self):
        """(55). Every test above asserts a refusal, so a `check_verdict` that
        rejected everything would pass all of them."""
        self.assertEqual([], DE.check_verdict(verdict(), MACHINE))


class ApplyTest(Harness):
    def test_the_python_table_becomes_the_measured_one(self):
        self.assertEqual(0, self.apply())
        text = self.stats.read_text(encoding="utf-8")
        self.assertIn('MEASURED_NOISE_FLOOR_BY_MACHINE["Q"] = {', text)
        for route, floor in zip(ROUTES, FLOORS):
            with self.subTest(route=route):
                self.assertIn(f'"{route}": {floor:.4f},', text)
        self.assertNotIn(
            f'MEASURED_NOISE_FLOOR_BY_MACHINE["{MACHINE}"] = {{\n'
            "    route: DEFAULT_NOISE_FLOOR for route", text,
            f"{MACHINE}'s provisional comprehension is still there")
        for other in REMAINING_PROVISIONAL:
            with self.subTest(untouched=other):
                self.assertIn(
                    f'MEASURED_NOISE_FLOOR_BY_MACHINE["{other}"] = {{\n'
                    "    route: DEFAULT_NOISE_FLOOR for route", text,
                    f"retiring {MACHINE} rewrote {other}'s table as well")

    def test_the_python_module_still_imports_and_reports_measured(self):
        """Textual replacement into a live module is only correct if the module
        still runs; this is the assertion that a stray brace would break."""
        self.assertEqual(0, self.apply())
        prog = (
            "import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('qrs', {str(self.stats)!r})\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "print(sorted(m.PROVISIONAL_MACHINES))\n"
            f"print(m.floor_is_measured('decode_m8_up', {MACHINE!r}))\n"
            f"print(m.noise_floor('decode_m8_up', {MACHINE!r}))\n"
        )
        proc = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                              text=True, timeout=120)
        self.assertEqual(0, proc.returncode, proc.stderr)
        sets, measured, floor = proc.stdout.split("\n")[:3]
        self.assertEqual(str(REMAINING_PROVISIONAL), sets,
                         f"{MACHINE} is still in PROVISIONAL_MACHINES, or "
                         "retiring it disturbed another epoch's flag")
        self.assertEqual("True", measured)
        self.assertEqual(FLOORS[ROUTES.index("decode_m8_up")], float(floor))

    def test_the_new_table_carries_its_provenance(self):
        self.assertEqual(0, self.apply())
        text = self.stats.read_text(encoding="utf-8")
        self.assertIn("MEASURED: 8 complete same-variant primed repeats", text)
        self.assertIn("bc7ea649e9ea3b7e", text)
        self.assertIn("tw003", text)

    def test_applying_twice_is_a_no_op(self):
        self.assertEqual(0, self.apply())
        first = self.stats.read_bytes()
        self.assertEqual(0, self.apply())
        self.assertEqual(first, self.stats.read_bytes())

    def test_check_reports_without_writing(self):
        before = self.stats.read_bytes()
        self.assertEqual(1, self.apply(mode="--check"))
        self.assertEqual(before, self.stats.read_bytes())
        self.assertEqual(0, self.apply())
        self.assertEqual(0, self.apply(mode="--check"), "check still reports work "
                                                        "after the work was done")


class AtomicityTest(Harness):
    def test_a_broken_anchor_leaves_the_file_exactly_as_it_found_it(self):
        """The tool makes two edits -- the table and the flag -- and renders the
        whole file before writing any of it. Without that, an anchor that failed
        on the second edit would leave a measured table sitting under a letter
        still listed as PROVISIONAL, which reports every route it just measured
        as unmeasured."""
        self.stats.write_text("PROVISIONAL_MACHINES = {'Q'}\n", encoding="utf-8")
        before = self.stats.read_bytes()
        self.assertEqual(3, self.apply())
        self.assertEqual(before, self.stats.read_bytes())

    def test_an_ambiguous_set_anchor_is_a_refusal_not_a_guess(self):
        text = self.stats.read_text(encoding="utf-8")
        self.stats.write_text(text + '\nPROVISIONAL_MACHINES = {"Q"}\n', encoding="utf-8")
        self.assertEqual(3, self.apply())


class ProseOwnershipTest(Harness):
    """The sentence above the table is retired by the same edit as the table.

    A comment reading "nothing has been measured here" directly above a measured
    table is worse than no comment, because it is the half a reader believes.
    The anchor therefore covers the comment block, not just the numbers.
    """

    def sentences(self) -> list[tuple[Path, str]]:
        """The live provisional sentence, word for word."""
        # Anchored on the epoch header rather than on the hand-written prose
        # that follows it. Each retiring epoch takes its own wording with it,
        # so quoting that prose word-for-word only survives until the next
        # apply. What does not change is that a provisional epoch's block opens
        # with its own header and the word PROVISIONAL.
        head = f"machine {MACHINE} -- {QRS.MACHINE_HOSTNAME[MACHINE]}. PROVISIONAL:"
        return [(self.stats, head)]

    def test_the_live_sources_carry_the_sentences_this_test_is_about(self):
        """(55). If the wording drifts, every assertion below starts passing by
        looking for something that was never there."""
        for path, sentence in self.sentences():
            with self.subTest(file=path.name):
                self.assertIn(sentence, path.read_text(encoding="utf-8"))

    def test_the_scan_sees_them_before_the_edit(self):
        """The other half of (55), on the scan rather than the fixture: a
        `stale_prose` that returned [] for everything would make the clean-exit
        assertion below pass without the prose ever being touched."""
        found = DE.stale_prose(MACHINE)
        self.assertEqual(1, len(found), f"expected the epoch's block, got {found}")

    def test_the_provisional_sentence_is_gone_from_the_file(self):
        self.assertEqual(0, self.apply(), "a clean apply should leave no stale prose")
        for path, sentence in self.sentences():
            with self.subTest(file=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn(sentence, text,
                                 "the table is measured and the comment above it "
                                 "still says there was nothing to measure with")
                self.assertIn("MEASURED: 8 complete", text)

    def test_a_clean_apply_leaves_the_scan_empty(self):
        self.assertEqual(0, self.apply())
        self.assertEqual([], DE.stale_prose(MACHINE))


class StaleProseTest(Harness):
    """Prose the anchors cannot own is reported, never quietly left behind."""

    def test_prose_elsewhere_in_the_file_fails_the_run(self):
        text = self.stats.read_text(encoding="utf-8")
        self.stats.write_text(
            text + f'\n# epoch {MACHINE} is PROVISIONAL until someone measures '
                   f'{QRS.MACHINE_HOSTNAME[MACHINE]}\n',
            encoding="utf-8")
        self.assertEqual(6, self.apply(),
                         "a sentence the anchor does not cover must still stop "
                         "the chain; it is the half a reader believes")
        left = DE.stale_prose(MACHINE)
        self.assertTrue(left)
        self.assertTrue(all(":" in line for line in left), left)

    def test_the_scan_clears_once_that_prose_is_fixed(self):
        self.test_prose_elsewhere_in_the_file_fails_the_run()
        kept = [l for l in self.stats.read_text(encoding="utf-8").splitlines()
                if "until someone measures" not in l]
        self.stats.write_text("\n".join(kept) + "\n", encoding="utf-8")
        self.assertEqual([], DE.stale_prose(MACHINE))

    def test_prose_about_another_epoch_is_not_this_epoch_s_problem(self):
        """The scan is per-machine on purpose. P is measured and Q is not; a
        scan that fired on any PROVISIONAL mention anywhere would report P's
        history as Q's stale prose and never clear."""
        text = self.stats.read_text(encoding="utf-8")
        self.stats.write_text(text + '\n# epoch P was PROVISIONAL once\n',
                              encoding="utf-8")
        self.assertEqual(0, self.apply())

    def test_the_scan_ignores_code_and_reads_only_comments(self):
        """`PROVISIONAL_MACHINES = {"Q"}` is code, and after the edit it is gone
        anyway; a scan that counted it would never clear and the exit code would
        be permanently 6, which is the same as having no signal."""
        self.stats.write_text('PROVISIONAL_MACHINES = {"Q"}\n', encoding="utf-8")
        self.assertEqual([], DE.stale_prose(MACHINE))


class ProducerContractTest(Harness):
    """The verdict is a contract between two files, and `verdict()` above is a
    fixture of one side of it.

    A fixture cannot notice the producer changing its keys. If
    `measure_noise_floor` starts writing `complete_repeats`, or moves the floor
    one level deeper, every test in this file keeps passing and the tool refuses
    the real sweep at whatever hour the GPU finally freed -- which is the one
    moment the refusal costs the most.
    """

    def produced(self, repeats: int = 8) -> dict:
        """A verdict assembled by the producer's own code, not by hand.

        The samples are synthetic -- `sweep()` needs a GPU -- but the route
        names come from `reference_routes()`, the rows go through `collect()`,
        and the table comes out of `build_table()`, so the shape is the
        producer's shape rather than this file's memory of it.
        """
        routes = sorted(MNF.reference_routes())
        rows_per_repeat = [
            [{"test_case_id": route, "speedup": 1.0 + 0.01 * ((i + j) % 5)}
             for j, route in enumerate(routes)]
            for i in range(repeats)
        ]
        by_route, problems = MNF.collect(rows_per_repeat)
        self.assertEqual([], problems, "the synthetic rows do not satisfy collect()")
        return {
            "ok": True,
            "stage": "sweep",
            "host": QRS.MACHINE_HOSTNAME[MACHINE],
            "host_machine": MACHINE,
            "source_hash": "bc7ea649e9ea3b7e",
            "repeats_requested": repeats,
            "repeats_complete": repeats,
            "routes": MNF.build_table(by_route),
            "problems": [],
        }

    def test_the_hand_fixture_has_the_producer_s_keys(self):
        self.assertEqual(sorted(self.produced()), sorted(verdict()))

    def test_the_hand_fixture_has_the_producer_s_row_keys(self):
        mine = verdict()["routes"]
        theirs = self.produced()["routes"]
        self.assertEqual(sorted(mine), sorted(theirs), "route sets differ")
        for route in sorted(theirs):
            with self.subTest(route=route):
                self.assertEqual(sorted(theirs[route]), sorted(mine[route]))

    def test_a_producer_built_verdict_installs(self):
        self.assertEqual(0, self.apply(self.produced()))
        text = self.stats.read_text(encoding="utf-8")
        for route in sorted(MNF.reference_routes()):
            with self.subTest(route=route):
                self.assertIn(f'"{route}": ', text)

    def test_the_producer_s_own_early_returns_are_refused(self):
        """`sweep()` bails before it has a table at all when correctness fails
        or the source hash moves. Those dicts have no `routes` and no
        `repeats_complete`, and they are exactly what an unattended chain pipes
        into `--apply` on a bad night."""
        for stage in ("correctness", "identity"):
            with self.subTest(stage=stage):
                bail = {"ok": False, "stage": stage, "problems": ["..."]}
                before = self.stats.read_bytes()
                self.assertEqual(2, self.apply(bail))
                self.assertEqual(before, self.stats.read_bytes())


class RendererReuseTest(unittest.TestCase):
    def test_the_tables_are_rendered_by_measure_noise_floor(self):
        """Not a style point. Two renderers for the same numbers drift, and the
        drift is invisible: both sides print plausible floats."""
        src = (HERE / "deprovisionalize_epoch.py").read_text(encoding="utf-8")
        self.assertIn("MNF.render_python(", src)
        self.assertNotIn("def render_python", src, "the tool grew its own renderer")


if __name__ == "__main__":
    unittest.main(verbosity=2)
