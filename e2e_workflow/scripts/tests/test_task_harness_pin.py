#!/usr/bin/env python3
"""The `PROVENANCE.frozen_harness` pin, executed rather than merely written down.

`examples/tasks/dense_bf16_gemm_fused/` is the one task tree that carries its own
copy of a lane module: `harness_lib.py`, byte-identical to
`e2e_workflow/scripts/harness_lib.py` today. The copy is deliberate -- the task's
`PROVENANCE.json` pins its sha256 -- and the reason is in that file's own
`repin_note`: the write probe that catches a kernel which allocates its output and
returns without writing it used to live only in the lane copy, while the runner
imported the task copy, so the gate was carried by an import-order accident. It
was fixed by moving the probe into the pinned file and by re-ordering the two
`sys.path.insert(0, ...)` calls.

Nothing checked any of that. Searching the tree for `frozen_harness` outside the
progress log returns exactly one hit, and it is a comment. The pin is a claim
about four things -- the file exists, its bytes still hash to the recorded value,
it is the copy the runner actually imports, and it contains the probe -- and all
four were being asserted by prose. This is (135) in its plainest form: not an
untested function, a guard nothing executes. The distinguishing symptom is that
it reads *better* than an untested function, because someone clearly thought
about it.

Scope note, and the (139) lesson from the module inventory: the task tree has no
unit-test convention to extend. Its `tests/` directory is empty and `knn`'s
`test_knn.py` is a GPU correctness fixture, not a unit test. So this does not
impose one -- the claim being checked is a cross-tree consistency claim, the lane
owns `harness_lib.py`, and the check lives with the lane's other tests.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TASKS = REPO_ROOT / "examples" / "tasks"
LANE = REPO_ROOT / "e2e_workflow" / "scripts"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_dirs() -> list[Path]:
    return sorted(p for p in TASKS.iterdir() if p.is_dir() and (p / "config.yaml").exists())


def provenance(task: Path) -> dict | None:
    path = task / "PROVENANCE.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pinned(task: Path) -> dict | None:
    prov = provenance(task)
    return (prov or {}).get("frozen_harness")


def pins() -> list[tuple[Path, dict]]:
    return [(t, p) for t in task_dirs() if (p := pinned(t))]


class PinIsExecutedTest(unittest.TestCase):
    def test_at_least_one_task_declares_a_pin(self):
        """(55). Every assertion below iterates the pins, so an empty list would
        make this whole file pass while checking nothing at all."""
        self.assertTrue(pins(), f"no task under {TASKS} declares frozen_harness; "
                                "if the pin was retired, delete this file rather "
                                "than leaving it reading as coverage")

    def test_every_pin_names_a_file_that_exists(self):
        for task, pin in pins():
            with self.subTest(task=task.name):
                self.assertIn("path", pin, "a pin with no path names nothing")
                self.assertTrue((task / pin["path"]).exists(),
                                f"{pin['path']} is pinned but not present")

    def test_every_pin_matches_the_bytes_on_disk(self):
        """The pin's whole job. An edit to the harness that forgets to re-pin
        leaves a hash naming a file that no longer exists in that form, and the
        next reader trusts it."""
        for task, pin in pins():
            with self.subTest(task=task.name):
                path = task / pin["path"]
                self.assertEqual(
                    pin["sha256"], sha256(path),
                    f"{path.relative_to(REPO_ROOT)} no longer hashes to its pin. "
                    "Either the edit was unintended, or PROVENANCE.frozen_harness "
                    "needs re-pinning with a repin_note saying what changed and why.")

    def test_the_hash_check_can_fail(self):
        """(55) again, on the comparison itself: `sha256(x) == sha256(x)` would
        pass this file forever."""
        task, pin = pins()[0]
        original = (task / pin["path"]).read_bytes()
        mutated = original + b"\n# one byte of drift\n"
        self.assertNotEqual(pin["sha256"], hashlib.sha256(mutated).hexdigest())


class ShadowInventoryTest(unittest.TestCase):
    """A task file that shadows a lane module is either pinned or an accident.

    Shadowing is not visible from either copy. `task_runner.py` puts the task
    directory ahead of the lane on `sys.path`, so a stray same-named file wins
    silently -- and the lane's tests, which load the lane copy by path, keep
    passing against a file the run never imports.
    """

    def shadows(self) -> list[tuple[Path, Path]]:
        out = []
        for task in task_dirs():
            for path in sorted(task.glob("*.py")) + sorted(task.glob("scripts/*.py")):
                if (LANE / path.name).exists():
                    out.append((task, path))
        return out

    def test_the_shadow_inventory_is_not_empty(self):
        """If this ever fails honestly -- no task carries a lane module any more
        -- the right response is to delete this class, not to keep it passing
        over nothing."""
        self.assertTrue(self.shadows(),
                        "no task shadows a lane module; drop ShadowInventoryTest")

    def test_every_shadow_is_pinned_by_its_task(self):
        for task, path in self.shadows():
            with self.subTest(shadow=str(path.relative_to(TASKS))):
                pin = pinned(task)
                self.assertIsNotNone(
                    pin, f"{path.name} shadows {LANE.name}/{path.name} but "
                         f"{task.name} declares no frozen_harness; an unpinned "
                         "shadow is how the import-order accident happened")
                self.assertEqual(
                    path, task / pin["path"],
                    f"{task.name} pins {pin['path']} but also shadows {path.name}; "
                    "the second shadow is unaccounted for")


class ImportOrderTest(unittest.TestCase):
    """The pinned copy is the one the runner imports.

    Asserted by running the runner's own path setup rather than by re-deriving
    it here: the bug being guarded against was two `sys.path.insert(0, ...)`
    calls in the wrong order, and a test that re-implements that ordering would
    reproduce whichever order its author believed in.
    """

    MARKER = "\nimport harness_lib"

    def runners(self) -> list[Path]:
        """Runners that import the harness -- not every runner in the tree.

        (139), a third time. `knn` is an older, self-contained task: its runner
        carries its own reference implementation and never touches harness_lib,
        so scoping this class by "has a task_runner.py" made a task that is not
        making the claim fail the check for the claim. The inventory has to be
        defined by the property being checked, not by directory membership.
        """
        out = []
        for task in task_dirs():
            runner = task / "scripts" / "task_runner.py"
            if runner.exists() and self.MARKER in runner.read_text(encoding="utf-8"):
                out.append(runner)
        return out

    def prefix(self, runner: Path) -> str:
        """The runner's source up to (not including) its harness import."""
        src = runner.read_text(encoding="utf-8")
        self.assertIn(self.MARKER, src, f"{runner.name} does not import harness_lib")
        return src.split(self.MARKER, 1)[0]

    def resolve(self, runner: Path) -> str:
        """Origin of `harness_lib` as that runner's path setup leaves it.

        Run in a subprocess so the test's own `sys.path` is untouched, with a
        stub `torch` so a CPU box and a GPU box give the same answer -- the same
        injection `test_harness_lib.py` uses.
        """
        prog = textwrap.dedent("""
            import sys, types, importlib.util
            sys.modules["torch"] = types.ModuleType("torch")
            __file__ = {runner!r}
        """).format(runner=str(runner)) + self.prefix(runner) + textwrap.dedent("""
            spec = importlib.util.find_spec("harness_lib")
            print(spec.origin if spec else "")
        """)
        proc = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                              text=True, timeout=120, cwd=str(REPO_ROOT))
        self.assertEqual(0, proc.returncode, proc.stdout + proc.stderr)
        return proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""

    def test_every_runner_resolves_harness_lib_somewhere(self):
        runners = self.runners()
        self.assertTrue(runners, "no task runners found")
        for runner in runners:
            with self.subTest(task=runner.parents[1].name):
                self.assertTrue(self.resolve(runner),
                                "the runner's path setup cannot find harness_lib at all")

    def test_a_pinned_task_imports_its_own_pinned_copy(self):
        for task, pin in pins():
            runner = task / "scripts" / "task_runner.py"
            with self.subTest(task=task.name):
                self.assertTrue(runner.exists(), f"{task.name} has no scripts/task_runner.py")
                self.assertEqual(
                    str((task / pin["path"]).resolve()), self.resolve(runner),
                    "the runner imports a harness other than the pinned one, so the "
                    "pin describes a file that does not run. This is the exact "
                    "failure PROVENANCE.repin_note records as already having "
                    "happened once.")

    def test_an_unpinned_task_falls_through_to_the_lane_copy(self):
        """The other half, and the reason the pinned case is not vacuous: with no
        shadow present the same path setup must land on the lane module."""
        checked = 0
        for runner in self.runners():
            task = runner.parents[1]
            if pinned(task):
                continue
            with self.subTest(task=task.name):
                self.assertEqual(str((LANE / "harness_lib.py").resolve()),
                                 self.resolve(runner))
                checked += 1
        if not checked:
            self.skipTest("every task with a runner is pinned")


class ProbeClaimTest(unittest.TestCase):
    """`contains_write_probe` and `write_probe` are claims about the pinned file.

    They are the reason the pin exists, so a pin whose hash matches a harness
    with no probe in it would be a correct pin on the wrong file -- which is
    precisely what the first pin was.
    """

    def test_a_task_claiming_a_write_probe_has_one(self):
        for task, pin in pins():
            if not pin.get("contains_write_probe"):
                continue
            with self.subTest(task=task.name):
                src = (task / pin["path"]).read_text(encoding="utf-8")
                self.assertIn("def assert_writes_output(", src,
                              "the pin claims a write probe and the file has none")

    def test_a_two_sentinel_claim_means_two_distinct_sentinels(self):
        """One sentinel was proven insufficient: the oracle output for
        `prefill_m512_up` contains an element exactly equal to POISON_A, so
        "still equals the sentinel" does not imply "never written". Two
        sentinels that differ in sign and exponent is the fix, and the claim in
        PROVENANCE has to keep meaning that."""
        for task, pin in pins():
            if pin.get("write_probe") != "two_sentinel":
                continue
            with self.subTest(task=task.name):
                path = task / pin["path"]
                mod = self.load(path)
                a, b = getattr(mod, "POISON_A"), getattr(mod, "POISON_B")
                self.assertNotEqual(a, b, "two names, one value")
                self.assertLess(a * b, 0, "the sentinels share a sign")
                sig = mod.assert_writes_output.__defaults__
                self.assertIn(a, sig, "POISON_A is not the probe's default")
                self.assertIn(b, sig, "POISON_B is not the probe's default")

    @staticmethod
    def load(path: Path):
        import importlib.util
        import types
        sys.modules.setdefault("torch", types.ModuleType("torch"))
        spec = importlib.util.spec_from_file_location("pinned_harness", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


if __name__ == "__main__":
    unittest.main(verbosity=2)
