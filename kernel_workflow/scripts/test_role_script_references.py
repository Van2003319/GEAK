#!/usr/bin/env python3
"""Scripts named in role prompts exist, parse, and are runnable as written.

The role prompts under `kernel_workflow/roles/` are the instructions the
unattended agents follow. Nine distinct `scripts/<name>` paths appear across
them -- `gpu_lock.sh` and `gpu_fence_run.sh` nine times each, `profile_kernel.sh`
three times -- and every one of those is a command an agent is told to run at
3am with nobody watching.

Nothing checked that they resolve. A rename in `scripts/` leaves the prompt
naming a file that is not there, and the failure does not look like a broken
pipeline: the roles all have prose fallbacks for "the tool did not work", so a
missing script reads as a soft degradation and the round continues with the
gate quietly skipped. That is (135) once more -- the guard is the prompt, and
nothing executes it.

Deliberately narrow, for the (139) reason. This checks only *path-shaped*
references (`scripts/foo.py`), not every filename mentioned in the prose. The
prompts also name `harness_lib.py`, `meta.json`, `unittest.py` and similar --
some are artifacts inside an eval workspace, some are vendored copies, some are
generic prose. Asserting that every filename-looking token resolves to a repo
file would invent a rule the prompts do not follow and then report its
violations as defects, which is exactly the mistake §53 and §54 each made once.

`scripts/` is resolved against every scripts root in the repo, not against
`kernel_workflow/scripts` alone: `scripts/task_runner.py` legitimately means the
task's own runner, and hard-coding one root would fail a reference that is
correct.
"""
from __future__ import annotations

import ast
import re
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
ROLES = REPO_ROOT / "kernel_workflow" / "roles"

# A reference is path-shaped: some directory component, then `scripts/`, then a
# filename with a script extension. Bare `foo.py` in prose is out of scope.
REFERENCE = re.compile(r"scripts/([A-Za-z0-9_.-]+\.(?:py|sh|js))")


def scripts_roots() -> list[Path]:
    roots = [REPO_ROOT / "kernel_workflow" / "scripts",
             REPO_ROOT / "e2e_workflow" / "scripts"]
    tasks = REPO_ROOT / "examples" / "tasks"
    if tasks.is_dir():
        roots += sorted(p / "scripts" for p in tasks.iterdir() if (p / "scripts").is_dir())
    return [r for r in roots if r.is_dir()]


def prompts() -> list[Path]:
    return sorted(p for p in ROLES.rglob("*.md") if p.is_file())


def references() -> dict[str, list[Path]]:
    """Referenced script name -> the prompts that name it."""
    out: dict[str, list[Path]] = {}
    for prompt in prompts():
        for name in set(REFERENCE.findall(prompt.read_text(encoding="utf-8"))):
            out.setdefault(name, []).append(prompt)
    return out


def resolve(name: str) -> list[Path]:
    return [root / name for root in scripts_roots() if (root / name).is_file()]


class ReferenceInventoryTest(unittest.TestCase):
    def test_the_prompt_set_is_not_empty(self):
        """(55). Every check below iterates the references, so a `roles/` that
        moved or a regex that stopped matching would make this file pass while
        reading exactly like coverage."""
        self.assertTrue(prompts(), f"no role prompts under {ROLES}")

    def test_the_reference_set_is_not_empty(self):
        found = references()
        self.assertGreaterEqual(
            len(found), 6,
            f"only {sorted(found)} matched; the prompts changed how they spell "
            "script paths and this file is now checking almost nothing")

    def test_the_fence_wrappers_are_still_named_by_the_prompts(self):
        """The two references whose disappearance would be silent and serious:
        an agent that stops being told to wrap GPU work does not error, it just
        runs unfenced."""
        found = references()
        for name in ("gpu_lock.sh", "gpu_fence_run.sh"):
            with self.subTest(script=name):
                self.assertIn(name, found,
                              "no role prompt names it any more; either the fence "
                              "moved and this test needs updating, or the roles "
                              "quietly stopped requiring it")


class ReferencesResolveTest(unittest.TestCase):
    def test_every_referenced_script_exists(self):
        for name, where in sorted(references().items()):
            with self.subTest(script=name):
                self.assertTrue(
                    resolve(name),
                    f"{name} is named by {[p.name for p in where]} but exists in no "
                    f"scripts/ root {[str(r.relative_to(REPO_ROOT)) for r in scripts_roots()]}. "
                    "The roles have prose fallbacks for a tool that fails, so this "
                    "does not surface as a broken pipeline -- it surfaces as a gate "
                    "that was skipped.")

    def test_the_resolution_can_fail(self):
        """(55) on the lookup itself: a `resolve` that returned something for any
        string would pass the test above for a prompt full of typos."""
        self.assertEqual([], resolve("no_such_script_hopefully.py"))


class ReferencesAreRunnableTest(unittest.TestCase):
    """Existing is not the same as working.

    A referenced script that no longer parses fails at the same place a missing
    one does -- inside the agent's `|| true`, on a night nobody is reading the
    log. Both are cheap to rule out without running anything.
    """

    def test_every_referenced_python_script_parses(self):
        for name in sorted(references()):
            if not name.endswith(".py"):
                continue
            for path in resolve(name):
                with self.subTest(script=str(path.relative_to(REPO_ROOT))):
                    try:
                        ast.parse(path.read_text(encoding="utf-8"))
                    except SyntaxError as exc:
                        self.fail(f"{path} does not parse: {exc}")

    def test_every_referenced_shell_script_parses(self):
        checked = 0
        for name in sorted(references()):
            if not name.endswith(".sh"):
                continue
            for path in resolve(name):
                with self.subTest(script=str(path.relative_to(REPO_ROOT))):
                    proc = subprocess.run(["bash", "-n", str(path)],
                                          capture_output=True, text=True, timeout=60)
                    self.assertEqual(0, proc.returncode, proc.stderr)
                    checked += 1
        self.assertGreater(checked, 0, "no shell reference was checked; the fence "
                                       "wrappers are shell and must be in this set")

    def test_every_referenced_script_is_readable_and_not_empty(self):
        for name in sorted(references()):
            for path in resolve(name):
                with self.subTest(script=str(path.relative_to(REPO_ROOT))):
                    self.assertGreater(path.stat().st_size, 0,
                                       "an empty file resolves and then does nothing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
