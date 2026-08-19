#!/usr/bin/env python3
"""Every module in `e2e_workflow/scripts` is named by at least one test file.

(135), ported. The kernel lane learned this the expensive way: three JS guards
were found unexecuted for three *different* reasons, and the lesson was that the
exposure is never "this file", it is "a guard nothing executes" -- so the guard
INVENTORY has to be the maintained thing rather than something rediscovered one
file at a time.

This directory cannot use the kernel lane's rule ("every `x.py` has a
`test_x.py`"), because its tests are named after the *behaviour* they check and
not after the module: `attribute_weights.py` is covered by
`test_attribute_weights_edges.py`, `parse_regime.py` by
`test_parse_regime_sources.py`. That is a fine convention, and it is also
exactly the shape finding (62) warns about -- a completeness claim scoped by a
naming convention, where "no file called test_<module>.py" and "nobody tests
this module" look identical from outside.

So the invariant is stated the weaker but true way: a module must be *mentioned*
somewhere under `tests/`. That catches a module nothing references at all, which
is the failure worth catching; it does not claim the mention amounts to
coverage, and this docstring is the place that says so rather than a reader
having to infer it from a green run.

Two things the first draft of this file got wrong, both worth keeping written
down because both were the same mistake -- inventing a rule and then reading its
violations as defects:

  * The inventory is not `*.py`. `bench_e2e.sh` and `server_teardown.sh` are
    modules of this directory in every sense that matters here, and two of the
    test files exist precisely to cover them.
  * The converse is not "every test names a module in *this* directory".
    `test_roofline_skill.py` legitimately imports
    `knowledge/analysis_skills/roofline/roofline_tools.py`; a test living beside
    the harness it exercises is not misfiled. The converse that is actually true
    -- and still catches the rot worth catching -- is that a test must name some
    artifact of `e2e_workflow/` that still exists.
"""
from __future__ import annotations

import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent
LANE = SCRIPTS.parent

# Extensions that make a file a module of this directory. Shell is here because
# two of them are shipped, staged by `roles/director.md`, and tested -- and a
# `.py`-only inventory would have reported a directory as fully covered while
# silently holding two scripts outside the count.
MODULE_SUFFIXES = (".py", ".sh")


class ModuleCoverageInventoryTest(unittest.TestCase):
    def modules(self) -> list[Path]:
        return sorted(p for p in SCRIPTS.iterdir()
                      if p.is_file() and p.suffix in MODULE_SUFFIXES
                      and not p.name.startswith("test_") and p.name != "__init__.py")

    def corpus(self) -> dict[str, str]:
        return {p.name: p.read_text(encoding="utf-8")
                for p in sorted(TESTS.glob("test_*.py"))
                if p.name != Path(__file__).name}

    def test_the_inventory_is_not_empty(self):
        """(55). An empty module list, or a suffix tuple that stopped matching
        anything, would make the check below pass silently and read as coverage
        of a directory nobody looked at."""
        names = [p.name for p in self.modules()]
        self.assertGreaterEqual(len(names), 8, names)
        for suffix in MODULE_SUFFIXES:
            with self.subTest(suffix=suffix):
                self.assertTrue(any(n.endswith(suffix) for n in names),
                                f"no {suffix} module left; drop the suffix or the "
                                "inventory is claiming to check a class of file that "
                                "is not there")

    def test_every_module_is_named_by_some_test_file(self):
        corpus = self.corpus()
        self.assertGreaterEqual(len(corpus), 8, sorted(corpus))
        for module in self.modules():
            with self.subTest(module=module.name):
                self.assertTrue(
                    any(module.stem in text for text in corpus.values()),
                    f"{module.name} is named by no file under tests/. Either it is "
                    "untested, or its tests moved and nothing says where -- and from "
                    "a passing run those two are indistinguishable.")

    def test_every_test_file_names_something_that_still_exists(self):
        """The converse, and the one that rots quietly: a test kept alive against
        a module that has been deleted or renamed still passes, still counts, and
        defends nothing.

        Scoped to the whole lane rather than to `scripts/`, because a test here
        may legitimately target a shell script, a role prompt, or a skill module
        in a sibling directory. The claim is only that it names *something real*.
        """
        stems = {p.stem for p in LANE.rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts}
        for name, text in sorted(self.corpus().items()):
            with self.subTest(test=name):
                self.assertTrue(
                    any(stem and stem in text for stem in stems),
                    f"{name} names no file under {LANE.name}/ -- whatever it was "
                    "written against has been renamed or deleted, and the test has "
                    "been passing ever since without defending it.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
