#!/usr/bin/env python3
"""Conventions that hold across the whole script directory, not inside one file.

Every module here has its own test file, and every one of those tests is written
from inside the module's own point of view. A convention shared by eight scripts
has no such home, so it goes unchecked in all eight at once -- which is a
different and worse failure than an untested function, because the eight look
individually well covered.

The one checked here came out of the mutation sweep. `noise_floor_stats.py`
survived `sort_keys=True -> False`: its CLI test re-ran one input twice and
compared, which passes whether or not the output is canonical, since CPython
dicts are deterministic in insertion order too. Fixing that module alone would
have been (57) again -- pinning the instance that bit rather than the invariant.
These receipts are meant to be diffed against each other and digested, and four
of the six emitters had nothing checking they still were.

This is a source-level check, and weaker than driving each CLI: it catches a
`sort_keys` that was dropped, not a writer that bypasses `json` entirely. Two of
the six (`noise_floor_stats`, `candidate_policy_scan`) also have behavioural
coverage in their own files.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent

# `json.dumps` calls that are not receipts, keyed by their exact source text and
# each carrying the reason. An entry here is a claim, not a silencer: the tests
# below fail if an exemption stops matching a real call, so a site that later
# becomes a receipt cannot keep its pass.
#
# Keyed by text rather than line number on purpose -- a line number drifts with
# every edit above it and would quietly re-point the exemption at whatever moved
# into its place.
EXEMPT = {
    "json.dumps(sources)":
        "escaping Python strings into a JS source template for mini-racer to "
        "evaluate; the consumer is a parser, not a diff",
    "json.dumps(str(dirname or SCRIPTS_DIR))":
        "same JS-literal escaping -- a single path string, where key order does "
        "not exist",
}


def json_dump_calls(tree: ast.AST):
    """Every `json.dumps(...)` / `json.dump(...)` call, with its line."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ("dump", "dumps") \
                and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == "json":
            yield node


def has_sorted_keys(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "sort_keys":
            return isinstance(kw.value, ast.Constant) and kw.value.value is True
    return False


# Scripts whose tests live under a different name, mapped to the file that
# actually covers them. An entry is a claim the test below verifies: the named
# file must exist and must mention the module, so a script cannot keep its pass
# by pointing at coverage that has moved or never existed.
#
# (62) is the reason this is a mapping and not a skip-list: that finding was a
# completeness claim scoped by a naming convention, which is exactly what
# "every module has a test_<module>.py" is. A module with no matching filename
# is either uncovered or covered elsewhere, and those two must not look alike.
# What counts as a module of this directory. Shell is here because it always
# was one in every sense but this tuple's: `gpu_lock.sh` is the chokepoint every
# GPU command passes through, and `profile_kernel.sh` is invoked by name from
# three role prompts.
MODULE_SUFFIXES = (".py", ".sh")

COVERED_ELSEWHERE = {
    "run_js_tests.py": (
        "test_js_suite.py",
        "the runner is not tested against a fixture but against the guards it "
        "hosts: test_js_suite.py calls RJT.run_body directly (one implementation "
        "of 'did the suite pass' is one more than can be trusted) and carries the "
        "disk-vs-registry inventory equality test",
    ),
}


class EveryScriptIsCoveredTest(unittest.TestCase):
    """(135), as a directory convention rather than as five separate discoveries.

    The JS side already learned this one: three guards were found unexecuted for
    three different reasons, and the fix was to make the guard *inventory* the
    maintained thing rather than to keep finding them one at a time. The Python
    side had the same exposure in the milder form -- a module whose test file
    simply does not exist reads, from any test-count summary, exactly like one
    whose tests all pass.
    """

    def modules(self):
        return sorted(p for p in HERE.iterdir()
                      if p.is_file() and p.suffix in MODULE_SUFFIXES
                      and not p.name.startswith("test_"))

    def test_the_inventory_covers_more_than_one_kind_of_script(self):
        """(55), and the hole this test class actually had. Globbing `*.py` made
        the directory look fully covered while three shell scripts sat outside
        the count -- one of them, `profile_kernel.sh`, invoked by name from three
        role prompts with no test of any kind."""
        names = [p.name for p in self.modules()]
        for suffix in MODULE_SUFFIXES:
            with self.subTest(suffix=suffix):
                self.assertTrue(any(n.endswith(suffix) for n in names),
                                f"no {suffix} module left; drop it from MODULE_SUFFIXES "
                                "rather than leaving the inventory claiming to check a "
                                "class of file that is not there")

    def test_every_script_has_a_test_file_or_a_named_substitute(self):
        for path in self.modules():
            with self.subTest(script=path.name):
                if path.name in COVERED_ELSEWHERE:
                    continue
                expected = f"test_{path.stem}.py"
                self.assertTrue(
                    (HERE / expected).exists(),
                    f"{path.name} has no {expected} and no COVERED_ELSEWHERE "
                    "entry naming where it is tested instead")

    def test_every_substitute_names_a_file_that_covers_the_module(self):
        for name, (where, reason) in sorted(COVERED_ELSEWHERE.items()):
            with self.subTest(script=name):
                self.assertTrue((HERE / name).exists(),
                                "this entry names no script in the directory; delete it")
                own = f"test_{Path(name).stem}.py"
                self.assertFalse((HERE / own).exists(),
                                 f"{own} exists now, so the substitute entry is "
                                 "stale and hides whichever of the two is rotting")
                target = HERE / where
                self.assertTrue(target.exists(), f"{where} does not exist")
                self.assertIn(name.removesuffix(".py"),
                              target.read_text(encoding="utf-8"),
                              f"{where} never mentions {name}; the claim that it covers "
                              "it is not checkable from the file itself")
                self.assertGreater(len(reason), 40,
                                   "an exemption without a stated reason is a silencer")


class CanonicalJsonTest(unittest.TestCase):
    def scripts(self):
        return sorted(p for p in HERE.glob("*.py") if not p.name.startswith("test_"))

    def all_calls(self):
        for path in self.scripts():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for call in json_dump_calls(tree):
                yield path, call, ast.unparse(call.func) + "(" + ", ".join(
                    ast.unparse(a) for a in call.args) + ")"

    def test_every_json_receipt_is_written_with_sorted_keys(self):
        for path, call, text in self.all_calls():
            if text in EXEMPT:
                continue
            with self.subTest(script=path.name, line=call.lineno):
                self.assertTrue(
                    has_sorted_keys(call),
                    f"{path.name}:{call.lineno} writes JSON without sort_keys=True, "
                    "so two runs holding the same data can emit different bytes and "
                    "the receipt stops being comparable by diff or digest. If this "
                    "call is not a receipt, add it to EXEMPT with the reason.")

    def test_the_convention_is_not_vacuous(self):
        """(55). If no script wrote JSON, or every call were exempt, the check
        above would pass silently and read as coverage of a convention nobody
        follows."""
        checked = [f"{p.name}:{c.lineno}" for p, c, t in self.all_calls() if t not in EXEMPT]
        self.assertGreaterEqual(len(checked), 8, checked)
        self.assertGreaterEqual(
            len({p.name for p, c, t in self.all_calls() if t not in EXEMPT}), 6)

    def test_every_exemption_still_matches_a_real_call(self):
        """A stale exemption is how a real emitter gets skipped later under a
        name someone recognises."""
        seen = {t for _, _, t in self.all_calls()}
        for text, reason in sorted(EXEMPT.items()):
            with self.subTest(call=text):
                self.assertIn(text, seen,
                              "this exemption matches no call in the directory; delete it")
                self.assertGreater(len(reason), 30,
                                   "an exemption without a stated reason is a silencer")

    def test_no_exemption_covers_a_call_that_reaches_a_stream(self):
        """The one way this exemption list could be abused: exempting a real
        receipt. A `json.dump(obj, stream)` always emits, so it can never be
        exempt regardless of what the reason says."""
        for path, call, text in self.all_calls():
            if text in EXEMPT:
                with self.subTest(script=path.name, line=call.lineno):
                    self.assertNotEqual("dump", call.func.attr,
                                        "json.dump writes to a stream; it is a receipt "
                                        "by construction and cannot be exempted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
