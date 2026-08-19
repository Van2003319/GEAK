#!/usr/bin/env python3
"""Mutation sweep over the Python helpers, the way `audit_pin_coverage.py`
swept the JS suite.

The JS side had a hand-written mutation corpus, so the question there was which
*assertions* it exercised. The Python side has 700-odd tests and no corpus at
all, so the question is the prior one: does a test go red when the module is
wrong? A green suite over an unmutated module is compatible with every
assertion in it being vacuous, and several of the JS findings ((82), (84)) were
exactly that shape -- so it is not a hypothetical failure mode in this tree, it
is the one that has actually kept recurring.

Method. For each module, generate mutants by systematic AST rewriting (relational
and boolean operators, arithmetic, `not`, numeric and boolean constants), run
that module's own test file against each, and record the mutants that survive.
A survivor is a change to the module that no test objects to. That is not
automatically a defect in the tests:

  * the mutated expression may be genuinely equivalent (`x < y` and `x <= y`
    where the values cannot be equal, a constant only used as a label),
  * it may be in a branch that is unreachable in practice, or a diagnostic
    string's arithmetic,
  * it may be dead code, in which case the finding is about the module, or
  * the behaviour may really be unchecked.

Only the last two need work. Like the pin auditor this is deliberately NOT a
test and its score is deliberately not gated: a mutation score is a number that
moves for legitimate reasons, and a ratchet on it would become a comment (55).

Isolation. The sweep never edits the working tree. The whole `kernel_workflow`
directory is mirrored into a scratch root once, and mutants are written there.
Files are restored by rewriting their original text rather than by deleting
anything.

This docstring used to add "the tests resolve their paths relative to their own
location, so a mirrored copy is self-contained", and that was wrong. Six test
files reach *above* `kernel_workflow` -- `parents[2] / "exp/..."`,
`parents[2] / "examples/tasks/..."` -- to check that a transcribed constant still
matches the report or task file it was transcribed from. Those siblings were not
mirrored, so all seven provenance tests skipped, and every constant they are the
sole pin for was unkillable for the whole sweep. The sweep then reported those
mutants as "killed only by a test outside this module's own file", which is a
false statement about the working tree, and would have reported SURVIVED for any
constant whose only pin is a provenance test -- a hole presented as a finding.

`_mirror` now symlinks those siblings, and `mirror_skips` reports whatever still
skips before the sweep starts, because a test that cannot run is a test that
cannot kill, and the sweep's own numbers do not say so (54).

Usage:
  python3 kernel_workflow/scripts/mutate_python.py                # every module
  python3 kernel_workflow/scripts/mutate_python.py noise_floor_stats
  python3 kernel_workflow/scripts/mutate_python.py --limit 40     # cap per module
"""
from __future__ import annotations

import ast
import copy
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE.parent                      # kernel_workflow/
SCRATCH = Path("/tmp/pymut")
MIRROR = SCRATCH / "kernel_workflow"

# Modules worth mutating: the ones carrying decision logic a wrong answer would
# propagate into a measurement. Excluded on purpose: `run_js_tests.py` and
# `audit_pin_coverage.py`, tooling that reports rather than decides.
MODULES = [
    "noise_floor_stats",
    "sol_card",
    "source_hash",
    "hip_twin_sync",
    "candidate_policy_scan",
]

# pytest short-summary verdicts meaning "a test objected", matched as PREFIXES.
# pytest-subtests writes the parameters into the verdict token itself --
# `SUBFAILED(i=1) test_x.py::T::test_sub` -- so an equality test against
# "SUBFAILED" drops every subtest failure on the floor. That matters more here
# than anywhere else: several of these suites keep most of their assertions
# inside `subTest`, and a dropped kill is reported as a survivor.
_VERDICTS = ("FAILED", "ERROR", "SUBFAILED")

_CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
        ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}
_BOOL = {ast.And: ast.Or, ast.Or: ast.And}


def _mutations(tree: ast.AST):
    """Yield (description, mutated_tree). One mutation per tree."""
    nodes = list(ast.walk(tree))
    for index, node in enumerate(nodes):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 \
                and type(node.ops[0]) in _CMP:
            yield (index, f"{type(node.ops[0]).__name__} -> "
                          f"{_CMP[type(node.ops[0])].__name__}",
                   "cmp", getattr(node, "lineno", 0))
        elif isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            yield (index, f"{type(node.op).__name__} -> {_BIN[type(node.op)].__name__}",
                   "bin", getattr(node, "lineno", 0))
        elif isinstance(node, ast.BoolOp) and type(node.op) in _BOOL:
            yield (index, f"{type(node.op).__name__} -> {_BOOL[type(node.op)].__name__}",
                   "bool", getattr(node, "lineno", 0))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            yield (index, "drop `not`", "not", getattr(node, "lineno", 0))
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            yield (index, f"{node.value} -> {not node.value}", "boolconst",
                   getattr(node, "lineno", 0))
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            yield (index, f"{node.value!r} -> {node.value + 1!r}", "numconst",
                   getattr(node, "lineno", 0))


def _apply(tree: ast.AST, index: int, kind: str) -> ast.AST:
    clone = copy.deepcopy(tree)
    node = list(ast.walk(clone))[index]
    if kind == "cmp":
        node.ops = [_CMP[type(node.ops[0])]()]
    elif kind == "bin":
        node.op = _BIN[type(node.op)]()
    elif kind == "bool":
        node.op = _BOOL[type(node.op)]()
    elif kind == "not":
        # Replace `not X` with `X` in place by swapping the node's contents for
        # its operand's. Rewriting the parent would need a parent map; this is
        # equivalent and keeps the walk index stable.
        operand = node.operand
        node.__class__ = operand.__class__
        node.__dict__ = operand.__dict__
    elif kind == "boolconst":
        node.value = not node.value
    elif kind == "numconst":
        node.value = node.value + 1
    return ast.fix_missing_locations(clone)


#: Sibling directories of `kernel_workflow` that tests reach into for evidence,
#: symlinked rather than copied: `exp/` is tens of gigabytes, and a copy would
#: also be a second thing to keep current. Every `parents[2]` use in the suite is
#: a read of a report or a task file; nothing writes there, and the mutation
#: itself only ever touches `MIRROR/scripts`.
EVIDENCE_SIBLINGS = ("exp", "examples")


def _mirror() -> None:
    MIRROR.parent.mkdir(parents=True, exist_ok=True)
    for name in EVIDENCE_SIBLINGS:
        real = SOURCE_ROOT.parent / name
        link = SCRATCH / name
        if real.is_dir() and not link.exists():
            link.symlink_to(real)
    for item in SOURCE_ROOT.iterdir():
        if item.name == "__pycache__":
            continue
        target = MIRROR / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__"))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _failures(target: Path | str, stop_early: bool) -> set[str]:
    """The set of failing test ids, so a mutant is judged on a DIFFERENCE.

    Returning a bool from `returncode == 0` was wrong and wrong in the flattering
    direction. The mirrored suite had one deterministic pre-existing failure --
    `ScratchTest`, which cannot hold inside the mirror -- and with `-x` pytest
    stopped there on every single run. Every mutant therefore looked like it had
    upset something, the confirmation pass reported "0 survived the whole suite"
    for three modules in a row, and that number was vacuous: it was measuring the
    baseline failure, not the mutant. A gate whose result is fixed is a comment
    (55), and this one read as an all-clear.

    So: collect ids, and let the caller subtract the ones that were already
    failing. `-x` is kept for the per-module pass, where any failure at all is a
    kill, and dropped for the whole-suite pass, where stopping early would hide
    every test after the first pre-existing failure.
    """
    argv = [sys.executable, "-m", "pytest", str(target), "-q", "--no-header",
            "--tb=no", "-rf", "-p", "no:cacheprovider"]
    if stop_early:
        argv.insert(4, "-x")
    proc = subprocess.run(argv, cwd=str(MIRROR / "scripts"),
                          capture_output=True, text=True, timeout=900)
    return parse_failures(proc.stdout, proc.returncode)


def parse_failures(stdout: str, returncode: int) -> set[str]:
    """Failing node ids from a `-rf` short summary.

    Separate from the subprocess call so it can be tested against real pytest
    output without a nine-minute sweep in the loop -- this parse is where the
    baseline subtraction lives or dies, and a subtraction that silently never
    matches restores exactly the bug it was written to fix.
    """
    ids = set()
    for line in stdout.splitlines():
        parts = line.split()
        # "<VERDICT> <nodeid> - <message>". Both the message and the subtest
        # parameters are dropped: they carry the mutated value, so keeping
        # either would make every mutant's failure look like a distinct failure
        # and nothing would ever match the baseline. Collapsing subtests to
        # their test errs toward calling a mutant a survivor, which is the
        # direction that reports more work rather than less.
        if len(parts) < 2:
            continue
        verdict = next((v for v in _VERDICTS if parts[0].startswith(v)), None)
        if verdict:
            ids.add(f"{verdict} {parts[1]}")
    if not ids and returncode not in (0, 1):
        # A crash, an import error, a collection failure: not a kill, and not a
        # pass either. Named so it cannot be mistaken for either.
        ids = {f"<pytest exited {returncode} with no reported failure>"}
    return ids


def baseline_failures() -> set[str]:
    """Tests already failing on the UNMUTATED mirror.

    Printed by `main` rather than swallowed: a large baseline means the
    confirmation pass is reading a mostly-broken suite, and the reader has to
    know that before trusting any "killed elsewhere" count.
    """
    return _failures(".", stop_early=False)


def null_mutant_failures(src_path, original: str, baseline: set[str]) -> set[str]:
    """Tests that fail on a mutant that changes NOTHING.

    Every mutant this module writes is `ast.unparse(tree)` -- the whole file
    round-tripped through the AST, not a surgical edit to one line. That is what
    makes the mutation semantic rather than textual, and it is also a trap: the
    round trip strips comments, normalises quotes to single, collapses
    multi-line calls, and re-spells f-strings. Any test that reads the module's
    *source text* rather than calling it therefore fails on every single mutant,
    and the sweep credits the kill to whatever the mutation happened to be.

    This is not hypothetical and it was not caught by the sweep's own numbers.
    A lane-parity guard scans a helper for the literal
    `verdict = "needs_fresh_elapsed"`; `ast.unparse` emits single quotes. Every
    one of that module's 96 mutants was scored killed, and 20 of them had no
    behavioural test anywhere -- reported as "killed elsewhere", which reads as
    defended. A tool that cannot fail flatters, and this one was flattering
    about the exact modules whose constants nobody had pinned.

    So: run the null mutant first and fold its failures into the baseline. What
    is left is unparse-blind and can only be killed by meaning. The count is
    returned rather than swallowed because it is a finding in its own right --
    a test that a null mutant kills is a test the whole sweep is blind behind.
    """
    src_path.write_text(ast.unparse(ast.parse(original)), encoding="utf-8")
    try:
        return _failures(".", stop_early=False) - baseline
    except subprocess.TimeoutExpired:
        return {"<null-mutant run timed out; every kill for this module is unverified>"}
    finally:
        src_path.write_text(original, encoding="utf-8")


def sweep(module: str, limit: int | None,
          baseline: set[str] | None = None) -> tuple[int, list[str], list[str], list[str]]:
    src_path = MIRROR / "scripts" / f"{module}.py"
    test_path = MIRROR / "scripts" / f"test_{module}.py"
    if not test_path.exists():
        return 0, [f"no test_{module}.py -- the module is unmutated because it is untested"], [], []
    if baseline is None:
        baseline = set()
    original = src_path.read_text(encoding="utf-8")
    # Anything the null mutant kills is text-sensitive, not meaning-sensitive,
    # and would otherwise be credited with killing every mutant in the file.
    blind = null_mutant_failures(src_path, original, baseline)
    baseline = baseline | blind
    tree = ast.parse(original)
    plans = list(_mutations(tree))
    if limit:
        # Evenly spread rather than truncated, so a cap still samples the whole
        # file instead of only its imports and constants.
        step = max(1, len(plans) // limit)
        plans = plans[::step][:limit]
    survivors = []      # survived its own test file; candidates only
    try:
        for index, label, kind, line in plans:
            try:
                mutated = ast.unparse(_apply(tree, index, kind))
            except Exception as exc:                      # noqa: BLE001
                survivors.append((index, kind, f"{module}.py:{line} [{label}] "
                                               f"UNPARSEABLE: {exc}", None))
                continue
            src_path.write_text(mutated, encoding="utf-8")
            try:
                # `test_path.name`, not the absolute path: pytest prints node
                # ids exactly as the target was spelled, and the baseline was
                # collected from `.`, so an absolute target would produce ids
                # that could never match it and the subtraction would be a
                # no-op in the flattering direction.
                if not (_failures(test_path.name, stop_early=True) - baseline):
                    survivors.append((index, kind, f"{module}.py:{line}  {label}", mutated))
            except subprocess.TimeoutExpired:
                survivors.append((index, kind, f"{module}.py:{line}  {label}  (TIMEOUT -- "
                                               f"possible infinite loop, itself a finding)", None))
            finally:
                src_path.write_text(original, encoding="utf-8")

        # Second pass, and the reason the first pass's number must never be
        # reported on its own. `test_<module>.py` is not the only test that can
        # object: the lane-parity guards cross-check several of these tables
        # against `kernel_lane.js`, and the JS runner checks others. A mutant
        # killed there is defended -- just not by the file named after the
        # module. Reporting the first-pass survivors as findings would
        # manufacture work of exactly the kind (53) says gets the whole report
        # ignored.
        confirmed, elsewhere = [], []
        for index, kind, text, mutated in survivors:
            if mutated is None:
                confirmed.append(text)
                continue
            src_path.write_text(mutated, encoding="utf-8")
            try:
                new = _failures(".", stop_early=False) - baseline
                (elsewhere if new else confirmed).append(text)
            except subprocess.TimeoutExpired:
                confirmed.append(text + "  (TIMEOUT in full suite)")
            finally:
                src_path.write_text(original, encoding="utf-8")
    finally:
        src_path.write_text(original, encoding="utf-8")
    return len(plans), confirmed, elsewhere, sorted(blind)


def parse_skips(stdout: str) -> list[str]:
    """`SKIPPED [n] file.py:line: reason` lines from a `-rs` short summary.

    Reported as `file.py:line` so the count is of distinct skipped tests rather
    than of pytest's grouped `[n]` prefix, which collapses identical reasons.
    """
    skips = []
    for line in stdout.splitlines():
        if line.startswith("SKIPPED"):
            rest = line.split("]", 1)[-1].strip()
            skips.append(rest.split(":", 2)[0] + ":" + rest.split(":", 2)[1]
                         if rest.count(":") >= 2 else rest)
    return skips


def mirror_skips() -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", ".", "-q", "--no-header", "--tb=no",
         "-rs", "-p", "no:cacheprovider"],
        cwd=str(MIRROR / "scripts"), capture_output=True, text=True, timeout=900)
    return parse_skips(proc.stdout)


USAGE = """mutate_python.py -- mutation sweep over the Python helpers.

  mutate_python.py                     every module in MODULES (a LONG sweep:
                                       it runs each module's test file once per
                                       mutant, hundreds of pytest invocations)
  mutate_python.py noise_floor_stats     one module
  mutate_python.py --limit 40          cap mutants per module

This is not a test and its score is not gated; see the module docstring.
"""


def main(argv: list[str]) -> int:
    # `--help` used to fall through the `not a.startswith("--")` filter below and
    # start the full sweep -- the one invocation whose entire purpose is to avoid
    # doing that. Unknown flags get the same treatment rather than being silently
    # ignored: a mistyped `--limt 40` would otherwise run uncapped for an hour.
    if {"-h", "--help"} & set(argv):
        sys.stdout.write(USAGE)
        return 0
    unknown = [a for a in argv if a.startswith("--") and a != "--limit"]
    if unknown:
        sys.stderr.write(f"unknown option(s): {' '.join(unknown)}\n\n{USAGE}")
        return 2
    limit = None
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit") + 1])
        argv = [a for i, a in enumerate(argv)
                if i not in (argv.index("--limit"), argv.index("--limit") + 1)]
    wanted = [a for a in argv if not a.startswith("--")] or MODULES
    _mirror()
    print(f"mirror: {MIRROR}  (the working tree is never mutated)")

    baseline = baseline_failures()
    if baseline:
        # Loud, because every later number is a difference against this set.
        print(f"\nbaseline: {len(baseline)} test(s) ALREADY failing on the unmutated "
              "mirror. Mutants are judged on failures beyond these:")
        for name in sorted(baseline):
            print(f"    pre-existing  {name}")
        print("  If this list is long the confirmation pass is reading a broken suite "
              "and its counts mean little.")
    else:
        print("baseline: the unmutated mirror is green.")

    # A test that cannot run cannot kill. Without this, a constant whose only
    # pin skips in the mirror is reported SURVIVED, which reads as "nothing
    # checks this" when the truth is "this run could not check it".
    skips = mirror_skips()
    if skips:
        print(f"\nUNCHECKED: {len(skips)} test(s) skip on the mirror and can kill "
              "no mutant this run. A survivor pinned only by one of these is a "
              "hole in the sweep, not a hole in the suite:")
        for name in sorted(set(skips)):
            print(f"    skipped  {name}")
    else:
        print("skips: none -- every test in the mirror can kill.")

    total = killed = far = 0
    for module in wanted:
        count, confirmed, elsewhere, blind = sweep(module, limit, baseline)
        total += count
        killed += count - len(confirmed)
        far += len(elsewhere)
        print(f"\n{module}: {count} mutants, {count - len(confirmed)} killed "
              f"({len(elsewhere)} of them only by a test outside test_{module}.py), "
              f"{len(confirmed)} survived the whole suite")
        if blind:
            print(f"    NULL-MUTANT: {len(blind)} test(s) fail on a rewrite of "
                  f"{module}.py that changes no behaviour at all. They read the "
                  "module's source text, so they would have killed every mutant "
                  "here for free; discounted from this module's kills:")
            for line in blind:
                print(f"        text-sensitive  {line}")
        for line in elsewhere:
            print(f"    killed elsewhere  {line}")
        for line in confirmed:
            print(f"    SURVIVED  {line}")
    if total:
        print(f"\ntotal: {killed}/{total} killed ({100.0 * killed / total:.1f}%), "
              f"{far} of those only by a test in another file")
    print("\nA survivor is a question, not a verdict -- see the module docstring "
          "for the four things it can mean.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
