#!/usr/bin/env python3
"""Which source-pinning assertions in the JS guard suites are load-bearing?

Finding (84) counted 47 source-matching assertions in the JS suite and observed
that ~42 of them pin an *expression* -- a comparison, a derivation, a guard --
by matching its text. That is finding (82)'s shape: the assertion is green while
the expression is wrong and turns red when you fix it. The note deferred the
sweep because "rewrite 42 assertions" without a per-site theory of the wrong
form produces motion rather than coverage.

This is that theory, computed instead of guessed. A pinning assertion is
load-bearing exactly when SOME mutation of the source it pins would flip it --
because that is the same thing as saying the assertion distinguishes the source
from a broken one. `test_js_suite.py` already carries the mutation corpus; this
script crosses the two and reports the pins no mutant in that corpus can flip.

An unflippable pin is not automatically wrong. It may be:
  * a literal that genuinely is the thing under test (a schema field name, a
    script path, an agent label),
  * a *negated* pin -- `ok(!/wrong form/...)`. These used to be reported as
    unexercisable by construction, on the argument that a corpus which removes
    correct forms can never introduce a forbidden one. That argument was wrong
    about its own corpus: a mutant may REPLACE a correct line with the forbidden
    form, and one now does, which is why `REQUIRE_ATOMIC_MANIFEST` moved out of
    this bucket and into the load-bearing one. So negated pins are now run
    through the corpus like every other pin, and what is reported here is the
    measured statement "no mutant introduces this" -- a fact about the corpus,
    fixable by writing an injection mutant -- rather than a claim about the
    method. A converse-absence pin is still the stronger form (84) recommended,
  * backed by an *executed* check elsewhere in the same suite, in which case the
    pin is documentation and the execution is the defense, or
  * a real (82) hole.
Only the last needs work, and the point of this script is that the four are
told apart by reading the named lines rather than by re-reading all 104.

The unit audited is the *clause*, not the assertion: these are mostly written
`ok(/a/.test(src) && (src.match(/b/g) || []).length === 3 && !/c/.test(src))`,
and an audit that reads only the leading conjunct sees about two thirds of them.
A clause with an unexercised pattern whose sibling IS exercised is reported
separately -- the assertion still fails on a real defect, so it is not a hole,
but that conjunct is not the reason it fails and should be read as a literal or
as redundancy. Counting clauses (`.match(/re/g).length === N`) are scored on the
count rather than on presence, because "declared at every site" is exactly the
claim a presence match cannot make: drop one of three declarations and presence
is still true.

Deliberately NOT a test. It reports a number that legitimately moves as the
suite grows, and a gate on that number would either be a ratchet nobody can
lower for a good reason or a comment (55). Run it, read it, act on it.

Every suite in `SUITES` is audited against the mutant corpus that defends it.

Usage: python3 kernel_workflow/scripts/audit_pin_coverage.py [--verbose]
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANE = HERE.parent / "kernel_lane.js"
WORKFLOW = HERE.parent / "kernel_workflow.js"
SUITE_JS = HERE / "test_qd_archive.js"
FLOOR_JS = HERE / "test_candidate_floor.js"
SUITE_PY = HERE / "test_js_suite.py"

# Which mutant corpus defends which suite. `test_candidate_floor.js` is here
# because auditing only the biggest guard measures the coverage of the file that
# already had the most attention: its eight pins were unmutated and one of them
# had been stale since the `verified` filter grew a metric arm, which nobody saw
# because the file was runnable only under node.
#
# `test_mode_dispatch.js` is deliberately absent, and its absence is not an
# omission: it pins no source text at all. It EXECUTES kernel_workflow.js
# against stubbed globals and asserts on the winner, the lane set, and the
# throws -- so there is nothing here to score, and its corpus
# (`DISPATCH_MUTANTS`) is checked by running the suite rather than by matching
# regexes. Listing it would print a row of zeros that reads like a gap.
SUITES = (
    (SUITE_JS, ("MUTANTS", "WF_MUTANTS")),
    (FLOOR_JS, ("FLOOR_MUTANTS",)),
)


def _mutant_corpus() -> dict[str, list]:
    """Read the mutant lists without importing the test module.

    Importing would pull in mini-racer and the whole runner for what is a
    literal list; `ast.literal_eval` on the source keeps this script runnable
    even when the JS engine is unavailable.
    """
    text = SUITE_PY.read_text(encoding="utf-8")
    out = {}
    for name in ("MUTANTS", "WF_MUTANTS", "FLOOR_MUTANTS"):
        match = re.search(rf"^{name} = \[(.*?)^\]", text, re.S | re.M)
        if not match:
            raise SystemExit(f"{SUITE_PY.name} no longer defines {name}")
        out[name] = ast.literal_eval("[" + match.group(1) + "]")
    return out


def _slices(js: str) -> dict[str, str]:
    """Identifiers bound to a `grab`bed slice of the lane, with the cutting regex.

    Narrowing the haystack is the STRONGER way to write a source pin -- a field
    can be present at the writer and missing from the reader, and a match
    against the whole file cannot tell those apart -- so an extractor that only
    understands `.test(src)` systematically misses the best pins in the suite.
    Resolving the cut here lets them be scored like any other: the mutant is
    applied to the lane, the slice is re-cut from the MUTATED lane, and the pin
    is matched against that. A mutant that moves the cut out of existence
    changes the slice to empty, which flips a positive pin -- correctly, since
    the assertion would throw at runtime.
    """
    consts = dict(re.findall(r"^const (\w+) =\n?\s*/((?:[^/\\\n]|\\.)+)/;", js, re.M))
    out: dict[str, str] = {}
    for name, pattern in re.findall(
            r"const (\w+) = grab(?:Group)?\(/((?:[^/\\\n]|\\.)+)/", js):
        out[name] = pattern
    for name, ref in re.findall(r"const (\w+) = grab(?:Group)?\((\w+),", js):
        if ref in consts:
            out[name] = consts[ref]
    for name, pattern in re.findall(
            r"const (\w+) = \w+\.match\(/((?:[^/\\\n]|\\.)+)/\)\[0\]", js):
        out[name] = pattern
    return out


def _cut(pattern: str, base: str) -> str | None:
    """Apply a slice's cutting regex to a (possibly mutated) source."""
    try:
        mo = re.search(pattern, base)
    except re.error:
        return None
    return mo.group(0) if mo else ""


def _pins(js: str, slices: dict[str, str] | None = None
          ) -> list[tuple[int, bool, str, str, int, str]]:
    """Every `/re/.test(src)` clause in the JS suite, with its enclosing `ok(`.

    Not just the first clause of each assertion. Most pins here are written
    `ok(/a/.test(src) && /b/.test(src) && !/c/.test(src), 'why')`, and an
    extractor anchored on `ok(` sees only `/a/`. On this file that hid 29 of 95
    patterns -- a third of the corpus -- in exactly the position where a
    conjunct gets bolted onto an existing assertion without anyone asking
    whether it distinguishes anything. The clause is the unit under audit; the
    enclosing assertion is reported alongside it so a clause that pins nothing
    while a sibling does is visibly a different situation from an assertion
    with nothing behind it at all.
    """
    targets = "|".join(
        ["src", "wfSrc"] + sorted(_slices(js) if slices is None else slices))
    starts = [mo.start() for mo in re.finditer(r"\bok\(", js)]
    found = []
    clause = re.compile(rf"(?:(!?)/((?:[^/\\\n]|\\.)+)/\.test\(({targets})\)"
                        rf"|({targets})\.match\(/((?:[^/\\\n]|\\.)+)/g\))")
    for mo in clause.finditer(js):
        owner = max((s for s in starts if s < mo.start()), default=-1)
        line = js[:mo.start()].count("\n") + 1
        if mo.group(2) is not None:
            found.append((line, bool(mo.group(1)), mo.group(2), mo.group(3), owner, "test"))
        else:
            # `(src.match(/re/g) || []).length === N`. A counting clause, and the
            # only form that can say "declared at every site" rather than "declared
            # somewhere" -- so it is precisely the strong pin (84) asks for, and an
            # extractor that only knows `.test` scores it as unexercised.
            found.append((line, False, mo.group(5), mo.group(4), owner, "count"))
    return found


def _unseen(js: str, scored: set[str]) -> list[tuple[int, str]]:
    """Source pins this script's extractor still cannot score, with their lines.

    Two forms used to slip past `_pins`. Grabbed-slice pins --
    `/x/.test(cellProj)` where `cellProj` was cut out of `src` -- are now
    scored (see `_slices`), because narrowing the haystack is the STRONGER way
    to write a pin and an extractor that skipped them was blindest exactly
    where the suite was strongest. What remains here is:

      * a pin built with `new RegExp(...)`, typically inside a `for` over a
        field list -- one line of source, N assertions at runtime. The pattern
        is assembled at runtime from loop variables, so there is no literal for
        this script to compile.
      * a slice whose own cutting regex this script cannot compile, and which
        therefore did not make it into `scored`.

    They are counted, not scored. A number that quietly excludes them is the
    failure this whole script is about, so silence here must mean "none".
    """
    holders = set(re.findall(r"const (\w+) = grab(?:Group)?\(", js))
    holders |= set(re.findall(r"const (\w+) = \w+\.match\(", js))
    holders -= {"src", "wfSrc"} | scored
    out = []
    for mo in re.finditer(r"(?:new RegExp\([^\n]*?\)|/(?:[^/\\\n]|\\.)+/)"
                          r"\.test\((\w+)\)", js):
        target = mo.group(1)
        dynamic = mo.group(0).startswith("new RegExp")
        if target in holders:
            out.append((js[:mo.start()].count("\n") + 1, "unreadable slice"))
        elif dynamic:
            out.append((js[:mo.start()].count("\n") + 1, "computed pattern"))
    return out


def audit(suite: Path, corpora: tuple[str, ...], verbose: bool) -> None:
    lane = LANE.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    corpus_by_name = _mutant_corpus()
    # The suite's own corpus defends its `src` pins; `WF_MUTANTS` only ever
    # applies to `wfSrc`. A suite listing one corpus gets it for `src` and an
    # empty one for `wfSrc`, which is honest: nothing it pins in the workflow
    # file (if it pinned anything there) would be exercised.
    lane_mutants = corpus_by_name[corpora[0]]
    wf_mutants = corpus_by_name[corpora[1]] if len(corpora) > 1 else []
    suite_js = suite.read_text(encoding="utf-8")
    name = suite.name
    # `grab` always cuts from the lane, so a slice pin is a lane pin that has
    # been narrowed. Unreadable cutting regexes are dropped here and their pins
    # fall through to the unscorable report below rather than being silently
    # scored against the whole file, which would overstate coverage in the one
    # place this script exists to measure it.
    slices = {k: v for k, v in _slices(suite_js).items()
              if _cut(v, lane) is not None}
    pins = _pins(suite_js, slices)

    unflippable: list[tuple[int, str, str, int]] = []
    flippable: list[tuple[int, str, str, int]] = []
    for line, negated, pattern, which, owner, kind in pins:
        if which in slices:
            base, corpus, cut = _cut(slices[which], lane), lane_mutants, slices[which]
        else:
            base = lane if which == "src" else workflow
            corpus = lane_mutants if which == "src" else wf_mutants
            cut = None
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            # JS regex this script cannot evaluate. Reported rather than
            # dropped: a pin nobody can audit is not a pin that is fine.
            unflippable.append((line, pattern, f"unreadable regex: {exc}", owner))
            continue
        if bool(rx.search(base)) == negated:
            # The assertion does not hold against the file on disk, yet the
            # suite passes -- so the two disagree about what is being matched.
            unflippable.append((line, pattern, "does not hold outside the JS engine", owner))
            continue
        # A `.test` clause flips when presence changes; a counting clause flips
        # when the number of matches changes at all, which is a strictly finer
        # signal -- dropping one of three declarations moves 3 to 2 while
        # presence stays true.
        before = len(rx.findall(base))
        killer = None
        for label, old, new in corpus:
            # A slice pin is applied to the WHOLE lane and then re-cut: a mutant
            # that lands outside the slice can still move the cut, and one that
            # deletes the cut entirely empties the slice -- which flips a
            # positive pin, correctly, because the assertion would throw.
            haystack = lane if cut else base
            if old not in haystack:
                continue
            mutated = haystack.replace(old, new)
            after = len(rx.findall(_cut(cut, mutated) if cut else mutated))
            flipped = (after != before) if kind == "count" else ((after > 0) == negated)
            if flipped:
                killer = label
                break
        if killer:
            flippable.append((line, pattern, killer, owner))
        else:
            # A negated pin gets its own label because the reason it is
            # unexercised is different in kind: not "nothing defends this
            # assertion" but "no mutant in the corpus INTRODUCES the forbidden
            # form". That is a statement about the corpus, and it is fixable by
            # adding an injection mutant -- which is how :987 stopped being in
            # this bucket.
            why = ("negated pin -- no mutant in the corpus introduces the "
                   "forbidden form" if negated else "no mutant in the corpus flips it")
            unflippable.append((line, pattern, why, owner))

    defended = {owner for _, _, _, owner in flippable}
    negated_count = sum(1 for _, _, why, _ in unflippable if why.startswith("negated pin"))
    bare = [row for row in unflippable
            if not row[2].startswith("negated pin") and row[3] not in defended]
    shielded = [row for row in unflippable
                if not row[2].startswith("negated pin") and row[3] in defended]
    print(f"{name}: {len(pins)} source-pinning clauses "
          f"in {len({row[4] for row in pins})} assertions")
    print(f"  {len(flippable)} are load-bearing -- some mutant flips them")
    print(f"  {negated_count} are converse-absence pins no mutant introduces")
    print(f"  {len(shielded)} pin nothing themselves but sit in an assertion a mutant does flip")
    print(f"  {len(bare)} are in assertions no mutant flips at all")
    unseen = _unseen(suite_js, set(slices))
    if unseen:
        kinds = {}
        for _, kind in unseen:
            kinds[kind] = kinds.get(kind, 0) + 1
        print(f"  (+{len(unseen)} source pins this extractor does not score: "
              + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())) + ")")
        if verbose:
            for line, kind in unseen:
                print(f"      {name}:{line}  {kind}")
    if verbose:
        print("\nload-bearing:")
        for line, pattern, killer, _ in flippable:
            print(f"  {name}:{line}  <- {killer}")
        print("\nunexercised clause, load-bearing sibling (weakest category: the "
              "assertion still fails on a real defect, this conjunct just is not "
              "the reason -- so it is a literal or it is redundant):")
        for line, pattern, why, _ in shielded:
            print(f"  {name}:{line}\n      /{pattern[:100]}/")
    # An empty backlog prints nothing. A heading over no rows reads like a
    # section that was not filled in, which is the same ambiguity finding (128)
    # is about: silence should mean "none", and it can only mean that if the
    # heading is absent when the list is.
    if bare:
        print("\nnothing in the enclosing assertion is exercised (read each; the "
              "docstring lists the four ways that happens, only one of which is a hole):")
        for line, pattern, why, _ in bare:
            print(f"  {name}:{line}  {why}\n      /{pattern[:100]}/")
    negated = [x for x in unflippable if x[2].startswith("negated pin")]
    if negated:
        print("\nconverse-absence pins no mutant in the corpus introduces (each is "
              "closable by an injection mutant; listed as a backlog, not as a limit):")
        for line, pattern, why, _ in negated:
            print(f"  {name}:{line}\n      /{pattern[:100]}/")


def main(argv: list[str]) -> int:
    verbose = "--verbose" in argv
    for i, (suite, corpora) in enumerate(SUITES):
        if i:
            print()
        audit(suite, corpora, verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
