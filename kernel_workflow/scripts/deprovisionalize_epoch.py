#!/usr/bin/env python3
"""Install a measured noise-floor table and retire an epoch's PROVISIONAL flag.

`measure_noise_floor.py` ends by printing two blocks and a sentence: paste this
into `qd_robust_stats.py`, paste that into `kernel_lane.js`, then remove the
letter from `PROVISIONAL_MACHINES` and from `QD_PROVISIONAL_MACHINES`. Four
hand-edits in two languages, at whatever hour a GPU finally frees. The parity
tests catch a half-applied edit, which is the right outcome and also a wedged
pipeline: the suite is red and the run that was waiting for the floors cannot
proceed until someone finishes the paste.

So the paste is a tool. It is all-or-nothing -- both files are rendered in full
and validated before either is written -- and idempotent, so a re-run after a
crash is a no-op rather than a second table.

It renders through `measure_noise_floor.render_python` / `render_js` rather than
formatting the numbers again here. A second renderer would be a second thing to
keep in agreement with the first, and the disagreement would be invisible: both
outputs are plausible tables of plausible floats.

Prose is handled by ownership, not by cleverness. The comment directly above
each table says in sentences that the epoch is unmeasured and that measuring it
is the next GPU work; after the edit that text is false, and false-but-confident
prose beside a correct table is worse than no prose at all. So that comment
block is part of the anchor in BOTH languages and is replaced wholesale by
generated provenance. Sentences the tool cannot anchor -- anywhere else in
either file -- it will not rewrite and will not ignore: it reports every
remaining PROVISIONAL mention of the machine and exits 6, stopping an unattended
chain at the point where a human sentence is genuinely required.

Owning the comment is also what makes the Python side idempotent. An anchor
starting at the assignment leaves the old header in place and inserts a new one
above it, so re-running after a crash stacks contradictory provenance instead of
doing nothing.

    deprovisionalize_epoch.py --verdict /tmp/floor_Q.json --machine Q --check
    deprovisionalize_epoch.py --verdict /tmp/floor_Q.json --machine Q --apply

Exit codes: 0 nothing to do / applied cleanly; 1 --check found pending edits;
2 the verdict is not usable; 3 an anchor did not match (nothing written);
6 applied, but stale PROVISIONAL prose remains and must be rewritten by hand.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
LANE = HERE.parent / "kernel_lane.js"
STATS = HERE / "qd_robust_stats.py"

sys.path.insert(0, str(HERE))

import measure_noise_floor as MNF  # noqa: E402
import qd_robust_stats as QRS  # noqa: E402


class AnchorError(RuntimeError):
    """A pattern that must match exactly once did not."""


def shortname(path: Path) -> str:
    """Repo-relative when it is in the repo, absolute when it is not.

    Tests point STATS/LANE at copies outside the tree, and a progress line that
    raises rather than prints would fail the run for a cosmetic reason.
    """
    try:
        return str(path.relative_to(HERE.parents[1]))
    except ValueError:
        return str(path)


def one(pattern: re.Pattern[str], text: str, what: str) -> re.Match[str]:
    found = list(pattern.finditer(text))
    if len(found) != 1:
        raise AnchorError(f"{what}: expected exactly 1 match, found {len(found)}")
    return found[0]


# --- validation -------------------------------------------------------------

def check_verdict(verdict: Mapping[str, Any], machine: str) -> list[str]:
    """Reasons this verdict must not become a floor table.

    A floor decides what the archive will accept forever after, so the bar is
    the sweep's own bar and not a softer one: the same `ok` flag, the same
    minimum repeat count, and -- the one this file adds -- exactly the routes
    the epoch already has. A table missing a route does not fail loudly; that
    route silently falls back to DEFAULT_NOISE_FLOOR, which is the widest floor
    anywhere, so the gap reads as a very quiet epoch with one stubborn route.
    """
    problems: list[str] = MNF.attribution_problems(verdict, machine)
    if not verdict.get("ok"):
        problems.append(f"verdict is not ok (stage={verdict.get('stage')!r}, "
                        f"problems={verdict.get('problems')})")
    complete = verdict.get("repeats_complete", 0)
    if complete < MNF.MIN_REPEATS:
        problems.append(f"{complete} complete repeats, need {MNF.MIN_REPEATS}")
    routes = verdict.get("routes") or {}
    if not routes:
        problems.append("no routes in the verdict")
    existing = QRS.MEASURED_NOISE_FLOOR_BY_MACHINE.get(machine)
    if existing:
        missing = sorted(set(existing) - set(routes))
        extra = sorted(set(routes) - set(existing))
        if missing:
            problems.append(f"routes measured nowhere in this sweep: {missing}; "
                            "each would fall back to DEFAULT_NOISE_FLOOR and read "
                            "as one stubborn route on an otherwise quiet epoch")
        if extra:
            problems.append(f"routes not in the epoch's table: {extra}")
    for route, stats in sorted(routes.items()):
        if not isinstance(stats, Mapping) or "floor" not in stats:
            problems.append(f"{route}: no floor in the verdict row")
            continue
        floor = stats["floor"]
        if not (MNF.MIN_FLOOR <= floor <= QRS.DEFAULT_NOISE_FLOOR * 4):
            problems.append(f"{route}: floor {floor} outside the sane band "
                            f"[{MNF.MIN_FLOOR}, {QRS.DEFAULT_NOISE_FLOOR * 4}]")
    return problems


def provenance(verdict: Mapping[str, Any], machine: str, comment: str) -> str:
    # The host the sweep actually ran on, not the one the letter is registered
    # to. Those agree only because check_verdict now refuses when they do not;
    # naming the measured box keeps the comment a record rather than a restated
    # assumption.
    host = (verdict.get("host") or QRS.MACHINE_HOSTNAME.get(machine)
            or "host not recorded")
    return "\n".join(f"{comment} {line}" for line in (
        f"machine {machine} -- {host}. MEASURED: "
        f"{verdict['repeats_complete']} complete same-variant primed repeats, "
        f"source_hash {verdict.get('source_hash', 'unrecorded')}.",
        "Installed by deprovisionalize_epoch.py from the sweep verdict; the "
        "statistic is 2*MAD(speedup)/median(speedup) per route, floors below "
        f"MIN_FLOOR ({MNF.MIN_FLOOR}) clamped up. Floors do not pool across a "
        "machine boundary, so this table is a reading of this box only.",
    ))


# --- the two edits ----------------------------------------------------------

PY_TABLE = re.compile(
    r'^(?P<comment>(?:#[^\n]*\n)*)'
    r'MEASURED_NOISE_FLOOR_BY_MACHINE\["(?P<m>[A-Z])"\] = \{.*?^\}',
    re.DOTALL | re.MULTILINE)
PY_SET = re.compile(r"^PROVISIONAL_MACHINES = (?P<body>\{[^}]*\}|set\(\))",
                    re.MULTILINE)
JS_TABLE = re.compile(
    r"^(?P<comment>(?:  //[^\n]*\n)*)  \['(?P<m>[A-Z])', new Map\(\[.*?^  \]\)\],",
    re.DOTALL | re.MULTILINE)
JS_SET = re.compile(
    r"^const QD_PROVISIONAL_MACHINES = new Set\((?P<body>\[[^\]]*\])\);",
    re.MULTILINE)


def drop_from_py_set(text: str, machine: str) -> str:
    match = one(PY_SET, text, "PROVISIONAL_MACHINES")
    body = match.group("body")
    letters = sorted(set(re.findall(r'"([A-Z])"', body)) - {machine})
    new = "{" + ", ".join(f'"{l}"' for l in letters) + "}" if letters else "set()"
    return text[:match.start("body")] + new + text[match.end("body"):]


def drop_from_js_set(text: str, machine: str) -> str:
    match = one(JS_SET, text, "QD_PROVISIONAL_MACHINES")
    letters = sorted(set(re.findall(r"'([A-Z])'", match.group("body"))) - {machine})
    new = "[" + ", ".join(f"'{l}'" for l in letters) + "]"
    return text[:match.start("body")] + new + text[match.end("body"):]


def replace_py_table(text: str, machine: str, block: str) -> str:
    """Replaces the assignment AND the comment block directly above it.

    Same anchor shape as the JS side, and for the same two reasons. The comment
    is the sentence saying the epoch is unmeasured, sitting immediately above
    the numbers that now say otherwise. And an anchor that starts *below* the
    comment is not idempotent: each run inserts a fresh provenance block above
    the previous one, so a re-run after a crash leaves a stack of contradictory
    headers rather than a no-op.
    """
    for match in PY_TABLE.finditer(text):
        if match.group("m") == machine:
            return text[:match.start()] + block + text[match.end():]
    raise AnchorError(f"no MEASURED_NOISE_FLOOR_BY_MACHINE[\"{machine}\"] assignment")


def replace_js_table(text: str, machine: str, block: str) -> str:
    """Replaces the entry AND the comment block directly above it.

    The comment is part of the anchor on purpose. It is the sentence that says
    the epoch is unmeasured, it sits immediately above the numbers, and leaving
    it in place beside a measured table is the failure this whole tool exists to
    avoid repeating by hand.
    """
    for match in JS_TABLE.finditer(text):
        if match.group("m") == machine:
            return text[:match.start()] + block + text[match.end():]
    raise AnchorError(f"no ['{machine}', new Map([...])] entry in {LANE.name}")


def render(verdict: Mapping[str, Any], machine: str) -> tuple[str, str]:
    table = verdict["routes"]
    py = provenance(verdict, machine, "#") + "\n" + MNF.render_python(machine, table)
    js = provenance(verdict, machine, "  //") + "\n" + MNF.render_js(machine, table)
    return py, js


def rewrite(verdict: Mapping[str, Any], machine: str) -> dict[Path, str]:
    """New text for each file. Raises AnchorError before writing anything."""
    py_block, js_block = render(verdict, machine)
    stats = STATS.read_text(encoding="utf-8")
    lane = LANE.read_text(encoding="utf-8")
    return {
        STATS: drop_from_py_set(replace_py_table(stats, machine, py_block), machine),
        LANE: drop_from_js_set(replace_js_table(lane, machine, js_block), machine),
    }


# --- the prose the tool refuses to fake ------------------------------------

WORD = re.compile(r"\bPROVISIONAL\b")

def stale_prose(machine: str) -> list[str]:
    """Lines that still call this epoch provisional, as `file:line: text`.

    A backstop, not the mechanism. The prose that matters sits in the comment
    block above each table and is replaced by the anchors; this catches the
    sentence somebody wrote about epoch Q three hundred lines away. It matches
    the literal word deliberately: a matcher that guessed at provisional-sounding
    language would find something it could never be made to stop finding, and an
    exit code that is permanently 6 carries exactly as much information as one
    that is permanently 0.
    """
    named = re.compile(rf"\b{machine}\b|`{machine}`|'{machine}'|\"{machine}\"")
    out = []
    for path in (STATS, LANE):
        for start, block in comment_blocks(path.read_text(encoding="utf-8")):
            text = "\n".join(block)
            # The bare word, not the substring: `deprovisionalize_epoch.py` and
            # `PROVISIONAL_MACHINES` both contain it, and the first of the two
            # appears in the provenance this tool writes -- so a substring test
            # makes every successful apply report its own signature as stale.
            if WORD.search(text.upper()) and named.search(text):
                out.append(f"{path.name}:{start}: {block[0].strip()}")
    return out


def comment_blocks(text: str) -> list[tuple[int, list[str]]]:
    """Runs of consecutive comment lines, as (1-based first line, lines).

    Blocks rather than lines because that is how these comments are actually
    written. The live entry says "machine Q -- tw003" on one line and
    "PROVISIONAL: nothing has been measured here" on the next, so a per-line
    scan for both terms together reads the paragraph that is entirely about an
    unmeasured epoch Q and finds nothing at all.
    """
    blocks: list[tuple[int, list[str]]] = []
    current: list[str] = []
    start = 0
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith(("#", "//")):
            if not current:
                start = n
            current.append(line)
        elif current:
            blocks.append((start, current))
            current = []
    if current:
        blocks.append((start, current))
    return blocks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verdict", required=True, help="JSON from measure_noise_floor --out")
    ap.add_argument("--machine", default=QRS.CURRENT_MACHINE)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report, write nothing")
    mode.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    machine = args.machine
    verdict = json.loads(Path(args.verdict).read_text(encoding="utf-8"))

    problems = check_verdict(verdict, machine)
    if problems:
        for problem in problems:
            print(f"REFUSED: {problem}", file=sys.stderr)
        print("no table installed -- the fail-closed default it would replace is "
              "wide, not wrong", file=sys.stderr)
        return 2

    try:
        planned = rewrite(verdict, machine)
    except AnchorError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        print("both files are left untouched; the source moved and the anchors "
              "need updating before this can be trusted to edit it", file=sys.stderr)
        return 3

    changed = [p for p, text in planned.items()
               if text != p.read_text(encoding="utf-8")]
    if not changed:
        print(f"epoch {machine} already carries this measured table; nothing to do")
        return 0

    if args.check:
        for path in changed:
            print(f"would rewrite {shortname(path)}")
        return 1

    for path, text in planned.items():
        path.write_text(text, encoding="utf-8")
    for path in changed:
        print(f"rewrote {shortname(path)}")

    left = stale_prose(machine)
    if left:
        print(f"\nSTALE PROSE: {len(left)} comment line(s) still describe epoch "
              f"{machine} as provisional. The table is now a measurement and "
              "these sentences say it is not:", file=sys.stderr)
        for line in left:
            print(f"  {line}", file=sys.stderr)
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
