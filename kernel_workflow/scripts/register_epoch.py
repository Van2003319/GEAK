#!/usr/bin/env python3
"""Register a PROVISIONAL epoch for the box this container is standing on.

Why this is a script. Registering an epoch is four hand-edits to
`noise_floor_stats.py`, and this lane's own log records what hand-edits cost:
one of them was a `.replace()` whose old-string had been guessed wrong, so the
edit silently did nothing and the run continued against another machine's
floors until a preflight caught it. Every edit here asserts that it changed the
text, and the whole thing is verified by re-importing the module afterwards --
a rewrite that did not take is a refusal, never a warning.

The letter is always NEW. A box that carried a retired epoch gets a fresh
letter when the container lands on it again; inheriting the old one reinstates
floors measured in a different container, which is finding (126).

    register_epoch.py --letter Y --host tw053            # writes
    register_epoch.py --letter Y --host tw053 --dry-run  # prints the diff
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path

BLOCK = '''# machine {L} -- {H}. PROVISIONAL: this box has never been measured for this
# lane, so every route sits at DEFAULT_NOISE_FLOOR and nothing narrower than
# that is admissible. This is the correct fail-closed state for a new box, not
# a fault. Registered as a NEW letter rather than by inheriting one this host
# may already have carried: a re-used box gets a new epoch, and reinstating the
# old letter's floors is finding (126). The FIRST GPU work here must be
# measure_noise_floor.py (8 same-variant full-suite primed repeats, the (105)
# debiased harness, 2*MAD/median), then deprovisionalize_epoch.py --apply,
# which replaces this comment and the table below it in a single edit.
MEASURED_NOISE_FLOOR_BY_MACHINE["{L}"] = {{
    route: DEFAULT_NOISE_FLOOR for route in MEASURED_NOISE_FLOOR_BY_MACHINE["{REF}"]
}}


'''


def edit(text: str, old: str, new: str, what: str) -> str:
    """Replace exactly once, or refuse. A no-op edit is the failure mode."""
    n = text.count(old)
    if n != 1:
        raise SystemExit(
            f"REFUSED: the anchor for {what} occurs {n} times, expected exactly 1.\n"
            f"  anchor: {old!r}\n"
            "  The file has moved under this script. Fix the anchor rather than\n"
            "  letting the edit silently not happen.")
    return text.replace(old, new, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--letter", required=True, help="the NEW epoch letter, e.g. Y")
    ap.add_argument("--host", required=True, help="hostname this epoch describes, e.g. tw053")
    ap.add_argument("--note", default="", help="one clause explaining why this epoch exists")
    ap.add_argument("--file", type=Path,
                    default=Path(__file__).resolve().parent / "noise_floor_stats.py")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    L, H = a.letter.strip().upper(), a.host.strip()
    if not re.fullmatch(r"[A-Z]", L):
        raise SystemExit(f"REFUSED: {L!r} is not a single epoch letter")

    path = a.file
    if not path.is_file():
        raise SystemExit(f"REFUSED: {path} does not exist")
    text = original = path.read_text(encoding="utf-8")

    if f'"{L}":' in text or f'["{L}"]' in text:
        raise SystemExit(
            f"REFUSED: epoch {L} already appears in {path.name}. Pick the next unused letter --\n"
            "  overwriting a letter is how one box's floors end up judging another's timings.")

    cur = re.search(r'^CURRENT_MACHINE = "([A-Z])"', text, re.M)
    if not cur:
        raise SystemExit("REFUSED: no CURRENT_MACHINE assignment found")
    ref = cur.group(1)

    # 1. the host table. Appended, because `machine_for_host` resolves a host to
    #    its NEWEST epoch by taking the last match -- order is load-bearing.
    host_tbl = re.search(r"^MACHINE_HOSTNAME = \{.*?^\}", text, re.S | re.M)
    if not host_tbl:
        raise SystemExit("REFUSED: no MACHINE_HOSTNAME table found")
    tail = host_tbl.group(0)[-2:]              # the "\n}" that closes it
    note = a.note or f"registered {L} for this box; see the {L} block below"
    text = edit(text, host_tbl.group(0),
                host_tbl.group(0)[:-2] + f'\n    "{L}": "{H}",     # {note}' + tail,
                "MACHINE_HOSTNAME")

    # 2. the provisional set.
    pset = re.search(r'^PROVISIONAL_MACHINES = (\{[^}]*\}|set\(\))', text, re.M)
    if not pset:
        raise SystemExit("REFUSED: no PROVISIONAL_MACHINES assignment found")
    letters = sorted(set(re.findall(r'"([A-Z])"', pset.group(1))) | {L})
    text = edit(text, pset.group(0),
                "PROVISIONAL_MACHINES = {" + ", ".join(f'"{x}"' for x in letters) + "}",
                "PROVISIONAL_MACHINES")

    # 3+4. the table (shaped exactly like a measured one so no caller needs a
    #      special case) and the pointer, in one anchor so they cannot diverge.
    text = edit(text, f'CURRENT_MACHINE = "{ref}"',
                BLOCK.format(L=L, H=H, REF=ref) + f'CURRENT_MACHINE = "{L}"',
                "CURRENT_MACHINE")

    if a.dry_run:
        import difflib
        sys.stdout.writelines(difflib.unified_diff(
            original.splitlines(True), text.splitlines(True),
            f"a/{path.name}", f"b/{path.name}"))
        return 0

    path.write_text(text, encoding="utf-8")

    # --- the part that makes this a registration rather than a text edit ----
    sys.path.insert(0, str(path.parent))
    for name in ("noise_floor_stats",):
        if name in sys.modules:
            del sys.modules[name]
    QRS = importlib.import_module("noise_floor_stats")
    problems = []
    if QRS.CURRENT_MACHINE != L:
        problems.append(f"CURRENT_MACHINE is {QRS.CURRENT_MACHINE!r}, not {L!r}")
    if QRS.MACHINE_HOSTNAME.get(L) != H:
        problems.append(f"MACHINE_HOSTNAME[{L!r}] is {QRS.MACHINE_HOSTNAME.get(L)!r}, not {H!r}")
    if QRS.machine_for_host(H) != L:
        problems.append(f"machine_for_host({H!r}) resolves to {QRS.machine_for_host(H)!r}, not {L!r}")
    if L not in QRS.PROVISIONAL_MACHINES:
        problems.append(f"{L} is not in PROVISIONAL_MACHINES, so its floors would read as measured")
    table = QRS.MEASURED_NOISE_FLOOR_BY_MACHINE.get(L) or {}
    if len(table) != 11 or set(table.values()) != {QRS.DEFAULT_NOISE_FLOOR}:
        problems.append(f"the {L} table is {len(table)} routes at {sorted(set(table.values()))}, "
                        f"expected 11 routes all at DEFAULT_NOISE_FLOOR")
    if problems:
        path.write_text(original, encoding="utf-8")
        raise SystemExit("REFUSED and REVERTED -- the edit did not take:\n  "
                         + "\n  ".join(problems))

    print(f"epoch {L} registered for {H}, PROVISIONAL, "
          f"11 routes at DEFAULT_NOISE_FLOOR = {QRS.DEFAULT_NOISE_FLOOR}")
    print(f"  seeded from epoch {ref}'s route names; CURRENT_MACHINE {ref} -> {L}")
    print("  next: check_measurement_frame.py should now exit 3, not 4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
