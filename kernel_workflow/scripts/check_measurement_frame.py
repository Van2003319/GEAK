#!/usr/bin/env python3
"""Preflight: is the ruler we are about to measure with the right ruler?

Run this BEFORE launching a wave, and after every restore. It answers one
question -- do the noise floors currently in force belong to the box we are
actually standing on -- and it exits non-zero when they do not.

Why this exists as a script and not as a test. The check already existed, in
`test_noise_floor_stats.py::EpochIdentityTest`. It was red, and it stayed red
across a whole wave, because nothing runs a unit test before a wave launches.
The post-restore integrity pass that did run verified the seed digest, the
oracle digest over 74 files, the lane HEAD and both twin sources -- everything
except the floor table -- and passed clean while the floors in use belonged to
another machine. A digest proves the code survived. It says nothing about
whether the ruler still fits the room.

This class has now bitten three times:
  * finding (107)  -- a table carried across a machine boundary on an argument;
                      re-measured, four of eleven routes came back NARROWER,
                      one by 3.3x.
  * finding (126)  -- CURRENT_MACHINE read "P" (tw008) while hostname returned
                      tw003, and nothing anywhere would have said so.
  * greedy lane, 2026-08-17 -- restored onto tw054 with CURRENT_MACHINE = "R"
                      (tw008). Cost nothing only because a foreign tenant held
                      all eight GPUs, so no timing was taken before it was
                      caught. Prospective damage would have been total.

Exit codes -- distinct on purpose, so a launcher can branch on them:
  0  frame is consistent and the floors are MEASURED. Cleared to measure.
  3  frame is consistent but the epoch is PROVISIONAL (0.072 everywhere).
     Not an error: it is the correct fail-closed state for a new box. But no
     win narrower than the default floor is admissible until the real floors
     are measured, so the first GPU work must be measure_noise_floor.py.
  4  frame is INCONSISTENT -- hostname resolves to a different epoch than
     CURRENT_MACHINE, or to no epoch at all. Do not measure. Anything timed in
     this state is compared against another machine's ruler.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import noise_floor_stats as QRS  # noqa: E402


def classify(host: str, resolved: str | None, current: str,
             provisional: set[str]) -> tuple[int, str]:
    """The whole decision, as a pure function of the frame. Returns (exit, why).

    Split out from `main` so every branch is reachable in a test without
    editing module constants. A preflight whose own logic is only exercised by
    running it on the one box it happens to be installed on is the same mistake
    one level up.
    """
    if resolved is None:
        return 4, f"{host} is registered to no epoch"
    if resolved != current:
        return 4, f"host resolves to {resolved} but CURRENT_MACHINE is {current}"
    if current in provisional:
        return 3, f"epoch {current} is provisional"
    return 0, f"epoch {current} is measured and matches {host}"


def state_continuity(state: Mapping[str, Any], host: str, epoch: str | None
                     ) -> tuple[str, dict[str, Any]]:
    """Has this lane's state been measured on more than one box?

    Reports a line and the block to persist. This is deliberately NOT part of
    `classify`: a machine change does not invalidate the lane. Every admission
    is a paired, interleaved, same-session ratio, and ratios compose across a
    restore -- the cumulative speedup survives a machine boundary intact.
    What does NOT survive is anything absolute: per-case microseconds, and the
    question "which epoch's floor was in force when round N was admitted".

    That second question cost a full session to answer by hand, from rocprofv3
    output directories (which happen to be named after the host) and `stat`
    birth times, because no round had ever recorded the box it ran on. This
    records it going forward so the archaeology is done once.
    """
    frame = dict(state.get("measurement_frame") or {})
    seen = list(frame.get("hosts_seen") or [])
    first = frame.get("first_host")
    if host not in seen:
        seen.append(host)
    frame["hosts_seen"] = seen
    frame["first_host"] = first or host
    frame["last_host"] = host
    frame["last_epoch"] = epoch
    frame["note"] = (
        "Paired same-session ratios compose across a machine boundary, so the "
        "cumulative speedup is intact. Absolute per-case microseconds from "
        "different hosts are NOT comparable, and a floor only describes the "
        "box it was measured on.")
    if len(seen) > 1:
        line = (f"lane state          : measured on {len(seen)} boxes "
                f"{seen} -- ratios compose, absolutes do not")
    else:
        line = f"lane state          : all recorded rounds on {host}"
    return line, frame


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=None,
                    help="override the hostname (for testing this script)")
    ap.add_argument("--state", default=None,
                    help="STATE.json to report host continuity for")
    ap.add_argument("--stamp", action="store_true",
                    help="write the continuity block back into --state")
    args = ap.parse_args()

    host = args.host or socket.gethostname()
    resolved = QRS.machine_for_host(host)
    current = QRS.CURRENT_MACHINE

    print(f"hostname            : {host}")
    print(f"resolves to epoch   : {resolved if resolved else '(unregistered)'}")
    print(f"CURRENT_MACHINE     : {current} -> {QRS.MACHINE_HOSTNAME.get(current)}")

    if args.state:
        state_path = Path(args.state)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        line, frame = state_continuity(state, host, resolved)
        print(line)
        if args.stamp:
            state["measurement_frame"] = frame
            state_path.write_text(
                json.dumps(state, indent=2, ensure_ascii=False,
                           sort_keys=True) + "\n",
                encoding="utf-8")
            print(f"                      : stamped into {state_path.name}")

    if resolved is None:
        print()
        print(f"STOP: {host} is registered to no epoch. The floors now in force were")
        print(f"measured on {QRS.MACHINE_HOSTNAME.get(current)} and do not describe this box.")
        print("Register a new PROVISIONAL epoch for this host (a new letter -- never")
        print("inherit a retired letter for the same box, that is finding (126) with")
        print("extra steps), then measure its real floors.")
        return 4

    if resolved != current:
        print()
        print(f"STOP: host resolves to epoch {resolved}, but CURRENT_MACHINE is {current}.")
        print("Whatever you time next would be judged against another machine's ruler.")
        return 4

    if current in QRS.PROVISIONAL_MACHINES:
        print()
        print(f"PROVISIONAL: epoch {current} has no measured table -- every route sits at")
        print(f"DEFAULT_NOISE_FLOOR = {QRS.DEFAULT_NOISE_FLOOR}. This is the correct")
        print("fail-closed state for a new box, not a fault. But nothing narrower than")
        print("that floor is admissible, so the FIRST GPU work must be:")
        print("  measure_noise_floor.py   (8 same-variant full-suite primed repeats,")
        print("                            the (105) debiased harness, 2*MAD/median)")
        print(f"  deprovisionalize_epoch.py --verdict <json> --machine {current} --apply")
        return 3

    n = len(QRS.MEASURED_NOISE_FLOOR_BY_MACHINE[current])
    print()
    print(f"OK: epoch {current} is measured over {n} routes and matches this host.")
    print("Cleared to measure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
