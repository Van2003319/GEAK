#!/usr/bin/env python3
"""Emit this epoch's measured per-route noise floors, in the shape the lane accepts.

Why this exists. `kernel_lane.js` cannot read the filesystem -- it is a workflow
script with agent calls and no `fs` -- so the floor table has to arrive inline
as `args.route_bands`. For seven waves nobody passed one, the per-route gate
logged nothing at all, and the only table on disk was six epochs stale. The lane
then derived its own from five baseline repeats, which measured the draw rather
than the route (8.8x too tight on one route, 2.9x too loose on another).

The table that is actually right already exists: `measure_noise_floor.py` takes
8 same-variant primed repeats and reports 2*MAD/median per route, and epoch
registration now re-measures it on every machine change. This is the two lines
that get it from there to the lane, so that "the right table exists" and "the
gate reads the right table" stop being different facts.

It refuses rather than guesses. An epoch whose table was never measured is
PROVISIONAL -- every route sits at the fail-closed default -- and passing that
to the gate would raise every route's bar to ~7%, silently turning the gate off
for a whole wave. That is exit 3, with the sweep command to run.

    route_floors.py                 # {route: floor} for the current epoch
    route_floors.py --arg           # the same, as a --print-able args fragment
    route_floors.py --machine Z     # another epoch, for comparison
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import noise_floor_stats as QRS  # noqa: E402


class FloorsUnavailable(ValueError):
    """No table this epoch can honestly hand to a gate."""


def resolve(machine: str | None = None, allow_provisional: bool = False,
            host: str | None = None) -> dict[str, float]:
    """The floors, or a refusal. Importable so `lane_args.py` can resolve the
    table at launch instead of anyone pasting a per-epoch number into a file
    that outlives the epoch -- which is how the one table on disk went six
    epochs stale while looking current."""
    live = machine is None
    machine = machine or QRS.CURRENT_MACHINE
    if live:
        here = host or socket.gethostname()
        resolved = QRS.machine_for_host(here)
        if resolved != machine:
            raise FloorsUnavailable(
                f"this host is {here} (epoch {resolved or 'unregistered'}) but CURRENT_MACHINE is "
                f"{machine} -> {QRS.MACHINE_HOSTNAME.get(machine)}. Run check_measurement_frame.py.")
    table = QRS.MEASURED_NOISE_FLOOR_BY_MACHINE.get(machine)
    if not table:
        raise FloorsUnavailable(f"epoch {machine} has no table at all")
    if machine in QRS.PROVISIONAL_MACHINES and not allow_provisional:
        raise FloorsUnavailable(
            f"epoch {machine} ({QRS.MACHINE_HOSTNAME.get(machine)}) is PROVISIONAL -- every route "
            f"sits at the fail-closed DEFAULT_NOISE_FLOOR = {QRS.DEFAULT_NOISE_FLOOR}, which is not "
            "a measurement. Passing it would raise every route's bar to that and turn the commit "
            "gate off for the whole wave without saying so. Measure the floors first: "
            f"measure_noise_floor.py --repeats 8 --machine {machine} --out <json>, then "
            f"deprovisionalize_epoch.py --verdict <json> --machine {machine} --apply")
    return {route: round(float(v), 6) for route, v in sorted(table.items())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", default=None,
                    help="epoch letter (default: the live CURRENT_MACHINE)")
    ap.add_argument("--arg", action="store_true",
                    help='print `"route_bands": {...}` ready to paste into a lane args file')
    ap.add_argument("--allow-provisional", action="store_true",
                    help="emit the fail-closed default table anyway (it will disable the gate)")
    a = ap.parse_args(argv)

    machine = a.machine or QRS.CURRENT_MACHINE
    host = QRS.MACHINE_HOSTNAME.get(machine)

    # The frame check belongs here too, not only in gpu_lock: a floor table is only about the box it
    # was measured on, and handing the lane another box's table is the same error as timing against
    # another box's ruler -- it just fails later and more quietly.
    try:
        floors = resolve(a.machine, allow_provisional=a.allow_provisional)
    except FloorsUnavailable as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        # 3 = registered but unmeasured (clearable by a sweep); 4 = wrong box or no epoch at all.
        return 3 if "PROVISIONAL" in str(exc) else 4

    if a.arg:
        body = json.dumps(floors, sort_keys=True, indent=4)
        print(f'  "_route_bands_provenance": "epoch {machine} ({host}), measured, '
              f'{len(floors)} routes -- from scripts/route_floors.py",')
        print(f'  "route_bands": {body},')
    else:
        print(json.dumps({"machine": machine, "host": host, "measured": True, "floors": floors},
                         sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
