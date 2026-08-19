#!/usr/bin/env python3
"""Filter a candidate GPU list down to the lanes this host can actually lock.

Motivation is in PIPELINE_PROGRESS §53.5. `/tmp/team_gpu_locks` is root-owned
and not group-writable here, so `gpu_lock.sh` cannot create a new lane file at
all, and `gpu_0.lock` / `gpu_1.lock` are root-owned. A run allocated GPU 0 dies
on the `200>"$LOCK_FILE"` redirect before its payload starts -- not "fails to
get the lock", fails to *open the file*. `sudo` needs a password and the wrapper
is frozen, so the defect is routed around rather than fixed, and this is where
the routing lives instead of in someone's memory at 3am.

Deliberately narrow. It answers exactly one question -- "can this lane be
locked?" -- and says nothing about whether a GPU is idle. `gpu_lock.sh` owns
idleness, and a second implementation of that gate would be a second thing to
keep in agreement with the first; when they disagreed, the more permissive one
would win by being the one that let the run start.

Usage:
    lockable_lanes.py 0 1 2 3        -> prints "2,3"
    lockable_lanes.py --from-watch-line "poll 61 idle=[0 4 5]"
    rocm-smi ... | lockable_lanes.py --stdin

Exit status is 1 and stdout is empty when nothing survives, so
`GEAK_GPU_ALLOWED=$(lockable_lanes.py ...)` cannot silently become an empty
allocation -- which `gpu_lock.sh` reads as "no fence at all".
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

LOCK_DIR = Path("/tmp/team_gpu_locks")


def is_lockable(gpu: int, lock_dir: Path = LOCK_DIR) -> bool:
    """Can `gpu_lock.sh` open this lane's file for writing?

    Mirrors the wrapper's own `exec {fd}>"$LOCK_DIR/gpu_$id.lock"`: append mode,
    which creates the file when the directory permits and fails when it does
    not. Anything unexpected is reported as NOT lockable -- the caller is about
    to hand this id to a measurement, and guessing optimistically here means
    discovering the answer from a dead run instead of from this function.
    """
    try:
        with open(lock_dir / f"gpu_{gpu}.lock", "a"):
            return True
    except OSError:
        return False


# The watcher writes `<ISO timestamp> poll 61 idle=[0 4 5]` and, on success,
# `GPU_IDLE_CONFIRMED: 0 4 5`. Both are accepted whole, because requiring the
# caller to cut the list out by hand is requiring it at the moment they are
# least likely to get it right.
WATCH_MARKERS = (re.compile(r"idle=\[([^\]]*)\]"),
                 re.compile(r"GPU_IDLE_CONFIRMED:([0-9 ,]*)"))


def narrow(text: str) -> str:
    """Cut a watcher line down to its id list, if it is one.

    The first version of this module skipped this step and simply pulled every
    integer out of the text. On the real log line that yields
    `2026,8,16,22,45,34,61,2,3` -- a timestamp and a poll counter promoted to
    GPU ids. Grabbing every number looked like generosity toward input shapes
    and was actually the opposite: it made a malformed input succeed.
    """
    for pattern in WATCH_MARKERS:
        found = pattern.findall(text)
        if found:
            return found[-1]  # a tail may hold several lines; the last is now
    return text


def parse_ids(text: str) -> list[int]:
    """Ids from an already-narrowed list, in order, de-duplicated.

    Strict: a token that is not a plain integer raises, rather than being
    skipped. A skipped token is a candidate the caller believes they passed.
    """
    seen, out = set(), []
    for tok in re.split(r"[,\s\[\]]+", narrow(text).strip()):
        if not tok:
            continue
        if not tok.isdigit():
            raise ValueError(f"not a GPU id: {tok!r}")
        gpu = int(tok)
        if gpu not in seen:
            seen.add(gpu)
            out.append(gpu)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", help="candidate GPU ids")
    ap.add_argument("--from-watch-line", default=None,
                    help="a line from the idle watcher; ids are read from it")
    ap.add_argument("--stdin", action="store_true", help="read candidates from stdin")
    ap.add_argument("--lock-dir", default=str(LOCK_DIR))
    ap.add_argument("--sep", default=",")
    args = ap.parse_args(argv)

    # Parsed per source, not concatenated first: narrowing a watcher line to its
    # bracket contents would otherwise throw away ids given on the command line.
    sources = [" ".join(args.ids)]
    if args.from_watch_line:
        sources.append(args.from_watch_line)
    if args.stdin:
        sources.append(sys.stdin.read())

    candidates: list[int] = []
    for source in sources:
        try:
            for gpu in parse_ids(source):
                if gpu not in candidates:
                    candidates.append(gpu)
        except ValueError as exc:
            print(f"cannot read candidate ids: {exc}", file=sys.stderr)
            return 2
    if not candidates:
        print("no candidate GPU ids given", file=sys.stderr)
        return 2

    lock_dir = Path(args.lock_dir)
    usable = [g for g in candidates if is_lockable(g, lock_dir)]
    refused = [g for g in candidates if g not in usable]
    if refused:
        print(f"refusing GPU(s) {','.join(map(str, refused))}: "
              f"{lock_dir}/gpu_<id>.lock is not writable by this user, so "
              "gpu_lock.sh would die on the flock redirect", file=sys.stderr)
    if not usable:
        print("no candidate lane is lockable on this host", file=sys.stderr)
        return 1
    print(args.sep.join(map(str, usable)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
