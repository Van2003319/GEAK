#!/usr/bin/env python3
"""Validate a committed launch-argument file and render the exact invocation.

Why this exists
---------------
The arguments a wave is launched with decide what that wave will ACCEPT, and on
this project they lived only in prose that was retyped by hand every wave. The
cost is on the record: one wave's invocation was retyped without
`min_improve: 0.005`, so it ran at the 0.02 default and refused a verified,
correctness-passing, policy-clean +1.58% integrated stack -- the largest result
of two days of GPU time. Nothing in the run said which threshold was live.

Three failures have to be separated, because they need three different cures:

1. **A misspelled key.** Silently ignored by a JS workflow. Cured by the
   `KNOWN_ARGS` check in `kernel_lane.js` / `kernel_workflow.js`, and by
   `--check` here, which refuses before anything is launched.
2. **An omitted key.** Takes its default, and NOTHING can distinguish that from
   a default that was intended -- not from inside the run. Cured by writing the
   arguments down once, in a file under version control, and by `--check
   --require k=v`, which asserts a protocol's load-bearing values are actually
   present and actually equal to what the protocol says.
3. **A key whose value drifted.** Cured by the same `--require`, plus the
   effective-config echo the lane prints at its Setup phase.

The accepted-argument set is EXTRACTED from the two JS entry points rather than
restated here. A third hand-maintained copy of that list is a third thing to
forget to update, and the whole point of this file is to stop relying on a human
transcribing a list correctly.

Usage
-----
    lane_args.py --check   lanes/greedy_bf16_gemm.json [--require min_improve=0.005 ...]
    lane_args.py --print   lanes/greedy_bf16_gemm.json     # the Workflow({...}) call to paste
    lane_args.py --json    lanes/greedy_bf16_gemm.json     # canonical args object, one line

The file itself is a plain JSON object of arguments, with two conveniences: keys
beginning with `_` are treated as comments and stripped, and `_require` may hold
the protocol's load-bearing values so `--check` needs no command line.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

WF_DIR = Path(__file__).resolve().parent.parent
LANE_JS = WF_DIR / "kernel_lane.js"
DISPATCH_JS = WF_DIR / "kernel_workflow.js"

# The lane throws on either of these being absent, so a file that omits one is
# broken before any knob is considered.
REQUIRED_KEYS = ("kernel_path", "workflow_dir")

_KNOWN_RE = re.compile(r"const KNOWN_ARGS = new Set\(\[(.*?)\]\);", re.S)


class LaneArgsError(ValueError):
    """The argument file is not one this workflow can be launched with."""


def known_args(source: Path) -> set[str]:
    """The accepted-argument set, read out of a JS entry point's own source."""
    try:
        text = source.read_text()
    except OSError as exc:
        raise LaneArgsError(f"cannot read {source}: {exc}") from exc
    m = _KNOWN_RE.search(text)
    if not m:
        raise LaneArgsError(
            f"{source.name} has no `const KNOWN_ARGS = new Set([...])` block. Either the entry point "
            "lost its argument check -- in which case a misspelled knob is silently ignored again -- "
            "or it was renamed and this extractor needs updating. Both are refusals, not defaults.")
    # Only quoted entries; comment lines inside the block are ignored by construction.
    return set(re.findall(r"'([a-z_][a-z0-9_]*)'", m.group(1)))


# A value that must be RESOLVED at launch rather than written down, because what it names changes
# faster than the file does.
#
# `route_bands` is the only one so far and it is the reason this exists: the per-route floors are a
# property of the box, this container changes box every few hours, and the one floor table ever
# committed to disk went six epochs stale while still looking like a current measurement. Writing
# today's numbers into a lane file would reproduce that exactly -- correct on the day, silently
# describing another machine by the next restore.
#
# Resolution happens in `--check` too, so an epoch whose floors were never measured is a refusal
# BEFORE the wave starts rather than a table of fail-closed defaults that quietly holds every route
# to ~7% for the whole run.
RESOLVERS: dict[str, str] = {"route_bands": "@current_epoch"}


def _resolve_directives(args: dict) -> dict:
    out = dict(args)
    for key, token in RESOLVERS.items():
        if out.get(key) != token:
            continue
        if key != "route_bands":
            raise LaneArgsError(f"no resolver implemented for {key!r}")
        try:
            import route_floors
        except ImportError as exc:                       # pragma: no cover - import guard
            raise LaneArgsError(f"cannot import route_floors to resolve {token}: {exc}") from exc
        try:
            out[key] = route_floors.resolve()
        except route_floors.FloorsUnavailable as exc:
            raise LaneArgsError(
                f"{key!r} is {token!r} but this epoch cannot supply a measured table: {exc}") from exc
    return out


def load(path: Path) -> tuple[dict, dict]:
    """Return (args, declared_requirements). `_`-prefixed keys are comments."""
    try:
        raw = json.loads(path.read_text())
    except OSError as exc:
        raise LaneArgsError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LaneArgsError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise LaneArgsError(f"{path} must hold a JSON OBJECT of arguments, got {type(raw).__name__}")
    require = raw.get("_require") or {}
    if not isinstance(require, dict):
        raise LaneArgsError(f"{path}: `_require` must be an object of {{key: expected_value}}")
    return _resolve_directives(
        {k: v for k, v in raw.items() if not k.startswith("_")}), require


def _parse_cli_require(items: list[str]) -> dict:
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise LaneArgsError(f"--require expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def check(path: Path, extra_require: dict | None = None) -> list[str]:
    """Every problem with the file, as a list. Empty means launchable."""
    args, declared = load(path)
    problems: list[str] = []

    accepted = known_args(LANE_JS) | known_args(DISPATCH_JS)
    for key in sorted(k for k in args if k not in accepted):
        problems.append(
            f"unknown argument {key!r}: no entry point reads it, so it would be silently ignored")

    for key in REQUIRED_KEYS:
        if not args.get(key):
            problems.append(f"missing required argument {key!r}")

    # The omission check. `_require` is the protocol's own statement of the values
    # that must not drift; an absent key here is the failure this whole module is
    # about, so it is reported as an OMISSION rather than as a mismatch against a
    # default -- the two read very differently to whoever is fixing it.
    want = dict(declared)
    want.update(extra_require or {})
    for key, expected in sorted(want.items()):
        if key not in args:
            problems.append(
                f"{key!r} is required by this lane's protocol at {expected!r} but is ABSENT. It would "
                "take its built-in default, which is indistinguishable from a chosen value once the "
                "run starts")
        elif args[key] != expected:
            problems.append(f"{key!r} is {args[key]!r} but this lane's protocol says {expected!r}")

    for key in ("kernel_path", "workflow_dir", "state_dir", "exp_root", "eval_dir"):
        val = args.get(key)
        if isinstance(val, str) and val and not val.startswith("/"):
            problems.append(f"{key!r} must be an absolute path, got {val!r}")
    return problems


def render(path: Path) -> str:
    """The exact Workflow() call, so nobody retypes the argument object."""
    args, _ = load(path)
    script = f"{args.get('workflow_dir', '<WF_DIR>')}/kernel_workflow.js"
    # Argument ORDER follows the file, because that is what a human reads and diffs. Nested keys are
    # sorted, so a structured value like `op_spec` renders byte-identically every time and two
    # renderings of one file can be compared by diff.
    body = ",\n".join(f"    {k}: {json.dumps(v, sort_keys=True)}" for k, v in args.items())
    return f'Workflow({{\n  scriptPath: "{script}",\n  args: {{\n{body}\n  }}\n}})'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="validate; exit 3 on any problem")
    mode.add_argument("--print", dest="show", action="store_true", help="render the Workflow() call")
    mode.add_argument("--json", action="store_true", help="canonical one-line args object")
    ap.add_argument("--require", action="append", default=[], metavar="KEY=VALUE",
                    help="assert an argument is present AND equal to this value")
    a = ap.parse_args(argv)

    try:
        if a.show:
            print(render(a.file))
            return 0
        if a.json:
            args, _ = load(a.file)
            print(json.dumps(args, sort_keys=True))
            return 0
        problems = check(a.file, _parse_cli_require(a.require))
    except LaneArgsError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    if problems:
        print(f"REFUSED: {a.file} is not launchable ({len(problems)} problem(s)):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 3
    args, declared = load(a.file)
    resolved = sorted(k for k, tok in RESOLVERS.items() if k in args and args[k] != tok)
    print(f"OK: {a.file} -- {len(args)} arguments, all accepted by the entry points"
          + (f"; {len(declared)} protocol value(s) pinned and matching" if declared else "")
          + ("".join(f"; {k} resolved at launch to {len(args[k])} entries" for k in resolved
             if isinstance(args[k], dict))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
