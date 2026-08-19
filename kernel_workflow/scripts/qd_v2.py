#!/usr/bin/env python3
"""Single fail-closed CLI for deterministic geak-qd-v2 helpers."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import qd_descriptor_v2 as descriptor
import qd_robust_stats as robust
import qd_route_priority as priority_mod
import qd_sol_card as sol
import qd_source_hash as source


def _json_arg(value: str) -> object:
    """Accept either inline JSON or a path to a JSON file.

    The branch is decided by the FIRST NON-SPACE CHARACTER, not by asking the
    filesystem. Asking cost every real descriptor: `Path(value).is_file()`
    raises `OSError(ENAMETOOLONG)` once `value` exceeds the filename limit
    (~255 bytes) -- `pathlib` swallows only ENOENT/ENOTDIR/EBADF/ELOOP -- and
    the `except OSError` below then reported a perfectly good seven-axis
    descriptor as "invalid JSON or JSON file ...: File name too long". Short
    descriptors parsed, long ones did not, which reads as malformed JSON and
    sends the caller looking at their quoting.

    `{` and `[` cannot begin a useful path here and every argument this CLI
    takes is a JSON object or array, so the test is exact rather than a guess.
    """
    stripped = value.lstrip()
    if stripped[:1] in ("{", "["):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise argparse.ArgumentTypeError(f"invalid inline JSON: {exc}") from exc
    try:
        return json.loads(Path(value).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON file {value!r}: {exc}") from exc


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _route_fields(value: Mapping[str, object]) -> dict[str, int]:
    """Accept only real `RouteFacts` integer fields, and reject the rest.

    Splatting caller JSON straight into the dataclass would turn a typo into a
    `TypeError` traceback, and -- worse for a refusal gate -- a misspelled
    `lds_bytes` would silently leave residency undetermined, which makes every
    rule abstain and the mutation read as allowed. A gate that fails open on a
    typo is not a gate, so unknown keys are an error here.
    """
    allowed = {f.name for f in dataclasses.fields(descriptor.RouteFacts)}
    unknown = sorted(k for k in value if k not in allowed)
    if unknown:
        raise ValueError(
            "unknown RouteFacts field(s): %s; legal fields are %s"
            % (", ".join(unknown), ", ".join(sorted(allowed))))
    out: dict[str, int] = {}
    for key, raw in value.items():
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"RouteFacts field {key!r} must be an integer")
        out[key] = raw
    return out


def _arch_argument(parser: argparse.ArgumentParser) -> None:
    """Add a `--arch` that has NO default, on every arch-sensitive subcommand.

    There used to be a default, and it was `gfx90a` on a fleet that has only
    ever been gfx942. Two things followed silently:

    1. `sol-card` scored against the gfx90a reference card, whose own `source`
       field calls it "not an effective measurement", instead of the gfx942
       card marked `measured: True`. On one fixed measurement that is
       `remaining_headroom` 0.241 vs 0.794 -- a route with four fifths of its
       headroom left reads as nearly closed, and the planner deprioritises it.
    2. `rasterization: xcd_remapped_grouped` -- the one mechanism in the
       vocabulary that describes gfx942's 8-XCD remap -- is unconditionally
       illegal on gfx90a, so every descriptor naming it was rejected with
       `rule:xcd_remap_requires_multi_die`. That reads as "this mechanism is
       wrong" when what was wrong was the unstated arch.

    Defaulting the other way would be the same bug pointed at the next box.
    A missing arch is not a value to guess, it is a question to ask, so this
    fails closed with argparse's own exit 2. `qd_route_priority.ARCH` is the
    only arch statement in this package that was right, and it is right
    because it is explicit.

    No `choices=` here on purpose: an unsupported arch is a refusal this CLI
    already reports as machine-readable JSON (`rule:unsupported_arch_or_dtype`,
    exit 2), and argparse's `choices` would replace that with prose on stderr.
    Required-ness is the part that must be enforced here, because argparse is
    the only layer that can see the flag is absent.
    """
    parser.add_argument(
        "--arch", required=True,
        help=f"target gfx arch, one of {', '.join(sorted(descriptor.SUPPORTED_ARCHES))}. "
             "REQUIRED and deliberately without a default: pass the card actually "
             "detected on-box.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    hash_tree = sub.add_parser("hash-tree", help="hash candidate-owned source/config/build inputs")
    hash_tree.add_argument("root")
    hash_tree.add_argument("--exclude", action="append", default=[])

    evidence = sub.add_parser("evidence", help="extract conservative source evidence for claims")
    evidence.add_argument("root")
    evidence.add_argument("--claim", action="append", required=True)
    evidence.add_argument("--metadata", type=_json_arg)
    evidence.add_argument(
        "--scope", action="append", metavar="PATH",
        help="restrict the search to this tree-relative file, or to a directory and "
             "everything under it (repeatable). Point it at the sources that actually "
             "build: a real workspace carries abandoned variants under research/ and "
             "saved snapshots beside the kernel itself (custom_gemm.hip.v100_...), "
             "whose text will happily ground a claim the shipped kernel does not "
             "support. Naming a file selects that file, not its snapshots. "
             "Documentation and data files are skipped regardless.")

    validate = sub.add_parser("validate-descriptor", help="validate one mechanism tuple")
    validate.add_argument("descriptor", type=_json_arg)
    validate.add_argument("--context")
    validate.add_argument("--known-context", action="append")
    _arch_argument(validate)
    validate.add_argument("--dtype", default="bf16")

    adjacent = sub.add_parser("adjacency", help="list legal named neighboring tuples")
    adjacent.add_argument("descriptor", type=_json_arg)
    _arch_argument(adjacent)
    adjacent.add_argument("--dtype", default="bf16")

    stats = sub.add_parser("robust-stats", help="compute per-context median/MAD intervals")
    stats.add_argument("samples", type=_json_arg,
                       help='JSON object or file: {"case": [sample, ...], ...}')

    floor = sub.add_parser(
        "noise-floor",
        help="smallest relative effect that is readable on a route (finding 26)")
    floor.add_argument("--context", action="append",
                       help="harness case id; repeatable. Omit for the whole table.")
    floor.add_argument("--effect", type=float, default=None,
                       help="proposed relative effect, e.g. 0.02 for 2%%. Exits 3 if it "
                            "is not readable on every named context.")

    priority = sub.add_parser(
        "route-priority",
        help="rank routes by remaining SOL slack against their own noise floor "
             "(finding 33)")
    priority.add_argument("--context", action="append",
                          help="harness case id; repeatable. Omit for the whole suite.")
    priority.add_argument("--elapsed-ms", type=_json_arg, default=None,
                          help='JSON object or file: {"case": ms, ...}. Overrides the '
                               "recorded ship-point latencies; pass this whenever the ship "
                               "point has moved.")
    priority.add_argument("--traffic-bytes", type=_json_arg, default=None,
                          help='JSON object or file: {"case": bytes, ...} of MEASURED DRAM '
                               "traffic (e.g. TCC_MISS_sum x 128). Without it a route falls "
                               "back to the compulsory minimum, which finding (34) measured "
                               "as 4.5x too small on the one route with counters.")

    verdict = sub.add_parser(
        "mutation-verdict",
        help="apply every measured refusal to one proposed route mutation")
    verdict.add_argument("--current", type=_json_arg, required=True,
                         help="JSON object or file: RouteFacts fields for the "
                              "route as it ships today")
    verdict.add_argument("--candidate", type=_json_arg, required=True,
                         help="JSON object or file: RouteFacts fields for the "
                              "mutation being proposed")

    card = sub.add_parser("sol-card", help="build one selected-route roofline card")
    card.add_argument("--flops", type=float, required=True)
    card.add_argument("--bytes", type=float, required=True, dest="bytes_min")
    card.add_argument("--elapsed-ms", type=float, required=True)
    card.add_argument("--dtype", default="bf16")
    # Required for the reason in `_arch_argument`, and most sharply here: this
    # is the subcommand whose answer changes by 3.3x between the two arches.
    # `choices` is kept because `build_sol_card` raises SOLCardError (not JSON)
    # on an unmodeled arch anyway, so argparse is the clearer of the two.
    card.add_argument("--arch", required=True, choices=sol.SUPPORTED_ARCHES)
    # Required for any arch whose ceiling is footprint-indexed (gfx942 is), and
    # NOT the same number as --bytes in general: --bytes is what the kernel
    # moved, the footprint is the distinct working set it moved it through.
    # They coincide for a streaming kernel, which is why omitting it defaults
    # to --bytes rather than failing.
    card.add_argument("--footprint-bytes", type=float, default=None)
    card.add_argument("--calibration", type=_json_arg)

    return parser


def _emit(payload: object) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "hash-tree":
            _emit({"schema": source.SCHEMA,
                   "source_hash": source.tree_hash(args.root,
                                                   extra_excluded_dirs=args.exclude)})
        elif args.command == "evidence":
            metadata = None if args.metadata is None else _mapping(args.metadata, "metadata")
            found = source.extract_descriptor_evidence(
                args.root, args.claim, metadata=metadata, scope=args.scope)
            # A bare null is two different answers wearing the same face, and
            # only one of them is actionable:
            #   - the claim has no grounding rule (`ungroundable`) -- expected,
            #     usually because the axis value IS an absence; carry on.
            #   - the claim has a rule and the rule found nothing
            #     (`unsubstantiated`) -- the source does not contain the
            #     construct the descriptor says it does. That is a mislabel
            #     signal, and a mislabeled cell poisons the archive worse than
            #     an empty one.
            # Reported, not enforced: exit stays 0. Whether a mislabel demotes
            # the route to "unclassified" is the verifier's call, because only
            # the verifier can see the disassembly that would overrule a text
            # match either way. A tool that cannot see the opcodes should not
            # be the one casting the deciding vote (finding 53).
            _emit({"schema": source.SCHEMA,
                   "evidence": found,
                   "ungroundable": {c: source.UNGROUNDABLE_CLAIMS[c]
                                    for c in found if c in source.UNGROUNDABLE_CLAIMS},
                   "unsubstantiated": sorted(
                       c for c, v in found.items()
                       if v is None and c in source.EVIDENCE_PATTERNS)})
        elif args.command == "validate-descriptor":
            value = _mapping(args.descriptor, "descriptor")
            # `reason` is the whole point of running this before submitting a
            # descriptor rather than after: `valid: false` alone tells an agent
            # that one of six rules refused and leaves it to guess which.
            # Finding (44) is the price of that guess -- every descriptor from
            # every agent was rejected for weeks over two axis names, and the
            # only visible symptom was an archive that stayed empty.
            reason = descriptor.descriptor_reject(value, arch=args.arch, dtype=args.dtype)
            valid = reason is None
            eligible = descriptor.coverage_eligible(value, arch=args.arch, dtype=args.dtype)
            cell = None
            if args.context is not None:
                cell = descriptor.cell_id(args.context, value,
                                          known_contexts=args.known_context,
                                          arch=args.arch, dtype=args.dtype)
            # A legal descriptor that is not coverage_eligible is the other
            # silent case: it occupies a cell but is never a directed-transition
            # target, which looks like being ignored unless it is said out loud.
            _emit({"classifier_version": descriptor.CLASSIFIER_VERSION, "valid": valid,
                   "reason": reason, "coverage_eligible": eligible, "cell": cell,
                   "ineligible_reason": (None if eligible or not valid
                                         else "rule:asymmetric_producer_consumer_not_a_target")})
            if not valid:
                return 2
        elif args.command == "adjacency":
            value = _mapping(args.descriptor, "descriptor")
            if not descriptor.descriptor_valid(value, arch=args.arch, dtype=args.dtype):
                raise ValueError("descriptor is not a legal geak-qd-v2 tuple")
            _emit({"classifier_version": descriptor.CLASSIFIER_VERSION,
                   "neighbors": [n.object() for n in descriptor.adjacency(
                       value, arch=args.arch, dtype=args.dtype)]})
        elif args.command == "noise-floor":
            names = args.context or sorted(robust.MEASURED_NOISE_FLOOR)
            rows = []
            for name in names:
                floor_value = robust.noise_floor(name)
                rows.append({
                    "context": name,
                    "noise_floor": floor_value,
                    "measured": name in robust.MEASURED_NOISE_FLOOR,
                    # An unmeasured route gets the widest floor measured on ANY
                    # machine -- which is wider than anything in the table above,
                    # since floors do not pool across a machine boundary -- so
                    # "readable" there is a claim about the fail-closed default,
                    # not about that route.
                    "readable": None if args.effect is None else args.effect > floor_value,
                })
            # The floors are epoch-specific, so a reader who cannot see which
            # epoch they came from cannot tell whether they apply to the numbers
            # being judged. Name it in the payload rather than leaving it to the
            # reader to know which box produced the JSON.
            payload = {"schema": robust.SCHEMA, "machine": robust.CURRENT_MACHINE,
                       "per_context": rows,
                       "default_noise_floor": robust.DEFAULT_NOISE_FLOOR,
                       "effect": args.effect}
            unreadable = [r["context"] for r in rows if r["readable"] is False]
            payload["unreadable_on"] = unreadable
            _emit(payload)
            if unreadable:
                # Exit 3, the same code `mutation-verdict` uses, because this is
                # the same kind of answer: a direction that cannot be measured
                # on its target route must not be built, however good it is.
                return 3
        elif args.command == "route-priority":
            elapsed = None
            if args.elapsed_ms is not None:
                elapsed = {k: float(v) for k, v in
                           _mapping(args.elapsed_ms, "elapsed-ms").items()}
            traffic = None
            if args.traffic_bytes is not None:
                traffic = {k: float(v) for k, v in
                           _mapping(args.traffic_bytes, "traffic-bytes").items()}
            rows = priority_mod.rank_routes(args.context, elapsed, traffic)
            closed = [r["context"] for r in rows if r["verdict"] == "closed"]
            # Finding (92): a row whose elapsed was defaulted from the shipped
            # kernel carries `needs_fresh_elapsed`, and the planner reads three
            # verdicts. Folding it into "not closed" would hand back exactly the
            # unlabelled optimism the finding is about, and folding it into
            # `closed` would resurrect the self-sealing loop. So it is a third
            # list, and `conditionally_closed` says what the stale number WOULD
            # have implied -- useful for deciding what to measure first, and
            # unusable as a reason to skip a route.
            unmeasured = [r["context"] for r in rows
                          if r["verdict"] == "needs_fresh_elapsed"]
            _emit({"schema": priority_mod.SCHEMA, "per_context": rows,
                   "richest": rows[0]["context"] if rows else None,
                   "closed": closed,
                   "needs_fresh_elapsed": unmeasured,
                   "conditionally_closed": [
                       r["context"] for r in rows
                       if r["verdict"] == "needs_fresh_elapsed"
                       and r["verdict_if_elapsed_confirmed"] == "closed"],
                   # Naming the routes ranked on a lower bound is the point of
                   # (34): their slack is understated and they are not
                   # comparable to a measured row without saying so.
                   "compulsory_traffic_routes": [r["context"] for r in rows
                                                 if r["traffic_basis"] == "compulsory"],
                   "elapsed_provenance": priority_mod.ELAPSED_PROVENANCE})
            # Exit 3 only when EVERY named route is closed, because that is the
            # case where the direction has nowhere to land. A mixed list is a
            # useful answer, not a refusal -- the planner drops the closed
            # entries from `target_cases` and keeps the rest.
            #
            # A `needs_fresh_elapsed` row can never make this fire, and that is
            # the (92) repair in one line: a route is refused only on its own
            # measurement, never on the shipped kernel's.
            if closed and len(closed) == len(rows):
                return 3
        elif args.command == "mutation-verdict":
            current = descriptor.RouteFacts(
                **_route_fields(_mapping(args.current, "current")))
            candidate = descriptor.RouteFacts(
                **_route_fields(_mapping(args.candidate, "candidate")))
            result = current.mutation_verdict(candidate)
            _emit({"classifier_version": descriptor.CLASSIFIER_VERSION,
                   "current": {"rounds": current.rounds,
                               "round_slack": current.round_slack,
                               "ctas": current.ctas,
                               "residency_slots": current.residency_slots},
                   "candidate": {"rounds": candidate.rounds,
                                 "round_slack": candidate.round_slack,
                                 "ctas": candidate.ctas,
                                 "residency_slots": candidate.residency_slots},
                   **result})
            if not result["allow"]:
                return 3
        elif args.command == "robust-stats":
            raw = _mapping(args.samples, "samples")
            samples = {}
            for name, values in raw.items():
                if not isinstance(name, str) or not isinstance(values, list):
                    raise ValueError("samples must map context strings to numeric arrays")
                samples[name] = values
            rows = robust.context_robust_stats(samples)
            _emit({"schema": robust.SCHEMA, "per_context": rows,
                   "combined": robust.combine_contexts(rows)})
        elif args.command == "sol-card":
            calibration = None if args.calibration is None else _mapping(
                args.calibration, "calibration")
            card = sol.build_sol_card(
                post_selection=True, achieved_flops=args.flops,
                achieved_bytes=args.bytes_min, elapsed_s=args.elapsed_ms / 1000.0,
                dtype=args.dtype, arch=args.arch, calibration=calibration,
                footprint_bytes=args.footprint_bytes)
            # (70). `validate_sol_card` existed, was thorough, was covered by its
            # own unit tests -- and was called by nothing outside them, which is
            # (55): a gate nothing invokes produces no output. It is the only
            # thing that enforces the v2 provenance fields, so the one path that
            # emits a card to an agent is where it has to run. A build that
            # cannot pass its own validator is a bug in this module, not a
            # degraded card to hand onward.
            problems = sol.validate_sol_card(card)
            if problems:
                raise ValueError("built an invalid SOL card: " + "; ".join(problems))
            _emit(card)
        else:  # pragma: no cover - argparse enforces the closed command set.
            raise ValueError(f"unknown command {args.command!r}")
    except (ValueError, OSError, TypeError) as exc:
        print(f"qd_v2: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
