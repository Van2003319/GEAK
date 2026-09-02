#!/usr/bin/env python3
"""Canonical same-session paired statistical commit gate.

The live workflow invokes this module and consumes its versioned receipt. No
JavaScript implementation of the statistical rule exists. Evidence is one raw,
paired block per route: candidate and untouched-incumbent observations carry
the lane-generated session and adapter-derived pair ids, and the complete
measured route set must match the suite manifest exactly. A mandatory
measurement binding pins the protocol hash, GPU lock, priming, slot/raw and
output-lifecycle invariants, immutable roots, and source task canonical SHA.

The estimand is a 20%-trimmed mean of ``log(control_ms / candidate_ms)``. A
fixed-seed paired percentile bootstrap supplies one-sided 95% lower confidence
bounds. The pinned algorithm pins 4,096 resamples, SplitMix64 stream derivation, R-type-7
5th-percentile interpolation, and seed 20260902; every parameter is repeated in
the v2 receipt. Acceptance is an intersection-union decision:

* every declared target route's lower bound exceeds its practical MDE;
* every other route's lower bound exceeds its predeclared non-regression margin;
* the suite lower bound exceeds its predeclared non-inferiority margin; and
* no route crosses the catastrophic-regression fence.

All endpoints must pass, so multiple targets/routes do not create an "any one
of many" false-positive path. Missing raw pairs, session provenance, control
observations, targets, or complete-suite coverage produces an INVALID_EVIDENCE
receipt and therefore refuses the candidate.

The ``legacy_*`` helpers at the bottom preserve offline compatibility for old
min-max/median-band notebooks. They are intentionally isolated and are never
consulted by :func:`decide` or the CLI; a spread band is descriptive calibration,
not a confidence interval or hypothesis test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

INPUT_SCHEMA = "geak.route-gate-input/v2"
RECEIPT_SCHEMA = "geak.route-gate-receipt/v2"
SCHEMA = RECEIPT_SCHEMA  # compatibility for receipt readers
ALGORITHM = "paired-log-trimmed-bootstrap/v1"

CONFIDENCE = 0.95
TRIM_FRACTION = 0.20
MIN_PAIRS = 8
BOOTSTRAP_RESAMPLES = 4096
BOOTSTRAP_SEED = 20260902
DEFAULT_PRACTICAL_MDE = 0.02
DEFAULT_NON_REGRESSION_MARGIN = 0.02
DEFAULT_SUITE_NON_INFERIORITY_MARGIN = 0.01
CATASTROPHIC_REGRESSION = 0.10
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RouteGateError(ValueError):
    """The supplied evidence cannot support a gate decision."""


@dataclass(frozen=True)
class PairedRoute:
    route: str
    pair_ids: tuple[str, ...]
    log_ratios: tuple[float, ...]


class _SplitMix64:
    """Tiny specified PRNG so bootstrap draws do not depend on Python's RNG."""

    _MASK = (1 << 64) - 1

    def __init__(self, seed: int):
        self.state = seed & self._MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self._MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self._MASK
        return (z ^ (z >> 31)) & self._MASK

    def randbelow(self, n: int) -> int:
        return self.next_u64() % n


def _algorithm_receipt() -> dict[str, Any]:
    return {
        "name": ALGORITHM,
        "effect": "log(control_ms/candidate_ms)",
        "estimator": f"{int(TRIM_FRACTION * 100)}% trimmed mean",
        "trim_rule": "drop floor(n*0.20) observations from each tail",
        "confidence": CONFIDENCE,
        "interval": "one-sided paired percentile bootstrap lower bound",
        "quantile": "linear interpolation at p=0.05 (R type 7)",
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "rng": "SplitMix64; stream seed is first 64 bits of SHA256(seed:label), big-endian",
        "index_sampler": "next_u64 modulo n",
        "min_pairs": MIN_PAIRS,
        "multiple_endpoints": "intersection-union: every declared endpoint must pass",
        "catastrophic_statistic": "trimmed-mean point estimate of candidate/control latency change",
    }


def _json_number_normalize(value: object) -> object:
    # JSON has one numeric type; V8 reserializes 1.0 as 1. Normalize before hashing so the
    # Python producer and the independent JavaScript consumer have one canonical digest.
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_json_number_normalize(item) for item in value]
    if isinstance(value, tuple):
        return [_json_number_normalize(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_number_normalize(item) for key, item in value.items()}
    return value


def _input_digest(spec: object) -> str | None:
    try:
        raw = json.dumps(_json_number_normalize(spec), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(raw).hexdigest()


def invalid_receipt(reason: str, spec: object | None = None) -> dict[str, Any]:
    """Return a versioned fail-closed receipt for malformed/missing evidence."""
    session_id = None
    expected: list[str] = []
    targets: list[str] = []
    if isinstance(spec, Mapping):
        if isinstance(spec.get("session_id"), str):
            session_id = spec.get("session_id")
        if isinstance(spec.get("expected_routes"), list):
            expected = [str(v) for v in spec["expected_routes"]]
        if isinstance(spec.get("target_routes"), list):
            targets = [str(v) for v in spec["target_routes"]]
    return {
        "schema": RECEIPT_SCHEMA,
        "input_schema": spec.get("schema") if isinstance(spec, Mapping) else None,
        "decision": "invalid_evidence",
        "accepted": False,
        "reason": reason,
        "session_id": session_id,
        "expected_routes": expected,
        "expected_route_count": (spec.get("expected_route_count")
                                 if isinstance(spec, Mapping) else None),
        "target_routes": targets,
        "pair_ids": [],
        "input_sha256": _input_digest(spec),
        "measurement": (spec.get("measurement")
                        if isinstance(spec, Mapping) and isinstance(spec.get("measurement"), Mapping)
                        else None),
        "algorithm": _algorithm_receipt(),
        "effective_margins": None,
        "route_weights": None,
        "suite": None,
        "routes": [],
        "target_passed": [],
        "regressed": [],
        "catastrophic": [],
        "failures": [reason],
    }


def _finite_positive(value: object, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) <= 0):
        raise RouteGateError(f"{label} must be a finite positive number, got {value!r}")
    return float(value)


def _fraction(value: object, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or not 0 <= float(value) < 1):
        raise RouteGateError(f"{label} must be a finite fraction in [0, 1), got {value!r}")
    return float(value)


def _names(value: object, label: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise RouteGateError(f"{label} must be an array")
    out: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or not raw.strip():
            raise RouteGateError(f"{label} contains a blank/non-string route: {raw!r}")
        if raw != raw.strip():
            raise RouteGateError(f"{label} route names must not have surrounding whitespace: {raw!r}")
        name = raw
        if name in out:
            raise RouteGateError(f"{label} contains duplicate route {name!r}")
        out.append(name)
    if not out and not allow_empty:
        raise RouteGateError(f"{label} is empty")
    return out


def _arm(arm: object, label: str, expected_session: str) -> dict[str, float]:
    if not isinstance(arm, Mapping):
        raise RouteGateError(f"{label} raw block is missing")
    session = arm.get("session_id")
    if not isinstance(session, str) or not session.strip():
        raise RouteGateError(f"{label}.session_id is missing")
    if session != session.strip():
        raise RouteGateError(f"{label}.session_id must not have surrounding whitespace")
    if session != expected_session:
        raise RouteGateError(
            f"{label}.session_id={session!r} does not match gate session {expected_session!r}")
    pair_ids = arm.get("pair_ids")
    samples = arm.get("samples_ms")
    if not isinstance(pair_ids, list) or not isinstance(samples, list):
        raise RouteGateError(f"{label} requires raw pair_ids[] and samples_ms[]")
    if len(pair_ids) != len(samples):
        raise RouteGateError(
            f"{label} has {len(pair_ids)} pair id(s) but {len(samples)} sample(s)")
    if len(pair_ids) < MIN_PAIRS:
        raise RouteGateError(
            f"{label} has {len(pair_ids)} paired sample(s); need at least {MIN_PAIRS}")
    out: dict[str, float] = {}
    for index, (pair_id, sample) in enumerate(zip(pair_ids, samples)):
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise RouteGateError(f"{label}.pair_ids[{index}] is blank/non-string")
        if pair_id != pair_id.strip():
            raise RouteGateError(
                f"{label}.pair_ids[{index}] must not have surrounding whitespace")
        if pair_id in out:
            raise RouteGateError(f"{label} repeats pair id {pair_id!r}")
        out[pair_id] = _finite_positive(sample, f"{label}.samples_ms[{index}]")
    return out


def _paired_routes(spec: Mapping[str, object], expected_routes: Sequence[str],
                   session_id: str) -> tuple[list[PairedRoute], tuple[str, ...]]:
    raw_routes = spec.get("paired_routes")
    if not isinstance(raw_routes, list) or not raw_routes:
        raise RouteGateError("paired_routes is missing or empty; raw candidate/control pairs are required")
    parsed: dict[str, PairedRoute] = {}
    common_pair_ids: tuple[str, ...] | None = None
    for index, raw in enumerate(raw_routes):
        if not isinstance(raw, Mapping):
            raise RouteGateError(f"paired_routes[{index}] is not an object")
        route = raw.get("route")
        if not isinstance(route, str) or not route.strip():
            raise RouteGateError(f"paired_routes[{index}] has no route name")
        if route != route.strip():
            raise RouteGateError(
                f"paired_routes[{index}] route must not have surrounding whitespace")
        if route in parsed:
            raise RouteGateError(f"paired_routes contains route {route!r} twice")
        candidate = _arm(raw.get("candidate"), f"{route}.candidate", session_id)
        control = _arm(raw.get("control"), f"{route}.control", session_id)
        if set(candidate) != set(control):
            missing_cand = sorted(set(control) - set(candidate))
            missing_ctrl = sorted(set(candidate) - set(control))
            raise RouteGateError(
                f"route {route!r} candidate/control pair ids differ"
                f" (candidate missing {missing_cand}, control missing {missing_ctrl})")
        pair_ids = tuple(sorted(candidate))
        if common_pair_ids is None:
            common_pair_ids = pair_ids
        elif pair_ids != common_pair_ids:
            raise RouteGateError(
                f"route {route!r} does not cover the same paired blocks as the full suite")
        # Subtract logs rather than dividing first. Both observations are finite and positive,
        # but their quotient can still overflow for otherwise valid IEEE-754 inputs.
        logs = tuple(math.log(control[pair_id]) - math.log(candidate[pair_id])
                     for pair_id in pair_ids)
        parsed[route] = PairedRoute(route=route, pair_ids=pair_ids, log_ratios=logs)

    expected_set = set(expected_routes)
    measured_set = set(parsed)
    if measured_set != expected_set:
        missing = sorted(expected_set - measured_set)
        extra = sorted(measured_set - expected_set)
        raise RouteGateError(
            "paired evidence is not the complete declared suite: "
            f"missing={missing or '[]'}, extra={extra or '[]'}")
    assert common_pair_ids is not None
    return [parsed[route] for route in sorted(parsed)], common_pair_ids


def _route_values(raw: object, routes: Sequence[str], all_routes: Sequence[str],
                  default: float, label: str) -> dict[str, float]:
    if raw is None:
        return {route: default for route in routes}
    if isinstance(raw, Mapping):
        if any(not isinstance(key, str) for key in raw):
            raise RouteGateError(f"{label} map keys must be route names")
        keys = set(raw)
        wanted, complete = set(routes), set(all_routes)
        # An exact applicable-route map is convenient for a single fixed target set. A complete
        # suite map is the reusable launch-time form: different planned candidates may target
        # different routes, and the gate selects only the entries applicable to this candidate.
        if keys not in (wanted, complete):
            raise RouteGateError(
                f"{label} map must name either exactly its applicable routes or the complete suite: "
                f"applicable_missing={sorted(wanted - keys)}, "
                f"suite_missing={sorted(complete - keys)}, extra={sorted(keys - complete)}")
        validated = {route: _fraction(raw[route], f"{label}[{route!r}]")
                     for route in keys}
        return {route: validated[route] for route in routes}
    value = _fraction(raw, label)
    return {route: value for route in routes}


def _weights(raw: object, routes: Sequence[str]) -> dict[str, float]:
    if raw is None:
        return {route: 1.0 / len(routes) for route in routes}
    if not isinstance(raw, Mapping):
        raise RouteGateError("route_weights must be an object naming every expected route")
    if any(not isinstance(key, str) for key in raw):
        raise RouteGateError("route_weights keys must be route names")
    keys, wanted = set(raw), set(routes)
    if keys != wanted:
        raise RouteGateError(
            "route_weights must name the complete suite: "
            f"missing={sorted(wanted - keys)}, extra={sorted(keys - wanted)}")
    vals = {route: _finite_positive(raw[route], f"route_weights[{route!r}]")
            for route in routes}
    total = sum(vals.values())
    if not math.isfinite(total) or total <= 0:
        raise RouteGateError("route_weights total must be finite and positive")
    return {route: vals[route] / total for route in routes}


def _hash_record(value: object, label: str) -> None:
    if not isinstance(value, Mapping):
        raise RouteGateError(f"{label} is missing")
    for kind in ("source", "asm", "binary"):
        row = value.get(kind)
        if (not isinstance(row, Mapping) or row.get("algorithm") != "sha256"
                or not isinstance(row.get("sha256"), str)
                or SHA256_RE.fullmatch(row["sha256"]) is None
                or isinstance(row.get("files"), bool) or not isinstance(row.get("files"), int)
                or row["files"] < 0 or not isinstance(row.get("present"), bool)
                or row["present"] != (row["files"] > 0)):
            raise RouteGateError(f"{label}.{kind} is not a canonical hash record")


def _lifecycle_row(value: object, label: str) -> None:
    if not isinstance(value, Mapping):
        raise RouteGateError(f"{label} is missing")
    ring, observed, unique = (value.get("ring_size"), value.get("observed_addresses"),
                              value.get("unique_addresses"))
    if (isinstance(ring, bool) or not isinstance(ring, int) or ring < 2
            or isinstance(observed, bool) or not isinstance(observed, int) or observed < 2
            or isinstance(unique, bool) or not isinstance(unique, int) or unique < 2
            or unique > observed or value.get("addresses_rotated") is not True):
        raise RouteGateError(f"{label} does not prove a rotating live output ring")


def _measurement_binding(spec: Mapping[str, object], routes: Sequence[str]) -> dict[str, Any]:
    raw = spec.get("measurement")
    if not isinstance(raw, Mapping) or raw.get("schema") != "geak.paired-verify-binding/v2":
        raise RouteGateError("measurement binding is missing or not paired-verify-binding/v2")
    if raw.get("producer_schema") != "geak.paired-verify-task-receipt/v2":
        raise RouteGateError("measurement producer schema is not paired-verify-task-receipt/v2")
    if raw.get("promotion_control") != "incumbent":
        raise RouteGateError("measurement promotion control is not the untouched incumbent")
    if raw.get("oracle_role") != "absolute_report_only":
        raise RouteGateError("measurement oracle is not report-only")
    canonical_input = raw.get("canonical_input_sha256")
    if not isinstance(canonical_input, str) or SHA256_RE.fullmatch(canonical_input) is None:
        raise RouteGateError("measurement canonical input SHA-256 is missing")
    protocol = raw.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("schema") != "geak.measurement-protocol/v2":
        raise RouteGateError("measurement protocol/v2 object is missing")
    protocol_payload = dict(protocol)
    protocol_digest = protocol_payload.pop("hash", None)
    recomputed = _input_digest(protocol_payload)
    if (not isinstance(protocol_digest, str) or protocol_digest != recomputed
            or raw.get("protocol_hash") != protocol_digest):
        raise RouteGateError("measurement protocol hash does not match canonical contents")
    if raw.get("metric_id") != "single_call_latency":
        raise RouteGateError("promotion measurement must be single_call_latency")
    lock = raw.get("gpu_lock")
    if (not isinstance(lock, Mapping) or lock.get("mode") != "external"
            or not isinstance(lock.get("path"), str) or not lock.get("path")):
        raise RouteGateError("measurement does not bind one external GPU lock")
    priming = raw.get("priming")
    warmups = protocol.get("warmup_cycles")
    if (not isinstance(priming, Mapping) or priming.get("required") is not True
            or priming.get("all_samples_primed") is not True
            or priming.get("timer") != "cuda_event"
            or priming.get("warmup_cycles") != warmups
            or isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0):
        raise RouteGateError("measurement priming/timer binding is incomplete")
    invariants = raw.get("invariants")
    required = (
        "all_arms_in_every_block", "same_slot_within_block",
        "same_outer_blocks_across_suite", "single_call_latency_inner_is_one",
        "semantic_slot_policy_bound",
    )
    if not isinstance(invariants, Mapping) or any(invariants.get(key) is not True for key in required):
        raise RouteGateError("measurement slot/raw invariants are incomplete")
    lifecycle = raw.get("output_lifecycle")
    per_route = lifecycle.get("per_route") if isinstance(lifecycle, Mapping) else None
    if (not isinstance(lifecycle, Mapping) or lifecycle.get("policy") != "bounded-live-ring"
            or lifecycle.get("all_arms_rotated") is not True or not isinstance(per_route, Mapping)
            or set(per_route) != set(routes)):
        raise RouteGateError("measurement output lifecycle does not cover the complete suite")
    for route in routes:
        arms = per_route.get(route)
        if not isinstance(arms, Mapping):
            raise RouteGateError(f"measurement output lifecycle is missing route {route!r}")
        _lifecycle_row(arms.get("candidate"), f"{route}.candidate output lifecycle")
        _lifecycle_row(arms.get("incumbent"), f"{route}.incumbent output lifecycle")
    receipt_hashes = raw.get("route_measurement_sha256")
    if (not isinstance(receipt_hashes, Mapping) or set(receipt_hashes) != set(routes)
            or any(not isinstance(value, str) or SHA256_RE.fullmatch(value) is None
                   for value in receipt_hashes.values())):
        raise RouteGateError("measurement receipt SHA-256 map does not cover the complete suite")
    if raw.get("incumbent_unchanged") is not True:
        raise RouteGateError("measurement does not prove the incumbent remained unmodified")
    _hash_record(raw.get("candidate_hashes"), "measurement.candidate_hashes")
    _hash_record(raw.get("incumbent_hashes"), "measurement.incumbent_hashes")
    _hash_record(raw.get("task_hashes"), "measurement.task_hashes")
    if not isinstance(raw.get("task_root"), str) or not raw["task_root"].startswith("/"):
        raise RouteGateError("measurement immutable task_root is missing")
    # Canonicalise through JSON so the receipt echoes exactly serialisable evidence and no
    # Mapping subclass can mutate under the decision.
    return json.loads(json.dumps(raw, sort_keys=True, allow_nan=False))


def _trimmed_mean(values: Sequence[float]) -> float:
    ordered = sorted(values)
    cut = int(len(ordered) * TRIM_FRACTION)
    kept = ordered[cut:len(ordered) - cut] if cut else ordered
    if not kept:
        raise RouteGateError("trimmed estimator has no observations")
    return sum(kept) / len(kept)


def _quantile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _stream_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _receipt_exp(value: float, label: str) -> float:
    """Exponentiate only when the result remains finite JSON receipt data."""
    try:
        result = math.exp(value)
    except OverflowError as exc:
        raise RouteGateError(f"{label} is outside the finite receipt range") from exc
    if not math.isfinite(result) or result <= 0:
        raise RouteGateError(f"{label} is outside the finite positive receipt range")
    return result


def _estimate(values: Sequence[float], label: str) -> tuple[float, float]:
    point = _trimmed_mean(values)
    rng = _SplitMix64(_stream_seed(label))
    n = len(values)
    boot: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        boot.append(_trimmed_mean([values[rng.randbelow(n)] for _ in range(n)]))
    lower = _quantile(boot, 1.0 - CONFIDENCE)
    return point, lower


def _route_record(route: PairedRoute, role: str, limit: float,
                  limit_kind: str) -> dict[str, Any]:
    estimate_log, lower_log = _estimate(route.log_ratios, f"route:{route.route}")
    estimate_ratio = _receipt_exp(estimate_log, f"route {route.route!r} estimate ratio")
    candidate_over_control = _receipt_exp(
        -estimate_log, f"route {route.route!r} latency change")
    catastrophic = estimate_log < -math.log1p(CATASTROPHIC_REGRESSION)
    passed = lower_log > limit and not catastrophic
    if catastrophic:
        status = "catastrophic"
    elif role == "target":
        status = "target_mde_pass" if passed else "target_mde_not_proven"
    else:
        status = "non_inferior" if passed else "non_inferiority_not_proven"
    return {
        "route": route.route,
        "role": role,
        "n_pairs": len(route.log_ratios),
        "bootstrap_stream_seed": f"0x{_stream_seed(f'route:{route.route}'):016x}",
        "estimate_log_ratio": estimate_log,
        "estimate_ratio": estimate_ratio,
        "estimate_latency_change": candidate_over_control - 1.0,
        "ci95_lower_log_ratio": lower_log,
        "ci95_lower_ratio": _receipt_exp(
            lower_log, f"route {route.route!r} lower-bound ratio"),
        "threshold_log_ratio": limit,
        "threshold_kind": limit_kind,
        "passed": passed,
        "catastrophic": catastrophic,
        "status": status,
    }


def decide(spec: Mapping[str, object]) -> dict[str, Any]:
    """Evaluate a v2 gate specification and return a canonical receipt."""
    if not isinstance(spec, Mapping):
        raise RouteGateError("gate input must be a JSON object")
    if spec.get("schema") != INPUT_SCHEMA:
        raise RouteGateError(
            f"input schema must be {INPUT_SCHEMA!r}, got {spec.get('schema')!r}")
    session_id = spec.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise RouteGateError("session_id is missing; cross-session evidence is not admissible")
    if session_id != session_id.strip():
        raise RouteGateError("session_id must not have surrounding whitespace")

    expected_routes = sorted(_names(spec.get("expected_routes"), "expected_routes"))
    expected_route_count = spec.get("expected_route_count")
    if (isinstance(expected_route_count, bool) or not isinstance(expected_route_count, int)
            or expected_route_count <= 0):
        raise RouteGateError("expected_route_count must be a positive integer from the suite manifest")
    if expected_route_count != len(expected_routes):
        raise RouteGateError(
            f"suite manifest declares {expected_route_count} route(s), but expected_routes has "
            f"{len(expected_routes)}; a partial baseline table cannot define the suite")
    target_routes = sorted(_names(spec.get("target_routes"), "target_routes"))
    unknown_targets = sorted(set(target_routes) - set(expected_routes))
    if unknown_targets:
        raise RouteGateError(f"target_routes are not in expected_routes: {unknown_targets}")
    non_target_routes = sorted(set(expected_routes) - set(target_routes))

    paired, pair_ids = _paired_routes(spec, expected_routes, session_id)
    measurement = _measurement_binding(spec, expected_routes)
    route_by_name = {route.route: route for route in paired}
    practical_mde = _route_values(spec.get("practical_mde"), target_routes, expected_routes,
                                  DEFAULT_PRACTICAL_MDE, "practical_mde")
    non_regression = _route_values(spec.get("non_regression_margin"), non_target_routes,
                                   expected_routes, DEFAULT_NON_REGRESSION_MARGIN,
                                   "non_regression_margin")
    suite_margin = _fraction(
        spec.get("suite_non_inferiority_margin",
                 DEFAULT_SUITE_NON_INFERIORITY_MARGIN),
        "suite_non_inferiority_margin")
    weights = _weights(spec.get("route_weights"), expected_routes)

    records: list[dict[str, Any]] = []
    for route in expected_routes:
        if route in practical_mde:
            limit = -math.log1p(-practical_mde[route])
            records.append(_route_record(route_by_name[route], "target", limit,
                                         "practical_mde"))
        else:
            limit = -math.log1p(non_regression[route])
            records.append(_route_record(route_by_name[route], "non_target", limit,
                                         "non_regression_margin"))

    # Resample complete pair blocks for the suite, preserving common-mode noise
    # and correlation between routes instead of pretending endpoints independent.
    suite_blocks: list[float] = []
    for pair_index, _pair_id in enumerate(pair_ids):
        suite_blocks.append(sum(
            weights[route] * route_by_name[route].log_ratios[pair_index]
            for route in expected_routes))
    suite_estimate, suite_lower = _estimate(suite_blocks, "suite")
    suite_limit = -math.log1p(suite_margin)
    suite_passed = suite_lower > suite_limit
    suite = {
        "n_pairs": len(pair_ids),
        "bootstrap_stream_seed": f"0x{_stream_seed('suite'):016x}",
        "estimate_log_ratio": suite_estimate,
        "estimate_ratio": _receipt_exp(suite_estimate, "suite estimate ratio"),
        "ci95_lower_log_ratio": suite_lower,
        "ci95_lower_ratio": _receipt_exp(suite_lower, "suite lower-bound ratio"),
        "non_inferiority_margin": suite_margin,
        "threshold_log_ratio": suite_limit,
        "passed": suite_passed,
    }

    target_passed = [r["route"] for r in records
                     if r["role"] == "target" and r["passed"]]
    catastrophic = [r["route"] for r in records if r["catastrophic"]]
    regressed = [r["route"] for r in records
                 if r["role"] == "non_target" and not r["passed"]]
    failures: list[str] = []
    if catastrophic:
        failures.append(
            f"catastrophic regression past {CATASTROPHIC_REGRESSION:.1%}: "
            + ", ".join(catastrophic))
    failed_targets = [r["route"] for r in records
                      if r["role"] == "target" and not r["passed"]]
    if failed_targets:
        failures.append("target MDE not proven at one-sided 95%: "
                        + ", ".join(failed_targets))
    if regressed:
        failures.append("non-target non-regression not proven at one-sided 95%: "
                        + ", ".join(regressed))
    if not suite_passed:
        failures.append("suite non-inferiority not proven at one-sided 95%")

    accepted = not failures
    if accepted:
        reason = ("all target MDE, non-target non-regression, and suite non-inferiority "
                  "bounds passed; no catastrophic regression")
        decision = "accept"
    else:
        reason = "; ".join(failures)
        decision = "refuse"

    return {
        "schema": RECEIPT_SCHEMA,
        "input_schema": INPUT_SCHEMA,
        "decision": decision,
        "accepted": accepted,
        "reason": reason,
        "session_id": session_id,
        "expected_routes": expected_routes,
        "expected_route_count": expected_route_count,
        "target_routes": target_routes,
        "pair_ids": list(pair_ids),
        "input_sha256": _input_digest(spec),
        "measurement": measurement,
        "algorithm": _algorithm_receipt(),
        "effective_margins": {
            "practical_mde": practical_mde,
            "non_regression_margin": non_regression,
            "suite_non_inferiority_margin": suite_margin,
            "catastrophic_regression": CATASTROPHIC_REGRESSION,
        },
        "route_weights": weights,
        "suite": suite,
        "routes": records,
        "target_passed": target_passed,
        "regressed": regressed,
        "catastrophic": catastrophic,
        "failures": failures,
    }


def decide_fail_closed(spec: object) -> dict[str, Any]:
    try:
        return decide(spec)  # type: ignore[arg-type]
    except (RouteGateError, KeyError, TypeError, ValueError, ArithmeticError) as exc:
        return invalid_receipt(str(exc), spec)


# ---------------------------------------------------------------------------
# LEGACY DESCRIPTIVE CALIBRATION ONLY -- NEVER USED BY decide() OR THE CLI.
# ---------------------------------------------------------------------------
def legacy_minmax_median_band(values: Sequence[float]) -> float:
    """Old descriptive ``(max-min)/median`` spread; NOT a statistical test."""
    vals = [_finite_positive(v, "legacy sample") for v in values]
    if len(vals) < 3:
        raise RouteGateError("legacy spread requires at least three values")
    return (max(vals) - min(vals)) / statistics.median(vals)


def legacy_bands_from_samples(per_case: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Compatibility reader for old offline reports; not accepted as gate evidence."""
    out: dict[str, float] = {}
    for row in per_case or []:
        route = row.get("name") or row.get("test_case_id")
        if not isinstance(route, str) or not route:
            raise RouteGateError("legacy row has no route name")
        if route in out:
            raise RouteGateError(f"legacy route {route!r} appears twice")
        samples = row.get("samples_ms")
        if not isinstance(samples, (list, tuple)):
            raise RouteGateError(f"legacy route {route!r} has no samples_ms")
        out[route] = legacy_minmax_median_band(samples)
    if not out:
        raise RouteGateError("legacy per_case is empty")
    return out


def legacy_bands_from_repeats(reports: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Old full-report transpose for offline calibration; never live gate evidence."""
    if len(reports) < 3:
        raise RouteGateError("legacy spread requires at least three reports")
    per_route: dict[str, list[float]] = {}
    expected: set[str] | None = None
    for report_index, report in enumerate(reports):
        rows = report.get("test_cases") or report.get("per_case")
        if not isinstance(rows, (list, tuple)) or not rows:
            raise RouteGateError(f"legacy report {report_index} has no per-case rows")
        current: set[str] = set()
        for row in rows:
            route = row.get("name") or row.get("test_case_id")
            if not isinstance(route, str) or not route or route in current:
                raise RouteGateError(f"legacy report {report_index} has an invalid/duplicate route")
            current.add(route)
            value = row.get("optimized_ms")
            if value is None:
                value = row.get("latency_ms")
            per_route.setdefault(route, []).append(
                _finite_positive(value, f"legacy report {report_index} route {route!r}"))
        if expected is None:
            expected = current
        elif current != expected:
            raise RouteGateError("legacy reports do not cover the same route set")
    return {route: legacy_minmax_median_band(values)
            for route, values in sorted(per_route.items())}


# Old import names retained for notebooks, explicitly pointing at legacy-only APIs.
bands_from_samples = legacy_bands_from_samples
bands_from_repeats = legacy_bands_from_repeats


def _print_human(receipt: Mapping[str, object]) -> None:
    print(f"{str(receipt.get('decision', 'invalid_evidence')).upper()}: {receipt.get('reason', '')}")
    suite = receipt.get("suite")
    if isinstance(suite, Mapping):
        print("suite estimate %.5fx, one-sided 95%% lower %.5fx" %
              (suite["estimate_ratio"], suite["ci95_lower_ratio"]))
    routes = receipt.get("routes")
    if isinstance(routes, list):
        for row in routes:
            print("%-24s %-10s estimate=%+.3f%% lower=%+.3f%% %s" % (
                row["route"], row["role"], row["estimate_latency_change"] * -100,
                (1.0 - 1.0 / row["ci95_lower_ratio"]) * 100, row["status"]))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="v2 JSON spec path, or '-' for stdin")
    ap.add_argument("--json", action="store_true", help="emit canonical JSON receipt")
    args = ap.parse_args(argv)
    spec: object | None = None
    try:
        if args.spec == "-":
            spec = json.load(sys.stdin)
        else:
            with open(args.spec, encoding="utf-8") as fh:
                spec = json.load(fh)
        receipt = decide_fail_closed(spec)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        receipt = invalid_receipt(f"cannot read gate input: {exc}", spec)
    if args.json:
        print(json.dumps(receipt, indent=1, sort_keys=True, allow_nan=False))
    else:
        _print_human(receipt)
    return 0 if receipt["accepted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
