#!/usr/bin/env python3
"""Per-route acceptance for a candidate, judged in absolute microseconds.

Replaces the arithmetic behind `kernel_lane.js`'s commit gate

    improved = winner.geomean > cumulative * (1 + MIN_IMPROVE)

which contradicts this repo's own claim rule. COMMANDMENT and the tech-lead
close-out both say a single-route mechanism must be judged by that route's
absolute microseconds, with the suite geomean used only to confirm nothing else
regressed. The gate did the opposite: it judged on the suite geometric mean,
which divides an eleven-route suite's single-route win by roughly eleven. A real
+7% route mechanism arrives at that gate reading +0.6%, i.e. under the noise,
and a 2% threshold on top of that is equivalent to "never commit".

One fact from the instrument calibration on this lane (24 repeats of one
UNCHANGED tree, §13.1 of PIPELINE_PROGRESS_GREEDY.md) sets the design:
per-route repeat bands differ by ~7x across routes -- 1.56% on
prefill_m128_square, 11.55% on decode_m2_square. A single threshold is therefore
wrong in both directions at once, so this module REFUSES a route it has no band
for instead of defaulting one.

KNOWN AND UNGUARDED: the same unchanged tree measures 1.5-3% differently between
invocations, and the candidate and the incumbent it is compared against come
from different invocations on whichever pool GPU was free. That exposure is real
and larger than the per-round gains being judged, and nothing here screens for
it -- deliberately, by decision of the run owner. Read a marginal verdict with
that in mind: the tighter fix is not a device check but comparing against a
control measured in the candidate's own session.

This module decides ACCEPT / REFUSE for one candidate against one incumbent. It
does not rank candidates and has no notion of a better candidate, so it cannot
be turned into a selection signal by accident.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from typing import Mapping, Sequence

SCHEMA = "geak.route-gate/v1"

# Both mirrored from measure_noise_floor.py rather than imported, because that
# module pulls in the machine/epoch tables and this one must stay usable with
# nothing but a per_case list. `test_route_gate.py` asserts the two agree, so the
# duplication cannot drift silently.
#
# MIN_REPEATS: two samples cannot distinguish a spread from a pair.
# MIN_BAND: "the tightest floor ever measured anywhere in this ledger is 0.005;
# anything below 0.002 is the sampler running out of resolution, not the GPU
# being quiet." A band of 0.0 -- which three coarse-timer repeats produce
# readily -- would make every rounding difference read as `improved`, i.e. the
# gate would bank noise on the quietest routes instead of on the loudest.
MIN_REPEATS = 3
MIN_BAND = 0.002


class RouteGateError(ValueError):
    """The gate was asked to decide something it has no evidence for."""


@dataclass(frozen=True)
class RouteVerdict:
    route: str
    incumbent_ms: float
    candidate_ms: float
    delta_frac: float          # >0 means the candidate is FASTER
    band: float
    status: str                # improved | regressed | flat


@dataclass
class Decision:
    accepted: bool
    reason: str
    routes: list[RouteVerdict] = field(default_factory=list)
    improved: list[str] = field(default_factory=list)
    regressed: list[str] = field(default_factory=list)
    suite_geomean_speedup: float | None = None


def _rows(per_case: Sequence[Mapping[str, object]], which: str) -> dict[str, float]:
    """{route: optimized_ms}. Refuses a row that cannot be read as a time."""
    out: dict[str, float] = {}
    for row in per_case or []:
        name = row.get("name") or row.get("test_case_id")
        if not name:
            raise RouteGateError(f"{which}: a per_case row has no route name: {row!r}")
        ms = row.get("optimized_ms")
        if ms is None:
            ms = row.get("candidate_ms")
        if not (isinstance(ms, (int, float)) and ms > 0):
            raise RouteGateError(
                f"{which}: route {name!r} has no positive optimized_ms/candidate_ms "
                f"(got {ms!r}); a route with no time cannot be judged")
        if str(name) in out:
            raise RouteGateError(f"{which}: route {name!r} appears twice")
        out[str(name)] = float(ms)
    if not out:
        raise RouteGateError(f"{which}: per_case is empty")
    return out


def _band_from_values(vals: Sequence[float], route: str) -> float:
    """The band statistic itself, in ONE place so the two entry points cannot drift.

    Full min-max spread over the repeats, as a fraction of the median. Full spread
    rather than a MAD multiple on purpose: the gate's job is to not bank noise, and
    on this lane the distribution is a tight core with a left tail, so a
    MAD-derived band would admit the tail.
    """
    med = statistics.median(vals)
    if not med > 0:
        raise RouteGateError(f"route {route!r} has a non-positive median ({med!r})")
    return max((max(vals) - min(vals)) / med, MIN_BAND)


def bands_from_repeats(reports: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Derive a per-route band from N full-suite reports of ONE unchanged tree.

    Needs at least MIN_REPEATS reports; two cannot distinguish a spread from a pair.
    """
    if len(reports) < MIN_REPEATS:
        raise RouteGateError(
            f"need at least {MIN_REPEATS} repeats of the same tree to derive bands, "
            f"got {len(reports)}")
    per_route: dict[str, list[float]] = {}
    for rep in reports:
        for route, ms in _rows(rep.get("test_cases") or rep.get("per_case") or [],
                              "repeat").items():
            per_route.setdefault(route, []).append(ms)
    bands: dict[str, float] = {}
    for route, vals in per_route.items():
        if len(vals) != len(reports):
            raise RouteGateError(
                f"route {route!r} appears in {len(vals)} of {len(reports)} repeats; "
                "a band derived from a subset would understate the spread")
        bands[route] = _band_from_values(vals, route)
    return bands


def bands_from_samples(per_case: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Derive per-route bands from a baseline table that carries `samples_ms`.

    Same statistic as `bands_from_repeats`, reading the ALREADY-TRANSPOSED shape
    the benchmark engineer reports: one row per route, each carrying its own
    repeat list (`baseline_per_case[].samples_ms`, benchmark_engineer.md step 5,
    "the raw per-repeat latency for that case -- at least three complete primed
    repeats, in the order measured, not a summary").

    Why this entry point exists at all. `bands_from_repeats` wants N whole suite
    reports keyed on `optimized_ms`, and no caller in the lane has that in hand at
    the moment the commit gate is configured -- which is why the per-route gate
    shipped with no reachable band source and had to be fed a hand-maintained
    per-host JSON that nobody passed. The baseline table has the repeats on EVERY
    run, for EVERY task, measured on the box that will judge the candidates. That
    makes the gate reachable without any task-specific or host-specific file.

    Raises rather than returning a partial table, and the asymmetry is deliberate:
    a table missing one route makes `decide()` refuse that route with
    "no noise band for ...", which reads as a candidate defect, while an absent
    table lets the caller fall back to the suite gate and say so once.
    """
    per_route: dict[str, list[float]] = {}
    for row in per_case or []:
        name = row.get("name") or row.get("test_case_id")
        if not name:
            raise RouteGateError(f"a per_case row has no route name: {row!r}")
        name = str(name)
        if name in per_route:
            raise RouteGateError(f"route {name!r} appears twice")
        raw = row.get("samples_ms")
        if not isinstance(raw, (list, tuple)):
            raise RouteGateError(
                f"route {name!r} has no samples_ms list (got {raw!r}); the band is a "
                "measurement of how much this tree wanders and cannot be inferred from "
                "a single latency")
        vals: list[float] = []
        for v in raw:
            if not (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and v > 0):
                raise RouteGateError(
                    f"route {name!r} has a non-positive or non-finite sample ({v!r})")
            vals.append(float(v))
        if len(vals) < MIN_REPEATS:
            raise RouteGateError(
                f"route {name!r} has {len(vals)} sample(s), need at least {MIN_REPEATS}; "
                "two cannot distinguish a spread from a pair")
        per_route[name] = vals
    if not per_route:
        raise RouteGateError("per_case is empty; no bands to derive")
    return {route: _band_from_values(vals, route) for route, vals in per_route.items()}


def suite_geomean(per_case: Sequence[Mapping[str, object]]) -> float | None:
    """Suite geometric mean of per-route speedups, for REPORTING only.

    Deliberately not an input to the decision. It is carried so a receipt can
    still show the number every prior round was gated on, which is what makes a
    changed verdict auditable against the old one.
    """
    vals = [row.get("speedup") for row in per_case or []]
    vals = [float(v) for v in vals if isinstance(v, (int, float)) and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def decide(
    *,
    candidate_per_case: Sequence[Mapping[str, object]],
    incumbent_per_case: Sequence[Mapping[str, object]],
    bands: Mapping[str, float],
    target_routes: Sequence[str] | None = None,
) -> Decision:
    """ACCEPT iff some route improves past its own band and none regresses past its own.

    `target_routes` narrows where the improvement must land. A direction that
    declared a mechanism on one route does not get to bank a win that showed up
    somewhere else -- that is how a measurement artefact gets committed as a
    mechanism.
    """
    cand = _rows(candidate_per_case, "candidate")
    inc = _rows(incumbent_per_case, "incumbent")

    missing = sorted(set(inc) - set(cand))
    if missing:
        return Decision(
            accepted=False,
            reason=(f"candidate did not measure {len(missing)} incumbent route(s): "
                    f"{', '.join(missing)}. Non-regression cannot be claimed for a route "
                    "that was not measured"))

    no_band = sorted(r for r in inc if r not in bands)
    if no_band:
        raise RouteGateError(
            f"no noise band for route(s) {', '.join(no_band)}. Bands on this lane range "
            "from 2.68% to 16.44%, so a default would be wrong by up to 6x; derive them "
            "with bands_from_repeats() on repeats of the unchanged tree")

    verdicts: list[RouteVerdict] = []
    for route in sorted(inc):
        band = float(bands[route])
        if not (band >= 0):
            raise RouteGateError(f"band for {route!r} must be >= 0, got {band!r}")
        i_ms, c_ms = inc[route], cand[route]
        delta = (i_ms - c_ms) / i_ms
        if delta > band:
            status = "improved"
        elif -delta > band:
            status = "regressed"
        else:
            status = "flat"
        verdicts.append(RouteVerdict(route=route, incumbent_ms=i_ms, candidate_ms=c_ms,
                                     delta_frac=delta, band=band, status=status))

    improved = [v.route for v in verdicts if v.status == "improved"]
    regressed = [v.route for v in verdicts if v.status == "regressed"]
    geo = suite_geomean(candidate_per_case)

    def out(accepted: bool, reason: str) -> Decision:
        return Decision(accepted=accepted, reason=reason, routes=verdicts,
                        improved=improved, regressed=regressed,
                        suite_geomean_speedup=geo)

    if regressed:
        return out(False, "regressed past its own band on: " + ", ".join(
            f"{v.route} ({-v.delta_frac:+.2%} vs band {v.band:.2%})"
            for v in verdicts if v.status == "regressed"))

    wanted = improved
    # Truthiness, not `is not None`: an EMPTY list means "this direction declared no target
    # route", which must not narrow anything. Under `is not None` an empty list narrowed to
    # nothing and refused every candidate, and the JS twin in kernel_lane.js -- which is the
    # gate that actually decides a commit -- reads the same input as "no narrowing" and
    # ACCEPTS. Two opposite verdicts on one piece of evidence, and it is a reachable shape:
    # engineer.md only says "omit when the direction names none", and an agent filling a
    # declared array field it has nothing to say about returns [] as readily as it omits it.
    if target_routes:
        wanted = [r for r in improved if r in set(target_routes)]
        if not wanted:
            return out(False, (
                "no declared target route improved past its own band"
                + (f"; incidental gains on {', '.join(improved)} are not the claimed mechanism"
                   if improved else "")))

    if not wanted:
        return out(False, "no route improved past its own band (all flat)")

    return out(True, "improved past band on: " + ", ".join(
        f"{v.route} ({v.delta_frac:+.2%} vs band {v.band:.2%})"
        for v in verdicts if v.route in set(wanted)))


def receipt(d: Decision) -> dict:
    return {
        "schema": SCHEMA,
        "accepted": d.accepted,
        "reason": d.reason,
        "improved": d.improved,
        "regressed": d.regressed,
        # Reported, never gated on. See suite_geomean().
        "suite_geomean_speedup_reported_only": d.suite_geomean_speedup,
        "routes": [{"route": v.route, "incumbent_ms": v.incumbent_ms,
                    "candidate_ms": v.candidate_ms, "delta_frac": v.delta_frac,
                    "band": v.band, "status": v.status} for v in d.routes],
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="JSON: {candidate_per_case, incumbent_per_case, bands, "
                                 "target_routes?}")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    with open(args.spec) as fh:
        spec = json.load(fh)
    d = decide(candidate_per_case=spec["candidate_per_case"],
               incumbent_per_case=spec["incumbent_per_case"],
               bands=spec["bands"], target_routes=spec.get("target_routes"))
    rec = receipt(d)
    if args.json:
        print(json.dumps(rec, indent=1, sort_keys=True))
    else:
        print("%-22s %10s %10s %9s %8s  %s"
              % ("route", "inc_us", "cand_us", "delta", "band", "status"))
        for v in d.routes:
            print("%-22s %10.1f %10.1f %+8.2f%% %7.2f%%  %s"
                  % (v.route, v.incumbent_ms * 1e3, v.candidate_ms * 1e3,
                     v.delta_frac * 100, v.band * 100, v.status))
        if d.suite_geomean_speedup is not None:
            print("\nsuite geomean (reported, NOT gated on): %.5f" % d.suite_geomean_speedup)
        print("\n%s: %s" % ("ACCEPT" if d.accepted else "REFUSE", d.reason))
    return 0 if d.accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())
