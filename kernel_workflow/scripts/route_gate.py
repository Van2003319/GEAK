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

# Owned here. These used to be mirrored from a per-machine noise-floor sweep, with a test asserting
# the two copies agreed; that sweep and its machine/epoch tables are gone, so this module is now the
# single definition and there is no copy left to drift from.
#
# MIN_REPEATS: two samples cannot distinguish a spread from a pair.
# MIN_BAND: "the tightest floor ever measured anywhere in this ledger is 0.005;
# anything below 0.002 is the sampler running out of resolution, not the GPU
# being quiet." A band of 0.0 -- which three coarse-timer repeats produce
# readily -- would make every rounding difference read as `improved`, i.e. the
# gate would bank noise on the quietest routes instead of on the loudest.
MIN_REPEATS = 3
MIN_BAND = 0.002

# THE LIVE GATE IS `routeGate` IN kernel_lane.js. This module is its twin, and the twin is the copy
# that follows. Both numbers below are read out of the JS by `test_route_gate.py`'s parity test, so a
# change there fails here rather than drifting silently -- which is what happened before: the two
# disagreed on four rules at once (missing band, the suite condition, the catastrophic fence, and
# whether target_routes vetoes) and no test on either side could see it, because each guarded only
# its own implementation.
MIN_ROUTE_WIN = 0.02
CATASTROPHIC_REGRESSION = 0.10


class RouteGateError(ValueError):
    """The gate was asked to decide something it has no evidence for."""


@dataclass(frozen=True)
class RouteVerdict:
    route: str
    incumbent_ms: float
    candidate_ms: float
    delta_frac: float          # >0 means the candidate is FASTER
    band: float                # the route's measured floor, 0 when none was supplied
    need: float                # max(MIN_ROUTE_WIN, band) -- the bar it actually had to clear
    status: str                # won | regressed | catastrophic | flat


@dataclass
class Decision:
    accepted: bool
    reason: str
    routes: list[RouteVerdict] = field(default_factory=list)
    improved: list[str] = field(default_factory=list)
    regressed: list[str] = field(default_factory=list)
    catastrophic: list[str] = field(default_factory=list)
    # Geomean of incumbent_ms/candidate_ms over the PAIRED rows, i.e. both arms from one session.
    # Distinct from `suite_geomean_speedup`, which is the harness's own speedup column against the
    # frozen baseline and is carried for the receipt only.
    suite_ratio: float | None = None
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

    # A route with no measured band is NOT a refusal. This module used to raise here, on the
    # argument that a default would be wrong by up to 7x against the 1.56%-11.55% spread the lane
    # once measured. The producer of those bands is deleted, so raising now means refusing every
    # candidate; `MIN_ROUTE_WIN` is a floor under the floor, and a measured band can only RAISE it.
    verdicts: list[RouteVerdict] = []
    log_sum = 0.0
    for route in sorted(inc):
        band = float(bands.get(route) or 0.0)
        if not (band >= 0):
            raise RouteGateError(f"band for {route!r} must be >= 0, got {band!r}")
        need = max(MIN_ROUTE_WIN, band)
        i_ms, c_ms = inc[route], cand[route]
        delta = (i_ms - c_ms) / i_ms
        log_sum += math.log(i_ms / c_ms)
        if delta > need:
            status = "won"
        elif -delta > CATASTROPHIC_REGRESSION:
            status = "catastrophic"
        elif -delta > band:
            status = "regressed"
        else:
            status = "flat"
        verdicts.append(RouteVerdict(route=route, incumbent_ms=i_ms, candidate_ms=c_ms,
                                     delta_frac=delta, band=band, need=need, status=status))

    improved = [v.route for v in verdicts if v.status == "won"]
    catastrophic = [v.route for v in verdicts if v.status == "catastrophic"]
    regressed = [v.route for v in verdicts if v.status == "regressed"] + catastrophic
    suite_ratio = math.exp(log_sum / len(verdicts)) if verdicts else None
    geo = suite_geomean(candidate_per_case)

    def out(accepted: bool, reason: str) -> Decision:
        return Decision(accepted=accepted, reason=reason, routes=verdicts,
                        improved=improved, regressed=regressed, catastrophic=catastrophic,
                        suite_ratio=suite_ratio, suite_geomean_speedup=geo)

    # The "we would never ship this" fence, and the only refusal that survives a positive suite
    # number. An order of magnitude outside every floor ever measured here, so it is not a noise
    # judgement: it exists so "the average improved" cannot ship one shape a third slower.
    if catastrophic:
        return out(False, "a route regressed catastrophically (past "
                   f"{CATASTROPHIC_REGRESSION:.2%}, an order of magnitude outside any measured "
                   "floor): " + ", ".join(f"{v.route} {-v.delta_frac:.2%}"
                                          for v in verdicts if v.status == "catastrophic"))

    # Condition 1: the suite average did not get worse. Computed over the PAIRED per-route times so
    # both arms come from one invocation, and the 1.5-3% an unchanged tree wanders between
    # invocations cancels instead of deciding the round.
    if not (suite_ratio and suite_ratio > 1):
        return out(False, f"the suite geomean did not improve ({suite_ratio:.5f} vs the incumbent "
                          "arm measured in the same session)")

    # Condition 2: something concrete moved. A suite average that improves while no single route
    # clears its own noise is what a round of pure measurement luck looks like.
    if not improved:
        nearest = max(verdicts, key=lambda v: v.delta_frac)
        return out(False, f"the suite improved ({suite_ratio:.5f}) but no single route cleared its "
                          f"own bar; the best was {nearest.route} {nearest.delta_frac:.2%} against "
                          f"a required {nearest.need:.2%}"
                          + (" (its measured floor, wider than the "
                             f"{MIN_ROUTE_WIN:.2%} default)" if nearest.need > MIN_ROUTE_WIN else ""))

    # Targeting is a NOTE, not a veto. It used to refuse a win that landed off the declared route;
    # `target_routes` came back on 1 of 26 engineer results, so in practice that rule refused work
    # over a missing label far more often than it caught an artefact. The caller logs the mismatch.
    off_target = (sorted(set(improved) - set(target_routes))
                  if target_routes else [])
    return out(True, f"suite {suite_ratio:.5f} and cleared its own bar on: " + ", ".join(
        f"{v.route} {v.delta_frac:.2%} (needed {v.need:.2%}"
        + (", its measured floor" if v.need > MIN_ROUTE_WIN else "") + ")"
        for v in verdicts if v.status == "won")
        + (f"; gave ground within noise on {', '.join(regressed)}" if regressed else "")
        + (f"; NOTE: {', '.join(off_target)} improved but was not a declared target route"
           if off_target else ""))


def receipt(d: Decision) -> dict:
    return {
        "schema": SCHEMA,
        "accepted": d.accepted,
        "reason": d.reason,
        "improved": d.improved,
        "regressed": d.regressed,
        "catastrophic": d.catastrophic,
        # The number the decision was actually made on: both arms from one session.
        "suite_ratio": d.suite_ratio,
        # Reported, never gated on. See suite_geomean().
        "suite_geomean_speedup_reported_only": d.suite_geomean_speedup,
        "routes": [{"route": v.route, "incumbent_ms": v.incumbent_ms,
                    "candidate_ms": v.candidate_ms, "delta_frac": v.delta_frac,
                    "band": v.band, "need": v.need, "status": v.status} for v in d.routes],
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
