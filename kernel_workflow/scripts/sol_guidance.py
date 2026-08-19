#!/usr/bin/env python3
"""Speed-of-Light guidance for the GREEDY lane: steer, budget, and screen.

`sol_card.py` already owns the ceiling arithmetic and its provenance, and it
is deliberately reachable only as post-selection reporting. This module does not
duplicate that. It adds the three things the greedy lane is missing, each of
which is a use of SOL that is *not* candidate ranking:

  1. STEER    -- per-route remaining headroom, so a round spends its directions
                 where the hardware still allows a win instead of where the last
                 round happened to look promising.
  2. BUDGET   -- SOL-gap stop / no-progress eligibility, so a route that has
                 reached its achievable ceiling stops consuming direction slots.
  3. SCREEN   -- a one-sided physical-plausibility filter, so a candidate that
                 reports a time below the bound is caught before it is banked.

Selection stays where it is. `winner` is still chosen by measured paired
speedup, exactly as before, and nothing here is allowed to reorder candidates:
an easy memory-bound route sitting at 95% of its ceiling must never outrank a
hard compute-bound route honestly at 40% of its own. That is the same rule
`sol_card.build_sol_card` enforces with `post_selection=True`; this module
keeps it by never accepting more than one candidate's measurements in any
function that returns a comparable number.

Method follows two published treatments and the deviations are named:

  * arXiv:2603.29010 (SOL guidance, MANTIS) -- gap g = t_best / t_SOL, the
    gap-aware ROI exponent, SOL-gap-stop / no-progress eligibility, and the
    "more than 10% below the bound is suspicious" screen.
  * arXiv:2607.14541 (Atrex-Bench) -- SOL from *reference semantics* rather
    than from the candidate's own profile, per-route achievement
    S = t_SOL / t_cand, route summary by median, and an aggregate that is a
    weighted ARITHMETIC mean. The arithmetic mean is the load-bearing choice:
    a geometric mean over eleven routes divides a single-route win by eleven
    and collapses outright on one zero, which is what made a real +7% route
    mechanism read as +0.6% suite noise on this lane.

DEVIATION, and the reason this module exists rather than a thin wrapper. The
compute ceiling in `sol_card` carries a WITNESS: a peak is attainable only
if something was observed to reach it, and the module refuses a ceiling that
sits below an achieved rate ("the ceiling is not a ceiling"). The bandwidth
ceiling has no such check, and on this suite that gap is not hypothetical --
four of the eleven routes run FASTER than the footprint-resolved bandwidth
ceiling, because that table was measured with an ordinary streaming read before
the lane adopted non-temporal loads. Screening on it unchanged would flag four
legitimate kernels as gaming. So the bandwidth side gets the same witness rule,
and a bound the hardware has already beaten is reported as a defect in the
BOUND (`ceiling_contradicted`), never as a defect in the kernel.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass, field, asdict
from typing import Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sol_card import SOLCardError, bandwidth_ceiling  # noqa: E402

SCHEMA = "geak.sol-guidance/v1"

# arXiv:2603.29010 §4.4. A candidate is suspicious once it is more than this far
# BELOW a witness-backed bound. Kept at the published value rather than retuned:
# the point of an external number here is that it was not chosen to fit this
# lane's results.
GAMING_MARGIN = 0.10

# arXiv:2603.29010 §4.2. ROI(h) = S^(1 + max(0, log10(g / ROI_GAP_PIVOT))) / (R_impl * R_perf).
# At g = 5 the exponent is 1, so ambition is amplified only once a route is
# more than 5x from its ceiling.
ROI_GAP_PIVOT = 5.0

# Fraction of the ridge point inside which a route is called balanced rather
# than compute- or memory-bound. Matches atrex-bench's `_RIDGE_BAND`.
RIDGE_BAND = 0.05

BYTES_PER_ELEM = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8_e4m3": 1, "fp8_e5m2": 1}


# --------------------------------------------------------------------------
# Semantic work and traffic. From the operator definition, never from a profile.
# --------------------------------------------------------------------------

def gemm_semantic_work_flops(m: int, n: int, k: int) -> int:
    """FLOPs a dense M x K by K x N GEMM must perform: one multiply-add per term."""
    _require_positive_shape(m, n, k)
    return 2 * m * n * k


def gemm_semantic_traffic_bytes(m: int, n: int, k: int, dtype: str = "bf16") -> int:
    """Best-case HBM bytes: every unique input element read once, output written once.

    This is a lower bound on traffic and therefore contributes an optimistic
    memory floor, which is what a Speed-of-Light bound is supposed to be. It is
    optimistic in a specific, nameable way: a tiling that re-reads a B panel per
    M-tile row moves a multiple of this, so a kernel with tiles_m > 1 cannot
    reach the memory floor computed here no matter how well it is written.
    """
    _require_positive_shape(m, n, k)
    if dtype not in BYTES_PER_ELEM:
        raise SOLCardError(
            f"unknown dtype {dtype!r}: known dtypes are {sorted(BYTES_PER_ELEM)}")
    w = BYTES_PER_ELEM[dtype]
    return w * (m * k + k * n + m * n)


def _require_positive_shape(m: int, n: int, k: int) -> None:
    for name, value in (("m", m), ("n", n), ("k", k)):
        if not (isinstance(value, int) and value > 0):
            raise SOLCardError(f"{name} must be a positive int, got {value!r}")


# --------------------------------------------------------------------------
# Ceilings, with the bandwidth witness this lane turned out to need
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Ceilings:
    """Resolved peaks for one route, plus what attests to each of them.

    `bandwidth_witness_bytes_s` is the fastest byte rate anything has been
    OBSERVED to reach at a comparable footprint, with `bandwidth_witness_by`
    naming it. Supplying it raises the memory floor's credibility in exactly the
    way the compute side already works; omitting it is honest but leaves the
    screen unable to distinguish a fast kernel from a slow bound.
    """

    peak_flops: float
    bandwidth_bytes_s: float
    bandwidth_confidence: str
    bandwidth_extrapolated: bool
    footprint_bytes: float
    compute_witnessed: bool
    compute_witness_by: str = ""
    bandwidth_witness_bytes_s: float | None = None
    bandwidth_witness_by: str = ""
    source: str = ""

    @property
    def effective_bandwidth_bytes_s(self) -> float:
        """The bound actually used: never below a rate already witnessed.

        A ceiling under an observed rate is not a ceiling. `sol_card` raises
        on exactly this for compute; here it is resolved upward instead of
        raised, because the caller is screening a real measurement and aborting
        would lose the measurement along with the bad bound.
        """
        if self.bandwidth_witness_bytes_s and \
                self.bandwidth_witness_bytes_s > self.bandwidth_bytes_s:
            return float(self.bandwidth_witness_bytes_s)
        return float(self.bandwidth_bytes_s)

    @property
    def bandwidth_is_witnessed(self) -> bool:
        return bool(self.bandwidth_witness_bytes_s)


def resolve_ceilings(
    *,
    footprint_bytes: float,
    peak_flops: float,
    bandwidth_table: Mapping[object, object] | None = None,
    bandwidth_scalar: float | None = None,
    compute_witness_by: str = "",
    bandwidth_witness_bytes_s: float | None = None,
    bandwidth_witness_by: str = "",
    source: str = "",
) -> Ceilings:
    """Resolve one route's ceilings. Exactly one bandwidth shape may be given."""
    if not (isinstance(peak_flops, (int, float)) and peak_flops > 0):
        raise SOLCardError(f"peak_flops must be positive, got {peak_flops!r}")
    if (bandwidth_table is None) == (bandwidth_scalar is None):
        raise SOLCardError(
            "supply exactly one of bandwidth_table or bandwidth_scalar: a scalar "
            "ceiling and a footprint-indexed one cannot both be the ceiling")
    if bandwidth_witness_bytes_s is not None:
        if not (isinstance(bandwidth_witness_bytes_s, (int, float))
                and bandwidth_witness_bytes_s > 0):
            raise SOLCardError(
                f"bandwidth_witness_bytes_s must be positive, got {bandwidth_witness_bytes_s!r}")
        if not str(bandwidth_witness_by).strip():
            raise SOLCardError(
                "bandwidth_witness_bytes_s requires bandwidth_witness_by naming what "
                "achieved the rate; an unattributed witness is the unevidenced claim "
                "the witness exists to replace")

    if bandwidth_table is not None:
        resolved = bandwidth_ceiling(float(footprint_bytes), bandwidth_table)
        bw = float(resolved["bytes_s"])
        confidence = str(resolved["confidence"])
        extrapolated = bool(resolved["extrapolated"])
    else:
        bw = float(bandwidth_scalar)  # type: ignore[arg-type]
        confidence = "scalar"
        extrapolated = False

    return Ceilings(
        peak_flops=float(peak_flops), bandwidth_bytes_s=bw,
        bandwidth_confidence=confidence, bandwidth_extrapolated=extrapolated,
        footprint_bytes=float(footprint_bytes),
        compute_witnessed=bool(str(compute_witness_by).strip()),
        compute_witness_by=str(compute_witness_by),
        bandwidth_witness_bytes_s=(float(bandwidth_witness_bytes_s)
                                   if bandwidth_witness_bytes_s is not None else None),
        bandwidth_witness_by=str(bandwidth_witness_by), source=str(source),
    )


# --------------------------------------------------------------------------
# SOL analysis for one route
# --------------------------------------------------------------------------

@dataclass
class RouteSol:
    """One route's Speed-of-Light bound and where the current kernel sits on it."""

    route: str
    m: int
    n: int
    k: int
    dtype: str
    work_flops: int
    traffic_bytes: int
    arithmetic_intensity: float
    ridge_point: float
    regime: str
    compute_floor_s: float
    memory_floor_s: float
    sol_s: float
    ceilings: Ceilings
    measured_s: float | None = None
    gap: float | None = None
    remaining_headroom: float | None = None
    achievement: float | None = None
    verdict: str = "no_measurement"
    notes: list[str] = field(default_factory=list)

    @property
    def sol_us(self) -> float:
        return self.sol_s * 1e6


def analyze_route(
    *,
    route: str,
    m: int,
    n: int,
    k: int,
    ceilings: Ceilings,
    dtype: str = "bf16",
    measured_s: float | None = None,
) -> RouteSol:
    """SOL bound for one route, and the current kernel's position on it.

    `measured_s` is optional: the bound is a property of the problem and the
    hardware, so it is computable before any kernel exists. That ordering is the
    point -- a direction can be triaged against remaining headroom before a
    single GPU second is spent on it.
    """
    work = gemm_semantic_work_flops(m, n, k)
    traffic = gemm_semantic_traffic_bytes(m, n, k, dtype)
    bw = ceilings.effective_bandwidth_bytes_s
    ai = work / traffic
    ridge = ceilings.peak_flops / bw
    t_compute = work / ceilings.peak_flops
    t_mem = traffic / bw
    sol_s = max(t_compute, t_mem)

    lo, hi = ridge * (1.0 - RIDGE_BAND), ridge * (1.0 + RIDGE_BAND)
    regime = "memory_bound" if ai < lo else ("compute_bound" if ai > hi else "balanced")

    out = RouteSol(
        route=route, m=m, n=n, k=k, dtype=dtype, work_flops=work,
        traffic_bytes=traffic, arithmetic_intensity=ai, ridge_point=ridge,
        regime=regime, compute_floor_s=t_compute, memory_floor_s=t_mem,
        sol_s=sol_s, ceilings=ceilings,
    )
    if ceilings.bandwidth_extrapolated:
        out.notes.append(
            "bandwidth ceiling is extrapolated past the measured footprint table; "
            "treat this bound as a prompt to measure the footprint, not a target")
    if not ceilings.compute_witnessed and regime != "memory_bound":
        out.notes.append(
            "compute ceiling has no witness, and this route is not memory-bound: "
            "the bound may be a nameplate nothing reaches, which ranks nothing")
    if measured_s is not None:
        out = _place_measurement(out, measured_s)
    return out


def _place_measurement(route: RouteSol, measured_s: float) -> RouteSol:
    if not (isinstance(measured_s, (int, float)) and measured_s > 0):
        raise SOLCardError(f"measured_s must be positive, got {measured_s!r}")
    route.measured_s = float(measured_s)
    route.gap = measured_s / route.sol_s
    route.remaining_headroom = 1.0 - route.sol_s / measured_s
    route.achievement = route.sol_s / measured_s
    route.verdict = _verdict(route)
    return route


def _verdict(route: RouteSol) -> str:
    """One of: ok, near_sol, ceiling_contradicted, gaming_suspected.

    The split between the last two is the whole point. Both are "the kernel
    reports a time below the bound", and they call for opposite actions: one
    means re-measure the ceiling, the other means audit the kernel. Collapsing
    them -- which a bare `t < 0.9 * t_SOL` check does -- would have condemned
    four legitimate non-temporal-load routes on this suite.
    """
    assert route.gap is not None
    if route.gap >= 1.0:
        return "ok"
    if route.gap > 1.0 - GAMING_MARGIN:
        # Inside the margin: a bound is an idealisation and landing just under it
        # is a modelling residue, not a finding either way.
        return "near_sol"
    binding_is_witnessed = (
        route.ceilings.compute_witnessed if route.regime == "compute_bound"
        else route.ceilings.bandwidth_is_witnessed)
    return "gaming_suspected" if binding_is_witnessed else "ceiling_contradicted"


# --------------------------------------------------------------------------
# Aggregation (arXiv:2607.14541 §3.6), with the geometric mean deliberately out
# --------------------------------------------------------------------------

def route_achievement(routes: Sequence[RouteSol]) -> float:
    """Median of per-route achievement S = t_SOL / t_cand over measured routes."""
    vals = [r.achievement for r in routes if r.achievement is not None]
    if not vals:
        return 0.0
    return statistics.median(vals)


def weighted_achievement(routes: Sequence[RouteSol],
                         weights: Mapping[str, float] | None = None) -> float:
    """Importance-weighted ARITHMETIC mean of per-route achievement.

    Arithmetic, not geometric, and not by preference: a geometric mean over
    eleven routes attenuates a single-route win by roughly eleven, and goes to
    zero if any route is unmeasured. Both failures are live on this lane -- the
    first is why route mechanisms worth +7% were read as suite noise.

    Routes with no measurement score 0 rather than being dropped, so a route the
    search broke cannot improve the aggregate by disappearing from it.
    """
    if not routes:
        return 0.0
    if weights is None:
        weights = {r.route: 1.0 for r in routes}
    total_w = sum(float(weights.get(r.route, 0.0)) for r in routes)
    if total_w <= 0:
        raise SOLCardError("weights must sum to a positive number over the given routes")
    acc = 0.0
    for r in routes:
        w = float(weights.get(r.route, 0.0))
        acc += w * (r.achievement if r.achievement is not None else 0.0)
    return acc / total_w


# --------------------------------------------------------------------------
# Steering: hypothesis triage (arXiv:2603.29010 §4.2)
# --------------------------------------------------------------------------

def gap_exponent(gap: float) -> float:
    """1 + max(0, log10(gap / 5)): amplify ambition only when far from the ceiling."""
    if not (isinstance(gap, (int, float)) and gap > 0):
        raise SOLCardError(f"gap must be positive, got {gap!r}")
    return 1.0 + max(0.0, math.log10(gap / ROI_GAP_PIVOT))


def hypothesis_roi(*, estimated_speedup: float, gap: float,
                   impl_risk: float, perf_risk: float) -> float:
    """ROI = S^(1 + max(0, log10(g/5))) / (R_impl * R_perf).

    Risks are multipliers >= 1, where 1 means "no risk". They divide rather than
    subtract so that a doubling of either halves the ROI regardless of scale.
    """
    if not (isinstance(estimated_speedup, (int, float)) and estimated_speedup > 0):
        raise SOLCardError(f"estimated_speedup must be positive, got {estimated_speedup!r}")
    for name, risk in (("impl_risk", impl_risk), ("perf_risk", perf_risk)):
        if not (isinstance(risk, (int, float)) and risk >= 1.0):
            raise SOLCardError(f"{name} must be a multiplier >= 1, got {risk!r}")
    return (estimated_speedup ** gap_exponent(gap)) / (impl_risk * perf_risk)


def triage(hypotheses: Sequence[Mapping[str, object]], *, gap: float) -> list[dict]:
    """Rank hypotheses by gap-aware ROI, highest first. Ties keep input order."""
    scored = []
    for idx, h in enumerate(hypotheses):
        roi = hypothesis_roi(
            estimated_speedup=float(h["estimated_speedup"]), gap=gap,
            impl_risk=float(h.get("impl_risk", 1.0)),
            perf_risk=float(h.get("perf_risk", 1.0)))
        scored.append((-roi, idx, {**dict(h), "roi": roi, "gap_exponent": gap_exponent(gap)}))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [item for _, _, item in scored]


# --------------------------------------------------------------------------
# Budgeting: eligibility (arXiv:2603.29010 §4.3)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    reason: str


def route_eligibility(*, gap: float, ahead_of_reference: bool,
                      rounds_without_progress: int,
                      sol_gap_epsilon: float = 1.0,
                      no_progress_window: int = 2) -> Eligibility:
    """Should this route still receive direction slots?

    Two rules, both from §4.3, and both conditioned on already being ahead of
    the reference implementation. A route that is still LOSING to the oracle is
    never made ineligible: being near its own ceiling while slower than the
    thing it has to replace means the bound is wrong or the route needs a
    different algorithm, and either way it is not finished.
    """
    if not (isinstance(sol_gap_epsilon, (int, float)) and sol_gap_epsilon >= 0):
        raise SOLCardError(f"sol_gap_epsilon must be >= 0, got {sol_gap_epsilon!r}")
    if not (isinstance(no_progress_window, int) and no_progress_window >= 1):
        raise SOLCardError(f"no_progress_window must be a positive int, got {no_progress_window!r}")
    if not ahead_of_reference:
        return Eligibility(True, "still behind the reference implementation")
    if gap <= 1.0 + sol_gap_epsilon:
        return Eligibility(
            False,
            f"within {sol_gap_epsilon:.0%} of its achievable ceiling (gap {gap:.2f}x)")
    if rounds_without_progress >= no_progress_window:
        return Eligibility(
            False,
            f"no progress for {rounds_without_progress} rounds while ahead of the reference")
    return Eligibility(True, f"gap {gap:.2f}x still open")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _ceilings_dict(c: Ceilings) -> dict:
    d = asdict(c)
    d["effective_bandwidth_bytes_s"] = c.effective_bandwidth_bytes_s
    d["bandwidth_is_witnessed"] = c.bandwidth_is_witnessed
    return d


def route_dict(r: RouteSol) -> dict:
    d = {k: v for k, v in asdict(r).items() if k != "ceilings"}
    d["ceilings"] = _ceilings_dict(r.ceilings)
    d["sol_us"] = r.sol_us
    return d


def report(routes: Sequence[RouteSol],
           weights: Mapping[str, float] | None = None) -> dict:
    contradicted = [r.route for r in routes if r.verdict == "ceiling_contradicted"]
    gaming = [r.route for r in routes if r.verdict == "gaming_suspected"]
    return {
        "schema": SCHEMA,
        "routes": [route_dict(r) for r in routes],
        "median_achievement": route_achievement(routes),
        "weighted_achievement": weighted_achievement(routes, weights),
        "ceiling_contradicted": contradicted,
        "gaming_suspected": gaming,
        "screen_passed": not gaming,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="JSON: {peak_flops, bandwidth_table|bandwidth_scalar, "
                                 "compute_witness_by, bandwidth_witness_bytes_s, "
                                 "bandwidth_witness_by, dtype, routes:[{route,m,n,k,measured_s}]}")
    ap.add_argument("--json", action="store_true", help="emit the machine-readable receipt")
    args = ap.parse_args(argv)

    with open(args.spec) as fh:
        spec = json.load(fh)

    table = spec.get("bandwidth_table")
    if isinstance(table, Mapping):
        table = {int(k): float(v) for k, v in table.items()}
    dtype = spec.get("dtype", "bf16")

    routes: list[RouteSol] = []
    for entry in spec["routes"]:
        m, n, k = int(entry["m"]), int(entry["n"]), int(entry["k"])
        footprint = gemm_semantic_traffic_bytes(m, n, k, dtype)
        ceilings = resolve_ceilings(
            footprint_bytes=footprint, peak_flops=float(spec["peak_flops"]),
            bandwidth_table=table, bandwidth_scalar=spec.get("bandwidth_scalar"),
            compute_witness_by=spec.get("compute_witness_by", ""),
            bandwidth_witness_bytes_s=spec.get("bandwidth_witness_bytes_s"),
            bandwidth_witness_by=spec.get("bandwidth_witness_by", ""),
            source=spec.get("source", ""))
        routes.append(analyze_route(
            route=str(entry["route"]), m=m, n=n, k=k, ceilings=ceilings, dtype=dtype,
            measured_s=(float(entry["measured_s"]) if entry.get("measured_s") else None)))

    rep = report(routes, spec.get("weights"))
    if args.json:
        print(json.dumps(rep, indent=1, sort_keys=True))
    else:
        print("%-22s %9s %9s %7s %7s  %-14s %s"
              % ("route", "meas_us", "sol_us", "gap", "S", "regime", "verdict"))
        for r in routes:
            print("%-22s %9s %9.1f %7s %7s  %-14s %s"
                  % (r.route,
                     ("%.1f" % (r.measured_s * 1e6)) if r.measured_s else "-",
                     r.sol_us,
                     ("%.2fx" % r.gap) if r.gap else "-",
                     ("%.3f" % r.achievement) if r.achievement else "-",
                     r.regime, r.verdict))
        print("\nmedian achievement   %.4f" % rep["median_achievement"])
        print("weighted achievement %.4f" % rep["weighted_achievement"])
        if rep["ceiling_contradicted"]:
            print("\nCEILING CONTRADICTED (re-measure the bound, do not audit the kernel):")
            for name in rep["ceiling_contradicted"]:
                print("  - %s" % name)
        if rep["gaming_suspected"]:
            print("\nGAMING SUSPECTED (witness-backed bound beaten by >%.0f%%):" % (GAMING_MARGIN * 100))
            for name in rep["gaming_suspected"]:
                print("  - %s" % name)
    return 0 if rep["screen_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
