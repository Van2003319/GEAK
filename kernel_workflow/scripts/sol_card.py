#!/usr/bin/env python3
"""Deterministic gfx90a "speed of light" (SOL) roofline card, strictly post-selection.

`sol_guidance.py` steers a round toward the routes where the hardware still
allows a win, and it needs a ceiling it can defend. This module is that
ceiling: the deterministic roofline arithmetic, with the provenance of every
peak it divides by, so a headroom number can be traced to a measurement rather
than to a nameplate.

SOL is diagnostic, not a selection signal: the roofline ceiling for a kernel's
own measured arithmetic intensity does not change whether that kernel beats
another candidate; it only explains HOW FAR a candidate (already chosen by the
harness's actual speedup/correctness gates) sits from its own hardware ceiling.
Feeding SOL into selection would let an easy, memory-bound case's near-100%
SOL rating quietly outrank a hard, compute-bound case's honestly-lower ratio --
this module keeps SOL contained to reporting by making the entry point refuse
to run at all unless the caller explicitly declares selection is already over:

    build_sol_card(..., post_selection=True)

`post_selection=False` (or omitted) raises immediately. There is deliberately
no variant of this function that accepts more than one candidate's
measurements, so it has no way to compare or rank candidates even by accident.
"""
from __future__ import annotations

from typing import Mapping, Sequence

SCHEMA = "geak.sol-card/v3"
# v3 adds the compute-ceiling witness ((89) item 2). The version string is
# bumped rather than the fields quietly appended because a consumer written
# against v2 reads a v3 card without noticing that its compute denominator now
# carries an attainability claim -- and per (92) a version string is the only
# thing that makes a stale consumer fail loudly instead of silently agreeing.
SCHEMA_V2 = "geak.sol-card/v2"
# v1 cards are still readable. They were written before the bandwidth ceiling
# became footprint-indexed, and rejecting them would make an old card look
# corrupt rather than merely old. What a v1 card may NOT do is claim a
# footprint-resolved ceiling, so `validate_sol_card` requires the v2 fields on
# v2 cards and forbids them on v1.
SCHEMA_V1 = "geak.sol-card/v1"
ACCEPTED_SCHEMAS = (SCHEMA, SCHEMA_V2, SCHEMA_V1)
SUPPORTED_ARCH = "gfx90a"

# Reference MI250X single-GCD figures. The in-repo CDNA2 card records 362.1
# TFLOP/s BF16/FP16/INT8 and 90.5 TFLOP/s matrix FP32/FP64 per OAM (both GCDs),
# hence the explicit half-OAM values here. INT8 is kept at the same documented
# 362.1 TOPS rather than assuming a second undocumented doubling. These are
# still physical reference peaks, not calibrated effective ceilings; callers
# should pass measured calibration from the actual allocated GCD/device.
REFERENCE_GFX90A_CARD: Mapping[str, object] = {
    "arch": SUPPORTED_ARCH,
    "peak_flops": {
        "bf16": 181.05e12, "fp16": 181.05e12, "fp32": 45.25e12,
        "fp64": 45.25e12, "int8": 181.05e12,
    },
    "peak_bandwidth_bytes_s": 1.6e12,
    "source": "reference MI250X single-GCD physical peaks derived from the in-repo "
              "CDNA2 per-OAM card; not an effective measurement -- pass `calibration` "
              "from the actually-profiled allocated device",
    # Finding (89) item 2. No `attainment` key, deliberately and visibly: nothing
    # has ever been observed to reach these numbers on this part, because they
    # are datasheet peaks. The absence is the honest state and it is what makes
    # a compute-bound gfx90a card come back `compute_ceiling_witnessed: False`.
}
SUPPORTED_DTYPES: tuple[str, ...] = tuple(REFERENCE_GFX90A_CARD["peak_flops"])  # type: ignore[arg-type]


# Measured on gfx942/MI300X, machine H, under the kernel harness's own timing
# regime (cold, flush_cache=True, warmup 10, repeats 50, median device ms).
# This was recorded as inert data for five stages because it is the first time
# anyone measured what the `calibration` argument is *for*, and because what it
# showed is that the card's model had a hole. Both are now closed: the hole by
# `bandwidth_ceiling` below, and the inertness by `ARCH_CARDS`, which makes this
# the default ceiling for `arch="gfx942"` with no calibration required.
#
# The paper figures for this part are 1307 TFLOP/s BF16 and 5.3 TB/s. Neither
# is reachable:
#
#   * BF16 matmul tops out at 668 TFLOP/s (8192 cubed), 51% of paper.
#   * Cold streaming read is strongly FOOTPRINT-DEPENDENT --
#     32 MB: 1.42 TB/s | 64 MB: 2.11 | 86 MB: 2.30 | 128 MB: 2.68
#     | 256 MB: 2.85-3.00 | 1024 MB: 3.94 -- i.e. 27% to 74% of paper,
#     reproduced twice within 2%.
#
# THE HOLE (closed in v2): `peak_bandwidth_bytes_s` was a scalar, so no
# calibration of this module could express a ceiling that moves by 2.8x across
# the footprints of a single benchmark suite. Feeding it the 1024 MB number made
# every small decode shape look like it had 2.8x of headroom it does not have;
# feeding it the 32 MB number made the large shapes look finished. Both were
# tried on the eleven-shape BF16 suite and both mis-rank it. Hence this card
# carries no scalar at all -- a gfx942 request must supply `footprint_bytes`,
# and the module refuses rather than picking one of the two wrong answers.
#
# Why it matters beyond this module: scored against the paper roofline every
# route in that suite sits 3.2x-6.4x from SOL and so does rocBLAS, at 3.5x-7.4x
# -- a ceiling nothing approaches ranks nothing. Rescored against the measured
# footprint-matched ceiling, two routes turn out to be finished (1.02x and
# 1.10x of achievable) that the paper roofline said had 3.8x and 4.1x left.
MEASURED_GFX942_CEILINGS: Mapping[str, object] = {
    "arch": "gfx942",
    "measured": True,
    "peak_flops": {"bf16": 668e12},
    # Finding (89) item 2: the ceiling's WITNESS -- what actually reached it.
    # `measured=True` above attests provenance ("these numbers came from this
    # box"), and provenance is not attainability: `rocminfo` reports 1307
    # TFLOP/s on this same box with impeccable provenance, and nothing has ever
    # got near it. What separates the two is whether anything achieved the rate.
    # 668e12 has an achiever, named here; 1307e12 does not and cannot acquire
    # one, which is why this check refuses a nameplate without inventing a
    # threshold to refuse it with.
    "attainment": {"bf16": {"achieved_flops": 668e12,
                            "by": "exp/opt_bf16_20260814/shapeceil.py, 8192^3 bf16 matmul, "
                                  "machine H, harness timing regime; reproduced twice within 2%"}},
    "peak_bandwidth_bytes_s_by_footprint": {
        32 << 20: 1.42e12, 64 << 20: 2.11e12, 86 << 20: 2.30e12,
        128 << 20: 2.68e12, 256 << 20: 2.92e12, 1024 << 20: 3.94e12,
    },
    "source": "exp/opt_bf16_20260814/bwceil.py and shapeceil.py, machine H, "
              "harness regime (cold, flush_cache, warmup 10, repeats 50, "
              "median device ms); no candidate linked",
}

# gfx942 is a first-class arch as of the v2 schema. Until then this table was
# DATA that `_resolve_peaks` refused to look at, reachable only by a caller
# passing it back in as `calibration` -- which meant the arch every measurement
# in this project was taken on was the one arch the module would not model.
#
# Note what the two cards differ in besides numbers. The gfx90a entry is a
# *reference physical peak* and is unmeasured; the gfx942 entry is measured on
# the box, and only for bf16, because bf16 is the only dtype anyone ran. The
# missing dtypes are absent rather than filled in from the spec sheet: a
# fabricated ceiling ranks worse than no ceiling, since it looks like evidence.
ARCH_CARDS: Mapping[str, Mapping[str, object]] = {
    "gfx90a": REFERENCE_GFX90A_CARD,
    "gfx942": MEASURED_GFX942_CEILINGS,
}
SUPPORTED_ARCHES: tuple[str, ...] = tuple(ARCH_CARDS)


class SOLCardError(ValueError):
    """A SOL card was requested or built in a way that violates its post-selection contract."""


def bandwidth_ceiling(footprint_bytes: float, table: Mapping[object, object]
                       ) -> dict[str, object]:
    """Resolve a footprint-indexed bandwidth ceiling. Closes the v1 hole above.

    Returns `{bytes_s, confidence, bracket, extrapolated}`.

    Interpolation is linear in **log(footprint), log(bandwidth)**, not linear in
    bytes. The measured curve is a cache-hierarchy curve -- 1.42 TB/s at 32 MB
    to 3.94 at 1024 MB, a 2.8x rise over a 32x footprint range -- and a straight
    line in bytes across it *contradicts the measurements it spans*: the chord
    from 32 MB to 1024 MB passes below the measured 256 MB point by a third.
    Log-log reproduces every measured point exactly and
    stays between its neighbours in between, which is the most a six-point table
    can honestly support. This is interpolation, not a model of the hardware.

    Outside the measured range the value is **clamped** to the nearest endpoint
    and flagged `extrapolated`, with `confidence: "low"`. Read the low end with
    particular care: below 32 MB the working set is latency- and launch-bound
    rather than bandwidth-bound, so clamping to 1.42 TB/s **overstates** the
    achievable rate, and a tiny decode shape scored against it will look like it
    has headroom it does not have. That is the same error the paper roofline
    made, one order of magnitude smaller. A low-confidence card is a prompt to
    measure the footprint, not a target to optimise toward.
    """
    points = sorted((float(k), float(v)) for k, v in table.items())
    if not points:
        raise SOLCardError("peak_bandwidth_bytes_s_by_footprint is empty")
    if not (isinstance(footprint_bytes, (int, float)) and footprint_bytes > 0):
        raise SOLCardError(f"footprint_bytes must be a positive number, got {footprint_bytes!r}")
    for size, rate in points:
        if not (size > 0 and rate > 0):
            raise SOLCardError(
                f"peak_bandwidth_bytes_s_by_footprint needs positive size/rate, got {size}: {rate}")
    x = float(footprint_bytes)
    if x <= points[0][0]:
        return {"bytes_s": points[0][1], "confidence": "low",
                "bracket": (points[0][0], points[0][0]),
                "extrapolated": x < points[0][0]}
    if x >= points[-1][0]:
        return {"bytes_s": points[-1][1], "confidence": "low",
                "bracket": (points[-1][0], points[-1][0]),
                "extrapolated": x > points[-1][0]}
    import math
    for (lo_x, lo_y), (hi_x, hi_y) in zip(points, points[1:]):
        if lo_x <= x <= hi_x:
            # Land a measured footprint on its measured rate bit-for-bit.
            # exp(log(y)) is off by an ulp or two, which is harmless as a
            # ceiling but makes a card disagree with the table it cites -- and
            # that discrepancy is exactly what someone chasing a stale number
            # would spend an hour on.
            if hi_x == lo_x or x == lo_x:
                rate = lo_y
            elif x == hi_x:
                rate = hi_y
            else:
                t = (math.log(x) - math.log(lo_x)) / (math.log(hi_x) - math.log(lo_x))
                rate = math.exp(math.log(lo_y) + t * (math.log(hi_y) - math.log(lo_y)))
            return {"bytes_s": rate, "confidence": "measured_interpolated",
                    "bracket": (lo_x, hi_x), "extrapolated": False}
    raise SOLCardError(f"footprint {x} fell through the bracket search")  # unreachable


def _resolve_peaks(arch: str, dtype: str, calibration: Mapping[str, object] | None,
                    footprint_bytes: float | None = None
                    ) -> tuple[float, float, str, dict[str, object]]:
    if arch not in ARCH_CARDS:
        raise SOLCardError(f"unsupported arch {arch!r}: modeled arches are {SUPPORTED_ARCHES}")
    card = ARCH_CARDS[arch]
    peak_flops_table = dict(card["peak_flops"])  # type: ignore[arg-type]
    attainment_table = dict(card.get("attainment") or {})  # (89) item 2
    if dtype not in peak_flops_table and not (
            calibration and isinstance(calibration.get("peak_flops"), Mapping)
            and dtype in calibration["peak_flops"]):  # type: ignore[index]
        # Named per arch, not against a global dtype list: gfx942's card holds
        # bf16 only because bf16 is all that was measured, and saying "fp32 is
        # not supported" would be wrong -- it is unmeasured *here*, and a
        # calibration may still supply it.
        raise SOLCardError(
            f"dtype {dtype!r} has no measured peak on {arch}: modeled dtypes are "
            f"{tuple(peak_flops_table)}; pass calibration.peak_flops[{dtype!r}] to supply one")
    peak_bandwidth = card.get("peak_bandwidth_bytes_s")
    source = card["source"]
    default_table = card.get("peak_bandwidth_bytes_s_by_footprint")
    # The ceiling comes from exactly one of two shapes -- a scalar or a
    # footprint table -- and the arch card and the calibration are both allowed
    # to supply either. Resolve which pair is in force first, then interpret it
    # once, so a calibrated scalar cannot end up silently layered over an arch
    # table (or the reverse).
    table = default_table if isinstance(default_table, Mapping) and default_table else None
    scalar = peak_bandwidth
    confidence_if_scalar = "measured_scalar" if card.get("measured") else "unmeasured"
    if calibration:
        cal_flops = calibration.get("peak_flops")
        if isinstance(cal_flops, Mapping) and dtype in cal_flops:
            peak_flops_table[dtype] = cal_flops[dtype]
        cal_bw = calibration.get("peak_bandwidth_bytes_s")
        cal_table = calibration.get("peak_bandwidth_bytes_s_by_footprint")
        # A footprint table and a scalar are mutually exclusive claims about the
        # same quantity. Taking one silently would leave the other as dead
        # config that still reads as authoritative in the calibration JSON,
        # which is exactly how a stale number survives (25e).
        if cal_bw is not None and isinstance(cal_table, Mapping) and cal_table:
            raise SOLCardError(
                "calibration supplies both peak_bandwidth_bytes_s and "
                "peak_bandwidth_bytes_s_by_footprint; drop one -- a scalar ceiling and a "
                "footprint-indexed one cannot both be the ceiling")
        # A calibration replaces the arch card's ceiling rather than merging
        # with it: it attests to the actual profiled box, and half of one box's
        # numbers combined with half of a reference card's is neither.
        if isinstance(cal_table, Mapping) and cal_table:
            table, scalar = cal_table, None
        elif cal_bw is not None:
            table, scalar = None, cal_bw
            confidence_if_scalar = "measured_scalar"
        if not calibration.get("measured"):
            raise SOLCardError(
                "calibration must set measured=True to attest these numbers came from the "
                "actual profiled box, not another unverified guess")
        source = str(calibration.get("source") or "caller-supplied calibration (measured=True)")
        # A calibration replaces the arch card's peaks, so it must also replace
        # the arch card's witness. Carrying the card's witness forward beside a
        # caller's peak would be the worst possible combination: an attainment
        # ratio computed from two different ceilings.
        if isinstance(cal_flops, Mapping):
            attainment_table = dict(calibration.get("attainment") or {})

    if table is not None:
        if footprint_bytes is None:
            raise SOLCardError(
                f"the bandwidth ceiling for {arch} is footprint-indexed, so build_sol_card needs "
                "footprint_bytes (the bytes the kernel actually moves through HBM for this shape); "
                "the whole point of the table is that no single number serves every shape")
        resolved = bandwidth_ceiling(float(footprint_bytes), table)
        peak_bandwidth = resolved["bytes_s"]
        resolution: dict[str, object] = {
            "bandwidth_ceiling_basis": "footprint_table",
            "bandwidth_ceiling_confidence": resolved["confidence"],
            "footprint_bytes": float(footprint_bytes),
            "bandwidth_ceiling_bracket": list(resolved["bracket"]),
            "bandwidth_ceiling_extrapolated": bool(resolved["extrapolated"])}
    else:
        # A scalar ceiling is the v1 behaviour; it is reported as "scalar" so a
        # card never silently claims footprint resolution it did not do.
        peak_bandwidth = scalar
        resolution = {"bandwidth_ceiling_basis": "scalar",
                      "bandwidth_ceiling_confidence": confidence_if_scalar,
                      "footprint_bytes": None,
                      "bandwidth_ceiling_bracket": None,
                      "bandwidth_ceiling_extrapolated": False}
    peak_flops = peak_flops_table[dtype]
    if peak_flops is None:
        raise SOLCardError(f"dtype {dtype!r} has no matrix-core peak on {arch} (no calibration override given)")
    if not (isinstance(peak_flops, (int, float)) and peak_flops > 0):
        raise SOLCardError(f"peak_flops for {dtype!r} must be a positive number, got {peak_flops!r}")
    if not (isinstance(peak_bandwidth, (int, float)) and peak_bandwidth > 0):
        raise SOLCardError(f"peak_bandwidth_bytes_s must be a positive number, got {peak_bandwidth!r}")

    # ---- (89) item 2: attainability, which provenance does not imply ----------
    #
    # Everything above this point tests where a ceiling CAME FROM. That is a
    # real check and it catches fabrication, but it cannot catch the failure
    # (89) is about: a vendor paper peak has flawless provenance -- it is on the
    # datasheet, `rocminfo` will read it off the card for you -- and it is still
    # unreachable by the vendor's own library, so it ranks nothing. Scored
    # against 1307 TFLOP/s every route in the suite sits 3.2x-6.4x from SOL and
    # so does rocBLAS at 3.5x-7.4x; a ceiling that says "everything has 4x left,
    # including the thing you are trying to beat" is not a weak ranking signal,
    # it is the absence of one.
    #
    # The line drawn here is not a threshold, because any threshold I picked
    # would be exactly the sort of unevidenced number this module exists to
    # refuse. It is a WITNESS: a ceiling is attainable if something has been
    # observed to reach it, and the observation is named. 668 TFLOP/s has an
    # achiever; 1307 does not and cannot acquire one. No arithmetic required.
    witness = attainment_table.get(dtype) if isinstance(attainment_table, Mapping) else None
    achieved = None
    if isinstance(witness, Mapping):
        achieved = witness.get("achieved_flops")
        if not (isinstance(achieved, (int, float)) and achieved > 0):
            raise SOLCardError(
                f"attainment[{dtype!r}].achieved_flops must be a positive number, got {achieved!r}")
        if not str(witness.get("by") or "").strip():
            raise SOLCardError(
                f"attainment[{dtype!r}] must name what achieved the rate in `by`; an unattributed "
                "witness is the same unevidenced claim the witness exists to replace")
        if float(achieved) > float(peak_flops) * (1 + 1e-9):
            # Not a rounding quibble: the ceiling is BELOW something already
            # achieved, so every card built from it reports sol_gap < 1 and the
            # lane refuses each case one at a time for a reason that points at
            # the kernel. The defect is the ceiling and it is caught here.
            raise SOLCardError(
                f"attainment[{dtype!r}].achieved_flops {achieved:.6g} exceeds the peak "
                f"{peak_flops:.6g} it witnesses: the ceiling is not a ceiling")
    resolution.update({
        "compute_ceiling_witnessed": achieved is not None,
        "compute_ceiling_attainment": (float(achieved) / float(peak_flops)) if achieved else None,
        "compute_ceiling_witness": (str(witness.get("by")) if isinstance(witness, Mapping) else ""),
    })
    return float(peak_flops), float(peak_bandwidth), source, resolution


def build_sol_card(*, post_selection: bool, achieved_flops: float, achieved_bytes: float,
                    elapsed_s: float, dtype: str, arch: str = SUPPORTED_ARCH,
                    calibration: Mapping[str, object] | None = None,
                    footprint_bytes: float | None = None) -> dict[str, object]:
    """The roofline card for ONE already-selected candidate's ONE measurement.

    Raises SOLCardError if `post_selection` is not True, or if any input is
    non-positive, or if `arch`/`dtype` is unsupported. There is intentionally
    no batch/list form: this signature cannot be handed a set of candidates to
    rank, only a single already-decided result to explain.
    """
    if post_selection is not True:
        raise SOLCardError(
            "build_sol_card() is post-selection reporting only; call it with "
            "post_selection=True after a candidate has already been chosen on its own "
            "correctness/speedup gates, never to help choose between candidates")
    for name, value in (("achieved_flops", achieved_flops), ("achieved_bytes", achieved_bytes),
                        ("elapsed_s", elapsed_s)):
        if not (isinstance(value, (int, float)) and value >= 0):
            raise SOLCardError(f"{name} must be a non-negative number, got {value!r}")
    if elapsed_s <= 0:
        raise SOLCardError("elapsed_s must be > 0")

    if footprint_bytes is None and achieved_bytes > 0:
        # For a single-pass GEMM the bytes the kernel moves ARE its footprint,
        # so this default is right for the suite this module was built against.
        # It is wrong for anything that re-reads an operand from HBM, which is
        # why the parameter exists and is not simply derived.
        footprint_bytes = achieved_bytes
    peak_flops, peak_bandwidth, source, resolution = _resolve_peaks(
        arch, dtype, calibration, footprint_bytes)
    achieved_flop_rate = achieved_flops / elapsed_s
    achieved_byte_rate = achieved_bytes / elapsed_s
    arithmetic_intensity = (achieved_flops / achieved_bytes) if achieved_bytes > 0 else float("inf")
    ridge_point = peak_flops / peak_bandwidth
    roofline_ceiling = min(peak_flops, arithmetic_intensity * peak_bandwidth) \
        if arithmetic_intensity != float("inf") else peak_flops
    regime = "compute_bound" if arithmetic_intensity >= ridge_point else "memory_bound"
    pct_of_peak = achieved_flop_rate / peak_flops
    pct_of_roofline = achieved_flop_rate / roofline_ceiling if roofline_ceiling > 0 else 0.0
    compute_floor_s = achieved_flops / peak_flops
    memory_floor_s = achieved_bytes / peak_bandwidth
    sol_s = max(compute_floor_s, memory_floor_s)
    if sol_s <= 0:
        raise SOLCardError("at least one of achieved_flops or achieved_bytes must be > 0")
    sol_gap = elapsed_s / sol_s
    remaining_headroom = 1.0 - sol_s / elapsed_s

    return {
        "schema": SCHEMA, "post_selection": True, "arch": arch, "dtype": dtype, "source": source,
        "peak_flops": peak_flops, "peak_bandwidth_bytes_s": peak_bandwidth, "ridge_point": ridge_point,
        "elapsed_s": elapsed_s, "achieved_flop_rate": achieved_flop_rate,
        "achieved_byte_rate": achieved_byte_rate,
        "arithmetic_intensity": arithmetic_intensity, "roofline_ceiling_flops": roofline_ceiling,
        "regime": regime, "pct_of_peak": pct_of_peak, "pct_of_roofline": pct_of_roofline,
        "compute_floor_s": compute_floor_s, "memory_floor_s": memory_floor_s, "sol_s": sol_s,
        "sol_gap": sol_gap, "remaining_headroom": remaining_headroom,
        **resolution,
    }


def validate_sol_card(card: Mapping[str, object]) -> list[str]:
    """Return a list of problems with `card` (empty means valid).

    Checked deterministically and defensively -- this accepts any mapping
    (e.g. one round-tripped through JSON from an agent) and never raises;
    every failure is reported as a string instead.
    """
    problems: list[str] = []
    schema = card.get("schema")
    if schema not in ACCEPTED_SCHEMAS:
        problems.append(f"schema must be one of {ACCEPTED_SCHEMAS!r}, got {schema!r}")
    v2_fields = ("bandwidth_ceiling_basis", "bandwidth_ceiling_confidence",
                 "footprint_bytes", "bandwidth_ceiling_bracket",
                 "bandwidth_ceiling_extrapolated")
    v3_fields = ("compute_ceiling_witnessed", "compute_ceiling_attainment",
                 "compute_ceiling_witness")
    if schema == SCHEMA:
        for key in v3_fields:
            if key not in card:
                problems.append(f"{key} is required on a {SCHEMA} card")
        witnessed = card.get("compute_ceiling_witnessed")
        if not isinstance(witnessed, bool):
            problems.append(
                f"compute_ceiling_witnessed must be a bool, got {witnessed!r}")
        attainment = card.get("compute_ceiling_attainment")
        if witnessed is True:
            if not str(card.get("compute_ceiling_witness") or "").strip():
                problems.append("compute_ceiling_witnessed is True but no witness is named")
            if not (isinstance(attainment, (int, float)) and 0 < attainment <= 1 + 1e-9):
                problems.append(
                    "compute_ceiling_attainment must be achieved/peak in (0, 1] on a witnessed "
                    f"card, got {attainment!r}")
        elif witnessed is False and attainment is not None:
            # Not pedantry: a ratio without a witness is a number with nothing
            # behind it, and it reads downstream exactly like a witnessed one.
            problems.append(
                f"compute_ceiling_attainment must be None when unwitnessed, got {attainment!r}")
    elif schema in (SCHEMA_V2, SCHEMA_V1):
        for key in v3_fields:
            if key in card:
                problems.append(
                    f"{key} cannot appear on a {schema} card; label it {SCHEMA}")
    if schema in (SCHEMA, SCHEMA_V2):
        for key in v2_fields:
            if key not in card:
                problems.append(f"{key} is required on a {schema} card")
        basis = card.get("bandwidth_ceiling_basis")
        if basis not in ("scalar", "footprint_table"):
            problems.append(f"bandwidth_ceiling_basis must be 'scalar' or 'footprint_table', got {basis!r}")
        if basis == "footprint_table":
            fp = card.get("footprint_bytes")
            if not (isinstance(fp, (int, float)) and fp > 0):
                problems.append(
                    f"a footprint_table card must carry a positive footprint_bytes, got {fp!r}")
    elif schema == SCHEMA_V1:
        # A v1 card that carries v2 fields is not an old card, it is a v2 card
        # with the wrong label -- and it would be read with the v1 assumption
        # that its ceiling is a scalar.
        for key in v2_fields:
            if key in card:
                problems.append(f"{key} cannot appear on a {SCHEMA_V1} card; label it {SCHEMA}")
    if card.get("post_selection") is not True:
        problems.append("post_selection must be True: a SOL card only ever describes an "
                         "already-selected candidate")
    if card.get("arch") not in ARCH_CARDS:
        problems.append(f"arch must be one of {SUPPORTED_ARCHES}, got {card.get('arch')!r}")
    # Deliberately the dtype *vocabulary*, not the arch card's measured dtypes:
    # a calibration may legitimately supply a peak for a dtype no card measured
    # (that is what `calibration.peak_flops` is for), and a validator that
    # rejected the resulting card would make the calibration path unusable.
    if card.get("dtype") not in SUPPORTED_DTYPES:
        problems.append(f"dtype {card.get('dtype')!r} is not one of {SUPPORTED_DTYPES}")
    for key in ("peak_flops", "peak_bandwidth_bytes_s", "ridge_point", "achieved_flop_rate",
                "achieved_byte_rate", "elapsed_s", "roofline_ceiling_flops", "compute_floor_s",
                "memory_floor_s", "sol_s"):
        value = card.get(key)
        if not (isinstance(value, (int, float)) and value >= 0):
            problems.append(f"{key} must be a non-negative number, got {value!r}")
    intensity = card.get("arithmetic_intensity")
    if not (isinstance(intensity, (int, float)) and (intensity >= 0 or intensity == float("inf"))):
        problems.append(f"arithmetic_intensity must be a non-negative number (or inf), got {intensity!r}")
    if card.get("regime") not in ("compute_bound", "memory_bound"):
        problems.append(f"regime must be 'compute_bound' or 'memory_bound', got {card.get('regime')!r}")
    peak_flops = card.get("peak_flops")
    ceiling = card.get("roofline_ceiling_flops")
    if (isinstance(peak_flops, (int, float)) and isinstance(ceiling, (int, float))
            and ceiling > peak_flops * (1 + 1e-9)):
        problems.append("roofline_ceiling_flops cannot exceed peak_flops")
    pct_of_roofline = card.get("pct_of_roofline")
    achieved = card.get("achieved_flop_rate")
    if (isinstance(pct_of_roofline, (int, float)) and isinstance(achieved, (int, float))
            and isinstance(ceiling, (int, float)) and ceiling > 0
            and abs(pct_of_roofline - achieved / ceiling) > 1e-6):
        problems.append("pct_of_roofline is inconsistent with achieved_flop_rate / roofline_ceiling_flops")
    compute_floor = card.get("compute_floor_s")
    memory_floor = card.get("memory_floor_s")
    sol = card.get("sol_s")
    if (isinstance(compute_floor, (int, float)) and isinstance(memory_floor, (int, float))
            and isinstance(sol, (int, float))
            and abs(sol - max(compute_floor, memory_floor)) > 1e-12):
        problems.append("sol_s is inconsistent with max(compute_floor_s, memory_floor_s)")
    gap = card.get("sol_gap")
    if not (isinstance(gap, (int, float)) and gap > 0):
        problems.append(f"sol_gap must be a positive number, got {gap!r}")
    elif gap < 1.0 - 1e-9:
        # Finding (59). sol_s is a lower bound on time, so elapsed >= sol and
        # gap >= 1. A gap below 1 means the kernel beat its own speed of light,
        # which is a broken model, not a fast kernel: the peak of the wrong arch
        # (see finding 55), the wrong dtype's peak, or an undercounted
        # footprint. It is invisible to the consistency checks below, because
        # such a card is entirely self-consistent -- the arithmetic agrees with
        # itself and describes something impossible.
        problems.append(
            f"sol_gap is {gap!r}, below 1: measured time is under the speed-of-light "
            f"bound, so the SOL model is wrong (check the arch/dtype peaks and the "
            f"byte footprint), and remaining_headroom would be negative")
    elapsed_s = card.get("elapsed_s")
    if (isinstance(elapsed_s, (int, float)) and elapsed_s > 0
            and isinstance(sol, (int, float)) and sol > 0
            and isinstance(gap, (int, float))
            and abs(gap - elapsed_s / sol) > 1e-6):
        problems.append("sol_gap is inconsistent with elapsed_s / sol_s")
    headroom = card.get("remaining_headroom")
    if not (isinstance(headroom, (int, float)) and headroom < 1.0):
        problems.append(f"remaining_headroom must be a number below 1, got {headroom!r}")
    elif isinstance(gap, (int, float)) and gap > 0 and abs(headroom - (1.0 - 1.0 / gap)) > 1e-6:
        problems.append("remaining_headroom is inconsistent with 1 - 1/sol_gap")
    return problems


def _parser():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--achieved-flops", type=float, required=True)
    p.add_argument("--achieved-bytes", type=float, required=True)
    p.add_argument("--elapsed-s", type=float, required=True)
    p.add_argument("--dtype", default="bf16")
    # No default. The gfx90a card here is a
    # physical-peak reference whose own `source` says it is not a measurement;
    # the gfx942 card is `measured: True`. Silently picking the former on a
    # gfx942 box understates remaining headroom by ~3.3x.
    p.add_argument("--arch", required=True, choices=SUPPORTED_ARCHES)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    import json
    import sys
    args = _parser().parse_args(argv)
    try:
        card = build_sol_card(post_selection=True, achieved_flops=args.achieved_flops,
                               achieved_bytes=args.achieved_bytes, elapsed_s=args.elapsed_s,
                               dtype=args.dtype, arch=args.arch)
    except SOLCardError as exc:
        print(f"sol_card: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(card, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
