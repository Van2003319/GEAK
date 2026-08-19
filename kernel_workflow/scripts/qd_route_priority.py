#!/usr/bin/env python3
"""Rank routes by what is left in them, not by how they compare to the vendor.

Finding (33). `speedup` measures this kernel against a moving opponent; the SOL
gap measures it against a fixed floor. Across the eleven-shape BF16 suite those
two rankings are **uncorrelated** (Spearman -0.08), and they disagree hardest at
the top -- which is the only part of a priority list anybody reads. Every route
this ledger has nominated as "the worst one" was nominated on the first
quantity, and the one nominated longest (`prefill_m128_square`) turns out to sit
at 67.4% of its achievable roofline with 11.7 us of slack in it.

Two numbers decide whether a route is worth a build, and neither is enough alone:

* **slack** -- how much time is there between where the route runs today and the
  hardware floor for its shape (finding 28/29's footprint-indexed SOL card).
* **noise floor** -- the smallest relative effect that is readable on that route
  at all (finding 26). It spans 3.5x across this suite on the current machine,
  and it is keyed by machine: re-measuring L -> N moved individual routes by up
  to 3.3x in both directions, so a floor quoted from another box is not a floor.
  No route's verdict changed across L -> N -- the ratio below is dominated by
  slack, not by the floor -- but two routes' `slack_to_floor` changed by more
  than 3x, so the ORDERING it induces is epoch-specific.

Their ratio is the thing. A route whose entire remaining headroom is smaller
than its own noise floor is **provably finished**: even reaching the SOL floor
exactly would produce a change no protocol used in this project could measure.

**A closure is a reading of a machine, not a property of a route.** On machines
L, N and O `decode_m2_square` was such a route, and proving it cost no GPU time.
On machine P (tw008) it is NOT: P's floor there is 0.0224 against O's 0.0416, so
its 2.90% headroom now clears its own floor at slack_to_floor 1.29 and it reads
`marginal`. Nothing about the kernel or the shape changed -- the box got quieter
on that route, and a route closed only because the room was loud reopens when
the room goes quiet. On P **no** route in the suite is closed; the two tightest
(`decode_m2_square` 1.29, `decode_m16_square` 1.40) are marginal. Do not carry a
remembered closure across an epoch boundary; re-read it from the current table.

**Provenance and staleness.** `SHIPPED_ELAPSED_MS` is a snapshot -- machine-L run
1670, ship point v98, the baseline recorded in the ledger banner. It is here so
the planner can rank routes without a run, not so anybody can quote a latency
from it. Every payload carries `elapsed_provenance`; when the ship point moves,
this table moves with it or the ranking silently describes an old kernel. Pass
`elapsed_ms_by_context` to override it with fresh numbers.
"""
from __future__ import annotations

import math
from typing import Mapping

import qd_robust_stats as robust
import qd_sol_card as sol

# v2, not v1: finding (92) added `needs_fresh_elapsed` to the verdict enum. A
# consumer written against v1 switches on `open|marginal|closed` and would take
# the new value for "not closed", which is the exact silent reading the finding
# is about. The version string is the only thing that can make that consumer
# fail loudly instead.
SCHEMA = "geak.qd-route-priority/v2"

#: M, N, K for every harness case id. Structural -- these come from the task
#: definition, not from a measurement, so they do not go stale with the kernel.
SUITE_SHAPES: Mapping[str, tuple[int, int, int]] = {
    "decode_m2_square": (2, 4096, 4096),
    "decode_m8_up": (8, 11008, 4096),
    "decode_m16_square": (16, 4096, 4096),
    "decode_m32_down": (32, 4096, 11008),
    "decode_m64_square": (64, 8192, 8192),
    "decode_m96_up": (96, 11008, 4096),
    "prefill_m128_square": (128, 4096, 4096),
    "prefill_m256_down": (256, 4096, 11008),
    "prefill_m512_up": (512, 11008, 4096),
    "prefill_m1024_down": (1024, 4096, 11008),
    "prefill_m2048_square": (2048, 4096, 4096),
}

#: Machine-L run 1670, ship point v98, `candidate_ms`. See the staleness note.
SHIPPED_ELAPSED_MS: Mapping[str, float] = {
    "decode_m2_square": 0.02456,
    "decode_m8_up": 0.05472,
    "decode_m16_square": 0.02552,
    "decode_m32_down": 0.06852,
    "decode_m64_square": 0.09260,
    "decode_m96_up": 0.06792,
    "prefill_m128_square": 0.03596,
    "prefill_m256_down": 0.11816,
    "prefill_m512_up": 0.16476,
    "prefill_m1024_down": 0.27096,
    "prefill_m2048_square": 0.18548,
}

ELAPSED_PROVENANCE = "machine-L run 1670, ship point v98, candidate_ms (full suite, autotune ON)"

ARCH = "gfx942"
DTYPE = "bf16"

#: A route is closed when reaching the SOL floor exactly would still be
#: unreadable. 1.0 is not a tuned threshold -- it is the definition.
CLOSED_RATIO = 1.0
#: Above the definition but inside the same order of magnitude: a real effect
#: there has to be nearly the whole remaining headroom to register.
MARGINAL_RATIO = 3.0


class RoutePriorityError(ValueError):
    """Raised for an unknown route or a non-positive elapsed time."""


def shape_workload(m: int, n: int, k: int) -> tuple[float, float]:
    """FLOPs and compulsory bytes for one BF16 GEMM shape.

    Bytes counts each operand and the output once. For these shapes that is
    also the working set, so the same number serves as the SOL card's traffic
    (`achieved_bytes`) and its footprint -- which is *not* true in general, and
    is why `build_sol_card` takes the two separately.
    """
    flops = 2.0 * m * n * k
    byte_count = 2.0 * (m * k + k * n + m * n)
    return flops, byte_count


#: Measured DRAM traffic (reads + writes), bytes, for every route. Finding (35):
#: rocprofv3 `FETCH_SIZE` + `WRITE_SIZE` -- which weight each L2->memory request
#: by its real 32/64/128 B size instead of assuming a line -- median over the
#: last 20 dispatches of the converged autotune configuration, one profile per
#: case, `exp/opt_bf16_20260814/one_prof_tcc_case.sh`. Cross-checked against
#: `TCC_MISS_sum` x 128 B, which agrees within 2% on ten of eleven routes
#: (`prefill_m1024_down` differs by 19%, so its misses are not all full lines
#: and the size-weighted figure is the one to trust).
MEASURED_TRAFFIC_BYTES: Mapping[str, float] = {
    "decode_m2_square": 34282880.0,
    "decode_m8_up": 119241472.0,
    "decode_m16_square": 39291584.0,
    "decode_m32_down": 116051648.0,
    "decode_m64_square": 183229536.0,
    "decode_m96_up": 124165408.0,
    "prefill_m128_square": 59912448.0,
    "prefill_m256_down": 245202560.0,
    "prefill_m512_up": 137272000.0,
    "prefill_m1024_down": 438361152.0,
    "prefill_m2048_square": 152730048.0,
}

TRAFFIC_PROVENANCE = ("FETCH_SIZE + WRITE_SIZE, one_prof_tcc_case.sh, median of the last 20 "
                      "dispatches of the converged config (machine L, ship point v98)")


def route_priority(context: str, elapsed_ms: float | None = None,
                   traffic_bytes: float | None = None) -> dict[str, object]:
    """SOL slack, noise floor, and the verdict their ratio implies.

    `traffic_bytes` overrides the compulsory minimum with what the memory
    system actually moved. Finding (34) measured 4.5x amplification on the one
    route that has L2 counters, which made its `t_memory` 3.4x too small -- so
    a compulsory-traffic floor is a *lower bound on time*, and every row says
    which of the two it used rather than leaving the reader to assume.
    """
    if context not in SUITE_SHAPES:
        raise RoutePriorityError(
            f"unknown harness case {context!r}: this table covers {tuple(SUITE_SHAPES)}. "
            "Contexts are exact harness case ids; a route cannot invent one.")
    elapsed_is_default = elapsed_ms is None
    if elapsed_is_default:
        elapsed_ms = SHIPPED_ELAPSED_MS[context]
        provenance = ELAPSED_PROVENANCE
    else:
        provenance = "caller-supplied"
    if not (isinstance(elapsed_ms, (int, float)) and math.isfinite(elapsed_ms) and elapsed_ms > 0):
        raise RoutePriorityError(f"elapsed_ms for {context!r} must be a positive finite number")

    m, n, k = SUITE_SHAPES[context]
    flops, compulsory_bytes = shape_workload(m, n, k)
    if traffic_bytes is None:
        traffic_bytes = MEASURED_TRAFFIC_BYTES.get(context)
        traffic_basis = "measured" if traffic_bytes is not None else "compulsory"
    else:
        traffic_basis = "measured"
    if traffic_bytes is None:
        traffic_bytes = compulsory_bytes
    elif not (isinstance(traffic_bytes, (int, float)) and math.isfinite(traffic_bytes)
              and traffic_bytes >= compulsory_bytes):
        # Below compulsory is not a tighter measurement, it is a broken one:
        # the operands have to cross the bus at least once.
        raise RoutePriorityError(
            f"traffic_bytes for {context!r} must be a finite number at least the compulsory "
            f"{compulsory_bytes:.0f}; got {traffic_bytes!r}")
    # Index the ceiling by TRAFFIC, not by the distinct working set. This
    # reverses what this function did before finding (35), and the reversal is
    # not a preference -- indexing by working set puts `decode_m16_square` and
    # `prefill_m128_square` at sol_gap 0.93 and 0.88, i.e. faster than the
    # hardware floor, which is impossible and therefore falsifies the choice.
    # The measured bandwidth curve of finding (28) was built by streaming N
    # bytes and dividing by time, so its x-axis was traffic all along; in a
    # streaming benchmark traffic and working set coincide and the two readings
    # are indistinguishable. A re-reading kernel is what tells them apart.
    # `max` and not plain `traffic_bytes`: traffic can never be below
    # compulsory (checked above), so this only guards the caller-supplied path.
    ceiling_index = max(compulsory_bytes, float(traffic_bytes))
    card = sol.build_sol_card(
        post_selection=True, achieved_flops=flops, achieved_bytes=float(traffic_bytes),
        elapsed_s=float(elapsed_ms) / 1000.0, dtype=DTYPE, arch=ARCH,
        footprint_bytes=ceiling_index)

    # A gap below 1.0 says the kernel finished faster than the memory system
    # could have delivered the bytes it is credited with -- impossible, so one
    # of the two inputs is wrong. Refuse rather than report it: the ranking
    # sorts such a row last as "closed", where a broken measurement is
    # indistinguishable from a finished route. Finding (35) is exactly this
    # check catching an indexing error that had already shipped.
    if float(card["sol_gap"]) < 1.0 - 1e-6:
        raise RoutePriorityError(
            f"{context!r} would run at sol_gap {float(card['sol_gap']):.4f}, i.e. faster than the "
            f"{ARCH} floor for {float(traffic_bytes):.0f} B in {float(elapsed_ms):.5f} ms. Either "
            "the traffic is over-counted, the elapsed is not this kernel's, or the measured "
            "ceiling table does not describe the machine that produced it.")

    floor = robust.noise_floor(context)
    headroom = float(card["remaining_headroom"])
    ratio = headroom / floor
    if ratio < CLOSED_RATIO:
        verdict = "closed"
    elif ratio < MARGINAL_RATIO:
        verdict = "marginal"
    else:
        verdict = "open"
    # Finding (92). Everything above is arithmetic on `elapsed_ms`, and when the
    # caller supplied none that number is `SHIPPED_ELAPSED_MS` -- a DIFFERENT
    # KERNEL's latency, measured at ship point v98 on machine L. The verdict is
    # then a statement about that kernel on that machine, dressed as a statement
    # about this candidate on this one.
    #
    # The direction of the error is what makes it serious rather than merely
    # imprecise. A faster shipped kernel leaves less headroom, so the default
    # pushes routes toward "closed" -- and a closed route sorts last, is not
    # proposed, and therefore never acquires the fresh measurement that would
    # reopen it. The mistake is self-sealing: it removes the evidence that would
    # correct it.
    #
    # So a defaulted elapsed does not produce a verdict at all. The computed one
    # is kept beside it, because it is a real reading of a real kernel and
    # throwing it away would just make the caller guess -- but it is labelled as
    # conditional, and nothing may sort on it.
    verdict_if_confirmed = verdict
    if elapsed_is_default:
        verdict = "needs_fresh_elapsed"
    return {
        "context": context,
        "m": m, "n": n, "k": k,
        "regime": card["regime"],
        "arithmetic_intensity": card["arithmetic_intensity"],
        "ridge_point": card["ridge_point"],
        "elapsed_ms": float(elapsed_ms),
        "elapsed_provenance": provenance,
        "sol_ms": float(card["sol_s"]) * 1000.0,
        "sol_gap": card["sol_gap"],
        "slack_ms": float(elapsed_ms) - float(card["sol_s"]) * 1000.0,
        "remaining_headroom": headroom,
        "noise_floor": floor,
        # Membership in the current table is NOT the same question. A
        # provisional epoch (finding 126) carries a structurally complete table
        # filled with the fail-closed default so that every code path behaves,
        # and every route in it would answer `True` to `context in ...` while
        # nothing on that box has been measured at all. Ask the module.
        "noise_floor_measured": robust.floor_is_measured(context),
        "slack_to_floor": ratio,
        "verdict": verdict,
        "verdict_if_elapsed_confirmed": verdict_if_confirmed,
        "elapsed_is_default": elapsed_is_default,
        "bandwidth_ceiling_confidence": card["bandwidth_ceiling_confidence"],
        "traffic_basis": traffic_basis,
        "traffic_bytes": float(traffic_bytes),
        "compulsory_bytes": compulsory_bytes,
        "traffic_amplification": float(traffic_bytes) / compulsory_bytes,
        "traffic_provenance": TRAFFIC_PROVENANCE if traffic_basis == "measured" else None,
    }


def rank_routes(contexts: list[str] | None = None,
                elapsed_ms_by_context: Mapping[str, float] | None = None,
                traffic_bytes_by_context: Mapping[str, float] | None = None,
                ) -> list[dict[str, object]]:
    """Every named route, richest first.

    Ordering is by absolute `slack_ms`, not by `sol_gap`: a 3x gap on a route
    that runs for 25 us is worth less than a 2x gap on one that runs for 270,
    and the suite geomean does not care which is which. Closed routes sort last
    regardless -- their slack is real but unreachable, so it must never head a
    priority list.

    Only a route CLOSED ON ITS OWN MEASUREMENT sorts last (92). A route whose
    elapsed was defaulted is `needs_fresh_elapsed`, and it sorts with the live
    ones: "we have not measured this" and "there is nothing here" are opposite
    states, and collapsing them buries precisely the routes that most need a
    measurement, permanently, because a buried route is never dispatched and so
    never measured.
    """
    names = list(contexts) if contexts else sorted(SUITE_SHAPES)
    supplied = dict(elapsed_ms_by_context or {})
    traffic = dict(traffic_bytes_by_context or {})
    for label, given in (("elapsed_ms_by_context", supplied),
                         ("traffic_bytes_by_context", traffic)):
        unknown = sorted(set(given) - set(SUITE_SHAPES))
        if unknown:
            raise RoutePriorityError(
                f"{label} names routes that are not harness cases: {unknown}")
    rows = [route_priority(name, supplied.get(name), traffic.get(name)) for name in names]
    rows.sort(key=lambda r: (r["verdict"] == "closed", -float(r["slack_ms"])))
    return rows
