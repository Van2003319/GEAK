#!/usr/bin/env python3
"""Deterministic per-context robust statistics for geak-qd-v2 measurements.

kernel_lane.js's QD v2 verify contract carries, per benchmark context,
`case_measurement_samples: [{name, samples, median, mad, lower, upper}, ...]`
(QD_CASE_SAMPLES_SCHEMA) and folds those into cell admission via
`qdCaseRobust`/`qdRobust`, which trust whatever median/mad/lower/upper an
agent reports. This module computes that tuple deterministically from the raw
repeated-measurement samples themselves (kernel_lane.js's own
QD_REPEAT_MEASUREMENTS: 3 is the expected sample count, but nothing here
assumes exactly 3), so admission decisions rest on arithmetic instead of a
model's self-report.

Robustness matters here because GPU wall-clock timings are noisy and
occasionally spike (a stalled SMI query, a cold cache, a neighbor process):
the median resists those spikes far better than a mean, and MAD-based bounds
widen instead of narrowing under an outlier, which is the conservative
direction kernel_lane.js already leans (`robust.lower > incumbent.robust.upper`
before a cell can be replaced).
"""
from __future__ import annotations

import math
import statistics
from typing import Mapping, Sequence

SCHEMA = "geak.qd-robust-stats/v1"

# Admission uses the workflow's deliberately conservative interval contract:
# median ± 2*MAD. This is not a normal-distribution confidence interval; it is
# a deterministic noise guard whose only purpose is to require clear separation
# before one route/context incumbent replaces another.
MAD_BOUND_MULTIPLIER = 2.0

# Speedups/latencies are strictly positive; a bound is clamped here rather
# than left to go to or below zero, which would otherwise make downstream
# log-space or ratio combinations undefined.
_MIN_BOUND = 1e-9

# Finding (26). MAD alone is not a sufficient noise guard, for two reasons that
# both bite hardest exactly where the route is smallest:
#
#   1. MAD can be *zero*. Any repeated value in an odd-sized sample can put the
#      median absolute deviation at 0 -- on `decode_m2_square` that happens in
#      3.2% of n=3 draws from a 20-sample same-variant pool -- and then
#      `lower == upper == median`, a zero-width interval that admits a
#      challenger which is better by one nanosecond. The gate is strictest on
#      paper and absent in practice at the same moment.
#   2. Even when MAD is non-zero, three samples of a multi-modal route
#      underestimate its spread. Splitting one variant's own samples into two
#      arms of three on `decode_m2_square` yields a median apparent difference
#      of 4.5%, and |difference| >= 4.19% just over half the time. A 4% "win"
#      there is a coin flip, and this ledger has read such numbers before.
#
# So the interval carries a floor: a relative half-width measured by repeating
# ONE variant and observing how far it moves on its own. These are the machine-L
# v98 numbers -- max relative deviation from the median over 4 same-variant full
# suite runs (runs 1672/1673/1676/1677), and over 20 isolated runs for
# `decode_m2_square` (1687-1721, which also showed GPU 2 and GPU 3 agreeing to
# 0.08%, so this is within-GPU process variance and not silicon).
#
# They span an order of magnitude -- 0.5% on `prefill_m256_down` against 7.2%
# on `decode_m2_square` -- which is the actual finding: a single repeat-count
# rule cannot serve both ends of this suite. Treat them as LOWER bounds; n=4
# cannot see the tail, and re-measure on a new machine, since these are
# machine-L values and nothing pools across a machine boundary.
#
# That last sentence was a comment, not a mechanism -- (55) -- and the machine
# did change. Re-measured on machine N (`tw035`, MI300X/gfx942) with the (105)
# debiased harness, 8 same-variant full-suite runs, in
# `exp/opt_bf16_20260814/noisefloor_tw035_20260816/`. The table moved in BOTH
# directions, which is why importing L's numbers onto N would have been worse
# than having none:
#
#   decode_m2_square     0.072 -> 0.038   L was ~2x too WIDE  (refused real wins)
#   prefill_m2048_square 0.011 -> 0.037   L was 3.3x too NARROW
#   decode_m64_square    0.007 -> 0.020   L was 2.8x too NARROW
#   prefill_m256_down    0.005 -> 0.014   L was 2.7x too NARROW
#   prefill_m512_up      0.007 -> 0.015   L was 2.1x too NARROW
#
# The narrow direction is the one the comment below always warned about, and
# `prefill_m2048_square` is the route carrying the single largest claimed win in
# this ledger (1.57). Under L's floor a 1.1% move there cleared the gate; the
# route's own same-variant spread on this box is 3.7%.
#
# So the floors are keyed by machine and nothing pools across the boundary. An
# unknown machine does not silently borrow another one's numbers -- it gets the
# widest value measured for that route anywhere, which is the fail-closed
# direction.
MEASURED_NOISE_FLOOR_BY_MACHINE = {
    # machine L -- runs 1672/1673/1676/1677 (+1687-1721 isolated), v98, gfx90a
    "L": {
        "decode_m2_square": 0.072,
        "decode_m16_square": 0.033,
        "prefill_m1024_down": 0.026,
        "decode_m8_up": 0.020,
        "decode_m32_down": 0.020,
        "prefill_m128_square": 0.019,
        "decode_m96_up": 0.011,
        "prefill_m2048_square": 0.011,
        "decode_m64_square": 0.007,
        "prefill_m512_up": 0.007,
        "prefill_m256_down": 0.005,
    },
    # machine N -- tw035, gfx942, ship point bc7ea649e9ea3b7e, 8 same-variant
    # runs under the (105) burn+ABBA harness, nf_1..nf_8.json
    "N": {
        "decode_m2_square": 0.0378,
        "prefill_m2048_square": 0.0368,
        "prefill_m128_square": 0.0229,
        "decode_m32_down": 0.0207,
        "decode_m16_square": 0.0203,
        "decode_m64_square": 0.0198,
        "decode_m8_up": 0.0181,
        "prefill_m512_up": 0.0149,
        "prefill_m256_down": 0.0137,
        "prefill_m1024_down": 0.0127,
        "decode_m96_up": 0.0108,
    },
    # machine O -- tw054, gfx942, container restored from a snapshot onto a new
    # host on 2026-08-16. SAME ship point bc7ea649e9ea3b7e as machine N, 8
    # same-variant repeats across 2 lock acquisitions on GPU 2, under the
    # discard+BCCB harness. Relative half-width, 2*MAD(speedup)/median(speedup).
    #
    # Note this table is not machine N's scaled by a constant -- it is wider on
    # the two shortest decode routes and NARROWER on almost everything else.
    # decode_m16_square goes 0.0203 -> 0.0588 (2.9x) while prefill_m2048_square
    # goes 0.0368 -> 0.0060 (6x the other way). A single global fudge factor
    # applied to N's table would have been wrong in both directions at once,
    # which is the argument for keying by machine rather than scaling.
    "O": {
        "decode_m16_square": 0.0588,
        "decode_m2_square": 0.0416,
        "prefill_m128_square": 0.0209,
        "decode_m32_down": 0.0114,
        "prefill_m256_down": 0.0097,
        "decode_m96_up": 0.0073,
        "prefill_m1024_down": 0.0062,
        "prefill_m2048_square": 0.0060,
        "decode_m64_square": 0.0046,
        "prefill_m512_up": 0.0045,
        "decode_m8_up": 0.0026,
    },
    # machine P -- tw008, gfx942, the host run 16 moved onto at ~16:22 when the
    # container was restored a second time. SAME ship point bc7ea649e9ea3b7e as
    # machines N and O, so the three tables compare directly.
    #
    # Derived from 13 same-variant full-suite PARENT runs already on disk from
    # run 16 -- round_1 verify perf_{2,3,5,8}_P, round_2/engineer_0
    # perf_{2,3,5}_P and rep_A_p{1,2,3}, round_2/engineer_1 interleave
    # parent_ws_r{1,2,3} -- every one of them `primed`, spanning ~2.5 hours and
    # at least four lock acquisitions. Same statistic as N and O:
    # 2*MAD(speedup)/median(speedup).
    #
    # Two honest caveats, because they push in opposite directions and a reader
    # should not have to reconstruct them:
    #
    #  - Mixed harness. N and O were each measured under one protocol; these 13
    #    come from three (the round-1 eight-run rotation, engineer_0's rep_A
    #    triples, engineer_1's P/C alternation). Spanning hours and lock
    #    acquisitions folds session-to-session DRIFT into the spread, so this
    #    table is conservative -- wider than a within-session floor would be.
    #    That is the direction this module already argues for: too wide costs a
    #    real improvement one more round, too narrow admits noise as an elite.
    #  - Free. No GPU time was spent; the samples were a byproduct of the round.
    #    That is also why they exist at all, and it is the reason this table
    #    should be re-derived from a dedicated same-protocol sweep when the GPU
    #    is next idle rather than treated as final.
    #
    # Against O, this is WIDER on ten of eleven routes -- prefill_m512_up 4.8x,
    # prefill_m256_down 4.4x, decode_m96_up 3.7x -- and 9x NARROWER on
    # decode_m16_square. Two consequences worth stating plainly. First, no
    # global scale factor relates the two tables, for the third machine
    # boundary running; keying by machine keeps earning itself. Second, the run
    # that produced these samples was gated on O's floors, which are too narrow
    # here on ten of eleven routes -- the noise-admitting direction. Nothing was
    # admitted in run 16, so nothing is retroactively suspect, but a run that
    # HAD admitted on O's floors while standing on tw008 would need re-checking.
    "P": {
        "prefill_m256_down": 0.0430,
        "prefill_m128_square": 0.0287,
        "decode_m96_up": 0.0270,
        "decode_m2_square": 0.0224,
        "prefill_m512_up": 0.0218,
        "decode_m32_down": 0.0178,
        "prefill_m1024_down": 0.0178,
        "prefill_m2048_square": 0.0109,
        "decode_m64_square": 0.0074,
        "decode_m16_square": 0.0066,
        "decode_m8_up": 0.0047,
    },
}

# An unmeasured context gets the widest floor measured for ANY route on ANY
# machine, not the mean and not zero. The asymmetry is deliberate: too wide only
# costs a real improvement one more round of measurement, while too narrow
# silently admits noise into the archive as an elite and every later comparison
# inherits it. Measuring the real floor for a new route costs four same-variant
# repeats -- eight if the route is short, because n=4 cannot see the tail.
#
# Computed over the MEASURED tables only, before any provisional table is added
# below -- a provisional table is made OF this value, so folding it back in
# would be circular.
DEFAULT_NOISE_FLOOR = max(v for table in MEASURED_NOISE_FLOOR_BY_MACHINE.values()
                          for v in table.values())

# --- which epoch is this, and who says so ----------------------------------
#
# Finding (126). `CURRENT_MACHINE` was a hand-set constant with the host name
# only in a comment beside it. That is a remembered fact about a machine, kept
# in a file that travels with the container, and it survived exactly as long as
# nobody moved the container: this line read "P" (tw008) while `hostname`
# returned tw003. Nothing anywhere would have said so. The floors would simply
# have been tw008's, applied to a different box, silently, in the direction that
# admits noise on whichever routes happen to be quieter there.
#
# So the host name is data now, and the epoch letter is checked against the box
# the process is actually running on (`test_the_epoch_letter_matches_the_host`).
# (123) is why this is even recoverable after the fact -- rocprofv3 stamps
# $(hostname) into its output directories as a side effect -- but provenance a
# gate reads beats provenance an archaeologist can reconstruct.
# Entries are APPENDED in epoch order, and that order is load-bearing: a host
# can carry more than one epoch. R is tw008, the same box that carried P -- the
# container was restored onto it a second time. `machine_for_host` therefore
# resolves a host to its NEWEST epoch, so re-registering a box does not quietly
# resurrect the older letter and, with it, the older floors.
MACHINE_HOSTNAME = {
    "L": None,        # gfx90a box, pre-dates the convention; hostname not recorded
    "M": None,        # ditto
    "N": "tw035",
    "O": "tw054",
    "P": "tw008",
    "Q": "tw003",
    "R": "tw008",     # same box as P, second container; see the R table below
    "S": "tw054",     # same box as O, second container; see the S table below
    "T": "tw046",     # wave 4 restore; a box this lane had never run on before
    "U": "tw049",     # wave 5 restore, MID-ROUND: see the U block below
    "V": "tw051",     # wave 6 restore; a box this lane had never run on before
    "W": "tw042",     # wave 7 restore; another box new to the lane
}

# Epochs whose table was never measured. Their floors are the fail-closed
# DEFAULT for every route: structurally a table, so nothing downstream needs a
# special case, but not a measurement and not reported as one.
#
# Membership of this set is what `floor_is_measured` reads. WHY a given epoch is
# in it belongs beside that epoch's own table, not here: installing a measured
# table has to be able to retire the claim and the sentence in one edit, and
# `deprovisionalize_epoch.py` can only own text it can anchor to.
PROVISIONAL_MACHINES = {"S"}
# machine Q -- tw003. MEASURED: 8 complete same-variant primed repeats, source_hash 943b15834616ca9b857a59b94c548a7392c621b89093a792b69c6d6cf8a5db75.
# Installed by deprovisionalize_epoch.py from the sweep verdict; the statistic is 2*MAD(speedup)/median(speedup) per route, floors below MIN_FLOOR (0.002) clamped up. Floors do not pool across a machine boundary, so this table is a reading of this box only.
MEASURED_NOISE_FLOOR_BY_MACHINE["Q"] = {
    "decode_m16_square": 0.0305,
    "decode_m2_square": 0.0189,
    "decode_m32_down": 0.0501,
    "decode_m64_square": 0.0047,
    "decode_m8_up": 0.0261,
    "decode_m96_up": 0.0064,
    "prefill_m1024_down": 0.0097,
    "prefill_m128_square": 0.0232,
    "prefill_m2048_square": 0.0072,
    "prefill_m256_down": 0.0720,
    "prefill_m512_up": 0.0159,
}
# machine R -- tw008. MEASURED: 8 complete same-variant primed repeats, source_hash f3da61b3e2b673f7cf2c2847a668432860f90c37b7eb848c863ad8fdacddb2fa.
# Installed by deprovisionalize_epoch.py from the sweep verdict; the statistic is 2*MAD(speedup)/median(speedup) per route, floors below MIN_FLOOR (0.002) clamped up. Floors do not pool across a machine boundary, so this table is a reading of this box only.
MEASURED_NOISE_FLOOR_BY_MACHINE["R"] = {
    "decode_m16_square": 0.0110,
    "decode_m2_square": 0.0093,
    "decode_m32_down": 0.0712,
    "decode_m64_square": 0.0139,
    "decode_m8_up": 0.0197,
    "decode_m96_up": 0.0248,
    "prefill_m1024_down": 0.0185,
    "prefill_m128_square": 0.0092,
    "prefill_m2048_square": 0.0029,
    "prefill_m256_down": 0.0364,
    "prefill_m512_up": 0.0245,
}


# machine S -- tw054. PROVISIONAL: the greedy lane's container was restored from
# a tw008 snapshot onto tw054, and this line still read "R" afterwards -- finding
# (126) recurring, caught by the host cross-check rather than by anyone noticing.
# The post-restore integrity pass verified digests, lane HEAD and the oracle but
# never the floor table, so it passed clean while the floors in use belonged to
# another box. It cost nothing only because the foreign tenant has held all eight
# GPUs since the restore: NO timing has been taken on tw054.
#
# Rounds 5-7 (1.3524 -> 1.37928 -> 1.39884) were NOT measured on tw008 under R --
# that was an inference from a restore record that omitted the hostname, and it
# was wrong. rocprofv3 names its output directory after the host, so every round
# left one on disk: wave 1 (rounds 1-4) ran on tw051 11:32-15:57 UTC 08-17, wave 2
# (rounds 5-7) on tw003 16:43-20:52, and tw054 only from the 21:29 restore. The
# epoch registered for tw003 carried no measured table -- 0.072 on every route --
# and the round 6 record corroborates the floor that was actually live ("against
# the 7% noise floor", PIPELINE_PROGRESS_GREEDY.md:2019; R is 0.0093/0.0110 on
# those routes). So those rounds stand A FORTIORI, having cleared a floor wider
# than any measured table -- not on the strength of R's narrow one. Note also that
# tw051 is registered to no epoch at all yet CURRENT_MACHINE named tw003's letter
# throughout wave 1: finding (126) twice over, harmless both times only because
# tw003's table was the fail-closed default.
#
# (Epochs are named here by host, not by letter, on purpose: `stale_prose` in
# deprovisionalize_epoch.py flags any comment block that pairs the word
# PROVISIONAL with a bare epoch letter, and this block is about S. Spelling
# tw003's letter here would make retiring that epoch report this paragraph as
# stale prose it cannot fix.)
#
# NOT resolved to O, which is the retired first container on this same box. That
# is precisely the re-registered-box trap `machine_for_host` documents below, and
# O disagrees with R by 2.1-5.3x in one direction and up to 7.6x in the other, so
# inheriting it would be a guess wearing a measurement's clothes.
#
# The cost is real and is the right way round: at 0.072 on every route this lane's
# ~1-1.4%-per-round wins are unreadable and cannot be admitted. So measuring the
# real tw054 floors is the FIRST GPU work of wave 3, ahead of round 8 -- 8
# same-variant full-suite repeats, the (105) debiased harness, same statistic
# (2*MAD(speedup)/median(speedup)) -- then `deprovisionalize_epoch.py --apply`,
# which replaces this comment along with the table below it.
MEASURED_NOISE_FLOOR_BY_MACHINE["S"] = {
    route: DEFAULT_NOISE_FLOOR for route in MEASURED_NOISE_FLOOR_BY_MACHINE["P"]
}

# machine T -- tw046. MEASURED: 8 complete same-variant primed repeats, source_hash 943b15834616ca9b857a59b94c548a7392c621b89093a792b69c6d6cf8a5db75.
# Installed by deprovisionalize_epoch.py from the sweep verdict; the statistic is 2*MAD(speedup)/median(speedup) per route, floors below MIN_FLOOR (0.002) clamped up. Floors do not pool across a machine boundary, so this table is a reading of this box only.
MEASURED_NOISE_FLOOR_BY_MACHINE["T"] = {
    "decode_m16_square": 0.0470,
    "decode_m2_square": 0.0438,
    "decode_m32_down": 0.0275,
    "decode_m64_square": 0.0208,
    "decode_m8_up": 0.0280,
    "decode_m96_up": 0.0121,
    "prefill_m1024_down": 0.0043,
    "prefill_m128_square": 0.0123,
    "prefill_m2048_square": 0.0118,
    "prefill_m256_down": 0.0458,
    "prefill_m512_up": 0.0079,
}

# machine U -- tw049. MEASURED: 8 complete same-variant primed repeats, source_hash f3da61b3e2b673f7cf2c2847a668432860f90c37b7eb848c863ad8fdacddb2fa.
# Installed by deprovisionalize_epoch.py from the sweep verdict; the statistic is 2*MAD(speedup)/median(speedup) per route, floors below MIN_FLOOR (0.002) clamped up. Floors do not pool across a machine boundary, so this table is a reading of this box only.
MEASURED_NOISE_FLOOR_BY_MACHINE["U"] = {
    "decode_m16_square": 0.0140,
    "decode_m2_square": 0.0121,
    "decode_m32_down": 0.0249,
    "decode_m64_square": 0.0071,
    "decode_m8_up": 0.0438,
    "decode_m96_up": 0.0059,
    "prefill_m1024_down": 0.0034,
    "prefill_m128_square": 0.0093,
    "prefill_m2048_square": 0.0032,
    "prefill_m256_down": 0.0448,
    "prefill_m512_up": 0.0077,
}


def machine_for_host(hostname: str | None = None) -> str | None:
    """The epoch letter registered for `hostname`, or None if it is a new box."""
    if hostname is None:
        import socket
        hostname = socket.gethostname()
    # The NEWEST epoch registered for this host, not the first. A box can be
    # re-used (tw008 carried both P and R), and returning the first match would
    # resolve a re-registered box to the retired letter -- reinstating floors
    # measured in a different container, which is finding (126) with extra
    # steps. Entries are appended in epoch order, so the last match is current.
    latest = None
    for letter, host in MACHINE_HOSTNAME.items():
        if host is not None and host == hostname:
            latest = letter
    return latest


# machine V -- tw051. MEASURED: 8 complete same-variant primed repeats, source_hash c4b6dba073440f108e3f07585272b1488850df3f95f8c7e3c926dccc1fc96355.
# Installed by deprovisionalize_epoch.py from the sweep verdict; the statistic is 2*MAD(speedup)/median(speedup) per route, floors below MIN_FLOOR (0.002) clamped up. Floors do not pool across a machine boundary, so this table is a reading of this box only.
MEASURED_NOISE_FLOOR_BY_MACHINE["V"] = {
    "decode_m16_square": 0.0087,
    "decode_m2_square": 0.0020,  # clamped to MIN_FLOOR
    "decode_m32_down": 0.0225,
    "decode_m64_square": 0.0132,
    "decode_m8_up": 0.0223,
    "decode_m96_up": 0.0090,
    "prefill_m1024_down": 0.0121,
    "prefill_m128_square": 0.0216,
    "prefill_m2048_square": 0.0093,
    "prefill_m256_down": 0.0120,
    "prefill_m512_up": 0.0107,
}


# machine W -- tw042. MEASURED: 8 complete same-variant primed repeats, source_hash f87a1ccd45be3f3ee060ce401f8119845ffd68efe9c45b2cc8475b97253d6786.
# Installed by deprovisionalize_epoch.py from the sweep verdict; the statistic is 2*MAD(speedup)/median(speedup) per route, floors below MIN_FLOOR (0.002) clamped up. Floors do not pool across a machine boundary, so this table is a reading of this box only.
MEASURED_NOISE_FLOOR_BY_MACHINE["W"] = {
    "decode_m16_square": 0.0101,
    "decode_m2_square": 0.0083,
    "decode_m32_down": 0.0045,
    "decode_m64_square": 0.0042,
    "decode_m8_up": 0.0043,
    "decode_m96_up": 0.0121,
    "prefill_m1024_down": 0.0042,
    "prefill_m128_square": 0.0136,
    "prefill_m2048_square": 0.0092,
    "prefill_m256_down": 0.0181,
    "prefill_m512_up": 0.0091,
}


CURRENT_MACHINE = "W"

MEASURED_NOISE_FLOOR = MEASURED_NOISE_FLOOR_BY_MACHINE[CURRENT_MACHINE]


def floor_is_measured(context: str | None, machine: str | None = None) -> bool:
    """Whether this route's floor on this epoch came from a measurement.

    A provisional table is shaped exactly like a measured one, which is what
    keeps every caller simple -- and is exactly why a caller that reports
    provenance must not infer it from `context in MEASURED_NOISE_FLOOR`.
    """
    m = CURRENT_MACHINE if machine is None else machine
    if m in PROVISIONAL_MACHINES:
        return False
    table = MEASURED_NOISE_FLOOR_BY_MACHINE.get(m)
    return bool(table) and context in table


def noise_floor(context: str | None, machine: str | None = None) -> float:
    """Relative interval half-width below which a difference is unreadable.

    `machine` selects the epoch's measured table and defaults to
    `CURRENT_MACHINE`. A machine with no measured table, or a route missing from
    the selected one, falls back to the widest floor measured anywhere rather
    than to that route's value on some other machine -- floors do not pool
    across a machine boundary any more than latencies do.
    """
    table = MEASURED_NOISE_FLOOR_BY_MACHINE.get(
        CURRENT_MACHINE if machine is None else machine)
    if table is None or context is None:
        return DEFAULT_NOISE_FLOOR
    return table.get(context, DEFAULT_NOISE_FLOOR)


def robust_stats(samples: Sequence[float], context: str | None = None
                 ) -> dict[str, float | int]:
    """{n, median, mad, bound_radius, lower, upper} for raw samples.

    `context` selects the measured noise floor (finding 26). The bound radius
    is `max(2*MAD, median * noise_floor(context))`, so a route that is quiet in
    three samples but loud in twenty cannot present a tight interval. Passing
    no context applies the widest measured floor, which is the fail-closed
    direction: it refuses admissions rather than granting them.

    - n == 0: every field is 0 -- there is no measurement to summarize.
    - n == 1: median equals the single sample, mad is 0, and the radius is the
      measured floor. One sample carries no spread information, so it is the
      last place that should present the narrowest bound.
    - n >= 2: mad is the median absolute deviation from the median; the radius
      is max(2*MAD, |median| * floor), clamped so lower never reaches zero.

    Parity note, and a warning about this docstring's own history. These lines
    used to claim the n == 1 case "matches kernel_lane.js's fallback of
    lower == upper == score". That stopped being true when finding (26) added
    the floor here, and the sentence stayed -- so anyone checking whether the
    two implementations agreed could read a confident parity claim and stop.
    The lane went on admitting elites with a bare 2*MAD radius for the whole
    period; see finding (58). `kernel_lane.js`'s `qdCaseRobust` now mirrors this
    function exactly, and `test_qd_lane_parity.py::NoiseFloorParityTest` plus
    `test_js_suite.py::RobustParityTest` are what keep that true -- one compares
    the tables, the other executes both implementations on the same samples.
    Do not restate parity in prose here; prose is not the mechanism.
    """
    xs = [float(x) for x in samples if isinstance(x, (int, float)) and not math.isnan(x)]
    n = len(xs)
    if n == 0:
        return {"n": 0, "median": 0.0, "mad": 0.0, "bound_radius": 0.0,
                "lower": 0.0, "upper": 0.0}
    median = statistics.median(xs)
    if n == 1:
        # Previously this returned a zero-width interval, matching
        # kernel_lane.js's point-estimate fallback. It now carries the measured
        # floor instead: one sample is the case with the *least* spread
        # information, so it is the last place that should present the
        # narrowest bound. Widening can only refuse admissions, never grant
        # one, so it cannot turn a rejection into an acceptance downstream.
        radius = abs(median) * noise_floor(context)
        return {"n": 1, "median": median, "mad": 0.0, "bound_radius": radius,
                "lower": max(_MIN_BOUND, median - radius), "upper": median + radius}
    mad = statistics.median([abs(x - median) for x in xs])
    bound_radius = max(mad * MAD_BOUND_MULTIPLIER,
                       abs(median) * noise_floor(context))
    lower = max(_MIN_BOUND, median - bound_radius)
    upper = median + bound_radius
    return {"n": n, "median": median, "mad": mad, "bound_radius": bound_radius,
            "lower": lower, "upper": upper}


def context_robust_stats(samples_by_context: Mapping[str, Sequence[float]]
                          ) -> list[dict[str, object]]:
    """`robust_stats` per context, shaped exactly like QD_CASE_SAMPLES_SCHEMA's
    items: {name, samples, median, mad, lower, upper}, sorted by context name
    for a deterministic, diff-friendly ordering.
    """
    out: list[dict[str, object]] = []
    for name in sorted(samples_by_context):
        xs = [float(x) for x in samples_by_context[name]]
        # The context name IS the noise-floor key, so a caller that already
        # groups samples per harness case gets the right floor for free and
        # cannot forget to pass it.
        stats = robust_stats(xs, context=name)
        out.append({"name": name, "samples": xs, "median": stats["median"],
                    "mad": stats["mad"], "lower": stats["lower"], "upper": stats["upper"]})
    return out


def combine_contexts(per_context: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Fold per-context {median, lower, upper, ...} rows into one overall
    {score, median, mad, lower, upper}, mirroring kernel_lane.js's
    qdContextScore averaging (mean across contexts, not a geomean -- QD v2's
    own score is a plain per-context average).

    The bound stays sound without any independence assumption: if every
    context's true value lies in [lower_i, upper_i], then by linearity the
    mean of the true values lies in [mean(lower_i), mean(upper_i)] regardless
    of how the contexts covary.
    """
    rows = list(per_context)
    if not rows:
        return {"score": 0.0, "median": 0.0, "mad": 0.0, "lower": 0.0, "upper": 0.0}
    median = sum(float(r["median"]) for r in rows) / len(rows)
    mad = sum(float(r.get("mad", 0.0)) for r in rows) / len(rows)
    lower = sum(float(r["lower"]) for r in rows) / len(rows)
    upper = sum(float(r["upper"]) for r in rows) / len(rows)
    return {"score": median, "median": median, "mad": mad, "lower": lower, "upper": upper}


def _parser():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("json_path", help="JSON file: {\"context_name\": [sample, sample, ...], ...}")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    import json
    import sys
    args = _parser().parse_args(argv)
    with open(args.json_path, encoding="utf-8") as fh:
        samples_by_context = json.load(fh)
    per_context = context_robust_stats(samples_by_context)
    payload = {"schema": SCHEMA, "per_context": per_context, "combined": combine_contexts(per_context)}
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
