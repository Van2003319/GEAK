#!/usr/bin/env python3
"""The mechanisms this kernel has already measured shut, and why.

`qd_route_priority.py` answers "which *route* is worth a slot". Nothing has ever
answered "which *mechanism* has already been tried on it". That asymmetry is the
self-sealing shape from finding (92) pointed the other way: a route recorded as
closed is never proposed and so never re-measured, while a mechanism closed
only in prose is re-proposed every time a planner reads the roofline and
rediscovers the same obvious idea. Most of the entries below were each rediscovered
at least once after they were closed, because the closure lived in a paragraph
and the proposal was generated from a number.

So the closures are data here, and a proposal can be checked against them.

Three rules govern this file, and they are what keeps it from becoming a way to
stop work rather than to aim it:

1. **A closure cites a measurement, never an argument.** Every entry names the
   finding, the variant that was built, the effect that was measured, and the
   negative control. An entry with a `reason` and no number does not belong here
   -- that is a hypothesis, and hypotheses are what the slots are for.
2. **A closure is only as tight as the floor that justified it** (finding 92,
   restated in `qd_route_priority`). `bound` records what the prize was bounded
   *at*, and `epoch` records whose noise floor did the bounding. A mechanism
   bounded at "under 2%" on a box whose floor was 7% is bounded at 7%, and it
   reopens on a quieter box. This is why the entries carry an epoch letter and
   not just a finding number.
3. **Every entry states what would reopen it.** A closure with no reopen
   condition is not a measurement, it is a ban. `reopens_when` is mandatory and
   the tests enforce that it is non-empty.

The checker is deliberately advisory-with-teeth: naming a closed mechanism is
allowed if the proposal also carries a `reopen_justification` that says which
`reopens_when` condition now holds. Refusing outright would make the file a
ratchet, and a ratchet on a machine that changes every epoch is wrong.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SUITE = "suite"


@dataclasses.dataclass(frozen=True)
class Closure:
    axis: str                     # the mechanism, in the planner's vocabulary
    aliases: tuple[str, ...]      # what a proposal is likely to call it
    routes: tuple[str, ...]       # (SUITE,) or the specific routes measured
    epoch: str                    # whose noise floor bounded the result;
                                  # one letter, or comma-separated letters
                                  # when a closure accumulated arms across
                                  # epochs (each arm bounded by its own box)
    finding: str                  # where the measurement is written down
    built: str                    # the variant that actually ran
    measured: str                 # the number, with its sign
    control: str                  # the negative control and what it did
    bound: str                    # what the prize is bounded at, honestly
    reopens_when: str             # mandatory

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


CLOSED: tuple[Closure, ...] = (
    Closure(
        axis="rasterization / L2 traffic reduction",
        aliases=("rasterization", "rasterisation", "xcd_remapped", "grouped_m",
                 "swizzle", "l2 traffic", "traffic amplification", "re-read",
                 "aligned tile", "tile alignment"),
        routes=("prefill_m1024_down", "prefill_m256_down"),
        epoch="N",
        finding="(38), demoting item 7",
        built="the aligned-tile experiment",
        measured="removed 36% of m1024_down's DRAM traffic and 25% of "
                 "m256_down's, lifting L2 hit rate by 12 and 16 points -- "
                 "medians moved +6.8% SLOWER and flat, respectively",
        control="decode_m8_up (1.1x re-read) did not move: 55.62 us both arms",
        bound="traffic is not the binding term on these routes; the "
              "'361 MB -> 135 MB' prize is not a time prize",
        reopens_when="the kernel's compute side changes enough that traffic "
                     "could become binding, or a route outside these two shows "
                     "both high re-read and a hit-rate deficit",
    ),
    Closure(
        axis="prefetch depth / s_waitcnt placement (memory side of 7b)",
        aliases=("prefetch", "s_waitcnt", "waitcnt", "prefetch depth",
                 "memory latency hiding", "lds double-buffer",
                 "double buffering", "pipeline depth"),
        routes=(SUITE,),
        epoch="N",
        finding="(73)",
        built="VmemLatency measurement across seven routes",
        measured="exposure is 0.05-0.27 of VmemLatency on every route measured "
                 "-- the pipeline already hides 73-95% of memory latency, and "
                 "that is an upper bound",
        control="decode_m8_up, chosen as the control BECAUSE it lacks the "
                "non-overlap signature, is the most exposed route at 0.27 -- "
                "which refutes the entry's own premise",
        bound="there is no ~135 us of un-hidden memory latency to recover. "
              "PARTLY FALSIFIED IN ROUND 7 OF THIS LANE, and left standing "
              "rather than deleted because the correction is the useful part: "
              "PF=4 on macro_gemm_128x64_bk64_nt recovered 6.14 us on "
              "decode_m96_up (66.48 -> 60.34, +10.2%). So the suite-wide "
              "'nothing to recover' is false at the ONE route that runs at "
              "2 CTA/CU. It survives everywhere else, and the reopen condition "
              "below was the wrong one -- exposure/VmemLatency did not predict "
              "this win; LDS residency did",
        reopens_when="a variant changes the load schedule enough to raise "
                     "exposure, measured as exposure/VmemLatency, above ~0.4; "
                     "or -- the condition that actually fired -- a route sits "
                     "at 2 CTA/CU or below, where see the PF>1 entry, which "
                     "supersedes this one on the prefetch-depth axis",
    ),
    Closure(
        axis="barrier count (barrier side of 7b)",
        aliases=("barrier", "__syncthreads", "s_barrier", "drain barrier"),
        routes=(SUITE,),
        epoch="N",
        finding="(74)",
        built="ws_bar2 = ws_a plus one forced drain barrier per stage, "
              "ABBA-interleaved against a same-epoch rebuild of the control",
        measured="one barrier per stage costs +0.9% geomean",
        control="the same-epoch rebuild of ws_a itself (finding 75: the control "
                "arm must be rebuilt in the epoch that times it)",
        bound="the kernel has one barrier, so the entire barrier prize is "
              "1-2% against a residual of tens of percent",
        reopens_when="a variant introduces several barriers per stage, where "
                     "1-2% each stops being negligible",
    ),
    Closure(
        axis="MFMA instruction shape (16x16x16 -> 32x32x8)",
        aliases=("mfma shape", "32x32x8", "mfma instruction mix",
                 "instruction mix", "rocwmma", "large mfma"),
        routes=(SUITE,),
        epoch="N",
        finding="(41)-(43)",
        built="v109, after (42) established the axis was unreachable through "
              "rocWMMA, which emulates the large shape rather than selecting it",
        measured="reaching the axis was the achievement; taking it is a "
                 "13-16% LOSS. (42)'s prediction is refuted.",
        control="the 16x16x16 shipped kernel, same build",
        bound="closed as a loss, not as a null",
        reopens_when="a compiler or rocWMMA release actually selects the large "
                     "shape rather than emulating it -- re-price, do not assume",
    ),
    Closure(
        axis="active-CU fraction as the clock residual explanation",
        aliases=("active cu", "cu_frac", "cu fraction", "clock residual",
                 "active-cu"),
        routes=(SUITE,),
        epoch="N",
        finding="(45)",
        built="the vendor rows pooled INTO the fit rather than extrapolated to "
              "-- a genuine hole in (39)'s dismissal",
        measured="absorbs a quarter of the vendor term, leaves a +6.1% arm "
                 "dummy standing, and lands at p = 0.134 against 20 000 random "
                 "relabelings of its own data",
        control="the 20 000 relabelings",
        bound="n = 22 cannot support the question being asked of it",
        reopens_when="the sample grows well past n = 22, or a direct per-CU "
                     "occupancy counter becomes available on this stack",
    ),
    Closure(
        axis="output_path on split-K routes",
        # Two vocabularies, and only the first was here. `output_path`,
        # `direct_store`, `atomic_fixup` are what the mutation is CALLED once
        # you already know this kernel. Nobody rediscovering the idea writes
        # those words -- they write the mechanism, from the reduce call site's
        # own comment ("the only way to remove it is to stop launching a second
        # kernel at all"): last CTA to arrive does the reduction, per-tile
        # arrival counter, fuse the fixup, one kernel instead of two. This
        # entry gave a clean pass to a proposal spelled exactly that way, which
        # is the expensive direction of the matcher trade-off -- a false clear
        # costs a build and a measured round, a false flag costs a sentence of
        # justification.
        aliases=("output_path", "direct_store", "atomic_fixup",
                 "lds_staged_store", "epilogue store path",
                 "last cta", "last arrival", "arrival counter", "fused fixup",
                 "fuse the reduc", "fuse the fixup", "inline the reduc",
                 "single kernel split", "single-kernel split",
                 "one kernel instead of two", "second launch",
                 "second kernel", "stop launching a second",
                 "remove the reduce", "remove the reduction",
                 "eliminate the reduce", "grid sync", "cooperative launch",
                 "threadfence reduc"),
        routes=("decode_m32_down", "prefill_m256_down", SUITE),
        epoch="P",
        finding="run 16, section 35",
        built="both directions: direct_store (removing the reduction) and "
              "atomic_fixup (inlining it)",
        measured="direct_store 0.8267 suite / 0.234x on prefill_m256_down; "
                 "atomic_fixup 1.0533 suite vs incumbent 1.0970. Replicated "
                 "far more sharply as v78 (run 480): -43.6% geomean, every "
                 "split-K route between -63% and -132%",
        control="the incumbent, same round. v78 adds the stronger one: the two "
                "routes that never split K cannot execute a line of the new "
                "code and moved -0.80% / +0.29%, opposite directions, both "
                "inside drift",
        bound="on this kernel slice counts run 4-20, so any mutation making "
              "per-slice work more expensive loses -- closed in BOTH "
              "directions. v78 priced WHY, and it is not the dispatch: "
              "winner-take-all collapses reduction parallelism by a factor of "
              "`slices` and spends it at the tail with nothing to overlap, and "
              "the handoff fence is device-scope on an 8-XCD part, so every "
              "strided partial read is guaranteed to miss L2. The added time "
              "tracks the FP32 partial plane at ~20x what that traffic costs "
              "at HBM speed. Trading `slices`-way parallelism and L2 residency "
              "for one 4.7 us launch is a bad trade on this part",
        reopens_when="the slice count drops to ~1-2, which would make the "
                     "per-slice cost argument stop applying",
    ),
    Closure(
        axis="raising the split-K slice count to fill idle CUs",
        # This entry exists to close the LAST reopen path the N-strip entry
        # below left open. That entry closed the N-strip mechanism and
        # explicitly invited "a mechanism that raises CTA count WITHOUT cutting
        # the strip" -- split-K with reduction being the obvious one. It is now
        # measured, so the invitation is withdrawn on THIS kernel. Aliases name
        # the mechanism (more slices) plus the motivation words the N-strip
        # entry deliberately refused, because here the motivation IS the axis.
        aliases=("more slices", "raise the slice", "increase the slice",
                 "higher slice count", "more split-k", "deeper split-k",
                 "split-k to fill", "fill the idle cu", "fill idle cu",
                 "idle cus", "idle cu", "cu occupancy", "fill the device",
                 "raise cta count", "more ctas", "persistent cta",
                 "grid-stride", "grid stride"),
        routes=("prefill_m128_square", SUITE),
        epoch="R",
        finding="run 17, section 74",
        built="no source change at all -- v98's own GEAK_DEBUG_FORCE_SLICES "
              "hook, which overrides both the analytic plan and the tuner",
        measured="prefill_m128_square is the ONLY route in the suite that "
                 "leaves CUs idle: 64x128 tiles, 32x2 grid, 4 slices = 256 "
                 "CTAs on 304 CUs (0.84/CU). Forcing s=8 fills it to 512 CTAs "
                 "(1.68/CU, exactly what every other route runs at) and costs "
                 "+18%: s=4 gave 37.08/37.07/36.99 us, s=8 gave "
                 "43.46/43.99/43.81 us, ABBA, autotune off on both arms",
        control="the oracle in the same rotation held at 33.85-34.37 us across "
                "all six runs, so there is no block drift; and the arms differ "
                "in one integer through a hook that touches nothing else",
        bound="on this kernel every other route ALREADY runs at 1.54-2.26 "
              "CTAs/CU -- `plan_slices` targets cu_count + cu_count/2 = 456 "
              "CTAs by construction -- so there are no idle CUs anywhere else "
              "to fill, and on the one route that has them, filling them "
              "loses. The shipped autotuner independently reaches the same "
              "answer: its ladder is planned/2..planned*2 = 2..8, it evaluates "
              "s=8 cold with a 512MB flush mirroring the harness, and it keeps "
              "4. Anything arguing from an idle-CU count on this kernel is "
              "arguing from the dead tall/generic table (section 72.3), not "
              "from the code that runs",
        reopens_when="a mutation changes the tile so a route drops well below "
                     "one CTA per CU for a reason other than slice count, or "
                     "the part changes so the per-slice re-read of A and B "
                     "stops being the binding cost",
    ),
    Closure(
        axis="shortening the N-strip to raise CU occupancy",
        # Deliberately NOT "occupancy" / "idle cus" / "cta count" on their own.
        # Those words also describe split-K-with-reduction and cross-CTA LDS
        # sharing, which are this entry's own `reopens_when` -- an alias list
        # that flagged the reopen condition would be the self-sealing shape the
        # module docstring exists to prevent. The aliases name the MECHANISM
        # (shrinking the N strip), not the motivation (filling CUs).
        aliases=("n-strip", "n strip", "narrow the n", "shorten the n",
                 "waves per cta", "waves per block", "n_waves", "nwaves",
                 "fewer columns per", "adaptive strip", "strip width",
                 "shrink the tile in n", "smaller n tile"),
        routes=(SUITE,),
        epoch="R",
        finding="(67)",
        built="nwaves: the N-strip width templated on Waves, halved at launch "
              "while grid_x*grid_y stays under the CU count -- 6 of 11 routes "
              "change, the other 5 are a PROVABLE identity",
        measured="all six changed routes got readably SLOWER: waves->1 costs "
                 "+8.3%/+11.1%/+79.6%, waves->2 costs +21.2%/+21.7%/+17.8%. "
                 "The pre-registered direction (66.3) is refuted, not merely "
                 "unconfirmed. Suite geomean 0.8864x of D1b.",
        control="the 5 waves==4 routes run byte-identical code and moved at "
                "most 0.44% against a +-2.48% bound -- while being readably "
                "1.69-1.91x apart from base on those SAME routes, so the "
                "rotation is shown able to see a difference and seeing none",
        bound="closed as a loss in the opposite direction, not as a null. The "
              "N-strip is a REUSE axis, not just a parallelism axis: cutting "
              "waves 4->1 quadruples A-tile loads, and the penalty tracks the "
              "reuse lost (worst on decode_m32_down, the largest-K route), not "
              "the CUs gained",
        reopens_when="a mechanism appears that raises CTA count WITHOUT cutting "
                     "per-CTA N reuse -- split-K with reduction, or an A tile "
                     "resident in LDS and shared across CTAs. Distinct from the "
                     "active-CU closure above: that one closed for lack of "
                     "statistical power, this one closed on a measured reversal",
    ),
    Closure(
        axis="replacing the inner-loop global load's 64-bit VGPR address with "
             "a buffer_load against an SGPR resource descriptor",
        # Mechanism words only. NOT "register pressure" / "vgpr" / "agpr" on
        # their own: those also describe accumulator placement and tile-shape
        # changes, which are open. And NOT "occupancy" / "ctas per cu", which
        # is this entry's own `reopens_when` (a) -- flagging the reopen
        # condition is the self-sealing shape this module exists to prevent.
        aliases=("buffer_load", "buffer load", "make_buffer_rsrc",
                 "buffer_rsrc", "buffer resource", "resource descriptor",
                 "raw_buffer_load", "buffer addressing", "sgpr address",
                 "descriptor in sgpr", "v# descriptor", "lshl_add_u64",
                 "64-bit address arithmetic", "64 bit address arithmetic"),
        routes=(SUITE,),
        epoch="R",
        finding="(147)",
        built="exp/v98_bufload_20260817: make_buffer_rsrc built once in the "
              "kernel prologue from three scalars every lane agrees on, "
              "__builtin_amdgcn_raw_buffer_load_b128 in load_panel, the bounds "
              "predicate deliberately kept so the change isolates the "
              "addressing mode alone. All 17 templates change; zero spills in "
              "both builds",
        measured="suite geomean TIME 1.0100x, i.e. the variant is 1.00% "
                 "SLOWER. Separated on median +-2*raw-MAD: 3/11 slower "
                 "(decode_m2_square +1.6%, decode_m8_up +2.0%, "
                 "prefill_m1024_down +4.3%), 0/11 faster. Per-slot geomean "
                 "speedup 1.0791/1.0810 against base's 1.0890/1.0954",
        control="ABBA in a single lock on lane 3, each slot internally BCCB. "
                "The oracle arm's baseline_ms spread across the four slots is "
                "<=1.52% on all 11 cases with no directional drift, so the "
                "rotation is shown quiet enough to read a 1% suite move",
        bound="closed as a measured loss, not a null -- and closed AGAINST "
              "four static screens that all said yes: clean compile, the "
              "intended buffer_load_dwordx4 in the disassembly, -2 to -20 VGPR "
              "on 16/17 routes with zero spills, and one route's CTA/CU "
              "doubling. Two facts kill the mechanism. (i) The instruction "
              "count did not fall: v_lshl_add_u64 went to 0 in all 17 loops "
              "but the compiler refilled the slots, so loop length was flat or "
              "WORSE (m128 169->175, <96,128,2,2,128> 414->424). (ii) The one "
              "route whose CTA/CU doubled, <96,128,2,2,64>, is NOT REACHED by "
              "any of the 11 cases -- decode_m96_up takes the sk128 arm. The "
              "static loop metric is also sign-inverted on the worst "
              "regression: prefill_m1024_down runs <128,160,2,2,32>, the one "
              "route that got shorter (174->171, VALU 32->29)",
        reopens_when="the LDS bill per CTA falls far enough that residency "
                     "stops being pinned at one block per CU by the 50,688 B "
                     "panel, so freed registers have something to buy; or a "
                     "shape appears in the suite that actually takes the "
                     "<96,128,2,2,64> arm; or a formulation is found where the "
                     "measured loop length falls rather than merely trading "
                     "one address instruction for another",
    ),
    Closure(
        axis="register prefetch depth PF>1 on an already-resident tile",
        # No bare "PF" and no "prefetch depth": matching is substring, so the
        # first is a false-positive generator and the second belongs to the
        # finding-(73) entry, which must stay the single owner of that phrase
        # or one proposal flags two closures.
        aliases=("PF=2", "PF=4", "PF>1", "register prefetch",
                 "multi-buffer prefetch", "deepen prefetch", "raise PF",
                 "ra[PF][APT]", "rb[PF][BPT]"),
        routes=("decode_m2_square", "decode_m8_up", "decode_m16_square",
                "decode_m32_down", "decode_m64_square",
                "prefill_m1024_down", "prefill_m2048_square"),
        epoch="N,Q",   # N bounded the bk64 sweep; Q the BK=32 arm

        finding="(round 7 of this lane, PIPELINE_PROGRESS_GREEDY.md 3350-3379 "
                "item 4) and round 8 r1_d0, which pre-registered the "
                "no-gain prediction from mfma_feed_model.py and confirmed it",
        built="`macro_gemm_body_bk64`'s fixed-depth prefetch generalised to a "
              "template depth `int PF`, then swept PER TILE -- the sweep that "
              "shipped PF=4 on <128,64,2,2> also measured PF>1 on <32,64,2,2> "
              "and <64,64,2,2>",
        measured="PF=1 was the per-tile optimum on BOTH bk64 siblings: PF>1 "
                 "lost on the two tiles that already have LDS residency. The "
                 "single winner, PF=4 on 128x64_bk64, moved decode_m96_up "
                 "66.48 -> 60.34 us (+10.2%) and nothing else outside the "
                 "+/-2.5% band. Round 8 then closed the LAST untested "
                 "residency point, the BK=32 128x128 body at 18432 B = "
                 "3 CTA/CU, with PF=2 (the ceiling: PF=4 costs an occupancy "
                 "wave): prefill_m1024_down 300.0 -> 302.2 us = -0.72% "
                 "against a MEASURED 0.97% floor (no effect, sign unstable "
                 "between cycles) and prefill_m2048_square 187.3 -> 192.5 us "
                 "= -2.79% against a 0.72% floor (a REGRESSION, both cycles "
                 "agreeing in sign, Welch p ~ 0.008), replicated by a second "
                 "8-block full-suite experiment at -0.20% / -2.04%",
        control="the same sweep on the same body in the same epoch -- the three "
                "bk64 tiles differ only in BM, so tile residency is the sole "
                "surviving explanatory variable. For the BK=32 arm the control "
                "was the unpatched canonical tree BUILT IN THE SAME SESSION "
                "and interleaved PCCP+CPPC, plus a whole-tree resource diff "
                "showing exactly one changed line (128x128 VGPR 74 -> 86, "
                "TotalSGPR 42 -> 44) and zero sibling drift, and nine "
                "bit-identical routes of which eight moved inside +/-0.6%",
        bound="PF buys in-flight bytes per CU, so it pays only where residency "
              "is the thing starving them. Measured monotone in CTA/CU at "
              "LDS granularity 512 B: 128x64_bk64 = 26112 B = 2 CTA/CU WON; "
              "64x64_bk64 = 17408 B = 3 CTA/CU LOST; 32x64_bk64 = 13312 B = "
              "4 CTA/CU LOST; and now 128x128_bk32 = 18432 B = 3 CTA/CU "
              "LOST. Four residency points measured, no untested member left. "
              "The BK=32 arm also separates WHY: prefill_m2048_square is "
              "splits=1, 512 WGs, one full-machine wave with no second CTA "
              "per CU to absorb the longer software pipeline, and it loses; "
              "prefill_m1024_down has 3 split-K slices and absorbs the same "
              "change to within noise. Registers were never the binding term -- every "
              "one of these tiles has 17-42 spare VGPR+AGPR and holds "
              "occupancy through PF=4, so an occupancy screen says yes to all "
              "three and is NOT evidence of a prize",
        reopens_when="a variant raises a tile's LDS bill until it sits at or "
                     "below 2 CTA/CU, which is the only residency class PF has "
                     "ever won in; or PF is proposed for a COMPUTE-bound route "
                     "(arithmetic intensity above the ~247 flop/byte ridge), "
                     "where the mechanism under test is MFMA feeding rather "
                     "than in-flight bytes and this measurement does not "
                     "speak -- note that on THIS kernel the second condition "
                     "is unreachable: CTA-level AI is BM*BN/(BM+BN) = 64 at "
                     "128x128 and no tile inside the LDS budget passes 128, "
                     "so it can only be met by a different kernel",
    ),
    Closure(
        axis="raising the workgroup count / machine fill on "
             "prefill_m2048_square by moving the dispatch gate onto a "
             "narrower (BN=64) tile",
        # Mechanism words for "make more CTAs on this route". NOT bare "fill"
        # or "occupancy": those belong to other axes that are still open.
        aliases=("kWideTileGate", "wide tile gate", "widetilegate",
                 "BN=64 dispatch", "narrower tile for m2048",
                 "raise the workgroup count", "more workgroups",
                 "machine fill deficit", "fill deficit", "tail wave",
                 "wave quantisation", "wave quantization"),
        routes=("prefill_m2048_square",),
        epoch="Q",
        finding="(round 9 r2_d0 of this lane)",
        built="kWideTileGate 400 -> 512, one integer in both mirrored HIP "
              "twins, which routes prefill_m2048_square off "
              "macro_gemm_128x128_kernel onto the EXISTING "
              "macro_gemm_128x64_kernel. An offline GateChangeTest (12/12) "
              "proved the gate moves exactly one of the eleven cases, and a "
              "whole-tree resource diff over 14 kernels x 8 fields came back "
              "126 lines BYTE-IDENTICAL between the arms -- a pure dispatch "
              "change with zero kernel-body movement",
        measured="prefill_m2048_square 186.08 -> 232.25 us = +24.81% SLOWER "
                 "over 8 interleaved --case blocks (PCCP+CPPC, same-session "
                 "control BUILD, all_primed on all 12 timed runs). The arms do "
                 "not overlap at all and the effect is ~34x the measured "
                 "0.72% epoch-Q floor on this route; an independent 4-block "
                 "full-suite experiment replicates it at 1.2399",
        control="ten routes are bit-identical by construction; seven sat "
                "inside +/-0.8% and the three sub-20us decode routes drifted "
                "3-6% against 1.89-3.05% floors (decode_m2_square is the "
                "known-void route of finding (42)). No bit-identical route "
                "moved in a way that could manufacture a 25% effect",
        bound="the fill deficit is REAL and the patch does fix it -- 512 WGs "
              "into 912 slots (56%) becomes 1024 into 1216 (84%) -- and it is "
              "still a 25% loss. So on this route fill is NOT the binding "
              "constraint at 56% of a single wave; the 128x128 tile's "
              "arithmetic efficiency is. Halving BN costs roughly 2x what it "
              "costs on prefill_m1024_down because at M=2048 the A panel is "
              "re-read across 16 tile-rows, so the doubled A stream is far "
              "larger, on top of halving the accumulators per wave. "
              "kWideTileGate = 400 is now MEASURED optimal on its upper side "
              "rather than merely shape-fitted -- do not raise it again",
        reopens_when="a variant makes the BN=64 body's A stream cheap enough "
                     "that halving BN no longer doubles a re-read panel (an "
                     "LDS-resident or cross-CTA-shared A tile), or the route "
                     "stops being one full-machine wave for a reason other "
                     "than tile width",
    ),
    Closure(
        axis="LDS bank-conflict removal in the BK=32 macro bodies",
        aliases=("bank conflict", "bank conflicts", "lds bank",
                 "xor swizzle", "swizzled lds", "lds padding",
                 "pad the lds stride", "conflict-free lds"),
        routes=("prefill_m512_up", "prefill_m1024_down",
                "prefill_m2048_square", "prefill_m256_down"),
        epoch="Q",
        finding="(round 9 r2_d1 of this lane)",
        built="nothing was built, and that is the result: the mandatory "
              "diagnosis step measured the counter before the edit and the "
              "premise died there. Workspace sources verified byte-identical "
              "to CANONICAL afterwards",
        measured="SQ_LDS_BANK_CONFLICT is EXACTLY 0 on "
                 "macro_gemm_128x128_kernel (prefill_m1024_down, "
                 "prefill_m2048_square) and on macro_gemm_128x64_kernel "
                 "(prefill_m512_up)",
        control="a hand-written POSITIVE control in the same profiling "
                "session read 96.9% conflict, proving the counter live and "
                "the zero real rather than an unarmed counter",
        bound="there is no conflict traffic to remove, so any swizzle, pad or "
              "stride change on these bodies is bounded at zero by "
              "construction -- and kMacroLdsStride=36 is already the unique "
              "legal stride at BK=32",
        reopens_when="a variant changes the LDS layout or the fragment access "
                     "pattern of these bodies (a new stride, a different "
                     "ds_read width, a second buffer) and re-measures the "
                     "counter above zero on the CHANGED body",
    ),
    Closure(
        axis="direct-to-LDS async global loads (delete the ds_write half of "
             "each K stage)",
        aliases=("direct-to-lds", "direct to lds", "global_load_lds",
                 "global_load_lds_dword", "lds dma", "async copy",
                 "asynchronous copy", "cp.async", "buffer_load_lds",
                 "bypass registers", "skip ds_write", "delete ds_write"),
        routes=(SUITE,),
        epoch="T",
        finding="(51)/(52), round 10 r1_d0 of this lane",
        built="three compiler probes with the in-tree toolchain (AMD clang "
              "22.0.0git, roc-7.2.3, HIP 7.2.53211) plus a full disassembly "
              "census of both macro bodies -- no patch, no GPU seconds",
        measured="__builtin_amdgcn_global_load_lds(size=4) compiles for "
                 "gfx942 and emits s_mov_b32 m0 / global_load_lds_dword; "
                 "size=16 is a HARD COMPILE ERROR on gfx942 ('size must be "
                 "1, 2, or 4') and compiles clean on gfx950. So the DMA path "
                 "moves 4 B/lane where the incumbent's global_load_dwordx4 "
                 "already moves 16. Replacement inflates the 128-instruction "
                 "128x128 inner loop to >=136 and realistically >=144 (+12.5%) "
                 "to delete a ds_write surface of 4 instructions = 3.1%; the "
                 "128x64 body agrees in sign (74 -> 80, +8.1%)",
        control="the incumbent census itself is the control: PAIRED "
                "ds_write2_b64 rather than ds_write_b128 confirms the 72-byte "
                "as[BM][36]/bs[BN][36] row stride that a contiguous "
                "M0+lane*4 DMA write cannot skip",
        bound="bounded at zero by arithmetic, not by a noise floor: the "
              "mechanism must pay a 4x global-instruction inflation to buy "
              "back 3.1% of the loop. No stride, swizzle, stage-count or "
              "lane-mapping variant changes that ratio",
        reopens_when="the target is gfx950 hardware, where the 16-byte width "
                     "compiles today and the trade becomes 1-for-1, so the "
                     "staging store is free to remove; or a loop whose global "
                     "reads are already 4 bytes per lane, so the hardware path "
                     "costs no extra issue slots -- neither exists on this tree",
    ),
    Closure(
        axis="instruction-scheduling hints inside the BK=64 macro body "
             "(iglp_opt / sched_group_barrier)",
        aliases=("iglp_opt", "iglp", "sched_group_barrier", "sched_barrier",
                 "instruction scheduling", "scheduling hint",
                 "software pipelining hint", "interleave mfma and ds_read",
                 "reorder ds_read"),
        routes=("decode_m96_up", "prefill_m128_square", "decode_m64_square"),
        epoch="T",
        finding="(54)/(55), round 10 r1_d1 of this lane",
        built="all four __builtin_amdgcn_iglp_opt values screened offline, "
              "then iglp_opt(2) -- the only resource-free value -- timed in "
              "eight interleaved PCCP+CPPC arms against a same-session "
              "control BUILD",
        measured="-0.31% / -0.03% / +0.75% on decode_m96_up / "
                 "prefill_m128_square / decode_m64_square; every delta inside "
                 "its epoch-T floor and every candidate/control arm range "
                 "fully overlapping. Whole-tree resource diff over 14 kernels "
                 "x 8 fields: 0 changed fields for V=2, versus 3 changed "
                 "(VGPR 150->174) for V=0/V=1 and 12 extra s_waitcnt for V=1",
        control="the 11 kernels off this body were byte-identical for all "
                "three values, and the control arm was built in the same "
                "session that timed it",
        bound="bounded by CONSTRUCTION, not by one null measurement: "
              "SQ_WAIT_INST_LDS is 0.050 of wave cycles, so at most 5 of the "
              "35 exposed-dependency-wait points sit on the ds_read->mma "
              "chain a scheduling hint can reorder inside an s_barrier-bounded "
              "region. The other ~30 points are on the global-load -> LDS "
              "staging edge, fenced by __syncthreads() on BOTH sides and "
              "unreachable by any intra-region hint. sched_group_barrier was "
              "deliberately not also spent: it is the same 5%",
        reopens_when="the barrier STRUCTURE of macro_gemm_body_bk64 changes so "
                     "that the global->LDS staging edge falls inside a single "
                     "scheduling region -- at which point the ~30% becomes "
                     "reachable and the hints are worth re-screening",
    ),
    Closure(
        axis="four split-K slices",
        aliases=("4 slices", "four slices", "splits=4", "split count 4",
                 "slice count 4"),
        routes=("decode_m96_up", "prefill_m128_square"),
        epoch="T",
        finding="(57), round 10 r1_d2 of this lane, closing round 9's open "
                "question",
        built="an isolated same-binary --case sweep of splits 2/3/4/5/6 on "
              "decode_m96_up and of 4 vs the incumbent 8 on the 2 CTA/CU "
              "route prefill_m128_square",
        measured="decode_m96_up 72.89 / 60.34 / 68.26 / 60.97 / 64.84 us for "
                 "splits 2/3/4/5/6 -- four slices is 13.1% worse than three "
                 "and 12.0% worse than five DESPITE having FEWER workgroups "
                 "than five (688 vs 860). prefill_m128_square: 4 loses 1.3% "
                 "to the incumbent 8",
        control="same binary across all arms -- only the host-side split "
                "count moved, so no codegen difference can explain the sign",
        bound="runtime is NON-MONOTONE in workgroup count, so the "
              "exposed-fraction model is a diagnostic and not a cost function "
              "(third model in this lane to fail that way, cf. the per-CU AI "
              "model and the fill model, finding (45)). On the BK=64 routes "
              "the dominant term is SLICE-LENGTH QUANTISATION: counts that "
              "divide the stage count cost 34.15/33.71 us and ragged ones "
              "37.11/36.82 us, a 9-10% penalty at identical residency",
        reopens_when="a tile or K-stage change makes 4 divide the stage count "
                     "evenly on a route where the incumbent count does not",
    ),
    Closure(
        axis="cross-N-tile A-panel reuse (stage the A macro-panel in LDS once "
             "and sweep several N tiles per CTA) on the 128x128 BK=32 body",
        aliases=("a-panel reuse", "a panel reuse", "n-sweep", "n sweep",
                 "a macro-panel", "a macro panel", "sweep n tiles",
                 "n tiles per cta",
                 "sweep several n tiles", "multiple n tiles per cta",
                 "cross-n-tile", "cross n tile", "reuse the a tile",
                 "lds-resident a tile", "a tile resident in lds",
                 "cross-cta-shared a tile", "amortise the a panel"),
        routes=("prefill_m1024_down",),
        epoch="T",
        finding="(70)/(71), round 11 r2_d0 of this lane",
        built="two SOURCE PROBES that price the ceiling instead of building "
              "the sweep: `gr = row0 + r` -> `gr = r` collapses all 8 A "
              "tile-rows onto one 128-row slab (A footprint 22.5 MB -> 2.8 MB, "
              "permanently L2-resident) leaving instruction stream, vector "
              "width and trip count untouched; the symmetric edit on "
              "`gc = col0 + r` does the same for B. Each is a STRICT UPPER "
              "BOUND on what infinite reuse of that operand could buy -- "
              "strictly more than any S-way sweep can capture. Four "
              "interleaved arms, 3 isolated --case repeats each, C/A/B/AB/C "
              "with the second control a fresh REBUILD from restored canonical",
        measured="control pooled median 0.30458 ms [0.30232, 0.30684]; A-free "
                 "0.29530 = -3.05%; B-free 0.27942 = -8.26%; both-free "
                 "0.25316 = -16.9%. All four ranges mutually non-overlapping, "
                 "every delta an order of magnitude above the route's "
                 "a-fortiori epoch-T floor of 0.0043 ms. The WHOLE A "
                 "substream is worth 9.3 us of a 306 us call, so with the "
                 "route arithmetic (GEMM 270.4 + finalize 14.4 + residue "
                 "19.4) the ideal A-reuse ceiling is 261.1 us against a 245 "
                 "us target; an S-way sweep captures only (1 - 1/S) of it -- "
                 "4.6 us at S=2, 7.0 us at S=4 -- and misses by 16 us at its "
                 "unreachable limit BEFORE paying any cost",
        control="the B-side probe run in the same session on the same tree is "
                "the negative control that makes the A number readable: same "
                "edit shape, 2.7x the effect, so a null on A is not a dead "
                "probe. Workspace verified byte-identical to CANONICAL at "
                "start AND at end, and the verifier independently sha256'd "
                "all eight modifiable files against canonical",
        bound="bounded by ARITHMETIC above its own ceiling, and the costs are "
              "priced separately: grid(32,8,3) = 768 CTAs on 912 slots fills "
              "the machine in one round, so dividing it by S gives 1.26 "
              "CTA/CU at S=2 (ragged two-round tail) and 37% idle CUs at S=4; "
              "the barrier-efficient form holds S*64 AGPR live across the K "
              "loop plus S B-buffers in LDS (18432 -> 27648 B, residency "
              "3 -> 2 CTA/CU), which is arithmetically the 128x256 tile "
              "already measured at -39%; the sequential single-B-buffer form "
              "keeps the register cost and adds 2S barriers per K stage on a "
              "body whose exposed wait is already dominated by the "
              "barrier-fenced staging edge (54). NOTE the direction of the "
              "finding: the A panel is re-read 32x but is ALREADY L2-resident "
              "under xcd_remapped_grouped, so a byte-denominated forecast "
              "predicts ~25% traffic removed and a large win, while pricing "
              "it in microseconds closes it at 3%",
        reopens_when="the mechanism is aimed at the B stream instead (an "
                     "M-SWEEP: B panel staged once, swept over 2..4 of the 8 "
                     "M tiles, priced ceiling 12.6 us at S=2 / 18.9 us at "
                     "S=4) AND the same engineer also holds the split-K slice "
                     "count, so the grid division the sweep causes can be "
                     "paid back -- splitting the reuse lever and the slice "
                     "lever across two lanes makes any sweep axis unbuildable "
                     "by construction; or a tile change removes the "
                     "one-round-fill constraint that the division violates",
    ),
    Closure(
        axis="the fixed per-call cost outside the two kernels (host prefix, "
             "inter-dispatch gap, call-cost-aware split count)",
        aliases=("fixed per-call cost", "per-call overhead", "residue outside "
                 "the gemm", "dispatch residue", "host overhead",
                 "inter-dispatch gap", "launch gap", "cheaper finalize launch",
                 "overlap the finalize launch", "call-cost-aware split",
                 "call cost aware slice count", "splits=1",
                 "single dispatch to remove the launch"),
        routes=("decode_m96_up", "prefill_m1024_down", "prefill_m128_square",
                "decode_m64_square"),
        epoch="T",
        finding="(72)/(73)/(74), round 11 r2_d1 of this lane",
        built="a direct decomposition of the cold call on decode_m96_up: "
              "rocprofv3 warm kernel times, a direct dispatch-gap trace, a "
              "host-prefix probe against an empty-frame floor in two "
              "sessions, and a spin-injection sensitivity sweep; then a "
              "planner sweep forcing the split count on four routes",
        measured="the books close exactly: warm GEMM 49.02 + warm finalize "
                 "5.84 + inter-dispatch gap 0.00 (30/30 pairs at 0 ns) + host "
                 "prefix 0.00 (prefix probe == empty-frame floor, twice) + "
                 "un-overlapped event-frame cost 0.82 + cold-cache ramp 3.40 "
                 "= 59.08 us = full_cold as measured. CEILING ON ANY HOST "
                 "EDIT IS 0.82 us against a -3 us target. "
                 "prefill_m1024_down's 19.4 us residue is 100% cold-cache "
                 "(19.40 and 19.68 us in two sessions). Making the split "
                 "count call-cost aware moves ZERO routes -- the count that "
                 "minimises the GEMM already minimises the full call on all "
                 "four routes swept -- and splits=1, the only option that "
                 "actually removes the second dispatch, costs +35% "
                 "decode_m96_up, +123% prefill_m128_square, +55% "
                 "prefill_m1024_down, +75% decode_m64_square",
        control="forcing the planner's own default choice reproduces the "
                "default (decode_m96_up 57.68/59.28 default vs 59.16 at "
                "forced s=3), which is the sanity check that the forcing "
                "harness itself is inert; the empty-frame floor measured in "
                "two separate sessions is the negative control for the host "
                "prefix; correctness PASS with write_probe_coverage 11/11 "
                "conclusive and the tree byte-identical to canonical",
        bound="THREE CORRECTIONS THAT OUTLIVE THE DIRECTION. (1) The ~4.3 us "
              "empty-frame floor is NOT additive: with real work present only "
              "0.82 us of it survives, so the standing assumption that a "
              "fixed 4-5 us frame cost sits inside every scored number is "
              "wrong. (2) The 1:1 host-sensitivity finding stands and is "
              "sharper: the prime buffer covers only ~1 us, above which the "
              "slope is 1.0 (spin 2/3/4/6/8/12 us -> +1.92/+2.88/+3.48/+6.12/"
              "+7.80/+11.96) -- host time genuinely pays, there is simply "
              "0.82 us of it left. (3) The '5.5 us finalize floor' overstates "
              "the call's actual cost: with the gap at 0 ns, suppressing the "
              "finalize dispatch recovers only 3.0-5.6 us, less than the "
              "kernel's own standalone 5.84 us. Any future direction quoting "
              "a residue from a warm-profile-minus-cold-harness subtraction "
              "must FIRST subtract the cold-cache term -- 3-4 us on the "
              "decode routes, 19-20 us on prefill_m1024_down",
        reopens_when="the harness stops flushing caches between timed samples "
                     "(which is what the cold-cache ramp is), or a gap above "
                     "zero is measured between the two kernels on some route",
    ),
    Closure(
        axis="LDS-neutral half-stage double buffering of macro_gemm_body_bk64 "
             "(split the 26112 B BK=64 panel into two BK=32 slots)",
        aliases=("half-stage double buffering", "half stage double buffer",
                 "split the bk64 panel", "two bk=32 slots", "bk32 halves",
                 "lds-neutral double buffering", "remove the "
                 "write-after-read fence", "staging edge fence",
                 "double-sided syncthreads", "barrier structure of the bk64 "
                 "body", "overlap the global to lds staging"),
        routes=("decode_m96_up", "prefill_m128_square"),
        epoch="T",
        finding="(75)/(76), round 11 r2_d2 of this lane",
        built="three built and timed arms with all three properties the "
              "direction demanded PROVEN, not assumed: LDS bytes 26112 in "
              "every arm; residency held at 2 CTA/CU by -Rpass-analysis "
              "(control 50 SGPR/150 VGPR/32 AGPR/occ 2; arm1 52/150/56/occ 2; "
              "arm2-3 33/144/56/occ 2, scratch 0 spills 0); and the barrier "
              "count held at 2 per 64 K, counted in the ISA (8 s_barrier in "
              "the PF=4-unrolled 4-stage loop in control, arm1 and arm2 "
              "alike). Whole-tree diff over 14 kernels x 8 fields moved only "
              "the three _bk64 kernels and only in SGPR/VGPR/AGPR",
        measured="arm 3 (the load-map re-cut ALONE, parent barrier structure "
                 "untouched) +31.5% / +14.7% on decode_m96_up / "
                 "prefill_m128_square; arm 2 (re-cut + the half-stage "
                 "restructure, with the ISA census made IDENTICAL to the "
                 "parent and total issued actually LOWER at 470 vs 577) is "
                 "WORSE at +35.1% / +22.7%, so the restructure adds nothing "
                 "even with its instruction cost removed; arm 1 (parent map "
                 "kept, lane-predicated half-store) costs 5-8% because the "
                 "compiler realises the predicate as a second full copy of "
                 "the store, ds_write2_b64 24 -> 48 and loop 577 -> 760 "
                 "issued (+32%) in a body that is 41% ISSUE WAIT (54)",
        control="a same-session freshly-built control workspace with its own "
                ".torch_ext, interleaved P C C P / C P P C, isolated --case, "
                "all_primed; no arm's range overlaps its control's on either "
                "route and every delta is 4x-30x the epoch-T floor. "
                "Correctness PASS on every arm, 11/11 write probes ok, "
                "hip_twin_sync 2 pairs in lockstep before every build",
        bound="an ARITHMETIC obstruction, not a tuning failure, and the same "
              "class as the direct-to-LDS 4 B/lane closure (51). Splitting a "
              "BK=64 panel into BK=32 halves forces a mutually exclusive "
              "choice and both branches are priced: keep the parent's "
              "coalesced map (8 consecutive threads read one row's whole "
              "128 B chunk) and idx%KV == tid%KV pins each thread to ONE "
              "half, so the half-store must be lane-predicated and duplicates "
              "the store stream; or re-cut the map so a thread's registers "
              "are contiguous within one half, and then a k-half of a row is "
              "64 B not 128 B, halving per-instruction contiguity to 4 lanes "
              "x 64 B. A BK=64 row contributes 128 B per stage and half of it "
              "is 64 B; no stride, padding, register-array or store-ordering "
              "variant changes that ratio. WITH (55) AND (67) THIS CLOSES THE "
              "LAST LIVE MECHANISM on the ~30% of wave cycles that (54) "
              "localises behind the double-sided __syncthreads() on the "
              "global->LDS staging edge: hints were closed by construction, a "
              "second LDS buffer a fortiori, and the LDS-neutral form now by "
              "direct measurement with residency and barrier count held. "
              "Reusable control census, inner loop of "
              "macro_gemm_128x64_bk64_nt_kernel at PF=4 (4 stages = 256 K): "
              "577 issued, 111 blocks, 8 s_barrier, 24 ds_write2_b64, 48 "
              "ds_read2_b64, 24 global_load_dwordx4, 128 "
              "v_mfma_f32_16x16x16_bf16, 17 s_waitcnt",
        reopens_when="the body's global reads for ONE K-half are already a "
                     "full contiguous line (BK >= 128, so a half-row is still "
                     "128 B), or the target arch has a lane-predicated LDS "
                     "store that does not cost a duplicated instruction "
                     "stream",
    ),
    Closure(
        axis="B-side M-sweep on the 128x128 BK=32 body (stage the B "
             "macro-panel in LDS once and sweep S of the 8 M tiles per CTA)",
        aliases=("m-sweep", "m sweep", "sweep several m tiles",
                 "b-panel reuse", "b panel reuse", "stage the b macro-panel",
                 "cross-m-tile reuse", "sweep the m tiles", "multi-tile m",
                 "reuse b across m tiles"),
        routes=("prefill_m1024_down",),
        epoch="T",
        finding="(82), round 12 r3_d0 of this lane",
        built="nothing -- closed by the register-file arithmetic BEFORE any "
              "GPU second was spent, then corroborated against an already "
              "measured point. Occupancy on gfx942 is "
              "floor(512 / (roundup(VGPR,8) + roundup(AGPR,8))) per SIMD; the "
              "parent 128x128 body is 80 VGPR + 64 AGPR = 144 -> 3 waves/SIMD, "
              "confirmed by -Rpass-analysis=kernel-resource-usage",
        measured="S=2 doubles the per-wave accumulator bill to 128 AGPR, so "
                 "holding 3 waves/SIMD demands VGPR <= 40 against 74 in use "
                 "today with the 8 A fragments and the staging registers "
                 "still unpaid. The configuration this forces -- 128x128 at "
                 "2 CTA/CU -- is exactly what round 7 r4_d0 measured at -6%",
        control="the LDS constraint the direction imposed does NOT rescue it: "
                "B-only staging is 9216 B, still 2 CTA/CU, because the "
                "collapse is forced by the REGISTER FILE and not by LDS. Both "
                "escapes were priced too -- BM=256/BN=64 holds accumulators "
                "constant but doubles tiles_n, and the whole A substream is "
                "9.3 us against a 12.6 us prize (69)/(71); the 8-wave "
                "256x128 form gives 1 CTA/CU = 8 waves/CU against today's 12",
        bound="UNSATISFIABLE, not merely unprofitable: no S>=2 M-sweep on "
              "gfx942 can hold the parent's 3 CTA/CU at any LDS budget. This "
              "closes the successor that (70)/(71) named when it redirected "
              "the reuse axis from A to B -- the redirection was right about "
              "WHICH stream binds (B by 2.7x, worth ~25 us of the 306 us "
              "call) and wrong that a sweep is the way to take it. The prize "
              "is real and unreachable by this mechanism",
        reopens_when="an architecture whose per-SIMD register file exceeds "
                     "512 registers, or a formulation that raises B reuse "
                     "WITHOUT raising the per-wave accumulator count (the "
                     "accumulators, not the panel, are what pays)",
    ),
    Closure(
        axis="epilogue store width / LDS-staged CShuffle transposed write-out",
        aliases=("cshuffle", "c shuffle", "epilogue store width",
                 "widen the global store", "wide coalesced stores",
                 "transposed shuffle epilogue", "lds-staged epilogue",
                 "staged write-out", "epilogue panel", "write-out formulation",
                 "global_store_dwordx4 epilogue"),
        routes=("prefill_m2048_square", "prefill_m512_up"),
        epoch="T",
        finding="(84), round 12 r3_d1 of this lane",
        built="the mechanism REACHED THE MACHINE CODE and was independently "
              "confirmed there: isa_signals diff mechanism_realized=true for "
              "widen_global_store, max global store per lane 2 B -> 16 B on "
              "the bf16 direct-store path, +2 global_store_dwordx4, K-loop "
              "census byte-for-byte unchanged (d(mfma)=0, d(ds_read2_b64)=0, "
              "d(ds_write2_b64)=0, d(global_load_dwordx4)=0), LDS high-water "
              "unchanged at 18432 B because the epilogue panel is unioned "
              "onto the dead staging panel, occupancy unchanged at 3 CTA/CU",
        measured="TIME-NEUTRAL. Against an interleaved same-session control "
                 "arm the two routes the patch actually changes move -0.43% "
                 "and +0.14% (verifier) / -0.87% and -0.29% (a dedicated "
                 "5-pair rotated single-case block), every delta inside its "
                 "route's floor and with overlapping ranges; an earlier full "
                 "variant gave the OPPOSITE sign on the suite (0.9959). Suite "
                 "median 1.38098 candidate vs 1.37933 control = +0.12%",
        control="a freshly built unpatched-canonical arm in the same session, "
                "rotated against the candidate; 6 candidate repeats and 3 "
                "control repeats, all_primed true on all nine",
        bound="the epilogue is ~11% of the issued instruction stream on "
              "prefill_m2048_square and returns at most ~1% of CALL time: "
              "these bodies are not issue-limited in the write-out window, "
              "the per-element stores were ALREADY coalescing at the cache "
              "line, so widening the instruction does not change bytes moved. "
              "Two structural limits make the reachable surface two routes "
              "wide: gfx942 exposes NO packed f32 atomic add (settled by "
              "compile probe, not documentation -- unsafeAtomicAdd(float2*) "
              "is 'no matching function' and __hip_atomic_fetch_add on a "
              "2-wide float ext_vector is 'invalid type'), so split-K routes "
              "cannot use it at all; and non-exact-tiled routes such as "
              "decode_m96_up never fire the specialisation. TRANSFERABLE AND "
              "MORE VALUABLE THAN THE NULL: dead code is not free -- "
              "instantiating the wide path in the BK=64 bodies where it is "
              "provably unreachable grew macro_gemm_32x64_bk64_kernel 550 -> "
              "752 instructions and cost +6.18% on decode_m16_square with the "
              "EXECUTED path unchanged",
        reopens_when="an arch with a packed f32 atomic add (which would put "
                     "the split-K routes in reach), or a measurement showing "
                     "a macro body issue-limited in its epilogue window",
    ),
    Closure(
        axis="de-predication / exec-mask guard deletion in the BK=64 macro "
             "bodies",
        aliases=("de-predicate", "depredicate", "de-predication",
                 "guard deletion", "delete the guard", "remove the guard",
                 "exec mask", "s_and_saveexec", "predication",
                 "branchless staging", "__builtin_assume"),
        routes=("decode_m96_up", "decode_m16_square", SUITE),
        epoch="V",
        finding="(99), round 13 r1_d0 of this lane",
        built="the EXACT de-predicated arm ported to all three BK=64 macro "
              "bodies, plus a second arm (B) that adds __builtin_assume and "
              "KEEPS the guards. ISA gate passed wide on arm A: "
              "128x64_bk64_nt K-loop 577 -> 335 issued, s_and_saveexec "
              "64 -> 0; 32x64_bk64 70 -> 43, 6 -> 0; 64x64_bk64 112 -> 58, "
              "12 -> 0; barriers, ds_write2/ds_read2, global_load_dwordx4 "
              "and mfma counts unchanged; occupancy of every arm equal to "
              "its parent's; 15 pre-existing kernels instruction-for-"
              "instruction identical",
        measured="SLOWER on both decode routes in all three rounds, "
                 "non-overlapping with the same-session control. The "
                 "regression tracks GUARD DELETION, not the assume: arm A "
                 "(zero exec regions) drives loop s_waitcnt UP 17 -> 22 and "
                 "5 -> 7; arm B (62-67% of exec regions gone, guards kept) "
                 "holds s_waitcnt flat or lower (17 -> 17, 5 -> 5, 9 -> 5) "
                 "and is measurement-neutral",
        control="a same-session control BUILD, interleaved isolated --case "
                "blocks; prefill_m128_square is insensitive in BOTH "
                "directions (A -0.3%, B -2.4%)",
        bound="240 saved issue slots are worth less than the added waits on "
              "these split-K memory-latency-bound routes (occ 2-5). The "
              "exec-masked staging region is a SCHEDULING BLOCK BOUNDARY "
              "keeping the global load separated from the LDS write; delete "
              "it and the two merge. Consequence for the issue-count axis "
              "generally: it is BODY-SPECIFIC, positive on the BK=32 128x128 "
              "body and negative here, so any future issue-count direction "
              "must name WHICH body and cite a measurement on THAT body",
        reopens_when="a formulation that PRESERVES the staging region's "
                     "scheduling block boundary (arm B, which keeps the "
                     "guards and only adds the compiler hint, is the neutral "
                     "form to build on), or a body where deleting them is "
                     "measured NOT to raise the loop's s_waitcnt count",
    ),
    Closure(
        axis="MFMA-register-native (unpadded) LDS staging layout for the "
             "BK=32 macro bodies",
        aliases=("mfma-register-native", "register-native lds",
                 "unpadded lds", "lds padding", "padded lds", "stride 36",
                 "kmacroldsstride", "fragment index", "fragment layout",
                 "ds_read2st64", "lds staging layout", "restage the lds"),
        routes=("prefill_m1024_down", "prefill_m2048_square", SUITE),
        epoch="V",
        finding="(101), round 13 r1_d2 of this lane",
        built="the unpadded MFMA-fragment-native LDS layout on "
              "macro_gemm_128x128_exact: LDS 18432 -> 16384 B, "
              "ds_read2_b64 -> ds_read2st64_b64, no stride multiply",
        measured="INSTRUCTION PRIZE 0 and OCCUPANCY PRIZE 0. One k0 "
                 "iteration is 77 issued on both sides with a "
                 "byte-identical histogram (32 v_mfma, 8 ds_read2, "
                 "4 ds_write2, 4 global_load_dwordx4, 8 v_lshl_add_u64, "
                 "7 s_waitcnt, 4 v_mov, 2 s_barrier); the sole difference is "
                 "the ds_read2 opcode. Occupancy 3 waves/SIMD both sides -- "
                 "that body is REGISTER-limited (88 VGPR + 64 AGPR = 152), "
                 "so the LDS saving buys nothing. Suite A-B-A 1.38890 / "
                 "1.39218 / 1.37870, candidate mean 0.9940 of parent with "
                 "the A1-A2 candidate spread (0.74%) EXCEEDING the "
                 "candidate-parent gap",
        control="the two interleaved A arms of the same A-B-A, rebuilt in "
                "session; smaller BK=32 tiles DID move on LDS "
                "(128x64 4 -> 5 CTA/CU, 32x128 5 -> 6) and the suite still "
                "measured null",
        bound="the brief's premise is FALSE IN THE COMPILED CODE: the "
              "parent's stride-36 row-major address is entirely "
              "loop-invariant (one hoisted base VGPR per operand) and the "
              "i/sk parts are immediates that already fit ds_read2_b64's "
              "8-byte-granular 255-slot field. The stride-36 layout costs "
              "nothing in the loop, which also closes the 'fragment index "
              "arithmetic' motivation for these bodies",
        reopens_when="the 128x128 body stops being register-limited (its "
                     "VGPR+AGPR total falls below the 3-wave boundary) so an "
                     "LDS saving can buy a wave; or a body whose staged "
                     "address is measured NOT to be loop-invariant",
    ),
    Closure(
        axis="slice-length quantisation as a PREDICTOR of the best split "
             "count",
        aliases=("slice-length quantisation", "slice length quantisation",
                 "slice quantisation", "divide the stage count",
                 "stage-aligned slice", "evenly divide k", "ragged slice",
                 "tail quantisation"),
        routes=("decode_m96_up",),
        epoch="V",
        finding="(97), round 13 r1_d1 of this lane",
        built="a same-session isolated --case sweep of splits "
              "2/3/4/5/6/8 at fixed BM=96, only the host-side split count "
              "moving",
        measured="the rule predicts 4 (64 stages / 4 = 16/16/16/16, exact) "
                 "and 4 is the SLOWEST point of the sweep: 0.069685 ms "
                 "against the RAGGED winner 3 (22/22/20) at 0.056645 / "
                 "0.057065 ms, +23%. Full sweep (ms): 2 = 0.067663, "
                 "3 = 0.056645, 4 = 0.069685, 5 = 0.065058, 6 = 0.065078, "
                 "8 = 0.071347",
        control="same binary across all arms; the 3-vs-4 sign reproduces the "
                "independently recorded (57) result on the older BM=128 tile",
        bound="the per-slice fp32 atomic image is 4.23 MB each, 20.8% of "
              "the route's 61.1 MB of DRAM traffic. NOTE: the earlier "
              "gloss that this image is therefore the BINDING term was "
              "arithmetic, not a measurement, and round 14 refuted it -- "
              "an oracle that drove the plane's 12.68 MB of DRAM writes "
              "to ~0 moved the route 0.0%. Only the split-count verdict "
              "here is measured. (57) stands only on the "
              "routes it was measured on -- do not carry it to a route "
              "without re-measuring, and do not re-sweep the split count on "
              "decode_m96_up, where 3 is pinned and measured",
        reopens_when="a route whose per-slice fp32 image is small relative "
                     "to its operand traffic, where the tail term can again "
                     "bind; or a tile/K-stage change on a route whose "
                     "incumbent count already divides the stages evenly",
    ),
    Closure(
        axis="split-K fp32 partial-plane DRAM locality (XCD-affine slice "
             "mapping, tile-major workspace, non-temporal finalize)",
        aliases=("split-k workspace", "splitk workspace", "partial plane",
                 "fp32 partial", "workspace footprint", "workspace locality",
                 "tile-major workspace", "xcd-affine", "xcd affine slice",
                 "slice-to-xcd", "affine slice mapping", "workspace layout",
                 "atomic workspace", "non-temporal finalize",
                 "nontemporal finalize", "shrink the workspace",
                 "delete the fp32 plane"),
        routes=("decode_m96_up", "decode_m8_up", "prefill_m1024_down"),
        epoch="V",
        finding="(109)-(112), round 14 r2_d0 of this lane",
        built="three arms. (1) the mechanism itself: a host-side XCD-aligned "
              "gridDim.x pad plus a workgroup-uniform col0>=n exit, making a "
              "tile's three partial streams land on one XCD's L2 (verified by "
              "an s_getreg HW_REG_XCC_ID probe: xcc_id == lin%8 for 516/516 "
              "and 528/528 blocks). (2) ORACLE A, a throwaway incorrect build "
              "aliasing every epilogue atomic into a 1 MB L2-resident window "
              "(& 0x3FFFF), driving the plane's DRAM writes to ~0 -- a strict "
              "upper bound on ANY workspace-locality mechanism. (3) ORACLE B, "
              "the finalize's two NT zero-restore stores deleted",
        measured="(1) null: decode_m96_up +0.3% against a 3.2% floor, "
                 "decode_m8_up -1.2% against a 4.6% floor, and rocprofv3 --pmc "
                 "over 30 dispatches is byte-identical to the pre-patch "
                 "reading (RDREQ_DRAM 756705, WRREQ_DRAM 198144, L2 hit "
                 "51.98%). (2) ORACLE A is 0.0% on decode_m96_up and -0.8% on "
                 "prefill_m1024_down, inside its 4.0% floor -- the plane's "
                 "DRAM bytes are NOT the binding term on either route. "
                 "(3) ORACLE B is -2.1% / -2.0%, i.e. even a FREE zero-restore "
                 "is below both routes' floors",
        control="same-session interleaved isolated --case blocks, 3 per arm, "
                "against a control BUILD; suite diagnostic 1.39098 sits inside "
                "the 2.8% session drift of (103)",
        bound="bounded at ZERO by an oracle, not by a floor: the whole "
              "workspace object -- footprint, layout, locality, slice-to-XCD "
              "mapping, and the finalize's read side -- is worth 0.0% on "
              "decode_m96_up and <=2% (below floor) on prefill_m1024_down. "
              "Sub-lever bookkeeping: non-temporal finalize was ALREADY in the "
              "tree (kFinalizeStreamGate), and a tile-major workspace cannot "
              "reduce the request count because there is ZERO write "
              "amplification already (198144 x 64 B = 12.68 MB = exactly "
              "3 x 4.23 MB; BN=64 is 256 B and n*4 = 44032 B is 128 B-aligned, "
              "so a tile row is two whole cache lines)",
        reopens_when="a route where the fp32 reduction image is a "
                     "materially larger share of DRAM traffic than the "
                     "20.8% measured here AND whose operand read stream "
                     "has been shown not to bind; or an epilogue that "
                     "eliminates the intermediate reduction buffer "
                     "altogether rather than relocating it -- but note "
                     "the winner-take-all form of that is separately "
                     "closed at -43.6%",
    ),
    Closure(
        axis="issue-count / ILP levers on macro_gemm_128x128_exact "
             "(MFMA reordering, longer dependency chains, 32-bit global "
             "offsets)",
        aliases=("mfma reorder", "reorder the mfma", "lengthen the "
                 "dependency chain", "ilp on 128x128", "accumulator chains",
                 "v_lshl_add_u64", "lshl_add_u64", "32-bit offset",
                 "saddr form", "host-side size guard for 32-bit offsets",
                 "issue count on the exact body", "mfma issue bound",
                 "wider bf16 mfma", "16x16x32", "full-rate bf16 opcode"),
        routes=("prefill_m1024_down", "prefill_m2048_square"),
        epoch="V",
        finding="(113)-(116), round 14 r2_d1 of this lane",
        built="no timed arm -- the regime was NAMED from counters first and "
              "both permitted levers then failed their own preconditions. "
              "SQ_WAVE_CYCLES on gfx942 partitions EXACTLY (100.0%, two "
              "independent routes) into SQ_ACTIVE_INST_ANY + SQ_WAIT_INST_ANY "
              "+ SQ_WAIT_ANY, which is a usable three-way regime test on a box "
              "with no rocprof-compute",
        measured="the body is NEITHER MFMA-issue-limited (it runs 2.23x / "
                 "2.24x above its MFMA-issue lower bound; MFMA busy ~45% even "
                 "on the maximally-loaded CUs after correcting the clock) NOR "
                 "DRAM-bandwidth-limited (807 GB/s = 15% of the measured HBM "
                 "roof, and (38) already measured a 36% traffic cut at +6.8% "
                 "SLOWER). It is LATENCY EXPOSURE AT 2-3 WAVES/SIMD: 84% "
                 "(m1024) / 79% (m2048) of every wave-cycle is spent not "
                 "issuing. Reordering has zero headroom in the COMPILED code "
                 "-- the post-barrier region is 29 consecutive v_mfma with "
                 "nothing between them and the 16 accumulator tiles already "
                 "have a 256-pipe-cycle reuse distance against a 16-cycle "
                 "latency. The 8 v_lshl_add_u64 are 10.4% of issued and issue "
                 "is only 16.1% of wave-cycles, so the whole (101b) prize is "
                 "1.7% of wave-cycles against a 4.0% route floor",
        control="the clock correction is itself the control: the earlier "
                "'26-42% MFMA busy' figure was a device average priced at the "
                "nominal 2.1 GHz, while GRBM_COUNT / 8 XCD / duration gives a "
                "MEASURED 1.478 GHz (m1024) / 1.521 GHz (m2048)",
        bound="bounded by arithmetic below the route floor, not by a null "
              "measurement. Also closed by ISA fact: "
              "v_mfma_f32_16x16x16_bf16 costs EXACTLY 16.000 SIMD-cycles here "
              "(SQ_VALU_MFMA_BUSY_CYCLES / SQ_INSTS_MFMA on both routes) and "
              "the full-rate CDNA4 opcode v_mfma_f32_16x16x32_bf16 does not "
              "exist on gfx942 (gfx950-insts only), so 16x16x16 IS the "
              "full-rate bf16 shape and no opcode swap can halve the floor",
        reopens_when="the body's wave count per SIMD rises above 3 (a "
                     "register-budget reduction from 152 to <=128 buys a 4th "
                     "wave) or its barrier structure is decoupled "
                     "(ping-pong / 8-wave interleave), because both change "
                     "the regime this closure is conditioned on; or a route "
                     "whose grid supplies well above 3 CTA/CU",
    ),
    Closure(
        axis="loop issue-count / exec-mask collapse / statically-countable "
             "global->LDS staging on macro_gemm_body_bk64 (the BK=64 macro "
             "bodies)",
        aliases=("countable staging", "statically countable", "staggered "
                 "vmcnt", "relaxed vmcnt", "vmcnt(n)", "waitcnt drains",
                 "collapse the exec regions", "exec mask on bk64",
                 "saveexec on bk64", "issue count on the bk64 body",
                 "loop instruction count", "mfma density", "unconditional "
                 "tail prefetch", "clamped prefetch", "prefetch clamping"),
        routes=("decode_m96_up",),
        epoch="V",
        finding="(122)-(126), round 15 r3_d0 of this lane",
        built="four arms on macro_gemm_96x64_bk64_nt, all measured against a "
              "same-session interleaved control. arm4 removes essentially the "
              "whole object the direction named: loop 479 -> 259 instructions "
              "(-46%), s_and_saveexec_b64 48 -> 0, s_waitcnt vmcnt(0) 18 -> 1, "
              "MFMA density 20.0% -> 37.1%, VGPR 152 -> 148 (BELOW the "
              "parent), LDS 21760 B and scratch/spill 0 on every arm",
        measured="0.0% on decode_m96_up. The mechanism WAS realised and the "
                 "aimed-at counter DID move -- SQ_ACTIVE_INST_ANY 6,963,387 "
                 "-> 4,547,869 (-34.7%, matching the instruction cut) at "
                 "identical SQ_INSTS_MFMA -- but the freed issue slots were "
                 "re-absorbed as instruction wait: SQ_WAIT_INST_ANY 26.91% -> "
                 "48.62%, SQ_WAIT_ANY 53.81% -> 35.35%. The (114) partition on "
                 "the parent is exact to 100.0%: 19.27% issuing / 26.91% "
                 "instruction wait / 53.81% other wait at a MEASURED 1.806 GHz",
        control="same-session interleaved control on decode_m96_up; the "
                "parent's 479/48/18 census was reproduced on two independent "
                "code objects after the build cache was moved aside",
        bound="bounded a priori BELOW the direction's own target by the (114) "
              "partition: price an issue-count edit as (target instructions / "
              "loop instructions) x SQ_ACTIVE_INST_ANY share before building "
              "-- 0.301 x 0.1927 = 5.8% for the stated mechanism and 0.46 x "
              "0.1927 = 8.9% for the total cut actually achieved, and both are "
              "UPPER bounds that assume issue is the critical path when 80.7% "
              "of wave-cycles say it is not. Two sub-levers separately priced: "
              "collapsing exec regions WITHOUT replacing the scheduling "
              "boundary costs 2.6% (all 20 load destinations hoisted into one "
              "live range, VGPR 152 -> 160, 40 accvgpr moves and 6 s_nop "
              "appear in a loop that had none) and a single "
              "__builtin_amdgcn_sched_barrier(0) at the seam restores it at "
              "zero instruction cost; a clamped unconditional tail prefetch "
              "costs 5.0% because PF*BK = 256 K-elements is 18-20% of this "
              "route's split-K slice (kchunk 1408 = 22 stages, last slice 20)",
        reopens_when="the route's SQ_ACTIVE_INST_ANY share rises well above "
                     "the 19.27% measured here (i.e. the body actually becomes "
                     "issue-limited), or a formulation whose priced upper "
                     "bound (target instrs / loop instrs x issue share) "
                     "exceeds the route floor with margin. Note also the "
                     "ANTI-SIGNAL recorded in `measured`: the two arms whose "
                     "disassembly showed the hoped-for partial-drain waits "
                     "(6 and 8 of them) are strictly SLOWER than the arm with "
                     "none, so that ISA property must never again be read as "
                     "evidence of a win on a wave that is 80.7% idle",
    ),
    Closure(
        axis="machine fill on prefill_m2048_square / split-K on the BN=128 "
             "wide-tile body / raising CTA/CU by adding split slices on "
             "macro_gemm_128x128_exact",
        aliases=("fill the machine", "machine fill", "1024 workgroup rule",
                 "more ctas", "raise cta/cu", "cta per cu on m2048",
                 "split-k on m2048", "split k on the wide tile",
                 "split the wide tile", "add split slices", "more slices",
                 "512 workgroups on 304 cus", "grid construction gap",
                 "workgroup count on prefill_m2048_square"),
        routes=("prefill_m2048_square",),
        epoch="V",
        finding="(127)-(129), round 15 r3_d2 of this lane",
        built="three oracles on one binary, 18 interleaved isolated --case "
              "blocks. ORACLE 1: the minimum split (s=2) on "
              "prefill_m2048_square, which is the CHEAPEST member of the "
              "family and therefore bounds all split counts. ORACLE 2: the "
              "finalize measured rather than estimated. ORACLE 3, the "
              "decisive one: at a CONSTANT 512-workgroup grid, change ONLY "
              "the output path from the direct bf16 store to the fp32 atomic "
              "workspace, then separately go 512 -> 1024 workgroups",
        measured="ORACLE 1: 0.26597 ms vs a 0.19183 ms same-session control = "
                 "-38.7% on the GEMM side ALONE, before a microsecond of "
                 "finalize, against a required +13-15% and a 4.9% route floor. "
                 "ORACLE 2: the 8.4 M-element finalize costs 25.1 us = 13.1% "
                 "of the 192 us call. ORACLE 3: the output-path change alone "
                 "costs +38.3% at CONSTANT grid, and DOUBLING the grid on top "
                 "of it then costs a further +0.4% -- so the machine-fill "
                 "hypothesis measures 0.0%",
        control="same binary across all arms, same-session interleaved "
                "isolated --case blocks; the 1.68 CTA/CU figure that motivated "
                "the direction is the thing ORACLE 3 refutes",
        bound="bounded at ZERO by ORACLE 3, independently of the reduction "
              "cost: prefill_m2048_square is not short of resident work in any "
              "way more CTAs can fix. The entire cost of split-K here is the "
              "reduction path (33.5 MB of global_atomic_add_f32 replacing "
              "16.8 MB of plain bf16 stores, ~70 us on a 192 us kernel) plus "
              "25 us of finalize. This also EXPLAINS the two previously-closed "
              "fill mechanisms: kWideTileGate 400 -> 512 (+24.8%) and raising "
              "an existing split count (+18%) lost not merely because of their "
              "side costs but because the fill they bought was worth nothing. "
              "Premise correction: prefill_m2048_square is NOT the only "
              "splits=1 route -- prefill_m512_up also runs splits=1 (688 tiles "
              "> kSplitTileGate, grid (172,4,1), 2.26 CTA/CU); m2048 is the "
              "lowest-CTA/CU non-split route and the only splits=1 route on "
              "the BN=128 body",
        reopens_when="a mechanism that raises resident WAVES without adding "
                     "CTAs and without touching the output path -- that is the "
                     "8-wave re-wave, which WON here at -4.7% (see the "
                     "separate entry); or a gfx generation with a packed f32 "
                     "atomic add making the reduction path cheap enough that "
                     "split-K's -38.7% GEMM-side cost is no longer the whole "
                     "story",
    ),
    Closure(
        axis="8-wave (512-thread) re-wave of macro_gemm_128x128_exact on "
             "grids that already supply >= ~2.5 CTA/CU",
        aliases=("8 wave interleave", "eight waves", "512 threads",
                 "re-wave the tile", "double the waves", "raise waves per "
                 "simd on m1024", "occupancy on prefill_m1024_down",
                 "4th wave on the exact body"),
        routes=("prefill_m1024_down",),
        epoch="V",
        finding="(130), round 15 r3_d1 of this lane -- the LOSING half of the "
                "round winner",
        built="macro_gemm_128x128_exact_w8_kernel (WAVES_M=2, WAVES_N=4, "
              "64x32 per-wave block, 512 threads, 26 SGPR / 78 VGPR / 0 AGPR / "
              "0 spill / 18432 B LDS, occupancy 6 waves/SIMD against 3, "
              "residency 3 CTA/CU x 8 waves = 24 waves/CU against 12). Whole-"
              "tree -Rpass-analysis diff: the w8 kernel is the ONLY line that "
              "differs; all 17 pre-existing kernels byte-identical",
        measured="+9.2% SLOWER on prefill_m1024_down (2.53 CTA/CU), more than "
                 "twice its 4.0% floor, with occupancy doubled. The SAME body "
                 "is -4.7% to -8.0% on prefill_m2048_square (1.68 CTA/CU) and "
                 "-2.4% (inside floor) on prefill_m512_up (1.13 CTA/CU). The "
                 "ordering of the three routes is exactly their CTA/CU",
        control="an unpatched same-session control BUILD from CANONICAL, "
                "interleaved A A A C A A C C A; the 17 unchanged kernels are a "
                "clean negative control by construction",
        bound="the tax is arithmetic and unconditional: at FM x FN = 4 x 2 a "
              "wave reads 6 fragments per 16-deep K step to issue 8 MFMA, "
              "against 8 reads for 16 MFMA at 4 waves -- 1.5x the LDS read "
              "traffic per MFMA. The wave-count benefit is collected only "
              "where the grid is short of waves, so the break-even on this "
              "suite sits between 1.68 and 2.53 CTA/CU",
        reopens_when="a formulation that raises waves/SIMD WITHOUT raising "
                     "LDS reads per MFMA -- i.e. the untried sub-lever A, a "
                     "register-budget cut 152 -> <=128 on the existing "
                     "4-wave body, whose ceiling is 4 waves/SIMD; or a route "
                     "whose grid drops below ~1.7 CTA/CU, where this body is "
                     "already measured a WIN and needs no new experiment, "
                     "only a dispatch predicate",
    ),
)


def _norm(text: str) -> str:
    return re.sub(r"[\s_\-]+", " ", text.lower())


def matches(text: str) -> list[Closure]:
    """Every closure whose vocabulary appears in `text`.

    Substring matching on normalised text, deliberately. A planner writes
    `rasterization: grouped_m -> xcd_remapped_grouped`, not a tidy axis name,
    and a matcher that only accepted the tidy name would be silent exactly when
    it mattered -- which is finding (128) in another costume.
    """
    hay = _norm(text)
    hits = []
    for closure in CLOSED:
        if any(_norm(alias) in hay for alias in closure.aliases):
            hits.append(closure)
    return hits


def relevant(closure: Closure, route: str | None) -> bool:
    """Whether this closure was measured on the route being proposed."""
    if route is None or SUITE in closure.routes:
        return True
    return route in closure.routes


def check(proposals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Flag proposals naming a mechanism already measured shut.

    A proposal carrying a non-empty `reopen_justification` is reported but not
    flagged: the registry aims work, it does not forbid it, and the epoch that
    justified a closure is not always the epoch running the proposal.
    """
    flagged, justified = [], []
    for i, proposal in enumerate(proposals):
        text = " ".join(str(proposal.get(k, "")) for k in
                        ("axis", "mechanism", "direction", "description",
                         "rationale", "title", "name", "why", "prompt"))
        route = proposal.get("route") or proposal.get("context") or proposal.get("test_case_id")
        for closure in matches(text):
            if not relevant(closure, route):
                continue
            record = {
                "proposal_index": i,
                "route": route,
                "axis": closure.axis,
                "finding": closure.finding,
                "measured": closure.measured,
                "bound": closure.bound,
                "epoch": closure.epoch,
                "reopens_when": closure.reopens_when,
            }
            if str(proposal.get("reopen_justification") or "").strip():
                record["reopen_justification"] = proposal["reopen_justification"]
                justified.append(record)
            else:
                flagged.append(record)
    return {"flagged": flagged, "justified": justified,
            "proposals_checked": len(proposals)}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--list", action="store_true", help="print the registry as JSON")
    p.add_argument("--proposals", help="JSON file: a list of proposal objects, "
                                       "or '-' for stdin")
    args = p.parse_args(argv)

    if args.list:
        print(json.dumps([c.as_dict() for c in CLOSED], indent=2, sort_keys=True))
        return 0

    if not args.proposals:
        p.error("one of --list or --proposals is required")

    raw = sys.stdin.read() if args.proposals == "-" else Path(args.proposals).read_text()
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        proposals = parsed
    else:
        # The TechLead emits `candidate_directions` in plan_seed and `directions`
        # in plan_round. Accepting only one of them would make the checker
        # silently pass an empty list on the other phase, which is the worst
        # possible failure for a tool whose job is to notice things.
        proposals = (parsed.get("candidate_directions")
                     or parsed.get("directions") or [])
    verdict = check(proposals)

    for record in verdict["justified"]:
        print(f"REOPENED: proposal {record['proposal_index']} names "
              f"{record['axis']} (closed by {record['finding']}) and states a "
              f"reopen justification: {record['reopen_justification']}",
              file=sys.stderr)
    for record in verdict["flagged"]:
        print(f"ALREADY MEASURED SHUT: proposal {record['proposal_index']} "
              f"({record['route'] or 'no route named'}) names {record['axis']}. "
              f"{record['finding']}: {record['measured']}. Bound: "
              f"{record['bound']}. Reopens when: {record['reopens_when']}.",
              file=sys.stderr)

    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 6 if verdict["flagged"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
