#!/usr/bin/env python3
"""Which macro tile could feed MFMA better than 128x128 BK=32, and at what cost?

The feed model (mfma_feed_model.py) showed that the two compute-bound routes
run a loop whose own arithmetic intensity is only

    CTA_AI = BM*BN / (BM+BN)        flop/byte

-- 64 at 128x128, far under the ~247 ridge -- and that the bandwidth the loop
would need to hold MFMA peak follows directly:

    demand = PEAK_FLOP_PER_CYCLE_PER_CU / CTA_AI      bytes/cycle/CU

Note what is NOT in that expression: CTA/CU, BK, and PF all cancel. The demand
per CU is a property of the tile's two edge lengths and nothing else. So more
prefetch depth and a deeper K-step cannot move it; only a larger, squarer tile
can. This enumerates that space against the constraints that actually bind.

The three constraints, all checkable without a GPU:

  LDS       ceil512((BM+BN) * (BK+PAD) * 2) <= 65536, and CTA/CU is the floor
            of that division. This is what killed 256x256 at first glance.
  registers accumulators are AGPRs and their count per lane is exact:
            AGPR = BM*BN / threads. gfx942 gives 512 unified VGPR+AGPR per
            lane, so waves/SIMD <= floor(512 / (VGPR + AGPR)).
  waves     32 wave slots per CU (8 per SIMD).

Occupancy is the MINIMUM of the LDS-limited and register-limited figures, and
the interesting tiles are the ones that raise CTA_AI without falling under
2 waves/SIMD -- below that there is nothing left to hide latency with.

VGPR is the one number not derivable here: it depends on addressing and
staging code the compiler emits. Rather than guess it, this reports the
occupancy as a function of a VGPR estimate and marks which candidates are
sensitive to it. The shortlist then gets compiled for real -- the same
discipline the PF sweep used, where the probe's PF=1 point reproduced the
incumbent's measured 74/64/0/3 exactly and that is what made it trustworthy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

PEAK_FLOP_PER_CYCLE_PER_CU = 2048.0
HBM_BYTES_PER_CYCLE_PER_CU = 8.30
LDS_BYTES_PER_CU = 65536
LDS_GRANULARITY = 512
REGS_PER_LANE = 512
# The 512 is a UNIFIED VGPR+AGPR budget, but neither half may exceed 256 on
# its own. 256x256 with 256 threads needs exactly 256 accumulators per lane and
# so sits precisely on that ceiling -- legal, but with nothing left over.
MAX_AGPR_PER_LANE = 256
WAVE_SLOTS_PER_CU = 32
WAVE = 64
RIDGE_AI = 247.0

# kMacroLdsStride: BK + 4 elements of padding, in the lane's own source.
PAD = 4


# Measured by the offline hipcc probe (tile_probe_suffix.hip appended to the
# shipped best/src/custom_gemm.hip, -Rpass-analysis=kernel-resource-usage,
# --offload-arch=gfx942). Keyed (BM, BN, BK, threads) -> (vgpr, agpr, scratch,
# compiler_occupancy, lds).
#
# The control row is the point of the table: probe_ctrl_128x128_w2x2_t256
# reproduced the shipped macro_gemm_128x128_kernel to the register
# (74/64/0/3/18432), so the other rows describe our kernel.
#
# Two things the analytic model got wrong, both worth keeping visible:
#  * AGPR/VGPR SPLIT. At 512 threads the compiler put every accumulator in
#    VGPRs and used no AGPRs at all. The model's per-lane accumulator COUNT was
#    right (256x256/512 = 128); which register file holds them was not the
#    model's to predict. Only the sum constrains occupancy, and there the
#    estimate was near-exact: predicted 128+86 = 214, measured 212.
#  * LDS ROUNDING. The compiler reports the DECLARED size (34560 for 224x256),
#    not the 512-granular allocation the model computes (34816). Both land on
#    the same CTA/CU, so this is a reporting difference, not a disagreement.
MEASURED: dict[tuple[int, int, int, int], tuple[int, int, int, int, int]] = {
    (128, 128, 32, 256): (74, 64, 0, 3, 18432),   # control == shipped kernel
    (256, 256, 32, 512): (212, 0, 0, 2, 36864),
    (224, 256, 32, 512): (180, 0, 0, 2, 34560),
    (256, 224, 32, 512): (180, 0, 0, 2, 34560),
    (128, 256, 32, 512): (126, 0, 0, 4, 27648),
    (192, 192, 32, 512): (148, 0, 0, 3, 27648),
}


def ceil512(n: int) -> int:
    return -(-n // LDS_GRANULARITY) * LDS_GRANULARITY


def cta_ai(bm: int, bn: int) -> float:
    return bm * bn / (bm + bn)


def demand_bytes_per_cycle(bm: int, bn: int) -> float:
    return PEAK_FLOP_PER_CYCLE_PER_CU / cta_ai(bm, bn)


def lds_bytes(bm: int, bn: int, bk: int) -> int:
    return ceil512((bm + bn) * (bk + PAD) * 2)


@dataclass(frozen=True)
class Tile:
    bm: int
    bn: int
    bk: int
    threads: int
    vgpr: int

    @property
    def waves_per_cta(self) -> int:
        return self.threads // WAVE

    @property
    def agpr(self) -> int:
        """Accumulators per lane. Exact: every output element is one fp32."""
        return self.bm * self.bn // self.threads

    @property
    def lds(self) -> int:
        return lds_bytes(self.bm, self.bn, self.bk)

    @property
    def cta_per_cu_lds(self) -> int:
        return LDS_BYTES_PER_CU // self.lds

    @property
    def waves_per_simd_regs(self) -> int:
        return min(8, REGS_PER_LANE // (self.vgpr + self.agpr))

    @property
    def cta_per_cu(self) -> int:
        """Resident CTAs, with the CTA treated as INDIVISIBLE.

        The trap this closes: at 1 CTA/CU a 1024-thread CTA is 16 waves, and if
        registers only support 12 resident waves you do not get 12 -- you get
        ZERO, because a block cannot be partially resident. Taking
        min(lds_waves, reg_waves) reports a plausible occupancy for a launch
        that cannot happen. 256x256 with 1024 threads lands exactly here:
        AGPR 64 + VGPR 86 = 150 per lane allows 3 waves/SIMD = 12 waves/CU,
        one short of the 16 the block needs.
        """
        by_regs_waves = self.waves_per_simd_regs * 4
        by_regs_ctas = by_regs_waves // self.waves_per_cta
        return min(self.cta_per_cu_lds, by_regs_ctas, WAVE_SLOTS_PER_CU // self.waves_per_cta)

    @property
    def waves_per_cu(self) -> int:
        return self.cta_per_cu * self.waves_per_cta

    @property
    def waves_per_simd(self) -> float:
        return self.waves_per_cu / 4

    @property
    def ai(self) -> float:
        return cta_ai(self.bm, self.bn)

    @property
    def demand(self) -> float:
        return demand_bytes_per_cycle(self.bm, self.bn)

    @property
    def hit_rate_needed(self) -> float:
        """Fraction of loop traffic that must come from cache, not HBM, for
        the tile to hold MFMA peak."""
        return max(0.0, 1.0 - HBM_BYTES_PER_CYCLE_PER_CU / self.demand)

    @property
    def vgpr_headroom(self) -> int:
        """How much the compiler's VGPR count may exceed the estimate before
        the tile loses a resident CTA.

        At 1 CTA/CU this is not a performance cliff, it is a LAUNCH cliff: one
        register over and the block has nowhere to live. The compile probe's
        job is to check the measured VGPR against this number, not to admire
        the occupancy column.
        """
        need_waves_per_simd = -(-self.waves_per_cta * self.cta_per_cu // 4)
        if need_waves_per_simd <= 0:
            return 0
        return REGS_PER_LANE // need_waves_per_simd - self.agpr - self.vgpr

    @property
    def feasible(self) -> bool:
        return (self.lds <= LDS_BYTES_PER_CU
                and self.agpr <= MAX_AGPR_PER_LANE
                and self.agpr + self.vgpr <= REGS_PER_LANE
                and self.waves_per_cu >= 4)


def enumerate_tiles(vgpr: int, bks=(32, 64)) -> list[Tile]:
    out = []
    for bm in (128, 160, 192, 224, 256):
        for bn in (128, 160, 192, 224, 256):
            for bk in bks:
                for threads in (256, 512, 1024):
                    t = Tile(bm, bn, bk, threads, vgpr)
                    # A wave must own at least one 32x32 chunk of the tile.
                    if t.bm * t.bn < t.threads * 16:
                        continue
                    if t.feasible:
                        out.append(t)
    return out


def best_per_shape(tiles: list[Tile]) -> list[Tile]:
    """One row per (BM,BN,BK): the thread count with the most waves/CU, and
    among ties the smallest, since more threads costs launch width."""
    best: dict[tuple[int, int, int], Tile] = {}
    for t in tiles:
        key = (t.bm, t.bn, t.bk)
        cur = best.get(key)
        if cur is None or (t.waves_per_cu, -t.threads) > (cur.waves_per_cu, -cur.threads):
            best[key] = t
    return sorted(best.values(), key=lambda t: (-t.ai, t.lds))


def effective_from_measured(bm: int, bn: int, bk: int, threads: int
                            ) -> dict[str, float]:
    """Re-derive residency from the MEASURED register total.

    The compiler's own "Occupancy [waves/SIMD]" column is a per-wave,
    register-limited figure. It is not CTA-aware and it does not know the LDS
    budget, so it can overstate: 192x192 at 512 threads reports 3, but the
    block is 8 waves and only one of them fits, which is 2 waves/SIMD in
    practice. Trusting that column directly is the same partial-residency error
    the analytic model had before `cta_per_cu` was fixed.
    """
    vgpr, agpr, _scratch, compiler_occ, _lds = MEASURED[(bm, bn, bk, threads)]
    t = Tile(bm, bn, bk, threads, vgpr + agpr)
    # Tile() adds agpr itself, so hand it the total as "vgpr" and zero it out.
    t = Tile(bm, bn, bk, threads, vgpr + agpr - t.agpr)
    return {"total_regs": vgpr + agpr, "compiler_occ": compiler_occ,
            "cta_per_cu": t.cta_per_cu, "waves_per_simd": t.waves_per_simd,
            "ai": t.ai, "demand": t.demand,
            "hit_rate_needed": t.hit_rate_needed}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vgpr", type=int, default=86,
                    help="VGPR estimate per lane (default: the measured PF=2 "
                         "128x128 figure, the most pessimistic real datapoint)")
    ap.add_argument("--all", action="store_true", help="do not collapse to best thread count")
    args = ap.parse_args()

    incumbent = Tile(128, 128, 32, 256, args.vgpr)
    print(f"incumbent 128x128 BK=32 thr=256: AI {incumbent.ai:.0f}, demand "
          f"{incumbent.demand:.0f} B/cyc/CU, LDS {incumbent.lds}, "
          f"{incumbent.cta_per_cu} CTA/CU, {incumbent.waves_per_simd:.0f} waves/SIMD, "
          f"needs {100 * incumbent.hit_rate_needed:.0f}% hit rate")
    print(f"(ridge AI is {RIDGE_AI:.0f}; a tile reaching it would need no cache at all)")
    print()

    tiles = enumerate_tiles(args.vgpr)
    rows = tiles if args.all else best_per_shape(tiles)
    print(f"{'BM':>4}{'BN':>5}{'BK':>4}{'thr':>6}{'AI':>7}{'demand':>8}"
          f"{'LDS':>7}{'CTA':>6}{'AGPR':>6}{'w/SIMD':>8}{'hit%':>6}  vs incumbent")
    for t in rows:
        gain = t.ai / incumbent.ai
        note = ""
        if t.waves_per_simd < 2:
            note = "TOO THIN"
        elif t.ai > incumbent.ai and t.waves_per_simd >= 2:
            note = f"AI x{gain:.2f}"
        elif t.ai <= incumbent.ai:
            note = "no AI gain"
        cta = (f"{t.cta_per_cu}" if t.cta_per_cu == t.cta_per_cu_lds
               else f"{t.cta_per_cu}/{t.cta_per_cu_lds}")
        print(f"{t.bm:>4}{t.bn:>5}{t.bk:>4}{t.threads:>6}{t.ai:>7.1f}"
              f"{t.demand:>8.1f}{t.lds:>7}{cta:>6}{t.agpr:>6}"
              f"{t.waves_per_simd:>8.1f}{100 * t.hit_rate_needed:>6.0f}  {note}")

    print()
    winners = [t for t in rows if t.ai > incumbent.ai and t.waves_per_simd >= 2]
    if not winners:
        print("Nothing raises CTA_AI while keeping >= 2 waves/SIMD.")
        return 0
    print("Shortlist -- raises AI and keeps >= 2 waves/SIMD:")
    for t in sorted(winners, key=lambda t: -t.ai)[:5]:
        print(f"  {t.bm}x{t.bn} BK={t.bk} thr={t.threads}: AI {t.ai:.1f} "
              f"(x{t.ai / incumbent.ai:.2f}), demand {t.demand:.1f} B/cyc "
              f"(vs {incumbent.demand:.0f}), {t.waves_per_simd:.0f} waves/SIMD "
              f"(vs {incumbent.waves_per_simd:.0f}), hit rate {100 * t.hit_rate_needed:.0f}% "
              f"(vs {100 * incumbent.hit_rate_needed:.0f}%)")
        cliff = ("LAUNCH FAILS" if t.cta_per_cu == 1 else "loses a CTA")
        print(f"      VGPR headroom {t.vgpr_headroom:+d} over the {t.vgpr} "
              f"estimate -> {cliff} beyond {t.vgpr + t.vgpr_headroom}")
    print()
    print("These are ANALYTIC. AGPR is exact and LDS is exact; VGPR is the "
          "estimate. Compile the shortlist before believing the occupancy "
          "column -- and reproduce a known tile's measured numbers in the same "
          "probe, or the probe is not measuring the kernel you think it is.")
    print()
    print("== MEASURED (offline hipcc probe, gfx942) ==")
    inc_m = effective_from_measured(128, 128, 32, 256)
    print(f"{'tile':<22}{'regs':>6}{'cc-occ':>8}{'CTA':>5}{'w/SIMD':>8}"
          f"{'AI':>7}{'demand':>8}{'hit%':>6}")
    for (bm, bn, bk, thr) in MEASURED:
        e = effective_from_measured(bm, bn, bk, thr)
        tag = f"{bm}x{bn} BK={bk} t{thr}"
        note = ""
        if e["cta_per_cu"] * (thr // WAVE) / 4 < e["compiler_occ"]:
            note = "  <- cc-occ overstates (block indivisible)"
        if (bm, bn, thr) == (128, 128, 256):
            note = "  <- CONTROL, == shipped kernel"
        print(f"{tag:<22}{e['total_regs']:>6}{e['compiler_occ']:>8}"
              f"{e['cta_per_cu']:>5}{e['waves_per_simd']:>8.1f}{e['ai']:>7.1f}"
              f"{e['demand']:>8.1f}{100 * e['hit_rate_needed']:>6.0f}{note}")
    print()
    print("The probe changes the ranking. 128x256 at 512 threads is the only")
    print("candidate that improves BOTH axes at once: AI 85.3 (x1.33) AND")
    print(f"4 waves/SIMD against the incumbent's {inc_m['waves_per_simd']:.0f}, because at")
    print("126 registers two 8-wave blocks still fit and LDS 27648 allows two.")
    print("Ranking by AI alone buried it under 256x256. Nothing spilled and no")
    print("kernel used scratch, so the launch cliff the analytic pass warned")
    print("about did not bind -- 256x256 came in at 212 registers, under the")
    print("256 that 2 waves/SIMD allows.")
    print()
    print("Two risks the arithmetic cannot see, both specific to 1 CTA/CU:")
    print(" * with one block per CU there is no second CTA to cover this one's")
    print("   barriers. The incumbent's 3 CTA/CU hides barrier stalls by")
    print("   construction; every shortlist entry gives that up.")
    print(" * halving demand only helps if the loop was bandwidth-limited. The")
    print("   feed model says PF=1 already satisfies Little's law, so if round 8")
    print("   also shows no gain, both results point at issue/barrier")
    print("   serialisation -- and a bigger tile will not fix that either.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
