#!/usr/bin/env python3
"""Is the 128x128 BK=32 inner loop starved for in-flight loads, or already fed?

Round 8's whole bet is that adding a second prefetch buffer (PF=2) improves
MFMA feeding on the two compute-bound routes. The ISA probe established WHAT
the patch does -- in-flight dwordx4 per wave goes 4 -> 8, barrier density
unchanged -- but not whether the extra in-flight bytes BUY anything. The
progress log recorded that as unanswerable offline.

Most of it is answerable offline. Little's law needs only three things, and we
have all three without a GPU:

  * tile geometry            (from the dispatch model, already derived)
  * peak issue and bandwidth (published MI300X figures)
  * in-flight loads per wave (counted in the ISA: 4 at PF=1, 8 at PF=2)

What stays unanswerable is the achieved cache hit rate, so this model reports a
BAND across the plausible range rather than one number, and states which end of
the band would have to hold for PF=2 to pay.

This is a model, not a measurement. It is written down BEFORE the paired timing
so that round 8 can refute it. A prediction made after the fact is not one.
"""

from __future__ import annotations

import argparse

# --- MI300X, published ------------------------------------------------------
CUS = 304
CLOCK_HZ = 2.1e9           # peak engine clock
PEAK_BF16_FLOPS = 1307.4e12  # dense, no sparsity
HBM_BYTES_PER_S = 5.3e12

# Latency in cycles. Vendor does not publish these; these are the usual
# measured ranges on CDNA3 and the model is reported across both ends.
LAT_L2_CYCLES = 500
LAT_HBM_CYCLES = 1200

# --- the tile under test ----------------------------------------------------
BM, BN, BK = 128, 128, 32
THREADS = 256
WAVE = 64
CTA_PER_CU = 3             # 18432 B LDS/CTA -> floor(65536/18432) = 3
BYTES_PER_DWORDX4_PER_WAVE = 16 * WAVE   # 16 B/lane


def flop_per_cycle_per_cu() -> float:
    return PEAK_BF16_FLOPS / (CUS * CLOCK_HZ)


def hbm_bytes_per_cycle_per_cu() -> float:
    return HBM_BYTES_PER_S / (CUS * CLOCK_HZ)


def stage() -> dict[str, float]:
    """Per K-stage, per CTA: the work and the traffic it implies."""
    flop = 2 * BM * BN * BK
    load_bytes = (BM * BK + BN * BK) * 2      # A and B tiles, bf16
    return {"flop": flop, "load_bytes": load_bytes,
            "cta_ai": flop / load_bytes}


def demand() -> dict[str, float]:
    """Bytes per cycle per CU the loop would need to run AT MFMA peak."""
    s = stage()
    fpc = flop_per_cycle_per_cu()
    cycles_per_cta_stage = s["flop"] / fpc          # if one CTA owned the CU
    cu_cycles = CTA_PER_CU * cycles_per_cta_stage   # 3 CTAs share it
    cu_bytes = CTA_PER_CU * s["load_bytes"]
    return {"cu_cycles_per_stage": cu_cycles,
            "cu_bytes_per_stage": cu_bytes,
            "bytes_per_cycle": cu_bytes / cu_cycles}


def inflight_bytes_per_cu(loads_per_wave: int) -> float:
    waves_per_cu = CTA_PER_CU * (THREADS // WAVE)
    return waves_per_cu * loads_per_wave * BYTES_PER_DWORDX4_PER_WAVE


def required_inflight(bytes_per_cycle: float, latency: float) -> float:
    """Little's law: to sustain B bytes/cycle with L cycles of latency you must
    keep B*L bytes outstanding."""
    return bytes_per_cycle * latency


def report() -> int:
    s, d = stage(), demand()
    fpc = flop_per_cycle_per_cu()
    hbm = hbm_bytes_per_cycle_per_cu()

    print("== per-CU peak ==")
    print(f"  bf16 flop/cycle/CU        : {fpc:.0f}")
    print(f"  HBM bytes/cycle/CU        : {hbm:.2f}")
    print()
    print(f"== one K-stage of a {BM}x{BN} BK={BK} CTA ==")
    print(f"  flop                      : {s['flop']:,.0f}")
    print(f"  bytes loaded (A+B)        : {s['load_bytes']:,.0f}")
    print(f"  CTA-level AI              : {s['cta_ai']:.0f} flop/byte")
    print()
    print(f"== to run at MFMA peak with {CTA_PER_CU} CTA/CU ==")
    print(f"  cycles/CU per stage       : {d['cu_cycles_per_stage']:.0f}")
    print(f"  bytes/CU per stage        : {d['cu_bytes_per_stage']:,.0f}")
    print(f"  DEMAND bytes/cycle/CU     : {d['bytes_per_cycle']:.1f}")
    print(f"  HBM alone supplies        : {hbm:.2f}  "
          f"({100 * hbm / d['bytes_per_cycle']:.0f}% of demand)")
    print()
    print("  -> These routes CANNOT reach MFMA peak on HBM alone. The CTA-level")
    print(f"     AI is only {s['cta_ai']:.0f} flop/byte, far under the ~247 ridge; the")
    print("     suite-level AI of 762/1024 is a statement about total DRAM")
    print("     traffic AFTER cache reuse, not about what the loop asks for.")
    print(f"     Closing the gap needs a cache hit rate of at least "
          f"{100 * (1 - hbm / d['bytes_per_cycle']):.0f}%.")
    print()

    print("== Little's law: is PF=1 already covering the latency? ==")
    print(f"  {'PF':>3} {'loads/wave':>11} {'in-flight B/CU':>15}  verdict vs need")
    need = {name: required_inflight(d["bytes_per_cycle"], lat)
            for name, lat in (("L2", LAT_L2_CYCLES), ("HBM", LAT_HBM_CYCLES))}
    for name, lat in (("L2", LAT_L2_CYCLES), ("HBM", LAT_HBM_CYCLES)):
        print(f"  need at {name:>3} latency ({lat:>4} cyc): {need[name]:>9,.0f} B/CU")
    print()
    for pf, loads in ((1, 4), (2, 8)):
        have = inflight_bytes_per_cu(loads)
        marks = " ".join(
            f"{name}:{'OK' if have >= need[name] else 'SHORT'}"
            f"({have / need[name]:.1f}x)" for name in ("L2", "HBM"))
        print(f"  {pf:>3} {loads:>11} {have:>15,.0f}  {marks}")
    print()

    have1 = inflight_bytes_per_cu(4)
    covered = all(have1 >= v for v in need.values())
    if covered:
        print("  PREDICTION (pre-registered, refutable by round 8's paired timing):")
        print("  PF=1 already keeps more bytes outstanding than Little's law")
        print("  requires at BOTH latency ends, so the extra in-flight bytes PF=2")
        print("  buys are slack on a constraint that is not binding. Expect NO")
        print("  significant gain on prefill_m1024_down / prefill_m2048_square.")
        print()
        print("  What would make this model wrong, and is worth watching for:")
        print("   * loads are not evenly spread -- a burst at the top of each")
        print("     stage drains, then MFMA runs dry before the next barrier;")
        print("     average in-flight bytes would then overstate the useful ones")
        print("   * the 2 barriers per stage serialise the wave's own loads with")
        print("     its own MFMA, which Little's law over a whole stage hides")
        print("   * MFMA issue and VMEM issue contend for the same wave slot")
        print("  All three are timing effects, and all three would show up as")
        print("  PF=2 winning anyway. That is the value of writing this down.")
    else:
        print("  PREDICTION: PF=1 is short of the Little's-law requirement, so")
        print("  PF=2 has a binding constraint to relieve. Expect a real gain.")
    return 0


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    raise SystemExit(report())
