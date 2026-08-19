#!/usr/bin/env python3
"""Does a bigger macro-tile still fill the machine on the routes it targets?

tile_design_space.py answers "which tile feeds MFMA best per CU" and picks
128x256 / 256x256. That model has no notion of HOW MANY workgroups the launch
produces, and on these two routes that is not a detail:

    workgroups = ceil(N/BN) * ceil(M/BM) * splits

Doubling BN halves the tile count. Split-K is the mechanism that normally
refills the machine, but it is capped by a fixup-traffic budget
(kSplitFixupBytes / (M*N*4)) that gets TIGHTER as the output grows -- and these
are the two largest outputs in the suite. So exactly where the AI argument is
strongest, the refill mechanism is weakest.

This reproduces the shipped dispatcher's own arithmetic (the gates and caps are
copied from best/src/custom_gemm.hip, and a test asserts the copies still match
that file) and reports occupancy of the MACHINE rather than of a CU.

Read together with tile_design_space.py: that one says a bigger tile needs less
bandwidth per CU, this one says whether there are enough CUs doing work for
that to matter.
"""

from __future__ import annotations

import argparse

CUS = 304

# --- copied from best/src/custom_gemm.hip; test_tile_fill_model.py checks ----
K_FILL_TARGET = 896
K_SPLIT_FIXUP_BYTES = 48 << 20
K_MAX_SPLITS = 24
K_SPLIT_TILE_GATE = 304
K_NARROW_TILE_GATE = 128
K_WIDE_TILE_GATE = 400
K_WORKSPACE_BYTES = 64 << 20

# The two routes round 9 targets, and the only two the AI argument is about.
ROUTES = {
    "prefill_m1024_down": (1024, 4096, 11008),
    "prefill_m2048_square": (2048, 4096, 4096),
}


def splits_for(m: int, n: int, k: int, tiles: int, bk: int) -> int:
    """The shipped dispatcher's split-K choice, reproduced exactly."""
    out_bytes = m * n * 4
    if not (tiles <= K_SPLIT_TILE_GATE and out_bytes <= K_WORKSPACE_BYTES):
        return 1
    splits = (K_FILL_TARGET + tiles // 2) // tiles      # NEAREST, not ceil
    max_by_bytes = K_SPLIT_FIXUP_BYTES // (out_bytes or 1)
    splits = min(splits, max_by_bytes)
    min_k = 256 if m <= 32 else 512
    splits = min(splits, k // min_k, K_MAX_SPLITS)
    splits = max(splits, 1)
    # The launcher then re-derives splits from a whole number of BK stages.
    if splits > 1:
        stages = k // bk
        kchunk = -(-stages // splits) * bk
        splits = -(-k // kchunk)
    return max(splits, 1)


def fill(m: int, n: int, k: int, bm: int, bn: int, bk: int, cta_per_cu: int
         ) -> dict[str, float]:
    tiles_n = -(-n // bn)
    tiles_m = -(-m // bm)
    tiles = tiles_n * tiles_m
    splits = splits_for(m, n, k, tiles, bk)
    wgs = tiles * splits
    capacity = CUS * cta_per_cu
    # Blocks land on CUs in waves of `capacity`. A partial last wave is the
    # cost that a raw wgs/CUs ratio hides.
    waves = -(-wgs // capacity)
    utilisation = wgs / (waves * capacity)
    return {"tiles": tiles, "splits": splits, "wgs": wgs,
            "capacity": capacity, "wgs_per_cu": wgs / CUS,
            "waves": waves, "tail_utilisation": utilisation,
            "cus_idle_in_tail": max(0, waves * capacity - wgs)}


# (label, bm, bn, bk, cta_per_cu) -- cta_per_cu MEASURED by the hipcc probe.
CANDIDATES = [
    ("128x128 t256 (shipped)", 128, 128, 32, 3),
    ("128x256 t512", 128, 256, 32, 2),
    ("256x256 t512", 256, 256, 32, 1),
    ("256x128 t512", 256, 128, 32, 2),
    # The opposite direction. Lower AI, but more tiles -- and the shipped
    # config is itself under-filled on both routes, which is the finding that
    # sent the search this way.
    ("128x64 BK=32 t256", 128, 64, 32, 4),
    ("128x64 BK=64 t256", 128, 64, 64, 2),
    ("64x128 t256", 64, 128, 32, 4),
    ("64x64 BK=64 t256", 64, 64, 64, 3),
]


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    for route, (m, n, k) in ROUTES.items():
        print(f"== {route}: M={m} N={n} K={k} ==")
        print(f"{'tile':<24}{'tiles':>7}{'splits':>8}{'WGs':>7}{'cap':>6}"
              f"{'WG/CU':>7}{'waves':>7}{'tail%':>7}{'idle':>6}")
        for label, bm, bn, bk, cta in CANDIDATES:
            f = fill(m, n, k, bm, bn, bk, cta)
            print(f"{label:<24}{f['tiles']:>7}{f['splits']:>8}{f['wgs']:>7}"
                  f"{f['capacity']:>6}{f['wgs_per_cu']:>7.2f}{f['waves']:>7}"
                  f"{100 * f['tail_utilisation']:>7.0f}"
                  f"{f['cus_idle_in_tail']:>6}")
        print()
    print("How to read this: 'cap' is CUs x measured CTA/CU, i.e. how many")
    print("workgroups the machine holds at once. 'waves' is how many times the")
    print("machine has to be refilled, and 'tail%' is how full the LAST wave is.")
    print("A tile that raises AI but drops the launch to a single partly-filled")
    print("wave has bought arithmetic intensity with idle CUs, which is not a")
    print("trade the per-CU model can see.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
