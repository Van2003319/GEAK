#!/usr/bin/env python3
"""Compute a kernel's per-shape launch facts from its source, not from memory.

Finding 144. Twice in round 17 a number from the QD descriptor's ledger was
about to be carried into a sentence about the fused BF16 GEMM kernel -- once a
ROUNDS LAW claim, once a whole planned experiment (D2, autotune decision
variance) aimed at a tuner that this kernel does not have. The ledger is
accurate and it is authoritative, and that is precisely what makes it dangerous:
it describes a *different kernel family*, one that tiles more coarsely and has a
tuning ladder, and the more trusted a source is the less often anyone re-checks
which object it is about.

The mechanizable half of that lesson is not "detect when someone cites the wrong
ledger" -- the ledger's per-route numbers live in prose, and a checker that
parsed prose to reach a verdict would be guessing (141). It is the other half:
make the subject's own numbers cheap to compute and always current, so there is
no occasion to reach for remembered ones. Everything here is derived from the
kernel source at call time. Nothing is hardcoded, which is the point.

The parsing reuses `lds_cast_alignment`'s declaration machinery -- already
tested, and already fail-loud on constants it cannot resolve rather than
guessing a value.

TWO WARNINGS, because this table invites two wrong readings.

1. `rounds` here is NOT the ROUNDS LAW (24). That law is stated over the QD
   lineage's `RouteFacts`, whose kernels tile far more coarsely; every shipped
   route in *that* descriptor sits at rounds 1. A number out of this table does
   not belong in a sentence about that law.

2. `occ%` is NOT recoverable time. Finding (23) measured exactly that
   reciprocal, twice, and both LOST: prefill_m512_up utilisation 0.754 -> 0.905
   cost 21.4%; prefill_m256_down 0.684 -> 0.912 cost 11.3%. Those CUs were not
   throughput-bound. Note the regime (23)'s control actually covers -- routes
   already touching every CU, 2 CTAs per CU versus 3 -- which is not the regime
   in the `CUs%` column, where a decode shape leaves most of the device with no
   work at all. That is a reason the question may still be open. It is not
   evidence that the answer is yes, and any experiment owes a control that can
   tell the two regimes apart.

    kernel_launch_facts.py path/to/kernel.hip
    kernel_launch_facts.py path/to/kernel.hip --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lds_cast_alignment import DECL, ELEM_SIZE, Unresolved, constants, dims  # noqa: E402

# gfx942 / MI300X.
CUS = 304
LDS_PER_CU = 64 * 1024

# The eleven harness cases, exact IDs. Kept as the suite defines them; a
# self-invented bucket ("M<64") is not a route.
CASES = [
    ("decode_m2_square", 2, 4096, 4096),
    ("decode_m8_up", 8, 11008, 4096),
    ("decode_m16_square", 16, 4096, 4096),
    ("decode_m32_down", 32, 4096, 11008),
    ("decode_m64_square", 64, 8192, 8192),
    ("decode_m96_up", 96, 11008, 4096),
    ("prefill_m128_square", 128, 4096, 4096),
    ("prefill_m256_down", 256, 4096, 11008),
    ("prefill_m512_up", 512, 11008, 4096),
    ("prefill_m1024_down", 1024, 4096, 11008),
    ("prefill_m2048_square", 2048, 4096, 4096),
]

# `__global__ __launch_bounds__(kThreads, 1) void tall_bf16_gemm_kernel(...)`.
# The `__launch_bounds__` clause has to be consumed explicitly: a lazy
# `[^;{]*?` before the name happily matches `__launch_bounds__(` itself and
# names every kernel that.
KERNEL = re.compile(
    r"__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s+)?"
    r"(?:template\s*<[^>]*>\s*)?"
    r"(?:void|__device__|inline|static)\s+"
    r"(?P<name>\w+)\s*\(", re.S)


class Unreadable(ValueError):
    """The source did not yield a fact. Never downgraded to a default."""


def kernel_regions(text: str) -> dict[str, tuple[int, int]]:
    """Byte ranges of each `__global__` function body, by kernel name.

    Both kernels in this family declare `as`, `bs` and `out`, so a file-wide
    scan collides them and silently reports one kernel's LDS for the other.
    That exact bug -- attributing a figure to the wrong kernel and producing a
    tidy, plausible, wrong table -- already happened once this round with the
    codegen metadata parser, so the split comes first here.
    """
    starts = [(m.group("name"), m.start()) for m in KERNEL.finditer(text)]
    out = {}
    for i, (name, pos) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        out[name] = (pos, end)
    return out


def lds_bytes(text: str, span: tuple[int, int], known: dict[str, int]) -> int:
    """Sum of the `__shared__` declarations in one kernel body.

    This is the declared footprint, which is what decides occupancy. It can
    differ from `.group_segment_fixed_size` if the compiler pads; when both are
    available they should be cross-checked rather than one assumed.
    """
    total = 0
    body = text[span[0]:span[1]]
    for m in DECL.finditer(body):
        typename = m.group("type").strip()
        elem = ELEM_SIZE.get(typename)
        if elem is None:
            raise Unreadable(f"element type {typename!r} of {m.group('name')!r} "
                             "is not in ELEM_SIZE; its size decides the footprint")
        try:
            extents = dims(m.group("dims"), known)
        except Unresolved as exc:
            raise Unreadable(f"{m.group('name')!r}: {exc}") from exc
        size = elem
        for e in extents:
            size *= e
        total += size
    if not total:
        raise Unreadable("no __shared__ declarations found in this kernel body")
    return total


def launch_targets(text: str) -> set[str]:
    """Kernel names this file actually launches.

    Both spellings count, because the hipify twin rewrites one into the other
    and either file may be the one handed to this tool:

        foo_kernel<<<grid, block, 0, stream>>>(...)
        hipLaunchKernelGGL(( foo_kernel), dim3(grid), ...)

    An empty set means no launch site was recognised. That is reported by the
    caller as "cannot tell" rather than "not launched" -- a parser that stops
    recognising launches must not start condemning files.
    """
    out = set()
    for m in re.finditer(r"<<<", text):
        # Walk left over whitespace, then over a balanced <...> template
        # argument list if one is there, then take the identifier. A plain
        # regex cannot do this: v98 launches
        #     tiled_bf16_gemm_kernel<WAVES_M, WAVES_N, ...>
        #         <<<grid, WAVES_M * WAVES_N * 64, 0, stream>>>(...)
        # where the name is on the previous line behind template arguments that
        # themselves contain `<`. Missing it would have left this function
        # reporting only `splitk_reduce_kernel` -- the right verdict reached on
        # incomplete evidence, which is not a state to leave a diagnostic in.
        i = m.start()
        while i > 0 and text[i - 1].isspace():
            i -= 1
        if i > 0 and text[i - 1] == ">":
            depth, i = 0, i
            while i > 0:
                ch = text[i - 1]
                if ch == ")":
                    # Skip a balanced parenthesised group wholesale. A template
                    # argument may legally contain a comparison -- `foo<(A < B)>`
                    # -- and counting that `<` as a template delimiter ends the
                    # walk in the middle of the argument list and yields `A` as
                    # the kernel name.
                    pdepth = 0
                    while i > 0:
                        c2 = text[i - 1]
                        if c2 == ")":
                            pdepth += 1
                        elif c2 == "(":
                            pdepth -= 1
                            if pdepth == 0:
                                i -= 1
                                break
                        i -= 1
                    else:
                        break
                    continue
                if ch == ">":
                    depth += 1
                elif ch == "<":
                    depth -= 1
                    if depth == 0:
                        i -= 1
                        break
                i -= 1
            else:
                continue  # unbalanced; do not guess a name
            while i > 0 and text[i - 1].isspace():
                i -= 1
        j = i
        while j > 0 and (text[j - 1].isalnum() or text[j - 1] == "_"):
            j -= 1
        if j < i:
            out.add(text[j:i])
    for m in re.finditer(r"hipLaunchKernelGGL\s*\(\s*\(?\s*([A-Za-z_]\w*)", text):
        out.add(m.group(1))
    return out


def read_source(path: Path) -> dict:
    """Constants and per-kernel LDS, all from the file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    known = constants(text)
    for name in ("kTile", "kWaves", "kTallM", "kStageK"):
        if name not in known:
            raise Unreadable(f"{name} is not a resolvable constexpr in {path}; "
                             "the launch geometry cannot be derived")
    regions = kernel_regions(text)
    tall = [n for n in regions if "tall" in n]
    generic = [n for n in regions if "generic" in n]
    if len(tall) != 1 or len(generic) != 1:
        raise Unreadable(f"expected one tall and one generic __global__, found "
                         f"tall={tall} generic={generic}")
    # Being DEFINED is not the same as being LAUNCHED, and the difference is the
    # whole value of this tool. The v98 ship point still carries `tall` and
    # `generic` verbatim from the 231-line seed, but launches neither -- it
    # dispatches `tiled_bf16_gemm_kernel` through a slice planner instead. The
    # checks above pass on that file and every grid, CTA count, occupancy and
    # reround figure below then describes code that cannot execute. That is the
    # failure this module's own docstring names ("attributing a figure to the
    # wrong kernel and producing a plausible table"), committed by the module
    # against a real, current input. A plausible table about dead code is worse
    # than an error, because it gets believed and planned against.
    launched = launch_targets(text)
    if launched:
        missing = [k for k in (tall[0], generic[0]) if k not in launched]
        if missing:
            raise Unreadable(
                f"{path}: {' and '.join(missing)} is DEFINED but never launched; "
                f"the launch sites in this file name {sorted(launched)}. This "
                "file is not the tall/generic family this tool models -- most "
                "likely the tall/generic kernels are dead code inherited from "
                "the seed. Every grid and occupancy figure would describe code "
                "that never runs, so none is reported.")
    return {
        "consts": known,
        "lds": {"tall": lds_bytes(text, regions[tall[0]], known),
                "generic": lds_bytes(text, regions[generic[0]], known)},
        "kernels": {"tall": tall[0], "generic": generic[0]},
    }


def rows(src: dict) -> list[dict]:
    c = src["consts"]
    kTile, kWaves, kTallM = c["kTile"], c["kWaves"], c["kTallM"]
    out = []
    for name, m, n, k in CASES:
        kernel = "generic" if m < kTallM else "tall"
        tile_m = kTile if m < kTallM else kTallM
        gx = (n + kWaves * kTile - 1) // (kWaves * kTile)
        gy = (m + tile_m - 1) // tile_m
        ctas = gx * gy
        slots_per_cu = LDS_PER_CU // src["lds"][kernel]
        slots = CUS * slots_per_cu
        out.append({
            "case": name, "M": m, "N": n, "K": k, "kernel": kernel,
            "lds_bytes": src["lds"][kernel],
            "grid": f"{gx}x{gy}", "ctas": ctas,
            "ctas_per_cu_cap": slots_per_cu, "slots": slots,
            "rounds": -(-ctas // slots),
            # The most generous reading -- spread before stacking -- which is the
            # right one for an argument that the number is nonetheless small.
            "cu_touched_pct": 100.0 * min(ctas, CUS) / CUS,
            "occupancy_pct": 100.0 * ctas / slots,
            "tile_rows_useful_pct": 100.0 * m / (gy * tile_m),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        src = read_source(Path(args.path))
    except Unreadable as exc:
        print(f"{args.path}: UNREADABLE {exc}", file=sys.stderr)
        return 2
    table = rows(src)
    if args.json:
        print(json.dumps({"source": args.path, "consts": src["consts"],
                          "lds": src["lds"], "rows": table}, sort_keys=True))
        return 0
    print(f"# {args.path}")
    print(f"# derived: " + ", ".join(f"{k}={src['consts'][k]}"
                                     for k in ("kTile", "kWaves", "kTallM", "kStageK")))
    print(f"# LDS: tall={src['lds']['tall']} generic={src['lds']['generic']}")
    print(f"{'case':22s} {'kern':8s} {'grid':>10s} {'ctas':>6s} {'slots':>6s} "
          f"{'rnds':>4s} {'CUs%':>6s} {'occ%':>6s} {'tileM%':>7s}")
    for r in table:
        print(f"{r['case']:22s} {r['kernel']:8s} {r['grid']:>10s} {r['ctas']:>6d} "
              f"{r['slots']:>6d} {r['rounds']:>4d} {r['cu_touched_pct']:>6.1f} "
              f"{r['occupancy_pct']:>6.1f} {r['tile_rows_useful_pct']:>7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
