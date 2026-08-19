#!/usr/bin/env python3
"""Check a latency table against physics, then classify each route's regime.

Two jobs, in this order, because the second is worthless without the first.

JOB 1 -- IS THIS TABLE A MEASUREMENT OF THIS KERNEL DOING THIS WORK?

A shape fixes two quantities no implementation can beat: the FLOPs it must
issue and the bytes it must move. Divide them by a claimed latency and you get
an implied rate, and a rate above the device's peak is not a fast kernel, it is
a number that is not measuring the work. This is decidable, costs nothing, and
catches a failure mode that reads as good news.

It has already caught one. A `seed_candidate_ms_median` column sat flat at
0.0081-0.0090 ms across every route from M=2 to M=2048 -- shapes whose work
differs by three orders of magnitude. On `prefill_m2048_square` (68.7 GFLOP)
that implies 8.5 PFLOP/s against a bf16 peak near 1.3, so the column is roughly
constant launch overhead, not the kernel computing anything. Nothing in the tree
consumed it, which is luck rather than a property. A flat column is the
signature: real work tracks the shape.

JOB 2 -- IS "MORE CUs WOULD HELP" A LIVE QUESTION ON THIS ROUTE?

Some routes hand work to a fraction of the device: the shipped kernel gives
`decode_m2_square` 64 workgroups on 304 CUs. The tempting reading is that the
reciprocal is available -- touch 4.7x the CUs, go 4.7x faster. Finding (23)
measured the reciprocal-of-utilisation idea twice and lost both times (-21.4%,
-11.3%), but in a DIFFERENT regime: those routes already touched every CU and
the change was 2 co-resident CTAs per CU versus 3. Stacking CTAs on a busy CU
and lighting up a CU that has nothing are not the same mechanism, so (23) does
not settle this one -- and that is a reason it may be open, not evidence it is.

The discriminator: if a route is limited by how many CUs are pulling, achieved
bandwidth should scale with CUs touched. `--experiment` evaluates that
prediction against routes already measured, which needs no device.

TWO BYTE MODELS, because one is not enough. `compulsory` counts A, B and C once
-- the traffic that must cross the pins. `requested` counts the re-reads the
grid issues (B once per tile row, A once per tile column). Real DRAM traffic is
between them, and they coincide only when the grid barely re-reads. Where they
do not, the route cannot carry a bandwidth argument and is marked unusable
rather than quietly ranked.

PROVENANCE IS MANDATORY. `--timing` and `--field` are required and echoed into
the output. An earlier draft of this file hardcoded a latency table and paired
it with a kernel it did not belong to -- committing finding (144) inside the
tool written against (144). Latencies now come from a named file and a named
field, and matching them to the kernel is the caller's declared claim, printed
where a reader can dispute it.

    bandwidth_regime.py kernel.hip --timing baseline_timing.json --field latency_ms
    bandwidth_regime.py kernel.hip --timing t.json --field baseline_ms --experiment

Exit: 0 plausible; 2 the source was unreadable; 3 the latency table is not
physically possible for this work and no classification was attempted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel_launch_facts import CUS, Unreadable, read_source, rows  # noqa: E402

BYTES_AB = 2
BYTES_C = 2

# gfx942 / MI300X. Spec peaks, stated rather than folded into a verdict so the
# discount to an achievable rate stays arguable. They are used ONLY as an
# impossibility bound -- exceeding them is disqualifying, approaching them is
# not by itself suspicious.
HBM_PEAK_TBS = 5.3
BF16_PEAK_TFLOPS = 1307.0


class Implausible(ValueError):
    """The latency table cannot be a measurement of this work."""


def traffic(r: dict) -> dict:
    m, n, k = r["M"], r["N"], r["K"]
    gx, gy = (int(x) for x in r["grid"].split("x"))
    a, b, c = m * k * BYTES_AB, k * n * BYTES_AB, m * n * BYTES_C
    return {"a": a, "b": b, "c": c,
            "compulsory": a + b + c,
            "requested": gy * b + gx * a + c,
            "flop": 2 * m * n * k}


def load_timings(path: Path, field: str) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("test_cases")
    if not isinstance(cases, list):
        raise Implausible(f"{path} has no `test_cases` list")
    # The route key: `task_runner.py performance` emits `test_case_id`, while
    # its `correctness` path emits `name`. This reader only knew `name`, so it
    # raised KeyError on every real performance JSON -- i.e. on its own intended
    # input. Accept either, prefer the performance spelling, and say which keys
    # were present when neither is, rather than dying on a bare KeyError.
    out = {}
    for c in cases:
        key = next((k for k in ("test_case_id", "name") if k in c), None)
        if key is None:
            raise Implausible(
                f"{path}: a case has neither `test_case_id` nor `name`; "
                f"keys present are {sorted(c)}")
        if field not in c or c[field] is None:
            raise Implausible(f"{path}: case {c[key]!r} has no {field!r}; "
                              "a partly-populated field cannot be compared across "
                              "routes")
        out[c[key]] = float(c[field])
    return out


def check_physics(table: list[dict]) -> list[str]:
    """Every way this latency table is impossible. Empty means it survived."""
    bad = []
    for r in table:
        if r["ms"] <= 0:
            bad.append(f"{r['case']}: latency {r['ms']} is not positive")
            continue
        secs = r["ms"] * 1e-3
        tflops = r["flop"] / secs / 1e12
        tbs = r["compulsory"] / secs / 1e12
        if tflops > BF16_PEAK_TFLOPS:
            bad.append(f"{r['case']}: implies {tflops:.0f} TFLOP/s against a bf16 "
                       f"peak of {BF16_PEAK_TFLOPS:.0f} -- {tflops / BF16_PEAK_TFLOPS:.1f}x "
                       "over, so this is not the kernel doing this shape's work")
        if tbs > HBM_PEAK_TBS:
            bad.append(f"{r['case']}: implies {tbs:.2f} TB/s of compulsory traffic "
                       f"against an HBM peak of {HBM_PEAK_TBS} -- the bytes cannot "
                       "arrive that fast")
    return bad


def flatness(table: list[dict]) -> str | None:
    """A column that ignores the shape is measuring something other than work.

    Advisory, not disqualifying: it is the signature the impossible column had,
    and it can be visible on a table where no single route is over peak.
    """
    ms = [r["ms"] for r in table if r["ms"] > 0]
    flops = [r["flop"] for r in table]
    if len(ms) < 3:
        return None
    ms_ratio = max(ms) / min(ms)
    flop_ratio = max(flops) / min(flops)
    if flop_ratio > 100 and ms_ratio < 2.0:
        return (f"latency varies {ms_ratio:.2f}x across routes whose FLOP counts "
                f"vary {flop_ratio:.0f}x -- a near-constant column is the signature "
                "of launch overhead, not of work")
    return None


def analyse(path: Path, timings: dict[str, float]) -> list[dict]:
    out = []
    for r in rows(read_source(path)):
        if r["case"] not in timings:
            raise Implausible(f"no latency for {r['case']!r} in the timing file; "
                              "a partial suite cannot be classified")
        t = traffic(r)
        gx, gy = (int(x) for x in r["grid"].split("x"))
        secs = timings[r["case"]] * 1e-3
        exact = gy == 1 and t["requested"] <= 1.25 * t["compulsory"]
        # A non-positive latency is one of the impossibilities `check_physics`
        # reports, so it must survive to be reported rather than raising here.
        # Infinity is the honest rate for "this work took no time".
        rate = (lambda x: x / secs / 1e12) if secs > 0 else (lambda x: float("inf"))
        out.append({**r, **t, "ms": timings[r["case"]], "exact": exact,
                    "reread": t["requested"] / t["compulsory"],
                    "compulsory_tbs": rate(t["compulsory"]),
                    "tflops": rate(t["flop"])})
    for r in out:
        r["pct_peak_bw"] = 100.0 * r["compulsory_tbs"] / HBM_PEAK_TBS
        r["regime"] = ("unusable(cache-mediated)" if not r["exact"]
                       else "closed-by-23(all CUs already pulling)"
                       if r["cu_touched_pct"] >= 99.0 else "open(CUs idle)")
    return out


def natural_experiment(table: list[dict]) -> list[str]:
    """Does bandwidth scale with CUs touched, on routes already measured?"""
    usable = [r for r in table if r["exact"]]
    if len(usable) < 2:
        return ["not enough exact-model routes to evaluate the prediction"]
    base = min(usable, key=lambda r: r["cu_touched_pct"])
    lines = [f"reference: {base['case']} at {base['cu_touched_pct']:.1f}% of CUs, "
             f"{base['compulsory_tbs']:.2f} TB/s compulsory",
             "prediction: bandwidth rises in proportion to CUs touched", "",
             f"{'case':22s} {'CUs%':>6s} {'CUx':>5s} {'pred':>7s} {'obs':>7s} "
             f"{'obs/pred':>9s}"]
    for r in sorted(usable, key=lambda r: r["cu_touched_pct"]):
        cux = r["cu_touched_pct"] / base["cu_touched_pct"]
        pred = base["compulsory_tbs"] * cux
        lines.append(f"{r['case']:22s} {r['cu_touched_pct']:>6.1f} {cux:>5.2f} "
                     f"{pred:>7.2f} {r['compulsory_tbs']:>7.2f} "
                     f"{r['compulsory_tbs'] / pred:>9.2f}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--timing", required=True,
                    help="JSON with a `test_cases` list; provenance is echoed")
    ap.add_argument("--field", required=True,
                    help="which per-case field to read as the latency in ms")
    ap.add_argument("--experiment", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        timings = load_timings(Path(args.timing), args.field)
        table = analyse(Path(args.path), timings)
    except Unreadable as exc:
        print(f"{args.path}: UNREADABLE {exc}", file=sys.stderr)
        return 2
    except Implausible as exc:
        print(f"IMPLAUSIBLE {exc}", file=sys.stderr)
        return 3

    impossible = check_physics(table)
    flat = flatness(table)

    if args.json:
        print(json.dumps({"kernel": args.path, "timing": args.timing,
                          "field": args.field, "impossible": impossible,
                          "flatness": flat, "rows": table}, sort_keys=True))
        return 3 if impossible else 0

    print(f"# kernel : {args.path}")
    print(f"# timing : {args.timing} field={args.field}")
    if impossible:
        print("\n!! THIS TABLE IS NOT A MEASUREMENT OF THIS KERNEL DOING THIS WORK",
              file=sys.stderr)
        for line in impossible:
            print(f"   {line}", file=sys.stderr)
        if flat:
            print(f"   {flat}", file=sys.stderr)
        print("   no regime classification attempted.", file=sys.stderr)
        return 3
    if flat:
        print(f"# WARNING: {flat}", file=sys.stderr)

    print(f"{'case':22s} {'CUs%':>6s} {'ms':>8s} {'compMB':>8s} {'reread':>7s} "
          f"{'TB/s':>6s} {'%peak':>6s} {'TFLOP/s':>8s}  regime")
    for r in table:
        print(f"{r['case']:22s} {r['cu_touched_pct']:>6.1f} {r['ms']:>8.5f} "
              f"{r['compulsory'] / 1e6:>8.1f} {r['reread']:>7.2f} "
              f"{r['compulsory_tbs']:>6.2f} {r['pct_peak_bw']:>6.1f} "
              f"{r['tflops']:>8.1f}  {r['regime']}")
    if args.experiment:
        print()
        print("-- scaling prediction, on routes whose byte model is exact --")
        for line in natural_experiment(table):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
