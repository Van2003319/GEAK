#!/usr/bin/env python3
"""Re-derive a QD archive's case-level admissions under a different noise floor.

Finding (107) is why this exists. The floor table in `qd_robust_stats.py` was
carried across a machine boundary on the strength of an argument -- the floors
are *relative*, so they should travel -- and when it was finally re-measured,
four of eleven routes came back NARROWER, one by 3.3x. Every admission made in
between rested on a floor that was not that machine's floor.

That raises a question no single-machine tool can answer: how much of what is in
the archive actually depended on the floor being right? This walks an archive
manifest, recomputes each case's `median +- max(2*MAD, |median| * floor)`
interval from the raw `verify_samples_ms` under each candidate table, and
reports which of them change their answer to "does this interval clear 1.0x?".

It counts DISTINCT MEASUREMENTS, not archive cases -- an elite that won K cells
carries K copies of one suite, and counting the copies inflates the evidence
base (6x on the first archive audited here). See `audit`.

Two things it deliberately does NOT do:

* It does not re-admit or rewrite anything. The archive is evidence; this is a
  reading of it.
* It does not tell you which table is "correct" for an archive measured on
  another machine. Floors do not pool across a machine boundary, so applying
  machine N's floors to a machine L archive is (107) committed a second time.
  The useful question is SENSITIVITY -- would a wrong floor have changed this
  archive? -- and that is machine-agnostic. Hence `--scale`, which asks the
  question without naming a second machine at all.

Exit status is 0 whatever it finds: this is an audit, not a gate. A gate here
would be a gate on history, which cannot be repaired by failing.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import qd_robust_stats as robust

SCHEMA = "geak.qd-floor-sensitivity/v1"

# The bar a case has to clear. A case whose whole interval sits above 1.0 is
# faster than its baseline by more than the route can fake; this is the same
# comparison `qdCaseRobust` feeds into admission, evaluated one case at a time.
NEUTRAL = 1.0


#: Every unit a timing has actually been written in anywhere in this pipeline,
#: and what one of it is worth in milliseconds. `ms` is the pipeline's unit
#: everywhere; the others are here to be RECOGNISED and refused, not supported.
UNIT_TO_MS: Mapping[str, float] = {"ms": 1.0, "s": 1000.0, "us": 1e-3, "ns": 1e-6}

#: How far the samples may sit from the candidate latency recorded beside them
#: before the declared unit is disbelieved. Deliberately loose: `optimized_ms`
#: is one summary of a run and the samples are its repeats, so a factor of two
#: either way is ordinary. The error this catches is not a few percent, it is
#: 1000x, and no honest spread reaches the gap between the two bands.
UNIT_TOLERANCE = 2.0


def declared_unit(case: Mapping[str, object], key: str) -> str | None:
    """The unit the case says its samples are in, or None if it never says.

    A key that ENDS in the unit declares it -- `verify_samples_ms`, `samples_ms`
    -- which is why those spellings are worth keeping. A bare `samples` declares
    nothing, and neither does an explicit `samples_unit` naming something this
    table has never heard of. Both come back None: "I do not know" and "the
    stated unit is wrong" are different questions, and only the second is a
    refusal, so they must not collapse here.
    """
    stated = case.get("samples_unit") or case.get("unit")
    if isinstance(stated, str):
        return stated.strip().lower() if stated.strip().lower() in UNIT_TO_MS else None
    for name in UNIT_TO_MS:
        if key.endswith(f"_{name}"):
            return name
    return None


def unit_disagrees(samples: Sequence[float], unit: str | None,
                   optimized_ms: object) -> str | None:
    """Why the values cannot be the unit they claim, or None if they can be.

    This is the one check in this module that is about the INPUT rather than
    about history, and it is the reason it can fail the run. A samples array in
    seconds read as milliseconds does not produce a wrong-looking answer -- it
    produces `baseline_ms / 0.000027`, a speedup of ~1000x, an interval far
    above 1.0, and a confident "every admission survives every floor". The
    fabrication is indistinguishable from a triumph.

    It needs a second number in a known unit to compare against, and
    `optimized_ms` is one by its own name. With no such number the answer is
    None: an unverifiable claim is not a refutable one, and refusing here would
    reject every legacy case rather than the broken ones.
    """
    if unit is None:
        return None
    xs = [float(x) for x in samples if isinstance(x, (int, float)) and x > 0]
    if not xs or not isinstance(optimized_ms, (int, float)) or optimized_ms <= 0:
        return None
    median_ms = statistics.median(xs) * UNIT_TO_MS[unit]
    ratio = median_ms / float(optimized_ms)
    if 1.0 / UNIT_TOLERANCE <= ratio <= UNIT_TOLERANCE:
        return None
    implied = min(UNIT_TO_MS, key=lambda u: abs(
        (statistics.median(xs) * UNIT_TO_MS[u]) / float(optimized_ms) - 1.0))
    return (f"samples declared in {unit!r} have median {median_ms:.6g} ms against an "
            f"optimized_ms of {float(optimized_ms):.6g} -- off by {ratio:.4g}x. "
            f"They would be consistent as {implied!r}.")


def iter_cases(manifest: Mapping[str, object]) -> Iterator[dict[str, object]]:
    """Every (elite, case) with raw per-sample timings, across manifest shapes.

    The archive format has changed more than once -- `per_case` at the cell, an
    `elite` sub-object, `case_measurement_samples` -- and an audit that silently
    read zero cases out of a shape it did not recognise would report a clean
    bill of health for an archive it never opened. So the shapes are enumerated
    here and the caller is told the count; see `test_a_shape_this_does_not_know_
    is_reported_as_unread_not_as_clean`.
    """
    cells = manifest.get("cells")
    if not isinstance(cells, dict):
        return
    for cell_key, cell in cells.items():
        if not isinstance(cell, dict):
            continue
        holder = cell.get("elite") if isinstance(cell.get("elite"), dict) else cell
        elite_id = holder.get("elite_id") or cell.get("elite_id") or cell_key
        cases = holder.get("per_case") or holder.get("case_measurement_samples") or []
        if not isinstance(cases, list):
            continue
        for case in cases:
            if not isinstance(case, dict):
                continue
            # `samples_ms` is what the v2 manifest writes; the other two are
            # earlier spellings. This list existing is not enough -- see the
            # `unread` check in main(), which is what actually caught the v2
            # rename. Between round 16's archive being written and that check
            # being added, this audit read the v2 manifest as zero cases and
            # printed `"cases_read":0,"flips":[]`, which reads exactly like
            # "no admission depended on the floor" and meant "I did not find
            # any admissions".
            samples, samples_key = None, ""
            for key in ("verify_samples_ms", "samples_ms", "samples"):
                if case.get(key):
                    samples, samples_key = case.get(key), key
                    break
            name = case.get("name")
            if not isinstance(samples, list) or not samples or not isinstance(name, str):
                continue
            # Round 1's engineer wrote its own `runs/speedup_samples.json` with a
            # twelfth key, `__suite_geomean__`, holding the per-rep geomean of
            # the eleven routes. That file is an engineer artifact and never
            # reaches the manifest, but nothing stops the next one from being
            # copied in wholesale, and the failure would be silent: the
            # aggregate has samples and a name, so it would be counted as a
            # twelfth route with an interval far tighter than any real one, and
            # "12 of 12 clear" would read as broader coverage than 11. A
            # dunder-fenced name is the pipeline's convention for "not a route".
            if name.startswith("__"):
                continue
            sp = case.get("speedup_samples")
            unit = declared_unit(case, samples_key)
            yield {"elite_id": elite_id, "cell": cell_key, "context": name,
                   # Normalised to ms, because everything downstream --
                   # `speedups`, `clears`, `measurement_key` -- assumes ms and
                   # says so in its own field names. A case that declares `s`
                   # AND is consistent with its own `optimized_ms` is not
                   # broken, it is merely in another unit, and the honest
                   # handling of that is to convert it, not to refuse it. The
                   # refusal is for values that can be NO declared unit.
                   "samples": [float(s) * UNIT_TO_MS.get(unit, 1.0) for s in samples],
                   "samples_unit": unit,
                   "unit_disagrees": unit_disagrees(samples, unit,
                                                    case.get("optimized_ms")),
                   "speedup_samples": ([float(s) for s in sp]
                                       if isinstance(sp, list) and sp else None),
                   "baseline_ms": case.get("baseline_ms"),
                   "optimized_ms": case.get("optimized_ms")}


def speedups(case: Mapping[str, object]) -> list[float] | None:
    """Per-sample speedups, or None when the case cannot yield them.

    `verify_samples_ms` are latencies; admission compares speedups. Dividing the
    recorded baseline by each sample keeps the per-sample spread, which is the
    whole input to the interval -- taking the ratio of the two MEDIANS instead
    would throw the spread away and make every interval look tight.

    `speedup_samples` is preferred where the manifest carries it, because it is
    the pipeline's own paired per-rep ratio (rep i's baseline over rep i's
    candidate) rather than this script's reconstruction of one.

    The `baseline_ms == optimized_ms` guard is not defensive padding. In round
    16's archive that field holds the CANDIDATE latency, identical to
    `optimized_ms` and `latency_ms`, with the real baseline in
    `baseline_samples_ms`. Dividing by it yields a per-sample ratio of ~1.0 for
    every case, so every interval straddles 1.0 and the audit reports zero
    admissions clearing under every floor -- a confident, uniform, entirely
    fabricated answer. A wrong baseline is worse than a missing one, because a
    missing one is counted in `cases_without_baseline` and a wrong one is not.
    """
    paired = case.get("speedup_samples")
    if paired:
        xs = [float(s) for s in paired if isinstance(s, (int, float)) and s > 0]
        if xs:
            return xs
    base = case.get("baseline_ms")
    if not isinstance(base, (int, float)) or base <= 0:
        return None
    opt = case.get("optimized_ms")
    if isinstance(opt, (int, float)) and opt == base:
        return None
    xs = [float(s) for s in case["samples"] if isinstance(s, (int, float)) and s > 0]
    return [base / x for x in xs] if xs else None


def measurement_key(case: Mapping[str, object]) -> tuple:
    """What makes two archive cases the SAME measurement rather than two of them.

    The route, the baseline it was compared against, and the raw samples. Not
    the elite id -- that is the cell the measurement was filed under, and one
    measurement gets filed under as many cells as its variant won.
    """
    return (case["context"], case.get("baseline_ms"), tuple(case["samples"]))


def clears(sp: Sequence[float], floor: float) -> dict[str, float | bool]:
    """`median - max(2*MAD, median*floor) > 1.0`, plus the numbers behind it."""
    median = statistics.median(sp)
    mad = statistics.median([abs(x - median) for x in sp])
    radius = max(mad * robust.MAD_BOUND_MULTIPLIER, abs(median) * floor)
    return {"median": median, "mad": mad, "radius": radius,
            "lower": median - radius, "clears": (median - radius) > NEUTRAL}


def audit(manifest: Mapping[str, object], tables: Mapping[str, Mapping[str, float]],
          reference: str) -> dict[str, object]:
    """Per-measurement verdicts under every named table, and the ones that disagree.

    **Cases are not evidence; measurements are.** An elite that wins K cells
    appears in the manifest K times, each copy carrying the SAME suite it was
    measured with once. Counting cases therefore multiplies one measurement by
    cell occupancy: the first archive audited here reads 132 cases that are 22
    distinct measurements -- two variants x eleven routes -- a 6x inflation, and
    "128 of 132 survive" would have been a far stronger claim than the evidence
    supports. So the headline counts and `flips` are per DISTINCT measurement,
    keyed by (context, baseline, samples), and each flip lists the cells that
    replicate it. `cases_read` is still reported, because the gap between the
    two numbers is itself the thing a reader needs to see.
    """
    rows: list[dict[str, object]] = []
    unusable = 0
    sources: dict[str, int] = {}
    unit_rejects: list[dict[str, object]] = []
    units: dict[str, int] = {}
    for case in iter_cases(manifest):
        units[str(case.get("samples_unit"))] = units.get(str(case.get("samples_unit")), 0) + 1
        # Before anything is computed from them. A case whose numbers cannot be
        # the unit they are labelled with is not a weak measurement to be
        # weighted down, it is a measurement in an unknown unit, and every
        # statistic downstream of it is arithmetic on two different quantities.
        if case.get("unit_disagrees"):
            unit_rejects.append({"elite_id": case["elite_id"], "context": case["context"],
                                 "declared_unit": case.get("samples_unit"),
                                 "reason": case["unit_disagrees"]})
            continue
        sp = speedups(case)
        if sp is None:
            unusable += 1
            continue
        # What the ratio was actually built from. A verdict of "8 of 11 clear"
        # means nothing until you know whether the denominator was the frozen
        # oracle or the elite's own parent, and those are different questions
        # with the same field names in the manifest.
        src = ("paired_speedup_samples" if case.get("speedup_samples")
               else "reconstructed_from_baseline_ms")
        sources[src] = sources.get(src, 0) + 1
        verdicts = {}
        for label, table in tables.items():
            floor = table.get(case["context"], robust.DEFAULT_NOISE_FLOOR)
            verdicts[label] = {**clears(sp, floor), "floor": floor}
        rows.append({"elite_id": case["elite_id"], "context": case["context"],
                     "key": measurement_key(case), "n": len(sp), "verdicts": verdicts})

    distinct: dict[tuple, dict[str, object]] = {}
    for row in rows:
        seen = distinct.get(row["key"])
        if seen is None:
            distinct[row["key"]] = {**{k: v for k, v in row.items() if k != "elite_id"},
                                    "elite_ids": [row["elite_id"]]}
        else:
            seen["elite_ids"].append(row["elite_id"])
    measurements = sorted(distinct.values(), key=lambda r: (r["context"], r["elite_ids"][0]))
    for m in measurements:
        m["elite_ids"] = sorted(m["elite_ids"])
        m["replicated_across_cells"] = len(m["elite_ids"])
        m.pop("key")

    ref = reference in tables
    flips = [m for m in measurements
             if ref and len({v["clears"] for v in m["verdicts"].values()}) > 1]
    return {
        "schema": SCHEMA,
        "cases_read": len(rows),
        "distinct_measurements": len(measurements),
        "cases_without_baseline": unusable,
        "speedup_sources": sources,
        "sample_units": units,
        "cases_rejected_bad_unit": len(unit_rejects),
        "unit_rejections": unit_rejects,
        "reference": reference,
        "clears_by_table": {label: sum(m["verdicts"][label]["clears"] for m in measurements)
                            for label in tables},
        "clears_by_table_cases": {label: sum(r["verdicts"][label]["clears"] for r in rows)
                                  for label in tables},
        "flips": flips,
        "rows": measurements,
    }


def scaled(table: Mapping[str, float], factor: float) -> dict[str, float]:
    return {k: v * factor for k, v in table.items()}


def _parser():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", help="path to a qd_archive/manifest.json")
    p.add_argument("--machine", action="append", default=None,
                   help="machine table to evaluate under; repeatable. Defaults to "
                        "the archive's own epoch ONLY if you name it -- there is no "
                        "safe default, since the manifest does not record its box.")
    p.add_argument("--scale", type=float, action="append", default=None,
                   help="also evaluate the first named machine's table multiplied "
                        "by this factor; repeatable. This is the machine-agnostic "
                        "sensitivity question: would a floor wrong by Nx have "
                        "changed any admission?")
    p.add_argument("--rows", action="store_true",
                   help="include the full per-measurement table, not just the flips")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    import sys
    args = _parser().parse_args(argv)
    machines = args.machine or [robust.CURRENT_MACHINE]
    tables: dict[str, Mapping[str, float]] = {}
    for name in machines:
        table = robust.MEASURED_NOISE_FLOOR_BY_MACHINE.get(name)
        if table is None:
            sys.stderr.write(f"unknown machine: {name}\n")
            return 2
        tables[name] = table
    for factor in args.scale or []:
        tables[f"{machines[0]}x{factor:g}"] = scaled(tables[machines[0]], factor)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    report = audit(manifest, tables, machines[0])
    if not args.rows:
        report.pop("rows")
    sys.stdout.write(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")

    # An archive with cells but no readable measurement is not a clean audit, it
    # is an unread one, and the two produce the same JSON: `flips: []`. Reporting
    # `cases_read: 0` inside the blob was supposed to cover this, but a count a
    # reader has to notice is not a defence -- round 16's manifest renamed the
    # sample field and this printed a clean report for an archive it had not
    # opened. Exit 4 and say so on stderr.
    # A unit refusal is the one finding here that is about the FILE and not
    # about history, so it is the one that can legitimately fail. Exit 5 before
    # the unread check, because a manifest whose every case was thrown out on
    # units would otherwise be reported as unread -- true, but it buries the
    # reason, and the reason is the actionable half.
    if report["cases_rejected_bad_unit"]:
        for reject in report["unit_rejections"]:
            sys.stderr.write(f"BAD UNIT: {reject['context']} in {reject['elite_id']}: "
                             f"{reject['reason']}\n")
        sys.stderr.write(
            f"REFUSED: {report['cases_rejected_bad_unit']} case(s) carry timings that "
            f"cannot be the unit they declare. Read as declared they do not produce a "
            f"wrong-looking answer, they produce a ~1000x speedup that clears every "
            f"floor -- so they are excluded rather than weighted, and the counts above "
            f"describe only the cases that survived.\n")
        return 5

    cells = manifest.get("cells")
    if isinstance(cells, dict) and cells and not report["cases_read"]:
        sys.stderr.write(
            f"UNREAD: {args.manifest} has {len(cells)} cell(s) but no case with raw "
            f"per-sample timings was recognised. The verdict above is not 'nothing "
            f"depended on the floor', it is 'nothing was examined'. Most likely the "
            f"manifest spells the samples under a key `iter_cases` does not know; "
            f"the ones it knows are verify_samples_ms, samples_ms, samples.\n")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
