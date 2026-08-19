#!/usr/bin/env python3
"""Measure an epoch's per-route noise floor, and refuse to guess one.

Findings (26) and (58) put a per-route floor under every admission decision:
`bound_radius = max(2*MAD, |median| * floor(route))`, so a route that is quiet
in three samples but loud in twenty cannot present a tight interval. The floor
is therefore a *measurement*, and a measurement belongs to the box it was taken
on -- floors do not pool across a machine boundary any more than `candidate_ms`
does.

Epoch Q (tw003) is currently running on the fail-closed DEFAULT floor of 0.072
on every route, because the container was restored onto a box whose eight GPUs
were all held at 96-98%% by a foreign job. That default is the honest choice and
it is expensive: at 0.072 a 2-5%% suite move is unreadable and cannot be
admitted. This script is the thing that ends it. It is authored ahead of the
device so that the moment one frees up the measurement is a single command and
not an authoring session.

What it does, in order:

1. Hash the task tree. The whole point is *same-variant* repeats: the spread
   being measured is the harness's, not the kernel's. If the tree changes
   mid-sweep the run is void, and the script says so rather than averaging two
   variants into one floor.
2. Run correctness once and require it to pass. Rule standing since (34):
   correctness passes before any timing is reported.
3. Run the full 11-case suite `--repeats` times, each in a FRESH process, so
   every repeat pays its own priming cost exactly as a real measurement does.
4. Per route, floor = 2*MAD(speedup) / median(speedup) -- the same statistic
   `noise_floor_stats.robust_stats` applies, expressed as a fraction of the
   median so it is dimensionless and survives the epoch it was measured in.
5. Emit the table two ways: JSON for machines, and a Python dict literal for
   `noise_floor_stats.py` -- the same numbers, in the shape of the diff they
   are about to become.

It fails closed everywhere it can:

- fewer than `MIN_REPEATS` complete repeats                       -> exit 4
- the route set does not exactly match the reference epoch's      -> exit 4
- the source hash moves mid-sweep                                 -> exit 3
- correctness fails                                               -> exit 3
- a speedup outside (1/100, 100) -- finding (132), a unit error
  does not produce a wrong-LOOKING answer, so it is caught here
  by magnitude rather than trusted                                -> exit 4

and it will not emit a floor tighter than `MIN_FLOOR`. A floor of exactly 0.0
is arithmetically possible (MAD is 0 when over half the samples are identical)
and it is never a true statement about a GPU; clamping is the direction that
can only refuse admissions, never grant one, and the clamp is recorded per
route rather than hidden.

Writing the table into `noise_floor_stats.py` is deliberately NOT automated.
Replacing a floor table is the step that decides what the archive will accept
for the rest of the epoch; it gets a human-legible diff, and `Q` comes out of
`PROVISIONAL_MACHINES` in the same edit.
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import noise_floor_stats as QRS  # noqa: E402
import source_hash as QSH  # noqa: E402

# Eight is the standing figure for a floor sweep: enough that the MAD is not
# itself dominated by the sample count, few enough to fit in one lock window.
DEFAULT_REPEATS = 8

# Three is the floor on the floor. Below this the MAD of the sample is a
# coin-flip and the resulting table would be a number with a measurement's
# authority and a guess's content.
MIN_REPEATS = 3

# The tightest floor ever measured anywhere in this ledger is 0.005
# (`prefill_m256_down` on machine L). Anything below 0.002 is the sampler
# running out of resolution, not the GPU being quiet.
MIN_FLOOR = 0.002

# Finding (132): seconds read as ms give a ~1000x speedup that clears every
# gate -- a fabrication indistinguishable from a triumph. A speedup is bounded
# by physics long before it is bounded by 100.
SPEEDUP_SANE_LO = 0.01
SPEEDUP_SANE_HI = 100.0

# The route set a table must cover, taken from the last epoch that actually
# measured one. Reading it from the reference table rather than hardcoding 11
# strings means a suite that gains a case fails this check loudly instead of
# quietly producing a table with a hole in it.
REFERENCE_EPOCH = "P"


def reference_routes() -> frozenset[str]:
    return frozenset(QRS.MEASURED_NOISE_FLOOR_BY_MACHINE[REFERENCE_EPOCH])


def mad(samples: Sequence[float]) -> float:
    """Median absolute deviation from the median, raw (no 1.4826 scaling)."""
    if not samples:
        return 0.0
    med = statistics.median(samples)
    return statistics.median([abs(s - med) for s in samples])


def floor_from_speedups(samples: Sequence[float]) -> dict[str, Any]:
    """`2*MAD/median` as a fraction, clamped up to MIN_FLOOR and said so.

    The statistic is deliberately the one `robust_stats` uses, divided by the
    median so it is a *relative* half-width. An absolute one would be a latency
    and would not survive the epoch boundary it exists to respect.
    """
    med = statistics.median(samples)
    raw = 2.0 * mad(samples) / med if med > 0 else float("inf")
    clamped = raw < MIN_FLOOR
    return {
        "n": len(samples),
        "median_speedup": med,
        "mad_speedup": mad(samples),
        "floor_raw": raw,
        "floor": max(raw, MIN_FLOOR),
        "clamped_to_min": clamped,
    }


def run_suite(task_dir: Path, python: str = sys.executable) -> list[dict[str, Any]]:
    """One full-suite performance run, in a fresh process. Raises on failure."""
    runner = task_dir / "scripts" / "task_runner.py"
    proc = subprocess.run([python, str(runner), "performance"],
                          cwd=str(task_dir), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"performance run failed (rc={proc.returncode})\n{proc.stdout[-2000:]}\n"
            f"{proc.stderr[-2000:]}")
    report = task_dir / "build" / "performance_report.json"
    if not report.is_file():
        raise RuntimeError(f"performance run wrote no report at {report}")
    rows = json.loads(report.read_text())["test_cases"]
    return rows


def run_correctness(task_dir: Path, python: str = sys.executable) -> bool:
    runner = task_dir / "scripts" / "task_runner.py"
    proc = subprocess.run([python, str(runner), "correctness"],
                          cwd=str(task_dir), capture_output=True, text=True)
    return proc.returncode == 0


def collect(rows_per_repeat: Sequence[Sequence[Mapping[str, Any]]]
            ) -> tuple[dict[str, list[float]], list[str]]:
    """Per-route speedup samples, plus every reason the sweep is not usable."""
    problems: list[str] = []
    by_route: dict[str, list[float]] = {}
    expected = reference_routes()
    for i, rows in enumerate(rows_per_repeat):
        names = set()
        for row in rows:
            name = row.get("test_case_id") or row.get("name")
            speedup = row.get("speedup")
            if not name:
                problems.append(f"repeat {i}: a row carries no test_case_id")
                continue
            names.add(name)
            if not isinstance(speedup, (int, float)) or not speedup > 0:
                problems.append(f"repeat {i}: {name} has no positive speedup")
                continue
            if not (SPEEDUP_SANE_LO < speedup < SPEEDUP_SANE_HI):
                problems.append(
                    f"repeat {i}: {name} speedup {speedup} is outside "
                    f"({SPEEDUP_SANE_LO}, {SPEEDUP_SANE_HI}) -- that is a unit "
                    f"error, not a result (finding 132)")
                continue
            by_route.setdefault(name, []).append(float(speedup))
        missing = expected - names
        extra = names - expected
        if missing:
            problems.append(f"repeat {i}: missing routes {sorted(missing)}")
        if extra:
            problems.append(f"repeat {i}: unknown routes {sorted(extra)}")
    return by_route, problems


def build_table(by_route: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    return {route: floor_from_speedups(list(samples))
            for route, samples in sorted(by_route.items())}


def render_python(machine: str, table: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [f'MEASURED_NOISE_FLOOR_BY_MACHINE["{machine}"] = {{']
    for route, stats in table.items():
        note = "  # clamped to MIN_FLOOR" if stats["clamped_to_min"] else ""
        lines.append(f'    "{route}": {stats["floor"]:.4f},{note}')
    lines.append("}")
    return "\n".join(lines)


def attribution_problems(verdict: Mapping[str, Any], machine: str) -> list[str]:
    """Reasons this verdict does not describe `machine`'s box.

    A verdict is a reading of one GPU on one host. Nothing in the numbers says
    which one, so until this check existed the letter came entirely from the
    `--machine` argument: pointing `--from-json` at another epoch's saved
    verdict re-rendered its floors under any letter asked for, and the
    installed table's own provenance comment named the letter's registered
    host rather than the host that took the reading. The forgery documented
    itself as legitimate. That is finding (107) -- floors carried across a
    machine boundary on an argument -- reachable in a single command.

    So the sweep stamps the box, and the stamp is checked wherever a verdict
    becomes a table. Fail closed on a MISSING stamp too: a verdict written
    before this check cannot be attributed after the fact, and re-measuring is
    cheap next to admitting a wave against another machine's ruler.
    """
    host = verdict.get("host")
    if not host:
        return ["verdict records no host, so it cannot be attributed to any "
                "box; it predates the stamp. Re-measure on the target machine "
                "rather than trusting the --machine argument"]
    resolved = QRS.machine_for_host(host)
    if resolved is None:
        return [f"verdict was taken on {host}, which is registered to no "
                f"epoch; it cannot be installed as {machine}'s table"]
    if resolved != machine:
        return [f"verdict was taken on {host} (epoch {resolved}), but would be "
                f"installed as epoch {machine} "
                f"({QRS.MACHINE_HOSTNAME.get(machine)}). Floors do not pool "
                f"across a machine boundary"]
    return []


def sweep(task_dir: Path, repeats: int, python: str = sys.executable
          ) -> dict[str, Any]:
    """Run the whole sweep and return a verdict dict. Never raises for a
    measurement outcome -- only the caller decides the exit code."""
    host = socket.gethostname()
    stamp = {"host": host, "host_machine": QRS.machine_for_host(host)}
    hash_before = QSH.tree_hash(task_dir)
    if not run_correctness(task_dir, python):
        return {"ok": False, "stage": "correctness", **stamp,
                "problems": ["correctness failed; no timing is reportable"]}
    rows_per_repeat = []
    errors = []
    for _ in range(repeats):
        try:
            rows_per_repeat.append(run_suite(task_dir, python))
        except Exception as exc:  # a lost repeat is data, not a crash
            errors.append(str(exc))
    hash_after = QSH.tree_hash(task_dir)
    if hash_before != hash_after:
        return {"ok": False, "stage": "identity", **stamp,
                "problems": [f"source hash moved mid-sweep: {hash_before} -> "
                             f"{hash_after}; these are not same-variant repeats"]}
    by_route, problems = collect(rows_per_repeat)
    problems = errors + problems
    complete = len(rows_per_repeat)
    if complete < MIN_REPEATS:
        problems.append(f"{complete} complete repeats, need at least {MIN_REPEATS}")
    table = build_table(by_route)
    short = [r for r, s in table.items() if s["n"] < MIN_REPEATS]
    if short:
        problems.append(f"routes with fewer than {MIN_REPEATS} samples: {sorted(short)}")
    return {
        "ok": not problems,
        "stage": "sweep",
        **stamp,
        "source_hash": hash_before,
        "repeats_requested": repeats,
        "repeats_complete": complete,
        "routes": table,
        "problems": problems,
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", required=True, help="task directory (dense_bf16_gemm_fused)")
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    p.add_argument("--machine", default=QRS.CURRENT_MACHINE,
                   help="epoch letter the emitted table belongs to")
    p.add_argument("--out", help="write the full JSON verdict here")
    p.add_argument("--from-json", help="skip the GPU and re-render a saved verdict")
    args = p.parse_args(argv)

    if args.from_json:
        verdict = json.loads(Path(args.from_json).read_text())
    else:
        verdict = sweep(Path(args.task).resolve(), args.repeats)

    # Checked on both paths, not just --from-json. On the sweep path it is a
    # tautology today, and it is exactly what stops being a tautology after a
    # restore moves the container to a box the letter no longer names.
    mismatch = attribution_problems(verdict, args.machine)
    if mismatch:
        verdict = dict(verdict)
        verdict["ok"] = False
        verdict["problems"] = list(verdict.get("problems") or []) + mismatch

    if args.out:
        Path(args.out).write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")

    for problem in verdict.get("problems", []):
        print(f"REFUSED: {problem}", file=sys.stderr)

    if not verdict.get("ok"):
        print("no floor table emitted -- a guessed floor is worse than the "
              "fail-closed default it would replace", file=sys.stderr)
        return 3 if verdict.get("stage") in {"correctness", "identity"} else 4

    print(f"# epoch {args.machine}, source_hash {verdict['source_hash']}, "
          f"{verdict['repeats_complete']} complete repeats")
    print()
    print("# --- noise_floor_stats.py ---")
    print(render_python(args.machine, verdict["routes"]))
    print()
    print(f"# then remove \"{args.machine}\" from PROVISIONAL_MACHINES -- a "
          f"measured table that still reads as provisional is the same lie in "
          f"the other direction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
