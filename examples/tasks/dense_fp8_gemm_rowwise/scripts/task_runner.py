#!/usr/bin/env python3
"""Build, validate, benchmark, and profile the rowwise-scaled FP8 GEMM task.

Structurally a sibling of dense_bf16_gemm_fused/scripts/task_runner.py -- same eleven shapes,
same v2 paired timing, same write/independence probes -- with three differences that fp8 forces:

1. The inputs are QUANTISED, so `_make_inputs` produces fp8 tensors plus per-token and
   per-channel fp32 scales, and the candidate takes four arguments instead of two.
2. The oracle is a per-shape best-of over two libraries rather than one library call, so every
   performance row records WHICH library it was scored against.
3. Correctness is checked against the oracle on the suite, and separately against dequantised
   fp32 truth on a deliberately awkward shape (N not a multiple of the 64-wide block) so a
   transposition or a tail-masking bug cannot hide behind an oracle that shares the mistake.
"""

import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

TASK_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TASK_DIR.parents[2]
BUILD_DIR = TASK_DIR / "build"
BUILD_DIR.mkdir(exist_ok=True)
# Order matters and the obvious order is wrong. Two insert(0, ...) calls leave the LAST one at
# position 0, so inserting the repo copy second put it ahead of the task's own frozen harness --
# and PROVENANCE pins the task copy. The gate that catches a no-op kernel then depended on which
# file happened to win an import race. The task's pinned harness comes first.
sys.path.insert(0, str(REPO_ROOT / "e2e_workflow" / "scripts"))
sys.path.insert(0, str(TASK_DIR))

import harness_lib as harness

FP8 = torch.float8_e4m3fnuz
FP8_MAX = 240.0                      # e4m3fnuz finite max

CASES = [
    {"id": "decode_m2_square", "regime": "decode", "M": 2, "N": 4096, "K": 4096, "seed": 101},
    {"id": "decode_m8_up", "regime": "decode", "M": 8, "N": 11008, "K": 4096, "seed": 102},
    {"id": "decode_m16_square", "regime": "decode", "M": 16, "N": 4096, "K": 4096, "seed": 108},
    {"id": "decode_m32_down", "regime": "decode", "M": 32, "N": 4096, "K": 11008, "seed": 103},
    {"id": "decode_m64_square", "regime": "decode", "M": 64, "N": 8192, "K": 8192, "seed": 104},
    {"id": "decode_m96_up", "regime": "decode", "M": 96, "N": 11008, "K": 4096, "seed": 109},
    {"id": "prefill_m128_square", "regime": "prefill", "M": 128, "N": 4096, "K": 4096, "seed": 105},
    {"id": "prefill_m256_down", "regime": "prefill", "M": 256, "N": 4096, "K": 11008, "seed": 110},
    {"id": "prefill_m512_up", "regime": "prefill", "M": 512, "N": 11008, "K": 4096, "seed": 106},
    {"id": "prefill_m1024_down", "regime": "prefill", "M": 1024, "N": 4096, "K": 11008, "seed": 107},
    {"id": "prefill_m2048_square", "regime": "prefill", "M": 2048, "N": 4096, "K": 4096, "seed": 111},
]

# One per oracle route, so a broken route is visible before the suite runs.
ORACLE_SANITY = [
    {"id": "oracle_sanity_m8", "regime": "sanity", "M": 8, "N": 64, "K": 64, "seed": 17},
    {"id": "oracle_sanity_m64", "regime": "sanity", "M": 64, "N": 64, "K": 64, "seed": 18},
]
# N=48 is a multiple of the 16-wide fragment but NOT of the seed's 64-wide block, so the last
# wave of every workgroup falls entirely outside the matrix. A candidate that forgets to mask
# its B columns writes out of bounds here and nowhere in the suite.
LAYOUT_SANITY = {"id": "layout_sanity", "regime": "sanity", "M": 32, "N": 48, "K": 64, "seed": 19}

TOL = 2e-2


def _write_report(name, report):
    path = BUILD_DIR / f"{name}_report.json"
    harness.atomic_write_json(path, report)
    return path


def _load_ops():
    from gemm_wrapper import dense_fp8_gemm
    from oracle_wrapper import best_fp8_gemm

    return dense_fp8_gemm, best_fp8_gemm


def _make_inputs(case, seed_offset=0):
    """Per-token / per-channel quantisation, deterministic in the case seed.

    Both arms are handed the SAME fp8 bytes and the SAME scales, so the comparison measures the
    GEMM and not two different quantisations. Scales are flat (M,) and (N,); the oracle reshapes
    to the (M,1)/(1,N) its libraries want, the candidate indexes them directly.
    """
    gen = torch.Generator(device="cuda").manual_seed(case["seed"] + seed_offset)
    a = torch.randn(case["M"], case["K"], generator=gen, device="cuda") * 0.1
    b = torch.randn(case["N"], case["K"], generator=gen, device="cuda") * 0.1
    a_scale = (a.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12).float().contiguous()
    b_scale = (b.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12).float().contiguous()
    a8 = (a / a_scale.view(-1, 1)).to(FP8)
    b8 = (b / b_scale.view(-1, 1)).to(FP8)
    return a8, b8, a_scale, b_scale


def _quantize_rows(x):
    scale = (x.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12).float().contiguous()
    return (x / scale.view(-1, 1)).to(FP8), scale


def _make_measurement_slots(case, protocol):
    weight_bytes = int(case["N"]) * int(case["K"])
    count = harness.weight_rotation_slot_count(weight_bytes, protocol)
    specs = harness.semantic_weight_slot_specs(case["seed"], count, protocol)
    a_gen = torch.Generator(device="cuda").manual_seed(specs[0]["semantic"]["a_seed"])
    a = torch.randn(case["M"], case["K"], generator=a_gen, device="cuda") * 0.1
    a8, a_scale = _quantize_rows(a)
    slots = []
    for spec in specs:
        b_gen = torch.Generator(device="cuda").manual_seed(spec["semantic"]["b_seed"])
        b = torch.randn(case["N"], case["K"], generator=b_gen, device="cuda") * 0.1
        b8, b_scale = _quantize_rows(b)
        slots.append({**spec, "weight_bytes": weight_bytes,
                      "args": (a8, b8, a_scale, b_scale)})
    return slots


def _dequant_reference(a8, b8, a_scale, b_scale):
    """The truth the fp8 inputs actually encode, computed in fp32 outside any GEMM library."""
    a = a8.float() * a_scale.view(-1, 1)
    b = b8.float() * b_scale.view(-1, 1)
    return (a @ b.t()).to(torch.bfloat16)


def _geomean(values):
    values = [float(v) for v in values if v is not None and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def run_preflight(load_extension=True):
    from oracle_wrapper import oracle_available

    rocm_home = Path(os.environ.get("ROCM_HOME", "/opt/rocm"))
    oracle_ok, oracle_note = oracle_available()
    report = {
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "pytorch_rocm_arch": os.environ.get("PYTORCH_ROCM_ARCH"),
        "hipcc": shutil.which("hipcc"),
        "rocprofv3": shutil.which("rocprofv3"),
        "rocwmma_header": str(rocm_home / "include" / "rocwmma" / "rocwmma.hpp"),
        "oracle_note": oracle_note,
        "checks": {},
    }
    checks = report["checks"]
    checks["rocm_build"] = bool(torch.version.hip)
    checks["gpu_visible"] = bool(torch.cuda.is_available())
    checks["hipcc"] = bool(report["hipcc"])
    checks["rocwmma_header"] = Path(report["rocwmma_header"]).is_file()
    checks["fp8_dtype"] = hasattr(torch, "float8_e4m3fnuz")
    checks["oracle_libraries"] = oracle_ok
    if checks["gpu_visible"]:
        try:
            torch.empty(1, device="cuda")
            torch.cuda.synchronize()
            report["device_name"] = torch.cuda.get_device_name(0)
            report["device_count"] = torch.cuda.device_count()
            checks["allocation"] = True
        except Exception as exc:
            checks["allocation"] = False
            report["allocation_error"] = repr(exc)
    else:
        checks["allocation"] = False

    # Validate the DENOMINATOR before anything else. Both oracle routes are exercised, against a
    # reference that shares no code with either library, and the first call also pays aiter's
    # module import so it never lands inside a timed region.
    if load_extension and all(checks.values()):
        try:
            _, oracle = _load_ops()
            routes = {}
            worst = 0.0
            from oracle_wrapper import oracle_route
            for case in ORACLE_SANITY:
                args = _make_inputs(case)
                out = oracle(*args)
                ref = _dequant_reference(*args)
                ok, err = harness.correct(out, ref, TOL)
                routes[case["id"]] = {"route": oracle_route(case["M"], case["N"], case["K"]),
                                      "correct": ok, "max_rel_err": err}
                worst = max(worst, err)
                checks[f"oracle_smoke_{case['id']}"] = bool(ok)
            report["oracle_sanity"] = routes
            report["oracle_max_rel_err"] = worst
        except Exception as exc:
            checks["oracle_smoke"] = False
            report["oracle_error"] = repr(exc)
    ok = all(checks.values())
    report["status"] = "ok" if ok else "fail"
    _write_report("preflight", report)
    return ok, report


def run_compile():
    ok, preflight = run_preflight(load_extension=True)
    report = {"status": "ok" if ok else "fail", "preflight": preflight}
    _write_report("compile", report)
    return ok, report


def _negative_tests(candidate):
    tests = []

    def expect_error(name, fn):
        try:
            fn()
        except Exception as exc:
            tests.append({"name": name, "passed": True, "error": type(exc).__name__})
        else:
            tests.append({"name": name, "passed": False, "error": None})

    a8, b8, sa, sb = _make_inputs({"M": 32, "N": 64, "K": 64, "seed": 23})
    expect_error("bf16_inputs", lambda: candidate(a8.to(torch.bfloat16), b8.to(torch.bfloat16), sa, sb))
    expect_error("noncontiguous", lambda: candidate(a8.t(), b8, sa, sb))
    expect_error("rank", lambda: candidate(a8.unsqueeze(0), b8, sa, sb))
    expect_error("k_mismatch", lambda: candidate(a8, b8[:, :-16].contiguous(), sa, sb))
    expect_error("cpu", lambda: candidate(a8.cpu(), b8.cpu(), sa.cpu(), sb.cpu()))
    expect_error("a_scale_length", lambda: candidate(a8, b8, sa[:-1].contiguous(), sb))
    expect_error("b_scale_length", lambda: candidate(a8, b8, sa, sb[:-1].contiguous()))
    expect_error("scale_dtype", lambda: candidate(a8, b8, sa.to(torch.float16), sb))
    return tests


def run_correctness():
    from oracle_wrapper import oracle_route

    candidate, oracle = _load_ops()
    results = []
    all_ok = True

    # 1. Is the oracle itself right? Both routes, against library-free fp32 truth.
    for case in ORACLE_SANITY:
        args = _make_inputs(case)
        ok, err = harness.correct(oracle(*args), _dequant_reference(*args), TOL)
        results.append({"case": case["id"], "draw": 0, "correct": ok, "max_rel_err": err,
                        "oracle": "fp32_dequant_truth",
                        "route": oracle_route(case["M"], case["N"], case["K"])})
        all_ok = all_ok and ok

    # 2. Is the candidate right on a shape whose tails the suite never exercises? Compared
    #    against fp32 truth, not the oracle, so a shared mistake cannot pass.
    args = _make_inputs(LAYOUT_SANITY)
    ok, err = harness.correct(candidate(*args), _dequant_reference(*args), TOL)
    results.append({"case": LAYOUT_SANITY["id"], "draw": 0, "correct": ok, "max_rel_err": err,
                    "oracle": "fp32_dequant_truth"})
    all_ok = all_ok and ok

    # 3. The suite: candidate against the live oracle, five independent draws per shape.
    probed = 0   # shapes the probe reached a VERDICT on -- pass AND fail both count as coverage
    for case in CASES:
        for draw in range(5):
            args = _make_inputs(case, seed_offset=draw * 1009)
            ref = oracle(*args).detach().clone()
            out = candidate(*args)
            ok, err = harness.correct(out, ref, TOL)
            results.append({"case": case["id"], "draw": draw, "correct": ok,
                            "max_rel_err": err, "oracle": "live_" +
                            oracle_route(case["M"], case["N"], case["K"])})
            all_ok = all_ok and ok
        # The poison probe, once per SHAPE. Running it only on the smallest case cannot see a
        # kernel that writes some shapes and skips others: an arch guard is all-or-nothing, but a
        # wrong tile bound or a masked epilogue is per-shape. The five draws above cannot catch
        # either one -- they compare against an oracle that was freed moments earlier, so the
        # allocator can hand its buffer back and a kernel that computes nothing scores 0.0.
        args = _make_inputs(case, seed_offset=9001)
        ok, status, frac = harness.assert_writes_output(lambda a_: candidate(*a_), args)
        entry = {"case": case["id"], "check": "output_write_probe", "correct": ok,
                 "status": status, "untouched_fraction": frac,
                 "note": harness._write_probe_note(status, frac)}
        if status in ("never_written", "partially_written"):
            all_ok = False
            probed += 1
        elif status == "ok":
            probed += 1
        else:
            # Inconclusive or raised: NOT a pass. correct=None so a consumer folding these with
            # all() fails closed instead of reading "not tested" as "tested and fine".
            entry["correct"] = None
            entry["untested"] = True
        results.append(entry)

    # If the allocator never handed the poisoned block back on ANY shape, the suite ran without
    # write coverage at all. That is a hole, and it must be visible rather than inferred.
    results.append({"case": "write_probe_coverage", "check": "output_write_probe",
                    "correct": None if probed == 0 else True,
                    "untested": probed == 0,
                    "shapes_conclusively_probed": probed, "shapes_total": len(CASES),
                    "note": "" if probed else
                            "no_write_coverage: the poison probe was inconclusive on every shape, "
                            "so nothing in this run tested whether the kernel writes its output."})

    case = min(CASES, key=lambda c: c["M"] * c["N"] * c["K"])
    args_a = _make_inputs(case, 7001)
    args_b = _make_inputs(case, 8009)
    independent, reason = harness.assert_independent_outputs(lambda args: candidate(*args),
                                                            args_a, args_b)
    results.append({"case": "output_independence", "correct": independent, "note": reason})
    all_ok = all_ok and independent

    negative = _negative_tests(candidate)
    all_ok = all_ok and all(item["passed"] for item in negative)
    report = {"status": "ok" if all_ok else "fail", "tol": TOL, "random_draws": 5,
              "results": results, "negative_tests": negative}
    _write_report("correctness", report)
    return all_ok, report


def _measurement_arms(candidate, backends):
    arms = {"candidate": lambda slot: candidate(*slot["args"])}
    for name, fn in backends.items():
        arms[name] = lambda slot, fn=fn: fn(*slot["args"])
    return arms


def _validate_measurement_receipt(receipt):
    if not all(receipt["invariants"].values()):
        raise RuntimeError(f"measurement invariant failed: {receipt['invariants']}")
    bad_timers = sorted({sample["timer"] for sample in receipt["samples"]
                         if sample["timer"] != "cuda_event"})
    if bad_timers:
        raise RuntimeError(f"device-event timing unavailable: {bad_timers}")
    if not all(sample["primed"] for sample in receipt["samples"]):
        raise RuntimeError("one or more paired samples were not dispatch-primed")
    stale = [name for name, row in receipt["output_lifecycle"].items()
             if row["observed_addresses"] >= 2 and not row["addresses_rotated"]]
    if stale:
        raise RuntimeError(f"outputs reused one address despite the live ring: {stale}")


def _metric_summary(receipt, route, flops):
    candidate_ms = receipt["medians_ms"]["candidate"]
    oracle_ms = receipt["medians_ms"][route]
    return {
        "metric_id": receipt["metric_id"],
        "unit": receipt["metric"]["unit"],
        "inner": receipt["metric"]["inner"],
        "candidate_ms_per_call": candidate_ms,
        "oracle_ms_per_call": oracle_ms,
        "speedup": oracle_ms / candidate_ms,
        "candidate_tflops": flops / (candidate_ms * 1e9),
        "oracle_tflops": flops / (oracle_ms * 1e9),
        "all_backend_medians_ms": receipt["medians_ms"],
    }


def run_performance(selected_case=None, protocol_name=None):
    from oracle_wrapper import oracle_backends, oracle_route, validate_oracle_routes

    protocol = harness.measurement_protocol(protocol_name)
    trusted, route_errors = validate_oracle_routes(protocol["hash"], CASES)
    if not trusted:
        raise RuntimeError("oracle route manifest is not trusted for this protocol: "
                           + "; ".join(route_errors)
                           + ". Run oracle-bakeoff under the GPU lock first.")
    candidate, _ = _load_ops()
    backends = oracle_backends()
    cases = [c for c in CASES if selected_case in (None, c["id"])]
    if not cases:
        raise ValueError(f"unknown case: {selected_case}")
    rows = []
    for case in cases:
        slots = _make_measurement_slots(case, protocol)
        arms = _measurement_arms(candidate, backends)
        latency = harness.measure_paired_arms(
            arms, slots, protocol=protocol, metric="single_call_latency")
        throughput = harness.measure_paired_arms(
            arms, slots, protocol=protocol, metric="sustained_throughput")
        _validate_measurement_receipt(latency)
        _validate_measurement_receipt(throughput)
        route = oracle_route(case["M"], case["N"], case["K"])
        flops = 2.0 * case["M"] * case["N"] * case["K"]
        latency_summary = _metric_summary(latency, route, flops)
        throughput_summary = _metric_summary(throughput, route, flops)
        legacy = harness.legacy_timing_aliases(
            latency, candidate_arm="candidate", baseline_arm=route)
        row = {
            "test_case_id": case["id"],
            "regime": case["regime"],
            "M": case["M"], "N": case["N"], "K": case["K"],
            "baseline_route": route,
            "baseline_tflops": flops / (legacy["baseline_ms"] * 1e9),
            "candidate_tflops": flops / (legacy["candidate_ms"] * 1e9),
            "score_metric": "single_call_latency",
            "single_call_latency": latency_summary,
            "sustained_throughput": throughput_summary,
            "measurement_receipts": {
                "single_call_latency": latency,
                "sustained_throughput": throughput,
            },
            **legacy,
        }
        rows.append(row)
        print(f"Perf: {legacy['candidate_ms']:.6f} ms ({case['id']})")
    timing_receipt = {
        **harness.legacy_timing_receipt(rows),
        "schema": harness.MEASUREMENT_RECEIPT_SCHEMA,
        "protocol_hash": protocol["hash"],
        "raw_receipts_embedded_per_case": True,
    }
    report = {
        "status": "ok",
        "protocol": protocol,
        "protocol_hash": protocol["hash"],
        "score_metric": "single_call_latency",
        "non_score_metric": "sustained_throughput",
        "metric_separation": "latency inner=1; throughput inner>1; no mixed aggregate",
        "baseline": "frozen per-shape best-of aiter CK/hipBLASLt rowwise",
        "test_cases": rows,
        "geomean_speedup": _geomean([r["single_call_latency"]["speedup"] for r in rows]),
        "decode_geomean_speedup": _geomean([
            r["single_call_latency"]["speedup"] for r in rows if r["regime"] == "decode"]),
        "prefill_geomean_speedup": _geomean([
            r["single_call_latency"]["speedup"] for r in rows if r["regime"] == "prefill"]),
        "sustained_geomean_speedup": _geomean([
            r["sustained_throughput"]["speedup"] for r in rows]),
        "timing_receipt": timing_receipt,
    }
    print(f"GEAK_TIMING_RECEIPT: {json.dumps(timing_receipt)}")
    _write_report("performance", report)
    return True, report


def run_oracle_bakeoff(selected_case=None, protocol_name=None):
    """Re-select and atomically freeze the FP8 denominator under the requested v2 protocol."""
    from oracle_wrapper import oracle_backend_metadata, oracle_backends
    from route_manifest import SCHEMA, write_route_manifest

    protocol = harness.measurement_protocol(protocol_name)
    cases = [c for c in CASES if selected_case in (None, c["id"])]
    if selected_case is not None and not cases:
        raise ValueError(f"unknown case: {selected_case}")
    if selected_case is not None:
        raise ValueError("a frozen oracle manifest requires all suite cases; omit --case")
    backends = oracle_backends()
    routes = {}
    receipts = {}
    metadata = {name: oracle_backend_metadata(name) for name in backends}
    for case in cases:
        slots = _make_measurement_slots(case, protocol)
        first_args = slots[0]["args"]
        ref = _dequant_reference(*first_args)
        correctness = {}
        for name, fn in backends.items():
            out = fn(*first_args)
            ok, err = harness.correct(out, ref, TOL)
            if not ok:
                raise RuntimeError(f"{name} failed correctness for {case['id']}: {err}")
            correctness[name] = {"correct": True, "max_rel_err": err}
        arms = {name: (lambda slot, fn=fn: fn(*slot["args"]))
                for name, fn in backends.items()}
        receipt = harness.measure_paired_arms(
            arms, slots, protocol=protocol, metric="single_call_latency")
        _validate_measurement_receipt(receipt)
        winner = min(backends, key=lambda name: receipt["medians_ms"][name])
        routes[case["id"]] = {
            "M": case["M"], "N": case["N"], "K": case["K"],
            "backend": winner,
            "correctness": correctness,
            "backend_measurements": {
                name: {
                    "median_ms": receipt["medians_ms"][name],
                    "samples_ms": [sample["ms"] for sample in receipt["samples"]
                                   if sample["arm"] == name],
                    "metadata": metadata[name],
                }
                for name in backends
            },
        }
        receipts[case["id"]] = receipt
    manifest = {
        "schema": SCHEMA,
        "task": "dense_fp8_gemm_rowwise",
        "frozen": True,
        "selection_state": "baked",
        "selection_metric": "single_call_latency",
        "protocol_hash": protocol["hash"],
        "protocol": protocol,
        "seed": protocol["seed"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(0),
        },
        "routes": routes,
    }
    path = TASK_DIR / "oracle_routes.v2.json"
    write_route_manifest(path, manifest)
    report = {"status": "ok", "manifest": str(path), "routes": routes,
              "protocol_hash": protocol["hash"], "raw_receipts": receipts}
    _write_report("oracle_bakeoff", report)
    return True, report


def run_profile(case_id, implementation, iterations):
    candidate, baseline = _load_ops()
    case = next((c for c in CASES if c["id"] == case_id), None)
    if case is None:
        raise ValueError(f"unknown case: {case_id}")
    call_impl = baseline if implementation == "baseline" else candidate
    args = _make_inputs(case)
    for _ in range(10):
        call_impl(*args)
    torch.cuda.synchronize()
    for _ in range(iterations):
        call_impl(*args)
    torch.cuda.synchronize()
    return True, {"status": "ok", "case": case_id, "implementation": implementation,
                  "iterations": iterations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "compile", "correctness", "performance",
                                         "oracle-bakeoff", "profile"])
    parser.add_argument("--case")
    parser.add_argument("--protocol", choices=["legacy_write_flush", "read_evict", "warm",
                                                "semantic_weight_rotating"])
    parser.add_argument("--confirm-oracle-bakeoff", action="store_true")
    parser.add_argument("--impl", choices=["baseline", "candidate"], default="candidate")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    try:
        if args.mode == "preflight":
            ok, report = run_preflight(load_extension=True)
        elif args.mode == "compile":
            ok, report = run_compile()
        elif args.mode == "correctness":
            ok, report = run_correctness()
        elif args.mode == "performance":
            ok, report = run_performance(args.case, args.protocol)
        elif args.mode == "oracle-bakeoff":
            if not args.confirm_oracle_bakeoff:
                parser.error("oracle-bakeoff requires --confirm-oracle-bakeoff")
            ok, report = run_oracle_bakeoff(args.case, args.protocol)
        else:
            if not args.case:
                parser.error("profile requires --case")
            ok, report = run_profile(args.case, args.impl, args.iterations)
        print(f"{args.mode.capitalize()}: {'PASS' if ok else 'FAIL'}")
        if args.mode in ("profile", "oracle-bakeoff"):
            print(json.dumps({"status": report.get("status"),
                              "manifest": report.get("manifest")}))
        raise SystemExit(0 if ok else 1)
    except Exception as exc:
        report = {"status": "fail", "error": repr(exc)}
        _write_report(args.mode.replace("-", "_"), report)
        print(f"{args.mode.capitalize()}: FAIL\nError: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
