#!/usr/bin/env python3
"""Build, validate, benchmark, and profile the BF16 GEMM task."""

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import torch

TASK_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TASK_DIR.parents[2]
BUILD_DIR = TASK_DIR / "build"
BUILD_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(REPO_ROOT / "e2e_workflow" / "scripts"))

import harness_lib as harness

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
SANITY_CASE = {"id": "layout_sanity", "regime": "sanity", "M": 3, "N": 5, "K": 7, "seed": 17}
TOL = 2e-2


def _write_report(name, report):
    path = BUILD_DIR / f"{name}_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def _load_ops():
    from gemm_wrapper import dense_bf16_gemm, rocblas_bf16_gemm

    return dense_bf16_gemm, rocblas_bf16_gemm


def _make_inputs(case, seed_offset=0):
    gen = torch.Generator(device="cuda").manual_seed(case["seed"] + seed_offset)
    a = (torch.randn(case["M"], case["K"], generator=gen, device="cuda") * 0.1).to(torch.bfloat16)
    b = (torch.randn(case["N"], case["K"], generator=gen, device="cuda") * 0.1).to(torch.bfloat16)
    return a, b


def _geomean(values):
    values = [float(v) for v in values if v is not None and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def run_preflight(load_extension=True):
    rocm_home = Path(os.environ.get("ROCM_HOME", "/opt/rocm"))
    report = {
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "pytorch_rocm_arch": os.environ.get("PYTORCH_ROCM_ARCH"),
        "hipcc": shutil.which("hipcc"),
        "rocprofv3": shutil.which("rocprofv3"),
        "rocblas_header": str(rocm_home / "include" / "rocblas" / "rocblas.h"),
        "rocblas_library": str(rocm_home / "lib" / "librocblas.so"),
        "checks": {},
    }
    checks = report["checks"]
    checks["rocm_build"] = bool(torch.version.hip)
    checks["gpu_visible"] = bool(torch.cuda.is_available())
    checks["hipcc"] = bool(report["hipcc"])
    checks["rocblas_header"] = Path(report["rocblas_header"]).is_file()
    checks["rocblas_library"] = Path(report["rocblas_library"]).is_file()
    if checks["gpu_visible"]:
        try:
            torch.empty(1, device="cuda")
            torch.cuda.synchronize()
            report["device_name"] = torch.cuda.get_device_name(0)
            report["device_count"] = torch.cuda.device_count()
            report["device_capability"] = list(torch.cuda.get_device_capability(0))
            checks["allocation"] = True
        except Exception as exc:
            checks["allocation"] = False
            report["allocation_error"] = repr(exc)
    else:
        checks["allocation"] = False
    if load_extension and all(checks.values()):
        try:
            _, rocblas = _load_ops()
            a, b = _make_inputs(SANITY_CASE)
            out = rocblas(a, b)
            ref = (a.float() @ b.float().t()).to(torch.bfloat16)
            ok, err = harness.correct(out, ref, TOL)
            checks["direct_rocblas_smoke"] = bool(ok)
            report["direct_rocblas_max_rel_err"] = err
        except Exception as exc:
            checks["direct_rocblas_smoke"] = False
            report["direct_rocblas_error"] = repr(exc)
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

    a = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(6, 8, device="cuda", dtype=torch.bfloat16)
    expect_error("fp32", lambda: candidate(a.float(), b))
    expect_error("noncontiguous", lambda: candidate(a.t(), b))
    expect_error("rank", lambda: candidate(a.unsqueeze(0), b))
    expect_error("k_mismatch", lambda: candidate(a, b[:, :-1].contiguous()))
    expect_error("cpu", lambda: candidate(a.cpu(), b.cpu()))
    return tests


def run_correctness():
    candidate, baseline = _load_ops()
    results = []
    all_ok = True

    a, b = _make_inputs(SANITY_CASE)
    base = baseline(a, b)
    fp32_ref = (a.float() @ b.float().t()).to(torch.bfloat16)
    ok, err = harness.correct(base, fp32_ref, TOL)
    results.append({"case": SANITY_CASE["id"], "draw": 0, "correct": ok, "max_rel_err": err,
                    "oracle": "fp32_layout_sanity"})
    all_ok = all_ok and ok

    for case in CASES:
        for draw in range(5):
            a, b = _make_inputs(case, seed_offset=draw * 1009)
            ref = baseline(a, b).detach().clone()
            out = candidate(a, b)
            ok, err = harness.correct(out, ref, TOL)
            results.append({"case": case["id"], "draw": draw, "correct": ok,
                            "max_rel_err": err, "oracle": "live_rocblas_gemm_ex"})
            all_ok = all_ok and ok

    case = min(CASES, key=lambda c: c["M"] * c["N"] * c["K"])
    args_a = _make_inputs(case, 7001)
    args_b = _make_inputs(case, 8009)
    independent, reason = harness.assert_independent_outputs(lambda args: candidate(*args), args_a, args_b)
    results.append({"case": "output_independence", "correct": independent, "note": reason})
    all_ok = all_ok and independent

    negative = _negative_tests(candidate)
    all_ok = all_ok and all(item["passed"] for item in negative)
    report = {"status": "ok" if all_ok else "fail", "tol": TOL, "random_draws": 5,
              "results": results, "negative_tests": negative}
    _write_report("correctness", report)
    return all_ok, report


def _time(call):
    detail = harness.time_op(call, warmup=10, repeats=50, inner=1, flush_cache=True, detail=True)
    if detail is None or detail.get("timer") != "cuda_event":
        raise RuntimeError(f"device-event timing unavailable: {detail}")
    return detail


def _timing_leg(detail):
    return {"primed": detail.get("primed"), "host_ms": detail.get("host_ms")}


def _timing_receipt(rows):
    cases = {
        row["test_case_id"]: {
            "baseline": _timing_leg(row["baseline_timing"]),
            "current": _timing_leg(row["candidate_timing"]),
        }
        for row in rows
    }
    legs = [leg for case in cases.values() for leg in case.values()]
    return {
        "all_primed": all(leg["primed"] is True for leg in legs),
        "timer_unprimed": any(leg["primed"] is None for leg in legs),
        "cases": cases,
    }


def run_performance(selected_case=None):
    candidate, baseline = _load_ops()
    cases = [c for c in CASES if selected_case in (None, c["id"])]
    if not cases:
        raise ValueError(f"unknown case: {selected_case}")
    rows = []
    for case in cases:
        a, b = _make_inputs(case)
        baseline(a, b)
        candidate(a, b)
        torch.cuda.synchronize()
        base_timing = _time(lambda: baseline(a, b))
        candidate_timing = _time(lambda: candidate(a, b))
        base_ms = base_timing["ms"]
        candidate_ms = candidate_timing["ms"]
        flops = 2.0 * case["M"] * case["N"] * case["K"]
        row = {
            "test_case_id": case["id"],
            "regime": case["regime"],
            "M": case["M"], "N": case["N"], "K": case["K"],
            "baseline_ms": base_ms,
            "candidate_ms": candidate_ms,
            "execution_time_ms": candidate_ms,
            "speedup": base_ms / candidate_ms,
            "baseline_tflops": flops / (base_ms * 1e9),
            "candidate_tflops": flops / (candidate_ms * 1e9),
            "baseline_timing": base_timing,
            "candidate_timing": candidate_timing,
        }
        rows.append(row)
        print(f"Perf: {candidate_ms:.6f} ms ({case['id']})")
    timing_receipt = _timing_receipt(rows)
    report = {
        "status": "ok",
        "baseline": "direct rocblas_gemm_ex",
        "test_cases": rows,
        "geomean_speedup": _geomean([r["speedup"] for r in rows]),
        "decode_geomean_speedup": _geomean([r["speedup"] for r in rows if r["regime"] == "decode"]),
        "prefill_geomean_speedup": _geomean([r["speedup"] for r in rows if r["regime"] == "prefill"]),
        "timing_receipt": timing_receipt,
    }
    print(f"GEAK_TIMING_RECEIPT: {json.dumps(timing_receipt)}")
    _write_report("performance", report)
    return True, report


def run_profile(case_id, implementation, iterations):
    candidate, baseline = _load_ops()
    case = next((c for c in CASES if c["id"] == case_id), None)
    if case is None:
        raise ValueError(f"unknown case: {case_id}")
    call_impl = baseline if implementation == "baseline" else candidate
    a, b = _make_inputs(case)
    for _ in range(10):
        call_impl(a, b)
    torch.cuda.synchronize()
    for _ in range(iterations):
        call_impl(a, b)
    torch.cuda.synchronize()
    return True, {"status": "ok", "case": case_id, "implementation": implementation,
                  "iterations": iterations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "compile", "correctness", "performance", "profile"])
    parser.add_argument("--case")
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
            ok, report = run_performance(args.case)
        else:
            if not args.case:
                parser.error("profile requires --case")
            ok, report = run_profile(args.case, args.impl, args.iterations)
        print(f"{args.mode.capitalize()}: {'PASS' if ok else 'FAIL'}")
        if args.mode == "profile":
            print(json.dumps(report))
        raise SystemExit(0 if ok else 1)
    except Exception as exc:
        report = {"status": "fail", "error": repr(exc)}
        _write_report(args.mode, report)
        print(f"{args.mode.capitalize()}: FAIL\nError: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
