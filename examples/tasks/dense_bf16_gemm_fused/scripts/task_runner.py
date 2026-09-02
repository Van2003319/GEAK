#!/usr/bin/env python3
"""Build, validate, benchmark, and profile the BF16 GEMM task."""

import argparse
import hashlib
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
# file happened to win an import race, and on the repo copy being dirty in the working tree: a
# clean checkout, a stash, or honouring the pin all silently removed it. The task's pinned harness
# comes first now, and it carries the write probe itself (see PROVENANCE.frozen_harness).
sys.path.insert(0, str(REPO_ROOT / "e2e_workflow" / "scripts"))
sys.path.insert(0, str(TASK_DIR))

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
    harness.atomic_write_json(path, report)
    return path


def _load_ops():
    from gemm_wrapper import dense_bf16_gemm
    from oracle_wrapper import best_bf16_gemm

    return dense_bf16_gemm, best_bf16_gemm


def _make_inputs(case, seed_offset=0):
    gen = torch.Generator(device="cuda").manual_seed(case["seed"] + seed_offset)
    a = (torch.randn(case["M"], case["K"], generator=gen, device="cuda") * 0.1).to(torch.bfloat16)
    b = (torch.randn(case["N"], case["K"], generator=gen, device="cuda") * 0.1).to(torch.bfloat16)
    return a, b


def _make_measurement_slots(case, protocol, count_override=None):
    weight_bytes = int(case["N"]) * int(case["K"]) * 2
    count = (harness.weight_rotation_slot_count(weight_bytes, protocol)
             if count_override is None else int(count_override))
    specs = harness.semantic_weight_slot_specs(case["seed"], count, protocol)
    a_gen = torch.Generator(device="cuda").manual_seed(specs[0]["semantic"]["a_seed"])
    a = (torch.randn(case["M"], case["K"], generator=a_gen, device="cuda") * 0.1).to(torch.bfloat16)
    slots = []
    for spec in specs:
        b_gen = torch.Generator(device="cuda").manual_seed(spec["semantic"]["b_seed"])
        b = (torch.randn(case["N"], case["K"], generator=b_gen, device="cuda") * 0.1).to(torch.bfloat16)
        slots.append({**spec, "weight_bytes": weight_bytes, "args": (a, b)})
    return slots


def _geomean(values):
    values = [float(v) for v in values if v is not None and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def run_preflight(load_extension=True):
    from oracle_wrapper import oracle_available

    rocm_home = Path(os.environ.get("ROCM_HOME", "/opt/rocm"))
    report = {
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "pytorch_rocm_arch": os.environ.get("PYTORCH_ROCM_ARCH"),
        "hipcc": shutil.which("hipcc"),
        "rocprofv3": shutil.which("rocprofv3"),
        "rocblas_library": str(rocm_home / "lib" / "librocblas.so"),
        "hipblaslt_library": str(rocm_home / "lib" / "libhipblaslt.so"),
        "checks": {},
    }
    checks = report["checks"]
    checks["rocm_build"] = bool(torch.version.hip)
    checks["gpu_visible"] = bool(torch.cuda.is_available())
    checks["hipcc"] = bool(report["hipcc"])
    checks["rocblas_library"] = Path(report["rocblas_library"]).is_file()
    checks["hipblaslt_library"] = Path(report["hipblaslt_library"]).is_file()
    available, note = oracle_available() if load_extension else (True, "not loaded")
    checks["direct_oracle_extension"] = available
    report["oracle_extension"] = note
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
            from oracle_wrapper import (hipblaslt_bf16_gemm, oracle_backend_metadata,
                                        rocblas_bf16_gemm)
            a, b = _make_inputs(SANITY_CASE)
            ref = (a.float() @ b.float().t()).to(torch.bfloat16)
            backend_rows = {}
            for name, fn in (("direct_rocblas", rocblas_bf16_gemm),
                             ("direct_hipblaslt", lambda x, y: hipblaslt_bf16_gemm(x, y, -1))):
                out = fn(a, b)
                ok, err = harness.correct(out, ref, TOL)
                checks[f"{name}_smoke"] = bool(ok)
                metadata = oracle_backend_metadata(a, b, name, frozen_algorithm=False)
                backend_rows[name] = {"correct": ok, "max_rel_err": err,
                                      "metadata": metadata}
                if name == "direct_hipblaslt" and ok:
                    frozen_out = hipblaslt_bf16_gemm(a, b, int(metadata["algo_index"]))
                    frozen_ok, frozen_err = harness.correct(frozen_out, ref, TOL)
                    checks["direct_hipblaslt_explicit_algo_replay"] = bool(frozen_ok)
                    backend_rows[name]["explicit_algo_replay_max_rel_err"] = frozen_err
            report["oracle_backends"] = backend_rows
        except Exception as exc:
            checks["oracle_backend_smoke"] = False
            report["oracle_backend_error"] = repr(exc)
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

    from oracle_wrapper import hipblaslt_bf16_gemm, oracle_route, rocblas_bf16_gemm

    probed = 0   # shapes the probe reached a VERDICT on -- pass AND fail both count as coverage
    for case in CASES:
        # Correctness-screen the newly added direct hipBLASLt backend on every scored shape,
        # independently of which route the current manifest selects.
        oracle_args = _make_inputs(case, seed_offset=31337)
        rocblas_ref = rocblas_bf16_gemm(*oracle_args).detach().clone()
        hipblaslt_out = hipblaslt_bf16_gemm(*oracle_args, algo_index=-1)
        oracle_ok, oracle_err = harness.correct(hipblaslt_out, rocblas_ref, TOL)
        results.append({"case": case["id"], "draw": "oracle_backend_parity",
                        "correct": oracle_ok, "max_rel_err": oracle_err,
                        "oracle": "direct_hipblaslt_vs_direct_rocblas"})
        all_ok = all_ok and oracle_ok
        for draw in range(5):
            a, b = _make_inputs(case, seed_offset=draw * 1009)
            ref = baseline(a, b).detach().clone()
            out = candidate(a, b)
            ok, err = harness.correct(out, ref, TOL)
            results.append({"case": case["id"], "draw": draw, "correct": ok,
                            "max_rel_err": err, "oracle": "live_" +
                            oracle_route(case["M"], case["N"], case["K"])})
            all_ok = all_ok and ok
        # The poison probe, once per SHAPE. Running it only on the smallest case (as the
        # output_independence check below does) cannot see a kernel that writes some shapes and
        # skips others: an arch guard is all-or-nothing, but a wrong tile bound or a masked
        # epilogue is per-shape, and that is exactly what "partially_written" means. The five
        # draws above cannot catch either one -- they compare against an oracle that was freed
        # moments earlier, so the allocator can hand its buffer back and a kernel that computes
        # nothing scores max_rel_err 0.0.
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
            # all() fails closed instead of reading "not tested" as "tested and fine" (54/55).
            entry["correct"] = None
            entry["untested"] = True
        results.append(entry)

    # If the allocator never handed the poisoned block back on ANY shape, the suite ran without
    # write coverage at all. That is a hole, and it must be visible rather than inferred from the
    # absence of a failure.
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
    independent, reason = harness.assert_independent_outputs(lambda args: candidate(*args), args_a, args_b)
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


def run_performance(selected_case=None, protocol_name=None, score_metric_id=None):
    from oracle_wrapper import oracle_backends, oracle_route, validate_oracle_routes

    protocol = harness.measurement_protocol(protocol_name)
    score_metric = harness.score_metric_config(score_metric_id, protocol=protocol)
    os.environ["GEAK_SCORE_METRIC"] = score_metric["id"]
    secondary_metric = harness.secondary_score_metric(score_metric, protocol=protocol)
    trusted, route_errors = validate_oracle_routes(
        protocol["hash"], score_metric, CASES)
    if not trusted:
        raise RuntimeError("oracle route manifest is not trusted for this protocol/score metric: "
                           + "; ".join(route_errors)
                           + ". Run oracle-bakeoff under the GPU lock first.")
    candidate, _ = _load_ops()
    backends = oracle_backends(frozen_algorithms=True)
    cases = [c for c in CASES if selected_case in (None, c["id"])]
    if not cases:
        raise ValueError(f"unknown case: {selected_case}")
    rows = []
    for case in cases:
        slots = _make_measurement_slots(case, protocol)
        arms = _measurement_arms(candidate, backends)
        receipts = {}
        for metric in (score_metric, secondary_metric):
            receipt = harness.measure_paired_arms(
                arms, slots, protocol=protocol, metric=metric["id"])
            _validate_measurement_receipt(receipt)
            if (receipt.get("metric_id") != metric["id"]
                    or receipt.get("metric", {}).get("inner") != metric["inner"]):
                raise RuntimeError("measurement receipt score metric/inner drift")
            receipts[metric["id"]] = receipt
        route = oracle_route(case["M"], case["N"], case["K"])
        flops = 2.0 * case["M"] * case["N"] * case["K"]
        summaries = {metric_id: _metric_summary(receipt, route, flops)
                     for metric_id, receipt in receipts.items()}
        score_receipt = receipts[score_metric["id"]]
        score_aliases = harness.score_timing_aliases(
            score_receipt, candidate_arm="candidate", baseline_arm=route)
        row = {
            "test_case_id": case["id"],
            "regime": case["regime"],
            "M": case["M"], "N": case["N"], "K": case["K"],
            "baseline_route": route,
            "baseline_tflops": flops / (score_aliases["baseline_ms"] * 1e9),
            "candidate_tflops": flops / (score_aliases["candidate_ms"] * 1e9),
            "score_metric": dict(score_metric),
            "secondary_metric": dict(secondary_metric),
            "single_call_latency": summaries["single_call_latency"],
            "sustained_throughput": summaries["sustained_throughput"],
            "measurement_receipts": receipts,
            **score_aliases,
        }
        if score_metric["id"] == "single_call_latency":
            row["legacy_metric_source"] = "single_call_latency"
        rows.append(row)
        print(f"Perf[{score_metric['id']}]: {score_aliases['candidate_ms']:.6f} ms/call "
              f"({case['id']})")
    timing_receipt = {
        **harness.score_timing_receipt(rows, score_metric),
        "schema": harness.MEASUREMENT_RECEIPT_SCHEMA,
        "protocol_hash": protocol["hash"],
        "raw_receipts_embedded_per_case": True,
    }
    if score_metric["id"] == "single_call_latency":
        timing_receipt["legacy_metric_source"] = "single_call_latency"
    report = {
        "status": "ok",
        "protocol": protocol,
        "protocol_hash": protocol["hash"],
        "score_metric": dict(score_metric),
        "score_metric_id": score_metric["id"],
        "secondary_metric": dict(secondary_metric),
        "non_score_metric": secondary_metric["id"],
        "metric_separation": "execution_time/speedup use score_metric only; secondary is report-only",
        "baseline": "frozen per-shape best-of direct rocBLAS/direct hipBLASLt",
        "test_cases": rows,
        "geomean_speedup": _geomean([r[score_metric["id"]]["speedup"] for r in rows]),
        "decode_geomean_speedup": _geomean([
            r[score_metric["id"]]["speedup"] for r in rows if r["regime"] == "decode"]),
        "prefill_geomean_speedup": _geomean([
            r[score_metric["id"]]["speedup"] for r in rows if r["regime"] == "prefill"]),
        "single_call_geomean_speedup": _geomean([
            r["single_call_latency"]["speedup"] for r in rows]),
        "sustained_geomean_speedup": _geomean([
            r["sustained_throughput"]["speedup"] for r in rows]),
        "timing_receipt": timing_receipt,
    }
    print(f"GEAK_TIMING_RECEIPT: {json.dumps(timing_receipt)}")
    _write_report("performance", report)
    return True, report


def _load_paired_candidate(root, label):
    """Load candidate-owned code from one immutable arm under a unique module name."""
    from torch.utils.cpp_extension import ROCM_HOME, load

    root = Path(root).resolve()
    sources = [root / "src/gemm_bindings.cpp", root / "src/dense_bf16_gemm.hip",
               root / "src/custom_gemm.hip"]
    if any(not path.is_file() for path in sources):
        raise RuntimeError(f"{label} root lacks BF16 candidate sources: {root}")
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    name = f"geak_paired_bf16_{label}_{digest.hexdigest()[:16]}"
    build = root / "build" / name
    build.mkdir(parents=True, exist_ok=True)
    ext = load(
        name=name, sources=[str(path) for path in sources], build_directory=str(build),
        extra_include_paths=[str(Path(ROCM_HOME or os.environ.get("ROCM_HOME", "/opt/rocm")) / "include")],
        extra_cflags=["-O3"], extra_cuda_cflags=["-O3"], verbose=True)
    return ext.dense_bf16_gemm


def run_paired_verify(protocol_name=None, score_metric_id=None):
    """Task-v2 producer: full suite, same slots/blocks, candidate+incumbent+oracle."""
    from oracle_wrapper import oracle_backends, oracle_route, validate_oracle_routes

    candidate_root = Path(os.environ["GEAK_PAIRED_CANDIDATE_ROOT"])
    incumbent_root = Path(os.environ["GEAK_PAIRED_INCUMBENT_ROOT"])
    output = Path(os.environ["GEAK_PAIRED_RAW_RECEIPT"])
    expected = json.loads(os.environ["GEAK_PAIRED_EXPECTED_ROUTES"])
    suite = [case["id"] for case in CASES]
    if sorted(expected) != sorted(suite):
        raise RuntimeError(f"paired verify requires the complete BF16 suite: {suite}")
    protocol = harness.measurement_protocol(protocol_name)
    score_metric = harness.score_metric_config(score_metric_id, protocol=protocol)
    os.environ["GEAK_SCORE_METRIC"] = score_metric["id"]
    secondary_metric = harness.secondary_score_metric(score_metric, protocol=protocol)
    trusted, errors = validate_oracle_routes(protocol["hash"], score_metric, CASES)
    candidate = _load_paired_candidate(candidate_root, "candidate")
    incumbent = _load_paired_candidate(incumbent_root, "incumbent")
    backends = oracle_backends(frozen_algorithms=True) if trusted else {}
    # Every route must expose the same outer block inventory. Use the largest slot count any
    # suite route needs to exceed the cache; routes are measured sequentially, so this does not
    # retain all suites' weights at once.
    bytes_per_weight = 2
    common_slot_count = max(harness.weight_rotation_slot_count(
        int(case["N"]) * int(case["K"]) * bytes_per_weight, protocol) for case in CASES)
    routes = []
    for case in CASES:
        arms = {
            "candidate": lambda slot, fn=candidate: fn(*slot["args"]),
            "incumbent": lambda slot, fn=incumbent: fn(*slot["args"]),
        }
        absolute = {"status": "unavailable", "reason": "; ".join(errors)}
        if trusted:
            route = oracle_route(case["M"], case["N"], case["K"])
            oracle_arm = f"oracle:{route}"
            arms[oracle_arm] = lambda slot, fn=backends[route]: fn(*slot["args"])
        slots = _make_measurement_slots(case, protocol, common_slot_count)
        receipts = {}
        for metric in (score_metric, secondary_metric):
            receipt = harness.measure_paired_arms(
                arms, slots, protocol=protocol, metric=metric["id"])
            _validate_measurement_receipt(receipt)
            if (receipt.get("metric_id") != metric["id"]
                    or receipt.get("metric", {}).get("inner") != metric["inner"]):
                raise RuntimeError("paired receipt score metric/inner drift")
            receipts[metric["id"]] = receipt
        score_receipt = receipts[score_metric["id"]]
        if trusted:
            absolute = {
                "status": "ok",
                "oracle_arm": oracle_arm,
                "score_metric": dict(score_metric),
                "candidate_ms": score_receipt["medians_ms"]["candidate"],
                "oracle_ms": score_receipt["medians_ms"][oracle_arm],
                "speedup": (score_receipt["medians_ms"][oracle_arm]
                            / score_receipt["medians_ms"]["candidate"]),
                "secondary_metric": dict(secondary_metric),
                "secondary_candidate_ms": receipts[secondary_metric["id"]]["medians_ms"]["candidate"],
                "secondary_oracle_ms": receipts[secondary_metric["id"]]["medians_ms"][oracle_arm],
            }
        routes.append({
            "route": case["id"],
            "score_metric": dict(score_metric),
            "execution_time_ms": score_receipt["medians_ms"]["candidate"],
            "incumbent_ms": score_receipt["medians_ms"]["incumbent"],
            "speedup": (score_receipt["medians_ms"]["incumbent"]
                        / score_receipt["medians_ms"]["candidate"]),
            "measurement_receipt": score_receipt,
            "secondary_metric": dict(secondary_metric),
            "secondary_measurement_receipt": receipts[secondary_metric["id"]],
            "absolute_report": absolute,
        })
    raw = {
        "schema": "geak.paired-measurement-suite/v2",
        "expected_routes": sorted(suite),
        "protocol_hash": protocol["hash"],
        "score_metric": dict(score_metric),
        "metric_id": score_metric["id"],
        "inner": score_metric["inner"],
        "secondary_metric": dict(secondary_metric),
        "promotion_control": "incumbent",
        "oracle_role": "absolute_report_only",
        "routes": routes,
    }
    harness.atomic_write_json(output, raw)
    return True, raw


def run_oracle_bakeoff(selected_case=None, protocol_name=None, score_metric_id=None):
    """Re-select and atomically freeze the denominator under one explicit score metric."""
    from oracle_wrapper import oracle_backend_metadata, oracle_backends
    from route_manifest import SCHEMA, write_route_manifest

    protocol = harness.measurement_protocol(protocol_name)
    score_metric = harness.score_metric_config(score_metric_id, protocol=protocol)
    os.environ["GEAK_SCORE_METRIC"] = score_metric["id"]
    secondary_metric = harness.secondary_score_metric(score_metric, protocol=protocol)
    cases = [c for c in CASES if selected_case in (None, c["id"])]
    if selected_case is not None and not cases:
        raise ValueError(f"unknown case: {selected_case}")
    if selected_case is not None:
        raise ValueError("a frozen oracle manifest requires all suite cases; omit --case")
    backends = oracle_backends(frozen_algorithms=False)
    routes = {}
    receipts = {}

    for case in cases:
        slots = _make_measurement_slots(case, protocol)
        first_args = slots[0]["args"]
        ref = (first_args[0].float() @ first_args[1].float().t()).to(torch.bfloat16)
        correctness = {}
        metadata = {}
        for name, fn in backends.items():
            out = fn(*first_args)
            ok, err = harness.correct(out, ref, TOL)
            if not ok:
                raise RuntimeError(f"{name} failed correctness for {case['id']}: {err}")
            correctness[name] = {"correct": True, "max_rel_err": err}
            metadata[name] = oracle_backend_metadata(
                *first_args, name, frozen_algorithm=False)
        arms = {name: (lambda slot, fn=fn: fn(*slot["args"]))
                for name, fn in backends.items()}
        metric_receipts = {}
        for metric in (score_metric, secondary_metric):
            receipt = harness.measure_paired_arms(
                arms, slots, protocol=protocol, metric=metric["id"])
            _validate_measurement_receipt(receipt)
            if (receipt.get("metric_id") != metric["id"]
                    or receipt.get("metric", {}).get("inner") != metric["inner"]):
                raise RuntimeError("oracle bakeoff receipt score metric/inner drift")
            metric_receipts[metric["id"]] = receipt
        score_receipt = metric_receipts[score_metric["id"]]
        secondary_receipt = metric_receipts[secondary_metric["id"]]
        winner = min(backends, key=lambda name: score_receipt["medians_ms"][name])
        routes[case["id"]] = {
            "M": case["M"], "N": case["N"], "K": case["K"],
            "backend": winner,
            "correctness": correctness,
            "backend_measurements": {
                name: {
                    "median_ms": score_receipt["medians_ms"][name],
                    "samples_ms": [sample["ms"] for sample in score_receipt["samples"]
                                   if sample["arm"] == name],
                    "score_metric": dict(score_metric),
                    "secondary": {
                        "metric": dict(secondary_metric),
                        "median_ms": secondary_receipt["medians_ms"][name],
                        "samples_ms": [sample["ms"] for sample in secondary_receipt["samples"]
                                       if sample["arm"] == name],
                    },
                    "metadata": metadata[name],
                }
                for name in backends
            },
        }
        receipts[case["id"]] = metric_receipts
    manifest = {
        "schema": SCHEMA,
        "task": "dense_bf16_gemm_fused",
        "frozen": True,
        "selection_state": "baked",
        "selection_metric": score_metric["id"],
        "score_metric": dict(score_metric),
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
    report = {
        "status": "ok", "manifest": str(path), "routes": routes,
        "protocol_hash": protocol["hash"], "score_metric": dict(score_metric),
        "secondary_metric": dict(secondary_metric), "raw_receipts": receipts,
    }
    _write_report("oracle_bakeoff", report)
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
    parser.add_argument("mode", choices=["preflight", "compile", "correctness", "performance",
                                         "oracle-bakeoff", "paired-verify", "profile"])
    parser.add_argument("--case")
    parser.add_argument("--protocol", choices=["legacy_write_flush", "read_evict", "warm",
                                                "semantic_weight_rotating"])
    parser.add_argument("--score-metric", choices=harness.SCORE_METRIC_IDS)
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
            ok, report = run_performance(args.case, args.protocol, args.score_metric)
        elif args.mode == "oracle-bakeoff":
            if not args.confirm_oracle_bakeoff:
                parser.error("oracle-bakeoff requires --confirm-oracle-bakeoff")
            ok, report = run_oracle_bakeoff(args.case, args.protocol, args.score_metric)
        elif args.mode == "paired-verify":
            if args.case:
                parser.error("paired-verify is full-suite only; --case is forbidden")
            ok, report = run_paired_verify(args.protocol, args.score_metric)
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
