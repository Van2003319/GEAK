#!/usr/bin/env python3
"""Locked GPU check that cached task dispatch performs no hashing and stays near bare launch."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from asm_tuner.contracts import TaskContract


def timed(call, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-manifest", required=True, type=Path)
    parser.add_argument("--task-root", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--max-ratio", type=float, default=1.20)
    args = parser.parse_args()
    if (os.environ.get("GEAK_GPU_LOCK_HELD") != "1"
            or os.environ.get("GEAK_GPU_FENCE_HELD") != "1"):
        raise RuntimeError("dispatch overhead check requires GPU lock and fence")
    task_root = args.task_root.resolve()
    contract = TaskContract.load(
        args.task_manifest, task_root=task_root, require_oracle_compatible=True,
    )
    sys.path.insert(0, str(task_root))
    sys.path.insert(0, str(task_root / "scripts"))
    import geak_tuner_runtime as runtime
    import gemm_wrapper
    import task_runner

    case = contract.case(args.case)
    inputs = task_runner._make_inputs(dict(case))
    entrypoint = str(contract.value["bindings"]["entrypoint"]["symbol"])
    task_call = lambda: getattr(gemm_wrapper, entrypoint)(*inputs)
    task_call()
    torch.cuda.synchronize()
    selected = runtime._select(
        runtime._load_table(), (int(case["M"]), int(case["N"]), int(case["K"])),
    )
    if selected is None:
        raise RuntimeError("overhead target is not a candidate route")
    route, entry = selected
    if route != args.case:
        raise RuntimeError("dispatch selected a different route")
    launch = runtime._FUNCTIONS[(route, str(entry["config_id"]))]
    shape = (int(case["M"]), int(case["N"]), int(case["K"]))

    def bare_call():
        out = torch.empty((shape[0], shape[1]), device=inputs[0].device, dtype=torch.bfloat16)
        stream = torch.cuda.current_stream(inputs[0].device)
        if contract.adapter_name == "bf16":
            a, b = inputs
            a_scale = b_scale = None
        else:
            a, b, a_scale, b_scale = inputs
        ptr = lambda tensor: ctypes.c_void_p(
            0 if tensor is None else int(tensor.data_ptr()))
        status = int(launch(
            shape[0], shape[1], shape[2], ptr(a), ptr(b), ptr(a_scale), ptr(b_scale),
            ptr(out), ctypes.c_void_p(int(stream.cuda_stream)),
        ))
        if status:
            raise RuntimeError("bare launch failed with status %d" % status)
        return out

    # Validation and CDLL/symbol resolution happened above. Any hash from this
    # point means a timed dispatch regressed to per-call file verification.
    runtime._sha256 = lambda _path: (_ for _ in ()).throw(
        AssertionError("timed dispatch attempted to hash a file"))
    for _ in range(10):
        task_call()
        bare_call()
    torch.cuda.synchronize()
    task_times, bare_times = [], []
    for repeat in range(7):
        if repeat % 2:
            bare_times.append(timed(bare_call, args.iterations))
            task_times.append(timed(task_call, args.iterations))
        else:
            task_times.append(timed(task_call, args.iterations))
            bare_times.append(timed(bare_call, args.iterations))
    task_ms = statistics.median(task_times)
    bare_ms = statistics.median(bare_times)
    ratio = task_ms / bare_ms
    report = {
        "schema": "geak.asm-dispatch-overhead/v1",
        "ok": ratio <= args.max_ratio,
        "task": contract.task,
        "case": args.case,
        "config_id": entry["config_id"],
        "iterations_per_repeat": args.iterations,
        "repeats": 7,
        "task_wrapper_ms": task_ms,
        "bare_launch_ms": bare_ms,
        "ratio": ratio,
        "max_ratio": args.max_ratio,
        "timed_hash_calls": 0,
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
