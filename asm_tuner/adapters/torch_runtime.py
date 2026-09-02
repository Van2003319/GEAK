from __future__ import annotations

import ctypes
import statistics
from pathlib import Path
from typing import Any, Dict

from ..schema import DType, KernelConfig, Route
from .base import AdapterError, Measurement, TaskAdapter


def _ptr(tensor: Any) -> ctypes.c_void_p:
    return ctypes.c_void_p(0 if tensor is None else int(tensor.data_ptr()))


def _check_result(torch: Any, config: KernelConfig, route: Route, a: Any, b: Any, a_scale: Any, b_scale: Any, out: Any) -> Dict[str, Any]:
    if route.m * route.n <= 65536:
        rows = torch.arange(route.m, device=out.device)
        cols = torch.arange(route.n, device=out.device)
    else:
        rows = torch.linspace(0, route.m - 1, min(route.m, 8), device=out.device).long().unique()
        cols = torch.linspace(0, route.n - 1, min(route.n, 16), device=out.device).long().unique()
    lhs = a.index_select(0, rows).float()
    rhs = b.index_select(0, cols).float()
    reference = lhs @ rhs.transpose(0, 1)
    if config.dtype is DType.FP8_E4M3_FNUZ:
        reference *= a_scale.index_select(0, rows).float()[:, None]
        reference *= b_scale.index_select(0, cols).float()[None, :]
    actual = out.index_select(0, rows).index_select(1, cols).float()
    diff = (actual - reference).abs()
    scale = reference.abs().clamp_min(1.0)
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    max_rel = float((diff / scale).max().item()) if diff.numel() else 0.0
    atol = max(0.5, float(reference.abs().max().item()) * 0.02) if reference.numel() else 0.5
    passed = bool(torch.allclose(actual, reference, rtol=0.03, atol=atol))
    fully_written = not bool(torch.isnan(out.float()).any().item())
    return {
        "checked": True,
        "mode": "full" if route.m * route.n <= 65536 else "sampled",
        "rows": int(rows.numel()),
        "cols": int(cols.numel()),
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
        "atol": atol,
        "rtol": 0.03,
        "fully_written": fully_written,
        "passed": passed and fully_written,
    }


def measure_torch(
    adapter: TaskAdapter,
    shared_object: Path,
    config: KernelConfig,
    route: Route,
    samples: int,
    warmups: int,
    device: int,
    check: bool,
) -> Measurement:
    if samples < 1 or warmups < 1:
        raise AdapterError("samples and warmups must both be positive")
    try:
        import torch
    except ImportError as exc:
        raise AdapterError("PyTorch ROCm is required to execute task adapters") from exc
    if not torch.cuda.is_available():
        raise AdapterError("ROCm device is not available through torch.cuda")

    torch.cuda.set_device(device)
    torch.manual_seed(20260902)
    target = "cuda:%d" % device
    if config.dtype is DType.BF16:
        a = torch.randn((route.m, route.k), device=target, dtype=torch.float32).to(torch.bfloat16)
        b = torch.randn((route.n, route.k), device=target, dtype=torch.float32).to(torch.bfloat16)
        a_scale = None
        b_scale = None
    else:
        fp8 = getattr(torch, "float8_e4m3fnuz", None)
        if fp8 is None:
            raise AdapterError("torch.float8_e4m3fnuz is unavailable")
        a = torch.randn((route.m, route.k), device=target, dtype=torch.float32).clamp_(-2.0, 2.0).to(fp8)
        b = torch.randn((route.n, route.k), device=target, dtype=torch.float32).clamp_(-2.0, 2.0).to(fp8)
        a_scale = torch.rand((route.m,), device=target, dtype=torch.float32) + 0.5
        b_scale = torch.rand((route.n,), device=target, dtype=torch.float32) + 0.5
    out = torch.empty((route.m, route.n), device=target, dtype=torch.bfloat16)
    out.fill_(float("nan"))

    library = ctypes.CDLL(str(shared_object))
    try:
        launch = getattr(library, adapter.launch_name(config))
    except AttributeError as exc:
        raise AdapterError("shared artifact lacks %s" % adapter.launch_name(config)) from exc
    launch.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 6
    launch.restype = ctypes.c_int
    stream = torch.cuda.current_stream(device)
    stream_ptr = ctypes.c_void_p(int(stream.cuda_stream))

    def invoke() -> None:
        status = int(launch(
            route.m, route.n, route.k, _ptr(a), _ptr(b), _ptr(a_scale),
            _ptr(b_scale), _ptr(out), stream_ptr,
        ))
        if status != 0:
            raise AdapterError("HIP launch failed with error code %d" % status)

    invoke()
    torch.cuda.synchronize(device)
    correctness = _check_result(torch, config, route, a, b, a_scale, b_scale, out) if check else {"checked": False}
    if check and not correctness["passed"]:
        raise AdapterError("correctness gate failed: %r" % correctness)

    for _ in range(warmups):
        invoke()
    torch.cuda.synchronize(device)
    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        invoke()
        end.record(stream)
        end.synchronize()
        timings.append(float(start.elapsed_time(end)) * 1000.0)

    timer = {
        "backend": "hip_event",
        "dispatch_primed": True,
        "warmups": warmups,
        "samples": samples,
        "device": device,
        "synchronized": True,
    }
    return Measurement(tuple(timings), float(statistics.median(timings)), correctness, timer)
