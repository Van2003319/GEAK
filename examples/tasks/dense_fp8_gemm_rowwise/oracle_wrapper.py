"""Rowwise FP8 oracle backends with an immutable-in-process route manifest.

The checked-in manifest preserves the historical routes but carries their old protocol identity.
A formal v2 performance run therefore requires ``oracle-bakeoff`` first; correctness may continue
to use the migrated routes while that deliberate bake-off is pending.
"""
from pathlib import Path

import torch

from route_manifest import FrozenRouteManifest

FP8 = torch.float8_e4m3fnuz
_TASK = "dense_fp8_gemm_rowwise"
_ROUTE_PATH = Path(__file__).with_name("oracle_routes.v2.json")
_ROUTE_MANIFEST = FrozenRouteManifest(_ROUTE_PATH, _TASK)


def _route_fallback(m: int) -> str:
    return "aiter_ck" if m <= 16 else "hipblaslt_rowwise"


def _check_inputs(a: torch.Tensor, b: torch.Tensor,
                  a_scale: torch.Tensor, b_scale: torch.Tensor) -> None:
    if not (a.is_cuda and b.is_cuda and a_scale.is_cuda and b_scale.is_cuda):
        raise ValueError("A, B and both scales must be GPU tensors")
    if a.dtype != FP8 or b.dtype != FP8:
        raise TypeError("A and B must have dtype torch.float8_e4m3fnuz")
    if a_scale.dtype != torch.float32 or b_scale.dtype != torch.float32:
        raise TypeError("scales must have dtype torch.float32")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be rank-2 tensors")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("A and B must be contiguous")
    if not a_scale.is_contiguous() or not b_scale.is_contiguous():
        raise ValueError("scales must be contiguous")
    if a.device != b.device or a_scale.device != a.device or b_scale.device != a.device:
        raise ValueError("A, B and both scales must be on the same device")
    if a.shape[1] != b.shape[1]:
        raise ValueError("A and B must have matching K dimensions")
    if a_scale.numel() != a.shape[0] or b_scale.numel() != b.shape[0]:
        raise ValueError("a_scale must hold M elements and b_scale must hold N elements")


def _hipblaslt_rowwise(a: torch.Tensor, b: torch.Tensor,
                       a_scale: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    return torch._scaled_mm(a, b.t(),
                            scale_a=a_scale.view(-1, 1), scale_b=b_scale.view(1, -1),
                            out_dtype=torch.bfloat16)


def _aiter_ck(a: torch.Tensor, b: torch.Tensor,
              a_scale: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    import aiter
    return aiter.gemm_a8w8_CK(a, b, a_scale.view(-1, 1), b_scale.view(1, -1))


_IMPLS = {"hipblaslt_rowwise": _hipblaslt_rowwise, "aiter_ck": _aiter_ck}


def oracle_available() -> tuple[bool, str]:
    try:
        import aiter  # noqa: F401
    except Exception as exc:
        return False, f"aiter unavailable ({type(exc).__name__}: {exc})"
    if not hasattr(torch, "_scaled_mm"):
        return False, "torch._scaled_mm unavailable"
    return True, "ok"


def best_fp8_gemm(a: torch.Tensor, b: torch.Tensor,
                  a_scale: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    _check_inputs(a, b, a_scale, b_scale)
    m, k = a.shape
    n = b.shape[0]
    return _IMPLS[oracle_route(m, n, k)](a, b, a_scale, b_scale)


def oracle_backends() -> dict[str, object]:
    return dict(_IMPLS)


def oracle_backend_metadata(backend: str) -> dict:
    if backend == "aiter_ck":
        import aiter
        return {
            "backend": backend,
            "solution": "aiter.gemm_a8w8_CK",
            "algo_index": "aiter_dispatch_not_exposed",
            "kernel": "runtime_selected_not_exposed",
            "version": getattr(aiter, "__version__", "unknown"),
            "workspace_bytes": "library_managed_not_exposed",
            "layout": "A_MK_B_NK_rowwise_scales_to_C_MN_bfloat16",
            "allocation": "fresh backend output per call",
        }
    if backend == "hipblaslt_rowwise":
        return {
            "backend": backend,
            "solution": "torch._scaled_mm_rowwise_hipblaslt",
            "algo_index": "pytorch_hipblaslt_dispatch_not_exposed",
            "kernel": "runtime_selected_not_exposed",
            "version": {"torch": torch.__version__, "hip": torch.version.hip},
            "workspace_bytes": "library_managed_not_exposed",
            "layout": "A_MK_B_NK_transpose_rowwise_scales_to_C_MN_bfloat16",
            "allocation": "fresh torch._scaled_mm output per call",
        }
    raise KeyError(backend)


def oracle_route(m: int, n: int, k: int) -> str:
    try:
        return _ROUTE_MANIFEST.backend(m, n, k)
    except KeyError:
        return _route_fallback(int(m))


def oracle_route_entry(m: int, n: int, k: int) -> dict:
    return _ROUTE_MANIFEST.entry(m, n, k)


def oracle_route_manifest() -> dict:
    return _ROUTE_MANIFEST.snapshot()


def validate_oracle_routes(protocol_hash: str, score_metric: dict | str,
                           cases: list[dict] | None = None) -> tuple[bool, list[str]]:
    return _ROUTE_MANIFEST.validate_for(protocol_hash, score_metric, cases)
