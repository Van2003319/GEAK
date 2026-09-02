"""BF16 oracle backends and the frozen per-shape route.

Both rocBLAS and hipBLASLt are called through direct C++ APIs in ``oracle_ext``. In particular this
module never calls ``torch.backends.cuda.preferred_blas_library``: candidate, control and backend
can coexist in one process without mutating process-global backend state.
"""
from pathlib import Path

import torch

from oracle_loader import oracle_ext
from route_manifest import FrozenRouteManifest

_TASK = "dense_bf16_gemm_fused"
_ROUTE_PATH = Path(__file__).with_name("oracle_routes.v2.json")
_ROUTE_MANIFEST = FrozenRouteManifest(_ROUTE_PATH, _TASK)


def _check_inputs(a: torch.Tensor, b: torch.Tensor) -> None:
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("A and B must be GPU tensors")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("A and B must have dtype torch.bfloat16")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be rank-2 tensors")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("A and B must be contiguous")
    if a.device != b.device:
        raise ValueError("A and B must be on the same device")
    if a.shape[1] != b.shape[1]:
        raise ValueError("A and B must have matching K dimensions")
    if min(*a.shape, b.shape[0]) <= 0:
        raise ValueError("M, N, and K must be positive")


def _shape(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int]:
    return int(a.shape[0]), int(b.shape[0]), int(a.shape[1])


def _frozen_hipblaslt_algo(m: int, n: int, k: int) -> int:
    entry = _ROUTE_MANIFEST.entry(m, n, k)
    detail = entry.get("backend_measurements", {}).get("direct_hipblaslt", {})
    return int(detail.get("metadata", {}).get("algo_index", -1))


def rocblas_bf16_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Direct rocblas_gemm_ex control; retained under its historical public name."""
    _check_inputs(a, b)
    return oracle_ext.rocblas_bf16_gemm(a, b)


def hipblaslt_bf16_gemm(a: torch.Tensor, b: torch.Tensor,
                         algo_index: int | None = None) -> torch.Tensor:
    """Direct hipblasLtMatmul with an explicit frozen algorithm when available."""
    _check_inputs(a, b)
    m, n, k = _shape(a, b)
    algo = _frozen_hipblaslt_algo(m, n, k) if algo_index is None else int(algo_index)
    return oracle_ext.hipblaslt_bf16_gemm(a, b, algo)


def best_bf16_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run the backend frozen for this exact suite shape."""
    _check_inputs(a, b)
    m, n, k = _shape(a, b)
    try:
        route = oracle_route(m, n, k)
    except KeyError:
        route = "direct_rocblas"  # correctness-only off-suite sanity shapes
    if route == "direct_rocblas":
        return oracle_ext.rocblas_bf16_gemm(a, b)
    if route == "direct_hipblaslt":
        return oracle_ext.hipblaslt_bf16_gemm(a, b, _frozen_hipblaslt_algo(m, n, k))
    raise RuntimeError(f"unsupported BF16 oracle route {route!r}")


def oracle_backends(*, frozen_algorithms: bool = True) -> dict[str, object]:
    """Backend arms for paired measurement; no function mutates global BLAS preference."""
    if frozen_algorithms:
        hip = lambda a, b: hipblaslt_bf16_gemm(a, b, None)
    else:
        hip = lambda a, b: hipblaslt_bf16_gemm(a, b, -1)
    return {"direct_rocblas": rocblas_bf16_gemm, "direct_hipblaslt": hip}


def oracle_backend_metadata(a: torch.Tensor, b: torch.Tensor, backend: str,
                            *, frozen_algorithm: bool = True) -> dict:
    _check_inputs(a, b)
    if backend == "direct_rocblas":
        return dict(oracle_ext.rocblas_bf16_gemm_info())
    if backend == "direct_hipblaslt":
        m, n, k = _shape(a, b)
        algo = _frozen_hipblaslt_algo(m, n, k) if frozen_algorithm else -1
        return dict(oracle_ext.hipblaslt_bf16_gemm_info(a, b, algo))
    raise KeyError(backend)


def oracle_route(m: int, n: int, k: int) -> str:
    return _ROUTE_MANIFEST.backend(m, n, k)


def oracle_route_entry(m: int, n: int, k: int) -> dict:
    return _ROUTE_MANIFEST.entry(m, n, k)


def oracle_route_manifest() -> dict:
    return _ROUTE_MANIFEST.snapshot()


def validate_oracle_routes(protocol_hash: str, cases: list[dict]) -> tuple[bool, list[str]]:
    return _ROUTE_MANIFEST.validate_for(protocol_hash, cases)


def oracle_available() -> tuple[bool, str]:
    required = ("rocblas_bf16_gemm", "hipblaslt_bf16_gemm",
                "rocblas_bf16_gemm_info", "hipblaslt_bf16_gemm_info")
    missing = [name for name in required if not hasattr(oracle_ext, name)]
    return (not missing, "ok" if not missing else f"oracle extension missing {missing}")
