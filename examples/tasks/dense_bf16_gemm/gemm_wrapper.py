import torch

from kernel_loader import gemm_ext


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


def dense_bf16_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute A[M,K] @ B[N,K].T with the editable HIP candidate."""
    _check_inputs(a, b)
    return gemm_ext.dense_bf16_gemm(a, b)


def rocblas_bf16_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute A[M,K] @ B[N,K].T through direct rocblas_gemm_ex."""
    _check_inputs(a, b)
    return gemm_ext.rocblas_bf16_gemm(a, b)
