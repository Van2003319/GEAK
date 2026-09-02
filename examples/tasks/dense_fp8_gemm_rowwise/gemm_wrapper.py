import torch

from geak_tuner_runtime import dispatch as _tuner_dispatch

FP8 = torch.float8_e4m3fnuz


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
    if min(*a.shape, b.shape[0]) <= 0:
        raise ValueError("M, N, and K must be positive")
    if a_scale.numel() != a.shape[0]:
        raise ValueError("a_scale must hold exactly M elements (per-token)")
    if b_scale.numel() != b.shape[0]:
        raise ValueError("b_scale must hold exactly N elements (per-channel)")


def _canonical_gemm(a: torch.Tensor, b: torch.Tensor,
                    a_scale: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    from kernel_loader import gemm_ext
    return gemm_ext.dense_fp8_gemm(a, b, a_scale, b_scale)


def dense_fp8_gemm(a: torch.Tensor, b: torch.Tensor,
                   a_scale: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    """Compute (A*a_scale) @ (B*b_scale).T in bf16 with the editable HIP candidate."""
    _check_inputs(a, b, a_scale, b_scale)
    tuned = _tuner_dispatch("fp8_rowwise", (a, b, a_scale, b_scale))
    if tuned is not None:
        return tuned
    # Shapes outside the exact suite use the declared canonical fallback.
    return _canonical_gemm(a, b, a_scale, b_scale)
