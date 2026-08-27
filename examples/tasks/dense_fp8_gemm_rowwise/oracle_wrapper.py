"""The immutable oracle: the fastest FP8 GEMM this box can actually deliver.

FROZEN. Do not edit. Its SHA256 is pinned in PROVENANCE.json and it is the denominator every
speedup in this task is measured against. Editing it does not make the candidate faster, it
makes every number in the run meaningless.

Why a per-shape best-of and not one library
-------------------------------------------
No single fp8 GEMM wins across the suite on gfx942. Measured twice, on an idle GPU, with the
same warmup / cache-flush / median-of-30 discipline the harness uses (see
PROVENANCE.oracle_selection for the full table -- the per-shape winner was identical in both
passes):

    M <= 16   aiter.gemm_a8w8_CK       20.3 / 25.0 / 21.6 us   vs hipBLASLt 23.0 / 28.2 / 23.0
    M >= 32   hipBLASLt rowwise        31.9 .. 118.8 us        vs aiter_ck  35.5 .. 225.4

Taking either library alone would hand the candidate a baseline 5% (hipBLASLt-only) or 30%
(aiter-only) softer than the machine's real best, and an easy baseline turns every later
speedup into a lie. So the oracle is the per-shape best-of, with the choice FROZEN from that
measurement rather than re-raced at runtime -- a runtime race is not reproducible and would let
the denominator drift under the candidate.

Paths measured and deliberately excluded
----------------------------------------
aiter.gemm_a8w8_bpreshuffle_ck is faster still (42.1 us geomean vs 43.0 for the best-of), but
only when the weight is shuffled ONCE and kept across calls. This task hands both arms a plain
contiguous B on every call and lets neither cache anything, so scoring the oracle on a
pre-shuffled weight would compare two different contracts. Measured with the shuffle inside the
timed region it is the SLOWEST path of all (88.0 us, 2.1x worse), which is what settles it.
Anyone reporting this task's results against production serving should quote 42.1 us as the
harder reference and say why.

aiter.gemm_a8w8_ASM is rejected by this build: "get_cfg Unsupported input_type: fp8,
out_type: bf16" (asm_gemm_a8w8.cu:52). The hand-asm a8w8 kernel here is int8-only.

Per-tensor scaling is not measured at all. These tensors are quantised per row, so one scalar
cannot dequantise them (max rel err ~1.0), and rowwise is what fp8 serving ships.
"""
import os

import torch

FP8 = torch.float8_e4m3fnuz

# Frozen per-shape route. Keyed (M, N, K) over the eleven suite cases; the value is the library
# that won BOTH measurement passes. `_route_fallback` covers anything outside the suite.
_FROZEN_ROUTE = {
    (2, 4096, 4096): "aiter_ck",
    (8, 11008, 4096): "aiter_ck",
    (16, 4096, 4096): "aiter_ck",
    (32, 4096, 11008): "hipblaslt_rowwise",
    (64, 8192, 8192): "hipblaslt_rowwise",
    (96, 11008, 4096): "hipblaslt_rowwise",
    (128, 4096, 4096): "hipblaslt_rowwise",
    (256, 4096, 11008): "hipblaslt_rowwise",
    (512, 11008, 4096): "hipblaslt_rowwise",
    (1024, 4096, 11008): "hipblaslt_rowwise",
    (2048, 4096, 4096): "hipblaslt_rowwise",
}


def _route_fallback(m: int) -> str:
    # The crossover sat between M=16 and M=32 on every measured N and K. Off-suite shapes take
    # the same rule so the oracle is total rather than raising on a shape the run invents.
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
    if a.shape[1] != b.shape[1]:
        raise ValueError("A and B must have matching K dimensions")
    if a_scale.numel() != a.shape[0] or b_scale.numel() != b.shape[0]:
        raise ValueError("a_scale must hold M elements and b_scale must hold N elements")


def _hipblaslt_rowwise(a: torch.Tensor, b: torch.Tensor,
                       a_scale: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    # torch._scaled_mm on ROCm routes to hipBLASLt's rowwise-scaled solutions.
    return torch._scaled_mm(a, b.t(),
                            scale_a=a_scale.view(-1, 1), scale_b=b_scale.view(1, -1),
                            out_dtype=torch.bfloat16)


def _aiter_ck(a: torch.Tensor, b: torch.Tensor,
              a_scale: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    import aiter
    return aiter.gemm_a8w8_CK(a, b, a_scale.view(-1, 1), b_scale.view(1, -1))


_IMPLS = {"hipblaslt_rowwise": _hipblaslt_rowwise, "aiter_ck": _aiter_ck}


def oracle_available() -> tuple[bool, str]:
    """Both libraries must be importable. A missing one is not a reason to fall back to the
    other: that silently softens the denominator, which is the one failure this task cannot
    detect from its own numbers."""
    try:
        import aiter  # noqa: F401
    except Exception as exc:
        return False, f"aiter unavailable ({type(exc).__name__}: {exc})"
    if not hasattr(torch, "_scaled_mm"):
        return False, "torch._scaled_mm unavailable"
    return True, "ok"


def best_fp8_gemm(a: torch.Tensor, b: torch.Tensor,
                  a_scale: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    """Compute the immutable comparison result with the fastest measured library for this shape."""
    _check_inputs(a, b, a_scale, b_scale)
    m, k = a.shape
    n = b.shape[0]
    route = _FROZEN_ROUTE.get((m, n, k)) or _route_fallback(m)
    if route == "aiter_ck" and os.environ.get("GEAK_FP8_ORACLE_HIPBLASLT_ONLY") == "1":
        # An explicit, loud escape hatch for a box without aiter. It makes the baseline softer
        # on the three smallest shapes, so it has to be asked for by name and recorded.
        route = "hipblaslt_rowwise"
    return _IMPLS[route](a, b, a_scale, b_scale)


def oracle_route(m: int, n: int, k: int) -> str:
    """Which library the oracle uses for a shape, so reports can name the denominator."""
    return _FROZEN_ROUTE.get((m, n, k)) or _route_fallback(m)
