#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <hip/hip_runtime.h>

// C[M,N] (bf16) = (A[M,K] * a_scale[M]) @ (B[N,K] * b_scale[N])^T
//
// A and B are e4m3fnuz -- the gfx942-native fp8 variant, the one v_mfma_f32_16x16x32_fp8_fp8
// consumes. a_scale is per-token (one per row of A), b_scale is per-channel (one per row of B).
// Both are fp32 and both are applied in the epilogue, so the MFMA accumulates in fp32 on the
// raw fp8 values.

void custom_fp8_gemm_launcher(
    int m,
    int n,
    int k,
    const c10::Float8_e4m3fnuz* a,
    const c10::Float8_e4m3fnuz* b,
    const float* a_scale,
    const float* b_scale,
    c10::BFloat16* c,
    hipStream_t stream);

void dense_fp8_gemm_launcher(
    int m,
    int n,
    int k,
    const c10::Float8_e4m3fnuz* a,
    const c10::Float8_e4m3fnuz* b,
    const float* a_scale,
    const float* b_scale,
    c10::BFloat16* c,
    int device,
    hipStream_t stream);
