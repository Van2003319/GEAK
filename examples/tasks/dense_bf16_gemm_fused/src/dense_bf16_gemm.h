#pragma once

#include <c10/util/BFloat16.h>
#include <hip/hip_runtime.h>

void custom_bf16_gemm_launcher(
    int m,
    int n,
    int k,
    const c10::BFloat16* a,
    const c10::BFloat16* b,
    c10::BFloat16* c,
    hipStream_t stream);

void dense_bf16_gemm_launcher(
    int m,
    int n,
    int k,
    const c10::BFloat16* a,
    const c10::BFloat16* b,
    c10::BFloat16* c,
    int device,
    hipStream_t stream);
