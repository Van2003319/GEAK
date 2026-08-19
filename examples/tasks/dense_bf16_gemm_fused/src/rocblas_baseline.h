#pragma once

#include <c10/util/BFloat16.h>
#include <hip/hip_runtime.h>

// Immutable oracle/denominator interface. Candidate sources must not include
// this header or call this launcher.
void rocblas_bf16_gemm_launcher(
    int m,
    int n,
    int k,
    const c10::BFloat16* a,
    const c10::BFloat16* b,
    c10::BFloat16* c,
    int device,
    hipStream_t stream);
