#include "dense_fp8_gemm.h"

#include <hip/hip_runtime.h>

// The task's entry point. It exists as its own translation unit so a specialisation can be
// added for one shape or one regime without touching the generic seed, which is how the bf16
// task grew its M=1 decode kernel. Right now every shape reaches the generic seed.
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
    hipStream_t stream) {
    (void)device;
    custom_fp8_gemm_launcher(m, n, k, a, b, a_scale, b_scale, c, stream);
}
