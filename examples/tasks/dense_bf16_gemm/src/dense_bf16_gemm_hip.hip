#include "dense_bf16_gemm.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace {

constexpr int kDecodeM1N = 4096;
constexpr int kDecodeM1K = 4096;
constexpr int kWaveSize = 64;
constexpr int kWavesPerBlock = 4;

__device__ __forceinline__ float wave64_sum(float value) {
#pragma unroll
    for (int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, kWaveSize);
    }
    return value;
}

__global__ __launch_bounds__(kWaveSize * kWavesPerBlock)
void decode_m1_4096_wave64_kernel(
    const hip_bfloat16* __restrict__ a,
    const hip_bfloat16* __restrict__ b,
    hip_bfloat16* __restrict__ c) {
    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    const int row = blockIdx.x * kWavesPerBlock + wave;
    if (row >= kDecodeM1N) {
        return;
    }

    const hip_bfloat16* __restrict__ b_row = b + row * kDecodeM1K;
    float sum = 0.0f;
#pragma unroll 4
    for (int kk = lane; kk < kDecodeM1K; kk += kWaveSize) {
        sum = fmaf(
            static_cast<float>(a[kk]),
            static_cast<float>(b_row[kk]),
            sum);
    }
    sum = wave64_sum(sum);
    if (lane == 0) {
        c[row] = hip_bfloat16(sum);
    }
}

void launch_decode_m1_4096(
    const c10::BFloat16* a,
    const c10::BFloat16* b,
    c10::BFloat16* c,
    hipStream_t stream) {
    constexpr int threads = kWaveSize * kWavesPerBlock;
    constexpr int blocks =
        (kDecodeM1N + kWavesPerBlock - 1) / kWavesPerBlock;
   hipLaunchKernelGGL(( decode_m1_4096_wave64_kernel), dim3(blocks), dim3(threads), 0, stream, 
        reinterpret_cast<const hip_bfloat16*>(a),
        reinterpret_cast<const hip_bfloat16*>(b),
        reinterpret_cast<hip_bfloat16*>(c));
}

}  // namespace

void dense_bf16_gemm_launcher(
    int m,
    int n,
    int k,
    const c10::BFloat16* a,
    const c10::BFloat16* b,
    c10::BFloat16* c,
    int device,
    hipStream_t stream) {
    if (m == 1 && n == kDecodeM1N && k == kDecodeM1K) {
        launch_decode_m1_4096(a, b, c, stream);
        return;
    }
    rocblas_bf16_gemm_launcher(m, n, k, a, b, c, device, stream);
}
