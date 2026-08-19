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

__device__ __forceinline__ float bf16_bits_to_float(unsigned bits) {
    return __uint_as_float(bits << 16);
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

    const uint2* __restrict__ a4 = reinterpret_cast<const uint2*>(a);
    const uint2* __restrict__ b4 =
        reinterpret_cast<const uint2*>(b + row * kDecodeM1K);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
#pragma unroll 4
    for (int i = lane; i < kDecodeM1K / 4; i += kWaveSize) {
        const uint2 av = a4[i];
        const uint2 bv = b4[i];
        sum0 = fmaf(bf16_bits_to_float(av.x & 0xffffu),
                    bf16_bits_to_float(bv.x & 0xffffu), sum0);
        sum1 = fmaf(bf16_bits_to_float(av.x >> 16),
                    bf16_bits_to_float(bv.x >> 16), sum1);
        sum2 = fmaf(bf16_bits_to_float(av.y & 0xffffu),
                    bf16_bits_to_float(bv.y & 0xffffu), sum2);
        sum3 = fmaf(bf16_bits_to_float(av.y >> 16),
                    bf16_bits_to_float(bv.y >> 16), sum3);
    }
    const float sum = wave64_sum((sum0 + sum1) + (sum2 + sum3));
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
    custom_bf16_gemm_launcher(m, n, k, a, b, c, stream);
}
