#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif

#include "dense_bf16_gemm.h"

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

namespace {

using Bf16 = rocwmma::bfloat16_t;
using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16,
                                Bf16, rocwmma::row_major>;
using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16,
                                Bf16, rocwmma::col_major>;
using FragC = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

constexpr int kTile = 16;
constexpr int kStageK = 32;
constexpr int kLdsStride = 36;
constexpr int kWaves = 4;
constexpr int kThreads = 256;
constexpr int kOutputElementsPerTile = kTile * kTile;
constexpr int kTallM = 2 * kTile;
static_assert(kStageK == 2 * kTile);
static_assert(kThreads == kWaves * 64);

// Library-free correctness seed. One workgroup computes four adjacent 16x16
// output tiles and stages one A tile plus four B tiles in LDS. This is a
// deliberately simple, synchronous schedule retained as the starting point
// for register blocking and load/compute pipelining.
__global__ __launch_bounds__(kThreads, 1)
void tall_bf16_gemm_kernel(
    int m,
    int n,
    int k,
    const Bf16* __restrict__ a,
    const Bf16* __restrict__ b,
    Bf16* __restrict__ c) {
#if defined(__gfx90a__)
    __shared__ __align__(16) Bf16 as[kTallM][kLdsStride];
    __shared__ __align__(16) Bf16 bs[kWaves][kTile][kLdsStride];
    __shared__ __align__(16) float out[kWaves][kTallM][kTile];

    const int tid = static_cast<int>(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int row0 = static_cast<int>(blockIdx.y) * kTallM;
    const int col0 = (static_cast<int>(blockIdx.x) * kWaves + wave) * kTile;

    FragC acc0;
    FragC acc1;
    rocwmma::fill_fragment(acc0, 0.0f);
    rocwmma::fill_fragment(acc1, 0.0f);

    for (int k0 = 0; k0 < k; k0 += kStageK) {
#pragma unroll
        for (int load = tid; load < kTallM * kStageK / 2; load += kThreads) {
            const int linear = load * 2;
            const int a_row = linear / kStageK;
            const int a_kk = linear % kStageK;
            auto* a_dst = reinterpret_cast<unsigned*>(&as[a_row][a_kk]);
            if (row0 + a_row < m) {
                *a_dst = *reinterpret_cast<const unsigned*>(
                    a + (row0 + a_row) * k + k0 + a_kk);
            } else {
                *a_dst = 0;
            }
        }

        const int b_linear = lane * 8;
        const int b_row = b_linear / kStageK;
        const int b_kk = b_linear % kStageK;
        *reinterpret_cast<uint4*>(&bs[wave][b_row][b_kk]) =
            *reinterpret_cast<const uint4*>(b + (col0 + b_row) * k + k0 + b_kk);
        __syncthreads();

#pragma unroll
        for (int sk = 0; sk < kStageK; sk += kTile) {
            FragA frag_a0;
            FragA frag_a1;
            FragB frag_b;
            rocwmma::load_matrix_sync(frag_a0, &as[0][sk], kLdsStride);
            rocwmma::load_matrix_sync(frag_a1, &as[kTile][sk], kLdsStride);
            rocwmma::load_matrix_sync(frag_b, &bs[wave][0][sk], kLdsStride);
            rocwmma::mma_sync(acc0, frag_a0, frag_b, acc0);
            rocwmma::mma_sync(acc1, frag_a1, frag_b, acc1);
        }
        __syncthreads();
    }

    rocwmma::store_matrix_sync(&out[wave][0][0], acc0, kTile,
                               rocwmma::mem_row_major);
    rocwmma::store_matrix_sync(&out[wave][kTile][0], acc1, kTile,
                               rocwmma::mem_row_major);
#pragma unroll
    for (int i = lane; i < kTallM * kTile; i += 64) {
        const int row = i >> 4;
        const int col = i & 15;
        if (row0 + row < m && col0 + col < n) {
            c[(row0 + row) * n + col0 + col] = Bf16(out[wave][row][col]);
        }
    }
#else
    (void)m;
    (void)n;
    (void)k;
    (void)a;
    (void)b;
    (void)c;
#endif
}

__global__ __launch_bounds__(kThreads, 1)
void generic_bf16_gemm_kernel(
    int m,
    int n,
    int k,
    const Bf16* __restrict__ a,
    const Bf16* __restrict__ b,
    Bf16* __restrict__ c) {
#if defined(__gfx90a__)
    __shared__ __align__(16) Bf16 as[kTile][kLdsStride];
    __shared__ __align__(16) Bf16 bs[kWaves][kTile][kLdsStride];
    __shared__ __align__(16) float out[kWaves][kTile][kTile];

    const int tid = static_cast<int>(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int row0 = static_cast<int>(blockIdx.y) * kTile;
    const int col0 = (static_cast<int>(blockIdx.x) * kWaves + wave) * kTile;

    FragC acc;
    rocwmma::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < k; k0 += kStageK) {
        const int a_linear = tid * 2;
        const int a_row = a_linear / kStageK;
        const int a_kk = a_linear % kStageK;
        auto* a_dst = reinterpret_cast<unsigned*>(&as[a_row][a_kk]);
        if (row0 + a_row < m) {
            *a_dst = *reinterpret_cast<const unsigned*>(
                a + (row0 + a_row) * k + k0 + a_kk);
        } else {
            *a_dst = 0;
        }

        const int b_linear = lane * 8;
        const int b_row = b_linear / kStageK;
        const int b_kk = b_linear % kStageK;
        *reinterpret_cast<uint4*>(&bs[wave][b_row][b_kk]) =
            *reinterpret_cast<const uint4*>(b + (col0 + b_row) * k + k0 + b_kk);
        __syncthreads();

#pragma unroll
        for (int sk = 0; sk < kStageK; sk += kTile) {
            FragA frag_a;
            FragB frag_b;
            rocwmma::load_matrix_sync(frag_a, &as[0][sk], kLdsStride);
            rocwmma::load_matrix_sync(frag_b, &bs[wave][0][sk], kLdsStride);
            rocwmma::mma_sync(acc, frag_a, frag_b, acc);
        }
        __syncthreads();
    }

    rocwmma::store_matrix_sync(&out[wave][0][0], acc, kTile,
                               rocwmma::mem_row_major);
#pragma unroll
    for (int i = lane; i < kOutputElementsPerTile; i += 64) {
        const int row = i >> 4;
        const int col = i & 15;
        if (row0 + row < m && col0 + col < n) {
            c[(row0 + row) * n + col0 + col] = Bf16(out[wave][row][col]);
        }
    }
#else
    (void)m;
    (void)n;
    (void)k;
    (void)a;
    (void)b;
    (void)c;
#endif
}

}  // namespace

void custom_bf16_gemm_launcher(
    int m,
    int n,
    int k,
    const c10::BFloat16* a,
    const c10::BFloat16* b,
    c10::BFloat16* c,
    hipStream_t stream) {
    const int tile_m = m < kTallM ? kTile : kTallM;
    const dim3 grid((n + kWaves * kTile - 1) / (kWaves * kTile),
                    (m + tile_m - 1) / tile_m);
    if (m < kTallM) {
        generic_bf16_gemm_kernel<<<grid, kThreads, 0, stream>>>(
            m, n, k, reinterpret_cast<const Bf16*>(a),
            reinterpret_cast<const Bf16*>(b), reinterpret_cast<Bf16*>(c));
    } else {
        tall_bf16_gemm_kernel<<<grid, kThreads, 0, stream>>>(
            m, n, k, reinterpret_cast<const Bf16*>(a),
            reinterpret_cast<const Bf16*>(b), reinterpret_cast<Bf16*>(c));
    }
}
