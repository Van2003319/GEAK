// torch.utils.cpp_extension compiles with -D__HIP_NO_HALF_OPERATORS__ -D__HIP_NO_HALF_CONVERSIONS__,
// which deletes the __half conversions that rocWMMA's vector registration needs
// (`static_cast<hfloat16_t>(0.0f)` in vector_impl.hpp). rocWMMA then fails to compile even though
// this kernel touches no fp16 at all. Undo them before the include, exactly as the bf16 seed does.
#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif

#include "dense_fp8_gemm.h"

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

namespace {

using F8 = rocwmma::float8_fnuz_t;
using Bf16 = rocwmma::bfloat16_t;

// 16x16x32 is the fragment shape that maps onto v_mfma_f32_16x16x32_fp8_fp8, verified by
// reading the emitted ISA rather than by trusting that rocWMMA picked the fp8 path: at
// BlockK=16 rocWMMA emits the SAME x32 instruction with half its K lanes wasted, so the
// natural translation of the bf16 seed's 16x16x16 would have thrown away half the MFMA.
using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 32, F8, rocwmma::row_major>;
using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 32, F8, rocwmma::col_major>;
using FragC = rocwmma::fragment<rocwmma::accumulator, 16, 16, 32, float>;

constexpr int kTile = 16;
constexpr int kBlockK = 32;
constexpr int kStageK = 128;
constexpr int kChunk = 16;                      // fp8 elements in one uint4
constexpr int kLdsStride = kStageK + kChunk;    // pad by a whole chunk, keeps &lds[r][kk] 16B-aligned
constexpr int kWaves = 4;
constexpr int kThreads = 256;
static_assert(kStageK % kBlockK == 0);
static_assert(kLdsStride % kChunk == 0);
static_assert(kThreads == kWaves * 64);

// Stage one 16-byte chunk of a row-major [* x k] matrix into LDS, zero-filling anything outside
// the matrix. The zero fill is what makes the MFMA safe on the M, N and K tails: an fp8 zero
// contributes nothing to the fp32 accumulator, so a padded tile computes the same answer as a
// masked one. The vector load is only taken when the row stride keeps it 16-byte aligned --
// a uint4 load at an odd offset faults, and every shape in this suite has k % 16 == 0, so the
// scalar path exists for generality rather than for the measured cases.
__device__ __forceinline__ void stage_chunk(
    F8* __restrict__ dst,
    const F8* __restrict__ row_base,
    bool row_valid,
    bool k_aligned,
    int k,
    int k_begin) {
    if (row_valid && k_aligned && k_begin + kChunk <= k) {
        *reinterpret_cast<uint4*>(dst) =
            *reinterpret_cast<const uint4*>(row_base + k_begin);
        return;
    }
    *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
    if (!row_valid) {
        return;
    }
#pragma unroll
    for (int e = 0; e < kChunk; ++e) {
        if (k_begin + e < k) {
            dst[e] = row_base[k_begin + e];
        }
    }
}

// Library-free correctness seed. One workgroup computes four adjacent 16x16 output tiles,
// staging one A tile plus four B tiles in LDS on a deliberately simple synchronous schedule.
// It is the starting point for register blocking, wider staging and load/compute pipelining --
// not an attempt at a good schedule.
__global__ __launch_bounds__(kThreads, 1)
void fp8_gemm_rowwise_kernel(
    int m,
    int n,
    int k,
    const F8* __restrict__ a,
    const F8* __restrict__ b,
    const float* __restrict__ a_scale,
    const float* __restrict__ b_scale,
    Bf16* __restrict__ c) {
// Gate on the capability actually required -- 16x16x32 fp8 MFMA -- which is CDNA3 (gfx942) and
// later. Guarding on the wrong arch macro compiles the body to nothing, i.e. a no-op that
// returns an allocator-recycled buffer and scores an unbounded fake speedup; that failure is
// recorded in the bf16 seed and the write probe in task_runner.py is what catches it.
#if defined(__gfx942__) || defined(__gfx950__)
    __shared__ __align__(16) F8 as[kTile][kLdsStride];
    __shared__ __align__(16) F8 bs[kWaves][kTile][kLdsStride];
    __shared__ __align__(16) float out[kWaves][kTile][kTile];

    const int tid = static_cast<int>(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int row0 = static_cast<int>(blockIdx.y) * kTile;
    const int col0 = (static_cast<int>(blockIdx.x) * kWaves + wave) * kTile;
    const bool k_aligned = (k % kChunk) == 0;

    FragC acc;
    rocwmma::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < k; k0 += kStageK) {
        for (int load = tid; load < kTile * kStageK / kChunk; load += kThreads) {
            const int linear = load * kChunk;
            const int row = linear / kStageK;
            const int kk = linear % kStageK;
            const bool valid = row0 + row < m;
            stage_chunk(&as[row][kk], a + static_cast<long>(row0 + row) * k + k0,
                        valid, k_aligned, k, kk);
        }

        for (int load = lane; load < kTile * kStageK / kChunk; load += 64) {
            const int linear = load * kChunk;
            const int row = linear / kStageK;
            const int kk = linear % kStageK;
            const bool valid = col0 + row < n;
            stage_chunk(&bs[wave][row][kk], b + static_cast<long>(col0 + row) * k + k0,
                        valid, k_aligned, k, kk);
        }
        __syncthreads();

#pragma unroll
        for (int sk = 0; sk < kStageK; sk += kBlockK) {
            FragA frag_a;
            FragB frag_b;
            rocwmma::load_matrix_sync(frag_a, &as[0][sk], kLdsStride);
            rocwmma::load_matrix_sync(frag_b, &bs[wave][0][sk], kLdsStride);
            rocwmma::mma_sync(acc, frag_a, frag_b, acc);
        }
        __syncthreads();
    }

    rocwmma::store_matrix_sync(&out[wave][0][0], acc, kTile, rocwmma::mem_row_major);

    // Rowwise dequantisation, fused into the epilogue: the accumulator holds the raw fp8
    // product, so the per-token and per-channel scales multiply in here rather than in a
    // separate pass over C.
    for (int i = lane; i < kTile * kTile; i += 64) {
        const int row = i >> 4;
        const int col = i & 15;
        if (row0 + row < m && col0 + col < n) {
            const float scaled = out[wave][row][col] * a_scale[row0 + row] * b_scale[col0 + col];
            c[static_cast<long>(row0 + row) * n + col0 + col] = Bf16(scaled);
        }
    }
#else
    (void)m;
    (void)n;
    (void)k;
    (void)a;
    (void)b;
    (void)a_scale;
    (void)b_scale;
    (void)c;
#endif
}

}  // namespace

void custom_fp8_gemm_launcher(
    int m,
    int n,
    int k,
    const c10::Float8_e4m3fnuz* a,
    const c10::Float8_e4m3fnuz* b,
    const float* a_scale,
    const float* b_scale,
    c10::BFloat16* c,
    hipStream_t stream) {
    const dim3 grid((n + kWaves * kTile - 1) / (kWaves * kTile),
                    (m + kTile - 1) / kTile);
    fp8_gemm_rowwise_kernel<<<grid, kThreads, 0, stream>>>(
        m, n, k,
        reinterpret_cast<const F8*>(a),
        reinterpret_cast<const F8*>(b),
        a_scale, b_scale,
        reinterpret_cast<Bf16*>(c));
}
