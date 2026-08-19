
// ---- offline resource probe (round 9 tile shortlist) -----------------------
// Appended to a copy of the lane's shipped custom_gemm.hip so it instantiates
// the SAME macro_gemm_body template the lane ships -- the numbers then describe
// our kernel, not a lookalike.
//
// The control instantiation duplicates the shipped 128x128 geometry exactly. If
// it does not reproduce the shipped kernel's measured usage, the probe is not
// measuring what it claims and no other row in the table may be believed. This
// is the same discipline that made the PF sweep trustworthy: its PF=1 point
// reproduced the incumbent's 74 VGPR / 64 AGPR / 0 scratch / occupancy 3.
#define PROBE(name, BM, BN, BK, WM, WN, THR)                       \
    __global__ __launch_bounds__(THR, 1)                           \
    void name(int m, int n, int k,                                 \
              const Bf16* __restrict__ a,                          \
              const Bf16* __restrict__ b,                          \
              Bf16* __restrict__ c,                                \
              float* __restrict__ work, int kchunk) {              \
        macro_gemm_body<BM, BN, BK, WM, WN, false>(                \
            m, n, k, a, b, c, work, kchunk);                       \
    }

PROBE(probe_ctrl_128x128_w2x2_t256, 128, 128, 32, 2, 2, 256)
PROBE(probe_256x256_w4x2_t512,      256, 256, 32, 4, 2, 512)
PROBE(probe_256x256_w2x4_t512,      256, 256, 32, 2, 4, 512)
PROBE(probe_224x256_w4x2_t512,      224, 256, 32, 4, 2, 512)
PROBE(probe_256x224_w2x4_t512,      256, 224, 32, 2, 4, 512)
PROBE(probe_128x256_w2x4_t512,      128, 256, 32, 2, 4, 512)
PROBE(probe_192x192_w4x2_t512,      192, 192, 32, 4, 2, 512)
