---
title: dense_gemm — tuning
kind: technique
operator: dense_gemm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, mxfp4]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - ROCm/aiter@HEAD:gradlib/gradlib/gemm_tuner.py
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
  - https://arxiv.org/abs/2511.08083
---

# dense_gemm — tuning

## TL;DR
The only tuning lever that engages the **live** sglang/vllm GEMM path is **aiter's per-shape DB**
(see [backends/aiter.md](backends/aiter.md)); everything below is the knob space that aiter/CK/triton
search over. Defaults worth testing in a hand kernel are **`mfma_16x16` against `32x32`**, tile
multiples compatible with the layout/XCD mapping, and `OPTIMIZE_EPILOGUE=1` where the backend exposes
that CShuffle path. Launch geometry is not a universal constant: derive it from the decomposition and
the measured bound class; **≥1024 workgroups is not a blanket requirement**.

## What dominates time
Prefill GEMMs (large M) are compute-bound; the win is keeping all 304 CUs busy at high MFMA occupancy.
MI300X sustains only ~45% of theoretical peak (third-party, narrowing) across fp8/bf16/fp16 vs ~93% on
H100/B200 — a software/clock-scaling gap, not hardware. So tuning chases the gap to the *best library
kernel*, not to peak.

## CDNA4 (MI355X / gfx950) is near-solved — the current SOTA bar
On CDNA4, hand and DSL kernels now hit the MFMA ceiling, so on gfx950 the bar is much higher than the
MI300X library numbers above (all figures AMD/HK-measured, M=N=K unless noted):
- **hipBLASLt FP8 bar:** ~**2750 TFLOPS** @ 4096, ~**3130** @ 8192 (MI355X, ROCm 7.1) — the no-tune
  default everyone targets.
- **HIP/C++ 8-wave ping-pong FP8:** **2680** @ 4096 (~97% of hipBLASLt), **3204** @ 8192 — *beats*
  hipBLASLt 3130 **with no assembly**. The 4-wave interleave variant (one wave/SIMD, full 512-VGPR
  budget, 128×128 tile, no `#pragma unroll`) is the robustness/perf successor.
- **Gluon (Triton-based) FP16:** **1489 TFLOPS @ 98.75% MFMA eff** (4096×4096×8192) — the practical
  FP16 ceiling (see [[operators/dense_gemm/backends/gluon]]).
- **HipKittens BF16:** **1610 TFLOPS** (M=N=K=8192, 256×256, no wave-spec) — academic SOTA, > TK-on-B200
  1538 / CUTLASS-B200 1570 (see [[operators/dense_gemm/backends/hipkittens]]).

The 8-wave ping-pong and 4-wave interleave scheduling patterns both **originate from HipKittens**
(arXiv 2511.08083) and were adopted into AMD's own CDNA4 GEMM blogs. NVIDIA-style wave specialization
*underperforms* on CDNA (static register allocation starves producers → ~80% peak BF16); use ping-pong
/interleave — see [[optimization/mfma_scheduling.md]].

## Knob space (per backend, what the tuners search)
- **MFMA instruction**: `mfma_16x16x16` usually beats `mfma_32x32x8` on MI300X for LLM N/K (better
  occupancy / less register pressure). In triton this is `matrix_instr_nonkdim=16`.
- **Tile (BLOCK_M/N/K)**: layout-compatible multiples (often 8 on the indexed dimensions) are a
  useful starting set; common candidates are 128×128×64, 256×128×64 and 256×256×64. Decode
  (skinny M) may justify small BLOCK_M (16/32) plus a reduction decomposition. Non-power-of-two tile,
  split, or grid values are **allowed candidates** when they remove a divisibility/tail/topology loss;
  they are not general winners and must be raced against neighboring power-of-two values.
- **Workgroup/grid construction**: first identify the mechanism. A fixed-output tile-parallel GEMM has
  `ceil(M/BM) * ceil(N/BN)` independent workgroups; manufacturing more requires smaller tiles or a
  reduction and may lose reuse/MFMA efficiency. A persistent tile kernel launches a bounded resident
  grid and grid-strides over **whole output tiles**. Stream-K instead grid-strides a flattened
  tile×K-iteration space and can require partial-tile fix-up. Uniform split-K gives every output tile
  the same K partitions and pays a reduction. These are different mechanisms, not spellings of one
  “more workgroups” rule.
- **XCD/L2 placement**: distribute workgroups that share data onto the **same XCD** to cut cross-die
  L2 traffic and stabilize timing.
- **SPLIT_K / Stream-K / persistent grid-stride**: candidates for a measured machine-fill or
  wave-quantization deficit, not defaults for every skinny or tall-K shape →
  see [[operators/splitk_streamk_gemm/overview.md]].
- **waves_per_eu / num_stages / num_warps** (triton): raise stages for K-deep prefill; lower
  waves_per_eu if register-spilling.
- **b_preshuffle**: pre-permute B to the MFMA-native layout to kill the LDS shuffle on the hot path
  (part of aiter's lookup key as `bpreshuffle`).
- **`OPTIMIZE_EPILOGUE=1`**: routes the C write-out away from the 512B Tagram hotspot (CShuffle path).

## Grid size is conditional on the bound class
- **Fixed-output tile-parallel**: preserve one owner per output tile when the natural tile grid already
  supplies enough resident waves. A grid below 1024 can still be optimal when MFMA efficiency, operand
  reuse, or reduction/output traffic is the binding term; compare active waves/CU and tail occupancy,
  not a raw threshold.
- **Persistent tile grid-stride**: choose a resident grid near the useful residency of the target part,
  then let each workgroup consume multiple complete output tiles. This attacks launch/tail overhead and
  does not, by itself, parallelize one tile's K reduction.
- **Persistent Stream-K**: distribute contiguous MAC-iteration ranges over a bounded grid. It attacks
  load imbalance/wave quantization, but partial tiles and their fix-up are part of the cost model.
- **Uniform split-K**: multiply work by a common K partition count and reduce partial outputs. It can fill
  a genuinely underoccupied grid, but the extra operand rereads, atomics/workspace and finalize pass can
  dominate even when occupancy rises.

Classify the current case before selecting among them: compute/MFMA-bound cases protect tile efficiency;
memory-bound cases protect reuse and bytes; reduction-bound cases avoid more partials; launch-bound cases
may favor a bounded persistent grid. A workgroup-count observation is evidence only for the body, shape,
dtype, toolchain and protocol that produced it. Re-measure after any of those changes.

## How to tune (the live lever)
1. Capture real shapes with `AITER_TUNE_GEMM=1` on a warm server (bias must match live calls).
2. Race kernels: `gradlib/gradlib/gemm_tuner.py --indtype bf16 --mp <ngpus>` (gate `err_ratio<0.05`).
3. Deploy by env `AITER_CONFIG_GEMM_BF16=<tuned.csv>`, verify `grep -c 'is tuned on cu_num'`.
Full recipe in [backends/aiter.md](backends/aiter.md). Validated **+2.23% e2e** on Qwen3.5-27B/sglang.

## Pitfalls
- Tuning synthesized `bias=true` shapes when live is `bias=false` → 100% lookup miss, 0 engagement.
- PyTorch TunableOp / `HIPBLASLT_TUNING_FILE` hook a dispatch aiter bypasses → 0 engagement.
- Racing ~1365 hipBLASLt solutions/shape is slow and can fork-storm the host; bucket-reduce M.
- Presenting theoretical peak as a target; the real bar is the best tuned library kernel.

## Sources
- MI300X candidate levers (mfma shape, grid sizing, layout-compatible tile multiples, XCD placement,
  Tagram): ROCm workload guide. Grid and tile values remain shape/decomposition-specific candidates.
- bf16/fp8 peak-utilization ceilings; ~45% MI300X sustained (third-party, narrowing): ScalarLM bench + arXiv 2510.27583 (cited in numerics.md).
- Live tuning recipe + +2.23%: `ROCm/aiter@HEAD`, perf_knowledge e2e run 2026-06-08 (see backends/aiter.md).
- HIP/C++ 8-wave ping-pong FP8 2680@4096 / 3204@8192 (>hipBLASLt 3130, no asm), MI355X ROCm 7.1: AMD CDNA4 GEMM blog + 4-wave interleave blog (cdna4-gemm-kernels, 4wave-fp8gemm).
- hipBLASLt FP8 bar ~2750@4096 / ~3130@8192 (MI355X, ROCm 7.1); Gluon FP16 1489@98.75%: AMD Gluon GEMM tutorial.
- HipKittens BF16 1610 TFLOPS, wave-spec-fails-on-CDNA, 8-wave/4-wave scheduling origin: HipKittens arXiv 2511.08083.
