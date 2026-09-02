---
title: splitk_streamk_gemm — tuning
kind: technique
operator: splitk_streamk_gemm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-05
sources:
  - https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
  - https://arxiv.org/abs/2301.03598
---

# splitk_streamk_gemm — tuning

## TL;DR
> The decision tree: compute `output_tiles = ceil(M/BM)*ceil(N/BN)`; if it's far below CU count, enable
> stream-K (or split-K for large-K) and size the persistent grid ≈ CU count (304 MI300X / 256 MI350X).
> Otherwise leave it off.

## Do not collapse the decomposition names
- **Uniform split-K**: every output tile is split into the same number of K partitions; all tiles pay
  partial-output storage/atomics and reduction. The grid grows with `output_tiles × split_count`.
- **Stream-K**: a bounded grid consumes balanced ranges of the flattened tile×K-iteration space. A
  workgroup can cross tile boundaries; only partial boundary tiles need fix-up. It is not uniform
  split-K with a different grid constant.
- **Static persistent tile grid-stride**: a bounded resident grid loops over complete output tiles.
  One workgroup owns each tile's whole K reduction unless a separate split mechanism is added, so it
  has no inherent partial-output reduction.

Record which mechanism, body hash, config, toolchain and measurement protocol produced a result. A
loss from uniform split-K's reduction traffic does not close Stream-K balancing or whole-tile static
persistence, and a win from one must not be generalized to the others.

## The levers
- **SPLIT_K** (split-K): number of K-partitions. Larger SPLIT_K → more parallelism but more reduction
  cost; tune 2/4/8/16 against large-K small-M shapes.
- **stream-K grid / `NUM_SMS`**: persistent workgroup count, set ≈ CU count so the flattened work divides
  evenly; this is what kills wave-quantization tails.
- **BLOCK_M/N/K**: smaller tiles raise the tile count (may avoid needing split-K) but lower MFMA
  efficiency; co-tune with the split/stream decision.
- **Reduction mode**: `atomic` (in-place add, lower latency, non-deterministic) vs `workspace +
  reduction kernel` (deterministic, extra pass). See [numerics.md](numerics.md).
- **mfma_16x16 vs 32x32**: 16x16 for small-M skinny tiles (fewer wasted lanes), 32x32 for throughput.
- **num_stages / num_warps / waves_per_eu** (triton): standard pipeline/occupancy.

## Heuristic
- Split-K when K is large relative to M·N and tiles ≪ CUs.
- Stream-K when tiles are slightly above or below a CU-count multiple (wave quantization) — it balances
  the remainder.
- Neither when tiles ≫ CUs (plain dense GEMM, [../dense_gemm/overview.md](../dense_gemm/overview.md)).

## Pitfalls
- Over-splitting K turns a compute-bound GEMM into a reduction-bound one — net slowdown.
- Atomic reduction on bf16/fp16 output risks precision loss and contention; prefer fp32 atomics or
  workspace.

## Verify
- Bench split/stream on vs off for each target shape; accept only on a non-overlapping win vs plain dense.

## Sources
- Triton stream-K / persistent matmul: https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
- Stream-K paper: https://arxiv.org/abs/2301.03598
