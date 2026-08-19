# Dense BF16 GEMM Fused Research Task

This source-only task consolidates the reproducible dense BF16 GEMM work that was previously distributed across `exp/`. The original experiment directories remain unchanged and are referenced by `PROVENANCE.json` and `research/index.json`.

## Candidate policy

AMD high-performance libraries are forbidden in candidate implementations. In particular, candidate source, wrappers, build flags, imported snapshots, and final patches may not call or link rocBLAS, hipBLAS, hipBLASLt, Tensile, Composable Kernel, or MIOpen.

The direct rocBLAS implementation in `src/rocblas_baseline.cpp` is the immutable baseline/oracle exception. It is used only for correctness and performance comparison and is not a candidate implementation.

Allowed candidate building blocks are HIP runtime APIs, HIP kernels, compiler/device intrinsics including MFMA, and the rocWMMA header-only fragment abstraction when the resulting artifact does not depend on a forbidden library.

## Current seed

The active candidate starts from the M=1 specialization:

- exact shape `(M,N,K)=(1,4096,4096)`;
- aligned `uint2` BF16 loads;
- four independent FP32 accumulator chains;
- wave64 shuffle reduction;
- no AMD high-performance library call in the specialized kernel.

Every non-M1 shape now reaches `src/custom_gemm.hip`, a candidate-owned synchronous rocWMMA/LDS correctness seed. Candidate and oracle compile into separate ELF artifacts: the candidate artifact has no rocBLAS object or link flag, while the immutable oracle artifact is built by `oracle_loader.py`. The synchronous seed is expected to be slow; it exists to make every scored shape eligible for library-free optimization rather than silently falling back to the oracle.

## Archived library routes

The historical rocBLAS solution-cache and cached hipBLASLt routes are indexed for research only. They must not be built, benchmarked as candidates, or selected as final implementations.

## Measurement

`harness_lib.py` is a frozen copy of the dispatch-primed timer. Its SHA256 is recorded in `PROVENANCE.json`. GPU commands must run through:

```bash
bash /home/yxh/GEAK/kernel_workflow/scripts/gpu_lock.sh 2 <command>
```

No result is publishable when the timing receipt is absent or unprimed.
