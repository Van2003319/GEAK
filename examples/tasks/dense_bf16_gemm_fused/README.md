# Dense BF16 GEMM Fused Research Task

This source-only task consolidates the reproducible dense BF16 GEMM work that was previously distributed across `exp/`. The original experiment directories remain unchanged and are referenced by `PROVENANCE.json` and `research/index.json`.

## Candidate policy

AMD high-performance libraries are forbidden in candidate implementations. In particular, candidate source, wrappers, build flags, imported snapshots, and final patches may not call or link rocBLAS, hipBLAS, hipBLASLt, Tensile, Composable Kernel, or MIOpen.

The direct rocBLAS and direct hipBLASLt implementations are immutable baseline/oracle exceptions. They are used only for correctness, a locked per-shape oracle bake-off, and performance comparison; neither is a candidate implementation.

Allowed candidate building blocks are HIP runtime APIs, HIP kernels, compiler/device intrinsics including MFMA, and the rocWMMA header-only fragment abstraction when the resulting artifact does not depend on a forbidden library.

## Current seed

The active candidate starts from the M=1 specialization:

- exact shape `(M,N,K)=(1,4096,4096)`;
- aligned `uint2` BF16 loads;
- four independent FP32 accumulator chains;
- wave64 shuffle reduction;
- no AMD high-performance library call in the specialized kernel.

Every non-M1 shape now reaches `src/custom_gemm.hip`, a candidate-owned synchronous rocWMMA/LDS correctness seed. Candidate and oracle compile into separate ELF artifacts: the candidate artifact has no BLAS object or link flag, while the immutable oracle artifact is built by `oracle_loader.py`. The synchronous seed is expected to be slow; it exists to make every scored shape eligible for library-free optimization rather than silently falling back to the oracle.

## Archived library routes

The historical rocBLAS solution-cache and cached hipBLASLt routes are indexed for research only. They must not be built, benchmarked as candidates, or selected as final implementations.

## Measurement

The task uses `geak.measurement-protocol/v2`; the formal default rotates semantic weight operand B instead of writing a synthetic 512 MiB buffer before every sample. Candidate, direct rocBLAS and direct hipBLASLt run in the same deterministic paired blocks and consume the same slot. Raw samples, slot/order, output addresses and protocol hash are embedded in the report. Single-call latency (`inner=1`) is the score; sustained throughput (`inner>1`) is a separate non-score metric.

The checked-in route manifest is deliberately untrusted because the historical BF16 oracle had only rocBLAS. Before a new lane, bake and freeze the per-shape best-of under the GPU lock:

```bash
cd /home/yxh/GEAK_measure
bash kernel_workflow/scripts/gpu_lock.sh 2 \
  python3 examples/tasks/dense_bf16_gemm_fused/scripts/task_runner.py \
  oracle-bakeoff --confirm-oracle-bakeoff
```

See `e2e_workflow/MEASUREMENT_PROTOCOL_V2.md` for protocol and adapter limits. A formal performance run fails closed if the route-manifest protocol hash does not match.
