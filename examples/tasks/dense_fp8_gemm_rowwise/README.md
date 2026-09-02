# Dense FP8 GEMM (rowwise-scaled) Research Task

`C[M,N]` in bf16 `= (A[M,K] * a_scale[M]) @ (B[N,K] * b_scale[N])^T`, with `A` and `B` in
`float8_e4m3fnuz` -- the gfx942-native fp8 variant, the one `v_mfma_f32_16x16x32_fp8_fp8`
consumes. Scales are fp32, per-token on `A` and per-channel on `B`: the quantisation real fp8
serving ships, and the layout both baseline libraries expect.

This is the fp8 sibling of `dense_bf16_gemm_fused`. Same eleven shapes, same timing discipline,
same probes, so the two tasks' curves are comparable.

## The baseline is measured, not assumed

The denominator is the fastest fp8 GEMM this box can actually deliver under the task's contract.
No single library wins across the suite, so the oracle is a per-shape best-of, frozen from
measurement (two independent passes on an idle GPU picked the SAME winner on every shape):

| shape | M | oracle route | oracle us | seed us | seed vs oracle |
|---|---|---|---|---|---|
| decode_m2_square | 2 | aiter_ck | 29.6 | 56.8 | 0.521 |
| decode_m8_up | 8 | aiter_ck | 43.1 | 65.7 | 0.657 |
| decode_m16_square | 16 | aiter_ck | 28.9 | 59.9 | 0.482 |
| decode_m32_down | 32 | hipblaslt_rowwise | 51.8 | 157.3 | 0.329 |
| decode_m64_square | 64 | hipblaslt_rowwise | 57.6 | 122.1 | 0.472 |
| decode_m96_up | 96 | hipblaslt_rowwise | 52.0 | 73.1 | 0.711 |
| prefill_m128_square | 128 | hipblaslt_rowwise | 32.9 | 59.8 | 0.549 |
| prefill_m256_down | 256 | hipblaslt_rowwise | 70.9 | 165.8 | 0.428 |
| prefill_m512_up | 512 | hipblaslt_rowwise | 97.0 | 285.7 | 0.339 |
| prefill_m1024_down | 1024 | hipblaslt_rowwise | 127.0 | 539.1 | 0.236 |
| prefill_m2048_square | 2048 | hipblaslt_rowwise | 110.4 | 328.9 | 0.336 |

`aiter_ck` is `aiter.gemm_a8w8_CK`; `hipblaslt_rowwise` is `torch._scaled_mm`, which on ROCm
routes to hipBLASLt's rowwise-scaled solutions. Suite geomean: oracle 43 us, seed 0.4384 of it
(re-measured 0.4360, so the seed control is ~0.437 with a sub-1% band).

Taking either library alone would have softened the baseline by 5% (hipBLASLt-only) or 30%
(aiter-only). The full selection evidence, including the paths measured and rejected, is in
`PROVENANCE.json` under `oracle_selection` and explained in `oracle_wrapper.py`.

### Why the fastest path on the box is NOT the oracle

`aiter.gemm_a8w8_bpreshuffle_ck` reaches 42.1 us geomean, beating the best-of. It needs the
weight shuffled into a proprietary layout, which in serving happens once at model load. This
task hands both arms a plain contiguous `B` on every call and lets neither cache anything across
calls, so scoring the oracle on a pre-shuffled weight would compare two different contracts.
With the shuffle inside the timed region the same kernel is the slowest path measured (88.0 us,
2.1x worse than its cached form).

When reporting this task against production serving, quote 42.1 us as the harder reference and
state why. A win over this task's oracle is not automatically a win over aiter's pre-shuffled path.

## Candidate policy

AMD high-performance libraries are forbidden in candidate implementations: candidate source,
wrappers, build flags, imported snapshots and final patches may not call or link rocBLAS,
hipBLAS, hipBLASLt, Tensile, Composable Kernel, MIOpen, or aiter.

`oracle_wrapper.py` and the route manifest are immutable oracle exceptions during a scoring
process. The wrapper is never a candidate implementation, and `oracle_routes.v2.json` freezes the
backend choice, protocol hash and shape inventory. Pass the wrapper to the policy scan explicitly:

```bash
python3 kernel_workflow/scripts/candidate_policy_scan.py <candidate paths> \
    --immutable examples/tasks/dense_fp8_gemm_rowwise/oracle_wrapper.py
```

Allowed candidate building blocks are HIP runtime APIs, HIP kernels, compiler/device intrinsics
including MFMA, and the rocWMMA header-only fragment abstraction -- header-only, nothing linked.

## Current seed

`src/custom_fp8_gemm.hip`: a library-free rocWMMA seed. One workgroup computes four adjacent
16x16 output tiles, staging one A tile plus four B tiles in LDS on a deliberately simple
synchronous schedule, with the rowwise dequantisation fused into the epilogue. It is the
starting point for register blocking, wider staging and load/compute pipelining -- not an
attempt at a good schedule. At 209 TFLOP/s on the largest prefill shape against the oracle's
622, and MI300X fp8 peak near 2600, there is room in both directions.

Two things the seed already gets right, and a rewrite must keep:

- **The fragment is 16x16x32, not 16x16x16.** At BlockK=16 rocWMMA emits the *same*
  `v_mfma_f32_16x16x32_fp8_fp8` with half its K lanes idle, so the natural translation of the
  bf16 seed would have thrown away half the MFMA. Verified by reading the emitted ISA.
- **The arch guard is `__gfx942__`/`__gfx950__`.** Guarding on the wrong macro compiles the
  body to nothing, i.e. a no-op that returns an allocator-recycled buffer and scores an
  unbounded fake speedup. That is not hypothetical -- it happened on the bf16 task. The poison
  write probe in `scripts/task_runner.py` is what catches it, and it reports conclusively on all
  eleven shapes today.

`kernel_loader.py` pins `--offload-arch` to this box's GPU. This image exports
`PYTORCH_ROCM_ARCH` as a nine-target fan-out, and rocWMMA's fp8 fragment does not compile for
gfx950; left alone the build failed depending on who invoked it. Pinning also cuts every
candidate rebuild from nine device compiles to one.

## Measurement

The task uses `geak.measurement-protocol/v2`; the formal default rotates real B weights and their rowwise scales. Candidate, aiter CK and hipBLASLt rowwise run in the same deterministic paired block on the same slot. Reports retain raw samples, slot/order, output addresses and protocol hash. Single-call latency (`inner=1`) is the only score metric; sustained throughput (`inner>1`) is reported separately.

The checked-in manifest preserves the historical table above but identifies it as an unversioned write-flush selection. Formal performance therefore requires a locked `oracle-bakeoff` first and fails closed on a stale protocol hash. See `e2e_workflow/MEASUREMENT_PROTOCOL_V2.md`.

## Gates

Run as user `yxh`; set `ROOT` to the exact clean worktree that will be launched:

```bash
ROOT=/absolute/path/to/the/expected/GEAK-worktree
cd "$ROOT"
GEAK_GPU_ALLOWED=5 bash "$ROOT/kernel_workflow/scripts/gpu_lock.sh" 5 \
  bash "$ROOT/kernel_workflow/scripts/gpu_fence_run.sh" \
  python3 "$ROOT/examples/tasks/dense_fp8_gemm_rowwise/scripts/task_runner.py" compile
GEAK_GPU_ALLOWED=5 bash "$ROOT/kernel_workflow/scripts/gpu_lock.sh" 5 \
  bash "$ROOT/kernel_workflow/scripts/gpu_fence_run.sh" \
  python3 "$ROOT/examples/tasks/dense_fp8_gemm_rowwise/scripts/task_runner.py" correctness
GEAK_GPU_ALLOWED=5 bash "$ROOT/kernel_workflow/scripts/gpu_lock.sh" 5 \
  bash "$ROOT/kernel_workflow/scripts/gpu_fence_run.sh" \
  python3 "$ROOT/examples/tasks/dense_fp8_gemm_rowwise/scripts/task_runner.py" \
    oracle-bakeoff --confirm-oracle-bakeoff
```

Run `performance` through the same lock after the v2 manifest has been baked.

Correctness checks, in order: both oracle routes against library-free fp32 dequant truth; the
candidate against that same truth on `layout_sanity` (M=32, N=48, K=64 -- N is a multiple of the
16-wide fragment but NOT of the seed's 64-wide block, so the last wave of every workgroup falls
entirely outside the matrix and an unmasked candidate writes out of bounds); then five draws per
suite shape against the live oracle, a poison write probe per shape, an output-independence
check, and eight negative tests.
