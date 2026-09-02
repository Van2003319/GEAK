# GEAK measurement protocol v2

## Audit that motivated v2

The BF16 and FP8 task copies were byte-identical to each other but behind the shared harness.
Their formal path called `time_op(..., inner=1, flush_cache=True)` for every arm and every sample.
`flush_cache` rewrote a persistent 512 MiB buffer with `zero_()`, so the eviction workload was much
larger than a short GEMM and did not model the semantic weight traffic. Both tasks then measured a
fixed `B` address using an ad-hoc `B C C B` palindrome. Receipts kept only four aggregate slots.

BF16 had one denominator, direct `rocblas_gemm_ex`; it was not a per-shape best-of. FP8 had a
hard-coded per-shape aiter/`torch._scaled_mm` route selected under the old write-flush procedure,
with no machine-readable protocol identity. The shared and pinned harness copies are now identical.

## Protocol identity

`harness_lib.measurement_protocol()` returns `geak.measurement-protocol/v2` data and a SHA-256 over
all resolved knobs. The default is `semantic_weight_rotating`; the fixed seed is 20260902 unless the
lane explicitly overrides it. Every raw receipt embeds both the protocol and its hash.

Supported cache/operand modes:

- `legacy_write_flush`: fixed operands and the historical 512 MiB `zero_()` before every arm sample.
- `read_evict`: fixed operands and a read reduction over an eviction buffer before every arm sample.
- `warm`: one fixed, warmed operand slot and no synthetic eviction.
- `semantic_weight_rotating` (formal default): shared `A`, rotating real weight operand `B`, no
  synthetic flush. The B pool targets 384 MiB by default, above the 256 MiB cache assumption.

For rowwise FP8, `b_scale` is generated and rotated with its B slot. Slot metadata records A/B and
scale seeds and a semantic fingerprint. Every paired block chooses one slot and runs every arm on
the exact same slot object. A deterministic Latin rotation balances first/last arm position across
cycles. The receipt contains raw samples, block order, slot IDs/fingerprints, host/device clocks,
output addresses and GPU-lock status.

Returned outputs enter a bounded live ring as soon as each call returns. This prevents the caching
allocator from handing every call the same output address; scoring rejects a receipt when multiple
observable outputs still use only one address.

## Metrics are separate

- `single_call_latency`: exactly `inner=1`, milliseconds per launch. This is the only score metric.
- `sustained_throughput`: `inner>1` back-to-back launches, milliseconds per call. It is reported as
  a separate non-score metric and never enters latency speedup or geomean.

The old `baseline_ms`, `candidate_ms`, `execution_time_ms` and `speedup` fields remain as explicit
aliases sourced only from the single-call latency receipt. The corresponding `baseline_tflops`,
`candidate_tflops`, timing legs, and top-level `all_primed`/`timer_unprimed` receipt keys are also
retained for old parsers. Existing `time_op` behavior remains available for old tasks.

## Oracle route lifecycle

Both tasks load `oracle_routes.v2.json` once per process. Mid-run file replacement cannot change a
route. A performance run requires a frozen manifest whose protocol hash and exact shape inventory
match the active protocol; otherwise it fails closed and asks for a bake-off.

Run the deliberate bake-off under the repository GPU lock before starting a new lane:

```bash
cd /home/yxh/GEAK_measure
bash kernel_workflow/scripts/gpu_lock.sh 2 \
  python3 examples/tasks/dense_bf16_gemm_fused/scripts/task_runner.py \
  oracle-bakeoff --confirm-oracle-bakeoff
# The FP8 task is independently re-bakeable under the same lock:
bash kernel_workflow/scripts/gpu_lock.sh 2 \
  python3 examples/tasks/dense_fp8_gemm_rowwise/scripts/task_runner.py \
  oracle-bakeoff --confirm-oracle-bakeoff
```

Run formal `performance` commands through the same wrapper after the manifest is frozen.

The bake-off uses the same semantic slots and paired scheduler as candidate scoring, correctness
screens every backend, saves raw receipts, and atomically replaces the route manifest.

BF16 compares direct rocBLAS and a direct hipBLASLt C++ adapter. The adapter uses row-major
`A[M,K]`, row-major `B[N,K]` with explicit transpose, BF16 output, FP32 accumulation, a zero-workspace
contract shared with the candidate and rocBLAS control, fresh `torch::empty` output per call, and cached plan/workspace outside timing.
The manifest records hipBLASLt algorithm index, solution name, kernel name, workspace, library
version, git revision and architecture. No `preferred_blas_library` global is read or changed.

FP8 can re-bake its aiter CK and hipBLASLt rowwise routes. Those Python entry points do not expose
an algorithm index, exact kernel name, or workspace size, so their manifest metadata says
`not_exposed` rather than inventing an identity. This is the remaining FP8 adapter limitation.

The checked-in BF16 manifest is an explicitly untrusted direct-rocBLAS migration placeholder, and
the checked-in FP8 manifest identifies the historical unversioned write-flush selection. This
change intentionally did not run a suite-wide benchmark. Therefore a fresh 48-hour lane must run
one locked v2 oracle bake-off first; formal performance refuses to reuse either historical route as
if it had been selected under v2.

## Kernel-lane paired promotion adapter

The bundled BF16 and FP8 task runners expose `paired-verify`. It is invoked only by
`kernel_workflow/scripts/paired_verify.py produce` with an immutable absolute task-runner path. The
producer supplies candidate/incumbent roots through `GEAK_PAIRED_*`, holds one repository GPU lock
around the complete suite, and seals `geak.paired-verify-task-receipt/v2`. Every route uses a common
slot count, so its candidate and untouched incumbent observations have one shared outer-block
inventory across the suite. The adapter derives session pair IDs; no role/model reports them.

Oracle arms are optional and marked `absolute_report_only`. A stale or unavailable oracle manifest
removes that absolute report but does not change or veto candidate-versus-incumbent promotion.
