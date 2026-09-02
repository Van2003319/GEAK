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

## Metrics are separate and the score is explicit

`score_metric` is a hashed `geak.score-metric/v1` object. It binds the metric id, unit, `inner`, and
measurement-protocol hash. The task CLI accepts `--score-metric`; automation may set
`GEAK_SCORE_METRIC`. If both are present they must agree. Omitting both is supported only as the
legacy-lane compatibility default and resolves to `single_call_latency`.

- `single_call_latency`: exactly `inner=1`, milliseconds per launch.
- `sustained_throughput`: `inner>1` back-to-back launches, milliseconds per call. Formal fresh BF16
  and FP8 lanes use this score with semantic B-weight same-slot rotation.

Both receipts remain separate. `execution_time_ms`, `speedup`, TFLOPS, route selection, geomeans,
successive halving, and promotion all come from the selected score receipt. The other metric is
embedded as `secondary_metric` report-only evidence and never enters those fields. Legacy latency
lanes retain their historical field values; no throughput result is silently presented as latency.
Existing `time_op` behavior remains available for old tasks.

## Oracle route lifecycle

Both tasks load `oracle_routes.v2.json` once per process. Mid-run file replacement cannot change a
route. A performance run requires a frozen manifest whose protocol hash, versioned score metric, and exact
shape inventory match the active contract; otherwise it fails closed and asks for a bake-off. Oracle
backend routes are selected by that metric; the secondary measurements cannot change the winner.

Run the deliberate bake-off under the repository GPU lock before starting a new lane:

```bash
cd /home/yxh/GEAK_metric
bash kernel_workflow/scripts/gpu_lock.sh 2 \
  python3 examples/tasks/dense_bf16_gemm_fused/scripts/task_runner.py \
  oracle-bakeoff --score-metric sustained_throughput --confirm-oracle-bakeoff
# The FP8 task is independently re-bakeable under the same lock:
bash kernel_workflow/scripts/gpu_lock.sh 2 \
  python3 examples/tasks/dense_fp8_gemm_rowwise/scripts/task_runner.py \
  oracle-bakeoff --score-metric sustained_throughput --confirm-oracle-bakeoff
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

The checked-in oracle route data remains the 2026-09-02 `single_call_latency` bake. It is not
relabeled as throughput evidence. The task and fresh-lane manifests now select
`sustained_throughput`, so integration must run and review one locked full-suite bake-off per task,
then commit the regenerated oracle manifests before launch. Until then, formal throughput search,
materialization, performance, and fresh-lane preflight fail closed.

## Kernel-lane paired promotion adapter

The bundled BF16 and FP8 task runners expose `paired-verify`. It is invoked only by
`kernel_workflow/scripts/paired_verify.py produce` with an immutable absolute task-runner path. The
producer supplies candidate/incumbent roots through `GEAK_PAIRED_*`, propagates the explicit score
metric, holds one repository GPU lock around the complete suite, and seals
`geak.paired-verify-task-receipt/v2`. The raw suite, sealed receipt, gate input, and gate receipt all
bind the same `metric_id` and `inner`. Every route uses a common
slot count, so its candidate and untouched incumbent observations have one shared outer-block
inventory across the suite. The adapter derives session pair IDs; no role/model reports them.

Oracle arms are optional and marked `absolute_report_only`. A stale or unavailable oracle manifest
removes that absolute report but does not change or veto candidate-versus-incumbent promotion.
