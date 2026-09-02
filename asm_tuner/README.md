# Canonical asm_tuner integration

`asm_tuner` is executable infrastructure for the two gfx942 GEMM tasks, not a synthetic config
package. The canonical contracts are:

- `manifests/tasks/dense_bf16_gemm_fused.json`
- `manifests/tasks/dense_fp8_gemm_rowwise.json`

Each contract binds the task digest to `task_runner.py:CASES`, the task input builder, the frozen
correctness function, protocol-v2, and the oracle manifest. Route and search-space manifests are
separate checked-in data.

The phase-independent bridge is `kernel_workflow/scripts/config_search.py`:

1. `search-asm` compiles declared variants and measures them as paired arms on the task's semantic
   input slots under measurement-protocol/v2. It must run inside `gpu_lock.sh` and
   `gpu_fence_run.sh`.
2. `create-asm` compiles a declared correctness seed without claiming it was tuned.
3. `select` re-hashes the release manifest, generated source, compile manifest, and shared artifact.
4. `materialize` installs immutable bytes in `.geak_tuner/releases/`, publishes one complete
   `geak.runtime_table/v2` with a single atomic replace, and writes the reviewable
   `asm_tuner_selection.json` candidate change.
5. `verify-materialization --rebuild` re-copies the task into an independent workspace, recomputes
   task/source/protocol/oracle/artifact/table hashes, rebuilds source, compares ISA, and loads the
   selected symbol.
6. `smoke` calls the canonical task wrapper and requires a dispatch receipt naming that symbol and
   config before accepting task correctness.

The BF16 and FP8 48-hour lane manifests set `asm_tuner_mode=required`. On a fresh state the lane
runs the manifest's bounded seed-vs-nondefault protocol-v2 bootstrap search, then atomically commits
the selection before measuring its baseline. It re-runs search/create/select/materialize under the
lock+fence before every Engineer; a missing or stale release prevents the Engineer from starting. The verifier independently materializes and rebuilds
that same registry id. Task calls consume the versioned runtime table through the task-local trusted
loader. Shapes outside the exact eleven-case suite use the explicitly declared canonical fallback.

Current supported executable axes are the geometry, LDS buffers/padding, vector widths, WGM,
stagger, and store mode represented in the generated body. PGR, PLR, and XCC must remain at
`1`, `0`, and `1`; GSU and StreamK must remain `1` and `0`. Schema-survival records may contain
other values, but adapters reject them as executable kernels. Current GPU evidence is recorded in
`evidence/gfx942_canonical_smoke_20260902.json`.
