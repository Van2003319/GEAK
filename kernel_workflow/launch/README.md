# Fresh BF16/FP8 48-hour launch

`kernel_workflow/scripts/lane_48h.py` is the only supported entry point for these
fresh lanes. It resolves committed templates against an explicit worktree root.
Both fresh templates explicitly select the hashed `geak.score-metric/v1`
`sustained_throughput` contract (`inner=16`); omission is reserved for legacy latency lane files.
The launcher refuses before creating state unless the oracle, tuner registry, runtime table, and
workflow all bind that same metric and protocol. `preflight` and `supervise --dry-run` are read-only:
they do not create the state directory, deadline, rendered config, lock file, or log.

## Required inputs

Choose two state paths that have never existed. All four lane paths
(`kernel_path`, `workflow_dir`, `state_dir`, and `sol_shapes`) are resolved below
the same `ROOT`; `HEAD` must equal the full `COMMIT` and the worktree must be
clean. The shared allocation is committed in `gpu_allocation.json`: BF16 uses
2,3,4 and FP8 uses 5,6,7. To adapt physical locks, edit that single file, keep
the sets disjoint, and commit it before calculating `COMMIT`.

The tuner integration publishes, per task, tracked
`asm_tuner_registry.json`, `asm_tuner_runtime_table.json`, and
`asm_tuner_selection.json` files at the task root. The registry's versioned
`launch_contract` binds task/source/oracle digests, protocol and score metric,
task-root-relative table and selection paths plus hashes, the tracked workflow
consumer and task-local loader, and an artifact-set digest. Both workflow entry
points accept the exact `tuner_registry` and `tuner_table` arguments. Content-addressed release
manifests, generated sources, compile manifests, and ELF artifacts remain untracked below the ignored
task-local `.geak_tuner/`; preflight permits that only when every path stays below `ROOT`, every hash
matches, and the tracked registry and table reference the same normalized set.
The table must contain the complete atomic 11-route suite and bind metric id/inner on every entry.
This branch intentionally does not fabricate empty tuner files: preflight stays
red until the real bridge has published them.

Get the allocation end on a host with SLURM tools and convert it to epoch
seconds. The allocation must cover the full 48 hours plus the 15-minute machine
margin:

```bash
END_TIME=$(scontrol show job -o "$SLURM_JOB_ID" | tr ' ' '\n' | awk -F= '$1=="EndTime" {print $2}')
SLURM_END_EPOCH=$(date -d "$END_TIME" +%s)
```

Run every container command as `yxh` with `HOME=/home/yxh`. Preflight checks the
credential file owner and requires mode 0600 or stricter, but never changes it.
It then sends one direct authenticated PONG without writing CLI cache files.

## Bake and commit oracle manifests first

These commands intentionally change each task's `oracle_routes.v2.json`; they
are prerequisites, not part of the launcher and are **not** run automatically.
Run them from the exact worktree that will later be passed as `ROOT`, under a GPU
lock and process-group fence. Use an idle GPU from that lane's allocation.

```bash
docker exec -u yxh -e HOME=/home/yxh yxh_geak bash -lc '
set -euo pipefail
ROOT=/absolute/path/to/the/expected/GEAK-worktree
cd "$ROOT"
GEAK_GPU_ALLOWED=2 bash "$ROOT/kernel_workflow/scripts/gpu_lock.sh" 2 \
  bash "$ROOT/kernel_workflow/scripts/gpu_fence_run.sh" \
  python3 "$ROOT/examples/tasks/dense_bf16_gemm_fused/scripts/task_runner.py" \
    oracle-bakeoff --score-metric sustained_throughput --confirm-oracle-bakeoff
GEAK_GPU_ALLOWED=5 bash "$ROOT/kernel_workflow/scripts/gpu_lock.sh" 5 \
  bash "$ROOT/kernel_workflow/scripts/gpu_fence_run.sh" \
  python3 "$ROOT/examples/tasks/dense_fp8_gemm_rowwise/scripts/task_runner.py" \
    oracle-bakeoff --score-metric sustained_throughput --confirm-oracle-bakeoff
'
```

Review and commit both frozen 11-route manifests (and any required provenance
updates). Their embedded protocol hash must equal
`6c479d8f24864215652dc11c625ab8817dd868aeba273d4ae0fd61881d7523ec`; their `score_metric` must be
`sustained_throughput`, `inner=16`, hash
`c104dd5f35ce8a7d4591d0952265a40ee72cf21e2addcb4bfd7762737dcc12b8`.
Do not relabel the committed latency measurements, and do not bake in one worktree and launch from
another. Rebuild/search, register, and materialize tuner releases only after this re-bake so stale
latency-ranked artifacts cannot enter the throughput lane.

## Publish and commit the tuner contract

Run `prepare` for each task after the oracle is committed. The command publishes the three reviewable
task-root files and leaves caches, releases, snapshots, locks, and receipts below `.geak_tuner/`.
Review and commit only the three published files.

```bash
ROOT=/absolute/path/to/the/expected/GEAK-worktree
TASK=dense_bf16_gemm_fused   # use dense_fp8_gemm_rowwise for the sibling
GPU=2                        # use 5 for FP8
TASK_ROOT="$ROOT/examples/tasks/$TASK"
GEAK_GPU_ALLOWED="$GPU" bash "$ROOT/kernel_workflow/scripts/gpu_lock.sh" "$GPU" \
  bash "$ROOT/kernel_workflow/scripts/gpu_fence_run.sh" \
  python3 "$ROOT/kernel_workflow/scripts/config_search.py" prepare \
    --registry "$TASK_ROOT/asm_tuner_registry.json" \
    --task-manifest "$ROOT/asm_tuner/manifests/tasks/$TASK.json" \
    --workspace "$TASK_ROOT" --cache "$TASK_ROOT/.geak_tuner/cache"
git -C "$ROOT" add \
  "examples/tasks/$TASK/asm_tuner_registry.json" \
  "examples/tasks/$TASK/asm_tuner_runtime_table.json" \
  "examples/tasks/$TASK/asm_tuner_selection.json"
git -C "$ROOT" commit -m "asm_tuner: publish $TASK launch contract"
```

## Read-only preflight / dry run

Example for BF16; use `--lane fp8` and the FP8 paths for its sibling:

```bash
docker exec -u yxh -e HOME=/home/yxh yxh_geak bash -lc '
set -euo pipefail
ROOT=/absolute/path/to/the/expected/GEAK-worktree
COMMIT=<full-40-character-clean-HEAD>
SLURM_END_EPOCH=<epoch-from-scontrol>
python3 "$ROOT/kernel_workflow/scripts/lane_48h.py" preflight \
  --lane bf16 --root "$ROOT" --expected-commit "$COMMIT" \
  --state-dir "$ROOT/exp/fresh_bf16_48h" \
  --tuner-registry "$ROOT/examples/tasks/dense_bf16_gemm_fused/asm_tuner_registry.json" \
  --tuner-table "$ROOT/examples/tasks/dense_bf16_gemm_fused/asm_tuner_runtime_table.json" \
  --slurm-end-epoch "$SLURM_END_EPOCH"
'
```

`supervise --dry-run` runs the identical checks and is also write-free. A green
report covers root/commit/cleanliness, fresh state, cold-start setting, manifests,
tuner task/protocol/score-metric contracts, authenticated API, credential permissions, user/HOME,
lane and GPU flocks, GPU idle/VRAM, disk/inodes, SLURM coverage, and non-overlapping GPUs. A metric
drift is a permanent manifest/configuration failure, not a retryable lane fault.

## Formal start (operator action only)

After both dry runs are green, replace `preflight` with `supervise`. Start each
lane in a separate durable service/session; the two commands may run in parallel
because their committed GPU sets and lane locks are disjoint.

The first successful formal start creates `<state>/.launch/` and persists one
shared deadline. Every wave receives `deadline_epoch=hard_end-30m`; the outer
supervisor terminates the whole process group at `hard_end`. Failed waves get at
most three restarts with 60/300/900-second backoff. Manifest/config failures are
permanent and are never retried. Exhausted failures return nonzero.

A supervisor process restart must be explicit and must reuse all original
arguments plus `--resume`; it reads the existing deadline instead of granting a
new 48 hours. It also resumes the monotonic wave number and the global three-
restart budget.

Persistent files are under `<state>/.launch/`:

- `deadline.json`, `lane.json`, `wave_state.json`
- `supervisor.log`, `wave_NNNN.log`
- `health.jsonl`, `gpu_health.jsonl`, `gpu_use.jsonl`

The watchdog probes the tunnel every 30 seconds and sends an authenticated PONG
every 10 minutes. Three tunnel failures or two authenticated failures terminate
the current wave and route it through the bounded restart policy.
