# FP8 rowwise GEMM lane — progress

## Status, 2026-08-27

**Built and verified. NOT LAUNCHED — blocked on an expired agent credential, not on anything in
this repo.**

The task, the oracle, the lane file and the two-day supervisor are all in place and all three
gates pass on-box. The run cannot start because the AMD Primus LLM gateway rejects the key in
`~/.claude/settings.json`:

```
POST https://global.primus-safe.amd.com:18443/api/v1/llm-proxy/v1/messages
  -> HTTP 401  {"errorCode":"Primus.00009","errorMessage":"Unavailable"}   (1.9 s)
```

The gateway is reachable and answers fast, so this is authorization, not a network problem. The
same failure has already killed the bf16 lane: `coldstart_ladder_20260820`'s session sat in
`Please run /login · API Error: 401 … Retrying in 19s · attempt 7/10`, holding GPU locks 2–7 and
writing nothing to `exp/` for over nine hours. All eight cards read 0% busy.

**To unblock:** renew `ANTHROPIC_API_KEY` (or restore gateway access) in
`~/.claude/settings.json` inside the `yxh_geak` container, then run the launch sequence below.
`fp8_two_day_run.sh` re-checks the credential before it starts and refuses in the first minute
rather than spending two days retrying — which is what the bf16 lane did.

## The baseline is measured, not assumed

No single fp8 GEMM wins across the suite on gfx942, so the oracle is a per-shape best-of, frozen
from two independent passes that picked the same winner on every shape:

| library | geomean | note |
|---|---|---|
| **per-shape best-of (THE ORACLE)** | **43.0 us** | aiter CK for M≤16, hipBLASLt rowwise for M≥32 |
| hipBLASLt rowwise (`torch._scaled_mm`) | 44.3 us | wins 8/11 shapes |
| aiter `gemm_a8w8_CK` | 54.9 us | wins the 3 smallest-M shapes, by 6–13% |
| aiter `gemm_a8w8` (tuned dispatch) | 55.5 us | no tuned config for these shapes, so ≡ CK |
| bf16 `torch.matmul`, for context | 69.2 us | the oracle is 1.60x faster |

hipBLASLt-only would have softened the baseline by 5%; aiter-only by 30%.

Measured and rejected, with reasons, in `PROVENANCE.json:oracle_selection`:

- `gemm_a8w8_bpreshuffle_ck` is the **fastest thing on the box (42.0 us)** but needs the weight
  pre-shuffled and kept across calls. This task hands both arms a plain contiguous B every call,
  so scoring the oracle on a pre-shuffled weight compares two different contracts. With the
  shuffle timed it is the *slowest* path measured (88.0 us, 2.1x worse). **Report 42.0 us as the
  harder production reference: beating this task's oracle is not automatically beating aiter's
  pre-shuffled path.**
- `bpreshuffle_ck` with layout (32,16) runs fast and computes garbage (max rel err 1.4). Every
  path here is correctness-screened before its time is allowed to count.
- `gemm_a8w8_ASM` is int8-only in this build (`get_cfg Unsupported input_type: fp8, out_type:
  bf16`).
- per-tensor scaling cannot dequantise per-row-quantised data (max rel err ~1.0).

## Verified on-box

| gate | result |
|---|---|
| compile | PASS, single-arch (`--offload-arch=gfx942` only) |
| correctness | PASS — oracle vs library-free fp32 truth on both routes (err 0.0); candidate vs fp32 truth on `layout_sanity` (9.25e-06); 55 draws vs live oracle (worst 0.0078 vs tol 0.02); poison write probe conclusive `ok` on 11/11 shapes; independence PASS; 8/8 negative tests raise |
| performance | PASS, `all_primed=true` |
| through `gpu_lock.sh 2,3,4,5,6,7` | PASS — the path the roles actually use, workspace-local `.torch_ext` rebuild included |
| `lane_args.py --check` | OK, 13 arguments, 8 protocol values pinned and matching |
| `fp8_two_day_run.sh` DRY_RUN | refuses at the credential check, exit 4 — the guard works |

Seed control: geomean **0.437** of the oracle (0.4384 / 0.4360 / 0.4380 across three runs, a
sub-1% band), min shape 0.238. Comparable to the bf16 task's seed at 0.3353, so the two runs
start a similar distance below their own baselines. Headroom: 209 TFLOP/s on the largest prefill
shape against the oracle's 622, with MI300X fp8 peak near 2600.

## Two things that would have broken the run, found by building it

1. **`PYTORCH_ROCM_ARCH` is a nine-target fan-out in this image**, and rocWMMA's fp8 16x16x32
   fragment does not compile for gfx950. The build therefore succeeded only when the caller
   happened to export a single arch — it passed under my `-e PYTORCH_ROCM_ARCH=gfx942` and failed
   the moment the pipeline would have invoked it plainly. `kernel_loader.py` now pins the arch to
   the detected device and treats a multi-target value as an image default. Side benefit: every
   candidate rebuild is one device compile instead of nine.
2. **The fp8 fragment must be 16x16x32, not 16x16x16.** At BlockK=16 rocWMMA emits the *same*
   `v_mfma_f32_16x16x32_fp8_fp8` with half its K lanes idle, so translating the bf16 seed's
   fragment shape directly would have thrown away half the MFMA. Found by reading the emitted
   ISA, not by trusting the type.

## Launch sequence, once the credential works

```bash
# in the yxh_geak container on tw008
cd /home/yxh/GEAK_asmopt

# 0. the task is currently UNTRACKED (12 source files, no build artifacts -- .torch_ext/,
#    build/, __pycache__/ and the hipified *_hip.hip are all already gitignored).
#    The lane seeds from kernel_path and its provenance is only checkable if that tree is
#    committed, so commit before launching.
git add examples/tasks/dense_fp8_gemm_rowwise \
        kernel_workflow/lanes/fp8_rowwise_20260827.json \
        kernel_workflow/scripts/fp8_two_day_run.sh \
        PIPELINE_PROGRESS_FP8_ROWWISE_20260827.md
git commit -m "feat(task): rowwise-scaled FP8 GEMM, with a measured per-shape best-of oracle"

# 1. rehearse. Runs every precondition and launches nothing; does not start the clock.
DRY_RUN=1 bash kernel_workflow/scripts/fp8_two_day_run.sh \
    kernel_workflow/lanes/fp8_rowwise_20260827.json 48

# 2. run, detached, so it survives the ssh session
tmux new-session -d -s fp8 \
    "bash kernel_workflow/scripts/fp8_two_day_run.sh \
         kernel_workflow/lanes/fp8_rowwise_20260827.json 48"

# 3. watch
tail -f /home/yxh/GEAK_asmopt/exp/state_fp8_rowwise_20260827/supervisor.log
```

The 48 hours are recorded **once**, in
`exp/state_fp8_rowwise_20260827/.wall_clock_deadline`. A restart resumes the same two days rather
than starting a fresh two — otherwise "two days" quietly becomes "two days per crash". Delete
that file to start a new budget deliberately.

### Why the deadline lives outside the lane

`kernel_lane.js` no longer takes `deadline_epoch` / `no_stop_s` / `max_forced_replans`; the
subsystem was removed when the launcher was decoupled, and those keys are now hard errors before
any GPU work. The lane's only stops left are `dispatched >= budget` and `deep_explore_failed`,
neither of which is a clock. So `budget` is oversized (120, matching the bf16 continuous lane) so
that the *wall clock* ends the run, and the supervisor owns the clock. If the lane still reports
`stopped_by=budget_exhausted` before the deadline, that is a real finding — the search ran out of
routes, not out of time.

Note the container's live `coldstart_ladder_20260820.json` still carries `deadline_epoch` and
`max_forced_replans`. That copy predates the removal; the container's `GEAK_asmopt` is three
commits behind this tree (`kb/` is empty and `sol_guidance.py` lacks `--arch`/`--basis`). The fp8
lane file deliberately uses no key that either version rejects, so it launches on both.

## GPU allocation

`gpu_ids: "2,3,4,5,6,7"` — deliberately the same pool as the bf16 lane. `gpu_lock.sh` flocks per
id, so if that lane ever resumes the two interleave correctly instead of corrupting each other's
timings. Lanes 0 and 1 are **not usable**: `gpu_lock.sh` opens the lock file for write
(`200>$LOCK_FILE`) and `/tmp/team_gpu_locks/gpu_{0,1}.lock` are root-owned mode 0644/0664, so the
flock fails for user `yxh`. That is the problem recorded in `scripts/lockable_lanes.py`, and it is
why the pool is six and not eight.

## Waves

_(append one entry per wave)_
