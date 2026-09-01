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

### Wave `fp8_rowwise_20260827` — launched 2026-08-27

- `lane_args.py --check` → exit 0 (13 arguments, 8 protocol values pinned and matching).
- Launched from the `--print` rendering verbatim, no hand-edited keys.
- Run ID `wf_a4e4c667-23d`; transcript under
  `~/.claude/projects/-home-yxh-GEAK-asmopt/fced9c0f-.../workflows/wf_a4e4c667-23d`.
- Config as launched: `mode=optimize`, `budget=120`, `min_improve=0.005`,
  `candidate_floor=0.37` (seed sits at 0.437, so the floor is genuinely below the seed and
  candidates stay visible), `isa_evidence=observe`, `ir_diagnostics=on`,
  `incremental_analyze=true`, `agent_timeout_ms=7200000`, `gpu_ids=2,3,4,5,6,7`,
  `state_dir=exp/state_fp8_rowwise_20260827`.
- Result: _pending — appended below on completion._

### Wave `fp8_rowwise_20260827` (relaunch) — 2026-08-27

The entry above records run `wf_a4e4c667-23d` as pending; that run belongs to an earlier session
and cannot be resumed from here (`resumeFromRunId` is same-session only). Relaunched from the same
lane file rather than adopting a run whose state I cannot inspect.

- `lane_args.py --check kernel_workflow/lanes/fp8_rowwise_20260827.json` → exit 0
  (13 arguments, all accepted; 8 protocol values pinned and matching).
- Invoked verbatim from the `--print` rendering — no key added, removed, or retyped.
- Run ID `wf_db33bc0d-067`; transcript under
  `~/.claude/projects/-home-yxh-GEAK-asmopt/fe256d15-.../workflows/wf_db33bc0d-067`.
- Config identical to the entry above (`mode=optimize`, `budget=120`, `min_improve=0.005`,
  `candidate_floor=0.37`, `isa_evidence=observe`, `ir_diagnostics=on`,
  `incremental_analyze=true`, `agent_timeout_ms=7200000`, `gpu_ids=2,3,4,5,6,7`,
  `state_dir=exp/state_fp8_rowwise_20260827`).
- Task string carries both standing rules: `oracle_wrapper.py` is the immutable denominator and is
  to be exempted with `--immutable` in every `candidate_policy_scan.py` run; rocBLAS / hipBLAS /
  hipBLASLt / Tensile / Composable Kernel / MIOpen / aiter are forbidden in candidates, rocWMMA
  (header-only) is allowed.
- Result: _pending — appended below on completion._

### Wave `fp8_rowwise_20260827` (relaunch #2) — ran 2026-08-31, 07:36 → 16:24 UTC

Preceding entries record `wf_a4e4c667-23d` and `wf_db33bc0d-067` as pending; both belong to
earlier sessions and `resumeFromRunId` is same-session only, so this is a fresh launch against the
same `state_dir` — which is what carried the incumbent forward (the lane resumed from
`STATE_DIR/best` at 1.1706, baseline commit `6e88438`, not from the 0.434 seed).

- `lane_args.py --check kernel_workflow/lanes/fp8_rowwise_20260827.json` → **exit 0**
  (14 arguments, all accepted by the entry points; 8 protocol values pinned and matching).
- Invoked verbatim from the `--print` rendering — no key added, removed, or retyped.
- Run ID `wf_2b4968aa-863`; transcript under
  `~/.claude/projects/-home-yxh-GEAK-asmopt/59e90b6c-.../workflows/wf_2b4968aa-863`.
- Config as launched: `mode=optimize`, `budget=120`, `min_improve=0.005`, `candidate_floor=0.37`,
  `isa_evidence=observe`, `ir_diagnostics=on`, `incremental_analyze=true`,
  `agent_timeout_ms=7200000`, `gpu_ids=2,3,4,5,6,7`,
  `state_dir=exp/state_fp8_rowwise_20260827`.

#### Result — completed, validation `accepted` / trust `verified`

| field | value |
|---|---|
| director-verified geomean (headline) | **1.1764** |
| tech-lead reported geomean | 1.1831 |
| arithmetic mean of per-case speedups | 1.2028 (tech lead: 1.2083) |
| correctness | pass, bit-exact |
| timing basis | device (`rocprofv3` dispatch timing) |
| rounds / budget | 8 rounds, 15 of 120 dispatches used |
| `stopped_by` | `deep_explore_failed` (not the clock — `deadline_hit=false`) |
| wall clock | 8h 49m, 70 agents, 0 errors |
| eval dir | `exp/team_dense_fp8_gemm_rowwise_20260831_073632_1898252_9464/dense_fp8_gemm_rowwise` |

The headline is the **director's** 1.1764, not the tech lead's 1.1831 — the two were measured in
different sessions and the director's is the independently re-timed one. Within-wave gain over the
1.1706 incumbent is therefore about +0.5%, and against the original 0.434 seed the cumulative
program stands at roughly 2.71x.

**Phase ladder:** `phase1_source` (r1) → `phase2_asm` (r3) → `deep_explore` (r8), each entered
because the TechLead's own stop gate fired (floor attacked, no compute case far above it, last
verified gain under ~3%). One cycle, zero forced replans.

**What actually paid.** Three wins banked, all three phase-2 assembly instruction-scheduling edits
inside per-tile-shape MFMA main loops; the shipped artifact of this wave is a single file,
`src/custom_fp8_gemm_hip.s`. No `.hip`/`.py`/`.cpp` source was touched, so the build routes through
`asm_roundtrip.py` restore.

| round | engineer | edit | banked as |
|---|---|---|---|
| 3 | `r3_d0` | hoist head `ds_read2_b64` burst above the K-guard in `.LBB3_346` (64x128x128); target +3.75% on `prefill_m1024_down` | `5606655` |
| 4 | `r4_d2` | 7±7 lines inside the 32x64x256 instantiation; correct and resource-neutral but **measured null** on the suite | `912b46d` |
| 5 | `r5_d0` | re-bank of round 4's dropped `.LBB2_464` sink (128x64x64); +7.77% on `prefill_m512_up`, 20/20 paired positive | `9cec716` |

Rounds 1, 2, 6, 7 banked nothing; round 8's `deep_explore` shipped no patch and closed the wave.

**Per-case, as validated (µs, oracle → ours):**

| route | shape | speedup |
|---|---|---|
| `decode_m2_square` | 2×4096×4096 | 1.931 |
| `decode_m8_up` | 8×11008×4096 | 1.141 |
| `decode_m16_square` | 16×4096×4096 | 1.462 |
| `decode_m32_down` | 32×4096×11008 | 1.051 |
| `decode_m64_square` | 64×8192×8192 | 1.196 |
| `decode_m96_up` | 96×11008×4096 | 1.234 |
| `prefill_m128_square` | 128×4096×4096 | 1.389 |
| `prefill_m256_down` | 256×4096×11008 | 1.010 |
| `prefill_m512_up` | 512×11008×4096 | **0.968** |
| `prefill_m1024_down` | 1024×4096×11008 | **0.953** |
| `prefill_m2048_square` | 2048×4096×4096 | 1.049 |

The two large-M prefill routes remain the only sub-parity cases and are where the remaining
headroom is.

**Bottleneck:** `latency` at wave start and at wave end — issue-slot latency at the head of the
K-loop body, not bandwidth and not occupancy. Every funded bandwidth-, occupancy-, and
tiling-shaped direction measured null or negative. That the bottleneck refused to shift under all
remaining cheap levers is what closed the wave.

**Compliance receipts:**

- `oracle_wrapper.py` unmodified — clean in `git status`, sha256
  `1059f13d2dfb7183b576cb9e655d0a47b0078896a1a1c037bbbc753cc96036d0`; the lane's own
  `oracle_digest` was constant at `161ef808…8239e` across every verify agent in all 8 rounds.
- `final_patch.diff` touches exactly one file (`src/custom_fp8_gemm_hip.s`) and contains zero
  occurrences of rocBLAS / hipBLAS / hipBLASLt / Tensile / Composable Kernel / MIOpen / aiter.

**Caveat carried forward:** rocprof-compute / omniperf are not installed on this host, so SoL, L1
and L2 fields are *absent* from the round metrics, not zero. Any future roofline argument must
install the missing profiler first rather than read absence as a value.

**Knowledge captured:** one new learned card,
`kernel_workflow/knowledge/learned/sink-the-next-global-load-to-the-earliest-legal-seam-in-hand-dense-gemm-gfx942-prefill.md`;
KB artifact written as `dense_fp8_gemm_rowwise__triton__gfx942` (champion, speedup 1.1764).
Direction entropy was low — 4 distinct directions over 14 issued (0.286).

### Wave `fp8_rowwise_20260827` (relaunch #3) — ran 2026-08-31, 16:26 → 20:27 UTC

Same `state_dir`, so the lane resumed from the previous wave's incumbent (1.1831 tech-lead frame /
1.1764 director frame) rather than from the 0.434 seed. Unattended launch.

- `lane_args.py --check kernel_workflow/lanes/fp8_rowwise_20260827.json` → **exit 0**
  (14 arguments, all accepted by the entry points; 8 protocol values pinned and matching).
- Invoked verbatim from the `--print` rendering — no key added, removed, or retyped.
- Run ID `wf_4436a030-e17`; transcript under
  `~/.claude/projects/-home-yxh-GEAK-asmopt/f7af746e-.../workflows/wf_4436a030-e17`.
- Config as launched: identical to relaunch #2 (`mode=optimize`, `budget=120`,
  `min_improve=0.005`, `candidate_floor=0.37`, `isa_evidence=observe`, `ir_diagnostics=on`,
  `incremental_analyze=true`, `agent_timeout_ms=7200000`, `gpu_ids=2,3,4,5,6,7`,
  `state_dir=exp/state_fp8_rowwise_20260827`).

#### Result — completed, **nothing banked**, validation `flagged`

| field | value |
|---|---|
| banked this wave | **none** — `IMPROVED=false`, `WINNER=null`, wave-local gain 1.0000 |
| director-verified geomean vs oracle | 1.1636 (arithmetic 1.1866) |
| tech-lead oracle-relative claim | 1.17653 |
| correctness | pass |
| timing basis | device (`GEAK_TIMING_RECEIPT all_primed=true`) |
| rounds / budget | 5 rounds, **8 of 120** dispatches used |
| `stopped_by` | `deep_explore_failed` (`deadline_hit=false`) |
| wall clock | 4h 01m, 33 agents, 0 errors |
| eval dir | `exp/team_dense_fp8_gemm_rowwise_20260831_162656_914334_984/dense_fp8_gemm_rowwise` |

Seven engineer directions over three funded rounds (r1 ×3 source→asm, r3 ×3 asm, r5 ×1
`deep_explore`); every one measured null or negative. The phase ladder advanced twice on the
TechLead's stop gate (`phase1_source` → `phase2_asm` at r3, → `deep_explore` at r5) and the run
ended the way the ladder says it should: a `deep_explore` round that banks nothing. Budget was
never the binding constraint — 112 dispatches went unspent.

Only 4 distinct directions over 7 issued (entropy 0.571): the search is repeating itself.

#### The one measured win this wave declined to bank

`r5_d0` built a new **128×64×128 (2×1)** rung for the long-K M≥256 corner, guarded by
`blocks(128,64) >= 200 && k > kShortK(6144) && k % 128 == 0`. Engineer measured +2.3% on
`prefill_m1024_down` (n=10/arm, 10/10 paired positive, t=6.36); the verifier independently
reproduced **+3.0%** at n=6/arm with **non-overlapping sample sets** (cand 129.98–132.72 µs vs ctrl
133.06–137.86 µs). Bit-exact on all 11 shapes, all gates green.

It was filed `regression` because the suite metric cannot see it: the guard fires on exactly one of
eleven shapes, so +3.0% there is **+0.27%** on an unweighted 11-case geomean against a measured
session spread of 0.85–1.0% ptp. The `verified_geomean` 0.9973 sits under a stored baseline table
the same-session *control* also could not reach (0.9966) — the ninth recurrence of oracle-denominator
drift on this lane. Precedent says re-bank: the last time this pattern occurred (w5r4 `r4_d0`), the
re-measurement came back **larger**, +4.4% → +7.8%, and it banked.

Patch is staged and re-checked (`git apply --check` rc=0 against the current canonical):
`exp/state_fp8_rowwise_20260827/pending_w6r5_prefill_m1024_down.diff`. Carry-forward scepticism:
the constant `200` is unexplained and fires on one suite shape, so the *guard* reads shape-fitted
even though the *mechanism* is supported by counters and non-overlapping latency distributions.

#### Deliverable defect found by the director, and fixed

The director flagged the wave `flagged` — not for correctness or policy (both clean), but because
`final_patch.diff` shipped only `src/*` and omitted `gemm_wrapper.py`. Applying it to
`KERNEL_PATH_ORIG` therefore reproduced a tree ~1.2% slower than the tree that was actually
measured (director A/B: patch-as-delivered 1.1602 mean vs patch+wrapper 1.1742, B>A in both
adjacent pairs, gap concentrated in decode). That accounts for essentially all of the 1.5%
shortfall between 1.1636 and the tech lead's 1.17653, and for the apparent regression against
relaunch #2's 1.1764 — **the program did not get slower; the patch got thinner.**

Regenerated `final_patch.diff` post-run (original preserved as `final_patch.diff.director_flagged.bak`).
Verified against a pristine copy of `KERNEL_PATH_ORIG`: `git apply -p1 --check` rc=0, apply rc=0,
and the resulting tree reproduces the measured workspace exactly —
`src/custom_fp8_gemm_hip.s` md5 `218ef54bfe8c307d0e07f3a478aad284`, `gemm_wrapper.py` identical,
`oracle_wrapper.py` untouched. `kernel_loader.py` is byte-identical to the original and correctly
absent from the patch. The 1.1636 headline is *not* re-measured after this fix — it should be
re-timed at the top of the next wave, and is expected to land near 1.176.

#### Tool defect found (round 4)

`isa_signals.py keypoint` truncates a kernel span at its **first** `s_endpgm`. Every tiled
instantiation has an early-exit `s_endpgm` ~29 lines in, so `windows_scored` was 1 — the kernarg
prologue — on every hot kernel. Worked around with a line-count-preserving scratch copy
(`windows_scored` 1 → 192, top window then agreed with the COMMANDMENT's named main loop).
**PHASE=keypoint is blind on this lane's hot kernels until this is fixed.**

#### Compliance receipts

- `oracle_wrapper.py` unmodified — clean in `git status`, sha256
  `1059f13d…c96036d0`, unchanged from the previous wave; lane `oracle_digest` constant at
  `161ef808…8239e` across every verify agent in all 5 rounds.
- All **5** `candidate_policy_scan.py` receipts list `oracle_wrapper.py` under `immutable`, and all
  5 report **0 findings, 0 advisory**.
- Regenerated `final_patch.diff` contains **zero** occurrences of rocBLAS / hipBLAS / hipBLASLt /
  Tensile / Composable Kernel / MIOpen / aiter.
- Knowledge: 2 learned cards cited (neither verified, neither became a winner); KB artifact written
  as `dense_fp8_gemm_rowwise__triton__gfx942`, exp `20260831_202712_1e5ceb`.

#### What the next wave should do first

1. **Re-bank `r5_d0`** from `pending_w6r5_prefill_m1024_down.diff`, judged on its declared route at
   n≥8/arm interleaved, with the suite as a tripwire only — not as the gate.
2. Fix the suite-metric blindness that dropped it, or the lane will keep discarding real per-route
   wins: a +3% on 1-of-11 is arithmetically unresolvable against 0.85–1.0% suite noise.
3. Fix `isa_signals.py keypoint`'s `s_endpgm` truncation.
4. Break the direction repetition — 4 distinct of 7 issued, and the schedule level (earliest-legal-seam
   sink family) is closed on all five reachable loops.
