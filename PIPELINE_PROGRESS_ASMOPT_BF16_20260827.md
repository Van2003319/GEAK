# asmopt_bf16_20260827 — progress

Lane file: `kernel_workflow/lanes/asmopt_bf16_20260827.json`
State dir: `exp/state_asmopt_bf16_20260827`
Task: fused BF16 GEMM on MI300X, 11-case suite geomean vs the immutable rocBLAS oracle.

Wave rules in force (violating any one voids the wave):
- `examples/tasks/dense_bf16_gemm_fused/oracle_wrapper.py` is the immutable oracle and the
  denominator of every speedup. Never edited. Exempted with `--immutable` in every
  `candidate_policy_scan.py` run.
- Candidates may not link rocBLAS / hipBLAS / hipBLASLt / Tensile / Composable Kernel / MIOpen /
  aiter. rocWMMA is header-only and allowed.
- Baseline / harness / oracle are not modified; the workflow copies its own workspace and
  `kernel_path` stays untouched.

---

## 2026-08-31 04:08Z — wave launched, after clearing a two-writer condition

### What was found before launching

`lane_args.py --check` passed (exit 0, 13 arguments, 8 pinned protocol values). The rendered
`Workflow(...)` was called verbatim, no key added or removed.

The check that mattered was the one before that. `supervisor.log` showed `wave 1 starting` at
03:59:00Z — three minutes old — and the process table showed **two supervisor stacks on this one
lane**:

| pid | what | age at 04:01Z |
|---|---|---|
| 385013/385014 | `claude -p` wave for asmopt_bf16, **parent = 1 (orphaned)** | 1h55m |
| 472643 → 472650 → 473599/473600 | a *second* run_lane.sh → lane_two_day_run.sh → this wave | 4m |

The mechanism, which is a real defect and not a race:

1. The earlier supervisor started its wave 3 at 02:07:31Z.
2. That supervisor died. Its `claude -p` child did **not** — the child is held by `timeout`, and
   a parent's death does not propagate to it. So `supervisor.log` never got a
   `wave 3 exited` line.
3. The controller keys on the presence of a supervisor, saw none for this lane, and started a
   fresh `run_lane.sh`, which started a fresh `lane_two_day_run.sh`, which opened wave 1 — this
   wave — against the **same `state_dir`** as the still-running orphan.

The orphan was genuinely live and productive at that point: eval dir
`exp/team_dense_bf16_gemm_fused_20260831_020930_385497_18181` with `round_2/engineer_{0,1,2}` and
round-5 profiling artifacts, and `STATE.json` + `best/` rewritten at 03:56:57Z. Launching a second
`Workflow` on top of it would have given the lane two writers for hours — the same failure recorded
on 2026-08-21, when a 13-minute overlap was survived only because `gpu_lock.sh`'s per-card flock
absorbed it. The real exposure then, and here, is the second wave committing a round from a stale
read of `STATE.json`.

So the wave was **held**, not launched.

### What changed while that was being resolved

Between 04:01Z and 04:06Z the orphan died on its own — `/proc/385013` gone, and `ps` for the lane
reduced to `473599/473600`. It wrote nothing to `wave_3.log` (still 43 bytes, dated Aug 29), so it
left no account of how it ended; with its supervisor already dead, nothing logged it either. That
is consistent with the transport-death class this launcher already knows about, but it is not
proven, and it should not be reported as proven.

With the lane down to a single writer, the wave was launched. Run ID `wf_817bc6d1-70a`.

### Guard added so this cannot recur

`kernel_workflow/scripts/lane_two_day_run.sh` had **nothing claiming the state dir**. Added two
guards, because there are two distinct things to exclude and the obvious one misses the case that
actually happened:

- `.supervisor.lock` — supervisor vs supervisor. `flock` held on fd 9 for the life of the process,
  so the kernel releases it however the process dies, including SIGKILL. A stale lock is
  impossible.
- `.wave.pid` — supervisor vs an **orphaned wave**. This is the case `flock` cannot see: the dead
  supervisor's lock died with it, but its wave did not. A wave now records its pid while running;
  a starting supervisor refuses if that pid is still alive *and* its `/proc/<pid>/cmdline` still
  names this lane, so a recycled pid cannot wedge the lane forever.

A duplicate supervisor now exits 5. `run_lane.sh` breaks on any rc≠4 and parks on `sleep infinity`,
so that is a clean stop and not a restart storm.

The file was replaced by atomic `mv`, deliberately: two shells were executing it at the time, and
`bash` reads a script by byte offset from an open fd — an in-place edit of a running script makes
it misparse partway through. The rename swaps the directory entry and leaves those shells on the
old inode.

The redundant supervisor `472650` was terminated (single pid, so this wave — its grandchild —
survived and kept running). Note the honest limit of the new guard: it protects against future
duplicate supervisors, but the orphan that caused this incident predated it and had recorded no
`.wave.pid`. It was registered by hand at the time, and this wave has since claimed the slot.

### Wave status

Launched 04:08Z, run `wf_817bc6d1-70a`. Binding stop is machine expiry at **07:29:43Z** (earlier
than the recorded 96h budget, which runs to 2026-09-02 03:57Z), giving this wave about 3h20m.
Because its supervisor was the redundant one and is now stopped, **no wave will follow this one** —
which costs nothing, since the machine expires inside this wave's own window anyway.

Starting point carried in from the seeded continuation: `cumulative_vs_rocblas = 1.5804`,
`budget_used = 28/120`, 127 ledger entries, 336 insights.

_Result appended below when the wave completes._

---

## 2026-08-31 04:59Z — run 1 died at init: `process is not defined`

`wf_817bc6d1-70a` failed after **50.8 minutes**, 4 agents, 399k subagent tokens:

```
▸ kernel-lane: ReferenceError: process is not defined
```

### Cause

`kernel_lane.js` read the ladder ENTRY override as `process.env.GEAK_START_PHASE`.
**Workflow scripts execute in a sandbox with no Node globals**, so that read threw at script init —
before the lane's first agent, taking the whole run down. The parent dispatcher had already spent
51 minutes of Setup by the time it called the worker, so the cost of a one-line sandbox violation
was the entire wave.

The feature looked wired up from every angle, which is why it survived unnoticed:

- `/pipe/asmopt/start_phase.asmopt_bf16_20260827` exists and reads `phase2_asm` (written 02:51Z).
- `run_lane.sh` reads that file and `export`s `GEAK_START_PHASE` for the whole subtree.
- `kernel_lane.js` has a careful comment block explaining the override semantics.

Every visible part was correct. The only thing that failed was the crossing into the sandbox, and
nothing on the outside can see that boundary. Worth noting this almost certainly also killed the
02:07Z orphan silently — it started before the 02:49Z edit but would have re-read the file when it
invoked the worker.

### Fix

`start_phase` is now a real **lane argument**, which is the only channel that reaches a workflow
script. The env var survives as a `typeof`-guarded fallback for non-sandbox callers, so nothing that
worked before stops working:

```js
const START_PHASE_ENV = (typeof process !== 'undefined' && process && process.env)
  ? String(process.env.GEAK_START_PHASE || '') : '';
const START_PHASE_RAW = String(
  A.start_phase != null && String(A.start_phase).trim() !== '' ? A.start_phase : START_PHASE_ENV,
).trim().toLowerCase();
```

Added `'start_phase'` to `KNOWN_ARGS` in **both** `kernel_lane.js` and `kernel_workflow.js` — the
optimize path forwards `{...A}` verbatim, so an argument the worker accepts and the dispatcher does
not is refused before it ever reaches the worker. `test_lane_args.py`: 26 passed.

Also fixed the phase-log reason string, which hardcoded `GEAK_START_PHASE=` as the source. It now
names whichever channel actually supplied the value — the file's own comment insists the run history
must not contradict the run, and after this change the env var is usually not the source.

### Run 2

`wf_17ca6f84-c0c`, launched 05:11Z with 14 args (13 rendered + `start_phase=phase2_asm`), verified
programmatically rather than typed. About 2h18m to machine expiry at 07:29:43Z.

## Two things the operator must fix on `/pipe` — I cannot

`/pipe` is owned by uid 300315 and I am uid 1000, so it is **read-only to me**. A syncer restores
`kernel_workflow/lanes/*.json` and `kernel_workflow/scripts/lane_two_day_run.sh` from `/pipe` about
once a minute. Measured: my lane-file edit was reverted 43s after I wrote it (05:05 → 05:07:08), and
again at 05:09:16.

Consequences, both of which will silently undo work already done:

1. **`start_phase` is not durable.** The key is in the repo copy of the lane file and in the running
   wave's captured args, so *this* wave honours it. The next wave will not. It must be added to
   `/pipe/asmopt/lanes/asmopt_bf16_20260827.json`:

   ```json
   "start_phase": "phase2_asm"
   ```

   Once it is there, `/pipe/asmopt/start_phase.asmopt_bf16_20260827` is redundant and should be
   deleted, so there is one source of truth rather than two that can disagree.

2. **The single-writer guard is gone.** The `flock` + `.wave.pid` guard described in the previous
   entry was reverted at 05:09:16 and `/pipe`'s copy never had it. Until it is applied to
   `/pipe/asmopt/scripts/lane_two_day_run.sh`, **the two-supervisor condition can recur**, which is
   what nearly put two waves on this `state_dir` at 04:01Z. This is the more serious of the two.

The repo-side edits to `kernel_lane.js` and `kernel_workflow.js` are NOT affected — those files are
not in the sync set (verified: mtime 05:04:35 survived two sync cycles).

---

# Wave 6 — 2026-08-31 (supervisor wave 1, run `wf_ebcc128e-4cd`)

Launched 07:36Z, finished 22:30Z. **14h 55m**, 64 agents (63 done, 1 errored), 10.45M subagent
tokens, 2742 tool calls. Config rendered by `lane_args.py --print` and called verbatim — 13
arguments, `--check` exit 0.

- eval dir: `exp/team_dense_bf16_gemm_fused_20260831_073702_587519_1211/dense_bf16_gemm_fused`
- report: `tech_lead_report.md` · patch: `final_patch.diff`

## Result: nothing banked

| | |
|---|---|
| director-verified geomean | **0.99378** (`validation_status=flagged`, `applied_to_original=false`, correctness pass) |
| tech_lead reported | 1.0 |
| rounds | 9 · `stopped_by=deep_explore_failed` |
| budget | 26 of 120 (lane) / `STATE.budget_used` 31 |
| `cumulative_vs_rocblas` | **1.5804265752 — unchanged** |
| git commits since 07:30Z | none |

Phase ladder, both escalations forced by the same trigger: `phase1_source` (r1) →
`phase2_asm` (r5, "4 consecutive non-improving rounds") → `deep_explore` (r9, same).

## Why it produced no gain

The round-8 assembly work was the only thing that got near the incumbent:

- **r8_d1** (producer arm `.LBB34_22` of `macro_rb_gemm_kernel<128,256,64,64,64,16,9,1>`,
  94.77% of `prefill_m1024_down`) — verified, correctness pass, policy pass,
  `verified_geomean=1.6188` against an incumbent `cumulative` of 1.61853. That is **+0.017%**,
  two orders of magnitude under `min_improve=0.005`. Not committed, correctly.
- **r8_d2** (depth-2 register staging of the B global load, `prefill_m2048_square`) — `regression`,
  0.99119.
- **r9_d0** deep_explore (ground-up rewrite of the large-prefill macro-GEMM path to break a measured
  4.56 TB/s panel-delivery ceiling) — `apply_failed`. No candidate existed to time, so no GPU was
  acquired and no arm was run. That failure is what stopped the lane.

`direction_entropy` was **0.16** (4 distinct directions across 25 issued). The search is re-treading
the same few axes; that is the number to watch next wave.

## What the wave did bank: knowledge, not speed

- insights 353 → **497**, hypothesis ledger 132 → **191**, `asm_ledger` **30 rows** (was 0 — this is
  the first wave in this tree to accumulate real machine-code evidence at volume)
- **8 ISA rules promoted + 4 anti-signals** into `perf_knowledge/isa_signals/learned_rules.md`.
  The synthesizer's own reason: five `phase2_asm` rounds produced "mostly bit-exact resource-neutral
  edits whose mechanism was verified in the linked binary before timing" — unusually clean
  confirmed/refuted evidence.
- 9 learned-KB citations filed; none became a winner.

## Wave rules: all three held

`policy_pass=true`, `findings: []`. `oracle_wrapper.py` was in the scan's `immutable` list and
`git status` on it is empty — untouched. No banned library linked. The only policy output was two
**advisory** hits, both the literal string "rocBLAS" inside source *comments* in
`skinny_bf16_gemm.hip` / `skinny_bf16_gemm_hip.hip` — text, not a link.

## The `/pipe` revert predicted in the Run-2 entry happened

`--check` reported **13 arguments**: `start_phase` is gone from the lane file again, exactly as the
previous entry warned it would be. The cost is measurable this time — the wave opened in
`phase1_source` and burned **rounds 1-4 non-improving** before the ladder let it into `phase2_asm`
at round 5, which is where all of its actual evidence came from. Roughly a third of a 15-hour wave
spent re-deriving that the source layer is exhausted.

Still an operator fix, still unchanged: add `"start_phase": "phase2_asm"` to
`/pipe/asmopt/lanes/asmopt_bf16_20260827.json`. The single-writer `flock` guard is likewise still
absent from `/pipe`'s `lane_two_day_run.sh`.

## Next wave

The supervisor holds ~29h against a deadline of 2026-09-02T03:57:13Z and will open wave 2 itself.
Two things worth changing before it matters:

1. `start_phase=phase2_asm` on the `/pipe` copy — otherwise wave 2 pays the same four-round tax.
2. r8_d1 is a real but sub-gate mechanism. It is worth re-running as a paired interleaved
   measurement rather than re-deriving: +0.017% at n=1 is indistinguishable from noise, and the
   question of whether it is a genuine small win has not actually been answered.
