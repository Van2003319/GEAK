# Learned ISA rules

Lane-produced candidates for promotion into `isa_signals/rule_cards/`. Written by
`tech_lead.md PHASE=synthesize_isa_lessons`; each entry is backed by machine-code evidence from one
run and names the archive a reviewer can re-check. Nothing here has been reviewed yet — do not treat
an entry as equal in authority to a card.

<!-- source run: dense_bf16_gemm_fused, gfx942, lane rounds 13-15 (wave-local 1-3), 2026-08-18 -->
<!-- archive root: /home/yxh/GEAK/exp/team_dense_bf16_gemm_fused_20260818_195602_10194_27057/dense_bf16_gemm_fused -->

## Price an instruction-count edit against the issue share before building it

- **Signal**: any loop-scoped instruction census that names a removable construct — here
  `s_and_saveexec_b64` 48/479 and `s_waitcnt vmcnt(0)` 18/479 in one K stage, i.e. `waits.drain_ratio
  == 1.0` with `waits.relaxed == 0`.
- **Change**: none yet. This is a **step-0 gate on the plan**, run before an arm exists. Price the
  edit as `(target instructions / loop instructions) × SQ_ACTIVE_INST_ANY share of SQ_WAVE_CYCLES`,
  using the `SQ_WAVE_CYCLES = SQ_ACTIVE_INST_ANY + SQ_WAIT_INST_ANY + SQ_WAIT_ANY` partition on the
  same dispatch. The result is an **upper bound** — it assumes issue is the critical path.
- **Outcome**: on the run's live route the gate priced the named mechanism at 5.8% and the total cut
  the arm actually achieved at 8.9%, against a 3.2% route floor and a 19.27%-issuing wave. The arm
  was built anyway, over-delivered in the ISA (loop 479→259 instructions, `s_and_saveexec_b64` 48→0,
  `vmcnt(0)` 18→1, MFMA density 20.0%→37.1%, VGPR 152→148), and measured **0.0%**. The gate would
  have closed the direction for zero GPU seconds.
- **Applies when**: the loop census comes from a real back edge, and one PMC pass on the target
  dispatch is affordable (it always is, relative to a build+bench arm).
- **Does NOT apply when**: the partition has not been read for *that* dispatch — a share carried
  over from a sibling kernel is not a price. Also does not apply to edits whose payer is residency,
  LDS footprint or memory traffic; those are not bounded by the issue share.
- **Evidence**: round 3 (lane 15), archive
  `…/round_3/engineer_0/` (`pmc_base_m96`, `pmc_arm4_m96`, `loopcensus.py`, `report.md`),
  attribution `…/round_3_isa_attribution.md`.

## ANTI-SIGNAL — `all_memory_waits_drain` justifies nothing on a wait-dominated wave

- **Signal**: `waits.drain_ratio == 1.0`, `waits.relaxed == 0`, ≥4 memory waits in the loop — the
  exact confirm condition of [all_memory_waits_drain](rule_cards/all_memory_waits_drain.md), raised
  here at 18 full `vmcnt(0)` drains per K stage against 20 `global_load_dwordx4`.
- **Change**: the card's direction was taken in full — staging made statically countable, exec
  regions collapsed, drains staggered.
- **Outcome**: **realized and null.** The aimed-at term moved exactly as the card predicts
  (`SQ_ACTIVE_INST_ANY` 6,963,387 → 4,547,869, −34.7%, at identical `SQ_INSTS_MFMA`) and wall clock
  did not: the freed issue slots were re-absorbed as instruction wait (`SQ_WAIT_INST_ANY`
  26.91%→48.62%, `SQ_WAIT_ANY` 53.81%→35.35%). An **accounted** null, not an unexplained one.
- **Applies when**: use this as a *precondition on the card*, not a replacement for it — before
  spending a round on drain removal, check the issuing share. Here the wave issued 19.27% of its
  cycles; removing waits from a wave that is 80.7% idle for other reasons redistributes the idleness.
- **Does NOT apply when**: the partition shows the wave actually issue-bound (high
  `SQ_ACTIVE_INST_ANY` share), or occupancy is high enough that another wave was ready to fill the
  drain. The card remains correct in those regimes; this entry bounds it, it does not retire it.
- **Evidence**: round 3 (lane 15), `…/round_3/engineer_0/report.md`, PMC dirs `pmc_base_m96` /
  `pmc_arm4_m96`, attribution `…/round_3_isa_attribution.md` (clauses C1/C2).

## ANTI-SIGNAL — "relaxed `vmcnt(N>0)` now appears" is not evidence of a win

- **Signal**: `waits.relaxed > 0` appearing in a loop that previously had only full drains — the
  standard read that the pipeline now overlaps.
- **Change**: three arms of the same direction produced different wait shapes on the same body.
- **Outcome**: **inverted.** The two arms that emitted partial-drain waits (6 and 8 relaxed
  `vmcnt(N>0)`) measured strictly **slower** than the arm that emitted **zero** relaxed waits and one
  drain. Relaxed-wait count ordered the arms backwards against wall clock.
- **Applies when**: the wave is latency-exposed for reasons other than VM return (low
  waves/SIMD, barrier-dense staging). Then a staggered wait only moves where the stall is named.
- **Does NOT apply when**: you have an independent measurement that VM return latency is on the
  critical path. Absent that, treat `waits.relaxed` as descriptive, never as a proxy metric — and
  never write it into a direction's discriminating clause as this run did.
- **Evidence**: round 3 (lane 15), arm logs `…/round_3/engineer_0/interleave_m96_arm2.log`,
  `interleave_m96_arm3.log`, `interleave_m96_arm4.log`; ISA in `…/round_3/engineer_0/isa`.

## COMPILER PRECONDITION — an exec region is a scheduling boundary; replace it when you delete it

- **Signal**: after collapsing per-load / per-LDS-write `s_and_saveexec_b64` regions, the loop gains
  `v_accvgpr_*` moves and `s_nop` that were not there before, and `resources.vgpr_count` rises
  (here 152 → 160, 40 accvgpr moves and 6 `s_nop` in a loop that had none).
- **Change**: the AMDGPU backend treats each exec region as a scheduling barrier. Removing the
  regions lets it hoist every load destination into one live range. Inserting a single
  `__builtin_amdgcn_sched_barrier(0)` at the seam restores the boundary at **zero** instruction cost
  and returns the register count.
- **Outcome**: **realized** — worth 2.6% of the loop's instruction budget on this body, and it is the
  difference between a de-predication edit that is register-neutral and one that pays occupancy for
  its own cleanup. (It did not by itself produce a wall-clock win here; see the two anti-signals
  above for why the whole direction was null.)
- **Applies when**: any gfx9xx kernel where predicated global→LDS staging is being de-predicated,
  flattened, or made statically countable, and register count or accvgpr traffic regresses.
- **Does NOT apply when**: the loop was never predicated, or the register rise is caused by real
  extra live values (deeper prefetch, wider tile) rather than by hoisting.
- **Evidence**: round 3 (lane 15), `…/round_3/engineer_0/arms/`, `refuted_countable_bk64_staging_arm4.diff`,
  `…/round_3/engineer_0/report.md`; earlier statement of the same effect in ledger row r1_d0
  (lane round 13), where guard deletion alone drove loop `s_waitcnt` 17 → 22.

## CONFIRMED — re-wave a register-light MFMA body when the grid, not the body, starves the machine

- **Signal**: `resources.vgpr_count` pinning waves/SIMD (152 VGPR → 3 waves/SIMD, LDS 18432 B → 3
  CTA/CU) on a loop the ISA shows is already clean (41.6% MFMA density, zero `s_and_saveexec_b64`,
  relaxed waits present) — i.e. the reordering/ILP axis is provably empty and only residency is left.
- **Change**: instantiate the *same* body on twice the threads (512-thread, 8-wave, WAVES_M=2 ×
  WAVES_N=4, halved per-wave register block) as an additional kernel symbol, host-dispatched only on
  the routes that qualify. Resulting symbol: 78 VGPR, 0 AGPR, 0 spill, 18432 B LDS, 6 waves/SIMD.
- **Outcome**: **realized**, −4.72% on the qualifying route with disjoint candidate/control sample
  sets; **+9.2% slower** on a sibling route running the identical instantiation.
- **Applies when**: occupancy-limited *and* the launch supplies **below ~1.7 CTA/CU**. Route order in
  this run was exactly CTA/CU order: −4.7% at 1.68, inside-floor at 1.13… 2.26, +9.2% at 2.53.
- **Does NOT apply when**: **CTA/CU ≥ ~2.5.** Halving the per-wave register block raises LDS reads
  per MFMA (at FM×FN = 4×2 a wave reads 6 fragments per 16-deep K step to issue 8 MFMA, vs 8 reads
  for 16 MFMA at 4 waves = 1.5× LDS read traffic per MFMA). That tax is unconditional; the wave-count
  benefit is only collected where the grid starves the machine. Ship it behind a dispatch predicate,
  never as a blanket replacement.
- **Evidence**: round 3 (lane 15), patch `…/round_3/engineer_1/best_patch.diff`
  (sha256 `0439e1a53015dd2595b55e84daf8228af95cb7c44300af43085f7f6d70b16ab7`), verify receipts under
  `…/round_3/engineer_1/verify/`; body-side ISA contrast in `…/round_3_isa_attribution.md`.

## Two ISA-evidence validity traps (both silently return the wrong answer)

- **Signal**: (a) `Compile: PASS` with ninja reporting "no work to do" immediately after a workspace
  restore from a tar archive; (b) `mechanism_verdict: refuted` on a patch whose per-kernel diff shows
  every shared symbol byte-identical.
- **Change**: (a) tar preserves mtimes, so restored sources are **older** than the stale `.o` — the
  re-captured "parent" ISA is actually the previous arm's. `touch` does not dislodge it; only
  `mv .torch_ext .torch_ext.stale_$(date +%s)_$$` plus a full rebuild does. Source-to-source lockstep
  checks (e.g. `hip_twin_sync`) prove nothing about source-to-object. (b) when a patch **adds a new
  kernel symbol** instead of editing existing ones, the ISA differ pairs only shared symbols, all of
  which are legitimately unchanged, so the verdict reads `refuted` by construction.
- **Outcome**: (a) an entire re-census returned the arm's ISA under the parent's label before it was
  caught; (b) a verified, correctness-passing, −4.72% patch carried a `refuted` mechanism verdict.
- **Applies when**: (a) **always**, before any ISA capture that follows a workspace restore, branch
  switch, or archive extraction; (b) any patch whose diff introduces a new `__global__` symbol.
- **Does NOT apply when**: (b) the patch edits existing kernels — there a `refuted` verdict is real
  evidence and must be treated as such. Do not use "new symbol" as a general excuse for a refuted
  mechanism; confirm from the diff that no pre-existing symbol changed, which doubles as a clean
  negative control on the other kernels.
- **Evidence**: (a) round 3 (lane 15), `…/round_3/engineer_0/isa_reverify/`; (b) round 3,
  `…/round_3/engineer_1/worker_result.json` + whole-tree `-Rpass-analysis` diff in
  `…/round_3/engineer_1/verify/`.

<!-- source run: dense_bf16_gemm_fused, gfx942, lane wave-6 rounds 1-9 (phase2_asm from round 5), 2026-08-31 -->
<!-- archive root: /home/yxh/GEAK_asmopt/exp/team_dense_bf16_gemm_fused_20260831_073702_587519_1211/dense_bf16_gemm_fused -->

## A full memory drain is only worth relaxing when DESTINATION REUSE caused it

- **Signal**: `waits.drain_ratio == 1.0`, `waits.relaxed == 0` — the confirm condition of
  [all_memory_waits_drain](rule_cards/all_memory_waits_drain.md) — **plus** the discriminator this run
  added: read the destination operands of the loads the drain guards. The rule fires only when N loads
  in the chain write the **same** register quad (the allocator's serialization surfacing as a wait).
- **Change**: give the loads distinct destinations first (rotate depth-4, or add a second staging set),
  then and only then replace `vmcnt(0)` with a staircase. A counter edit alone on a reused destination
  is a no-op or a correctness break.
- **Outcome**: **realized and confirmed** where destinations were reused — 16 serial
  `global_load_dwordx4` on one quad → 4 quads + `vmcnt(3)`, **+2.44%** on the route, zero resource
  change; and a second-destination-set depth-2 staging on another body, in-loop `vmcnt(0)` 2→0,
  **−2.69%** on its route, reproduced five times. **Realized and null** every time destinations were
  already distinct: four separate bodies, all bit-exact, all resource-free, all inside their route MDE
  (+0.24%, +0.37%, −0.03%, −0.13%).
- **Applies when**: gfx9xx, the drained loads share a destination register range, and the consumer is
  in the same block. The check costs one `grep` over the window and is strictly cheaper than a build.
- **Does NOT apply when**: the loads already have distinct destinations and the drain sits in the
  shadow of ≥8 MFMA — then the wait is already covered and relaxing it buys nothing. Also does not
  apply where the body is at the memory roofline (see the roofline entry below) or where the wait is a
  **pre-barrier** drain protecting another wave group's panel re-entry, which is a data race to relax,
  not a conservatism.
- **Evidence**: wave-6 rounds 1 and 5, `…/round_5/engineer_2/` (patch, `runs/`, `objs/`) for the win;
  null counterparts in `…/round_5/engineer_1/arm1.s`+`arm2.s` and `…/round_7/engineer_1/armA_reduce8.s`.

## ANTI-SIGNAL — moving VMEM across an `s_barrier` re-synchronises waves and costs ~1-1.6%

- **Signal**: a mainloop with **two or more `s_barrier` per K stage** and symmetric waves, where a VMEM
  issue or its `s_waitcnt` sits between the barriers with visible slack. Every latency heuristic
  (`waits.drain_ratio`, low cover before the consumer) says to hoist it earlier.
- **Change**: hoist the wait above the first barrier; or split the load into its own block so the issue
  crosses the barrier; or stage it a full iteration early past both barriers.
- **Outcome**: **realized and a regression, on three structurally unrelated kernels.** Wait-hoist
  **+1.007%** (t=+3.24); issue-hoist **+1.580%** (t=+7.71, 16/16 legs, two dies within 0.01%);
  both together an **exact null** (−0.087%, dies opposite in sign) — the cancellation proving each is a
  real, oppositely-directed shift rather than noise. On a second kernel a full-iteration prefetch across
  the barrier cost **+0.38%**, same sign on both dies. All arms bit-exact, all resource-neutral.
- **Applies when**: the barriers are what *staggers* the waves' LDS-write and L2 traffic. Released
  together, the four waves issue as one burst and the LDS write port / L2 serialises them; released
  behind their own VMEM returns they pipeline. Treat barrier count per stage as scheduling structure,
  not just synchronisation, before funding any placement move.
- **Does NOT apply when**: the stage has a single barrier, the CTA is one wave, or the panel is
  double-buffered so the second barrier is gone. Nor does it forbid *depth* changes that keep issue
  order relative to the barriers unchanged.
- **Evidence**: wave-6 round 7 `…/round_7/engineer_2/armA.s`,`armC.s`,`runs/`,`runs2/`; independent
  replication wave-6 round 8 `…/round_8/engineer_0/candidate_skinny_depth2.s`, `…/round_8/engineer_0/runs/`.

## ANTI-SIGNAL — `narrow_lds_access` on a producer store path: widening to `ds_write_b128` lost

- **Signal**: `lds_access_bytes.max < 16` with many LDS ops — the confirm condition of
  [narrow_lds_access](rule_cards/narrow_lds_access.md), here six `ds_write2_b64 … offset1:1` writing
  contiguous 16 B-aligned pairs, i.e. a textbook widening candidate.
- **Change**: convert each pair to a single `ds_write_b128` (alignment proven legal by a correctness
  PASS on the built binary).
- **Outcome**: **realized and a regression** — **+3.26%** on the anchor and **+0.96%** on a relaxed-wait
  variant of the same body, with resource deltas 0/0/0/0.
- **Applies when**: the LDS *store* side of a producer that shares its cycle window with a consumer's
  `ds_read2_b64` traffic. The split 8-byte pair spreads bank pressure across cycles that one 16 B/lane
  access collides on; instruction count is not the metric on this path.
- **Does NOT apply when**: the narrow access is on the **read** side, or nothing else is contending for
  LDS in the same window — the card's original regime. This entry bounds the card on stores; it does
  not retire it.
- **Evidence**: wave-6 round 8, `…/round_8/engineer_1/` (`work/`, `runs/`, `verify/`).

## ANTI-SIGNAL — a scratch spill can be load-bearing; check what counts its reloads before deleting it

- **Signal**: `resources.scratch_bytes > 0` with a nonzero `vgpr_spill_count` — the confirm condition of
  [spill_to_scratch](rule_cards/spill_to_scratch.md).
- **Change**: two things must be checked before an arm exists. (1) **Price it**: the reloads executed
  per output tile against the mainloop's dynamic instruction count. Here 13 scratch ops against ~16.7k
  mainloop instructions is a **0.07% dynamic share**, so deleting the spill *entirely* was worth ~0.1%
  against a 1.1% route MDE — the direction was unfundable before it was built. (2) **Trace the
  counter**: a downstream `s_waitcnt vmcnt(N)` may be counting the *reloads*, not global loads.
- **Outcome**: **realized and a regression, −0.97%** from a pure re-encode (26 scratch ops → 14 by
  coalescing 12 `dwordx2` into `dwordx4`/`x3`), with every nameplate resource unchanged; six swizzled
  wide scratch reads drain more slowly than twelve narrow ones, and a downstream `vmcnt` must retire
  all of them. Separately, **removing** the spill silently turned that entry `vmcnt` into a no-op, left
  a pointer register consumed as garbage, and produced `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`.
- **Applies when**: any gfx9xx body where the spill sites and their reload block are separated by a
  large scheduled region, and the block after the reloads opens with a `s_waitcnt vmcnt(N>0)`.
- **Does NOT apply when**: the spill is inside the hot loop (then the dynamic share is large and the
  card's direction is live), or the register file has genuine free headroom — full elimination here
  needed 23 free VGPRs against a census showing 0, under a VGPR cap that was a *launch* requirement
  rather than an occupancy preference.
- **Evidence**: wave-6 round 6, `…/round_6/engineer_1/` (`cand.s`, `base_r5d2.s`, `logs/`, `runs/`);
  the fault signature at wave-6 round 4, `…/round_4/engineer_1/`.

## ANTI-SIGNAL — at or above the DRAM roofline, MLP and non-temporal edits are null by construction

- **Signal**: a body flagged latency-exposed by wait counters (here ~85% `wait_any`) with the purest
  possible destination-reuse pattern — 4 `global_load_dwordx4` into one quad, four serialised
  `vmcnt(0)`, one load in flight.
- **Change**: **compute effective bytes/second for the dispatch first** (bytes moved ÷ kernel time from
  `rocprofv3`). If it is at or above HBM peak, the stream is being served from LLC/Infinity Cache and
  the wait share is bandwidth *queueing*, not per-wave latency exposure.
- **Outcome**: 101.4 MB in 14.9 µs = **6.81 TB/s effective, above HBM peak**. Rotating to four distinct
  destinations with a 3/2/1/0 staircase (bit-exact, occupancy-free) measured **−0.47% on the kernel
  itself**, inside the control's own leg spread. Adding per-operand `nt` hints produced a **real ISA
  delta** (nt-tagged loads 116→124 on the linked `.so`) and an **exact null** (−0.03%): a non-temporal
  hint can only ask the cache to discard what it is already hitting in.
- **Applies when**: split-K reduce kernels, epilogue passes, and any consumer reading a plane a producer
  wrote microseconds earlier in the same dispatch chain — architecturally single-touch *within* the
  kernel, but a re-touched tile at the LLC level.
- **Does NOT apply when**: the effective rate is well under HBM peak — a sibling body at **2.28 TB/s
  (43% of peak)** genuinely did have latency headroom, and the destination check was the right one to
  run there (it lost for a different reason: see the barrier entry).
- **Evidence**: wave-6 round 7, `…/round_7/engineer_1/armA_reduce8.s`, `armB_reduce8_4.s`, `runs/`.

## COMPILER PRECONDITION — source load placement and `sched_barrier` cannot control cross-backedge rotation

- **Signal**: an in-loop `s_waitcnt vmcnt(0)` that the source says should be covered, where the emitted
  schedule shows exactly **one** load hoisted above the back edge.
- **Change**: the AMDGPU backend's machine-sink / loop-rotation hoists one load across the back edge
  regardless of where the source issues it; splitting the MFMA body to place the load between halves
  only relocates **which** stream pays. `__builtin_amdgcn_sched_barrier(0)` is a *within-region* fence
  and cannot stop it. The cover budget per stage is set by the **barrier structure**, not by source
  placement.
- **Outcome**: **refuted** — the split variant read −0.27% against −3.12% for the unsplit arm in the
  same interleaved frame, and a second split variant **disassembled identically** to the first.
- **Applies when**: any HIP mainloop being software-pipelined from the source side on gfx9xx; use it as
  a reason to do the pipelining **in assembly** (or to change the barrier structure) rather than to
  spend arms on source-side placement.
- **Does NOT apply when**: the loop is fully unrolled (no back edge to rotate across), or the edit
  changes the number of barriers per stage — that does move the budget.
- **Evidence**: wave-6 round 2, `…/round_2/engineer_0/`; the barrier-structure half separately refuted
  at wave-6 round 3, `…/round_3/engineer_0/`.

## COMPILER PRECONDITION — the VGPR high-water mark is owned by the loop; epilogue edits cannot free depth

- **Signal**: `resources.vgpr_count` at or just under a granulation step (here 126 of a 128 cap) on a
  body where a deeper pipeline needs ≥10 more registers, with an epilogue full of foldable address
  vectors that looks like the cheap source of them.
- **Change**: before funding the "enabler", **census which block owns the peak**. Map the peak index to
  its defining block and count how many of the allocated registers the loop touches.
- **Outcome**: **refuted for zero GPU time.** Folding 20 `v_lshl_add_u64` epilogue address vectors into
  the stores' 13-bit immediates worked exactly as described (epilogue 282→261, bit-exact, correctness
  PASS) and freed **zero** registers: `next_free_vgpr` 126→126, because the peak was an MFMA operand set
  *inside* the loop while the epilogue's own peak was 32 registers lower. The loop touched 122 of 126
  indices — there was no 10-register hole anywhere in the body. Timed anyway: **−0.011%**, dies opposite
  in sign, a true zero on the third measurement of the same fold.
- **Applies when**: any register-pressure "enabler" argument on a gfx9xx kernel; the census is a symbol
  scan and costs no build.
- **Does NOT apply when**: the peak genuinely lives in the epilogue, or the body is not VGPR-pinned. And
  do **not** buy the depth by raising `next_free_vgpr` past the granule — 126 granulates to 128
  regs/lane = 4 waves/SIMD here, 136 would give 3.
- **Evidence**: wave-6 round 6, `…/round_6/engineer_0/` (`asm_store/`, `runs/`, `per_case.json`).

## Charge a pipelining arm for its STRUCTURE with a control that realises no extra distance

- **Signal**: a depth-2 (or deeper) staging arm on a mainloop that requires a **2× unroll** to alternate
  staging parity — visible in the ISA as a duplicated MFMA tail, an extra branch pair, and +8 VGPR.
- **Change**: build a **structure-only control**: the same unroll and the same extra destination quads,
  with every loop-head wait left at `vmcnt(0)` so **no additional distance is realised**. The delta
  between control and parent is the structural tax; the delta between arm and control is what the
  distance is actually worth.
- **Outcome**: the depth-2 arm lost 2.2%/4.3% on two dies; the structure-only control lost **1.9%/1.2%**
  — i.e. the unroll was essentially the whole loss and the extra distance was worth **zero**. Without the
  control the round would have concluded "prefetch distance does not help here", which is the wrong
  lesson: the correct one is that any future pipelining scheme on this body must avoid paying for a 2×
  unroll (e.g. by reusing an already-dead quad for the alternate parity).
- **Applies when**: any register-staged pipelining arm whose parity alternation forces an unroll — this
  is the standard shape of depth-2 staging in hand-written assembly and in Gluon-style source pipelines.
- **Does NOT apply when**: the parity is handled by existing double-buffered addressing so the unroll is
  not introduced by the arm; then arm-vs-parent is already the clean comparison.
- **Evidence**: wave-6 round 8, `…/round_8/engineer_2/` (`work/`, `runs/`, `verify/`).
