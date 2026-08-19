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
