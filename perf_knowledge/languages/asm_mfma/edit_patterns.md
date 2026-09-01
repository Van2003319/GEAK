---
title: Post-compilation edit patterns — what to change, and the discriminator that decides
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, int8]
regimes: [both]
status: competitive
updated: 2026-09-01
sources:
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
  - https://llvm.org/docs/AMDGPU/gfx9_waitcnt.html
  - https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX940.html
  - https://arxiv.org/abs/2511.08083
---

# Post-compilation edit patterns

## TL;DR
A catalogue of edits to already-compiled AMDGCN, **indexed by the static stall signal that points at
the window** so lookup is mechanical. Read [editing_compiled_asm.md](editing_compiled_asm.md) first:
its three pre-flight gates close most of these before they are worth building.

Every entry carries a **discriminator** — the precondition that separates a win from a null. That
field is the point of this page. Naming a pattern ("relax a wait") predicts almost nothing; the
discriminator ("only where the loads it guards share a destination register range") predicts most of
it.

**Measured base rate — 63 window edits, two gfx942 GEMM lanes, 8 paid.** Counts are by the edit id as
recorded, so the table also shows where the vocabulary split:

| recorded edit id | tried | paid | what separated them |
|---|---|---|---|
| `relax_waitcnt` | 23 | 3 | the guarded loads share a destination range (D1) |
| `hoist_global_load` *(+ `hoist_load_above_use` — same edit, two ids)* | 14 | 4 | supplied occupancy > ~2 WG/CU (M1) |
| `reorder_independent_valu` | 5 | 0 | nothing found; MFMA order is load bearing (M4) |
| `widen_global_load` / `widen_global_store` | 4 | 0 | alignment was provable; bank pressure inverted it (N1) |
| `remove_conversion_burst` / `reduce_instruction_count` / `remove_branch_divergence` | 5 | 0 | epilogue dynamic share too small (Gate 1) |
| `none_observable` | 4 | 0 | cache-policy and arbitration axes, closed on paper (A3, A6) |
| `reduce_barriers` | 2 | 0 | barrier count is handshake arity (A4) |
| `reduce_branches` | 2 | 0 | guards were SALU beside a 16-deep MFMA run |
| `reduce_vgpr` / `remove_spill` | 2 | 0 | the loop owns the high-water mark |
| `reuse_load_destination` | 1 | 0 | destinations were already distinct |
| `critical_arm_probe_symmetric_delay` | 1 | (1) | a **diagnostic**, not a candidate — see M0 |

Two entries in the common folklore — *remove a redundant `s_nop`* and *drop an
`accvgpr_read`/`write` pair* — were **never attempted** in 63 rounds. Both turn out to be misstated
rather than merely untested: every `s_nop` in default compiler output is correctness-required (O4, and
the legitimate version is O5), and the accvgpr problem is largely fixable with two compiler flags
without touching assembly at all (O3).

Note the two synonym pairs (`hoist_global_load`/`hoist_load_above_use`,
`reduce_branches`/`remove_branch_divergence`) and the four ids with no checker behind them
(`widen_global_store`, `reduce_instruction_count`, `reuse_load_destination`,
`critical_arm_probe_symmetric_delay`). A split id makes the same edit look like two experiments and
is the main way a table like this degrades.

**Entry markers:** ★ measured win · ○ realized, free, and measured null · ✗ measured loss · ◇ never
attempted here. A ○ is still useful — it names a mechanism this body does *not* have, which is what
keeps the next round from re-buying it.

---

## M0. Diagnostics — buy the answer instead of guessing it

Not every assembly edit is a candidate. Some are the cheapest way to *learn which arm to optimise*,
and they are worth a round on their own.

**Symmetric delay injection.** On a producer/consumer (phase-offset) kernel, inject an identical
per-stage delay into each arm in turn and compare the cost. The arm that costs more is the one
arriving at the barrier last, i.e. the critical one. Measured here: an identical 256-cycle per-stage
injection cost **+14.93%** in the producer arm and **+6.57%** in the consumer arm — so the producer is
critical, *and* both arms are sensitive, so the pair is nearly balanced and there is little slack
anywhere. That one probe explained why every subsequent producer-side wait edit on that body was null.

The same idea generalises: if you cannot tell whether a window is on the critical path, make it
**worse** by a known amount and measure. A window that costs nothing when slowed is a window that will
gain nothing when sped up.

---

## Signal: `drain` — a full-count `s_waitcnt` serialising the pipeline

### D1. Relax a drain caused by DESTINATION REUSE  ★ pays
- **Signal.** `waits.drain_ratio == 1.0`, `waits.relaxed == 0`, ≥4 memory waits in the window.
- **Why this is the only relax_waitcnt pattern that pays, mechanically.** On CDNA the compiler is
  **not** being conservative with `vmcnt` — `counterOutOfOrder(LOAD_CNT)` returns false
  unconditionally on gfx942/gfx950, so `SIInsertWaitcnts` already emits a precise count (see
  [hazards_and_waitcnt.md](hazards_and_waitcnt.md)). A `vmcnt(0)` is therefore genuinely required
  *given the register assignment*. There is no conservatism to recover; there is a **register
  assignment to change**, after which the exact count is looser by itself.
- **Discriminator — read the destination operands before anything else.** The pattern fires only when
  N loads in the chain write the **same** register range. That is the register allocator's
  serialisation surfacing as a wait. If the destinations are already distinct, the count is exact and
  there is nothing to recover.
- **Hard precondition.** No `flat_*` outstanding in the window: a pending FLAT forces *both*
  `vmcnt(0)` and `lgkmcnt(0)` on GFX9-generation parts. `global_*` / `buffer_*` / `scratch_*` are
  strictly in-order against each other and relaxable. Count **stores** too — VM_CNT is shared with
  stores on CDNA (no VSCNT), so a forgotten store is an off-by-one.
- **Edit.** Give the loads distinct destinations *first* (rotate onto 4 quads, or add a second staging
  set), *then* replace `vmcnt(0)` with a staircase. **A counter edit alone on a reused destination is
  a no-op or a correctness break** — the wait was doing real work.
- **Evidence.** 16 serial `global_load_dwordx4` on one quad → 4 quads + `vmcnt(3)`: **+2.44% faster**
  on the route, zero resource change. A second-destination-set depth-2 staging on another body:
  **2.69% lower route latency**, reproduced five times. **Null every time the destinations were already
  distinct** — four separate bodies, all bit-exact and resource-free, all inside their route MDE
  (+0.24%, +0.37%, −0.03%, −0.13%).
- **Anti-signals.** (a) destinations already distinct; (b) the drain sits in the shadow of ≥8 MFMA, so
  it is already covered; (c) the body is at the memory roofline (Gate 2); (d) ≥4 waves/SIMD, where
  another wave covers it (Gate 3); (e) the wait is a **pre-barrier** drain protecting another wave
  group's panel re-entry — relaxing that is a data race, not a relaxation; (f) the wait was placed for
  **fence or atomic ordering** — `applyPreexistingWaitcnt` never weakens those because they are
  memory-model obligations, and they still look satisfiable from register dataflow alone, so this one
  fails silently.
- **Verify.** `grep -c 'vmcnt(0)'` in the window before/after, and confirm every operand actually read
  next is still covered by the new count.

### D2. Delete a provably-dead `s_waitcnt`  ○ free, worth nothing
- **Signal.** A `s_waitcnt vmcnt(N)` at a block head where **both** predecessors — preheader and latch
  — reach it having already drained at their final `ds_write`.
- **Edit.** Delete it. An already-satisfied `s_waitcnt` costs exactly one issue slot.
- **Evidence.** Deleted across five symbols (census 7→5, 7→6, 9→8, 17→15, 10→9) and measured nothing.
- **Use.** Keep the deletion because it is free and it keeps the wait census monotone for the next
  round's guard test. Do not fund it as a lever.

### D3. Unmerge a CFG join to license a tighter count  ○ measured null
- **Signal.** A `vmcnt` count tighter than steady state because the compiler merged a predicated path
  and took the conservative arm's count.
- **Edit.** Delete the `s_cbranch_execz` so the count is unconditional, then relax one level.
- **Evidence.** Bit-exact, resource-free, worth nothing: the loads were issued a full 65-instruction /
  32-MFMA block before their consumer, so the wait was already covered.
- **Note.** The *mechanism* is real — `SIInsertWaitcnts` does merge pessimistically at CFG joins.
  Whether recovering it is worth anything is a separate question, and here it was not.

### D4. Relax an `lgkmcnt` drain  ○ where the slack actually is, and still null here
- **Where the compiler really is conservative on CDNA.** `lgkmcnt` shares one timeline across SMEM and
  LDS, so with both in flight the pass emits `lgkmcnt(0)` where `lgkmcnt(i)` would do — its own header
  comment says so. Unlike `vmcnt` (D1) this is genuine compiler slack.
- **But the hardware limit is real.** CDNA3 §4.4 and §8.3: scalar memory reads return **out of order**.
  If the window genuinely mixes `s_load*` with `ds_*`, `lgkmcnt(0)` is required by hardware and is not
  recoverable at any effort.
- **Evidence — recovering the slack still bought nothing.** Converting full `lgkmcnt(0)` drains into
  exact staircases was measured on four bodies: 4/3/2/1/0 over four `ds_read2_b64` (+0.24% suite,
  inside the null band, −0.9% on one decode route); a 2/1/0 read-order swap (+0.37% slower, exactly
  null); the module's worst instance, 15 of 16 MFMA issuing out of a drain converted to a perfect
  3/2/1/0 with 12 MFMA of cover (−1.6%, inside its own spread); and full drains per K stage 2→1 with
  MFMA-out-of-drain 14→4 (−1.3%, null).
- **Why.** All four ran at ≥4 waves/SIMD — Gate 3. The slack was real, the recovery was real, and
  another resident wave was already covering the drain. Every row's reopen condition is *occupancy
  falls to 1–2 waves/SIMD*.
- **Reading for a new body:** `lgkmcnt` slack is worth pursuing only where Gate 3 says intra-wave
  scheduling is a currency at all. Otherwise it is a correct edit with no payer.

---

## Signal: `mfma_starved` — a matrix op issuing straight out of a drain

### M1. Sink the next-stage global-load group to the EARLIEST LEGAL SEAM  ★★ the biggest measured win
- **Signal.** At the head of a K-stage block, an address-computation + `global_load` group issues
  *before* the `ds_read` burst that feeds the first MFMA.
- **Discriminator — supplied occupancy, not block shape.** Both winners ran on rungs supplying **>2
  WG/CU**; all three nulls/losses supplied ~1.68. A block-census screen (staged loads × head-burst
  length × K-step depth) predicted the exact opposite ordering and was falsified at >6σ.
- **Edit.** Move the whole group to *immediately after the last `ds_read` of the first burst*.
  **Position response is monotone and the earliest legal seam is the argmax** — measured
  +3.75% / +3.00% / +0.91% at increasing sink depth on one body.
- **Enabler (register-free).** If an address temp aliases a `ds_read` destination, remap it onto the
  **low pair of its own load's destination quad** — the compiler already does this for one load per
  group. Zero new VGPRs. **Check for the alias first**: on two of five rungs the temps were already
  disjoint and the whole edit was a bare 12-line move.
- **Evidence.** **+7.8%** on the 128×64×64 rung (n=20v20 ABBA, t=9.31, 20/20 paired positive, median
  1.064) and **+3.75%** on 64×128×128 (reproduced independently at n=8v8). Exactly 0.0% on 32×64×256
  and 32×32×256; **−1.0%** on 128×128×64.
- **The effect is variance-shaped, not just mean-shaped.** Control latency was bimodal with a heavy
  slow tail (sd 3.91%); the candidate is unimodal (sd 1.42%). Best-decile moved only ~2% while the
  slow mode disappeared — **a best-of-n or min-latency estimator under-reads this by ~2 points.**
- **Anti-signal.** A body where the staged loads must complete before a `ds_write` in the *same*
  iteration and only 2 loads are in flight: there is no surplus slack, so delaying issue directly
  lengthens the intra-iteration critical path (measured −1.0%).
- **Currency.** Not memory latency. Every tested position shortens load-to-use distance, yet the
  shallowest sinks win — the gain comes from clearing the first issue slots at the head of the block,
  immediately after the barrier where all waves rendezvous and nothing else can issue.

### M2. Sink deeper than the first burst  ○ dominated
Seam after the first MFMA quad: +3.00%. Seam after the first 8-MFMA half: +0.91%, null against the
route's bar. Monotone decay with depth — never the position to choose. Deeper seams are bounded above
by this null.

### M3. Split reads so each half re-issues earlier  ✗ refuted
Splitting `ds_read2_b64` into `ds_read_b64` so each 64-bit operand half re-issues the instant its last
consumer retires (slots 2/3/5/6 rather than 6 and 7), buying the second half 5 MFMA of cover instead
of 1: mechanism verified present in the linked object, measured **+0.09%**. The head-of-block currency
does not generalise to a mid-block burst — that burst is already inside a 16-deep MFMA run another
wave covers. Do not re-fund narrowing `ds_read2_b64` on issue-timing grounds.

### M4. Permute the MFMA run  ✗✗ refuted twice, and it is also a correctness hazard
- **⚠ Read this before permuting anything.** CDNA3 §7.5 grants **zero** wait states between an XDL
  write and an XDL read of SrcC only when "vDst and vSrc start from the same offset and same VGPR
  size." Break that exact match into a *partial* overlap and the requirement jumps to **9 or 17** wait
  states depending on pass count — and because the hazard recogniser does not run at assembly time,
  nothing inserts them for you. A permutation can therefore be simultaneously slower and wrong. Table
  in [hazards_and_waitcnt.md](hazards_and_waitcnt.md).
- Regrouping 16 MFMA srcA-major instead of the compiler's srcB-major, a pure permutation with an
  identical instruction multiset: **−1.03%**. The lane's own note guessed SrcC bank contention; the
  exact-match rule above is a second candidate and fits the sign. Neither was isolated, so treat the
  *magnitude* as measured and the *mechanism* as open.
- A one-slot legal improvement inside the same run: +0.46% against a 3.43% bar. The arithmetic bounds
  it: freeing any two of four operand quads requires exactly 6 of the 8 first-half MFMAs to have
  retired, so slot 6 is a **hard floor** and the compiler was already at slot 7.

---

## Signal: `narrow` — an access narrower than the widest this kernel already uses

### N1. Widen an access  ✗ the naive rule is inverted on store paths
- **Legality first — and the thresholds are lower than folklore says.** Full table in
  [hazards_and_waitcnt.md](hazards_and_waitcnt.md). The three that matter here:
  `global_load_dwordx4` needs only **4-byte** alignment on gfx942, so merging contiguous
  `global_load_dword` needs *no new alignment proof*; `buffer_*` needs **natural (16 B)** alignment
  *and* changes out-of-bounds behaviour, because four `buffer_load_dword` are each clamped against
  `num_records` while one straddling `dwordx4` is not; and the destination VGPR tuple must be
  **even-aligned** (`v[5:8]` is rejected by the assembler, `v[6:9]` is fine).
- **⚠ Widening shifts every downstream counter.** Four loads become one, so every `vmcnt(N)` later in
  the same bracket drops by three. This is the easiest silent correctness bug in the catalogue.
- **A worked negative, with a caveat on its own reasoning.** LDS row stride `0x108` = 264 B and
  264 mod 16 = 8, so every odd row base is only 8-byte aligned; the lane closed six structurally
  perfect `ds_write2_b64` → `ds_write_b128` merges on paper for zero GPU time on the grounds that
  16-byte alignment was not provable. ◇ **That bar may have been too high** — LLVM's ISel rule for
  128-bit DS access is `Align(8)`, not 16, precisely because an 8-byte-aligned 16-byte access is
  expressible. The merge may have been legal after all. It would still very likely have lost on bank
  pressure (next bullet), so the conclusion survives while the stated reason does not.
- **Then bank pressure, which is the part that surprises.** Where the merge *was* legal (correctness
  PASS proves the alignment), widening six `ds_write2_b64 … offset1:1` to `ds_write_b128` cost
  **+3.26%**; eight conversions on a relaxed variant cost +0.96%. The split 8-byte pair spreads LDS
  bank pressure across cycles that a single 16 B/lane access collides on.
- **Read the `narrow` flag as a question, not a finding.** On a store path into LDS it has now been
  measured as a loss twice.
- **Where widening is still the right instinct:** a genuinely scalar `global_load_dword` on a
  contiguous stream — and per the alignment note above, that one needs only 4-byte alignment, so it is
  usually legal on sight. This is the case [triton_amd/isa_verify.md](../triton_amd/isa_verify.md)
  describes, and AMD's own ISA checklist asks for it ("ensure `global_load_dwordx4` is used in the
  ISA, especially when the global memory load happens in the loop"). It did not occur in these lanes —
  every load was already `dwordx4`, which is why the only measured instances were LDS *stores*, where
  the rule inverts.

### N2. Fold per-store address arithmetic into the 13-bit immediate  ○ free, measured null ×3
- **Signal.** Repeated 64-bit address adds feeding stores whose offsets are constant-separated. The
  compiler's own use of `v_or` rather than `v_add` **proves** the low index bits are zero.
- **Edit.** Drop the extra address vectors; put the separation in the store's immediate offset.
- **Evidence.** −21 epilogue instructions, zero resource cost, bit-exact: −0.248% alone, −0.365%
  stacked with N3, −0.011% re-measured after a change that raised the epilogue's dynamic share ~1.5×.
  Three independent nulls retire the dynamic-share reopen for that window.
- **Do not price it as a register enabler.** It freed exactly zero VGPRs — see the high-water-mark
  note in [editing_compiled_asm.md](editing_compiled_asm.md).

---

## Signal: `overhead` / `nop_cycles` — moves and conversions doing no arithmetic

### O1. Collapse exec diamonds (branch-free de-predication)  ○ bit-exact, free, null
- **Signal.** Repeated `s_and_saveexec_b64` / `s_xor_b64` / `s_andn2_saveexec_b64` / `s_or_b64` around
  single stores or scalar conversions.
- **Edit.** Replace with `v_cmp` + `v_cndmask_b32`. Leave any diamond that ends in `s_cbranch_execz`
  to a different block.
- **Evidence.** 59 of 60 collapsed: 177 EXEC writes removed, block −15.6% instructions, resources
  byte-identical, output bit-exact on four shapes — **+0.076%**. Separately, 216 conversion sites at
  14→11 instructions each: suite did not move. Gate 1 predicted both before the build.
- **⚠ Correctness trap, the only one measured in 63 edits.** Deleting the branch-condition producer
  alongside the branch faults with a write to nil: consecutive conversions are back to back and
  nothing between them names `vcc`, so a bounded *instruction-count* backward scan silently deletes
  the **preceding** conversion's still-live compare and corrupts that block's exec. **Bound the scan
  at the previous `s_or_b64 exec`, never by instruction count.**

### O2. Replace a shift/mask/sdwa pack with `v_perm_b32`  ○ bit-exact, free, null
Four VALU of bf16 packing → two `v_perm_b32` (selector `0x07060302`), −32 of 283 epilogue
instructions: **−0.333%**. An 11.3% cut in epilogue issue slots returning under a third of a percent
is the signature of a window that is latency- and occupancy-bound rather than issue-bound.

### O3. Remove an `accvgpr_read`/`write` round trip  ◇ untested here — and try two compiler flags first
Since gfx90a the ACC bit lets MFMA read and write architectural VGPRs directly, so an AGPR round trip
is a register-allocation artifact rather than an ISA requirement (CDNA4 §7). LLVM is known to emit
spurious ones at large tiles: [llvm#131954](https://github.com/llvm/llvm-project/issues/131954)
reports that on gfx942, pushing a matmul block past 128×256 fills the inner loop with
`v_accvgpr_read_b32` / `v_accvgpr_write_b32` and "degrade[s] performance back to a level of 128x128."
The maintainer diagnosis is a register-coalescer failure that lengthens live ranges.

**Do not start in the assembly.** AMD's own gfx950 Gluon GEMM tutorial measures the same problem and
fixes most of it with two flags:

| configuration | TFLOPS | VGPRs | MFMA efficiency |
|---|---|---|---|
| scheduler only | 1377 | 494 | 87.00% |
| + RA hints (`amdgpu-agpr-alloc=256`, `amdgpu-mfma-vgpr-form=false`) | 1426 | 464 | **95.64%** |
| + `amdgcnas` post-assembly peephole | 1433 | 512 | 98.25% |

Its `performance_philosophy.md` states those two hints alone "close 75–85% of the MFMA-efficiency
gap." A recompile is cheaper than a hand edit and reaches most of the value, so the order is: try
`-mllvm -amdgpu-agpr-alloc=256 -mllvm -amdgpu-mfma-vgpr-form=false`, re-read the ISA, and only then
consider editing what is left.

- **Discriminator — the flags remove *copy traffic*, so first check that copy traffic exists.** Read
  `accvgpr_moves` from the ISA archive (or `grep -c accvgpr`). The Gluon case that gained 8.6
  percentage points was a 494-VGPR kernel whose inner loop was full of AGPR↔VGPR shuffles. **If
  `accvgpr_moves == 0`, there is nothing for these flags to remove.**
- **Worked negative from this lane, and it is the useful half.** The bf16 body has
  `accvgpr_moves = 0`, `num_agpr = 0`, `accum_offset = 128`, and its 64 accumulators live in arch
  VGPRs in `vgprcd` form. An L3 attribution ruled the axis out on the right grounds: **gfx942 has a
  unified register file**, so relocating accumulators into AGPRs leaves the *unified* total — the
  number that sets occupancy — unchanged. A parallel phase2 row hoped an AGPR move would free ≥10
  registers for a third staging set; it would not, for the same reason. Both the flags and the hand
  edit are closed on this body by the register-file architecture, not by measurement.
- **So the ordering rule is:** `accvgpr_moves > 0` → try the two flags before touching assembly.
  `accvgpr_moves == 0` and unified-file occupancy is the binding term → the axis is closed and neither
  a flag nor an edit will move it.
- **Never attempted in 63 rounds here**, consistent with the above. If you do reach a body with real
  copy traffic, the costs to price against: `v_accvgpr_write` → dependent MFMA is 2 wait states, and
  MFMA → `v_accvgpr_read` of its result is `NumPasses + 3` (5/7/11/19 on gfx942).

### O4. Delete a redundant `s_nop`  ✗ no default-build instance exists
**Every `s_nop` in compiler output is correctness-required.** The only padding source in the backend is
`checkMFMAPadding`, gated on `-amdgpu-mfma-padding-ratio`, which **defaults to 0** and returns
immediately when unset; `-amdgpu-snop-padding` is likewise 0 and marked debug-only. Unless the build
passed one of those, deleting an `s_nop` is a correctness change — and since the hazard recogniser runs
pre-emit rather than at assembly time, nothing puts it back.

So the folklore entry "remove a redundant `s_nop` the backend inserted for a hazard your reordering
removed" is only reachable *after* you have actually removed the hazard, which makes it a consequence
of another edit rather than an edit of its own. Never attempted in 63 rounds.

### O5. Rename a destination to remove the hazard, and the NOP with it  ◇ the legitimate version
- **Signal.** An `s_nop N` sitting between an MFMA and a later instruction that **reuses the MFMA's
  destination register tuple** — a WAR hazard created by register allocation, not by the algorithm.
  Canonical shape from [llvm PR #156943](https://github.com/llvm/llvm-project/pull/156943):

  ```asm
  v_mfma_f32_16x16x32_fp8_fp8 v[26:29], v[102:103], v[6:7], v[134:137]
  ...
  s_nop 5                       ; hazard mitigation
  ds_read_b128 v[26:29], v125 offset:10240
  ```
- **Edit.** Point the `ds_read` at a free, **even-aligned** tuple that does not overlap the MFMA
  destination. The `s_nop` then has no hazard to mitigate and can go with it.
- **Why this one is different from O4.** It is a *register* edit, not a schedule edit: the dependency
  is removed rather than the mitigation. Cost is VGPRs, and the granulation cliffs in
  [editing_compiled_asm.md](editing_compiled_asm.md) decide whether that is free.
- **Untested in this lane**, but it is the shape most likely to be a clean win where free registers
  exist, because it removes issue slots *and* stall cycles at once.

---

## Cross-cutting anti-patterns

These fire no keypoint signal, get proposed anyway, and are measured losses.

### A1. Moving VMEM issue across an `s_barrier`  ✗✗ two independent kernels
Hoisting a drain above a barrier: **+1.007% slower** (11/12 legs, both dies). Issuing a prefetch
before the barrier: **+1.580% slower** (t=7.71, 16/16 legs, two dies agreeing to within 0.01%) — the
tightest signal in the whole ledger, and it is a regression. Replicated on a structurally unrelated
skinny split-K kernel at +0.38%, same sign on both dies.

**Mechanism.** A VMEM issue placed before the barrier makes all waves' L2 requests arrive as one
synchronised burst; issuing after lets barrier-release skew stagger them. *The barriers are not just a
sync — they are what staggers the memory traffic.* The paper argument that the hoist is
provably-not-worse (`max(arrive + vm) ≤ max arrive + max vm`) is wrong on real hardware.

**Control that proves both are real:** the two moves *together* measured −0.087% with the dies
disagreeing in sign. Two independently harmful perturbations summing to null is only possible if each
is a real, oppositely-directed shift.

### A2. Deepening the software pipeline (prefetch depth ≥2)  ✗ rung-local, and never escalate
**−5.7%** on 64×128×128 with the mechanism fully realized: 6 outstanding `global_load_dwordx4` per
wave instead of 3, i.e. 96 rather than 48 in-flight per CU at identical total traffic — **the extra
memory-system queue occupancy is worth more than the latency it hides.** Exactly null (+0.4%) on
128×64×64, for 12 VGPRs. Depth-2 is now measured on the two deepest rungs of one family, −5.7% and 0;
neither is a reason to price depth-3.

### A3. `s_setprio` arbitration brackets  ✗ both directions null at 4 waves/SIMD
Priority raised over the MFMA burst: +0.59%. Priority raised over the *memory* phase instead: −0.78%.
Both inside noise on the same route in the same sessions, on the best-supplied rung in the suite
(2.263 WG/CU). Because **both signs** are null, the axis is closed rather than mis-signed: adding
priority does not create issue slots that 4 waves/SIMD was not already filling.

**Why the knob is weaker than it looks.** CDNA3 §12.5: the overall wave priority is
`{SPIPrio[1:0], UserPrio[1:0], WaveAge[3:0]}` — `s_setprio` sets only the **middle** field, so it is
dominated by the SPI-assigned priority and only breaks ties ahead of wave age. Note also that LLVM's
`AMDGPUSetWavePriority` pass is off by default, so an `s_setprio` already in a CDNA kernel came from
source intrinsics or hand-written asm, not from the default pipeline.

**Where it does get used in production**, for contrast: hipBLASLt/TensileLite gates it on the
`StorePriorityOpt` parameter — `s_setprio 3` at kernel entry through the main loop, dropped to `0`
entering the store/epilogue path. Triton's AMD ping-pong scheduler uses only 0/1 around the dot
cluster, to stop a second warp entering the cluster while the first is still in it. ◇ No published
measurement isolates `s_setprio`'s own contribution on CDNA — every reported number bundles it with
barrier and cluster restructuring, so treat any specific attribution as unsourced.

### A4. Removing a barrier in a warp-specialised kernel  ✗ refuted by construction
gfx942 has **no named barriers**, so producer/consumer kernels phase-offset panels instead of
double-buffering. The producer and consumer loops each execute exactly two `s_barrier` per K stage and
pair 1:1 — *barrier count is the handshake arity.* "This loop contains no `ds_write`, so its barriers
are redundant" is exactly backwards: the writes are on the other wave group, which is **why** the
barriers exist. Both `s_waitcnt lgkmcnt(0)` in such a loop are pre-barrier drains, not data drains.

### A5. Maximal-overlap read schedules at high occupancy  ✗ costs 0.9–1.8%
Rotating all eight `ds_read2_b64` onto private quads and relaxing four MFMA gates from
`lgkmcnt(1)/(1)/(1)/(0)` to `(6)/(4)/(2)/(0)` — the textbook schedule — cost 0.9–1.2% (t=−6.14).
Bursting twelve 1 KiB LDS reads from eight resident waves congests the LDS return path and pushes the
first gate behind ten unwanted returns. **The compiler's tight register recycling is a rate-matched
schedule, not a WAR mistake to be undone.**

### A6. Non-temporal hints on a re-read stream  ✗ null
Tagging loads `nt` produced a real ISA delta and measured −0.03%. The stream was 90 MB of partial
planes written microseconds earlier by the producer kernel and served from the 256 MB Infinity Cache
— marking it non-temporal can only tell the cache to discard what it is already hitting in. **A
traffic-shaped profile does not imply removable or mis-cached traffic.**

### A7. Buying a mechanism with a 2× unroll  ⚠ always run the structure-only control
On one body the unroll itself was the entire ~2% tax (extra branch pairs, a second copy of the
16-MFMA tail, +8 VGPRs) and the relaxation it enabled paid back nothing. On another body a *verbatim*
duplication was exactly free, and 12 extra VGPRs were free because LDS bound occupancy. Rung-local in
both directions — the only way to know is a control arm carrying the same structure with the mechanism
switched off.

---

## How to use this page in a round

1. Run the three gates in [editing_compiled_asm.md](editing_compiled_asm.md). Most windows die here.
2. Look up the window's dominant signal above. Read the **discriminator** before the edit.
3. If the discriminator does not hold, say so and stop — that is a result, and a cheap one.
4. Declare the mechanism **before** measuring. A claim fitted to a result is not evidence.
5. Build a **negative control** whenever the expected effect is under ~3%: a pure permutation with the
   same registers and the same dependency graph, or the same structure with the mechanism switched
   off. In this lane a control caught a "win" that was frame drift, and a zero-code-change arm twice
   caught a false attribution — see [asm_edit_verify.md](asm_edit_verify.md).

## Sources
- AMD CDNA3 ISA Reference Guide — §4.4 counter ordering, §7.5 MFMA wait-state tables, §8.3 scalar memory ordering, §10.2 FLAT counters, §12.5 `s_nop` / `s_setprio` composite priority: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- AMD CDNA4 ISA Reference Guide — §7 Acc VGPR file and the ACC bit: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
- LLVM gfx9 `s_waitcnt` semantics: https://llvm.org/docs/AMDGPU/gfx9_waitcnt.html
- LLVM gfx940/gfx942 instruction syntax (`ds_read2_b64`, `ds_write_b128`, `global_load_dwordx4`, `v_perm_b32`): https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX940.html
- `llvm/llvm-project` — `SIInsertWaitcnts.cpp` (`counterOutOfOrder`, `applyPreexistingWaitcnt`), `GCNHazardRecognizer.cpp` (`checkMFMAPadding`, MFMA wait-state formulas), `SIISelLowering.cpp` (alignment rules), `AMDGPU.td` (`FeatureVscnt`, `FeatureRequiresAlignedVGPRs`): https://github.com/llvm/llvm-project
- llvm#131954 (spurious `v_accvgpr` / spills at large MFMA tiles): https://github.com/llvm/llvm-project/issues/131954
- llvm PR #156943 (hazard NOP removable by renaming a `ds_read` destination — pattern O5): https://github.com/llvm/llvm-project/pull/156943
- AsmEvo (arXiv 2608.20711, AMD — agentic assembly-level optimisation of AMD GPU kernels; hot-window localisation, 5.1% candidate survival): https://arxiv.org/abs/2608.20711
- ROCm gfx950 Gluon GEMM tutorial (`amdgpu-agpr-alloc` / `amdgpu-mfma-vgpr-form` RA hints, measured MFMA-efficiency table — pattern O3): https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
- `hipBLASLt` TensileLite `KernelWriterAssembly.py` (`StorePriorityOpt`: `s_setprio 3` through the main loop, `0` entering the store path): https://github.com/ROCm/hipBLASLt
- HipKittens (arXiv 2511.08083 — inter-wave cover, wave specialisation on CDNA3/4): https://arxiv.org/abs/2511.08083
- Measured evidence: GEAK asmopt lanes, 63 window edits on `macro_bf16_gemm` / `macro_rb_gemm` / `fp8_gemm_tiled` families @ MI300X gfx942, ROCm 7.2.3, amdclang++ 22.0.0git, 2026-08-27..2026-08-31. Every percentage above is a drift-corrected paired same-die A/B; per-row archives under `exp/state_asmopt_bf16_20260827/` and `exp/state_fp8_rowwise_20260827/`.
