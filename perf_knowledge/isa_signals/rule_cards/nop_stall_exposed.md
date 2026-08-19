---
title: nop_stall_exposed — requested stall cycles from s_nop
kind: technique
gens: [gfx942, gfx950]
dtypes: [any]
regimes: [compute]
status: competitive
updated: 2026-08-18
sources:
  - perf_knowledge/optimization/mfma_scheduling.md
  - https://llvm.org/docs/AMDGPUUsage.html
---

# nop_stall_exposed

**Severity: advisory.**

## Signal

`nop_stall_cycles` above the budget (default 32): the sum of the immediates on `s_nop` in the kernel.
Counted the same way `asm_loop_audit.py` counts it, so the two tools' numbers stay comparable.

## Mechanism

`s_nop` is the compiler inserting a hardware-required wait for a FIXED-latency hazard — classically an
MFMA writing a register a VALU instruction then reads, where the architecture requires N cycles of
separation. The compiler could find nothing useful to put in the gap, so it stalled explicitly.

Large requested stall cycles therefore mean: a known hazard exists, and nothing was available to hide
it behind.

## Required source condition

**The fix is more work to interleave, not reordering.** This is the part most often got backwards.
Reordering cannot create slack; if there were an independent ready instruction, the scheduler would
already have put it there. What changes the outcome is having more independent work available:

- more unroll, so a second independent accumulator chain exists to interleave
- higher occupancy, so another wave issues while this one is stalled
- more independent accumulator chains (instruction-level parallelism), not more instructions

## Anti-signals

- **A handful of `s_nop` in a prologue.** Whole-kernel figure again; scope to the loop.
- **Reading it as a scheduling defect.** It is the opposite: the scheduler already tried. Proposing "let
  the compiler reorder this" wastes a round.
- **Occupancy is capped elsewhere.** If registers or LDS cap occupancy, "raise occupancy" is not
  available as a lever until that cap moves; check `resources` before proposing it.
- **Busy-cycle MFMA utilisation looking healthy.** It will. `profile_engineer.md` records the reason:
  on gfx942 bf16 the MFMA pipe retires a fixed 512 FLOPs per busy cycle regardless of fragment shape,
  so busy-cycle ratios cannot see the instruction mix and two builds differing only in shape measured
  bit-identical busy cycles while one ran 15% slower. A high MFMA-busy fraction does not contradict
  this signal.

## Verifying the fix

Re-capture and compare `nop_stall_cycles` and the MFMA↔VALU interleave from `asm_loop_audit.py`. There
is no `--claim` for this one: "hide a latency hazard" is not a mechanical before/after predicate, and
inventing one would be a claim the diff cannot honestly judge.
