---
title: accvgpr_moves_in_kernel — accumulator crossing the AGPR/ArchVGPR boundary
kind: technique
gens: [gfx942]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [compute]
status: competitive
updated: 2026-08-18
sources:
  - perf_knowledge/languages/asm_mfma/register_alloc.md
  - perf_knowledge/languages/triton_amd/isa_verify.md
---

# accvgpr_moves_in_kernel

**Severity: advisory**, and more dependent on scope than any other card here.

## Signal

`accvgpr_moves > 0` — `v_accvgpr_read` / `v_accvgpr_write` are present. `isa_verify.md` §2 lists
"`v_accvgpr_read/write` inside the loop" in the bad column, against "acc stays in AGPR (`a[0:n]`)".

## Mechanism

On CDNA3 the matrix core accumulates into AGPRs. Moving an accumulator between the AGPR and ArchVGPR
files costs instructions that do no arithmetic. In a prologue or epilogue that is the normal cost of
getting values in and out. Inside the K loop it is pure overhead repeated every iteration, and it
usually means the allocator could not keep the accumulator resident — often because ArchVGPR pressure
elsewhere forced it out.

## Required source condition

Reduce the pressure that evicted the accumulator, rather than trying to move the moves:

- fewer live non-accumulator values across the loop body
- a smaller accumulator tile if the block genuinely does not fit
- check `resources.scratch_bytes` at the same time: an accumulator being shuffled and a kernel spilling
  are usually the same pressure problem seen from two sides, and fixing one often fixes the other

## Anti-signals

- **Whole-kernel count, loop-local claim.** This is the decisive one. `checks` counts the entire
  kernel; prologue and epilogue moves are expected and carry no per-iteration cost. **Scope with
  `asm_loop_audit.py` before proposing anything** — an `accvgpr` count that is entirely epilogue
  justifies no change at all.
- **Not gfx942.** The AGPR/ArchVGPR split is a CDNA property. On an arch without a separate accumulator
  file the opcode does not appear, and its absence says nothing.
- **A deliberate epilogue.** Reading accumulators out to store them is what the end of a GEMM does.

## Verifying the fix

`isa_signals.py diff --claim remove_accvgpr_moves` returns `indeterminate` when the parent had none —
the claim is ill-posed rather than refuted, and blaming the engineer for the planner's mistake is a
distinction the diff keeps on purpose.
