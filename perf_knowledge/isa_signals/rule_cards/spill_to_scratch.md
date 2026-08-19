---
title: spill_to_scratch — nonzero private segment
kind: technique
gens: [gfx942, gfx950]
dtypes: [any]
regimes: [both]
status: competitive
updated: 2026-08-18
sources:
  - perf_knowledge/languages/triton_amd/isa_verify.md
  - perf_knowledge/optimization/occupancy_and_registers.md
  - https://llvm.org/docs/AMDGPUUsage.html
---

# spill_to_scratch

**Severity: high — the only non-advisory rule in the table.** Every other card is a judgement call.
This one is not, because the threshold is not a matter of opinion: `isa_verify.md` states the private
segment MUST be 0.

## Signal

`resources.scratch_bytes > 0`, read from `.private_segment_fixed_size` in the AMDGPU metadata of the
binary that was measured.

## Mechanism

The register allocator ran out and spilled to scratch, which lives in HBM. A spill inside a hot loop
converts a register access into a memory round trip, and it does so on the path the loop repeats most.
It also compounds: scratch traffic competes with the loads the kernel actually needs.

## Why the profiler often misses it

It does not present as "memory-bound". Occupancy may look fine, HBM utilisation may look unremarkable,
and the kernel simply runs slower than its arithmetic says it should. That is why this is the first
thing to read off an archive and not something to escalate toward.

## Required source condition

The next edit must reduce live values across the spill point, not merely move code around. Effective
levers, in the order they are usually worth trying:

- fewer simultaneously-live accumulators (tile the N or M dimension of the accumulator block)
- shorter live ranges: sink loads to their use, avoid hoisting everything above the loop
- tile the K loop so operand registers are reused rather than co-resident
- `__launch_bounds__` / `waves_per_eu` to tell the backend the occupancy you actually want

## Anti-signals — when this is NOT the thing to fix

- **A deliberate trade.** Raising `waves_per_eu` to buy occupancy can introduce a small spill on
  purpose. `isa_verify.md` §3 names this explicitly: if setting `waves_per_eu` produced the nonzero
  private segment, you went one step too far — back off rather than restructure the kernel.
- **Prologue-only spill.** Scratch is a whole-kernel figure. A spill on a setup path executed once is
  not the loop's problem; confirm with the loop audit before restructuring.
- **Chasing it below the real limiter.** If occupancy is capped by LDS rather than by registers,
  removing the spill buys nothing until the LDS cap moves too. Both limiters are reported; read both.

## Verifying the fix

Re-capture and diff. `isa_signals.py diff --claim remove_spill` reports `realized` only when scratch
actually returned to 0 — and reports `indeterminate`, not success, when the metadata could not be read.
