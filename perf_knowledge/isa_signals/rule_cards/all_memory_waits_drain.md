---
title: all_memory_waits_drain — every wait fully drains its counter
kind: technique
gens: [gfx942, gfx950]
dtypes: [any]
regimes: [both]
status: competitive
updated: 2026-08-18
sources:
  - perf_knowledge/optimization/memory_pipelining.md
  - https://llvm.org/docs/AMDGPUUsage.html
---

# all_memory_waits_drain

**Severity: advisory.** This one is the most often misread card in the table, so read the anti-signals
before proposing anything.

## Signal

`waits.drain_ratio == 1.0` with at least 4 memory waits: every `s_waitcnt` in the kernel names a
counter value of 0, so no load is ever left outstanding across other work.

Two ISA spellings are counted, and both must be, or the ratio is fiction:
`s_waitcnt vmcnt(0) lgkmcnt(1)` (gfx9, counter named, value in parens) and `s_wait_dscnt 0x1`
(gfx11/12, counter in the mnemonic). ALU-dependency waits (`s_wait_alu`, `s_delay_alu`) resolve a
register hazard rather than a memory op and are deliberately excluded from the ratio.

## Mechanism

A relaxed wait (`cnt > 0`) lets some loads stay in flight while the kernel does other work. A full
drain does not: issue the load, stop, wait for all of it, continue. A loop built entirely from full
drains has serialised its memory behind its compute even when both units were free.

## Required source condition

The edit must create work the wait can overlap WITH, then let the wait be relaxed:

- prefetch the next tile's loads before consuming the current one (the loads must be issued early
  enough that there is something to overlap)
- deepen the software pipeline / raise `num_stages` so a stage's loads and another stage's math coexist
- unroll enough that independent loads exist at all

## Anti-signals — when full drains are STRUCTURAL

- **No slack to relax into.** If every load feeds the very next instruction, there is no independent
  ready operation to overlap with, and relaxing the wait would be incorrect rather than faster. This is
  the common case in short elementwise kernels and it is not a defect.
- **A dependency chain, not a scheduling miss.** `asm_loop_audit.py` states the distinction the counts
  cannot: scan the hot loop for an exposed single-class run and ask whether any independent ready op
  could fill it. No → structural. Yes → scheduling. That judgement needs the dependency context, which
  a ratio does not have.
- **The waits are ALU-dependency waits.** Those are counted separately for exactly this reason; many of
  them means VALU/SALU dependency chains, which unroll or wider vectors hide — not waitcnt relaxation.
- **A parser gap inflating the ratio.** `unrecognised_counted_as_drain > 0` means a wait spelling this
  parser does not know was conservatively counted as a drain. The ratio is then an upper bound; report
  the spelling rather than acting on the number.

## Verifying the fix

`isa_signals.py diff --claim relax_waitcnt` compares the ratio before and after, and returns
`indeterminate` — not success — when either build had no memory waits to take a ratio of.
