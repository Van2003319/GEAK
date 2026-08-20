---
title: gfx942 instruction availability — what the ISA does not have
kind: invariant
gens: [gfx942, gfx950]
dtypes: [bf16, fp32]
regimes: [both]
status: settled
updated: 2026-08-20
sources:
  - kernel_workflow/roles/tech_lead.md (closed-mechanism ledger, findings 84, 51/52, 113-116)
  - perf_knowledge/isa_signals/learned_rules.md
---

# gfx942: instructions that are not there

A backend invariant is the cheapest possible answer to an L4 question: it settles "why did the compiler
not emit X" with "X does not exist on this part", at zero rebuild cost. Each entry below was
established by a **compile probe or an ISA reading in this repository**, not from a datasheet, and each
names the arch on which it flips.

These were sitting in `tech_lead.md`'s closed-mechanism ledger, where they were reachable only by a
planner reading a table. Collected here so the compiler role can consult them as Tier 0 — which is the
layer the paper says should resolve most questions and which this lane did not have.

## No packed f32 atomic add

`unsafeAtomicAdd(float2*)` does not resolve on gfx942 — "no matching function", settled by compile
probe. There is no packed 2×f32 atomic add instruction to select.

**Consequence for L4:** any question of the form "why did the compiler not vectorize this atomic
accumulation" is closed here. Split-K routes that would need a packed f32 atomic are out on this part
regardless of how the source is written.

**Flips on:** an arch that exposes a packed f32 atomic add.

## `v_mfma_f32_16x16x32_bf16` does not exist on gfx942

The available bf16 MFMA is `v_mfma_f32_16x16x16_bf16`, and it costs exactly 16.000 SIMD-cycles on this
part (measured, finding 113-116). The `...x32...` shape is not selectable.

**Consequence for L4:** "why did the matrix pass not pick a wider MFMA" is closed. No opcode swap
halves the MFMA floor on gfx942, so an MFMA-issue argument has to come from count or overlap, not from
shape.

**Flips on:** gfx950 and later parts that add the wider bf16 shape.

## `global_load_lds` is 4 bytes per lane on gfx942

`__builtin_amdgcn_global_load_lds(size=4)` compiles and emits `global_load_lds_dword`.
`size=16` is a **hard compile error** on gfx942 and compiles clean on gfx950.

**Consequence for L4:** direct-to-LDS async loads move 4 B/lane here, which is *narrower* than the
`global_load_dwordx4` a normal staged load already achieves. A question about why the compiler did not
use the DMA path is closed by arithmetic before it reaches legality.

Second constraint on the same path: the DMA destination is `M0 + inst_offset + lane*4` — 256 contiguous
bytes — so a padded LDS row layout cannot be the destination at all.

**Flips on:** gfx950, where `size=16` compiles.

## How to add to this file

Only from a probe or a reading, with the command and its output. An invariant recalled from what the
ISA "usually" has is worse than no entry: it will be quoted as settled by a role that escalated here
precisely because it had run out of things it could check.
