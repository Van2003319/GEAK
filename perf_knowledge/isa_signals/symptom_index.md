---
title: Profiler symptom → ISA signal → rule card
kind: reference
gens: [gfx942, gfx950]
dtypes: [any]
regimes: [both]
status: competitive
updated: 2026-08-18
sources:
  - perf_knowledge/languages/triton_amd/isa_verify.md
  - perf_knowledge/profiling/reading_a_kernel_bottleneck.md
  - https://llvm.org/docs/AMDGPUUsage.html
---

# Symptom → signal → card

The routing table for `PHASE=isa_attribution`. A profiler tells you a kernel is scalar-heavy,
transfer-heavy or stalled; it cannot tell you which construct in the machine code produced that. This
maps each symptom to the ISA signal that would **confirm or kill** it, so the analyst reads one or two
cards instead of the whole library.

Read this first, then at most the one or two cards it points at. Signals come from
`kernel_workflow/scripts/isa_signals.py signals|checks` over the round's archive; every field named
below is a key in that JSON.

## How to use a row

A row is a hypothesis test, not a diagnosis. The **kill** column matters as much as the confirm column:
a symptom whose ISA signal is absent is a symptom you have just explained *is not* what you thought,
which is the outcome the evidence ladder produces most often and the one that saves benchmark rounds.

| Profiler symptom | Check this signal | Confirms if | Kills the hypothesis if | Card |
|---|---|---|---|---|
| latency-bound, issue-wait dominant | `resources.scratch_bytes`, `resources.vgpr_count` | scratch > 0, or vgpr just over a 16-granule step | scratch == 0 and vgpr well inside a step | [spill_to_scratch](rule_cards/spill_to_scratch.md) |
| memory-bound but HBM utilisation low | `global_load_bytes.max`, `global_load_bytes.histogram` | max < 16 on a contiguous tile | max == 16, or the access pattern is a real gather | [narrow_global_load](rule_cards/narrow_global_load.md) |
| LDS-bound / bank conflicts suspected | `lds_access_bytes.max`, `lds_multi_access` | max < 16 with many LDS ops | max == 16 (`ds_read_b128` already) | [narrow_lds_access](rule_cards/narrow_lds_access.md) |
| low compute/memory overlap, "waiting" | `waits.drain_ratio`, `waits.relaxed` | drain_ratio == 1.0 with ≥4 waits | relaxed > 0, i.e. the pipeline already overlaps | [all_memory_waits_drain](rule_cards/all_memory_waits_drain.md) |
| MFMA pipe fed but kernel still slow | `accvgpr_moves`, `mfma_shapes` | accvgpr moves inside the hot loop | moves are prologue/epilogue only | [accvgpr_moves_in_kernel](rule_cards/accvgpr_moves_in_kernel.md) |
| stalls the counters cannot localise | `nop_stall_cycles`, `nops` | large requested stall cycles | few or zero `s_nop` | [nop_stall_exposed](rule_cards/nop_stall_exposed.md) |
| "this dtype should be faster" | `mfma_shapes`, `conversions` | wrong fragment shape, or a `v_cvt` burst | expected shape and few conversions | [isa_verify.md](../languages/triton_amd/isa_verify.md) §4 |
| a round measured flat and you do not know why | `isa_signals.py diff` vs the parent archive | `unchanged_machine_code: true` | codegen moved | this is the falsification gate, not a card |

## Three standing warnings

**A card's confirm condition is necessary, not sufficient — read what finished runs learned about it.**
`isa_signals/learned_rules.md` carries lessons distilled from completed runs by
`tech_lead.md PHASE=synthesize_isa_lessons`, and its ANTI-SIGNAL entries exist precisely to bound the
cards this table routes to. They are unreviewed and rank BELOW a card: an entry never licenses a
direction a card does not, and its only power is to stop one. Grep it for the card's name before
acting. The standing example is [all_memory_waits_drain](rule_cards/all_memory_waits_drain.md), whose
confirm condition was met in full, whose direction was taken in full, whose ISA moved exactly as the
card predicts — and which measured **0.0%**, because that wave was issuing only 19.27% of its cycles
and removing waits from a wave 80.7% idle for other reasons just relocates the idleness. Reading the
card without the entry buys that round a second time.

**`checks` counts are whole-kernel.** A prologue and a hot loop are not the same evidence. Before
acting on `accvgpr_moves` or a wait ratio, scope to the loop with
`perf_knowledge/expert_skills/skills/gluon_authoring/scripts/asm_loop_audit.py`, which finds the
innermost back-edge and reports per-iteration. An `accvgpr` count that is entirely epilogue justifies
nothing.

**An unavailable signal is not a zero.** `resources.available: false` means `llvm-readelf` could not be
read, not that the kernel uses no registers; `drain_ratio: null` means there were no memory waits to
take a ratio of, not that the pipeline is perfect. `isa_signals.py` never fills these in, and neither
may the analysis — the same rule the SOL branch of `profile_engineer.md` enforces on ceilings, for the
same reason: a confident wrong number outranks a missing one in every later reader's mind.

## What this table is not

It does not rank kernels, produce a headroom figure, or feed the fitness function. Verified geomean
remains the only authority on performance. These signals explain and reject; they never score.
