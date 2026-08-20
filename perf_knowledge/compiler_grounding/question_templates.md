---
title: IR symptom → one narrowed compiler question
kind: reference
gens: [gfx942, gfx950]
dtypes: [any]
regimes: [both]
status: competitive
updated: 2026-08-20
sources:
  - perf_knowledge/compiler_grounding/navigation_map.md
  - kernel_workflow/scripts/ir_signals.py
---

# Turning an IR symptom into a question with a stopping condition

L4 is the most expensive step in the lane and the only one with no natural end: "why is the compiler
doing this" can be researched forever. What bounds it is the question, and a question is only bounded
when it names a pass and a structure. This file is the translation table.

**The test a question must pass:** you can say in advance what an answer would look like, and you can
say what would make it unanswerable. "Why is this kernel slow" fails both. "Which legality condition
stops `si-load-store-opt` from merging these two 4-byte global loads into one 8-byte load" passes
both — the answer is a condition, and the unanswerable case is "the pass never considered them
adjacent", which is itself a finding.

## The form

> Why did **`<pass>`** at **stage A → B** introduce / fail to remove **`<structure>`** in
> `<kernel>`, and what condition must the source satisfy for it to do otherwise?

Three slots, all from `ir_signals.py`. If any is empty the escalation is premature and the honest
return is `inconclusive`.

## Templates

### Width did not survive

Symptom: `widest load falls 16 -> 4 bytes at stage N (P)`, or the census never shows a wide access at
all.

- If P is `SROAPass`: *"Which aggregate did SROA scalarize at stage N, and what would have had to be
  true — alignment, address-taken, or a proven contiguous index — for the wide access to survive as a
  memory operation?"* The wide load frequently never existed; see navigation_map §2.
- If the width is still wide entering `amdgpu-isel` and narrow after: *"Why did instruction selection
  choose N narrow loads over one wide one — which alignment or contiguity fact was not established
  upstream?"*
- If it is narrow throughout and `si-load-store-opt` did not merge: *"Which legality condition stops
  `si-load-store-opt` from merging these accesses — adjacency, alias, or address-space?"*

Existing card: `perf_knowledge/isa_signals/rule_cards/narrow_global_load.md`.

### Sync grew

Symptom: `sync ops +N at stage M (si-insert-waitcnts)`.

*"Which dependency forces `si-insert-waitcnts` to place a wait at this point — and is it the memory
ordering the source requested, or a schedule the scheduler chose?"*

Do not ask why the pass inserts waits in general. It inserts them to satisfy a dependency that already
exists; the question is which one, and that is answerable from the MIR at stage M−1.

### Pointers stayed generic

Symptom: `address_spaces` still shows non-`addrspace(1)`/`addrspace(3)` accesses after
`InferAddressSpacesPass`.

*"Which pointer could `InferAddressSpacesPass` not infer, and what provenance would the source have to
make visible for it to succeed?"* This is a legality question and one of the few that genuinely reaches
Tier 2.

### Scratch appeared

Symptom: `scratch first appears at stage N (prologepilog | virtregrewriter)`.

*"What is the register demand at the point of the first spill, and which live range would have to
shorten for allocation to fit?"* Note this is usually **not** a compiler-legality question — it is a
pressure question with a source answer, so Tier 0 should hand it straight back to the planner. Card:
`rule_cards/spill_to_scratch.md`.

### Accumulator moves survived

Symptom: `accvgpr` still nonzero at the last stage.

*"Which AGPR-to-VGPR move survived `amdgpu-prepare-agpr-alloc`, and what would remove the need for
it?"* Check the cycle first (navigation_map §7): intermediate churn is normal and only the final-stage
count matters. Card: `rule_cards/accvgpr_moves_in_kernel.md`.

### The optimization simply never ran

Symptom: the pass you expected does not appear in `list-stages` at all.

*"Did `<pass>` run and decline, or never see this function?"* This is the one symptom where Tier 0
cannot answer and Tier 1 is the right next step: `-fsave-optimization-record` distinguishes a refusal
(which carries a reason string) from an absence.

## Questions that are NOT for this role

- "Which rewrite should I make." L4 recovers a precondition; the TechLead plans and an Engineer writes.
- "Is this kernel memory- or compute-bound." That is L2, and if it is still open the escalation was
  premature.
- "Why is the generated code slow." Not narrowable, no stopping condition. Return `inconclusive` and
  say the L3 attribution did not isolate a pass.
