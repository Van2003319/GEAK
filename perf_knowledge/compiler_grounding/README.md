---
title: Compiler grounding — Tier 0 of the compiler-source escalation
kind: index
gens: [gfx942, gfx950]
dtypes: [any]
regimes: [both]
status: competitive
updated: 2026-08-20
sources:
  - kernel_workflow/roles/compiler_engineer.md
  - kernel_workflow/docs/compiler_grounded_evidence_ladder.md
---

# Compiler grounding

The layer the compiler role reads **before** it rebuilds anything and long before it reads any backend
source. Three artifacts, in the order they are used:

1. [`navigation_map.md`](navigation_map.md) — which pass owns which decision. Turns a pass name from
   `ir_signals.py find-changes` into the right subsystem to ask about. Built from the passes actually
   observed on this image, not from recall.
2. [`question_templates.md`](question_templates.md) — turns an IR symptom into one question with a
   stopping condition. An unnarrowed compiler investigation has no end and will consume the round.
3. [`invariants/`](invariants) — settled facts about this backend and this toolchain. A matched
   invariant answers the question outright, at zero rebuild cost.

## Why this layer exists

The evidence ladder's deepest step used to go straight from "the machine code looks like this" to
rebuilding with diagnostic flags, and then to reading backend source that this image does not ship. So
its most expensive tier was also its least likely to produce anything, and the tier that should have
answered most questions was missing entirely.

These are three different kinds of knowledge and they are separated on purpose:

- the **map** is stable and about the compiler,
- the **templates** are about how to ask,
- the **invariants** are measured facts that close a question outright.

Collapsing them into one file would mean every consultation reads all three, which is how a knowledge
layer becomes a per-round tax.

## The standing rule for adding to this directory

Every entry names how it was established — a probe with its output, an ISA reading, or a captured
trajectory. Nothing here may be written from what the backend is believed to do. The role that reads
this directory escalated to it *because it had run out of things it could check*, so an entry it cannot
distinguish from a verified one is worse than no entry at all.
