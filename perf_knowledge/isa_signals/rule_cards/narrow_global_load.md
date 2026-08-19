---
title: narrow_global_load — widest global load under dwordx4
kind: technique
gens: [gfx942, gfx950]
dtypes: [any]
regimes: [memory]
status: competitive
updated: 2026-08-18
sources:
  - perf_knowledge/optimization/vectorization_and_coalescing.md
  - perf_knowledge/languages/triton_amd/isa_verify.md
---

# narrow_global_load

**Severity: advisory.** This rule reports a fact and cannot supply the judgement that turns it into a
defect. Read the anti-signals before acting.

## Signal

`global_load_bytes.max < 16` with `global_load_bytes.accesses >= 4` — the widest global load in the
kernel is narrower than `global_load_dwordx4` / `global_load_b128`.

## Mechanism

A contiguous tile read 4 bytes at a time issues four times the memory instructions for the same bytes,
and each one carries its own address arithmetic and its own slot in the outstanding-request budget.
`isa_verify.md` §2 lists `global_load_dword` in the "bad → retune" column for exactly this reason.

The usual causes, in the order they are worth checking:

- the compiler could not prove alignment, so it declined to widen
- the compiler could not prove contiguity (an index expression it cannot analyse)
- the element type genuinely is narrow and nothing was packed
- a tail/boundary path that predicates per element instead of using buffer bounds

## Required source condition

The edit must give the backend something it can *prove*, not merely an intention:

- an alignment guarantee it can see (`__builtin_assume_aligned`, an aligned type, `__restrict__` where
  aliasing is what blocks it)
- an index expression whose contiguity is syntactically evident
- an explicit vector type on the staging copy — but see `lds_cast_alignment.py` first, because
  promising an alignment the stride does not have is finding (143) and fails silently

## Anti-signals — when a narrow load is CORRECT

- **A genuine gather.** Indirect or strided access has no contiguous run to widen. Widening it would
  read bytes the kernel does not want. This is the common false positive and the reason the rule is
  advisory.
- **Narrow elements with nothing to pack.** A `ushort` load of scattered values is 2 bytes because the
  data is 2 bytes.
- **A tail path.** A boundary remainder handled scalar-wise, executed once, is not the loop's cost.
  Scope to the hot loop before acting.
- **Already at the dtype's natural width** for the access pattern in question.

Decide from the access pattern in the source, not from this count. The count tells you where to look.

## Verifying the fix

`isa_signals.py diff --claim widen_global_load` compares the widest load before and after. If it comes
back `false` the compiler declined again — that is a Tier-1 question for the compiler role
(`-fsave-optimization-record` will usually name the refusal), not a reason to try the same edit harder.
