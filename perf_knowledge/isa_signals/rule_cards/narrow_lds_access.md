---
title: narrow_lds_access — widest LDS access under b128
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: competitive
updated: 2026-08-18
sources:
  - perf_knowledge/optimization/lds_and_bank_conflicts.md
  - perf_knowledge/languages/triton_amd/isa_verify.md
---

# narrow_lds_access

**Severity: advisory.**

## Signal

`lds_access_bytes.max < 16` with `lds_access_bytes.accesses >= 4` — no `ds_read_b128` / `ds_write_b128`
where dot-operand staging would normally use them.

## Mechanism

`isa_verify.md` §5 states the expectation directly: on gfx942 `kpack=2` should turn dot-operand LDS
reads into `ds_read_b128`; if they are still `b64` or `b32`, either `BLOCK_K` is too small (< 64) or the
swizzle did not apply. On gfx950 `ds_read_b128` is expected without `kpack`, which is deprecated there.

Narrow LDS reads cost twice: more instructions for the same bytes, and a bank-conflict pattern that a
wider access would have avoided by construction.

## Required source condition

- `BLOCK_K` large enough for the packed read to exist at all (≥ 64 on gfx942 for the usual bf16 shapes)
- the swizzle/`kpack` actually applied to the operand layout, not just requested
- a staging layout whose row stride permits a 16-byte access at every index the cast covers

## Anti-signals

- **Alignment the stride cannot carry.** This is finding (143), and it is the dangerous one: casting a
  destination to `uint4*` when the row stride is not a multiple of 16 makes the compiler emit
  `ds_write_b128` for addresses that are only 8-byte aligned. On gfx942 that does not fault — the
  address is truncated inside the granule and the tile is silently wrong. Run
  `kernel_workflow/scripts/lds_cast_alignment.py` before widening any LDS staging copy. A green count
  here with a wrong tile is worse than a narrow read.
- **LDS not on the dot-operand path.** A reduction scratchpad or a small broadcast buffer has no reason
  to reach b128.
- **`read2`/`write2` forms.** These move two elements per instruction; `lds_multi_access` counts them
  separately precisely so they are not read as narrow single accesses.
- **gfx950 with `kpack` still set** — deprecated there; the absence of `kpack` is not the cause.

## Verifying the fix

`isa_signals.py diff --claim widen_lds_access`, and re-run `lds_cast_alignment.py` as a delta against
the frozen original: exit 1 means the candidate introduced a hazard the baseline did not have, and that
outcome must fail the round rather than be traded for the wider access.
