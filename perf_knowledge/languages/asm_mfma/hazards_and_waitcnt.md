---
title: Hazards and wait counters — the rules an assembly editor owns
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, int8, fp64]
regimes: [both]
status: competitive
updated: 2026-09-01
sources:
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
  - https://github.com/llvm/llvm-project
  - https://llvm.org/docs/AMDGPUUsage.html
---

# Hazards and wait counters

## TL;DR — the fact that reframes the whole activity

`GCNHazardRecognizer` runs as `PostRAHazardRecognizerID` inside `addPreEmitPass()`
(`llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp`), **not in the assembler**. When you reassemble an
edited `.s`, nothing re-derives hazard NOPs and nothing re-derives wait counts.

**You own every `s_nop` and every `s_waitcnt` in the file.** The assembler will not put back one you
delete, and it will not add one your reordering newly requires. This is the difference between
editing compiled assembly and every other level in this directory, and it is why
[edit_patterns.md](edit_patterns.md) treats a reorder as a correctness change until proven otherwise.

## Wait counters — when relaxing is legal

### On CDNA, `vmcnt` is already exact. The conservative counter is `lgkmcnt`.

This inverts the common assumption and it is worth stating plainly.
`WaitcntBrackets::counterOutOfOrder(LOAD_CNT)` returns **false unconditionally** on gfx942/gfx950,
because it early-returns on `!hasVscnt()` and `FeatureVscnt` appears only in the GFX10/11/12/13
feature lists — never in `FeatureGFX9` or `FeatureISAVersion9_4_Common`/`9_5_Common`
(`llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp`, `AMDGPU.td`). So `SIInsertWaitcnts` emits a precise
`vmcnt(N)` whenever it can.

**Consequence:** a `vmcnt(0)` in gfx942 output is usually *not* compiler laziness. It is genuinely
required by the dataflow, a FLAT artifact, or a memory-model obligation. If a window's loads are
serialised, look at the **register assignment** rather than the counter — see D1 in
[edit_patterns.md](edit_patterns.md), where the fix was to give the loads distinct destinations and
the exact count then loosened by itself.

`lgkmcnt` is the one with real slack. The pass keeps one timeline per counter, so when SMEM and LDS
are both in flight it emits a conservative `lgkmcnt(0)` where `lgkmcnt(i)` would do — stated in the
pass's own header comment. But note the hardware limit: CDNA3 §4.4 and §8.3 say scalar memory reads
return **out of order**, so if the window genuinely mixes `s_load*` with `ds_*`, `lgkmcnt(0)` is
required by hardware and not recoverable.

### The FLAT trap

`hasFlatLgkmVMemCountInOrder()` is `getGeneration() > GFX9`, and gfx942/gfx950 are GFX9 generation —
so it is **false**, and any pending FLAT in the bracket forces *both* `vmcnt(0)` and `lgkmcnt(0)`.

The hardware documentation agrees: CDNA3 §10.2 — FLAT "are executed as both an LDS and a Buffer
instruction; so, they increment both VM_CNT and LGKM_CNT"; §10.2.1 — "Flat instructions can complete
out of order with each other"; §10.2.2 — "the only sensible S_WAITCNT value to use after FLAT
instructions is zero." By contrast §10.4/§10.5: `global_*` and `scratch_*` "do not access LDS, only
VM_CNT is used."

**Rule:** `global_*`, `buffer_*`, `scratch_*` are strictly in-order against each other and relaxable.
One `flat_*` anywhere in the window closes the whole question.

### The four conditions that license `vmcnt(0)` → `vmcnt(N)`

CDNA3 §4.4 is the ground truth: "Memory reads and writes return in the order they were issued,
including mixing reads and writes."

1. **No FLAT outstanding** in the window.
2. **Count every VMEM issued after the producer, in program order — including stores.** VM_CNT is
   shared with stores on CDNA (there is no VSCNT), so a store you forgot to count is an off-by-one.
3. The wait you weaken is not also gating a **different** consumer before the next `s_waitcnt`.
4. The wait was not placed for **fence/atomic ordering**. `applyPreexistingWaitcnt` "conservatively
   assumes that these preexisting waits are required for correctness" and never weakens them; a
   memory-model obligation still looks satisfiable from register dataflow alone, so this one fails
   silently.

Useful for testing a hypothesis before editing: `-mllvm -amdgpu-waitcnt-forcezero` and
`-mllvm -amdgpu-waitcnt-load-forcezero` bracket the two extremes with a recompile.

## `s_nop` — every one the compiler emitted is correctness

`s_nop N` inserts **N+1** wait states (CDNA3 §12.5, SIMM16[3:0]); `s_nop 15` is the 16-cycle maximum.

The only padding source in the backend is `checkMFMAPadding`, gated on `-amdgpu-mfma-padding-ratio`
which **defaults to 0** and returns immediately when unset (`GCNHazardRecognizer.cpp`). There is also
`-amdgpu-snop-padding`, also default 0, marked "for debugging purposes only". Unless the build passed
one of those, **deleting any `s_nop` in compiler output is a correctness change**, and the folklore
entry "remove a redundant `s_nop` the backend inserted" has no default-build instance.

The legitimate move is the inverse: remove the *hazard*, and the NOP becomes deletable. See "Renaming
away a hazard" below.

### MFMA wait states, gfx942 vs gfx950

`NumPasses` is the MFMA's scheduling latency (2/4/8/16 for the common shapes). "XDL" is the matrix
pipeline (`SIInstrInfo::isXDL`, which excludes DGEMM and both `v_accvgpr_*`).

| Producer → consumer | gfx942 (2/4/8/16 pass) | gfx950 |
|---|---|---|
| XDL write → XDL reads SrcC, **exact same** vDst | **0** (2 for 4×4) | same |
| XDL write → XDL reads SrcC, **overlapped** | 3 / 5 / 9 / 17 | 4 / 6 / 10 / 18 |
| XDL write → S/DGEMM reads SrcC | 3 / 5 / 9 / 17 | 3 / 6 / 10 / 18 |
| XDL write → MFMA reads SrcA/SrcB | 5 / 7 / 11 / 19 | 5 / 8 / 12 / 20 |
| XDL write → VALU/VMEM/DS read, or VALU WAW | 5 / 7 / 11 / 19 | 5 / 8 / 12 / 20 |
| SGEMM (non-XDL) write → SMFMA reads SrcC overlapped | 2 / 4 / 8 / 16 | same |
| SGEMM write → MFMA reads SrcA/SrcB | 4 / 6 / 10 / 18 | same |
| MFMA reads SrcC → VALU **writes** that reg (WAR) | 1 / 3 / 7 / 15 | same |
| `V_MFMA_16x16x4_F64` → S/DGEMM reads SrcC overlapped | 9 | **17** |
| `V_MFMA_16x16x4_F64` → reads SrcA/SrcB | 11 | **19** |
| Non-DL VALU writes VGPR → MFMA reads it | 2 | 2 |
| VALU writes EXEC (e.g. `v_cmpx`) → MFMA | 4 | 4 |

Sources: `GCNHazardRecognizer.cpp` (`checkMAIHazards90A`, `checkMAIVALUHazards`); CDNA3 §7.5 Tables
37ff.; CDNA4 §7.6 Table 38ff. The LLVM formulas and the ISA tables agree exactly.

**Two consequences for an editor.**

**The zero-wait exact-match rule is why accumulating MFMA chains carry no NOPs.** CDNA3 §7.5 requires
"the two V_MFMA must be the same number passes and vDst and vSrc start from the same offset and same
VGPR size." Reorder MFMAs so a consumer's SrcC only *partially* overlaps the producer's vDst and you
flip from **0 to 9 or 17** wait states — a large silent regression, and a correctness hazard if you do
not also insert the NOPs. This is the most likely mechanism behind the measured −1.03% MFMA
permutation in [edit_patterns.md](edit_patterns.md) M4 (the lane's own note guessed SrcC bank
contention; both are consistent with the sign, and neither was isolated).

**gfx950 needs one more cycle** than gfx942 on the 4/8/16-pass XDL cases. Assembly hand-tuned on
gfx942 and moved to gfx950 without re-running the hazard analysis is under-NOP'd by exactly 1.

### `v_accvgpr_write` → MFMA

On gfx90a+ the `v_accvgpr_*` ops are handled as ordinary VALU, so `v_accvgpr_write` feeding a
dependent MFMA is covered by the "Non-DL VALU writes VGPR → MFMA reads it = 2" row — i.e. it needs
`s_nop 1`. The reverse (MFMA → `v_accvgpr_read` of the result) costs `NumPasses + 3`.

### Non-MFMA hazards a reorder can trip

CDNA3 §4.5 Table 11, "The hardware does not check for the following dependencies":

| First | Second | Wait states |
|---|---|---|
| VALU writes SGPR | VMEM reads that SGPR | **5** |
| VALU sets VCC or EXEC | VALU using VCCZ/EXECZ as a data source | 5 |
| VALU writes SGPR/VCC | `v_readlane`/`v_writelane` using it as lane-select | 4 |
| VALU writes SGPR/VCC | VALU reads SGPR as a constant | 2 |
| VALU writes VGPR | VALU DPP reads that VGPR | 2 |
| `s_setreg` | `s_getreg` / `s_setreg` | 2 |
| `{flat,global,buffer}_store_dwordx3/x4` | SALU overwrites the writedata VGPRs | 1 |
| same | **VALU** overwrites those VGPRs | 2 |

The ISA adds a caveat on the last two: "BUFFER_STORE_* operations that use an SGPR for offset do not
require any wait states."

⚠ **Known discrepancy.** `llvm/llvm-project` PR #197267 (merged 2026-06-01) contradicts that caveat —
it widens the MUBUF/MTBUF source-VGPR WAR hazard on the gfx940 family to fire when SOFFSET comes from
an SGPR (1 wait state), keeping 2 for literal/absent SOFFSET. The PR states the behaviour was
reproduced on gfx950 only and that "gfx942 (MI300) was not validated on hardware." On this one rule
the compiler is stricter than the published ISA and the gfx942 half is an extrapolation. Reproduce
before relying on either version.

### Renaming away a hazard — the safe deletion

`llvm/llvm-project` PR #156943 gives the canonical shape:

```asm
v_mfma_f32_16x16x32_fp8_fp8 v[26:29], v[102:103], v[6:7], v[134:137]
...
s_nop 5                       ; hazard mitigation
ds_read_b128 v[26:29], v125 offset:10240
```

The `ds_read` reuses the MFMA's destination tuple, so the WAR hazard forces `s_nop 5`. **Renaming the
`ds_read` destination to a non-overlapping even-aligned tuple removes the NOP entirely** — a real
edit, safe, and it needs only free VGPRs. This is the one legitimate form of "remove an `s_nop`", and
it is a *register* edit rather than a *schedule* edit.

## What you cannot insert

`SCHED_BARRIER`, `SCHED_GROUP_BARRIER` and `IGLP_OPT` are metadata pseudos with `Size = 0` and
`isMeta = 1` (`SIInstructions.td`). They assemble to **nothing**. By the time you hold a `.s`, their
effect is already baked into the instruction order, and writing one into the file changes nothing.

A source round can use them — for example, restoring a scheduling boundary with
`__builtin_amdgcn_sched_barrier(0)` after deleting an exec region — but that is a phase-1 edit, not
one available here. Mask bits, for reading a source file: `ALU 0x1`, `VALU 0x2`, `SALU 0x4`,
`MFMA 0x8`, `VMEM 0x10`, `VMEM_READ 0x20`, `VMEM_WRITE 0x40`, `DS 0x80`, `DS_READ 0x100`,
`DS_WRITE 0x200`, `TRANS 0x400`, `LDSDMA 0x800`; `0` is a total barrier.

## Widening — the alignment rules are not what most people assume

| Access | Alignment actually required | Notes |
|---|---|---|
| `global_load_dwordx4` | **4 bytes** | `SIISelLowering`: wide global ops "perform better than multiple smaller memory ops — even when misaligned". Widening contiguous `global_load_dword` needs **no new alignment proof**. |
| `buffer_load_dwordx4` | **natural (16 B)** unless `hasRelaxedBufferOOBMode()` | And the OOB semantics differ — see below. |
| VGPR tuple destination | **even-aligned** (`vgpr-align2`, in `FeatureISAVersion9_4_Common`) | An `AssemblerPredicate`, so `llvm-mc` rejects a violation rather than miscompiling. `v[5:8]` illegal, `v[6:9]` fine. |
| `ds_read_b128` / `ds_write_b128` | **8 bytes** per `SIISelLowering` (`RequiredAlignment = Align(8)`) | ◇ but see the open question below. |

**The buffer OOB difference is a real semantic change, not a technicality.** Four separate
`buffer_load_dword` are each clamped against `num_records`; one `buffer_load_dwordx4` straddling the
boundary is not equivalent — "in the edge case where an access starts out-of-bounds and then enters
in-bounds, the entire access would be treated as out-of-bounds." If a `num_records` boundary can fall
inside the widened span, widening changes behaviour.

**Widening shifts every downstream counter.** Four loads become one, so every `vmcnt(N)` later in the
same bracket must drop by three. This is the easiest silent correctness bug in the whole page.

◇ **Open question, flagged rather than asserted.** `useDS128()` is `HasCIInsts && EnableDS128`, and
`enable-ds128` is *not* in the gfx942/gfx950 feature sets — so SelectionDAG should not form
`ds_read_b128` by default. Yet real gfx942 kernels plainly contain it (AMD's own scheduling blog
describes `ds_write_b128`/`ds_read_b128` staging pairs). Whether those come from `SILoadStoreOptimizer`
merging later in the pipeline, from explicit intrinsics, or from `-mllvm -amdgpu-ds128` was not
established. Until it is, treat the 8-byte figure as the *ISel* rule and confirm empirically before
using it to license an LDS merge — the one measured LDS widening in this lane was a **loss** on bank
pressure grounds regardless of legality ([edit_patterns.md](edit_patterns.md) N1).

## Sources
- AMD CDNA3 ISA Reference Guide — §4.4 data dependency resolution & counter ordering, §4.5 Table 11 manual-NOP dependencies, §7.5 MFMA wait-state tables, §8.3 scalar memory ordering, §9.1.7 dword alignment, §10.2/§10.4/§10.5 FLAT vs global/scratch counters, §12.5 `s_nop` and `s_setprio`: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- AMD CDNA4 ISA Reference Guide — §7 Acc VGPR file and the ACC bit, §7.6 Table 38ff. gfx950 wait states: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
- `llvm/llvm-project@d770d1ee` — `lib/Target/AMDGPU/SIInsertWaitcnts.cpp` (counter timelines, `counterOutOfOrder`, `applyPreexistingWaitcnt`), `GCNHazardRecognizer.cpp` (wait-state formulas, `checkMFMAPadding`), `GCNSubtarget.h` (`hasFlatLgkmVMemCountInOrder`, `useDS128`), `SIISelLowering.cpp` (alignment rules), `SIInstructions.td` (`SCHED_BARRIER` is `isMeta`), `AMDGPU.td` (`FeatureVscnt`, `FeatureRequiresAlignedVGPRs`), `AMDGPUTargetMachine.cpp` (hazard recognizer in `addPreEmitPass`): https://github.com/llvm/llvm-project
- llvm#131954 (spurious `v_accvgpr` / spills at large MFMA tiles): https://github.com/llvm/llvm-project/issues/131954
- llvm PR #156943 (hazard NOP removable by renaming a `ds_read` destination): https://github.com/llvm/llvm-project/pull/156943
- llvm PR #197267 (MUBUF SOFFSET WAR hazard; gfx942 half unvalidated on hardware): https://github.com/llvm/llvm-project/pull/197267
- LLVM AMDGPU user guide — `sched_barrier` / `sched_group_barrier` / `iglp_opt` mask bits and semantics: https://llvm.org/docs/AMDGPUUsage.html
