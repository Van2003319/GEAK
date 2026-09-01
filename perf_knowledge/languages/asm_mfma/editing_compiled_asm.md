---
title: Editing compiled AMDGCN — the post-compilation edit discipline
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, int8]
regimes: [both]
status: competitive
updated: 2026-09-01
sources:
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
  - https://llvm.org/docs/AMDGPU/gfx9_waitcnt.html
  - https://llvm.org/docs/AMDGPUUsage.html
  - https://arxiv.org/abs/2511.08083
---

# Editing compiled AMDGCN

## TL;DR
A different activity from everything else in this directory. [raw_asm.md](raw_asm.md) and
[mfma_intrinsics.md](mfma_intrinsics.md) are about **writing** assembly; this page is about taking the
`.s` the compiler already emitted, moving or rewriting a few instructions, and reassembling — the last
resort after source-level rewriting stops paying, because the remaining opportunities are the ones the
source cannot express.

**The base rate is low and the reason is structural.** Across 63 measured window edits on two gfx942
GEMM lanes, 8 paid and 55 did not (see [edit_patterns.md](edit_patterns.md) for the per-pattern
breakdown). Almost every null had the same cause: the edit was real in the binary, bit-exact, and
resource-free, and the thing it improved was **not what set the kernel's time**. Three cheap
arithmetic gates below predict most of those nulls before a single GPU second is spent. Run them
first; they are the highest-value part of this page.

**One fact before you touch anything.** `GCNHazardRecognizer` runs in the backend's pre-emit pass, not
in the assembler. When you reassemble an edited `.s`, **nothing re-derives hazard `s_nop`s and nothing
re-derives wait counts** — you own every one in the file. A reorder that newly creates a hazard
assembles cleanly and produces a wrong kernel. The rules you are now responsible for are in
[hazards_and_waitcnt.md](hazards_and_waitcnt.md); read it before the first edit, not after the first
correctness failure.

**External calibration for the base rate.** AMD's own agentic assembly optimiser reports that **only
5.1% of candidate edits become verified commits** (AsmEvo, arXiv 2608.20711 — same problem, same
hot-window localisation, MI300X/MI308X), against 8/63 ≈ 13% here. A low yield is intrinsic to the
activity rather than a defect in the setup, which is the argument for spending the budget on the gates
instead of on more candidates.

**And the published size of the prize.** Triton ships a post-assembly peephole for exactly this layer
(`python/triton/tools/amdgcnas.py`, `TRITON_ENABLE_AMDGCN_AS=1`) that does post-assembly LICM on LDS
address arithmetic plus MFMA–SALU interleaving. Its documented contribution on gfx950 is **+2
percentage points of MFMA efficiency on FP16/BF8 GEMM and +6 on MXFP4**, where the scale pipeline
creates denser SALU activity. That is the honest order of magnitude to expect from schedule-level
editing on a kernel the compiler already did well — single-digit percent, larger where there is more
scalar work to pack into MFMA gaps.

Worth knowing why the layer exists at all: `ROCm/aiter` ships **3159 prebuilt `.co` code objects and
zero `.s` files**. For those kernels there is no source-level knob because there is no source.

**And one fact before you touch anything.** `GCNHazardRecognizer` runs in the backend's pre-emit pass,
not in the assembler. When you reassemble an edited `.s`, **nothing re-derives hazard `s_nop`s and
nothing re-derives wait counts** — you own every one in the file. A reorder that newly creates a
hazard will assemble cleanly and produce a wrong kernel. The rules you are now responsible for are in
[hazards_and_waitcnt.md](hazards_and_waitcnt.md); read it before the first edit, not after the first
correctness failure.

For external calibration: AMD's own agentic assembly optimiser reports that **only 5.1% of candidate
edits survive verification** (AsmEvo, arXiv 2608.20711), against 8/63 ≈ 13% here. A low yield is
intrinsic to the activity, not a defect in the setup — which is exactly why the gates matter.

**And one fact before you touch anything.** `GCNHazardRecognizer` runs in the backend's pre-emit pass,
not in the assembler. When you reassemble an edited `.s`, **nothing re-derives hazard `s_nop`s and
nothing re-derives wait counts** — you own every one in the file. A reorder that newly creates a
hazard will assemble cleanly and produce a wrong kernel. The rules you are now responsible for are in
[hazards_and_waitcnt.md](hazards_and_waitcnt.md); read it before the first edit, not after the first
correctness failure.

For external calibration: AMD's own agentic assembly optimiser reports that **only 5.1% of candidate
edits survive verification** (AsmEvo, arXiv 2608.20711), against 8/63 ≈ 13% here. A low yield is
intrinsic to the activity, not a defect in the setup — which is exactly why the gates matter.

## The three pre-flight gates

Run all three on paper. Each one has closed a direction in this lane for zero GPU time, and each one
has also been *skipped* at least once and cost a full round.

### Gate 1 — dynamic share bounds the prize

An instruction-count edit can never return more than the region's share of the dynamic issue stream:

```
ceiling ≈ (instructions removed / region instructions)
        × (region's dynamic share)
        × SQ_ACTIVE_INST_ANY / SQ_WAVE_CYCLES
```

with `SQ_WAVE_CYCLES = SQ_ACTIVE_INST_ANY + SQ_WAIT_INST_ANY + SQ_WAIT_ANY` read on **the same
dispatch**. A share carried over from a sibling kernel is not a price.

Worked example that should have opened its round rather than closing it: a once-per-tile epilogue of
751 instructions against a mainloop of 172 stages × ~97 instructions is **~4% of the dynamic stream**,
so a 15.6% instruction cut there has a **~0.6% ceiling** — already below that route's 1.1% minimum
detectable effect. The edit was built anyway, came back bit-exact and free, and measured +0.076%.

The gate is an **upper** bound: it assumes issue is the critical path. If it says "too small", stop.
If it says "big enough", you have learned nothing yet — Gates 2 and 3 still apply.

### Gate 2 — the memory roofline closes the issue axis

Compute the body's effective bandwidth from bytes moved and measured time. If it is at or above the
DRAM roof, **there is no issue-pattern deficit to remove** and a high `wait_any` is bandwidth
queueing, not per-wave latency exposure.

Measured instance: a split-K reduce kernel moved 90.2 MB read + 11.3 MB write in 14.9 µs = **6.81 TB/s
effective**, which is *above* MI300X HBM peak (~5.3 TB/s) — i.e. the partial planes were being served
out of the 256 MB Infinity Cache. Its 85% `wait_any` looked like a textbook latency problem. Four
serialised loads on one destination quad were rotated onto four quads with a perfect `vmcnt` staircase
— the purest instance of the highest-value pattern in [edit_patterns.md](edit_patterns.md) — and it
measured −0.47% on the kernel itself.

The same arithmetic closes non-temporal hints: a stream that is being *served* from LLC cannot be
improved by telling the cache to discard it.

### Gate 3 — occupancy decides whether intra-wave scheduling is a currency at all

**This is the gate that explains most of the 55 nulls, and it is the most transferable fact on this
page.** At ≥4 waves/SIMD, the wave that stalls is not the wave that costs time: another resident wave
covers the drain, so *where a wait sits inside one wave's instruction stream* buys nothing.

Refuted independently five times in this lane at 4 waves/SIMD — three `s_waitcnt` staircases on three
different fp8 bodies, plus both directions of an `s_setprio` arbitration bracket. Every one of those
rows carries the same reopen condition: *only if occupancy falls to 1–2 waves/SIMD*.

Read the supplied occupancy, not just the cap:

```bash
# cap: LDS and register limits
grep -E '\.group_segment_fixed_size|\.vgpr_count|\.agpr_count|accum_offset' kernel.s
# supply: grid size / CU count  — 512 workgroups on 304 CUs = 1.68 WG/CU
```

**Supply is the axis that sorted the evidence.** The same earliest-legal-seam edit was measured on
five rungs of one kernel family: it paid +7.8% and +3.75% on the two rungs supplying >2 WG/CU, and
0.0% / 0.0% / −1.0% on the three supplying ~1.68. A block-census screen (staged loads × head-burst
length × K-step depth) predicted the opposite ordering and was falsified — the lean-looking rung lost.

## The round-trip contract

Nothing you do here reaches a number unless the edited `.s` is reassembled into the object the build
already links. Establish that **before** the first edit:

1. **Round-trip gate.** Emit the `.s`, reassemble it *unedited*, and require the instruction stream to
   be identical. A round trip that already changes the binary attributes its own damage to your edit.
   This is a property of the toolchain, not of your kernel, and it is checked once.
2. **Provenance stamp.** Keep the first line of an emitted `.s` — an AMDGCN comment naming the
   translation unit, the arch, and the source hash. It assembles to nothing and it is how a later
   step knows which build the assembly belongs to, and whether the source has moved under it.
3. **One translation unit per candidate.** Two edited `.s` for one object make an accepted result
   unattributable.
4. **Staleness is not an error you should have to notice.** If the source changed after the `.s` was
   emitted, the assembly describes a program that no longer exists; re-emit rather than reconcile.

## What you may move, and what the compiler owns

| Safe to move | Why | Owned by the compiler — do not touch |
|---|---|---|
| Independent VALU between dependent MFMA | no register or memory dependence | MFMA **order** — see the SrcC-bank note below |
| A load above work that does not read it | as long as the counter still covers every operand actually read next | `s_nop` whose hazard is still live (correctness on CDNA, not padding) |
| An already-satisfied `s_waitcnt` (delete) | costs one issue slot; both predecessors already drained | a pre-barrier drain in a producer/consumer kernel — that is handshake, not data |
| Address arithmetic into a store's 13-bit immediate | when the offsets are provably constant-separated | a spill whose reload a later `vmcnt` is counting |

Two of those deserve their own warning because both were learned the expensive way.

**MFMA order is not free to permute.** Regrouping 16 MFMA from the compiler's srcB-major order into
srcA-major — a pure permutation, identical instruction multiset — cost **1.03%**. The most plausible
mechanism is that the compiler's order spreads the accumulator families so consecutive MFMA never
contend on the same SrcC bank, and the "tidier" grouping destroys that. Treat MFMA order as load
bearing until a measurement says otherwise.

**A scratch spill can be load bearing.** In one body the `s_waitcnt vmcnt(10)` opening the hot block
was not counting global loads at all — it counted 12 scratch reloads issued ~1100 instructions
earlier, and existed so a pointer had landed before its consumer. Halving the reload count silently
turned that wait into a no-op and produced an address fault. Before deleting a spill, find what counts
its reloads.

## Locality, and why the window is the unit

Keep the edit inside one window — a straight-line run bounded by basic-block edges — for two reasons
that are both about attribution rather than safety. It keeps the reasoning small enough to be right,
and it keeps an accepted result explainable by one mechanism. An edit that also tidied three unrelated
places is a result nobody can build on.

Before changing anything, read the window widened to **the definitions it reads and the first consumer
of what it writes**, clipped to the basic block. An edit that reorders or removes an instruction whose
result is consumed later in that span is a correctness bug the assembler will not catch. Locate a
window by the opcodes it starts with, never by line number: your own first edit moves every line
below it.

One caveat the window model does not express: **two windows can be one experiment**. A destination
rotation and the `vmcnt` constants fitted to it are a single pipeline even when the localizer names
them as two separate sites 48 instructions apart. Report them as one result rather than splitting a
number across both.

## Resource neutrality is a hard constraint, not a preference

VGPR/AGPR/LDS/scratch are occupancy. Check the code object, not your intent:

```bash
llvm-readelf --notes kernel.hsaco | grep -E 'vgpr_count|agpr_count|vgpr_spill|group_segment|private_segment'
```

Three facts about the register budget that changed decisions in this lane:

- **Allocation granulates off `next_free_vgpr`, not off live values.** One body had ~74 architectural
  VGPRs reachable at *zero* occupancy cost by raising `accum_offset` alone, because 169 and 176 both
  granulate to the same registers/lane.
- **The loop owns the high-water mark.** An epilogue edit that removed 12 address vectors freed
  exactly **zero** registers, because the allocation peak sat inside the loop block. Confirm where the
  peak lives before pricing an edit as a register enabler.
- **Granulation boundaries are cliffs.** 126 → 128 registers/lane is 4 waves/SIMD; 136 is 3. Twelve
  extra VGPRs are free when LDS binds occupancy first and catastrophic when registers do.

## Verify

The full receipt chain — proving the edit reached the timed binary, proving it is a pure reorder,
and the measurement discipline that keeps a per-route number honest — is in
[asm_edit_verify.md](asm_edit_verify.md). The minimum before any timing:

```bash
# 1. it assembled into the object the build links
llvm-objdump -d --arch-name=amdgcn kernel.hsaco > after.txt
# 2. resources did not move
llvm-readelf --notes kernel.hsaco | grep -E 'vgpr|agpr|spill|segment'
# 3. a pure reorder has an identical opcode multiset
awk '{print $1}' before.txt | sort | uniq -c > b.cnt
awk '{print $1}' after.txt  | sort | uniq -c > a.cnt
diff b.cnt a.cnt        # empty for a reorder
# 4. correctness BEFORE timing — a fast wrong kernel is the one result this layer must refuse
```

## Sources
- AMD CDNA3 ISA Reference Guide (waitcnt semantics, hazards, buffer/global addressing): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- LLVM gfx9 `s_waitcnt` semantics ("wait until ≤N remaining"): https://llvm.org/docs/AMDGPU/gfx9_waitcnt.html
- LLVM AMDGPU backend user guide (resource-usage attributes, code-object metadata): https://llvm.org/docs/AMDGPUUsage.html
- HipKittens (arXiv 2511.08083 — 256 VGPR/256 AGPR split at one wave/SIMD; wave specialisation underperforms on CDNA3/4): https://arxiv.org/abs/2511.08083
- ROCm MI300X workload optimization (occupancy math, 512 VGPR/EU, 16-granule): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- AsmEvo (arXiv 2608.20711, AMD — 5.1% of candidate assembly edits survive verification; hot-window localisation; round-trip gate as a precondition): https://arxiv.org/abs/2608.20711
- Triton `amdgcnas` post-assembly peephole (`TRITON_ENABLE_AMDGCN_AS=1`; +2pp MFMA efficiency on FP16/BF8 GEMM, +6pp on MXFP4): https://github.com/triton-lang/triton
- ROCm/aiter (3159 prebuilt `.co`, zero `.s` — why a binary-level layer exists): https://github.com/ROCm/aiter
- ROCm blog, Memory Instruction Scheduling for Lock-Stepped Kernels on MI300X (the canonical gfx942 GEMM main-loop shape: 3 tiles in flight, 2 barriers per iteration, AGPR-resident accumulators): https://rocm.blogs.amd.com/software-tools-optimization/scheduling_memory_ops_gfx942/README.html
- Measured evidence: GEAK asmopt lanes, 63 window edits on `macro_bf16_gemm` / `macro_rb_gemm` / `fp8_gemm_tiled` families @ MI300X gfx942, ROCm 7.2.3, amdclang++ 22.0.0git, 2026-08-27..2026-08-31. Per-row archives under `exp/state_asmopt_bf16_20260827/` and `exp/state_fp8_rowwise_20260827/`.
