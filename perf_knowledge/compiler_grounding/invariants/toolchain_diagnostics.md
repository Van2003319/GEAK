---
title: What this toolchain will and will not tell you
kind: invariant
gens: [gfx942, gfx950]
dtypes: [any]
regimes: [both]
status: settled
updated: 2026-08-20
sources:
  - measured on this image: amdclang++ 22.0.0git (roc-7.2.3), ROCm 7.2.3
  - kernel_workflow/scripts/ir_capture.py
---

# Toolchain diagnostics available on this image

Which questions this ROCm build can answer directly, and which it cannot. Both halves are measured;
neither is inferred from what LLVM upstream supports, because ROCm ships a **release** build and the
difference between a release and an assertions build is exactly the set of flags a compiler
investigation reaches for first.

## Available

**`-mllvm -print-changed=quiet`** — a full IR dump after every pass that changed anything. Ordinary
`cl::opt`, present in the release build. Measured: 148 changed passes on a trivial kernel unfiltered,
42 with `--cuda-device-only` and a function filter, 96 on the real `tall_bf16_gemm_kernel`.

**`-mllvm -filter-print-funcs=<mangled symbol>`** — narrows the above to one function. This is the
difference between a navigable trajectory and an unreadable one.

**`--cuda-device-only`** — keeps the host half's passes out of the trace, where they would otherwise
appear as real stages of a function that never runs on the GPU.

**Coverage note:** the trace spans the *whole* pipeline, not just the middle end. Both LLVM IR passes
(`SROAPass`, `InferAddressSpacesPass`, `GVNPass`) and MachineFunction passes (`si-fold-operands`,
`register-coalescer`, `prologepilog`, `si-insert-waitcnts`) appear. So codegen-level questions —
register allocation, wait insertion, exec-mask lowering — are answerable from the same capture as
middle-end ones.

**`-Xclang -disable-llvm-passes -S -emit-llvm`** — the front end's own output, before any optimization.
`-print-changed` has no "before the first pass" dump, so without this the earliest stage in an archive
is already optimized.

**`-fsave-optimization-record -foptimization-record-file=<f>.yaml`** — per source line, what each pass
did and what it declined to do, with the pass's own reason string. This is Tier 1, and it is the only
thing that distinguishes "the pass ran and refused" from "the pass never saw this function".

**`-Rpass-analysis=kernel-resource-usage`** — register, LDS and occupancy figures as compiler remarks.

## Not available

**`-mllvm -debug-only=<pass>`** fails outright:

```
clang (LLVM option parsing): Unknown command line argument '-debug-only=load-store-vectorizer'.
```

because that flag is compiled out of a non-assertions build. Measured, not assumed. It is named here so
the next reader does not spend a round rediscovering it — and so that its absence is not mistaken for
the *trajectory* being unavailable, which was the standing misreading this file exists to correct.
`-print-changed` is a plain `cl::opt` and is present.

**AMDGPU backend sources.** `/opt/rocm/llvm` contains the built toolchain, not
`llvm/lib/Target/AMDGPU`. Tier 2 is therefore structurally unavailable on this image unless an operator
supplies `COMPILER_SOURCE_DIR`. Do not `git clone` llvm-project, do not fetch over the network, and do
not reconstruct pass behaviour from memory: a recalled pass name set beside real tool output reads
exactly like a verified finding, and nothing downstream can tell them apart.

## The one methodological trap

Everything above requires a **rebuild**, while the ISA archive was read out of the artifact that
actually ran. The reporting flags do not change codegen, so the two should agree — but "should" is not
evidence. `ir_capture.py` re-derives the object without the evidence flags and compares it to the
measured archive per kernel; until that comparison passes, a trajectory describes *a* program and not
*the* program, and the manifest says so in `provenance`.
