---
title: AMDGPU pass navigation map — which pass owns which decision
kind: reference
gens: [gfx942, gfx950]
dtypes: [any]
regimes: [both]
status: competitive
updated: 2026-08-20
sources:
  - measured on this image: amdclang++ 22.0.0git (roc-7.2.3), ROCm 7.2.3, gfx942
  - kernel_workflow/scripts/ir_capture.py (-mllvm -print-changed=quiet)
  - https://llvm.org/docs/AMDGPUUsage.html
---

# Pass navigation map

Tier 0 of the compiler role. `ir_signals.py find-changes` hands you a pass name; this says what that
pass is *for*, so the narrowed question you send to Tier 1 or Tier 2 is about the right subsystem.

**Every name below was observed, not recalled.** The list is the 63 distinct passes that actually
changed the IR of `tall_bf16_gemm_kernel` on this image, captured by `ir_capture.py`. That matters
because a pass map written from memory is indistinguishable from one written from the toolchain, and
the whole reason L4 exists is that the lane ran out of things it could check. If a pass appears in a
trajectory and not here, add it *from the trajectory* rather than from what LLVM upstream is believed
to do.

## How to use a row

A pass name answers "which subsystem", never "why". `SROAPass` removing 71 loads is what SROA is for;
it becomes a finding only when the loads it removed were the ones you wanted vectorized. So: find the
family, read what the pass is entitled to do, and then ask whether the observed change is that
entitlement working normally or an optimization that did not happen.

## The pipeline in order

A trajectory crosses instruction selection once. Before it the stages are LLVM IR and the passes have
`Pass`-suffixed class names; after it they are MIR and the passes have lowercase flag names. That
boundary is where an IR-level explanation stops and a codegen-level one starts, and `ir_signals.py`
flags it explicitly (`crosses_ir_boundary`) because counts either side are not comparable.

### 1. Inlining, IPO and cleanup (LLVM IR)

`InlinerPass`, `GlobalOptPass`, `GlobalDCEPass`, `IPSCCPPass`, `SCCPPass`,
`PostOrderFunctionAttrsPass`, `TailCallElimPass`

Owns whether a device function body is present at all. **`InlinerPass` is where a kernel's MFMA and
barrier counts first become nonzero** — an intrinsic inside a helper is invisible until it inlines, so
a "matrix ops +N at InlinerPass" line is normally the body arriving, not a transformation.

### 2. Scalar replacement and redundancy (LLVM IR)

`SROAPass`, `EarlyCSEPass`, `InstCombinePass`, `InstSimplifyPass`, `GVNPass`, `ReassociatePass`,
`SimplifyCFGPass`

`SROAPass` breaks aggregates into scalars. This is the pass that most often *destroys* a wide access
the source appeared to express: an array or struct staged in registers becomes individual values, and
the vector load the author expected never existed in the first place. If the trajectory shows loads
collapsing here, the question is whether the aggregate could have stayed in memory with a proven
alignment, not why a later pass failed to re-vectorize.

`GVNPass` is the biggest single mover in a real GEMM trajectory (magnitude 788 on the measured kernel)
because it hoists and re-materializes across the loop nest. Large deltas here are normal.

### 3. Address spaces (LLVM IR)

`InferAddressSpacesPass`

Owns whether a generic (flat) pointer becomes `addrspace(1)` global or `addrspace(3)` LDS. A kernel
whose accesses stay generic pays flat addressing for every one of them. If `address_spaces` in the
census still shows generic pointers after this pass, the source gave it a pointer it could not prove
the provenance of — that is a legality question and a legitimate Tier 2 escalation.

### 4. Loop structure (LLVM IR)

`LoopSimplifyPass`, `LCSSAPass`, `LoopRotatePass`, `LICMPass`, `IndVarSimplifyPass`, `LoopUnrollPass`

`LICMPass` hoists loop-invariant work out; `LoopRotatePass` and `IndVarSimplifyPass` restructure the
loop and its induction variables. **These three routinely show symmetric ± swings in `sync` and
`matrix` counts** as blocks are cloned and merged — on the measured kernel, sync went +1/+1 at LICM,
−2 at LoopRotate, −2 at IndVarSimplify, all of it structural churn. Do not read a single ± here as a
pass adding or deleting barriers; read the net across the group.

### 5. Instruction selection (the boundary)

`amdgpu-isel`, `finalize-isel`, `si-fix-sgpr-copies`, `si-i1-copies`

`amdgpu-isel` chooses the actual instructions: this is where `global_load_dwordx4` versus four
`global_load_dword` is decided, and where LDS traffic first appears as `DS_*` (+34 ops on the measured
kernel). A width question that survives to here is a question about what isel was *able* to select,
which depends on alignment and contiguity facts established upstream.

### 6. Machine-level optimization (MIR)

`machine-cse`, `machine-sink`, `early-machinelicm`, `peephole-opt`, `si-fold-operands`,
`si-shrink-instructions`, `si-load-store-opt`, `si-peephole-sdwa`, `dead-mi-elimination`,
`detect-dead-lanes`, `processimpdefs`, `unreachable-mbb-elimination`

`si-load-store-opt` merges adjacent memory operations into wider ones and forms clauses — it is the
*second* chance at width, after isel. `si-fold-operands` folds immediates and copies, and is also
where AGPR-versus-VGPR placement starts to move.

### 7. AGPR / accumulator placement (MIR)

`si-fold-operands`, `amdgpu-prepare-agpr-alloc`, `postrapseudos`

On the measured kernel these three form a visible cycle: +16 accvgpr at `si-fold-operands`, −16 at
`amdgpu-prepare-agpr-alloc`, +16 again at `postrapseudos`. Accumulator moves that survive to the final
stage are the ones `perf_knowledge/isa_signals/rule_cards/accvgpr_moves_in_kernel.md` is about;
intermediate churn is not.

### 8. Register allocation (MIR)

`livevars`, `liveintervals`, `si-opt-vgpr-liverange`, `phi-node-elimination`,
`twoaddressinstruction`, `register-coalescer`, `machine-scheduler`,
`si-optimize-exec-masking-pre-ra`, `si-form-memory-clauses`, `livedebugvars`, `virtregrewriter`

`phi-node-elimination` inserts copies (+130 on the measured kernel) and `register-coalescer` removes
them again (−179). Judging either in isolation is meaningless. **This is the group that owns VGPR
pressure and therefore occupancy**, and `prologepilog` below is where failure to fit becomes scratch.

### 9. Control flow and exec mask (MIR)

`si-lower-control-flow`, `si-optimize-exec-masking`, `si-fix-vgpr-copies`, `branch-folder`,
`block-placement`

Owns `s_and_saveexec` and friends. Note the measured finding already in `tech_lead.md`: an exec-masked
region is also a *scheduling block boundary*, so deleting guards can raise loop `s_waitcnt` rather
than lower it.

### 10. Frame and spill (MIR)

`prologepilog`

Where a register allocation that did not fit becomes `scratch`. `scratch first appears at stage N` in
`ir_signals.py performance-signals` is the exact location of a spill's birth, and it is almost always
here or in `virtregrewriter`.

### 11. Memory model, waits and hazards (MIR, last)

`si-memory-legalizer`, `si-insert-waitcnts`, `si-pre-emit-peephole`, `post-RA-hazard-rec`,
`postmisched`, `si-post-ra-bundler`

`si-insert-waitcnts` is where `s_waitcnt` is placed (+14 on the measured kernel, the single largest
sync contribution in the whole trajectory). If the complaint is "too many waits", this is the pass
that inserted them — but it inserts them to satisfy a dependency the schedule created, so the question
belongs to the scheduler or to the memory ordering the source asked for, not to this pass's own logic.

`si-memory-legalizer` implements the memory model: `__threadfence`, atomics ordering, and volatile all
turn into cache-control bits and waits here.

## What is NOT in this map

`LoadStoreVectorizer` did not appear in the measured trajectory, because on that kernel it changed
nothing. Its absence from a trajectory is itself informative — it means the pass either did not run or
had nothing it could widen — and the way to tell the two apart is
`-fsave-optimization-record` (Tier 1), not this file.
