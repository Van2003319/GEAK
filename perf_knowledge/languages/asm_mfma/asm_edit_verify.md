---
title: Verifying an assembly edit — did it land, is it inert, did it actually move the clock
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [both]
status: competitive
updated: 2026-09-01
sources:
  - https://llvm.org/docs/AMDGPUUsage.html
  - https://rocm.docs.amd.com/projects/omniperf/en/amd-staging/what-is-rocprof-compute.html
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
---

# Verifying an assembly edit

## TL;DR
An assembly edit has three separate things to prove, and they fail in different ways:
**(1) it reached the binary that was timed**, **(2) it changed nothing it was not supposed to change**,
and **(3) the number it produced is the edit and not the box.** Most of the wasted effort in this lane
came from (3) — a route carrying a ~1% per-install offset on top of a 3–5% run-to-run spread will
manufacture a convincing win for an edit that cannot possibly have caused it.

Rule of thumb from 63 measured edits: if the expected effect is under ~3%, the control arm is not
optional.

## 1. Did it land?

No build edge compiles a `.s`. The only reason an edited assembly matters is that a post-build step
overwrites the object the edge names — so "the patch applied and the number came back flat" is a
shape that must be distinguished from "the edit never reached the binary."

```bash
# extract the code object that actually ran and disassemble it
roc-obj-ls kernel.hsaco
llvm-objdump -d --arch-name=amdgcn kernel.hsaco > after.txt
# the marker: grep for the exact instruction you introduced or removed
grep -c 'vmcnt(3)' after.txt
```

Verify against the **linked** artifact, not the `.s`. A staged `.s` that never made it into the object
disassembles perfectly and proves nothing.

### Two validity traps that silently return the wrong answer

**Stale object after a workspace restore.** `tar` preserves mtimes, so restored sources are *older*
than a leftover `.o` and ninja reports "no work to do" while `Compile: PASS`. The re-captured "parent"
ISA is then the previous arm's. `touch` does not dislodge it — move the build directory aside and
rebuild. Source-to-source lockstep checks prove nothing about source-to-object.

**A new symbol reads as `refuted` by construction.** If a patch *adds* a kernel symbol rather than
editing existing ones, a differ that pairs only shared symbols finds every one of them legitimately
unchanged and reports the mechanism refuted. A verified, correctness-passing −4.72% patch was
misjudged this way. Confirm from the diff that no pre-existing symbol changed — which doubles as a
clean negative control on the other kernels.

**A third, softer trap: claim-vocabulary mismatch.** A mechanism checker that looks for `v_cvt*` will
report `refuted` on a kernel that has none, even though the edit is plainly in the binary. That is a
vocabulary gap, not a compiler refusal, and the licensed verdict is still `dead-end` rather than
`not-realized`.

## 2. Is it inert everywhere it should be?

### Resource neutrality
```bash
llvm-readelf --notes kernel.hsaco | \
  grep -E 'vgpr_count|agpr_count|accum_offset|vgpr_spill_count|group_segment_fixed_size|private_segment_fixed_size'
```
Compare against the parent field by field. Then recompute occupancy rather than reading it off a
register count: `65536 / group_segment_fixed_size` gives the LDS cap in WG/CU, and the unified
register total granulates to a wave count with cliffs (128 regs/lane → 4 waves/SIMD, 136 → 3).

### A pure reorder has an identical opcode multiset
```bash
awk '{print $1}' before.txt | sort | uniq -c > b.cnt
awk '{print $1}' after.txt  | sort | uniq -c > a.cnt
diff b.cnt a.cnt          # empty  ⇒  pure reorder
```
Also check the instruction *count* per symbol. A win reported at "identical instruction count, exactly
the 12 moved lines" is a much stronger claim than one reported as a diff.

### Blast radius, per symbol
```bash
for s in $(llvm-objdump -d --arch-name=amdgcn kernel.hsaco | grep -oP '^[0-9a-f]+ <\K[^>]+'); do
  ... # disassemble each symbol separately and diff parent vs candidate
done
```
The target claim is *exactly one symbol DIFF out of N, with `.note` metadata byte-identical across all
kernels*. This is what makes a per-route attribution defensible.

### Output bit-exactness
A reorder must be bit-exact. Hash the **raw integer view** of the output, not a float comparison:
```bash
python3 -c "import torch,hashlib;t=torch.load('out.pt');print(hashlib.sha256(t.view(torch.int16).numpy().tobytes()).hexdigest())"
```
Run it on several shapes, and in a candidate/control/candidate sequence so an install error cannot
masquerade as agreement.

### Census the steady-state extent by hand
A static signal reader counts a whole symbol or a whole basic block. Neither is the loop body you
edited — the true steady-state extent usually excludes a peeled head and a guarded tail. Every wait
census in this lane that mattered was counted by hand over that extent; the automated census
disagreed and was wrong about the direction of the change more than once.

## 3. Is the number the edit, or the box?

### The controls, in order of how often they caught something

**Zero-code-change blast-radius arm.** Build an install whose only difference is in a symbol the
measured route provably never reaches. It *cannot* change the route, so whatever it reads is the
install-to-install offset. On one route this arm read **−1.1%** — larger than the candidate's own
movement — and it caught a false attribution twice. Cost: one build. On any route with known spread,
this is the single highest-value control available.

**Pure-permutation negative control.** Same registers, same dependency graph, instructions moved from
earliest-legal to latest-legal position. If it measures the same as the candidate, the candidate's
"win" was frame drift. This is how a +1.4% arm was correctly retired.

**Structure-only control.** The same unroll, the same extra staging registers, with the mechanism
switched off (counters left at their original values). It separates the cost of the *vehicle* from the
value of the *mechanism* — and on one body the vehicle was the entire ~2% loss.

**Common-mode channel.** Carry a set of routes the edit cannot touch. Report
`corrected = direct / common_mode`. A candidate reading 1.0777 direct with a common mode of 1.00006 is
a very different claim from the same 1.0777 with a common mode of 1.006.

### Interleaving and power
Run **ABBA** (or both orders), same die, same session, run-by-run interleaved rather than arm-by-arm.
Report `n` per arm, the paired positive count, and `t`. The wins in this lane that survived scrutiny
were n=20v20 with 20/20 paired positive and t=9.31; the ones that did not were n=4v4 single-order.

**Two dies disagreeing in sign is the signature of a true zero**, not of a small win. A real effect
reproduces in sign on both dies — in the strongest negative here, two dies agreed to within 0.01%.

### The effect can be variance-shaped
Do not report only the mean. One winning edit moved best-decile latency ~2% while eliminating a slow
mode entirely: control bimodal over 103.2–115.7 µs (sd 3.91%), candidate unimodal over 97.9–102.7 µs
(sd 1.42%), with per-quantile ratios running ~1.05 through the lower three quartiles and ~1.125 across
the upper one. **A best-of-n or min-latency estimator under-reads that edit by about 2 points.** Plot
the quantiles before choosing an estimator.

### Know the route's MDE before you build
Minimum detectable effects measured on these suites ranged from 0.8% to 5.37% per route. Gate 1 in
[editing_compiled_asm.md](editing_compiled_asm.md) gives an instruction-count *ceiling*; if the ceiling
is under the route's MDE, the experiment cannot produce information no matter how it turns out.

## Reading a null honestly

A null from a fully realized, bit-exact, resource-free edit is a **real result about the kernel**, and
it is worth more than an unexplained one. Record which of the three gates it retires:

| the edit was… | what the null means |
|---|---|
| realized, bit-exact, resource-free | the mechanism is not this body's binding term — record the gate that now closes |
| realized, but resources moved | confounded; re-run resource-neutral before concluding anything |
| not in the binary | no information about the mechanism at all — `not-realized`, not `dead-end` |
| in the binary but the route was noisy | no information until a control arm is run |

The distinction in the last two rows is the expensive one. `dead-end` is the strongest claim available
and the only one that removes a technique from future planning; spending it on an experiment that did
not happen discards the technique for free.

## Sources
- LLVM AMDGPU backend user guide (code-object metadata, resource-usage attributes, `llvm-objdump --arch-name=amdgcn`): https://llvm.org/docs/AMDGPUUsage.html
- AMD CDNA3 ISA Reference Guide (occupancy/register granularity, LDS segment sizing): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- rocprof-compute / omniperf (per-kernel timing and PMC capture): https://rocm.docs.amd.com/projects/omniperf/en/amd-staging/what-is-rocprof-compute.html
- ROCm MI300X workload optimization (occupancy math, granulation): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Measured evidence: GEAK asmopt lanes @ MI300X gfx942, ROCm 7.2.3, amdclang++ 22.0.0git, 2026-08-27..2026-08-31 — the blast-radius arm, the permutation control and the variance-shaped win are rows in `exp/state_asmopt_bf16_20260827/` and `exp/state_fp8_rowwise_20260827/`.
