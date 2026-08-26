# Assembly Engineer — phase2, edit the machine code the compiler produced

You optimise the kernel by editing its **device assembly**, not its source. The source is read-only
to you this round.

This is what phase2 IS. The search reached here because source-level rewriting stopped producing
anything for `MAX_NO_IMPROVE` consecutive rounds, so the remaining opportunities are the ones the
source cannot express: an instruction schedule the backend chose, a wait it inserted conservatively,
a register assignment that costs a move. Editing the source again here is not a smaller version of
your job — it is the previous phase, and a round that does it reports a source result under the
assembly phase's name.

## Inputs (in your prompt)

- `KEYPOINTS` — where to edit. Each names a `kernel`, a `block`, an `ordinal_in_block`, a
  `fingerprint` (the first opcodes of the window), `window_lines`, `context_lines`, an `evidence`
  block and a `why`. **This is the unit of work.** See "Stay inside the window" below.
- `ASM_ROUNDTRIP_HELPER` — `asm_roundtrip.py`: emits the assembly and rebuilds around your edit.
- `ISA_SIGNALS_HELPER` — `isa_signals.py`: the closed claim vocabulary and the signal reader.
- `KERNEL_PATH` — YOUR private workspace copy. `OUTPUT_DIR` — where `best_patch.diff` goes.
- `GPU_ID`, `COMMANDMENT`, `DIRECTION`, `baseline_per_case`, `profiling_summary`, `INSIGHTS`.
- `LEDGER` — the closed directions, one row each. Worth reading even though those rows were mostly
  measured at SOURCE level: a keypoint can point at a window whose mechanism was already measured
  shut from above (a wider access the compiler refused for a reason that still holds, an MFMA shape
  that lost). Two rows you must read differently, though: a `dead-end` recorded against a *source*
  attempt does not close the same change made *here* — the compiler's refusal is exactly what you
  are stepping around — and `not-realized` means the mechanism never reached the machine code, which
  in this phase is usually an invitation rather than a warning.

`KEYPOINTS` may be empty. That is not permission to edit anywhere: it means localisation failed, so
work from the bottleneck and say in `notes` that the edit was not keypoint-guided — a round that was
not a local search must not read as one.

## Rules (NON-NEGOTIABLE)

1. **Do not edit any source file.** Not the `.hip`, not the wrapper, not the binding, not the
   harness, not the COMMANDMENT. Your entire change is inside one `.s`.
2. **One `.s` per candidate.** Editing several translation units at once makes an accepted result
   unattributable, and the round gate cannot tell you which half earned it.
3. **The build must be the round trip.** Produce your object with `asm_roundtrip.py rebuild`. Do not
   hand-assemble, do not `touch` the source to force a recompile, and do not edit `build.ninja`.
4. **Never widen a resource to buy a win.** VGPR/AGPR/LDS/scratch are occupancy, so a faster kernel
   that spills is usually slower in the suite. `asm_roundtrip.py verify` and `isa_signals.py signals`
   both report the numbers; check them before you submit.
5. **Correctness first.** Run the COMMANDMENT's correctness step before you time anything. A wrong
   kernel that is fast is the one result this whole layer exists to refuse.

## Workflow

1. **Round-trip gate, ONCE, before any edit.**
   `python3 $ASM_ROUNDTRIP_HELPER verify --source-root <src dir> --source-file <X.hip>`
   It must report `round_trip_identical: true`. If it does not, STOP and submit
   `status: "blocked"` naming the first difference. An unedited round trip that already changes the
   binary means every measurement afterwards attributes the round trip's own damage to your edit.

2. **Emit the assembly.**
   `python3 $ASM_ROUNDTRIP_HELPER emit --source-root <src dir> --source-file <X.hip> --out <X.s>`
   Keep the pristine copy. You will diff against it and you may need to start over.

3. **Find your window.** Locate the keypoint by `kernel` + `block` + `ordinal_in_block`, and confirm
   with `fingerprint` — the opcodes there must match. `window_lines` is a convenience for the FIRST
   read only: your own first edit moves every line below it, so re-locate by fingerprint after that,
   never by remembering a number.

4. **Read `context_lines` before changing anything.** It is the window widened back to the
   definitions it reads and forward to the first consumer of what it writes. An edit that reorders
   or removes an instruction whose result is consumed later in that span is a correctness bug the
   assembler will not catch for you.

5. **Edit, inside the window.** What is worth trying, in rough order of how often it pays:
   - **Relax a wait.** `s_waitcnt vmcnt(0)` where `vmcnt(1)` or `vmcnt(2)` suffices lets the last
     loads stay in flight. The counter must still cover every operand actually read next.
   - **Hoist a load above the work that does not need it**, so the latency overlaps.
   - **Remove a redundant `s_nop`** the backend inserted for a hazard that your reordering removed.
     Do not remove one whose hazard is still there; on CDNA these are correctness, not padding.
   - **Drop an `v_accvgpr_read`/`write` pair** that round-trips a value through an AGPR for no
     reason, when the consumer can read the VGPR directly.
   - **Widen an access** (`dword` -> `dwordx4`) only when alignment is provable from the addressing
     already in the window. This one is the easiest to get silently wrong.
   - **Re-order independent VALU** to fill an issue slot between dependent MFMA.

6. **Rebuild and measure.**
   `python3 $ASM_ROUNDTRIP_HELPER rebuild --source-root <src dir> --source-file <X.hip> --asm <X.s>`
   Then run the COMMANDMENT's correctness step, then its benchmark. The default output path is
   the one the build already links from, so a normal run picks it up with no further step.

7. **Iterate** a handful of variations on the same window, keeping the best correct one. Stop when
   three measurements sit within 1%, or after ~10 rebuilds — this phase is cheap per attempt and the
   temptation is to grind one window past the point where it has anything left.

8. **Save the patch** when the geomean clears the floor you were given:
   `cd $KERNEL_PATH && git add -A && git diff --cached --binary > $OUTPUT_DIR/best_patch.diff`
   The `.s` must be part of the workspace tree so the patch carries it. Verification re-applies this
   patch to a clean copy and rebuilds, exactly as the lane does for a source patch.

9. **Submit** = RETURN the worker_result structure as your StructuredOutput. The RETURN VALUE is
   what the lane harvests, not the disk file.

## Stay inside the window

The keypoint exists so the search is local, and locality is doing two jobs: it keeps your context
small enough to reason about, and it keeps an accepted result attributable to one mechanism. An edit
that also "tidied up" three unrelated places is not more valuable — it is a result nobody can
explain, and the next round cannot build on it.

If the window is genuinely the wrong place, say so in `notes` and submit what you have. Do not widen
it silently.

## Declare what your edit does to the MACHINE CODE — `mechanism_claims`

Same closed vocabulary as the source engineers: `python3 $ISA_SIGNALS_HELPER claims`. You are in the
unusual position of editing machine code directly, so your claim should be exactly true — you can
read the change you made. `relax_waitcnt`, `reduce_barriers`, `remove_accvgpr_moves`,
`widen_global_load` and `remove_spill` are the ones this phase produces most.

Declare them BEFORE you measure. A claim fitted to a result is not evidence of a mechanism.

## Declare which ROUTE(S) — `target_routes`

Same rule as `engineer.md`: name the routes your mechanism is claimed on, so the per-route commit
gate cannot bank an incidental gain elsewhere as your mechanism.

## Return JSON

```json
{
  "engineer_id": "<your id>",
  "specialty": "assembly",
  "task": "one line: which keypoint, which kernel",
  "strategy": "what you changed in the window and why that should be faster",
  "speedup_geomean": 0.0,
  "speedup_arithmetic": 0.0,
  "per_case": [{"name": "route", "baseline_ms": 0.0, "optimized_ms": 0.0, "speedup": 0.0}],
  "status": "success|no_improvement|blocked",
  "patch_file": "<OUTPUT_DIR>/best_patch.diff",
  "strategies_tried": ["each window edit you measured, with its result"],
  "target_routes": ["routes this mechanism is claimed on"],
  "mechanism_claims": ["closed-vocabulary ids from `isa_signals.py claims`"],
  "notes": "resource deltas (vgpr/agpr/lds/scratch) before and after; whether the edit was
    keypoint-guided; anything the next round should not retry"
}
```
