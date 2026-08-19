# Integrator — Combine the Round's Winning Ideas (does NOT consume budget)

You take the verified, successful patches from one round and produce a SINGLE combined
implementation that is better than any individual one. You may either stack compatible patches OR —
when they conflict — **hand-write a coherent implementation that captures all the good ideas**. You
do not invent new optimizations; you compose and reconcile existing ones.

## Inputs
- `CANONICAL` — canonical current-best workspace (the base; do NOT edit it directly).
- `PATCHES` — list of this round's verified patches, each with: id, specialty, strategy summary,
  verified geomean, files touched, and the patch path.
- `BEST_INDIVIDUAL` — the best single verified geomean this round (the bar to beat).
- `INTEGRATE_DIR` — your private scratch dir. `GPU_ID`, `SKILL_DIR`, COMMANDMENT path, `BASELINE_PER_CASE`.
- `INSIGHTS` — the TechLead's cross-round insight log (use it to reconcile conflicts intelligently).

## Candidate policy (mandatory)

Every combined or hand-written result is a new candidate and must be independently scanned even when all parents previously passed. It must not include, call, import, dynamically load, or link rocBLAS, hipBLAS, hipBLASLt, Tensile, Composable Kernel/CK, or MIOpen, nor delegate through PyTorch matmul/mm/bmm/linear. Only separately frozen immutable baseline/oracle paths are exempt. HIP runtime/language APIs, MFMA/device intrinsics, inline AMDGCN assembly, and header-only rocWMMA remain allowed when the candidate ELF has no forbidden dependency. Run `python3 $SKILL_DIR/scripts/candidate_policy_scan.py` on the materialized candidate sources/build files before correctness and on every candidate ELF after build. Save the receipts in `INTEGRATE_DIR`; any finding, inspection error, or absent receipt fails closed and the combination cannot be returned or archived.

## Strategy
1. Work in a private copy:
   ```bash
   # NO `rm` (prompts + blocks autonomous runs). Unique private ws each time; tar-copy EXCLUDING build
   # artifacts (.torch_ext build.ninja has absolute paths to CANONICAL), so nothing stale is inherited.
   WS="$INTEGRATE_DIR/ws_$(date +%s)_$$"; mkdir -p "$WS"
   ( cd "$CANONICAL" && tar --exclude='./.git' --exclude='*/.git' --exclude=./build --exclude='*/build' \
       --exclude=./__pycache__ --exclude='*/__pycache__' --exclude=./.torch_ext --exclude='*/.torch_ext' \
       --exclude='*.so' --exclude='*.o' -cf - . ) | ( cd "$WS" && tar -xf - )
   cd "$WS"
   git init -q && \
     git -c user.email=team@workflow -c user.name=team add -A && \
     git -c user.email=team@workflow -c user.name=team commit -q --allow-empty -m workspace_baseline || exit 1
   ```
2. Sort patches by verified speedup (best first). Check compatibility using
   `optimization_strategies.md` (compatible: template+launch-bounds, tiling+coalescing, warp-coop +
   native-layout/wrapper; incompatible: two tiling schemes, two warp-coop schemes).
3. **Incremental stack**: `git apply` the best patch, then try adding each next patch. After each
   add: clear cache → correctness → benchmark (gpu_lock). Keep an add only if it stays correct and
   improves geomean.
4. **Hand-merge on conflict**: if `git apply` rejects, read both patches and manually implement both
   ideas in a compatible way (e.g. fold a host_runtime native-layout change into an algorithm
   engineer's templated kernel). This is encouraged — the best result is often a hand-merge, not a
   diff stack. Respect hipify safety (template dispatch, no `<<<>>>` in macro if/else).

**Never weaken the lock's idleness gate to unblock yourself — finding (128).** When `gpu_lock.sh`
blocks, or exits `no free+idle GPU in pool [...]`, the pool is held by a job it can see through
sysfs and `flock` cannot: on this host that has meant a second container running all cards at
95–100% busy with ~152 GB resident each. Setting `GEAK_GPU_REQUIRE_IDLE=0`, raising
`GEAK_GPU_MAX_BUSY_PCT` / `GEAK_GPU_MAX_VRAM_MB`, or exporting `HIP_VISIBLE_DEVICES` around the
lock all "work" — they hand you a GPU, your commands run, and every number you report afterwards
was measured against someone else's load. That is strictly worse than being blocked: a blocked
run announces itself, a contaminated one does not, and it is indistinguishable from a real
result for as long as anyone believes it. `GEAK_GPU_POOL_WAIT` (how long to WAIT) is the only one
of these you may raise. If the wait expires, STOP and report the contention AS your result —
the pool was busy, for how long, and what sysfs said. "Could not measure" is a usable answer;
a number produced under contention is not.

5. Always clear cache before benchmarking; always correctness before benchmark; gpu_lock for all
   benchmarks. Compute per-case speedup vs `BASELINE_PER_CASE`, geomean = `exp(mean(log(...)))`.
   **If the COMMANDMENT's METRIC is the time-weighted ratio-of-sums (workload-aligned), ALSO report
   `weighted = Σ weight_i / Σ (weight_i / speedup_i)`; that is the number compared to
   `BEST_INDIVIDUAL` (which is already the primary metric).**

6. **Two receipts travel with the result, and without them it is discarded (findings (69) and (87)).**
   A merge is a new candidate — it is where a link line gets rewritten and where an edit lands in one
   of a hipify pair — so neither parent's clean record carries over.
   - Copy the `summary` block of your post-build `candidate_policy_scan.py` receipt **verbatim** into
     `policy_postbuild`. The orchestrator re-derives the verdict from its numbers; a `policy_pass:true`
     with no block behind it is refused, and a path in `policy_receipts` does not travel with the report.
   - On a HIP/CUDA lane, run `python3 $SKILL_DIR/scripts/hip_twin_sync.py <your ws src dir>` after the
     final build and before the timing you report, and return `hip_twin_sync` with the tool's real
     `exit_code`, `pairs` and `drifted`. **Exit 2 means no pair was found and nothing was checked — it is
     the tool's `HOLE:`, not its `ok:`, and it is not a pass.** Exit 1 means the merge you timed is not
     the merge you wrote. Both are refused, as is a fabricated `exit_code:0` with `pairs:0`.

## Output
If the best combination beats `BEST_INDIVIDUAL`, save it:
```bash
cd "$WS" && git add -A && git diff --cached --binary > "$INTEGRATE_DIR/integrated_patch.diff"   # $WS = the unique private ws from step 1
```

## Return JSON
```json
{
  "attempted": true,
  "combos_tried": [
    {"patches": ["r1_d0","r1_d2"], "method": "incremental|hand_merge",
     "correctness": "pass|fail", "geomean": 0.0}
  ],
  "best": {"patches": ["..."], "geomean": 0.0, "arithmetic": 0.0, "weighted": 0.0,
            "patch_file": "<INTEGRATE_DIR>/integrated_patch.diff",
            "per_case": [{"name":"...","baseline_ms":0.0,"optimized_ms":0.0,"speedup":0.0,"weight":0.0}]},
  "improved_over_best_individual": true,
  "policy_pass": true,
  "policy_receipts": {"prebuild": "path/to/prebuild.json", "postbuild": "path/to/postbuild.json"},
  "policy_postbuild": {"schema": "...", "passed": true, "findings": 0, "advisory": 0,
    "inspected": 0, "files": 0, "elf": 0, "unreadable": 0},
  "hip_twin_sync": {"exit_code": 0, "pairs": 1, "drifted": 0, "checked": ["custom_gemm.hip"]},
  "conclusion": "improved|no_improvement|all_failed|policy_failed",
  "notes": "what combined well / what conflicted"
}
```
If nothing beats the best individual, return `conclusion:"no_improvement"` and no patch_file.
