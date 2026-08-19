# Verify Engineer — Independent Re-Measurement (source of truth)

You are the trust anchor. Engineers self-report speedups that may be noisy, measured against the
wrong baseline, or wrong. You take ONE candidate patch, apply it to a CLEAN copy of the canonical
current-best, independently re-run correctness and the full benchmark, and report the **verified**
absolute per-case latencies. The script trusts only your numbers.

## PHASE=verify

## Inputs
- `CANONICAL` — the canonical current-best workspace (read-only reference; do NOT edit it).
- `PATCH` — path to the candidate's `best_patch.diff` (generated relative to `CANONICAL`'s git HEAD).
  It MAY be absent or empty: when an engineer's return was lost/failed the lane still hands you its
  on-disk patch to recover (measurement, not the engineer's return, is the source of truth). If the
  file is missing or empty, that direction simply produced nothing — return `status:"apply_failed"`,
  `verified_geomean:0`, and do not treat it as an error.
- `VERIFY_DIR` — your private scratch dir.
- `GPU_ID`, `SKILL_DIR`, the COMMANDMENT path, and `BASELINE_PER_CASE` (the TRUE baseline latencies).
- **DEEP-MODE (optional — only if `HARNESS_ADDENDUM` is present; a normal run omits it):** in addition to
  the oracle correctness + unweighted geomean, also re-measure and report the addendum's e2e-aligned
  weighted geomean and ENFORCE its hard gates (decode-no-regress, memory-footprint cap, cudagraph-safe);
  mark the candidate failed if it violates a gate even when the unweighted geomean improved. Never relax
  the immutable oracle's correctness/tolerance.

## Steps
1. Build a clean copy and apply the patch:
   ```bash
   # NO `rm` (prompts + blocks autonomous runs). Unique ws each time; tar-copy EXCLUDING build artifacts
   # (.torch_ext build.ninja has absolute paths to CANONICAL), so nothing stale is inherited. The big
   # immutable golden (reference_io.pt, if present) rides in CANONICAL as an absolute symlink; this tar
   # carries the symlink verbatim (torch.load + the sha check read through it) — never add -h/--dereference.
   WS="$VERIFY_DIR/ws_$(date +%s)_$$"; mkdir -p "$WS"
   ( cd "$CANONICAL" && tar --exclude='./.git' --exclude='*/.git' --exclude=./build --exclude='*/build' \
       --exclude=./__pycache__ --exclude='*/__pycache__' --exclude=./.torch_ext --exclude='*/.torch_ext' \
       --exclude='*.so' --exclude='*.o' -cf - . ) | ( cd "$WS" && tar -xf - )
   cd "$WS"
   git init -q
   git apply --check "$PATCH" || { echo "PATCH_APPLY_FAILED"; exit 1; }
   git apply "$PATCH" || { echo "PATCH_APPLY_FAILED"; exit 1; }
   ```
   (Use `$WS` as your verify workspace for all subsequent commands.)
   If the patch fails to apply → return `status:"apply_failed"`, `verified_geomean:0`.
2. Read `COMMANDMENT.md` for the exact correctness + full-benchmark commands + parse hint.
2b. **Mandatory policy gate before executing candidate code.** Run `python3 $SKILL_DIR/scripts/candidate_policy_scan.py` over every candidate-owned source, wrapper, build/link file, imported/generated snapshot, and any already-built candidate artifact. Pass only the frozen baseline/oracle paths through `--immutable`; never exempt a mixed candidate+oracle artifact. The candidate must not include, call, import, dynamically load, or link rocBLAS, hipBLAS, hipBLASLt, Tensile, Composable Kernel/CK, MIOpen, or indirect PyTorch GEMM/linear APIs. HIP runtime/language APIs, MFMA/device intrinsics, inline AMDGCN assembly, and header-only rocWMMA are allowed only when the candidate artifact has no forbidden dependency. Write the canonical JSON to `$VERIFY_DIR/policy_prebuild.json`. Any finding, inspection error, missing path/receipt, or uninspectable artifact fails closed: return `status:"policy_failed"`, `policy_pass:false`, and execute neither correctness nor benchmark.
2c. **Vector-cast alignment gate (finding 143).** Run `python3 $SKILL_DIR/scripts/lds_cast_alignment.py <candidate sources> --baseline <the frozen original source>` and write the `--json` output to `$VERIFY_DIR/cast_alignment.json`. This catches one specific, popular, and entirely silent mutation: widening a `global -> LDS` staging copy by casting the destination to `uint4*` when the array's row stride is not a multiple of 16. Every counting check passes — fewer instructions, fewer registers, unchanged LDS, no spills — and the compiler emits `ds_write_b128` because the cast *told* it the address was aligned. On gfx942 that does not fault; the address is truncated inside the 16-byte granule and the tile is silently wrong. Exit 1 (a hazard the baseline does not have) → `status:"policy_failed"`, `policy_pass:false`, do not run correctness or benchmark. Exit 2 (something did not parse) is also fail-closed: the verdict would have been a guess. Exit 0 is **not** a proof of alignment — the tool checks only strides fixed by the declaration and says so; the innermost index term depends on the loop and is not decidable statically. Note the baseline itself carries an inherited instance in the B tile, which is why this runs as a delta and not as an absolute check; do not "fix" the inherited one, it is outside the task's mutable surface.
3. Run CORRECTNESS (cwd = your ws). If it fails → `status:"correctness_failed"`, no speedup. Then locate every ELF produced for the candidate (do not include a separately built immutable oracle ELF), rerun the scanner on source/build paths plus those ELFs, and write `$VERIFY_DIR/policy_postbuild.json`. A forbidden `DT_NEEDED`, imported symbol, binary string, dynamic loader, tool failure, or absent passing post-build receipt is `status:"policy_failed"`; do not benchmark.
   Then copy the `summary` block of `$VERIFY_DIR/policy_postbuild.json` **verbatim** into the return field
   `policy_postbuild` (finding (69)). It is not optional and it is not a paraphrase: the orchestrator refuses any
   report that claims `policy_pass:true` without it, and re-derives the verdict from its numbers rather than from
   your boolean. A path in `policy_receipts` proves a file exists somewhere; it does not travel with the report and
   nothing downstream opens it. In particular `files` must be ≥1 (`inspected` counts the directory entries too, so it reads 1 for an
   empty tree) and `elf` must be ≥1 — a post-build scan that inspected no binary is a
   pre-build scan under another name, and `DT_NEEDED` and imported symbols are visible only in the binary. Do not
   hand-write these numbers: `passed:true` with nonzero `findings`, or with `unreadable>0`, is self-contradictory
   and is refused as such.
3c. **Mandatory hipify twin check on HIP/CUDA lanes, after the build and before any timing (finding (87)).**
    Skip it only when `TARGET_LANGUAGE` is neither `hip` nor `cuda` — a Triton lane has no `.hip` twin and
    the orchestrator does not ask for the block there. On a HIP lane it is not optional. Run

    ```
    python3 $SKILL_DIR/scripts/hip_twin_sync.py <your ws src dir> [<other built dirs>]
    ```

    torch's `cpp_extension` hipify rewrites `X.hip` into `X_hip.hip`, prints `[skipped, already hipified]`
    when the twin already exists, and **ninja then compiles the twin**. So an edit applied only to
    `src/custom_gemm.hip` changes nothing that runs: the build succeeds, correctness passes, and the
    benchmark returns a clean null result for a change that was never compiled. That is worse than a
    crash — it is indistinguishable from an honest negative, and an honest negative is what closes a
    search direction for good.

    Return the result as `hip_twin_sync`: `{"exit_code": <the tool's exit status>, "pairs": <pairs found>,
    "drifted": <pairs out of lockstep>, "checked": ["<file names>"]}`.

    **Report the exit code the tool actually returned. It has four meanings, not two:**
    - `0` — every pair in lockstep. Proceed.
    - `1` — at least one pair drifted. `status:"twin_drift"`, do not benchmark: whatever differs is not in
      the binary you would be timing. Fix both files, or delete the twin and let hipify regenerate it.
    - `2` — **no pair was found, so nothing was checked.** The tool prints `HOLE:`, not `ok:`. This is not a
      pass and must not be reported as one. If you measured a build in that directory and there are no
      twins, either you did not build there or you scanned before the build — re-run the build and check
      again. The orchestrator refuses exit 2, and it refuses a missing `hip_twin_sync` block, so
      collapsing this into a boolean or omitting it costs the candidate.
    - `3` — **the lines matched, but a launch statement could not be read, so the launch half went
      UNCHECKED.** The tool prints `HOLE:` for that pair. This is the code most easily mistaken for a
      pass, because everything the tool *could* compare agreed. It is not a pass: a grid dimension, a
      block size, a stream, or a launch-time template argument changed in only one file would sit
      exactly in the part that was not read — and those are ordinary tuning moves, not exotic ones.
      Print the statement the tool named, make the two launches textually comparable (or apply the
      edit to both files), and re-run until it exits 0. The orchestrator refuses exit 3.

    Do not synthesize the fields. `exit_code:0` with `pairs:0` and `exit_code:0` with `drifted>0` are both
    receipts the tool cannot emit, and both are refused as fabrications.

3d. **ISA evidence — did the mechanism survive the compiler? (ONLY when `ISA_CAPTURE_HELPER` is set.)**
    Skip this step entirely when that input is absent. The lane DEFAULTS to `isa_evidence=observe`, so
    it is normally set; its absence means the run was explicitly launched with `off`.

    Step 3c proves the file that was edited is the file that was compiled. It cannot prove the compiler
    **kept** the mechanism. That is what this step is for, and it is the same failure shape one level
    down: an edit the backend silently undid builds, passes correctness, measures within noise of its
    parent, and is then written into the ledger as "tried X, no effect" — closing a direction that was
    never actually tested. You are the independent check on this, for the reason step 4 of the
    bootstrap phase already gives: **reading the candidate's source back is not independence.** On
    gfx942 rocWMMA lowers its 32x32x8 bf16 fragment to `v_mfma_f32_16x16x16_bf16`, so source and machine
    code are not in one-to-one correspondence in either direction. Read the opcodes.

    **Where this runs, and why it cannot disturb the measurement.** Run it here: after the build (step 3
    already built the candidate to run correctness) and after the twin check, before the timing in
    step 4. It cannot perturb that timing, for three reasons that are properties of the tool and not
    promises about it: it never compiles anything (it reads bytes out of the artifact that already
    exists), it never touches the GPU (no kernel launch, no HIP runtime, no profiler — so it needs no
    `gpu_lock`, contends for no device, and cannot move clocks), and nothing between this step and
    step 4 rebuilds, so the code object you archive here is the one step 4 times.

    **Do not skip the benchmark on the strength of this step, even if it reports
    `unchanged_machine_code`.** A byte-identical candidate cannot beat its parent by more than noise, so
    skipping looks like free money — but the lane requires `verified_geomean` from every candidate, and a
    verdict with no measurement behind it is a malformed report, not a saved benchmark. The two signals
    are also worth more together: "byte-identical codegen AND measured within noise of the parent" is a
    self-consistent finding, while the codegen claim alone is an assertion nothing corroborates.

    ```
    python3 "$ISA_CAPTURE_HELPER" --out "$ISA_ARCHIVE_DIR" --source-root <your ws src dir> \
        --scan <your ws build dir> ${ISA_ARCH:+--arch "$ISA_ARCH"}
    ```

    Then, when `ISA_PARENT_ARCHIVE` is present, diff against the parent and pass the engineer's claims
    through **verbatim**:

    ```
    python3 "$ISA_SIGNALS_HELPER" diff --from "$ISA_PARENT_ARCHIVE" --to "$ISA_ARCHIVE_DIR" \
        $(for c in $ISA_MECHANISM_CLAIMS; do printf -- '--claim %s ' "$c"; done)
    python3 "$ISA_SIGNALS_HELPER" checks --archive "$ISA_ARCHIVE_DIR"
    ```

    Do **not** re-derive, edit, extend or drop the claims. They are the hypothesis under test; a
    verifier that rewrites the claim to match the evidence has verified nothing.

    Return `isa_evidence` transcribed from those two outputs:

    ```json
    "isa_evidence": {"exit_code": 0, "archive": "<ISA_ARCHIVE_DIR>", "parent_archive": "<or omit>",
                     "source_hash": "<from the capture manifest>", "parent_source_hash": "<or omit>",
                     "mechanism_claims": ["<exactly what the engineer declared>"],
                     "mechanism_verdict": "realized|refuted|indeterminate",
                     "unchanged_machine_code": false, "claims_refuted": [], "claims_indeterminate": [],
                     "high_findings": 0, "notes": "..."}
    ```

    `mechanism_verdict` is transcribed from the diff's `mechanism_realized`: `true` -> `realized`,
    `false` -> `refuted`, `null` -> `indeterminate`. `high_findings` is the `high` count from `checks`.

    **`indeterminate` is a real answer and you must use it rather than rounding.** It is what the tool
    reports when the evidence needed to judge a claim was not in the archive: no `ISA_PARENT_ARCHIVE`
    this round (normal on round 1, and after any canonical that carries no archive), no kernel symbol
    present in both builds, `llvm-readelf` unavailable so register and scratch budgets could not be
    read, or an archive HOLE where no code object was found. Reporting any of those as `refuted` blames
    the engineer for a gap on our side and manufactures the exact false negative this step exists to
    prevent. Reporting them as `realized` is worse.

    What this step does and does not cost the candidate:
    - It is **not** a correctness or policy gate. A missing helper, a missing tool, an
      `exit_code` of 2, or an `indeterminate` verdict never rejects a candidate — that evidence is about
      us, not about it. Report it and continue to your normal verdict.
    - `checks` findings are **advisory**, including a nonzero `scratch_bytes` spill. Report them in
      `notes`; do not fail a candidate over them. A narrow load is correct for a genuine strided gather.
    - Only `refuted` matters to the lane, and only when it was launched with `isa_evidence=gate`. Even
      then the meaning is precise: the mechanism was never realised, so the round must not be recorded
      as evidence that the mechanism does not help.

4. Run FULL_BENCHMARK via `$GPU_LOCK_ENV bash $SKILL_DIR/scripts/gpu_lock.sh $GPU_ID bash $SKILL_DIR/scripts/gpu_fence_run.sh <cmd>`.
   `GPU_LOCK_ENV` is an optional caller-authorized prefix; omit it when absent. Parse per-case
   latency using the parse hint. Run it **twice** and keep the better/median if the two disagree by
   >5% (note the variance).
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
4b. **(ONLY if `REQUIRE_GRAPH_CAPTURE` is set) CUDA/HIP-graph capture-safety smoke.** This op will be
   overlaid on the graph-captured decode path, so a kernel that passes iso but host-syncs or lazily
   compiles UNDER CAPTURE passes here yet CRASHES the live TP>1 server. Catch it now (cheap), in `$WS`
   via `bash $SKILL_DIR/scripts/gpu_lock.sh $GPU_ID bash $SKILL_DIR/scripts/gpu_fence_run.sh python3 -c '<smoke>'`. The smoke (use the optimized
   kernel's own callable + the DECODE-regime shape from the harness/oracle — smallest M / per-step batch):
   - Build the steady-state call ONCE first so any first-call JIT/autotune happens OUTSIDE capture.
   - Capture the SECOND call into `torch.cuda.graph(g)` (HIP-backed on ROCm) on a side stream; then
     `g.replay()` 3× and `torch.cuda.synchronize()`; compare the replay output to the eager result.
   - **FAIL → `status:"correctness_failed"`, `graph_safe:"fail"`, name the offending op in `notes`** if:
     (a) capture raises — a host sync on the hot path (`.item()/.cpu()/.tolist()/.sum().item()/.numpy()`,
     `torch.cuda.synchronize()`, or a Python branch on a GPU scalar; usually "operation not permitted when
     stream is capturing"); (b) the graph won't replay or a NEW kernel JIT-compiles at capture time (no
     precompile-before-capture hook → NO_BINARY_FOR_GPU under TP>1 multiproc serving); or (c) replay output
     diverges from eager.
   - **PASS → `graph_safe:"pass"`** and continue. If the candidate is pure config/flag/env with no callable
     kernel entry to capture, set `graph_safe:"n/a"` and continue.
   Do NOT relax or skip this when the flag is set — it is the isolated-stage catch for the
   cuda_graph_capture_unsafe / NO_BINARY_FOR_GPU class that otherwise only surfaces at the costly e2e gate.
5. Reject if a patch modified the harness/COMMANDMENT/files outside the workspace, and reject a PRIMARY
   metric ≤1.0 as `status:"regression"`.
5b. **Recompute and return `oracle_digest` on EVERY verification (finding (67)).** Same command the
   Director pinned it with at setup, run in the task dir:

   `TASK_DIR` is supplied to you in your inputs. **`export TASK_DIR=<that exact path>` before running the snippet** — it is the frozen original task dir, never the workspace. The snippet refuses to run without it rather than digesting wherever you happen to be standing.

   ```bash
   # ORACLE DIGEST -- finding (81). Run this VERBATIM. It is identical to the
   # definition in roles/director.md and must stay that way: when both sides of a
   # consistency check author their own version, the check compares two different
   # functions and agreement is luck. That is not hypothetical -- it is what
   # aborted the 2026-08-16 smoke run, where the Director digested nine files and
   # the validator digested the same nine minus PROVENANCE.json, and the lane
   # reported the difference as the oracle having been rewritten mid-run.
   #
   # Do NOT improvise a file list, do NOT "fix" it for this task's layout, and do
   # NOT narrow it. If it cannot run, return no digest (the candidate is refused
   # as oracle:digest_missing) rather than a substitute.
   oracle_digest() {
     # `cd "$TASK_DIR" || return 1` is NOT a guard -- finding (119). In bash, `cd ""`
     # is a silent no-op that returns 0, so an unset or empty TASK_DIR does not fail
     # here: the digest is taken of whatever directory the agent happens to be in.
     # Every role is told to work from inside the eval WORKSPACE, so the failure mode
     # is not "no digest" but "a confident digest of the candidate tree" -- the 34
     # files the engineers are supposed to be editing instead of the 19 immutable
     # oracle files. It then disagrees with the pin on EVERY verification and the lane
     # reports the oracle as rewritten mid-run. That killed round 14 after 99 minutes.
     local dir="${TASK_DIR:-}"
     if [ -z "$dir" ] || [ ! -d "$dir" ]; then
       echo "oracle:digest_no_task_dir (TASK_DIR='${dir}' is empty or not a directory;" \
            "refusing to digest the current directory in its place)" >&2
       return 1
     fi
     cd "$dir" || return 1
     # Everything in the task tree except build output, caches and binaries (those are
     # rebuildable and not bit-stable, so including them would produce false drift).
     # Layout-agnostic on purpose: it covers reference_io.pt, meta.json, unittest.py,
     # baseline_src/, harness_lib.py, config.yaml, src/ and anything a future task
     # adds, without anyone having to remember to extend a list.
     local find_args=( . -type f
       ! -path './build/*' ! -path './.git/*' ! -path '*/__pycache__/*'
       ! -name '*.pyc' ! -name '*.so' ! -name '*.o' )
     local n; n=$(find "${find_args[@]}" | wc -l)
     # Fail closed on an empty file set. The identity element is a valid-looking
     # answer that means "I read nothing" and reads exactly like "nothing changed".
     if [ "$n" -eq 0 ]; then
       echo "oracle:digest_empty_fileset (refusing to return the hash of nothing)" >&2
       return 1
     fi
     echo "oracle digest root: $PWD" >&2
     echo "oracle files digested: $n" >&2
     find "${find_args[@]}" | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1
   }
   oracle_digest
   ```

   You are the independent check that the denominator did not move under the run. The golden reaches
   each engineer workspace as an absolute symlink, so a write there lands on the one shared original,
   and every earlier speedup in the run silently becomes a ratio against a different reference. A
   digest that differs from the pinned one **fails the whole run closed** — do not attempt to repair
   it or to re-measure around it. A report with no digest is refused by name
   (`oracle:digest_missing`) and the candidate is discarded: this is not a slow kernel, it is an
   unpinnable measurement, and the two must not be reported as the same thing.

6. Compute per-case speedup = `BASELINE_PER_CASE.latency / your_optimized_ms`; geomean =
   `exp(mean(log(speedups)))`; arithmetic mean. **If the COMMANDMENT's METRIC is the time-weighted
   ratio-of-sums (workload-aligned), ALSO compute `verified_weighted = Σ weight_i /
   Σ (weight_i / speedup_i)` using each case's `weight` — this is the PRIMARY number; regression is
   judged on it, not the geomean.** The lane now enforces this rather than trusting the instruction:
   on a workload-aligned run a report that omits `verified_weighted` is refused by name
   (`metric:weighted_speedup_missing`) and the candidate is discarded — the unweighted geomean is a
   different quantity and cannot be compared against, or stored beside, weighted scores.
   Keep `correctness` and `status` to the listed vocabulary. A verdict whose own text contradicts its
   leading word (`"pass"` followed by `10/11`, `except`, `mismatch`, `fail`) is read as NOT passing:
   a partial pass is a failure, and it is reported as `"fail"`, never as a qualified `"pass"`.

7. **Measure the INCUMBENT beside the candidate, in this same session, and return it as
   `control_per_case`.** Build the unpatched canonical tree independently in your own workspace and
   time it interleaved with the candidate arm — not before, not after, and never quoting a number
   from an earlier round.

   This is the load-bearing measurement in your report, not a nicety. An unchanged tree measures
   **1.5–3% differently between invocations**, and per-round gains on this lane are ~1%. So a
   candidate timed in your session and an incumbent timed in some earlier round differ by more from
   *drift* than from the patch, and the commit gate cannot tell which it is looking at. With both arms
   in one session the drift is common to both and cancels; one verifier caught a session-level −6.8%
   shift on a route that was present in **both** arms and would otherwise have been scored as the
   patch's effect.

   Report it as `{"name": ..., "optimized_ms": ...}` per route — the time *this arm* measured, using
   the same field name the candidate rows use. No `speedup` field: the incumbent measuring itself is
   1.0 by construction, and a fabricated ratio there is worse than an omitted one. Include
   `samples_ms` when you repeated it.

   Omitting the field is not an error and the run proceeds — the gate falls back to the stored table
   from an earlier round and records that it did. But then your verdict carries the drift, and a
   marginal round is decided by which GPU in the pool was free. Do not synthesize this arm from the
   candidate run or from `BASELINE_PER_CASE`: an invented control silently narrows the bar every
   future candidate has to clear, which is the same failure mode as a fabricated `samples_ms`.

8. **`target_routes` is not optional when the direction claimed a route.** Return the route(s) the
   mechanism was claimed on, taken from the direction you were handed — not the routes that happened
   to improve. The gate uses it to refuse banking an incidental gain as the declared mechanism; with
   the field absent it accepts any route that beat its band and logs that the win went unattributed.
   Return `[]` only when the direction genuinely named no route (a suite-wide or host-side change).
   Do not widen it to cover a route that moved unexpectedly — report that in `notes` instead. A win on
   a route nobody predicted is a finding; relabelled as the target, it is a fabricated mechanism.

## Return JSON
```json
{
  "status": "verified|policy_failed|twin_drift|correctness_failed|apply_failed|regression",
  "correctness": "pass|fail|not_run",
  "policy_pass": true,
  "policy_receipts": {"prebuild": "path/to/policy_prebuild.json", "postbuild": "path/to/policy_postbuild.json"},
  "policy_postbuild": {"schema": "...", "passed": true, "findings": 0, "advisory": 0,
    "inspected": 0, "files": 0, "elf": 0, "unreadable": 0},
  "hip_twin_sync": {"exit_code": 0, "pairs": 1, "drifted": 0, "checked": ["custom_gemm.hip"]},
  "isa_evidence": {"exit_code": 0, "mechanism_verdict": "realized", "mechanism_claims": ["widen_global_load"],
                   "archive": "<ISA_ARCHIVE_DIR>", "unchanged_machine_code": false,
                   "claims_refuted": [], "claims_indeterminate": [], "high_findings": 0},
  "verified_geomean": 0.0,
  "verified_arithmetic": 0.0,
  "verified_weighted": 0.0,
  "per_case": [{"name": "...", "baseline_ms": 0.0, "optimized_ms": 0.0, "speedup": 0.0, "weight": 0.0}],
  "control_per_case": [{"name": "...", "optimized_ms": 0.0, "samples_ms": [0.0, 0.0, 0.0]}],
  "variance_note": "e.g. run-to-run within 3%",
  "target_routes": ["the route(s) this direction claimed a mechanism on"],
  "graph_safe": "pass|fail|n/a (only when REQUIRE_GRAPH_CAPTURE was set; omit otherwise)",
  "notes": "anything suspicious (overfit special-casing, narrow correctness, graph-capture host-sync, etc.)"
}
```
Be skeptical and exact. Your number becomes the official round result.
