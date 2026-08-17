export const meta = {
  name: 'kernel-lane',
  description: 'SINGLE-LANGUAGE kernel optimization worker (Director/TechLead/specialist Engineers) with budget-controlled rounds, independent verification, integration, and optional route-aware QD search. Optimizes ONE kernel in ONE language (mode=optimize) or authors a fresh seed then optimizes it (mode=author). This is the worker invoked per lane by the kernel-workflow dispatcher (kernel_workflow.js) and by e2e_workflow; prefer calling kernel-workflow directly unless you specifically want one unchanged lane. Target: AMD Instinct MI-series GPUs; ordinary greedy runs auto-detect the card, while geak-qd-v2 currently supports gfx90a and gfx942 explicitly.',
  whenToUse: 'Internal single-language worker. Prefer the kernel-workflow dispatcher (kernel_workflow.js) as the entry point; invoke this directly only to run one unchanged lane. Pass args.kernel_path (required), args.mode, args.target_language, args.budget, args.gpu_ids, args.gpu_mode, args.task.',
  phases: [
    { title: 'Setup', detail: 'director builds the isolated eval dir + canonical workspace' },
    { title: 'Author', detail: 'author_engineer writes a fresh optimize-loop seed (only when mode=author); speedup denominator stays the frozen online kernel' },
    { title: 'Analyze', detail: 'tech_lead analyzes kernel + writes roadmap' },
    { title: 'Benchmark', detail: 'benchmark_engineer builds the COMMANDMENT + baseline' },
    { title: 'Profile', detail: 'profile_engineer classifies the bottleneck' },
    { title: 'Optimize', detail: 'budget loop: tech_lead plans, specialist OR deep_explore engineers optimize, reprofile' },
    { title: 'Verify', detail: 'each candidate patch independently re-benchmarked' },
    { title: 'Merge', detail: 'integrator combines the round winners' },
    { title: 'Report', detail: 'tech_lead writes the final report + patch' },
    { title: 'Validate', detail: 'director independently validates vs the true baseline' },
  ],
};

// ---------------------------------------------------------------------------
// Args + defaults. (The script cannot touch the filesystem or read its own
// path; agents do all FS work, and every path is supplied/derived from args —
// nothing about the install location is hard-coded.)
// ---------------------------------------------------------------------------
const A = args || {};
if (!A.kernel_path) throw new Error('args.kernel_path is required (absolute path to the kernel/model directory)');

// WORKFLOW_DIR = the directory that holds this script + roles/ + knowledge/ + scripts/.
// A JS workflow script can't read its own path, so the caller passes it (it is just the
// dirname of the scriptPath used to launch the workflow).
const WORKFLOW_DIR = String(A.workflow_dir || '').replace(/\/+$/, '');
if (!WORKFLOW_DIR) {
  throw new Error('args.workflow_dir is required: absolute path to the directory containing ' +
    'kernel_workflow.js, roles/, knowledge/, scripts/ (i.e. the dirname of this script).');
}
// EXP_ROOT = where timestamped run dirs are written. Default: a sibling "exp/" next to the
// kernel_workflow dir (…/<parent>/kernel_workflow -> …/<parent>/exp). Override with args.exp_root.
const EXP_ROOT = String(A.exp_root || (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/exp')).replace(/\/+$/, '');

const KERNEL_PATH_ORIG = A.kernel_path;
const BUDGET = parseInt(A.budget != null ? A.budget : 6, 10);
// Minimum verified geomean improvement over the cumulative best for a round winner to be COMMITTED
// into the canonical workspace (default 2%). Kept as a knob rather than a hard-coded constant so the
// gate is tunable per run (e.g. raise it on a noisy box, lower it to bank small compounding wins).
const MIN_IMPROVE = (() => {
  const v = parseFloat(A.min_improve != null ? A.min_improve : 0.02);
  return Number.isFinite(v) && v >= 0 ? v : 0.02;
})();
// Minimum verified speedup for a candidate to enter the round's candidate list (default 1.0 = only a
// candidate that beats the baseline is worth looking at). A knob for the same reason MIN_IMPROVE is:
// a transcription (plain Triton -> Gluon/TileLang/HIP) lands BELOW the comparator by construction, and
// at 1.0 its recovery phase is invisible -- no patch saved, no verify, `winner` null every round. The
// COMMIT gate is separate and still requires beating `cumulative` by MIN_IMPROVE, so a sub-baseline
// candidate can be TRACKED but never BANKED.
const CANDIDATE_FLOOR = (() => {
  const v = parseFloat(A.candidate_floor != null ? A.candidate_floor : 1.0);
  return Number.isFinite(v) && v > 0 ? v : 1.0;
})();
// Rendered into the Optimize prompt, where `${1.0}` would stringify to "1" and silently reword a
// prompt that has always said "geomean>1.0". Keeps the default run byte-identical.
const CANDIDATE_FLOOR_TXT = Number.isInteger(CANDIDATE_FLOOR)
  ? CANDIDATE_FLOOR.toFixed(1) : String(CANDIDATE_FLOOR);
// How far BELOW the best candidate ever seen a round may land and still count as "the search is
// advancing" (default +MIN_IMPROVE = the historical test). A knob because a climb that starts under
// the comparator advances for many rounds without ever clearing `cumulative`, and scoring those as
// stalls ends it at MAX_NO_IMPROVE however much budget was given -- and no static counter substitutes,
// since it would have to pre-guess how many rounds the climb takes. Negative admits a round that gives
// ground (a layout experiment that does not pay is information, not a stall).
const PROGRESS_DELTA = (() => {
  const v = parseFloat(A.progress_delta != null ? A.progress_delta : MIN_IMPROVE);
  return Number.isFinite(v) && v > -1 ? v : MIN_IMPROVE;
})();
// Budget cost of ONE `deep_explore` direction. The deep-explore engineer does far more than a single
// specialist — broad rewrite authority, its own multi-iteration measure→profile→rewrite loop — so it
// is charged more than 1 against the direction budget (default 2). It also always runs in a DEDICATED
// round (no other directions that round), enforced below.
const DEEP_COST = (() => {
  const v = parseInt(A.deep_cost != null ? A.deep_cost : 2, 10);
  return Number.isFinite(v) && v >= 1 ? v : 2;
})();
const GPU_IDS = String(A.gpu_ids != null ? A.gpu_ids : '0');
const GPU_LIST = GPU_IDS.split(',').map(s => s.trim()).filter(Boolean);
// Every GPU consumer gets the WHOLE pool rather than a pinned lane. gpu_lock.sh
// resolves a comma spec by flocking whichever lane is free AND idle at acquire
// time, so placement follows what work actually costs instead of an index fixed
// before the cost is known. Measured task costs span 23x on campaign20, and a
// pinned round ends with the slowest LANE, not the slowest task -- at 4-way that
// idles ~43% of lane time. Pinning is also fragile on a shared box: GPU_LIST[0]
// has no fallback when lane 0 has a foreign tenant.
// gpu_mode='pin' restores the pre-scheduler behavior (direction i pinned to GPU_LIST[i % n]).
// It exists ONLY so the scheduler can be A/B'd as a single-variable change against the arm it
// replaced; leaving it out would make the "before" arm unreachable and force the comparison to be
// made across two different scripts, where any other drift would be indistinguishable from the
// scheduler's effect. Default 'pool'.
const GPU_MODE = String(A.gpu_mode || 'pool') === 'pin' ? 'pin' : 'pool';
const GPU_POOL = GPU_MODE === 'pin' ? GPU_LIST[0] : GPU_LIST.join(',');
// Optional caller-authorized environment prefix for gpu_lock.sh (for example a known-idle card that
// retains allocator VRAM). Empty by default, so ordinary runs preserve the strict idle/ownership policy.
const GPU_LOCK_ENV = String(A.gpu_lock_env || '').trim();
const GPU_LOCK_INPUT = GPU_LOCK_ENV ? { GPU_LOCK_ENV } : {};
const TASK = A.task || '';
const EVAL_DIR_OVERRIDE = A.eval_dir || '';
const APPLY_TO_ORIGINAL = String(A.apply_to_original != null ? A.apply_to_original : 'false');
const KERNEL_NAME_HINT = KERNEL_PATH_ORIG.replace(/\/+$/, '').split('/').pop();

// --- author mode: when there is NO existing source, write a fresh from-scratch SEED first, then optimize it.
// mode=optimize (default) keeps the exact original behavior (backward compatible). mode=author seeds
// the workspace from an op task dir (immutable oracle + frozen online kernel in baseline_src/), the
// author_engineer writes a passing seed, then the SAME optimize loop runs — always timing against the
// frozen online kernel, never against the seed's own language. KERNEL_KNOWLEDGE_DIR is the AMD authoring
// knowledge base — REFERENCE ONLY (facts/how-to, never decisions; the author always measures regardless). Default:
// sibling perf_knowledge/ so standalone runs use it too; empty if WORKFLOW_DIR is unset (no behavior change).
const MODE = String(A.mode != null ? A.mode : 'optimize').trim() || 'optimize';
const TARGET_LANGUAGE = String(A.target_language != null ? A.target_language : 'triton').trim() || 'triton';
// Quality-diversity search is opt-in. Any absent or unknown value preserves the historical greedy path.
const SEARCH_STRATEGY = String(A.search_strategy != null ? A.search_strategy : 'greedy').trim().toLowerCase() === 'qd_archive'
  ? 'qd_archive' : 'greedy';
const QD_ENABLED = SEARCH_STRATEGY === 'qd_archive';
const QD_CLASSIFIER_VERSION = 'geak-qd-v2';
// No default. `qd_arch` used to fall back to 'gfx90a' on a fleet that has only
// ever been gfx942, and the fallback was invisible: the run started, every
// stage reported normally, and two things were quietly wrong for its whole
// length. (a) The SOL card scored against the gfx90a reference peaks -- whose
// own `source` field says "not an effective measurement" -- instead of the
// gfx942 card marked measured; on one fixed measurement that is
// remaining_headroom 0.241 vs 0.794, so a route with four fifths of its
// headroom left is ranked as nearly closed. (b) `xcd_remapped_grouped`, the
// only mechanism in the vocabulary that describes this chip's 8-XCD remap, is
// illegal on gfx90a, so every descriptor proposing it came back
// `rule:xcd_remap_requires_multi_die` -- which reads as "that mechanism is
// wrong" rather than "you never said which chip". Defaulting to gfx942
// instead would be the identical bug aimed at the next box, so QD runs now
// state the arch or do not start.
const QD_ARCH = String(A.qd_arch || '').trim().toLowerCase();
const QD_DTYPE = String(A.qd_dtype || 'bf16').trim().toLowerCase();
const QD_SOL_CALIBRATION = A.qd_sol_calibration && typeof A.qd_sol_calibration === 'object'
  ? A.qd_sol_calibration : null;
// (79). The arch default was removed directly above for the reason in that
// comment; the calibration-version default, which NAMES an arch, was left behind
// -- so a gfx942 run with no explicit calibration labelled its SOL card
// `gfx90a-reference-single-gcd-v1`. The label is what a warm-started run reads to
// decide whether a persisted card is reusable, so a card computed against gfx942
// peaks (668 TFLOPs, 2.576 TB/s) advertises itself as a gfx90a card. Derive the
// label from the arch that is now mandatory, and refuse a supplied label whose
// arch prefix disagrees with it: a version string that names the wrong chip is
// worse than no version string, because it is confidently wrong.
const QD_SOL_CALIBRATION_VERSION = String(
  (QD_SOL_CALIBRATION && QD_SOL_CALIBRATION.version) ||
  (QD_ARCH ? `${QD_ARCH}-reference-v1` : 'unpinned-arch-v1')
);
if (QD_ENABLED && QD_ARCH && !QD_SOL_CALIBRATION_VERSION.startsWith(`${QD_ARCH}-`)) {
  throw new Error(
    `qd_sol_calibration.version="${QD_SOL_CALIBRATION_VERSION}" does not name qd_arch=${QD_ARCH}. ` +
    'The SOL ceiling is arch-specific and this label is the only thing a warm-started run checks ' +
    'before reusing a persisted card, so a mislabelled calibration silently re-uses another chip\'s ' +
    'roofline. Fix the label or omit it to derive one.');
}
// Optional read-only QD warm-start. The Director copies the referenced archive into this run,
// validates its manifest/classifier, and returns untrusted candidates for current-harness re-verification.
const QD_STATE_DIR = String(A.qd_state_dir || '').replace(/\/+$/, '');
const QD_RECLASSIFY = String(A.qd_reclassify != null ? A.qd_reclassify : 'false') === 'true';
const QD_SUPPORTED_ARCHES = ['gfx90a', 'gfx942'];
// Architectures whose dispatcher hands consecutive workgroup ids to different
// dies round-robin, so blockIdx adjacency is not last-level-cache adjacency and
// un-shuffling it before grouping is a real traffic mechanism. gfx942 has 8
// XCDs behind 8 L2s; gfx90a's two GCDs are separate HIP devices that one kernel
// never spans, so there is nothing to undo.
const QD_MULTI_DIE_ARCHES = ['gfx942'];
if (QD_ENABLED && !QD_ARCH) {
  throw new Error(
    'geak-qd-v2 requires an explicit qd_arch (one of ' + QD_SUPPORTED_ARCHES.join(', ') +
    '); pass the card detected on-box. There is deliberately no default: the ' +
    'arch silently selects both the SOL ceiling and whether xcd_remapped_grouped ' +
    'is a legal mechanism at all.');
}
if (QD_ENABLED && !QD_SUPPORTED_ARCHES.includes(QD_ARCH)) {
  throw new Error(`geak-qd-v2 supports qd_arch in {${QD_SUPPORTED_ARCHES.join(', ')}}, got ${QD_ARCH}`);
}
if (QD_ENABLED && !['bf16', 'fp16', 'fp32', 'fp64', 'int8'].includes(QD_DTYPE)) {
  throw new Error(`geak-qd-v2 does not support qd_dtype=${QD_DTYPE} on ${QD_ARCH}`);
}
const QD_CELL_GUARDRAIL = (() => {
  const v = parseFloat(A.qd_cell_guardrail != null ? A.qd_cell_guardrail : 0.80);
  return Number.isFinite(v) && v > 0 ? v : 0.80;
})();
const QD_CANONICAL_GUARDRAIL = (() => {
  const v = parseFloat(A.qd_canonical_guardrail != null ? A.qd_canonical_guardrail : 0.95);
  return Number.isFinite(v) && v > 0 ? v : 0.95;
})();
const OP_SPEC = A.op_spec || {};
// When the op will run on the CUDA/HIP-graph-captured decode path (e2e sets op_spec.cuda_graph_safe=true),
// the isolated oracle alone CANNOT catch a kernel that passes iso but host-syncs or lazily-compiles under
// graph capture — the "wins isolated, crashes serving" class (cuda_graph_capture_unsafe / NO_BINARY_FOR_GPU).
// This turns on an OPTIONAL capture+replay smoke in the verify step so that failure is caught at the cheap
// isolated stage. Unset (standalone single-kernel runs / non-graph ops) => byte-identical to before.
const REQUIRE_GRAPH_CAPTURE = !!(OP_SPEC && OP_SPEC.cuda_graph_safe === true);
// WORKLOAD ALIGNMENT (optional). When the caller supplies the real-workload shape/dtype case
// distribution, the benchmark harness benchmarks EXACTLY those (shape, dtype) cases, weights each
// by its total time contribution in the workload (weight = count * baseline_latency), and the
// optimization target becomes the time-weighted ratio-of-sums instead of an unweighted geomean.
//   workload_spec_path : path to a workload-v1 json (produced by parse_profile.py --workload-out,
//                        or hand-written). The benchmark_engineer reads it (JS can't touch FS).
//   op_spec.workload   : inline cases, same shape as a workload-v1 "kernels[].cases" list (or the
//                        full object). Takes precedence; weight_source becomes "caller".
// Both unset => unweighted behavior, byte-identical to before. Correctness ALWAYS stays on the
// frozen immutable oracle (parity vs baseline_src/, + a recorded reference_io.pt where the task dir
// came from e2e's kernel_extractor); this only shapes the PERFORMANCE measurement.
const WORKLOAD_SPEC_PATH = String(A.workload_spec_path || (OP_SPEC && OP_SPEC.workload_path) || '').trim();
const WORKLOAD_SPEC = (OP_SPEC && OP_SPEC.workload) || A.workload || null;
const HAS_WORKLOAD = !!(WORKLOAD_SPEC_PATH ||
  (Array.isArray(WORKLOAD_SPEC) && WORKLOAD_SPEC.length) ||
  (WORKLOAD_SPEC && Array.isArray(WORKLOAD_SPEC.kernels) && WORKLOAD_SPEC.kernels.length));
// PRIMARY-metric selector: prefer the time-weighted number when a workload spec is in play and the
// agent reported one; otherwise fall back to the geomean (unweighted runs => unchanged behavior).
// Finding (62). `verified_weighted` is NOT in VERIFY_SCHEMA's required list, so on a
// workload-aligned run a Verifier that omits it reaches this fallback and the round
// gates on -- and the archive STORES as an elite's `geomean` -- the unweighted number,
// beside weighted scores from other candidates. That is the (59c) shape: not a default
// for a missing value but a silently different QUANTITY. The fallback is kept here
// because this selector has a second, advisory caller (the engineer's own self-report at
// the skip-verify check, whose schema requires only `speedup_geomean`, and where being
// lenient errs toward sending the candidate to the oracle). Every AUTHORITATIVE caller
// now refuses the ambiguous case by name -- see `primWeighted` in `qdAdmissionCheck`
// and the `verified` / integrate gates.
const primWeighted = (o) => {
  if (!o) return null;
  const w = o.verified_weighted != null ? o.verified_weighted
          : (o.speedup_weighted != null ? o.speedup_weighted : null);
  return Number.isFinite(w) ? w : null;
};
// A named refusal for the sites where the primary metric decides admission or is stored.
const primMetricReason = (o) => (HAS_WORKLOAD && primWeighted(o) === null
  ? 'metric:weighted_speedup_missing(this run is workload-aligned, so the primary metric is the '
    + 'time-weighted ratio-of-sums; the unweighted geomean is a different quantity and cannot gate '
    + 'against, or be stored beside, weighted scores)'
  : null);
const primSpeedup = (o) => {
  if (!o) return 0;
  const w = primWeighted(o);
  if (HAS_WORKLOAD && Number.isFinite(w)) return w;
  const g = o.verified_geomean != null ? o.verified_geomean : o.speedup_geomean;
  return Number.isFinite(g) ? g : 0;
};
// Gate fields like `correctness` / `status` are free strings in the schemas, so an agent legitimately
// answers "PASS - 15/15 draws" and a `=== 'pass'` test silently drops a genuinely verified candidate.
// Match the leading word instead: "PASS - ..." / "passed" gate open, "FAIL"/"did not pass" stay shut.
//
// Finding (62). The leading-word match alone is not enough, and it fails OPEN on
// the one gate that must never do so. Executed against realistic phrasings,
// `says('passes 10/11 cases', 'pass')` and `says('pass rate 10/11', 'pass')` both
// returned true -- a correctness FAILURE opening the correctness gate -- while
// 'partially passes' returned false. The same fact stated two ways gave opposite
// verdicts, so the rule was not tracking meaning at all. `says('verified: FAILED
// correctness', 'verified')` opened the status gate for the same reason.
// The prefix stays (it is what makes "PASS - 15/15 draws" work); what is added is
// a veto on text that contradicts its own leading word. A fraction is a
// contradiction only when it is not n/n, so "15/15" passes and "10/11" does not.
const saysContradicted = (v) => {
  const s = String(v == null ? '' : v);
  if (/\b(fail\w*|mismatch\w*|incorrect|wrong|error\w*|partial\w*|except|nan|inf)\b/i.test(s)) return true;
  const frac = /(\d+)\s*\/\s*(\d+)/g;
  for (let m = frac.exec(s); m; m = frac.exec(s)) if (Number(m[1]) !== Number(m[2])) return true;
  return false;
};
const says = (v, w) =>
  String(v == null ? '' : v).trim().toLowerCase().startsWith(w) && !saysContradicted(v);
const KERNEL_KNOWLEDGE_DIR = String(A.perf_knowledge_dir ||
  (WORKFLOW_DIR ? WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/perf_knowledge' : '')).replace(/\/+$/, '');
// Expert skills = human-authored, validated kernel recipes (perf_knowledge/expert_skills/). ADVISORY
// priors only: a matched `validated` skill is a HIGH-PRIOR author/optimize candidate the planning/author
// roles reproduce, then gate by the isolated A/B vs the oracle — it NEVER overrides measurement. Default
// OFF (opt-in: pass use_expert_skills="true"). When OFF (the default) NOTHING is injected -> byte-identical
// to a build without this feature. When invoked by the e2e layer the flag + dir are passed down.
const USE_EXPERT_SKILLS = String(A.use_expert_skills != null ? A.use_expert_skills : 'false') === 'true';
const EXPERT_SKILLS_DIR = String(A.expert_skills_dir ||
  (KERNEL_KNOWLEDGE_DIR ? KERNEL_KNOWLEDGE_DIR + '/expert_skills' : '')).replace(/\/+$/, '');
// Only planning + authoring roles consult skills; every other role gets no injection.
const EXPERT_SKILL_ROLES = new Set(['tech_lead', 'author_engineer', 'engineer', 'deep_engineer']);

// ---------------------------------------------------------------------------
// DEEP-MODE continuation + cross-backend / e2e-feedback hooks. ALL OPTIONAL.
// When none are passed (every normal/fast e2e run, and every standalone run) these are '' / the
// current defaults and are NEVER threaded into a prompt — so behavior is byte-identical to the
// pre-feature build. They are set ONLY by e2e_workflow's deep_mode head scheduler.
//   STATE_DIR        a STABLE dir for THIS (kernel,backend) ACROSS deep waves. When set the run
//                    RESUMES: director seeds the canonical from STATE_DIR/best (the cumulative-best
//                    code) and returns prior_state (cumulative + history) so re-invocation CONTINUES
//                    instead of restarting (no lost experience, no re-explored directions). The frozen
//                    oracle baseline (immutable unittest.py/meta.json) stays the reference, so speedups
//                    remain comparable to the TRUE baseline across waves. update_memory writes STATE.json
//                    + syncs best/ each round.
//   SHARED_KB        cross-backend blackboard file (read by plan+engineers, appended by update_memory).
//   GLOBAL_KB        run-global cross-KERNEL technique blackboard (deep): techniques that generalize
//                    across head ops/backends. Optional; unset (default/fast) => byte-identical prompts.
//   E2E_FEEDBACK     path to the latest end-to-end A/B result + problems from e2e_workflow (engaged?,
//                    cudagraph behavior, mem footprint, decode regression, e2e delta) — steers planning.
//   HARNESS_ADDENDUM path to an e2e-refined harness addendum (timing-weight / cudagraph-capture / hard
//                    constraint gates). The IMMUTABLE oracle is NEVER touched; this only refines what the
//                    perf bench emphasizes so the isolated target aligns with e2e.
//   MAX_NO_IMPROVE   consecutive non-improving rounds before stopping (default 2 = current behavior).
const STATE_DIR = String(A.state_dir || '').replace(/\/+$/, '');
const SHARED_KB = String(A.shared_kb || '').trim();
const GLOBAL_KB = String(A.global_kb || '').trim();   // run-global cross-KERNEL technique blackboard (deep)
const E2E_FEEDBACK = String(A.e2e_feedback || '').trim();
const HARNESS_ADDENDUM = String(A.harness_addendum || '').trim();
// P2 (deep continuation): on a RESUMED wave (STATE_DIR holds prior work) the cold re-derivation of
// Analyze + baseline Profile is largely redundant. INCREMENTAL_RESUME tells those two ADVISORY agents to
// load the prior roadmap/profile and return it with only delta updates instead of re-deriving from
// scratch, so each burst spends its budget on optimization ROUNDS, not re-analysis. Benchmark is NEVER
// incremental (it re-pins a fresh in-window baseline every wave \u2014 the matched-A/B correctness rail).
// Unset (default/fast / first deep burst) => spreading {} adds nothing => byte-identical prompts.
const INCREMENTAL = !!STATE_DIR && String(A.incremental_analyze || '') === 'true';
const RESUME_INPUT = INCREMENTAL ? { INCREMENTAL_RESUME: '1' } : {};
const MAX_NO_IMPROVE = Math.max(1, parseInt(A.max_no_improve != null ? A.max_no_improve : 2, 10));
// Conditional inputs: spreading {} adds NOTHING to a prompt (byte-identical) when a hook is unset.
const KB_INPUTS = {
  ...(SHARED_KB ? { SHARED_KB } : {}),
  ...(GLOBAL_KB ? { GLOBAL_KB } : {}),
  ...(E2E_FEEDBACK ? { E2E_FEEDBACK } : {}),
  ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
};

// ---------------------------------------------------------------------------
// Reusable JSON-schema fragments.
// ---------------------------------------------------------------------------
const perCase = {
  type: 'array',
  items: {
    type: 'object',
    properties: {
      name: { type: 'string' },
      baseline_ms: { type: 'number' },
      optimized_ms: { type: 'number' },
      speedup: { type: 'number' },
      // Workload-alignment fields (present only when a WORKLOAD_SPEC drives the harness; absent
      // on a normal unweighted run). weight = this case's baseline time SHARE in the real workload;
      // it is the coefficient of the time-weighted metric Σweight / Σ(weight/speedup). count is
      // optional/informational (regime-attributed cases have no per-call count).
      weight: { type: 'number' },
      count: { type: 'number' },
      dims: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
      dtypes: { type: 'array', items: { type: 'string' } },
      weight_source: { type: 'string' }, // trace | regime | regime_floor | prior | caller
    },
    required: ['name', 'speedup'],
  },
};
const obj = (props, required) => ({ type: 'object', properties: props, required: required || [], additionalProperties: true });

const SETUP_SCHEMA = obj({
  eval_dir: { type: 'string' }, workspace: { type: 'string' }, baseline_dir: { type: 'string' },
  kernel_name: { type: 'string' }, source_files: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
  // Frozen-baseline verdict (BOTH modes). The unittest's timing + random-value parity baseline MUST be
  // the real online kernel — the immutable baseline_src/ dir OR an importable meta.baseline_callable —
  // never kernel_src/ (the candidate's own scaffold). The director sets baseline_frozen=true after it
  // copies baseline_src/ + confirms meta.baseline_callable; the script aborts the run if neither holds.
  baseline_frozen: { type: 'boolean' }, baseline_callable: { type: 'string' },
  // DEEP-MODE resume only: populated by the director ONLY when STATE_DIR was provided AND a prior best
  // exists there. Lets a continued wave restore its cumulative speedup + insight/ledger history so it
  // does not re-explore dead directions. Absent (undefined) on a fresh run -> no behavior change.
  resumed: { type: 'boolean' },
  prior_state: obj({
    cumulative: { type: 'number' }, insights: { type: 'array', items: { type: 'string' } },
    ledger: { type: 'array', items: { type: 'object', additionalProperties: true } },
    bottleneck_now: { type: 'string' }, best_per_case: perCase,
  }, []),
  // QD warm-start candidates are copied under this run's archive and are explicitly untrusted:
  // their historical scores never seed cells/global_best until the verifier re-measures them.
  // Finding (67). sha256 over the immutable oracle (unittest.py, meta.json, reference_io.pt,
  // baseline_src/) recorded BEFORE any engineer runs. Candidates carry `source_hash` provenance;
  // the denominator and correctness reference carried none, so a write through the golden's
  // absolute symlink would silently rescore every measurement AND every archived elite.
  oracle_digest: { type: 'string' },
  qd_import_status: { type: 'string' },
  qd_import_source: { type: 'string' },
  qd_import_manifest: { type: 'object', additionalProperties: true },
  qd_import_candidates: { type: 'array', items: obj({
    cell: { type: 'string' }, elite_id: { type: 'string' }, snapshot: { type: 'string' },
    context_id: { type: 'string' }, route_id: { type: 'string' }, source_hash: { type: 'string' },
    descriptor: { type: 'object', additionalProperties: true },
    route_descriptors: { type: 'array', items: { type: 'object', additionalProperties: true } },
    historical_geomean: { type: 'number' },
    historical_robust: { type: 'object', additionalProperties: true },
    policy_pass: { type: 'boolean' }, policy_receipt: { type: 'string' },
    needs_reclassification: { type: 'boolean' },
  }, ['snapshot', 'source_hash', 'policy_pass']) },
}, ['eval_dir', 'workspace', 'kernel_name']);

const AUTHOR_SCHEMA = obj({
  authored: { type: 'boolean' }, target_language: { type: 'string' }, correctness: { type: 'string' },
  baseline_ms: { type: 'number' }, kernel_src_path: { type: 'string' }, entry_point: { type: 'string' },
  build: { type: 'boolean' }, notes: { type: 'string' },
}, ['authored', 'correctness']);

const ANALYZE_SCHEMA = obj({
  kernel_type: { type: 'string' }, kernel_file: { type: 'string' }, entry_point: { type: 'string' },
  modifiable_files: { type: 'array', items: { type: 'string' } },
  bottleneck_guess: { type: 'string' }, roadmap_summary: { type: 'string' },
  candidate_directions: { type: 'array', items: { type: 'object', additionalProperties: true } },
  // perf_knowledge resolution (REFERENCE ONLY): the operator/language this kernel maps to in the
  // AMD perf_knowledge base, plus the most relevant card paths, so engineers read focused context
  // instead of re-navigating the whole base. Empty string / [] / null when no card applies.
  kk_operator: { type: ['string', 'null'] }, kk_language: { type: ['string', 'null'] },
  kk_refs: { type: 'array', items: { type: 'string' } },
}, ['kernel_type', 'roadmap_summary']);

const BENCH_SCHEMA = obj({
  commandment_path: { type: 'string' }, correctness_cmd: { type: 'string' },
  benchmark_cmd: { type: 'string' }, profile_cmd: { type: 'string' }, parse_hint: { type: 'string' },
  baseline_per_case: { type: 'array', items: { type: 'object', additionalProperties: true } },
  baseline_geomean_ms: { type: 'number' }, num_test_cases: { type: 'number' },
  // Workload-aligned outputs: present when a WORKLOAD_SPEC drove case selection + weights.
  // baseline_weighted_total_ms = the baseline time the weights represent (Σ weight_i in time units).
  // The metric is Σ weight_i / Σ (weight_i/speedup_i). workload_aligned flags weights are real (not 1).
  workload_aligned: { type: 'boolean' },
  baseline_weighted_total_ms: { type: 'number' },
  weights_provenance: { type: 'string' }, // e.g. "trace" | "regime" | "regime_floor" | "prior" | "caller" | "mixed"
  reliable: { type: 'boolean' }, notes: { type: 'string' },
}, ['commandment_path', 'baseline_per_case', 'baseline_geomean_ms']);

const PROFILE_SCHEMA = obj({
  bottleneck: { type: 'string' }, profiler_used: { type: 'string' }, dispatch_count: { type: 'number' },
  // The accelerator detected on-box (e.g. "MI300X / gfx942 / CDNA3, 304 CU, ~5.3 TB/s"), so the
  // roofline ceiling + grid-sizing advice downstream use the real card instead of an assumed MI300X.
  device: { type: 'string' },
  key_metrics: { type: 'object', additionalProperties: true },
  top_kernels: { type: 'array', items: { type: 'object', additionalProperties: true } },
  top_opportunities: { type: 'array', items: { type: 'string' } },
  summary_path: { type: 'string' }, shift_note: { type: 'string' },
}, ['bottleneck', 'top_opportunities']);

// Finding (89). The baseline/reprofile profile branch has no SOL contract: no
// `qd_sol_card.py` call, no ceiling provenance, no gate. It nevertheless
// published `sol_gap_x: 12` and `peak_pct: 8.36` for a route whose gated card
// says 6.2x, because it hand-rolled a roofline from the vendor nameplate --
// and its own two axes used different undeclared bases (5.3 TB/s named,
// 3.5 TB/s implied), which is the shape no single correction can repair.
//
// The danger is the asymmetry: the gated card is read by the admission
// arithmetic, the roadmap profile is read by the Director and every Engineer.
// The wrong number is the one people plan against, and it never meets the gate
// that would refuse it.
//
// So this strips rather than refuses. The counters in that report are real
// measurements and were adopted; only the denominators are ungrounded, and
// dropping a candidate over them would throw away the good half. (76): a field
// with no contract behind it should not exist -- not be quietly believed.
const SOL_SHAPED_KEY = /^(sol_|peak_pct$|pct_of_peak|compute_floor|memory_floor|remaining_headroom|roofline)/;
const profileSolStrip = (rep, where) => {
  if (!rep || typeof rep !== 'object') return rep;
  const dropped = [];
  const scrub = (o, prefix) => {
    if (!o || typeof o !== 'object') return;
    for (const k of Object.keys(o)) {
      if (SOL_SHAPED_KEY.test(k)) { dropped.push(prefix + k); delete o[k]; }
    }
  };
  scrub(rep, '');
  scrub(rep.key_metrics, 'key_metrics.');
  if (dropped.length) {
    log(`  ${where}: dropped ungrounded SOL fields [${dropped.join(', ')}] -- this branch `
      + `does not call qd_sol_card.py, so it has no ceiling provenance to publish (89). `
      + `The measured counters are kept.`);
    rep.sol_note = 'SOL-shaped fields were removed: the roadmap profile has no ceiling '
      + 'provenance. Use the selected_cell_sol card for any headroom claim.';
  }
  return rep;
};

const QD_SELECTION_SCHEMA = obj({
  stop: { type: 'boolean' }, reasoning: { type: 'string' },
  selections: { type: 'array', items: obj({
    id: { type: 'string' }, parent_elite_id: { type: 'string' },
    selected_cell: { type: 'string' }, context_id: { type: 'string' },
    parent_source_hash: { type: 'string' },
  }, ['id', 'parent_elite_id', 'selected_cell', 'context_id', 'parent_source_hash']) },
}, ['stop', 'selections']);

const PLAN_SCHEMA = obj({
  stop: { type: 'boolean' }, reasoning: { type: 'string' },
  directions: {
    type: 'array',
    items: obj({
      id: { type: 'string' }, title: { type: 'string' },
      specialty: { type: 'string', enum: ['algorithm', 'memory', 'compute', 'host_runtime', 'deep_explore'] },
      focus_files: { type: 'array', items: { type: 'string' } },
      expected_speedup: { type: 'number' }, prompt: { type: 'string' },
      kk_refs: { type: 'array', items: { type: 'string' } }, // optional: perf_knowledge card paths for THIS direction (REFERENCE ONLY)
      operator: { type: 'string', enum: ['local_mutation', 'directed_transition', 'deep_mutation', 'semantic_crossover', 'parameter_tuning'] },
      parent_elite_id: { type: 'string' },
      parent_elite_ids: { type: 'array', items: { type: 'string' } },
      parent_source_hash: { type: 'string' }, selected_cell: { type: 'string' },
      context_id: { type: 'string' }, target_cell: { type: 'string' },
      target_descriptor: { type: 'object', additionalProperties: true },
      target_cases: { type: 'array', items: { type: 'string' } },
      changed_dimensions: { type: 'array', items: { type: 'string' } },
      preserved_dimensions: { type: 'array', items: { type: 'string' } },
      mutation_scale: { type: 'string', enum: ['local', 'structural'] },
      sol_gap_before: { type: 'number' }, target_regime: { type: 'string' },
      expected_effect: { type: 'string' }, required_evidence: { type: 'array', items: { type: 'string' } },
      strategy_capsule: { type: 'object', additionalProperties: true },
      // The receipt from the mandatory `qd_v2.py mutation-verdict` gate. It is
      // requested here, and required by the orchestrator, because a gate that
      // lives only in a prompt is a gate the planner can forget: findings (19)
      // through (24) were each encoded in that helper and re-proposed anyway,
      // every run, for exactly this reason. `ctas` and `residency_slots` are
      // carried so the orchestrator can re-derive `rounds` instead of taking
      // the planner's word for the arithmetic.
      residency_receipt: obj({
        allow: { type: 'boolean' },
        current: obj({ ctas: { type: 'number' }, residency_slots: { type: 'number' }, rounds: { type: 'number' } },
          ['ctas', 'residency_slots', 'rounds']),
        candidate: obj({ ctas: { type: 'number' }, residency_slots: { type: 'number' }, rounds: { type: 'number' } },
          ['ctas', 'residency_slots', 'rounds']),
        refusals: { type: 'array', items: { type: 'string' } },
      }, ['allow', 'current', 'candidate']),
      // Finding (68). The receipt from the third mandatory gate, `qd_v2.py
      // route-priority`. Same reason as `residency_receipt` above and the same
      // shape: the planner reports the two inputs it measured and the verdict
      // it drew, and the orchestrator re-derives the verdict rather than taking
      // it. `noise_floor` is carried even though the orchestrator owns that
      // table, precisely so the two copies can be compared -- a receipt whose
      // floor is not this route's floor is a receipt about a different route.
      priority_receipt: obj({
        per_context: { type: 'array', items: obj({
          context: { type: 'string' },
          remaining_headroom: { type: 'number' },
          noise_floor: { type: 'number' },
          slack_to_floor: { type: 'number' },
          verdict: { type: 'string', enum: ['open', 'marginal', 'closed'] },
        }, ['context', 'remaining_headroom', 'noise_floor', 'verdict']) },
      }, ['per_context']),
    }, ['id', 'title', 'specialty', 'prompt']),
  },
}, ['stop', 'directions']);

// Finding (44): every axis of QD_VOCAB must appear here, and appear as
// REQUIRED. `qdDescriptorValid` tests `Object.entries(QD_VOCAB).every(...)`, so
// an axis the agent is not asked for comes back `undefined`, fails `.has()`,
// and `qdRouteCells` drops the whole route on a bare `continue` -- no error, no
// failing test, and an archive that silently never fills. `rasterization`,
// `plan_binding` and the `lds_deep_single` value were each added to the
// validator and the adjacency graph without being added here, which made them
// simultaneously mandatory and unrequestable.
// Guarded by test_qd_lane_parity.AgentFacingVocabularyParityTest.
const QD_DESCRIPTOR_SCHEMA = obj({
  compute_primitive: { type: 'string', enum: ['valu', 'rocwmma', 'native_mfma'] },
  wave_schedule: { type: 'string', enum: ['independent', 'symmetric_interleave', 'symmetric_pingpong', 'asymmetric_producer_consumer'] },
  k_pipeline: { type: 'string', enum: ['direct_global', 'lds_single', 'lds_reg_prefetch', 'lds_pingpong', 'lds_deep_single', 'lds_multistage'] },
  decomposition: { type: 'string', enum: ['tile_grid', 'persistent_output', 'split_k', 'stream_k'] },
  output_path: { type: 'string', enum: ['direct_store', 'lds_staged_store', 'atomic_fixup', 'workspace_fixup'] },
  rasterization: { type: 'string', enum: ['linear', 'grouped_m', 'xcd_remapped_grouped'] },
  plan_binding: { type: 'string', enum: ['static', 'runtime_tuned'] },
  evidence: { type: 'array', items: { type: 'string' } },
}, ['compute_primitive', 'wave_schedule', 'k_pipeline', 'decomposition', 'output_path',
    'rasterization', 'plan_binding']);

const QD_RESOURCE_SCHEMA = obj({
  vgpr: { type: 'number' }, agpr: { type: 'number' }, lds_bytes: { type: 'number' },
  scratch_bytes: { type: 'number' }, occupancy: { type: 'number' },
}, []);
const QD_ROUTE_SCHEMA = obj({
  route_id: { type: 'string' },
  case_ids: { type: 'array', items: { type: 'string' } },
  predicate_evidence: { type: 'array', items: { type: 'string' } },
  kernel_symbols: { type: 'array', items: { type: 'string' } },
  descriptor: QD_DESCRIPTOR_SCHEMA,
  descriptor_evidence: { type: 'array', items: { type: 'string' } },
  resource_signature: QD_RESOURCE_SCHEMA,
  classification_status: { type: 'string' },
}, ['route_id', 'case_ids', 'descriptor']);
// Finding (69). `policy_pass` is the single most consequential boolean in this
// project -- the entire "a candidate may not link rocBLAS" constraint reduces to
// it -- and it was consumed at five sites as a bare self-report. `policy_receipts`
// existed beside it in three schemas, carried nothing but *file paths*, and was
// read by nothing: a report of `policy_pass: true` with no receipts at all passed
// every one of those five gates.
//
// The orchestrator has no filesystem and cannot open the receipt, so this is the
// same cross-agent shape as the oracle digest (67) and the two planner receipts:
// the agent copies the scanner's derived `summary` block in, and the orchestrator
// checks it against itself and against what a post-build scan must look like.
// `elf` is the field with teeth. A forbidden library arrives as a DT_NEEDED entry
// or an imported symbol, both of which are visible only in the built binary, so
// `elf: 0` on a post-build receipt means the one scan that could have caught the
// cheat did not look at the thing the cheat lives in.
const POLICY_SUMMARY_SCHEMA = obj({
  schema: { type: 'string' }, passed: { type: 'boolean' },
  findings: { type: 'number' }, advisory: { type: 'number' },
  inspected: { type: 'number' }, elf: { type: 'number' }, unreadable: { type: 'number' },
}, ['passed', 'findings', 'inspected', 'elf']);
// (87). The exit code is the field with teeth, and it has three meanings, not
// two: 0 in lockstep, 1 drifted, 2 nothing was checked. `pairs` and `drifted`
// are carried because they let the orchestrator refuse a receipt that could not
// have come from running the tool -- exit 0 with zero pairs, or exit 0 with
// drift. See `qdTwinReject`.
const HIP_TWIN_SCHEMA = obj({
  exit_code: { type: 'number' }, pairs: { type: 'number' }, drifted: { type: 'number' },
  checked: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
}, ['exit_code', 'pairs']);
const QD_CASE_SAMPLES_SCHEMA = {
  type: 'array', items: obj({
    name: { type: 'string' }, samples: { type: 'array', items: { type: 'number' } },
    median: { type: 'number' }, mad: { type: 'number' }, lower: { type: 'number' }, upper: { type: 'number' },
  }, ['name', 'samples', 'median', 'mad', 'lower', 'upper']),
};

const ENG_SCHEMA = obj({
  engineer_id: { type: 'string' }, specialty: { type: 'string' }, task: { type: 'string' },
  strategy: { type: 'string' }, speedup_geomean: { type: 'number' }, speedup_arithmetic: { type: 'number' },
  // Time-weighted ratio-of-sums vs the TRUE baseline (PRIMARY metric when workload_aligned).
  // = Σ weight_i / Σ (weight_i / speedup_i). Omitted on unweighted runs.
  speedup_weighted: { type: 'number' },
  per_case: perCase, status: { type: 'string' }, patch_file: { type: 'string' },
  strategies_tried: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
  descriptor: QD_DESCRIPTOR_SCHEMA, descriptor_evidence: { type: 'array', items: { type: 'string' } },
  route_descriptors: { type: 'array', items: QD_ROUTE_SCHEMA }, source_hash: { type: 'string' },
}, ['status', 'speedup_geomean']);

const VERIFY_SCHEMA = obj({
  status: { type: 'string' }, correctness: { type: 'string' },
  verified_geomean: { type: 'number' }, verified_arithmetic: { type: 'number' },
  verified_weighted: { type: 'number' }, // time-weighted ratio-of-sums (PRIMARY when workload_aligned)
  per_case: perCase, variance_note: { type: 'string' }, notes: { type: 'string' },
  graph_safe: { type: 'string' }, descriptor: QD_DESCRIPTOR_SCHEMA,
  route_descriptors: { type: 'array', items: QD_ROUTE_SCHEMA },
  source_hash: { type: 'string' }, seed_source_hash: { type: 'string' },
  // Finding (67): the setup digest, recomputed independently at verify time. Two agents at two
  // different points in the run reporting the same oracle is materially stronger evidence than
  // one agent asserting it once -- and it is the only evidence available to a script with no FS.
  oracle_digest: { type: 'string' },
  measurement_samples: { type: 'array', items: { type: 'number' } },
  case_measurement_samples: QD_CASE_SAMPLES_SCHEMA,
  robust_median: { type: 'number' }, robust_mad: { type: 'number' },
  robust_lower: { type: 'number' }, robust_upper: { type: 'number' },
  min_case_speedup: { type: 'number' }, descriptor_evidence: { type: 'array', items: { type: 'string' } },
  policy_pass: { type: 'boolean' },
  policy_receipts: { type: 'object', additionalProperties: true },
  // (69): the summary of `$VERIFY_DIR/policy_postbuild.json`, the scan that sees the ELFs.
  policy_postbuild: POLICY_SUMMARY_SCHEMA,
  // (87): `hip_twin_sync.py` over the measured tree -- did ninja compile what was edited?
  hip_twin_sync: HIP_TWIN_SCHEMA,
}, ['status', 'verified_geomean', 'policy_pass']);

const INTEGRATE_SCHEMA = obj({
  attempted: { type: 'boolean' },
  combos_tried: { type: 'array', items: { type: 'object', additionalProperties: true } },
  best: { type: 'object', additionalProperties: true },
  improved_over_best_individual: { type: 'boolean' },
  policy_pass: { type: 'boolean' }, policy_receipts: { type: 'object', additionalProperties: true },
  policy_postbuild: POLICY_SUMMARY_SCHEMA,   // (69)
  hip_twin_sync: HIP_TWIN_SCHEMA,            // (87)
  conclusion: { type: 'string' }, notes: { type: 'string' },
}, ['attempted', 'conclusion', 'policy_pass']);

const MEMORY_SCHEMA = obj({
  insights: { type: 'array', items: { type: 'string' } },
  ledger: { type: 'array', items: { type: 'object', additionalProperties: true } },
  bottleneck_now: { type: 'string' }, suggest_next: { type: 'string' },
}, ['insights']);

const QD_IMPORT_VERIFY_SCHEMA = obj({
  imported_elite_id: { type: 'string' }, imported_source_hash: { type: 'string' },
  source_cell: { type: 'string' },
  status: { type: 'string' }, correctness: { type: 'string' },
  policy_pass: { type: 'boolean' }, policy_receipts: { type: 'object', additionalProperties: true },
  policy_postbuild: POLICY_SUMMARY_SCHEMA,   // (69)
  verified_geomean: { type: 'number' }, verified_arithmetic: { type: 'number' },
  verified_weighted: { type: 'number' }, per_case: perCase,
  measurement_samples: { type: 'array', items: { type: 'number' } },
  robust_median: { type: 'number' }, robust_mad: { type: 'number' },
  robust_lower: { type: 'number' }, robust_upper: { type: 'number' },
  min_case_speedup: { type: 'number' }, descriptor: QD_DESCRIPTOR_SCHEMA,
  route_descriptors: { type: 'array', items: QD_ROUTE_SCHEMA },
  source_hash: { type: 'string' }, seed_source_hash: { type: 'string' },
  case_measurement_samples: QD_CASE_SAMPLES_SCHEMA,
  descriptor_evidence: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
  hip_twin_sync: HIP_TWIN_SCHEMA,   // (87)
}, ['imported_elite_id', 'source_cell', 'status', 'correctness', 'policy_pass', 'verified_geomean']);

// Finding (86). The archive models mechanisms as EDGES -- "this move, on this
// route, had this measured effect" -- and an edge-only model has no way to say
// "the arch guard in this kernel rejects gfx942 and must be widened before
// anything builds". That fact is true of every descriptor and every one of the
// 11 routes, so there is no `capsule_key` a later planner would think to read
// it under, and it is established at preflight, which never reaches the
// admission block where capsules are written. Every fresh run therefore
// rediscovered it: this run's entire `baseline.patch` was that one fix, again.
//
// So: a SECOND, flat namespace keyed by `qd_arch`, written from bootstrap, read
// on the same planner path as the capsule ledger. Deliberately NOT a widened
// `capsule_key` -- (71) is on the record about what happens when one identity is
// made to serve as two slots, and a key that accepts both edges and
// preconditions makes "failed twice on this route" unreadable.
const PRECONDITION_KINDS = ['build_guard', 'toolchain', 'runtime_env', 'attainable_ceiling', 'harness'];
const PRECONDITION_SCHEMA = obj({
  id: { type: 'string' }, kind: { type: 'string', enum: PRECONDITION_KINDS },
  statement: { type: 'string' }, evidence: { type: 'string' },
  established_by: { type: 'string' },
}, ['id', 'kind', 'statement', 'evidence']);

const QD_SEED_SCHEMA = obj({
  status: { type: 'string' }, policy_pass: { type: 'boolean' }, source_hash: { type: 'string' },
  // (69): the seed is the one report whose policy claim is never re-checked by a
  // later stage -- every elite in the archive descends from it, so an unbacked
  // pass here is inherited by everything and questioned by nothing.
  policy_postbuild: POLICY_SUMMARY_SCHEMA,
  // (86). Optional and NOT in the required list on purpose: a run on an arch
  // that needed nothing must be able to say so by omission, and making this
  // mandatory would manufacture a record for every bootstrap.
  preconditions: { type: 'array', items: PRECONDITION_SCHEMA },
  route_descriptors: { type: 'array', items: QD_ROUTE_SCHEMA }, notes: { type: 'string' },
}, ['status', 'policy_pass', 'source_hash', 'route_descriptors']);
// (70). The five provenance fields `qd_sol_card.py` emits about the bandwidth
// ceiling it resolved, carried verbatim. Before this they were computed by the
// tool, printed to the agent, and dropped here -- so at the orchestrator a
// clamped guess and a measured interpolation were the same number. They are not
// the same number: below the measured range the clamp OVERSTATES the achievable
// rate, which understates `memory_floor_ms`, which overstates `sol_gap` and
// `remaining_headroom` -- the quantity route priority is computed from. The
// error points toward "this route has headroom", i.e. it manufactures work.
const SOL_CEILING_SCHEMA = obj({
  basis: { type: 'string' },          // 'scalar' | 'footprint_table'
  confidence: { type: 'string' },     // 'measured_interpolated' | 'measured_scalar' | 'unmeasured' | 'low'
  extrapolated: { type: 'boolean' },
  footprint_bytes: { type: 'number' },
  bracket: { type: 'array', items: { type: 'number' } },
}, ['basis', 'confidence', 'extrapolated']);
// (89) item 2. The bandwidth half of a SOL card has had a provenance block
// since (70); the compute half had nothing, and provenance would not have been
// enough for it anyway. `rocminfo` reports 1307 TFLOP/s bf16 on this box with
// impeccable provenance and nothing has ever come near it -- scored against it
// every route in the suite sits 3.2x-6.4x from SOL and so does rocBLAS at
// 3.5x-7.4x, which is not a weak ranking signal but the absence of one. What
// separates a ceiling from a nameplate is whether anything was observed to
// reach it, so the card must name the achiever. `qd_sol_card.py` emits these
// three fields; they are transcribed here the same way the bandwidth block is.
const SOL_COMPUTE_CEILING_SCHEMA = obj({
  witnessed: { type: 'boolean' },
  attainment: { type: 'number' },   // achieved/peak of the witness, <= 1
  witness: { type: 'string' },      // what achieved it, named
}, ['witnessed']);
const QD_SOL_CASE_SCHEMA = obj({
  name: { type: 'string' }, measured_ms: { type: 'number' },
  compute_floor_ms: { type: 'number' }, memory_floor_ms: { type: 'number' }, sol_ms: { type: 'number' },
  sol_gap: { type: 'number' }, remaining_headroom: { type: 'number' }, roof: { type: 'string' },
  profile_regime: { type: 'string' }, confidence: { type: 'string' },
  ceiling: SOL_CEILING_SCHEMA, compute_ceiling: SOL_COMPUTE_CEILING_SCHEMA,
  evidence: { type: 'array', items: { type: 'string' } },
}, ['name', 'measured_ms', 'compute_floor_ms', 'memory_floor_ms', 'sol_ms', 'sol_gap', 'remaining_headroom', 'roof', 'profile_regime', 'confidence', 'ceiling', 'compute_ceiling']);
const QD_SOL_CARD_SCHEMA = obj({
  selected_cell: { type: 'string' }, context_id: { type: 'string' },
  parent_elite_id: { type: 'string' }, parent_source_hash: { type: 'string' },
  calibration_version: { type: 'string' }, cases: { type: 'array', items: QD_SOL_CASE_SCHEMA },
}, ['selected_cell', 'context_id', 'parent_elite_id', 'parent_source_hash', 'calibration_version', 'cases']);

const QD_ARCHIVE_SCHEMA = obj({
  persisted_elite_ids: { type: 'array', items: { type: 'string' } },
  rejected_elite_ids: { type: 'array', items: { type: 'string' } },
  manifest_path: { type: 'string' }, global_best_path: { type: 'string' },
  notes: { type: 'string' },
  // (96). `verification` is not the persister's opinion of its own work: it is
  // `qd_persist_manifest.py` re-reading the file it just wrote and counting
  // what is in it. The lane holds the same archive in memory, so these numbers
  // are checkable rather than reportable -- see `qdVerifyPersisted`. The agent
  // that runs the script must return this block byte-for-byte.
  verification: obj({
    readable: { type: 'boolean' }, generation: { type: 'number' },
    cells: { type: 'number' }, cell_keys_sha256: { type: 'string' },
    elite_ids_sha256: { type: 'string' }, challengers: { type: 'number' },
    capsules: { type: 'number' }, lineage: { type: 'number' },
    recent_transitions: { type: 'number' }, bytes: { type: 'number' },
    error: { type: 'string' },
  }, []),
  artifacts: { type: 'object' },
  failures: { type: 'array', items: { type: 'object' } },
}, ['persisted_elite_ids']);

const COMMIT_SCHEMA = obj({
  committed: { type: 'boolean' }, current_best_diff: { type: 'string' }, note: { type: 'string' },
}, ['committed']);

const REPORT_SCHEMA = obj({
  final_speedup_geomean: { type: 'number' }, final_speedup_arithmetic: { type: 'number' },
  final_speedup_weighted: { type: 'number' }, // time-weighted ratio-of-sums (PRIMARY when workload_aligned)
  rounds: { type: 'number' }, budget_used: { type: 'number' },
  report_path: { type: 'string' }, final_patch: { type: 'string' }, per_case: perCase,
}, ['final_speedup_geomean', 'report_path', 'final_patch']);

const VALIDATE_SCHEMA = obj({
  kernel_name: { type: 'string' },
  director_verified_speedup_geomean: { type: 'number' },
  director_verified_speedup_arithmetic: { type: 'number' },
  director_verified_speedup_weighted: { type: 'number' }, // PRIMARY when workload_aligned
  tech_lead_reported_speedup_geomean: { type: 'number' },
  validation_status: { type: 'string' }, correctness: { type: 'string' },
  per_case: perCase, applied_to_original: { type: 'string' },
  arbitration_note: { type: 'string' }, final_patch: { type: 'string' },
  // Finding (65). director.md: "`timing_basis` is REQUIRED in `director_validation.json`, and any
  // campaign summary that quotes the speedup must carry it — an unlabelled number is read as a clean
  // device-time win." It was in neither the schema nor any consumer, so the one number a human reads
  // was exactly that unlabelled number. Declared here rather than *required* on purpose: a hard schema
  // failure at the last phase would throw away the whole run's GPU work over a missing label, and
  // director.md already fixes the safe default itself — "Absence is not evidence of priming."
  timing_basis: { type: 'string' },
  oracle_digest: { type: 'string' },   // (67): the arbiter's own recomputation of the pinned oracle
}, ['director_verified_speedup_geomean', 'validation_status']);

// ---------------------------------------------------------------------------
// Prompt helpers. Every agent reads its role file from WORKFLOW_DIR and the
// relevant knowledge files itself; the script only passes paths + JSON inputs.
// ---------------------------------------------------------------------------
const cfg = (o) => Object.entries(o).map(([k, v]) =>
  `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');

// --- Hung-agent guard ------------------------------------------------------
// An agent LLM call that HANGS (no response, no terminal error) blocks a
// parallel()/pipeline() round-barrier forever (observed: engineer agents frozen
// mid-turn wedged the whole optimize round for >30min). The harness resolves
// terminal API errors to null but NOT an indefinite hang. So bound every agent()
// call: if it has not returned after AGENT_TIMEOUT_MS, resolve it to null (which
// every .filter(Boolean)/null-check downstream already tolerates) and let the
// round proceed. VERY generous default (60min): a true hang never returns, so this only fires on a
// hang, NEVER on a legitimately-long agent. Inner agents include benchmark/profile/verify that build
// (hipcc/ninja) and run benches — minutes, well under 60min — plus the LLM-heavy optimize engineers
// (the ones observed hanging). A too-short bound would kill legit long agents (e.g. a slow rocprof or
// build), so keep it large. Cache keys (prompt, opts) are unchanged so resume still works. Falls back
// to raw agent() if setTimeout is unavailable. args.agent_timeout_ms=0 disables.
// API-FAULT TOLERANCE: a transient API failure (gateway 4xx/5xx, rate-limit, dropped connection, the
// model API going down mid-run) must NOT crash the whole workflow. agentT retries the call up to
// AGENT_RETRIES times on a thrown API/agent error, then resolves to null (every .filter(Boolean)/
// null-check downstream — incl. the Director validate + final report — already degrades on null rather
// than exiting). A timeout (hang) resolves null immediately and is NOT retried (a real hang would just
// burn another full timeout window). args.agent_retries tunes the count. If the failure is PERSISTENT
// (e.g. an auth/header requirement the client doesn't send), retries are exhausted then the run
// degrades — re-run with Workflow({resumeFromRunId}) once the client/API is fixed; cached agent results
// make resume cheap.
const AGENT_TIMEOUT_MS = parseInt(A.agent_timeout_ms != null ? A.agent_timeout_ms : 3600000, 10);
const AGENT_RETRIES = Math.max(1, parseInt(A.agent_retries != null ? A.agent_retries : 4, 10));
async function agentT(p, o) {
  const label = (o && o.label) ? o.label : 'agent';
  for (let attempt = 1; attempt <= AGENT_RETRIES; attempt++) {
    try {
      if (typeof setTimeout !== 'function' || !(AGENT_TIMEOUT_MS > 0)) return await agent(p, o);
      let to;
      const guard = new Promise((resolve) => {
        to = setTimeout(() => {
          log(`  [hung-agent guard] ${label} exceeded ${Math.round(AGENT_TIMEOUT_MS / 60000)}min with no return — resolving null so the round proceeds.`);
          resolve(null);
        }, AGENT_TIMEOUT_MS);
      });
      // A timeout resolves null (returned as-is, no retry). An API/agent error rejects -> caught below.
      return await Promise.race([
        agent(p, o).then((r) => { clearTimeout(to); return r; }, (e) => { clearTimeout(to); throw e; }),
        guard,
      ]);
    } catch (e) {
      const msg = String(e && e.message ? e.message : e).slice(0, 200);
      if (attempt < AGENT_RETRIES) {
        log(`  [api-fault guard] ${label} attempt ${attempt}/${AGENT_RETRIES} hit an API/agent error (${msg}) — retrying so a transient outage doesn't kill the run.`);
        continue;
      }
      log(`  [api-fault guard] ${label} still failing after ${AGENT_RETRIES} attempts (${msg}) — resolving null so the workflow degrades gracefully instead of exiting.`);
      return null;
    }
  }
  return null;
}

// Expert-skills injection. PURELY ADDITIVE: '' when OFF or the role is not a skills consumer, so both
// call sites (roleAgent, and the inline Optimize prompt) are byte-identical to the pre-feature build in
// those cases. When ON, appends an advisory pointer telling the agent to Read the fragment + query the
// skills index (scripts have no fs access).
function expertSkillsBlock(role) {
  if (!USE_EXPERT_SKILLS || !EXPERT_SKILL_ROLES.has(role) || !EXPERT_SKILLS_DIR) return '';
  return `\n\n## Expert skills (ADVISORY — opt-in, enabled this run)\n` +
    `Also Read ${WORKFLOW_DIR}/roles/_fragments/expert_skills.md and follow it: query ` +
    `${EXPERT_SKILLS_DIR}/index.yaml for skills whose \`match\` fits this op (operator/dtype/regime, and ` +
    `from_backend->to_backend for migration skills) and whose validation_status is \`validated\`, and ` +
    `treat each as a HIGH-PRIOR candidate to reproduce — advisory only, never overriding your isolated ` +
    `A/B vs the oracle, never reducing a result below the measured baseline.`;
}

function roleAgent(role, phase, intro, inputs) {
  const base = `You are the ${role}. PHASE=${phase}.
First Read ${WORKFLOW_DIR}/roles/${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${WORKFLOW_DIR}/knowledge/.
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}

## Inputs
${cfg(inputs)}

Return ONLY the structured JSON the role file specifies (a StructuredOutput tool is forced).`;
  return base + expertSkillsBlock(role);
}

// ===========================================================================
// PHASE: Setup
// ===========================================================================
phase('Setup');
const setup = await agentT(
  roleAgent('director', 'setup', 'Build the isolated evaluation environment.', {
    KERNEL_PATH_ORIG, TASK_DIR: KERNEL_PATH_ORIG,
    EXP_ROOT, EVAL_DIR_OVERRIDE, KERNEL_NAME_HINT, TASK, SKILL_DIR: WORKFLOW_DIR,
    MODE, TARGET_LANGUAGE, OP_SPEC,
    ...(STATE_DIR ? { STATE_DIR } : {}),
    ...(QD_ENABLED && QD_STATE_DIR ? { QD_STATE_DIR, QD_CLASSIFIER_VERSION, QD_RECLASSIFY } : {}),
  }),
  { phase: 'Setup', label: 'director:setup', schema: SETUP_SCHEMA });
if (!setup || !setup.eval_dir) throw new Error('Setup failed: director did not return an eval_dir');
const EVAL_DIR = setup.eval_dir;
const CANONICAL = setup.workspace;       // canonical current-best workspace (advances each round)
const KERNEL_NAME = setup.kernel_name;
const COMMANDMENT = `${EVAL_DIR}/COMMANDMENT.md`;
log(`Setup done. EVAL_DIR=${EVAL_DIR}`);

// ---------------------------------------------------------------------------
// Enforce a FROZEN REAL-ONLINE BASELINE in BOTH modes (author AND same-language
// optimize). The immutable unittest times + parity-checks the candidate against
// baseline_src/ / meta.baseline_callable (the live online kernel); if neither
// exists it would silently fall back to timing kernel_src/ against itself — the
// "optimized-HIP vs naive-HIP = fake 15.7×" bug this harness exists to prevent.
// The script has no FS access, so we trust the director's structured verdict
// (it copied baseline_src/ + confirmed the callable). Missing -> abort/re-extract.
// ---------------------------------------------------------------------------
const hasBaseline = setup.baseline_frozen === true ||
  (typeof setup.baseline_callable === 'string' && setup.baseline_callable.trim().length > 0);
if (!hasBaseline) {
  const reason = `no frozen baseline (baseline_src/ or meta.baseline_callable) for ${KERNEL_NAME} — ` +
    `re-extract; refusing to time the candidate against kernel_src/ (fake-win risk)`;
  log(`Setup ABORT: ${reason}`);
  return {
    mode: MODE, authored: false, target_language: TARGET_LANGUAGE,
    eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
    final_geomean: 0, final_patch: '', validation_status: 'no_baseline', reason,
  };
}

// ---------------------------------------------------------------------------
// Finding (67). The oracle is the denominator of every speedup in the run AND the
// correctness reference for every candidate AND, because the archive persists, the
// yardstick every FUTURE run's warm-started elites were scored on. Four role prompts
// say "NEVER modify unittest.py / meta.json / reference_io.pt", director.md asks the
// Director to check `reference_io.pt`'s sha256 at validate time -- and nothing in the
// lane had ever looked. It is not a hypothetical: the golden is shipped into each
// engineer workspace as an ABSOLUTE symlink (the tars deliberately do not dereference
// it), so a write to that path inside a sandboxed workspace writes through to the one
// shared original. One engineer regenerating a reference silently rescores the entire
// run and every elite already in the archive, with no error anywhere.
//
// The script has no filesystem, so the check is cross-agent rather than direct: setup
// pins the digest before any engineer runs, and each verify recomputes it. That is not
// proof, but two independent agents at two points in the run is the same standard the
// seed-hash provenance guard already holds candidates to, and it is strictly more than
// the zero checks the denominator had.
// ---------------------------------------------------------------------------
// (81). Digests that are well-formed, stable, and cover nothing.
//
// The command both roles used to prescribe named four paths from the generic task
// layout. On a task lacking all four, `find` matched nothing, `xargs` invoked
// `sha256sum` with no arguments, sha256sum read empty STDIN, and the pipeline
// returned a constant independent of every byte of the oracle. Pinning it and
// comparing it to itself would have reported the oracle immutable without ever
// reading it -- a green light wired to nothing. The roles now fail closed on an
// empty file set; this is the second line of defence, because the role file is a
// prompt and a prompt is a request, not a guarantee.
const ORACLE_DEGENERATE = new Set([
  // sha256("")  -- xargs with no args, sha256sum reading an empty stdin
  'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
  // sha256 of the single line sha256sum prints for that empty stdin: the exact
  // value the old prescribed pipeline returned on this task.
  'abcfa6a9d4df344d1781bc2560b5e4cdcae08b39ed303063535e7e1e926a304a',
]);
const oracleDigestUsable = (v) =>
  typeof v === 'string' && /^[0-9a-f]{64}$/.test(v.trim()) && !ORACLE_DEGENERATE.has(v.trim());
const ORACLE_DIGEST = oracleDigestUsable(setup.oracle_digest) ? setup.oracle_digest.trim() : null;
if (typeof setup.oracle_digest === 'string' && setup.oracle_digest.trim() && !ORACLE_DIGEST) {
  // Named, not silently downgraded: "setup returned nothing" and "setup returned
  // the hash of nothing" are different failures and only one of them is a bug in
  // the digest command.
  log(`  [oracle] REFUSED a digest that covers no bytes (${setup.oracle_digest.trim().slice(0, 16)}…). ` +
      'This is the empty-file-set constant, not a measurement of the oracle: the digest command ' +
      'matched zero files. Treating the run as UNPINNED rather than certifying an oracle nothing read.');
}
if (ORACLE_DIGEST) {
  log(`Oracle pinned: ${ORACLE_DIGEST.slice(0, 16)}… (every verify recomputes it)`);
} else {
  // Not fatal, and deliberately not silent. Nothing measured after this point can be
  // shown to have used the oracle setup built, so the run's verdict cannot be `verified`.
  log('  [oracle] UNPINNED: setup returned no oracle_digest, so no measurement in this run can be ' +
      'shown to share a denominator with any other. Results are reported but cannot win a bake-off.');
}
// Refuses by name (60), and distinguishes "drifted" from "never checked" from "matched".
const oracleDrift = (o) => {
  if (!ORACLE_DIGEST) return null;           // already named once, above; not re-litigated per candidate
  const got = o && typeof o.oracle_digest === 'string' ? o.oracle_digest.trim() : '';
  if (!got) {
    return 'oracle:digest_missing(this report carries no oracle digest, so it cannot be shown to have '
      + 'measured against the oracle setup pinned; an unpinnable denominator is not a slow kernel)';
  }
  // (81): name the empty-file-set constant as itself. Falling through to the
  // drift branch would report "the oracle changed during the run" -- a false and
  // maximally alarming diagnosis -- when what actually happened is that this
  // agent's digest command matched no files.
  if (ORACLE_DEGENERATE.has(got)) {
    return 'oracle:digest_empty_fileset(this report\'s digest is the hash of an empty file set, not a '
      + 'measurement of the oracle; the digest command matched zero files. The oracle is not implicated)';
  }
  if (got !== ORACLE_DIGEST) {
    // Not a candidate-level refusal. If the oracle changed mid-run then every measurement
    // ALREADY taken was against a different denominator, including ones already admitted to
    // an archive that outlives this run. There is no subset of the results still worth
    // keeping, so this fails the run closed rather than quietly dropping one candidate.
    throw new Error(
      `oracle:digest_drift — the immutable oracle changed during the run (pinned ${ORACLE_DIGEST}, `
      + `now ${got}). Every speedup and every correctness verdict in this run, and every elite `
      + `already admitted to the archive, was scored against a denominator that no longer exists. `
      + `Refusing to continue or to persist. Re-extract the task and start a fresh run.`);
  }
  return null;
};

// ===========================================================================
// PHASE: Author (mode=author only) — write a fresh from-scratch impl as the
// optimize loop's CODE SEED. On success, HEAD of CANONICAL becomes that seed
// (what the optimize loop diffs its edits against) and the rest of the pipeline
// (Analyze/Benchmark/Profile/optimize loop) runs UNCHANGED on it. The SPEEDUP
// denominator is NEVER the seed — it is the frozen REAL ONLINE kernel in
// baseline_src/ (meta.baseline_callable), regardless of TARGET_LANGUAGE.
// On failure (no correct seed), abort early with a structured result so the
// e2e caller drops this language.
// ===========================================================================
if (MODE === 'author') {
  phase('Author');
  const authored = await agentT(
    roleAgent('author_engineer', 'author', 'Write the simplest correct baseline in the target language.', {
      TARGET_LANGUAGE, OP_SPEC, WORKSPACE: CANONICAL, TASK_DIR: KERNEL_PATH_ORIG,
      GPU_ID: GPU_POOL, ...GPU_LOCK_INPUT, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, KERNEL_KNOWLEDGE_DIR,
    }),
    { phase: 'Author', label: `author:${TARGET_LANGUAGE}`, schema: AUTHOR_SCHEMA });
  if (!authored || !authored.authored || !says(authored.correctness, 'pass')) {
    log(`Author mode FAILED for ${TARGET_LANGUAGE}: ${authored ? authored.notes || authored.correctness : 'no result'}. Aborting (no seed to optimize).`);
    return {
      mode: 'author', authored: false, target_language: TARGET_LANGUAGE,
      eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
      final_geomean: 0, final_patch: '', validation_status: 'author_failed',
      reason: authored ? authored.notes || 'author produced no correct baseline' : 'author returned nothing',
    };
  }
  log(`Author mode: ${TARGET_LANGUAGE} seed written (correct, seed ${authored.baseline_ms || '?'} ms; denominator = frozen online kernel). Optimizing it now.`);
}

// ===========================================================================
// PHASE: Analyze + Roadmap (TechLead)
// ===========================================================================
phase('Analyze');
const analysis = await agentT(
  roleAgent('tech_lead', 'analyze', 'Analyze the kernel and write the roadmap.', {
    WORKSPACE: CANONICAL, EVAL_DIR, TASK, SKILL_DIR: WORKFLOW_DIR,
    KERNEL_KNOWLEDGE_DIR,
    ...RESUME_INPUT,
  }),
  { phase: 'Analyze', label: 'tech_lead:analyze', schema: ANALYZE_SCHEMA });
log(`Analyze done. kernel_type=${analysis ? analysis.kernel_type : '?'}`);

// perf_knowledge pointers resolved by the TechLead in analyze (REFERENCE ONLY; threaded to the
// planner + engineers so they read focused op/language cards instead of the whole base). Empty when
// no operator card applies (e.g. point-cloud HIP ops) or KERNEL_KNOWLEDGE_DIR is unset → no change.
const KK_OPERATOR = (analysis && analysis.kk_operator) || '';
const KK_LANGUAGE = (analysis && analysis.kk_language) || '';
const KK_REFS = (analysis && Array.isArray(analysis.kk_refs)) ? analysis.kk_refs : [];

// ===========================================================================
// PHASE: Benchmark setup (Benchmark Engineer)
// ===========================================================================
phase('Benchmark');
const bench = await agentT(
  roleAgent('benchmark_engineer', 'setup', 'Build the COMMANDMENT and record a reliable baseline.', {
    WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL,
    ANALYSIS: analysis,
    ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
    ...(WORKLOAD_SPEC_PATH ? { WORKLOAD_SPEC_PATH } : {}),
    ...(WORKLOAD_SPEC ? { WORKLOAD_SPEC } : {}),
  }),
  { phase: 'Benchmark', label: 'benchmark_engineer', schema: BENCH_SCHEMA });
if (!bench || !bench.baseline_per_case) throw new Error('Benchmark setup failed: no baseline recorded');
const BASELINE_PER_CASE = bench.baseline_per_case;
const BASELINE_GEOMEAN_MS = bench.baseline_geomean_ms;
log(`Benchmark done. ${bench.num_test_cases || BASELINE_PER_CASE.length} cases, baseline geomean ${BASELINE_GEOMEAN_MS} ms, reliable=${bench.reliable}`);

// ===========================================================================
// PHASE: Baseline profile (Profile Engineer)
// ===========================================================================
phase('Profile');
let profileSummary = await agentT(
  roleAgent('profile_engineer', 'baseline', 'Profile the baseline and classify the bottleneck.', {
    WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL, ROUND: 0,
    COMMANDMENT,
    ...RESUME_INPUT,
  }),
  { phase: 'Profile', label: 'profile_engineer:baseline', schema: PROFILE_SCHEMA });
profileSummary = profileSolStrip(profileSummary, 'baseline profile');
log(`Baseline bottleneck: ${profileSummary ? profileSummary.bottleneck : '?'} (dispatch_count=${profileSummary ? profileSummary.dispatch_count : '?'})`);

// ===========================================================================
// PHASE: Optimization loop (budget-controlled)
// ===========================================================================
let dispatched = 0;          // counts ONLY optimization-direction engineers (the budget)
let round = 0;
let cumulative = 1.0;        // best verified geomean speedup vs the TRUE baseline
let bestSeen = 0;            // best verified geomean of any candidate, committed or not
let noImprove = 0;
let bestPerCase = BASELINE_PER_CASE;
let finalWinner = null;      // {geomean, arithmetic, per_case, patch, source}
const history = { insights: [], ledger: [], rounds: [], bottleneck_now: profileSummary ? profileSummary.bottleneck : 'unknown', suggest_next: '' };

// QD v2 describes one observed workload context + one AMD mechanism tuple. It is a sparse
// graph: knobs/resources are metadata, never extra axes, and an unsupported tuple has no cell.
const QD_CONTEXT_IDS = new Set((BASELINE_PER_CASE || []).map(c => c && (c.name || c.test_case_id)).filter(Boolean));
// Findings (34c)/(122). This used to be `.find(...)` over the three field
// names: a PRECEDENCE order, which quietly answers the question "which of these
// numbers is the authoritative baseline" whenever more than one is present. The
// three names do not mean the same thing. `baseline_ms` in a bench frame is the
// frozen oracle's latency; `baseline_ms` in the QD archive's elite rows is the
// PARENT's own latency; on a seed round those coincide and on every later round
// they do not. A ranking picks one and reports nothing, and the reader of the
// number downstream cannot tell which frame they got -- which is exactly the
// near-miss recorded in 34c, where a suite geomean was computed against the
// parent while being described as against the oracle.
//
// Disagreement is therefore refused, not ranked. Agreement to within 0.1%
// relative is treated as the same number reported twice, which is the ordinary
// case: harnesses commonly emit `latency_ms` and `execution_time_ms` for one
// measurement. Anything wider is two different measurements wearing one name,
// and the run stops rather than choosing.
// The other half of (34c). Refusing a contradiction only helps where two
// numbers are present to contradict each other; a row carrying ONE number
// under an ambiguous name is silently unambiguous to the code and ambiguous
// to every reader. So each elite says, on the row itself, which denominator
// its statistics were computed against. It is `oracle` at all three
// construction sites because the lane recomputes `geomean`/`robust`/
// `min_case`/`suite_robust` from QD_BASELINE_MS and never from the parent --
// deliberately, since a chain of parent-relative gains multiplies its own
// measurement error and cannot be compared across cells. `per_case` is the
// verifier's verbatim rows and may say something else; the label describes
// the lane's arithmetic, which is what admission actually sorts on.
const QD_ROBUST_BASELINE_FRAME = 'oracle';
const QD_BASELINE_FIELDS = ['latency_ms', 'baseline_ms', 'execution_time_ms'];
const qdAuthoritativeBaseline = (c, name) => {
  const offered = QD_BASELINE_FIELDS
    .map(f => [f, c && c[f]])
    .filter(([, v]) => Number.isFinite(v) && v > 0);
  if (!offered.length) return undefined;
  const [, first] = offered[0];
  const disagreeing = offered.filter(([, v]) => Math.abs(v - first) > 1e-3 * first);
  if (disagreeing.length) {
    throw new Error(
      `QD bootstrap failed closed: case ${name} offers more than one baseline latency and they `
      + `disagree -- ${offered.map(([f, v]) => `${f}=${v}`).join(', ')}. These field names do not `
      + `mean the same thing (a bench frame's baseline_ms is the frozen oracle; an archive elite's `
      + `baseline_ms is the parent), so the lane will not choose between them. Emit exactly one, or `
      + `name the frames apart (oracle_baseline_ms / parent_baseline_ms).`);
  }
  return first;
};
const QD_BASELINE_MS = new Map((BASELINE_PER_CASE || []).map(c => {
  const name = c && (c.name || c.test_case_id);
  return [name, name ? qdAuthoritativeBaseline(c, name) : undefined];
}).filter(([name, latency]) => name && Number.isFinite(latency) && latency > 0));
// Per-case repeats, when the benchmark engineer recorded them. Optional by
// design: an older bench result, or a harness that reports one number per case,
// must still bootstrap -- it just gets the floored seed interval instead of the
// measured one (95), and says so in `interval_provenance`.
const QD_BASELINE_SAMPLES = new Map((BASELINE_PER_CASE || []).map(c => [
  c && (c.name || c.test_case_id),
  (c && [c.samples_ms, c.baseline_samples_ms, c.latency_samples_ms].find(Array.isArray)) || [],
]).filter(([name, xs]) => name && xs.length));
if (QD_ENABLED && (QD_CONTEXT_IDS.size === 0 || QD_BASELINE_MS.size !== QD_CONTEXT_IDS.size)) {
  throw new Error('QD bootstrap failed closed: every exact harness context needs a positive authoritative baseline latency');
}
// Finding (58). MIRROR of `MEASURED_NOISE_FLOOR` in scripts/qd_robust_stats.py,
// which is the authoritative table; `test_qd_lane_parity.py` fails if the two
// drift apart. The floor lived only on the Python side for the whole life of
// finding (26), which is to say it lived in the copy that ANALYSES results and
// not in the copy that ADMITS them.
//
// Why a floor is needed at all: `median +- 2*MAD` is not a 95% interval. At the
// n=3 this protocol actually runs, MAD is `min(b-a, c-b)` for sorted samples --
// the smaller of the two gaps -- so 2*MAD averages 0.91 sigma (measured; the
// n->inf asymptote is only 1.35 sigma, against 1.96 for a real 95% half-width).
// Executing this lane's own qdCaseRobust on 20k pairs of arms that are
// IDENTICAL in truth, `robust.lower > incumbent.robust.upper` fired 9.34% of
// the time on `decode_m2_square` and 8.90% on the quietest route in the suite.
// That is the archive replacing its elite with noise roughly one round in
// eleven, and every mutation descended from that elite inheriting the mistake.
// With the floor applied: 0.00% of 20k.
//
// This block used to claim: "They are RELATIVE, so they survive the machine
// boundary that absolute milliseconds do not." That was reasoning, not a
// measurement, and re-measuring on machine N falsified it. The same eleven
// relative floors moved by up to 3.3x across the L->N boundary, in BOTH
// directions -- and the narrow direction is the one that admits noise:
// `prefill_m2048_square`, the route carrying the largest claimed win in the
// ledger, has a machine-L floor of 0.011 against a machine-N same-variant
// spread of 0.037. Relative quantities cross a machine boundary no better than
// absolute ones. Floors are therefore keyed by machine here exactly as they are
// in `qd_robust_stats.py`, and the two structures are checked for parity.
const QD_NOISE_FLOOR_BY_MACHINE = new Map([
  // machine L -- runs 1672/1673/1676/1677 (+1687-1721 isolated), v98, gfx90a
  ['L', new Map([
    ['decode_m2_square', 0.072], ['decode_m16_square', 0.033], ['prefill_m1024_down', 0.026],
    ['decode_m8_up', 0.020], ['decode_m32_down', 0.020], ['prefill_m128_square', 0.019],
    ['decode_m96_up', 0.011], ['prefill_m2048_square', 0.011], ['decode_m64_square', 0.007],
    ['prefill_m512_up', 0.007], ['prefill_m256_down', 0.005],
  ])],
  // machine N -- tw035, gfx942, ship point bc7ea649e9ea3b7e, 8 same-variant
  // runs under the (105) burn+ABBA harness
  ['N', new Map([
    ['decode_m2_square', 0.0378], ['prefill_m2048_square', 0.0368],
    ['prefill_m128_square', 0.0229], ['decode_m32_down', 0.0207],
    ['decode_m16_square', 0.0203], ['decode_m64_square', 0.0198],
    ['decode_m8_up', 0.0181], ['prefill_m512_up', 0.0149],
    ['prefill_m256_down', 0.0137], ['prefill_m1024_down', 0.0127],
    ['decode_m96_up', 0.0108],
  ])],
  // machine O -- tw054, gfx942, container restored from a snapshot onto a new
  // host on 2026-08-16. SAME ship point bc7ea649e9ea3b7e as machine N, 8
  // same-variant repeats across 2 lock acquisitions on GPU 2, discard+BCCB.
  // Wider than N on the two shortest decode routes, narrower on nearly all the
  // rest -- see qd_robust_stats.py for why that rules out a global scale factor.
  ['O', new Map([
    ['decode_m16_square', 0.0588], ['decode_m2_square', 0.0416],
    ['prefill_m128_square', 0.0209], ['decode_m32_down', 0.0114],
    ['prefill_m256_down', 0.0097], ['decode_m96_up', 0.0073],
    ['prefill_m1024_down', 0.0062], ['prefill_m2048_square', 0.0060],
    ['decode_m64_square', 0.0046], ['prefill_m512_up', 0.0045],
    ['decode_m8_up', 0.0026],
  ])],
  // machine P -- tw008, gfx942, the host run 16 moved onto at ~16:22. SAME ship
  // point bc7ea649e9ea3b7e as N and O. 13 same-variant full-suite parent runs
  // already on disk from run 16, all primed, ~2.5 hours and 4+ lock
  // acquisitions, mixed harness -- so it folds in session drift and is
  // conservative. Wider than O on ten of eleven routes (up to 4.8x) and 9x
  // narrower on decode_m16_square: no scale factor relates them, for the third
  // machine boundary running. See qd_robust_stats.py for the derivation and the
  // caveats; that table is authoritative and this one mirrors it.
  ['P', new Map([
    ['prefill_m256_down', 0.0430], ['prefill_m128_square', 0.0287],
    ['decode_m96_up', 0.0270], ['decode_m2_square', 0.0224],
    ['prefill_m512_up', 0.0218], ['decode_m32_down', 0.0178],
    ['prefill_m1024_down', 0.0178], ['prefill_m2048_square', 0.0109],
    ['decode_m64_square', 0.0074], ['decode_m16_square', 0.0066],
    ['decode_m8_up', 0.0047],
  ])],
  // machine Q -- tw003, gfx942, the host the container was restored onto after
  // run 16. PROVISIONAL: nothing has been measured here. Every route carries
  // the fail-closed default, so admission on this epoch refuses any effect
  // smaller than the widest spread ever measured on any box. That is correct
  // and it is also expensive: at 0.072 a 2-5% suite move is unreadable and
  // cannot be admitted at all. Measuring the real tw003 table -- 8 same-variant
  // full-suite primed repeats, finding (105) debiased harness, the same
  // 2*MAD(speedup)/median(speedup) statistic -- is the first GPU work when a
  // device frees up. Mirrors PROVISIONAL_MACHINES in qd_robust_stats.py.
  ['Q', new Map([
    ['prefill_m256_down', 0.072], ['prefill_m128_square', 0.072],
    ['decode_m96_up', 0.072], ['decode_m2_square', 0.072],
    ['prefill_m512_up', 0.072], ['decode_m32_down', 0.072],
    ['prefill_m1024_down', 0.072], ['prefill_m2048_square', 0.072],
    ['decode_m64_square', 0.072], ['decode_m16_square', 0.072],
    ['decode_m8_up', 0.072],
  ])],
  // machine R -- tw008. MEASURED: 8 complete same-variant primed repeats, source_hash f3da61b3e2b673f7cf2c2847a668432860f90c37b7eb848c863ad8fdacddb2fa.
  // Installed by deprovisionalize_epoch.py from the sweep verdict; the statistic is 2*MAD(speedup)/median(speedup) per route, floors below MIN_FLOOR (0.002) clamped up. Floors do not pool across a machine boundary, so this table is a reading of this box only.
  ['R', new Map([
    ['decode_m16_square', 0.0110],
    ['decode_m2_square', 0.0093],
    ['decode_m32_down', 0.0712],
    ['decode_m64_square', 0.0139],
    ['decode_m8_up', 0.0197],
    ['decode_m96_up', 0.0248],
    ['prefill_m1024_down', 0.0185],
    ['prefill_m128_square', 0.0092],
    ['prefill_m2048_square', 0.0029],
    ['prefill_m256_down', 0.0364],
    ['prefill_m512_up', 0.0245]
  ])],
]);
// The machine whose floors apply. One line, set deliberately; nothing infers it.
// Finding (126): it used to be a remembered constant with the hostname only in
// a comment, and it read 'P' (tw008) while the process was running on tw003.
// `test_the_epoch_letter_matches_the_host` now checks this letter against the
// box, via MACHINE_HOSTNAME in qd_robust_stats.py.
const QD_CURRENT_MACHINE = 'R';
const QD_NOISE_FLOOR = QD_NOISE_FLOOR_BY_MACHINE.get(QD_CURRENT_MACHINE);
// Epochs whose table is a fail-closed PLACEHOLDER rather than a measurement:
// structurally complete so that every code path behaves, every route at the
// widest floor ever measured anywhere, and nothing in it is a fact about that
// box. MIRROR of PROVISIONAL_MACHINES in scripts/qd_robust_stats.py, which is
// authoritative; `test_qd_lane_parity.py` fails if the two sets drift apart.
// Anything asking "is this route's floor a measurement" must consult this and
// not table membership -- on a provisional epoch every route is present.
const QD_PROVISIONAL_MACHINES = new Set(['Q']);
const qdFloorIsMeasured = (contextId) =>
  !QD_PROVISIONAL_MACHINES.has(QD_CURRENT_MACHINE) && QD_NOISE_FLOOR.has(contextId);
// A provenance string that says "measured: 8 primed baseline repeats" states
// how many and not WHERE, and a floor is a reading of a box (26)/(58). Two
// epochs' seed intervals are not comparable, and an interval quoted out of an
// archive with no epoch on it is a number the reader has to date by hand from
// the run id. So the stamp travels with the interval.
//
// Mirrored from `qd_robust_stats.MACHINE_HOSTNAME` and kept equal to it by
// `test_qd_lane_parity.py`, for finding (126)'s reason: an epoch letter that is
// never checked against the host it claims is not provenance, it is a comment.
// `null` for the two gfx90a-era epochs, which pre-date the convention -- an
// unrecorded host is stated as unrecorded, not backfilled with a guess.
const QD_MACHINE_HOSTNAME = new Map([
  ['L', null], ['M', null], ['N', 'tw035'], ['O', 'tw054'], ['P', 'tw008'], ['Q', 'tw003'], ['R', 'tw008'],
]);
const QD_EPOCH_STAMP = (() => {
  const host = QD_MACHINE_HOSTNAME.get(QD_CURRENT_MACHINE);
  const provisional = QD_PROVISIONAL_MACHINES.has(QD_CURRENT_MACHINE)
    ? ', floor table PROVISIONAL' : '';
  return ` [epoch ${QD_CURRENT_MACHINE}${host ? ` (${host})` : ' (host unrecorded)'}${provisional}]`;
})();
// An unmeasured context gets the WIDEST floor measured on ANY machine, not the
// mean, not zero, and not the widest on this machine. Too wide costs a real
// improvement one more round of measurement; too narrow admits noise as an
// elite and every later comparison inherits it. An unmeasured route belongs to
// no epoch, so it cannot borrow this epoch's spread either.
const QD_DEFAULT_NOISE_FLOOR = Math.max(
  ...[...QD_NOISE_FLOOR_BY_MACHINE.values()].flatMap((t) => [...t.values()]));
const qdNoiseFloor = (contextId) => QD_NOISE_FLOOR.has(contextId)
  ? QD_NOISE_FLOOR.get(contextId) : QD_DEFAULT_NOISE_FLOOR;
const QD_VOCAB = {
  compute_primitive: new Set(['valu', 'rocwmma', 'native_mfma']),
  wave_schedule: new Set(['independent', 'symmetric_interleave', 'symmetric_pingpong', 'asymmetric_producer_consumer']),
  // lds_deep_single is not a weaker pingpong: past a certain stage depth the
  // double buffer stops fitting in LDS at all, so "deeper stage, one buffer" is
  // the only way further along this axis, and on a grid that cannot fill the
  // machine the residency it gives up was never real. Measured 0.76 -> 0.88 on
  // the shape it fires on.
  k_pipeline: new Set(['direct_global', 'lds_single', 'lds_reg_prefetch', 'lds_pingpong', 'lds_deep_single', 'lds_multistage']),
  decomposition: new Set(['tile_grid', 'persistent_output', 'split_k', 'stream_k']),
  output_path: new Set(['direct_store', 'lds_staged_store', 'atomic_fixup', 'workspace_fixup']),
  // Which output tile a workgroup id maps to. Identical arithmetic, identical
  // occupancy, different set of panels per last-level cache.
  rasterization: new Set(['linear', 'grouped_m', 'xcd_remapped_grouped']),
  // When the launch configuration is chosen: on the host from shape and CU
  // count, or by measuring the candidates on the real stream and keeping the
  // winner. Trades host time and decision variance for per-shape fit.
  plan_binding: new Set(['static', 'runtime_tuned']),
};
// Returns null when the descriptor is legal, otherwise a short stable token
// naming the axis or rule that refused it.
//
// This is the ONLY copy of the legality rules; `qdDescriptorValid` is a
// predicate over it. That matters more than the diagnostic: finding (44) was a
// vocabulary that existed twice and drifted, and a second copy written "just to
// explain the failure" would drift the same way and lie about the reason.
//
// The reason exists because the refusal was invisible. `qdRouteCells` dropped
// an illegal route with a bare `continue`: no error, no failing test, and an
// archive that silently never filled. Three axes were mandatory-and-
// unrequestable for weeks and the only symptom was a search that found nothing.
// A refusal that cannot say why is indistinguishable from an empty search.
const qdDescriptorReject = (d) => {
  if (!d) return 'descriptor:absent';
  for (const [k, values] of Object.entries(QD_VOCAB)) {
    if (!values.has(d[k])) {
      return `axis:${k}=${d[k] === undefined ? '<missing>' : JSON.stringify(d[k])}`;
    }
  }
  const reduction = d.decomposition === 'split_k' || d.decomposition === 'stream_k';
  const fixup = d.output_path === 'atomic_fixup' || d.output_path === 'workspace_fixup';
  // Finding (61). This was one token, `rule:reduction_fixup_coupling`, for both
  // directions. The rule is one rule, but the two violations need opposite
  // corrections -- add a fixup, or drop it -- and the planner reading this token
  // is the thing that has to pick. Per (60), refusals of different kinds must be
  // distinguishable, and "which side is wrong" is a different kind.
  if (reduction && !fixup) return 'rule:reduction_without_fixup';
  if (fixup && !reduction) return 'rule:fixup_without_reduction';
  if ((d.wave_schedule === 'symmetric_interleave' || d.wave_schedule === 'symmetric_pingpong') &&
      d.compute_primitive === 'valu') return 'rule:pingpong_requires_matrix_core';
  if (!QD_SUPPORTED_ARCHES.includes(QD_ARCH) ||
      !['bf16', 'fp16', 'fp32', 'fp64', 'int8'].includes(QD_DTYPE)) return 'rule:unsupported_arch_or_dtype';
  // Un-shuffling a die round-robin is only a mechanism where the hardware
  // shuffles; elsewhere it is the identity plus a scalar prologue, i.e. cost.
  if (d.rasterization === 'xcd_remapped_grouped' && !QD_MULTI_DIE_ARCHES.includes(QD_ARCH)) {
    return 'rule:xcd_remap_requires_multi_die';
  }
  // Runtime tuning needs something to tune, and the only launch parameter
  // chosen per call rather than per route is the K-reduction slice count.
  if (d.plan_binding === 'runtime_tuned' && !reduction) return 'rule:runtime_tuned_requires_reduction';
  return null;
};
const qdDescriptorValid = (d) => qdDescriptorReject(d) === null;
const qdCoverageEligible = (d) => qdDescriptorValid(d) && d.wave_schedule !== 'asymmetric_producer_consumer';
const qdHashValid = (value) => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);
const QD_AXIS_ORDER = {
  compute_primitive: ['valu', 'rocwmma', 'native_mfma'],
  wave_schedule: ['independent', 'symmetric_interleave', 'symmetric_pingpong', 'asymmetric_producer_consumer'],
  k_pipeline: ['direct_global', 'lds_single', 'lds_reg_prefetch', 'lds_pingpong', 'lds_deep_single', 'lds_multistage'],
  decomposition: ['tile_grid', 'persistent_output', 'split_k', 'stream_k'],
  output_path: ['direct_store', 'lds_staged_store', 'atomic_fixup', 'workspace_fixup'],
  rasterization: ['linear', 'grouped_m', 'xcd_remapped_grouped'],
  plan_binding: ['static', 'runtime_tuned'],
};
// (80). Descriptor identity, as a VALUE rather than as a byte string.
//
// Every descriptor comparison in this file used to be `JSON.stringify(a) ===
// JSON.stringify(b)`, which serialises keys in insertion order and is therefore
// an ordering test, not a value test. It cost a whole run. `qdAdjacency` builds
// neighbours as `{ ...parent, [axis]: v }`, so they inherit the parent's key
// order (alphabetical, as persisted); the planner is an agent writing JSON and
// wrote its axes in the order it reasoned about them. Every legal single-axis
// edge compared unequal to itself, the legality gate rejected all of them, and
// the round dispatched nothing while reporting a clean `rounds: 0`.
//
// The same bug pointed the other way in the `targetIsCurrent` no-op gate, which
// FAILED OPEN: a target identical in content to the parent but written in a
// different key order was not recognised as current and would have bought a
// full build-and-verify cycle to re-measure the parent.
//
// So: one canonical key, over exactly the seven axes and in a fixed order, used
// for every descriptor identity in the lane. Extra keys are ignored on purpose
// -- the descriptor IS its seven axes (`qdCellId` already takes that view), and
// an agent decorating one with a comment field must not thereby invent a new
// mechanism. An invalid descriptor returns null, which never compares equal to
// anything, including another null: unequal is the safe answer for "this is not
// a descriptor", and `===` on two nulls would have quietly made all garbage
// identical.
const QD_AXES = Object.keys(QD_AXIS_ORDER);
const qdDescriptorKey = (d) => {
  if (!d || typeof d !== 'object') return null;
  const parts = [];
  for (const axis of QD_AXES) {
    const v = d[axis];
    if (typeof v !== 'string' || !QD_AXIS_ORDER[axis].includes(v)) return null;
    parts.push(`${axis}=${v}`);
  }
  return parts.join('|');
};
// Value equality for descriptors. Null-safe in the refusing direction, per above.
const qdDescriptorSame = (a, b) => {
  const ka = qdDescriptorKey(a);
  return ka !== null && ka === qdDescriptorKey(b);
};
// The orchestrator's half of the rounds-law gate (finding 24).
//
// `qdAdjacency` answers "is this edge legal in the vocabulary". It cannot
// answer "has this already been measured and lost", which is what
// `qd_v2.py mutation-verdict` knows -- and that helper ran only inside the
// planner's prompt. Findings (19)-(24) were each encoded there and re-proposed
// by the search anyway, every run, because a gate that lives only in a prompt
// is a gate the planner can forget. So the planner now returns its receipt and
// this re-checks it.
//
// **What this can and cannot check, stated plainly.** It cannot verify that the
// facts describe the candidate: the candidate is not built yet, and `ctas` /
// `residency_slots` are properties of an object that does not exist. Nothing
// available at plan time can. What it *can* do is refuse a receipt that is
// absent, that reports a refusal, or whose own arithmetic does not hold -- and
// the last of those is the one worth having, because a planner that fabricates
// a passing receipt has to fabricate three consistent numbers per side to get
// past it. Returns null when the receipt is acceptable, else a reason token.
const qdResidencyReject = (receipt) => {
  if (!receipt) return 'residency:receipt_absent';
  if (receipt.allow !== true) {
    const why = (receipt.refusals || []).join(', ') || 'no refusals listed';
    return `residency:gate_refused(${why})`;
  }
  const sides = { current: receipt.current, candidate: receipt.candidate };
  for (const [side, facts] of Object.entries(sides)) {
    if (!facts) return `residency:${side}_facts_absent`;
    const { ctas, residency_slots: slots, rounds } = facts;
    for (const [field, value] of [['ctas', ctas], ['residency_slots', slots], ['rounds', rounds]]) {
      if (!Number.isInteger(value) || value <= 0) return `residency:${side}.${field}=${JSON.stringify(value)}`;
    }
    if (rounds !== Math.ceil(ctas / slots)) {
      return `residency:${side}_arithmetic(rounds=${rounds}, ceil(${ctas}/${slots})=${Math.ceil(ctas / slots)})`;
    }
  }
  // (24): both configurations that ever crossed from one round to two lost
  // 27-43%. `max(1, ...)` rather than a bare 1 so a route that already needs
  // two rounds is held at two instead of being declared unfixable.
  if (receipt.candidate.rounds > Math.max(1, receipt.current.rounds)) {
    return `residency:rounds_raised(${receipt.current.rounds} -> ${receipt.candidate.rounds})`;
  }
  return null;
};

// Finding (68). `qd_v2.py route-priority` is the third gate tech_lead.md makes
// mandatory, and the only one that answers "is this route worth a build at
// all?". Its rule is per-entry -- "drop the `closed` entries from
// `target_cases` and keep the rest" -- and exit 3 fires only in the degenerate
// case where every entry is closed. Nothing outside the prompt had ever looked:
// `route-priority` and `richest` had zero occurrences in this file, so the one
// gate whose whole job is to stop a wasted round rested entirely on the planner
// remembering to obey it. That is the same fail-open the residency gate above
// was written to close, and (19)-(24) are the evidence that a planner does not
// remember.
//
// A closed route is not merely a poor bet. `closed` means the entire remaining
// headroom is smaller than the route's own noise floor, so reaching the
// hardware floor *exactly* would produce a change no protocol in this project
// can read. A round aimed there is unreadable by construction, and any apparent
// win on it is an artifact of a kind the interval test does not catch, because
// the interval test measures spread and this is bias.
//
// **What is re-derived and what is not.** The orchestrator has no measurements,
// so it cannot recompute `remaining_headroom`; that comes from the SOL card and
// the route's elapsed time. It CAN recompute the two things that turn the
// receipt from an assertion into a checkable claim: the verdict is a pure
// function of the two reported numbers, and one of those numbers -- the noise
// floor -- is a table this file already owns. So a fabricated receipt has to
// state this route's real floor and then draw the arithmetic conclusion that
// floor implies, which leaves nowhere to put a thumb except `remaining_headroom`
// itself. Mirroring the elapsed/traffic tables here instead was considered and
// rejected: they are machine-L snapshots, and importing a cross-machine
// absolute into a gate is the error this ledger keeps warning about.
//
// Returns `{ cases }` with the closed entries removed, or `{ reason }`.
const QD_CLOSED_RATIO = 1.0;    // headroom < floor: unreadable even at the floor.
const QD_MARGINAL_RATIO = 3.0;  // same order of magnitude as the floor.
const qdPriorityVerdict = (headroom, floor) => {
  const ratio = headroom / floor;
  return ratio < QD_CLOSED_RATIO ? 'closed' : (ratio < QD_MARGINAL_RATIO ? 'marginal' : 'open');
};
const qdPriorityFilter = (receipt, targetCases) => {
  const wanted = (targetCases || []).filter(c => typeof c === 'string' && c);
  if (!wanted.length) return { reason: 'priority:no_target_cases' };
  if (!receipt || !Array.isArray(receipt.per_context)) return { reason: 'priority:receipt_absent' };
  const rows = new Map();
  for (const row of receipt.per_context) {
    if (row && typeof row.context === 'string') rows.set(row.context, row);
  }
  // Finding (62): a completeness claim scoped by what happens to be named omits
  // whatever was named differently. The receipt covering *some* target case is
  // not the receipt covering this direction's target cases.
  const uncovered = wanted.filter(c => !rows.has(c));
  if (uncovered.length) return { reason: `priority:uncovered(${uncovered.join(', ')})` };
  const kept = [];
  for (const context of wanted) {
    const row = rows.get(context);
    const headroom = row.remaining_headroom;
    const floor = row.noise_floor;
    if (!Number.isFinite(headroom) || headroom < 0 || headroom >= 1) {
      return { reason: `priority:${context}.remaining_headroom=${JSON.stringify(headroom)}` };
    }
    if (!Number.isFinite(floor) || floor <= 0) {
      return { reason: `priority:${context}.noise_floor=${JSON.stringify(floor)}` };
    }
    const own = qdNoiseFloor(context);
    if (Math.abs(floor - own) > 1e-6 * Math.max(1, own)) {
      return { reason: `priority:floor_mismatch(${context}: receipt ${floor}, table ${own})` };
    }
    const derived = qdPriorityVerdict(headroom, own);
    // Finding (92). `needs_fresh_elapsed` is what the helper now reports when
    // nobody supplied `--elapsed-ms`, i.e. when the headroom was computed from
    // the SHIPPED kernel's latency on machine L rather than from this
    // candidate. The arithmetic is still checkable -- the row carries the
    // verdict that number WOULD have implied -- so the receipt is still
    // verified, against `verdict_if_elapsed_confirmed` instead.
    //
    // What must not happen is the drop. A route nobody measured is not a route
    // with nothing left in it, and dropping it here would close the loop that
    // makes the mistake permanent: never proposed, therefore never measured,
    // therefore closed forever on a number about a different kernel. So an
    // unmeasured target is KEPT even when its conditional verdict is "closed".
    const unmeasured = row.verdict === 'needs_fresh_elapsed';
    const claimed = unmeasured ? row.verdict_if_elapsed_confirmed : row.verdict;
    if (claimed !== derived) {
      return { reason: `priority:verdict_arithmetic(${context}: receipt "${claimed}", `
        + `${headroom}/${own} implies "${derived}")` };
    }
    // Finding (126), same argument one level out. On a PROVISIONAL epoch the
    // floor this closure was derived from is a placeholder, not a measurement
    // taken on this box, so "closed" here means "unreadable against the widest
    // spread ever seen anywhere" -- which is a statement about the missing
    // measurement, not about the route. Dropping the route on it would close
    // the same permanent loop the paragraph above refuses to close.
    if (unmeasured || derived !== 'closed' || !qdFloorIsMeasured(context)) kept.push(context);
  }
  // Exit 3's case, and the only one that costs the whole direction: with every
  // target closed there is nowhere for it to land.
  if (!kept.length) return { reason: `priority:all_targets_closed(${wanted.join(', ')})` };
  return { cases: kept };
};

// Finding (69). The post-build policy receipt, checked rather than assumed.
// See POLICY_SUMMARY_SCHEMA for why this is a summary and not the file.
//
// **What this can and cannot do, stated plainly.** It cannot prove the scan
// happened: nothing available to a script with no filesystem can. What it can
// do is refuse the shapes a *missing* scan actually takes, and those are not
// exotic -- they are what you get from an agent that skipped the step, ran the
// pre-build scan twice, or pointed the scanner at the wrong directory:
//   - no summary at all (the overwhelmingly likely form of "I skipped it")
//   - a summary whose `passed` contradicts the `policy_pass` beside it
//   - `passed: true` with a non-zero finding count, which the scanner itself
//     can never emit (`passed = not findings`), so it is not a receipt
//   - `inspected: 0` -- a scan of nothing passes trivially (66)
//   - `elf: 0` on a POST-build scan -- the source scan cannot see DT_NEEDED,
//     so this receipt is not evidence about linkage at all
//   - `unreadable > 0` -- verify_engineer.md 2b already says an uninspectable
//     artifact fails closed; it just had nowhere to be enforced
// A fabricated pass now requires stating a positive ELF count for a scan the
// agent did not run, which is a materially different act from omitting a field.
const qdPolicyReject = (rep, label) => {
  const where = label || 'verify';
  // Only speaks about reports that CLAIM a pass. An honest `policy_failed` is
  // already refused by the caller's own `policy_pass !== true`, and complaining
  // about its receipt here would rename that refusal into a paperwork problem
  // -- the one thing (60) says a refusal must never do.
  if (!rep || rep.policy_pass !== true) return null;
  const s = rep.policy_postbuild;
  if (!s || typeof s !== 'object') {
    return `policy:postbuild_summary_absent(${where} asserted policy_pass with no post-build receipt; `
      + 'the pre-build source scan cannot see DT_NEEDED, so nothing here is evidence about linkage)';
  }
  if (typeof s.schema === 'string' && s.schema && !/policy/i.test(s.schema)) {
    return `policy:foreign_schema("${s.schema}")`;
  }
  if (s.passed !== true) {
    return `policy:receipt_contradicts_policy_pass(policy_pass true, receipt passed ${JSON.stringify(s.passed)})`;
  }
  if (!Number.isFinite(s.findings) || s.findings !== 0) {
    return `policy:passing_receipt_with_findings(${JSON.stringify(s.findings)}; the scanner defines `
      + 'passed as "no findings", so these two cannot both be true)';
  }
  // `files`, not `inspected`: the latter counts directory entries, so it reads 1
  // for an empty tree. Found by executing the gate, not by reading it (59).
  if (!Number.isFinite(s.files) || s.files < 1) {
    return `policy:inspected_nothing(files=${JSON.stringify(s.files)}; a scan that opened no file passes trivially, `
      + 'and `inspected` cannot see this because it counts the directory itself)';
  }
  if (!Number.isFinite(s.elf) || s.elf < 1) {
    return `policy:no_binary_inspected(${JSON.stringify(s.elf)}; a post-build scan that saw no ELF is a `
      + 'pre-build scan under another name, and linkage is only visible in the binary)';
  }
  if (Number.isFinite(s.unreadable) && s.unreadable > 0) {
    return `policy:uninspectable_artifacts(${s.unreadable}; verify_engineer.md 2b fails closed on these)`;
  }
  return null;
};

const qdAdjacency = (d) => {
  if (!qdDescriptorValid(d)) return [];
  const out = [], seen = new Set();
  const add = (candidate, axes, direction) => {
    if (!qdDescriptorValid(candidate)) return;
    const key = qdDescriptorKey(candidate);   // (80): value identity, not key order
    if (key === null || seen.has(key)) return;
    seen.add(key); out.push({ descriptor: candidate, axes, direction });
  };
  for (const [axis, order] of Object.entries(QD_AXIS_ORDER)) {
    const i = order.indexOf(d[axis]);
    for (const [step, direction] of [[-1, 'prev'], [1, 'next']]) {
      const j = i + step;
      if (j >= 0 && j < order.length) add({ ...d, [axis]: order[j] }, [axis], direction);
    }
  }
  const fixups = ['atomic_fixup', 'workspace_fixup'];
  const nonFixups = ['direct_store', 'lds_staged_store'];
  if (d.decomposition === 'persistent_output' && nonFixups.includes(d.output_path)) {
    for (const output_path of fixups) add({ ...d, decomposition: 'split_k', output_path },
      ['decomposition', 'output_path'], 'coupled');
  } else if (d.decomposition === 'split_k') {
    // Dropping the reduction also drops the only parameter the runtime tuner
    // binds, so plan_binding has to come back to static in the same step or the
    // edge lands on an illegal descriptor and silently disappears.
    for (const output_path of nonFixups) {
      const tuned = d.plan_binding === 'runtime_tuned';
      add({ ...d, decomposition: 'persistent_output', output_path, plan_binding: tuned ? 'static' : d.plan_binding },
        tuned ? ['decomposition', 'output_path', 'plan_binding'] : ['decomposition', 'output_path'], 'coupled');
    }
  }
  return out;
};
const qdCellId = (contextId, d) => {
  if (!QD_CONTEXT_IDS.has(contextId) || !qdDescriptorValid(d)) return null;
  return [contextId, d.compute_primitive, d.wave_schedule, d.k_pipeline,
    d.decomposition, d.output_path, d.rasterization, d.plan_binding].join('|');
};
// Finding (87), wired. `hip_twin_sync.py` has existed and been correct for
// several rounds while nothing called it, which is the same fail-open as (68):
// a check nobody runs is a comment.
//
// What it is checking is the worst shape a measurement bug can take. torch's
// hipify rewrites `X.hip` into `X_hip.hip`, prints "[skipped, already
// hipified]" when the twin exists, and ninja then compiles THE TWIN. An edit
// applied only to the primary changes nothing that runs: the build succeeds,
// correctness passes, and the benchmark reports a clean null result for a
// change that was never compiled. It is indistinguishable from an honest
// negative -- and an honest negative is what closes a search direction. So the
// cost of missing this is not one wasted round, it is a mechanism written off.
//
// **Exit 2 is not a pass, and that is the whole reason this needs a gate rather
// than a boolean.** The tool returns 2 when it found no twins to compare, and
// prints `HOLE:` rather than `ok:` precisely so the distinction survives; an
// agent summarizing its own run into `twin_sync_pass: true` erases it. A
// verifier that measured a build and found no twins either did not build in the
// directory it scanned or scanned before the build -- both mean the measurement
// is unbacked, and unbacked in the direction that manufactures null results.
// The parent-provenance gate. Every direction the planner returns claims a
// parent, a cell and a SOL card; this resolves those claims against the live
// archive and refuses the direction if any of them does not hold.
//
// Why it is a named function. The other two gates in this map -- residency and
// route-priority -- were extracted so the JS suite could *execute* them against
// fabricated receipts. This one, the oldest and the most load-bearing of the
// three, stayed inline, so the only thing defending it was a regex asserting
// that its log message exists. `audit_pin_coverage.py` is what surfaced that:
// the message pin at `test_qd_archive.js:548` is flipped by no mutant in the
// corpus, and the section it lives in is one of the few not marked `executed`.
// Two of three extracted is (57) exactly -- the invariant was applied at the
// sites that came up rather than at every site it holds for.
//
// It also now says *which* claim failed. The single lumped message named four
// possibilities and committed to none, and the four are not equally alarming: a
// planner aiming at an illegal descriptor is a prompt problem, while a
// `source_hash` that does not match the live cell means the archive moved under
// the round and is the one that invalidates results already collected.
//
// Returns `{ selection, parent, card, donors }` when the direction is sound, or
// `{ reason }`. `donors` is carried out because the caller needs the resolved
// elites and resolving them twice is how the two copies drift.
const qdParentReject = (d, selections, solCards, cells) => {
  const selection = (selections || []).find(s => s.id === d.id ||
    (s.parent_elite_id === d.parent_elite_id && s.selected_cell === d.selected_cell));
  if (!selection) return { reason: `parent:no_selection_for_direction(${d.id || '<unnamed>'})` };
  const parent = selection.parent;
  if (!parent) return { reason: `parent:selection_carries_no_parent(${selection.selected_cell})` };
  const card = (solCards || []).find(c => c.selected_cell === selection.selected_cell &&
    c.context_id === selection.context_id && c.parent_source_hash === selection.parent_source_hash);
  if (!card) {
    return { reason: `parent:no_sol_card(${selection.selected_cell}|${selection.context_id}` +
      `|${selection.parent_source_hash})` };
  }
  if (d.parent_source_hash !== parent.source_hash) {
    return { reason: `parent:source_hash_moved(direction=${d.parent_source_hash}, ` +
      `live cell=${parent.source_hash})` };
  }
  const donorIds = Array.isArray(d.parent_elite_ids) ? d.parent_elite_ids : [];
  const donors = donorIds.map(id => Object.values(cells || {}).find(e => e.elite_id === id));
  if (d.operator === 'directed_transition') {
    if (d.target_descriptor && qdDescriptorSame(d.target_descriptor, parent.descriptor)) {
      return { reason: 'parent:transition_target_is_the_parent_itself' };
    }
    const legal = qdAdjacency(parent.descriptor);
    if (!legal.some(n => qdDescriptorSame(n.descriptor, d.target_descriptor))) {
      return { reason: `parent:transition_illegal(${JSON.stringify(d.target_descriptor || null)})` };
    }
  }
  if (d.operator === 'semantic_crossover') {
    // The parent counts as one of the sources: crossing an artifact with itself
    // is a no-op dressed as a crossover, and it is the cheap way for a planner
    // to spend a build and report a direction attempted.
    const distinct = donors.length === donorIds.length && donors.every(Boolean) &&
      new Set([parent.source_hash, ...donors.map(e => e.source_hash)]).size === donors.length + 1;
    if (!distinct) {
      return { reason: `parent:crossover_sources_not_distinct(${donorIds.length} donor id(s), ` +
        `${donors.filter(Boolean).length} resolved)` };
    }
  }
  return { selection, parent, card, donors };
};

// Scope, and it is narrow on purpose. The twin only exists because torch's
// cpp_extension hipifies `.hip` SOURCES; a Triton lane -- which is the default
// `target_language` -- has no `X.hip`/`X_hip.hip` pair to compare and would
// return exit 2 forever. Requiring the receipt there would refuse every
// candidate in the lane for a hazard that cannot occur in it, which is not
// fail-closed, it is fail-shut. So the gate is armed by language, and the
// arming condition is a literal the parity test can read.
const TWIN_LANGUAGES = ['hip', 'cuda'];
const TWIN_APPLICABLE = TWIN_LANGUAGES.includes(String(TARGET_LANGUAGE).toLowerCase());
const qdTwinReject = (rep, label) => {
  if (!TWIN_APPLICABLE) return null;
  const where = label || 'verify';
  // Same scoping rule as `qdPolicyReject`: only reports claiming a pass. A
  // verifier that already failed is refused on its own terms, and adding a
  // paperwork complaint on top renames the refusal (60).
  if (!rep || rep.status === 'failed' || rep.correctness === 'FAIL') return null;
  const s = rep.hip_twin_sync;
  if (!s || typeof s !== 'object') {
    return `twin:receipt_absent(${where} reported a measurement without running hip_twin_sync.py; `
      + 'ninja compiles the _hip.hip twin, so an unchecked tree may have measured the parent)';
  }
  if (!Number.isInteger(s.exit_code)) {
    return `twin:no_exit_code(${JSON.stringify(s.exit_code)})`;
  }
  if (s.exit_code === 1) {
    return `twin:drifted(${where}: ${JSON.stringify(s.drifted)} pair(s) out of lockstep; the edit is `
      + 'not in the binary that was measured)';
  }
  if (s.exit_code === 2) {
    return `twin:nothing_checked(${where}: exit 2 is the tool's HOLE, not its pass -- no X.hip/X_hip.hip `
      + 'pair was found, so this measurement has no evidence that what was edited is what was compiled)';
  }
  if (s.exit_code === 3) {
    return `twin:launch_unreadable(${where}: exit 3 means the line halves matched but a launch `
      + 'statement could not be normalized, so the launch half went UNCHECKED -- a grid, block, '
      + 'stream or launch-time template edit applied to only one file would still be invisible)';
  }
  if (s.exit_code !== 0) {
    return `twin:unknown_exit(${s.exit_code})`;
  }
  // Exit 0 has two arithmetic consequences the tool itself guarantees, so a
  // receipt that violates either was not produced by running it.
  if (!Number.isInteger(s.pairs) || s.pairs < 1) {
    return `twin:zero_pairs_with_exit_zero(${JSON.stringify(s.pairs)}; the tool returns 2, not 0, when it `
      + 'finds no pairs, so this receipt is not one it can have emitted)';
  }
  if (Number.isFinite(s.drifted) && s.drifted > 0) {
    return `twin:exit_zero_with_drift(${s.drifted}; the tool returns 1 when any pair drifts)`;
  }
  return null;
};

const qdMedian = (xs) => {
  const values = (xs || []).filter(Number.isFinite).slice().sort((a, b) => a - b);
  if (!values.length) return null;
  const mid = Math.floor(values.length / 2);
  return values.length % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
};
const qdCaseRobust = (ver, contextId) => {
  const supplied = (ver && ver.case_measurement_samples || []).find(c => c && c.name === contextId);
  const baselineMs = QD_BASELINE_MS.get(contextId);
  const optimizedSamples = supplied && Array.isArray(supplied.samples)
    ? supplied.samples.filter(x => Number.isFinite(x) && x > 0) : [];
  if (!Number.isFinite(baselineMs) || optimizedSamples.length < 3) return null;
  const speedups = optimizedSamples.map(ms => baselineMs / ms);
  const median = qdMedian(speedups);
  const mad = median == null ? null : qdMedian(speedups.map(x => Math.abs(x - median)));
  if (median == null || mad == null) return null;
  // Finding (58): the radius is max(2*MAD, median * noise_floor), matching
  // qd_robust_stats.robust_stats. Taking 2*MAD alone lets three samples that
  // happened to agree present an interval far narrower than the route's own
  // run-to-run spread, and admission compares intervals.
  const radius = Math.max(2 * mad, Math.abs(median) * qdNoiseFloor(contextId));
  return { score: median, median, mad, bound_radius: radius,
    lower: Math.max(1e-9, median - radius), upper: median + radius };
};
// Finding (95). The seed used to carry a hardcoded `{median:1, mad:0, lower:1,
// upper:1}`. That is a definition, not a measurement -- the seed IS the
// baseline, so its speedup is 1 by construction -- but admission compares
// intervals, and a zero-width one makes every comparison against the seed
// one-sided. Any candidate clearing 1.0 by a hair displaces it. That is how
// `r1_s1_decode_m64_square` was admitted on a median of 1.0091, on a route
// whose own noise floor is 0.007.
//
// Two independent repairs, and they compose:
//
//   (a) measure it. If the benchmark engineer recorded repeats, the seed's
//       interval comes from `baseline_ms / sample_i` through the SAME function
//       every candidate goes through. This is the real fix: it is a null
//       comparison, so its width is that route's genuine run-to-run spread.
//   (b) floor it. Even with no repeats the radius is floored at the route's
//       noise floor, exactly as `qdCaseRobust` floors a candidate's.
//
// (b) alone would be tempting and wrong to stop at: it substitutes a table
// written on another epoch's hardware for a measurement of this one. It is
// here as the fallback for when (a) has nothing to work with, never as a
// reason to skip (a) -- which is why `interval_provenance` is recorded and
// why the two cases are separately asserted in the parity tests.
const qdSeedRobust = (contextId) => {
  const baselineMs = QD_BASELINE_MS.get(contextId);
  const samples = (QD_BASELINE_SAMPLES.get(contextId) || [])
    .filter(x => Number.isFinite(x) && x > 0);
  let median = 1, mad = 0,
    provenance = `floored_at_route_noise_floor: no baseline repeats recorded${QD_EPOCH_STAMP}`;
  if (Number.isFinite(baselineMs) && baselineMs > 0 && samples.length >= 3) {
    const speedups = samples.map(ms => baselineMs / ms);
    const m = qdMedian(speedups);
    const d = m == null ? null : qdMedian(speedups.map(x => Math.abs(x - m)));
    if (m != null && d != null) {
      median = m; mad = d;
      provenance = `measured: ${samples.length} primed baseline repeats${QD_EPOCH_STAMP}`;
    }
  }
  const radius = Math.max(2 * mad, Math.abs(median) * qdNoiseFloor(contextId));
  return { score: median, median, mad, bound_radius: radius,
    lower: Math.max(1e-9, median - radius), upper: median + radius,
    interval_provenance: provenance };
};
// The suite roll-up averages the per-context rows, the same way `qdRobust`
// does for a candidate -- so a seed measured on eleven routes with eleven
// different noise floors gets the mean of those floors, not the worst one.
const qdSeedSuiteRobust = () => {
  const rows = [...QD_CONTEXT_IDS].map(qdSeedRobust);
  const mean = (f) => rows.reduce((sum, r) => sum + r[f], 0) / rows.length;
  return { score: mean('median'), median: mean('median'), mad: mean('mad'),
    lower: mean('lower'), upper: mean('upper'),
    interval_provenance: rows.every(r => r.interval_provenance.startsWith('measured'))
      ? `measured: primed baseline repeats on every context${QD_EPOCH_STAMP}`
      : `floored_at_route_noise_floor on at least one context${QD_EPOCH_STAMP}` };
};
const qdRobust = (ver) => {
  const perContext = [...QD_CONTEXT_IDS].map(contextId => qdCaseRobust(ver, contextId));
  if (!perContext.length || perContext.some(x => !x)) return null;
  const mean = (field) => perContext.reduce((sum, row) => sum + row[field], 0) / perContext.length;
  return { score: mean('median'), median: mean('median'), mad: mean('mad'),
    lower: mean('lower'), upper: mean('upper') };
};
const qdMinCase = (ver) => {
  const robust = [...QD_CONTEXT_IDS].map(contextId => qdCaseRobust(ver, contextId));
  return robust.length && robust.every(Boolean)
    ? Math.min(...robust.map(row => row.median)) : 0;
};
// Finding (60). Both admission paths used to drop a verified candidate with a
// bare `continue`/`return false` covering three unrelated conditions at once:
// the suite interval could not be formed, an admitted route context had fewer
// than three usable samples, or the candidate genuinely regressed a context
// below the cell guardrail. The first two are HARNESS faults and the third is a
// PERFORMANCE verdict, and a round log that shows neither makes an archive that
// never filled because the Verifier returned two samples look exactly like an
// archive that never filled because nothing was fast enough. (48): a refusal
// that cannot name itself is indistinguishable from a search that found nothing.
// Returns {reason} to refuse, or {suiteRobust} to admit -- one object rather
// than a boolean plus a second qdRobust call, so there is no redundant guard
// downstream to go stale.
const qdAdmissionCheck = (ver, routeCells) => {
  const suiteRobust = qdRobust(ver);
  if (!suiteRobust) {
    return { reason: 'measurement:suite_interval_unavailable(a harness context has fewer than 3 usable samples)' };
  }
  // No separate per-route sample check: `qdCellId` refuses any context outside
  // QD_CONTEXT_IDS, so every routeCell's context is in the set `qdRobust` just
  // required three samples on. The old inline conjunction carried that
  // redundant clause and so did the first draft of this helper; the mutation
  // probe caught it by surviving. (56) -- an unreachable branch is an
  // unexercised one, and here it was also a second, silently weaker copy of a
  // rule that already holds.
  // Finding (62). Both admission paths store `primSpeedup(ver)` as the elite's
  // `geomean`. On a workload-aligned run that number must be the weighted one, or
  // the archive holds two incomparable quantities under one field name and every
  // later parent selection compares them.
  const metricReason = primMetricReason(ver);
  if (metricReason) return { reason: metricReason };
  // Finding (67). The archive outlives the run: an elite admitted here is a future run's warm
  // start and a future parent. Admitting one whose denominator cannot be tied to the pinned
  // oracle exports this run's uncertainty into every run after it.
  const oracleReason = oracleDrift(ver);
  if (oracleReason) return { reason: oracleReason };
  const minCase = qdMinCase(ver);
  if (minCase < QD_CELL_GUARDRAIL) {
    return { reason: `performance:worst_context_speedup_${minCase.toFixed(4)}_below_cell_guardrail_${QD_CELL_GUARDRAIL}` };
  }
  // (63): the callers store min_case; return the value that was actually gated on
  // rather than letting each site re-derive it from the same samples.
  return { suiteRobust, minCase };
};

// Finding (64). Durability is asserted by a TechLead receipt: the lane cannot stat the
// archive from the workflow sandbox, so it cannot confirm the files landed. What it can
// do -- and did not -- is check the receipt against the list it just handed out, and say
// out loud what the receipt costs. Written once because there are two callers and (63)
// is what a second hand-inlined copy costs.
const qdPersistenceReceipt = (persisted, admissions, where) => {
  const reported = persisted && Array.isArray(persisted.persisted_elite_ids)
    ? persisted.persisted_elite_ids : null;
  if (reported === null) {
    // Not "nothing was persisted" -- "the persister did not answer". Under the old code
    // this became an empty array and rolled everything back in silence, which is
    // indistinguishable in the log from a round that admitted nothing (60).
    log(`  [qd] persistence receipt MISSING (${where}): the archive updater returned no ` +
        `persisted_elite_ids, so all ${admissions.length} admitted cell(s) are being rolled back. ` +
        `This is a harness/agent fault, not a search result — do not read it as "found nothing".`);
  }
  const admittedIds = new Set(admissions.map(e => e.elite_id));
  const unknown = (reported || []).filter(id => !admittedIds.has(id));
  if (unknown.length) {
    // A receipt naming things that were never admitted is not a trustworthy receipt for
    // the things that were; it is honoured only for the intersection.
    log(`  [qd] persistence receipt (${where}) names ${unknown.length} elite id(s) never admitted ` +
        `here (${unknown.slice(0, 4).join(', ')}${unknown.length > 4 ? ', …' : ''}); ignoring them ` +
        `and honouring only the admitted intersection`);
  }
  return (reported || []).filter(id => admittedIds.has(id));
};
// (96). Serialization has no judgement in it, so it has no place for a model.
// `qd_persist_manifest.py` does the whole write -- materialize artifacts by
// reapplying each patch to the frozen baseline, check the reapplied tree hashes
// to its own content address, merge this round's cells onto what is on disk,
// write atomically, then re-read and describe what landed. What is left for the
// agent is the one thing this sandbox cannot do at all: touch the filesystem.
// It writes the payload and runs the command. It is not asked to decide
// anything, and below it is not believed either.
//
// FNV-1a/32 over ASCII. Must stay identical to `fnv1a32` in
// qd_persist_manifest.py. There is no crypto in the workflow JS sandbox, so
// this is what a checkable digest costs here; it is not defending against an
// adversary, it is catching a paraphrase.
// ASCII only, and it says so rather than quietly disagreeing: this side hashes
// UTF-16 code units and the python side hashes UTF-8 bytes, so the two agree on
// every input below U+0080 and on nothing above it. Every caller here feeds it
// `qdAsciiJson` output, which is pure ASCII by construction; the throw is for
// the caller that one day does not.
const qdFnv1a32 = (text) => {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c > 0x7f) throw new Error(`qdFnv1a32: non-ASCII input at ${i} (U+${c.toString(16)})`);
    h = Math.imul(h ^ c, 0x01000193) >>> 0;
  }
  return ('0000000' + h.toString(16)).slice(-8);
};
// JSON with every non-ASCII code unit escaped, so the text is pure ASCII and a
// character index is a byte index on both sides of the transport.
const qdAsciiJson = (obj) => JSON.stringify(obj).replace(/[\u0080-\uffff]/g,
  (c) => '\\u' + ('000' + c.charCodeAt(0).toString(16)).slice(-4));
// ...and the escape above is the fallback, not the plan -- finding (120).
//
// Round 15's persist payload was emitted as 92453 bytes of pure ASCII and
// arrived on disk as 92450. Exactly one substitution accounts for the
// difference: a single `\u2014` -- six ASCII characters -- was written out as a
// literal em-dash, three UTF-8 bytes. (92453 - 6 + 3 = 92450, and the file has
// exactly one non-ASCII character in it, at offset 8892, inside an evidence
// string that read "correctness fails silently \u2014 no route survives".) The
// agent's own note blamed context compaction; the bytes on disk say otherwise,
// and the bytes were there the whole time.
//
// A `\uXXXX` escape is the one construct in this transport that a transcriber
// can "correct" while believing it is copying: the escape and the character it
// denotes are the same string to a reader and different bytes to a checksum. So
// the escape is not made more emphatic, it is removed from the payload -- the
// prose is folded to ASCII before serialization, and `qdAsciiJson` then finds
// nothing above U+007F to escape. What is lost is typography inside evidence
// strings; an em-dash becomes `--`. The archive is prose that will be printed
// into prompts and retyped many more times, so ASCII is the right normal form
// for it regardless.
const QD_ASCII_FOLD = {
  '\u2014': '--', '\u2013': '-', '\u2012': '-', '\u2011': '-', '\u2212': '-',
  '\u2018': "'", '\u2019': "'", '\u201a': "'", '\u201b': "'",
  '\u201c': '"', '\u201d': '"', '\u201e': '"', '\u201f': '"',
  '\u2026': '...', '\u00a0': ' ', '\u2009': ' ', '\u200a': ' ', '\u202f': ' ',
  '\u2192': '->', '\u2190': '<-', '\u21d2': '=>', '\u00d7': 'x',
  '\u2265': '>=', '\u2264': '<=', '\u2260': '!=', '\u2248': '~=',
  '\u00b1': '+/-', '\u00b5': 'u', '\u03bc': 'u', '\u00b0': ' deg',
  '\u2022': '*', '\u00b7': '.', '\u2032': "'", '\u2033': '"',
};
const qdAsciiFoldText = (s) => s.replace(/[\u0080-\uffff]/g, (c) =>
  // Anything without a spelling becomes '?'. Dropping it silently would let two
  // distinct strings fold together; '?' at least shows where the character was.
  Object.prototype.hasOwnProperty.call(QD_ASCII_FOLD, c) ? QD_ASCII_FOLD[c] : '?');
const qdAsciiFold = (v) => {
  if (typeof v === 'string') return qdAsciiFoldText(v);
  if (Array.isArray(v)) return v.map(qdAsciiFold);
  if (v && typeof v === 'object') {
    const out = {};
    // Keys too: a cell key or elite id is ASCII by construction, but the payload
    // also carries free-form maps whose keys came from agent output.
    for (const k of Object.keys(v)) out[qdAsciiFoldText(k)] = qdAsciiFold(v[k]);
    return out;
  }
  return v;
};
// The digests the lane can recompute for itself, over the archive it holds in
// memory. `qd_persist_manifest.py` computes the same two by re-reading the file
// it wrote, which is what turns "the persister said it worked" into a
// comparison. Sorted, newline-joined, exactly as the script does it.
const qdCellDigests = () => {
  const keys = Object.keys(qdArchive.cells).sort();
  const eliteIds = keys.map(k => String(qdArchive.cells[k].elite_id)).sort();
  return {
    cells: keys.length,
    cell_keys_fnv1a32: qdFnv1a32(keys.join('\n')),
    elite_ids_fnv1a32: qdFnv1a32(eliteIds.join('\n')),
  };
};
// Merge mode: this round's admitted cells, plus the top-level fields, and NOT
// the cell map. Carrying the unchanged cells forward is then something the
// script does by reading the manifest, rather than something a transcriber is
// asked to remember -- which is the exact step that lost eleven cells and a
// replacement on round 1 of qd_v2_bf16_smoke_20260816b_tw054.
const qdPersistPayload = (admissions) => {
  const summary = qdSummary();
  const fields = {};
  for (const k of Object.keys(summary)) if (k !== 'cells') fields[k] = summary[k];
  const cellUpdates = {};
  for (const e of admissions) cellUpdates[e.cell] = summary.cells[e.cell];
  return {
    archive_dir: qdArchive.archive_dir,
    immutable_baseline: `${EVAL_DIR}/baseline`,
    manifest: fields,
    cell_updates: cellUpdates,
    admissions: admissions.map(e => ({
      elite_id: e.elite_id, cell: e.cell, source_hash: e.source_hash,
      patch: e.patch || null, parent_workspace: e.parent_workspace || null,
      // No patch means this is not a delta against the baseline but a tree in
      // its own right -- the bootstrap seed, and a warm import, which arrives
      // as a re-verified snapshot rather than as a diff.
      source_tree: e.patch ? null : (e.parent_workspace || null),
      generation: e.generation, operator: e.operator || null,
      parent_elite_id: e.parent_elite_id || null,
    })),
  };
};
// The prompt is deliberately two mechanical steps and no description of intent.
// The previous version of this call site asked the tech lead to "materialize
// content-addressed artifacts, cell references, challengers, and an atomic v2
// manifest" -- a description of a goal, which is an invitation to achieve it
// some other way, and it did.
// Finding (120), second half. The ASCII fold above removes the specific thing
// that corrupted round 15's payload. This removes the reason a corruption of any
// kind was fatal.
//
// The payload used to be printed as one blob under "write this to a file, byte
// for byte", 92453 bytes of it. On a mismatch the prompt asked for the whole
// thing again. So one wrong character cost two full re-transcriptions of ~30k
// tokens each, the persister ran out of room, and a run that had already spent
// 48 minutes and produced a perfectly good seed threw at the bootstrap gate.
//
// The payload now goes in parts, each with its own checksum. The agent writes
// each part as it reads it -- a few KB between reading a byte and writing it,
// instead of the whole file -- and a failed attempt is repaired by rewriting the
// two or three parts that are wrong, not all twenty-six. `qd_payload_assemble.py`
// checks every part, names the ones to redo, and refuses to produce the file at
// all unless every one of them matches; the end-to-end checksum on the assembled
// file is unchanged, so nothing `qd_persist_manifest.py` will accept has been
// loosened. See `scripts/test_qd_payload_assemble.py`.
const QD_PAYLOAD_CHUNK_BYTES = 3500;
const qdPersistPrompt = (payload, tag) => {
  const text = qdAsciiJson(qdAsciiFold(payload));
  const bytes = text.length;
  const dir = `${qdArchive.archive_dir}/.staging`;
  const file = `${dir}/payload_${tag}.json`;
  const partsDir = `${dir}/parts_${tag}`;
  const parts = [];
  for (let i = 0; i < text.length; i += QD_PAYLOAD_CHUNK_BYTES) {
    parts.push(text.slice(i, i + QD_PAYLOAD_CHUNK_BYTES));
  }
  if (!parts.length) parts.push('');
  const expectSpec = parts.map(p => `${qdFnv1a32(p)}:${p.length}`).join(',');
  const pad = (i) => ('00' + i).slice(-3);
  const blocks = parts.map((p, i) =>
    `----- PART ${pad(i)} (${p.length} bytes) -> ${partsDir}/${pad(i)}.part\n${p}`).join('\n');
  return {
    prompt:
`Do exactly these three steps. Do not summarize, reformat, reorder, pretty-print, ` +
`abbreviate, or "fix" anything in the payload, and do not write the manifest yourself.

The payload is one line of pure 7-bit ASCII JSON, ${bytes} bytes, split below into
${parts.length} parts. It is split so that a mistake costs you one part and not the whole file.
There is no character above U+007F anywhere in it and no \\uXXXX escape for one. If what
you write contains a typographic character -- an em-dash, a curly quote, an ellipsis --
you introduced it, and the checksum will say so.

1. mkdir -p ${partsDir}
2. For each PART below, in order, write its body to the file named in its header,
   exactly as given. Use a quoted heredoc (cat > FILE <<'EOF') so nothing is expanded.
   Write each part IMMEDIATELY after you read it -- do not read them all first and
   write them afterwards, and do not join them yourself. Each part is a fragment of
   one long line: it will not parse as JSON on its own, it may start or end mid-token,
   and that is correct. Never add a line break inside a part.
3. Run, from anywhere:
     python3 ${WORKFLOW_DIR}/scripts/qd_payload_assemble.py \\
       --parts-dir ${partsDir} --out ${file} \\
       --expect ${expectSpec} \\
       --expect-total ${qdFnv1a32(text)}:${bytes}
   If it exits 3 it prints which part numbers are wrong. Rewrite ONLY those parts
   from the text below and run it again. Do not edit any part to make a checksum
   pass, and do not hand-assemble ${file}.
4. Once it prints ASSEMBLED, run:
     python3 ${WORKFLOW_DIR}/scripts/qd_persist_manifest.py --payload ${file} \\
       --expect-checksum ${qdFnv1a32(text)}:${bytes}
   Return its stdout as your JSON answer, verbatim, including the "verification" block.

If step 4 exits 3, return the receipt it printed unchanged -- a reported failure is
useful and a repaired one is not. If step 3 keeps failing on the same part after two
rewrites, return {"verification": {"readable": false, "error": "part <NNN> could not be
transcribed"}, "persisted_elite_ids": [], "failures": [...]} rather than working around it.

${blocks}`,
    expect: { ...qdCellDigests(), generation: payload.manifest.generation },
  };
};
// The receipt says what is on disk in numbers the lane can check against the
// archive it holds. A mismatch is not a smaller success: the file is some other
// archive than the one this round produced, and every elite in it is suspect.
// Returns true only if the write is confirmed.
const qdVerifyPersisted = (receipt, expect, where) => {
  const v = receipt && receipt.verification;
  if (!v || v.readable !== true) {
    log(`  [qd] persistence verification MISSING (${where}): the persister returned no readable ` +
        `verification block${v && v.error ? ` (${v.error})` : ''}. Treating the write as failed.`);
    return false;
  }
  const bad = [];
  for (const k of ['cells', 'cell_keys_fnv1a32', 'elite_ids_fnv1a32', 'generation']) {
    if (v[k] !== expect[k]) bad.push(`${k}: on disk ${JSON.stringify(v[k])}, in memory ${JSON.stringify(expect[k])}`);
  }
  if (bad.length) {
    log(`  [qd] persistence verification FAILED (${where}): ${bad.join('; ')}. ` +
        `The manifest on disk is not the archive this round produced — this is the (96) ` +
        `failure mode and it is a harness fault, not a search result.`);
    return false;
  }
  if ((receipt.failures || []).length) {
    log(`  [qd] persistence reported ${receipt.failures.length} failure(s) (${where}): ` +
        receipt.failures.map(f => f.reason).join('; '));
  }
  return true;
};
// A cell that was measured, admitted and won is being taken back out of the archive.
// Unlike the bootstrap path -- which throws, because a run with no durable seed cannot
// proceed -- a round survives a persistence failure by rolling back. That is deliberate,
// but it must not be silent: this is the exact shape that makes an archive look like it
// "never filled" (44) for a reason that has nothing to do with the search.
const qdLogRollback = (entry, where) =>
  log(`  [qd] ${entry.elite_id} rolled back (${where}): persistence:not_in_receipt ` +
      `(cell ${entry.cell} ${entry.previous_elite ? 'restored to its previous elite' : 'left empty'})`);
const qdSolCaseValid = (c, contextId) => {
  if (!c || c.name !== contextId || !Number.isFinite(c.measured_ms) || c.measured_ms <= 0 ||
      !Number.isFinite(c.sol_ms) || c.sol_ms <= 0 || !Number.isFinite(c.sol_gap) || c.sol_gap <= 0 ||
      !Number.isFinite(c.remaining_headroom)) return false;
  // Finding (59). `sol_ms` is a speed-of-light LOWER BOUND on time, so
  // measured_ms >= sol_ms and sol_gap >= 1, always. A card reporting gap < 1
  // says the kernel outran the roofline, which is never a fast kernel and
  // always a broken model -- most plausibly the peak of the wrong arch (see
  // (55)), a wrong dtype peak, or an undercounted footprint. The formula check
  // below cannot catch it: such a card is perfectly self-consistent, and
  // self-consistency is not validity. Refused rather than flagged, because the
  // number the planner would receive is a NEGATIVE remaining_headroom, which
  // is not a weak signal but a meaningless one.
  if (c.sol_gap < 1 - 1e-9) return false;
  const expectedGap = c.measured_ms / c.sol_ms;
  const expectedHeadroom = 1 - 1 / c.sol_gap;
  return Math.abs(c.sol_gap - expectedGap) <= 1e-6 * Math.max(1, Math.abs(expectedGap)) &&
    Math.abs(c.remaining_headroom - expectedHeadroom) <= 1e-6 * Math.max(1, Math.abs(expectedHeadroom));
};
// (70). `qdSolCaseValid` above checks the card against ITSELF: gap >= 1, gap ==
// measured/sol, headroom == 1 - 1/gap. Every one of those holds by construction
// for a card whose `sol_ms` was computed from a fabricated or clamped bandwidth
// ceiling -- which is (59) again: self-consistency is not validity. This gate
// checks the card against the tool that was supposed to produce it, using the
// provenance block the tool emits and the schema above now carries.
//
// The substantive clause is the last one. A ceiling is only the denominator
// when the memory floor is the binding one; for a compute-bound case the
// bandwidth number does not enter `sol_ms` at all, so demanding a measured
// ceiling there would be a gate that cries wolf (53). When it IS binding and
// the ceiling is clamped or unmeasured, the case is refused rather than
// flagged: the number handed to the planner would not be weak evidence, it
// would be an overstatement pointing in the direction of more work. The fix is
// to measure that footprint -- which is what `profile_engineer.md` already
// says, and (50) is exactly why saying it there was not enough.
//
// Returns a reason string, or null when the ceiling backs the number.
const QD_SOL_CONFIDENCE = ['measured_interpolated', 'measured_scalar', 'unmeasured', 'low'];
const QD_SOL_BACKED = ['measured_interpolated', 'measured_scalar'];
const qdSolCeilingReject = (c, contextId) => {
  const label = contextId ? `${contextId}` : 'sol case';
  const k = c && c.ceiling;
  if (!k || typeof k !== 'object') {
    return `sol:ceiling_absent(${label}; the card states a sol_ms but not what bandwidth ceiling `
      + 'it divided by, so nothing here can tell a measured denominator from a guess)';
  }
  if (k.basis !== 'scalar' && k.basis !== 'footprint_table') {
    return `sol:ceiling_basis(${label}; basis=${JSON.stringify(k.basis)}, `
      + "expected 'scalar' or 'footprint_table')";
  }
  if (QD_SOL_CONFIDENCE.indexOf(k.confidence) < 0) {
    return `sol:ceiling_confidence(${label}; confidence=${JSON.stringify(k.confidence)} is not a value `
      + `qd_sol_card.py emits (${QD_SOL_CONFIDENCE.join('|')}), so it was not copied off a real card)`;
  }
  if (typeof k.extrapolated !== 'boolean') {
    return `sol:ceiling_extrapolated_flag(${label}; extrapolated=${JSON.stringify(k.extrapolated)})`;
  }
  // The tool clamps outside the measured range and stamps `low`. A block
  // claiming both an extrapolated value and a measured confidence did not come
  // off that tool, whichever half is the lie.
  if (k.extrapolated && k.confidence !== 'low') {
    return `sol:ceiling_contradiction(${label}; extrapolated=true with confidence=`
      + `${JSON.stringify(k.confidence)}; the tool stamps "low" whenever it clamps)`;
  }
  if (k.basis === 'footprint_table') {
    if (!Number.isFinite(k.footprint_bytes) || k.footprint_bytes <= 0) {
      return `sol:ceiling_footprint(${label}; a footprint_table ceiling without a positive `
        + `footprint_bytes (${JSON.stringify(k.footprint_bytes)}) names no point on the table)`;
    }
    const b = k.bracket;
    if (!Array.isArray(b) || b.length !== 2 || !b.every(x => Number.isFinite(x) && x > 0) || b[0] > b[1]) {
      return `sol:ceiling_bracket(${label}; bracket=${JSON.stringify(b)}, expected two ascending `
        + 'positive footprints from the measured table)';
    }
    // A non-clamped read lands strictly inside the bracket it reports; the tool
    // returns a degenerate [p, p] bracket only at or outside an endpoint, and
    // stamps `low` there.
    if (!k.extrapolated && (k.footprint_bytes < b[0] || k.footprint_bytes > b[1])) {
      return `sol:ceiling_bracket_excludes_footprint(${label}; ${k.footprint_bytes} is not within `
        + `[${b[0]}, ${b[1]}] yet the read is not marked extrapolated)`;
    }
  }
  // (89) item 2, the compute half, arranged to mirror the memory clause below.
  // A witness is demanded only where the compute floor is the binding one --
  // elsewhere the peak does not enter `sol_ms` and demanding one would cry
  // wolf (53). Where it IS binding and unwitnessed, the case is refused rather
  // than flagged, for the same reason as the memory clause: an unreachable
  // denominator does not weaken the headroom number, it inverts what it means.
  // Refusing only that intersection is also what keeps this from being the
  // (87) mistake -- an arch whose card carries no witness keeps every
  // memory-bound case it has, and loses exactly the cases (89) says rank
  // nothing.
  const cc = c && c.compute_ceiling;
  const computeBinding = Number.isFinite(c.compute_floor_ms) && Number.isFinite(c.memory_floor_ms)
    && c.compute_floor_ms > c.memory_floor_ms;
  if (!cc || typeof cc !== 'object' || typeof cc.witnessed !== 'boolean') {
    return `sol:compute_ceiling_absent(${label}; the card divides by a compute peak but does not say `
      + 'whether anything was ever observed to reach it, and a peak nothing reaches ranks nothing)';
  }
  if (cc.witnessed) {
    if (!String(cc.witness || '').trim()) {
      return `sol:compute_witness_unnamed(${label}; witnessed=true with no witness named, which is `
        + 'the unevidenced claim the witness field exists to replace)';
    }
    // achieved/peak > 1 means something already outran the ceiling, so the
    // ceiling is wrong -- and wrong in the direction that makes every kernel
    // look finished. `qd_sol_card.py` raises on this; a card that reached here
    // with it was not produced by that tool.
    if (!Number.isFinite(cc.attainment) || cc.attainment <= 0 || cc.attainment > 1 + 1e-9) {
      return `sol:compute_attainment(${label}; attainment=${JSON.stringify(cc.attainment)}, expected `
        + 'achieved/peak in (0, 1])';
    }
  } else if (computeBinding) {
    return `sol:unwitnessed_compute_peak(${label}; compute_floor_ms=${c.compute_floor_ms} sets sol_ms, `
      + 'and nothing has been observed to reach the peak it was divided by. A nameplate peak has '
      + 'perfect provenance and is still unreachable, so the headroom this case reports is not a '
      + 'measurement of this kernel. Witness the peak or leave the case out.)';
  }
  const memoryBinding = Number.isFinite(c.memory_floor_ms) && Number.isFinite(c.compute_floor_ms)
    && c.memory_floor_ms >= c.compute_floor_ms;
  if (memoryBinding && QD_SOL_BACKED.indexOf(k.confidence) < 0) {
    return `sol:unmeasured_denominator(${label}; memory_floor_ms=${c.memory_floor_ms} sets sol_ms, `
      + `and its bandwidth ceiling is ${JSON.stringify(k.confidence)}`
      + `${k.extrapolated ? ' (clamped, so it overstates the achievable rate)' : ''}. `
      + 'Measure the footprint; do not optimise toward a headroom this card invented.)';
  }
  return null;
};
// `rejects`, when passed, receives one entry per route that produced no cell,
// naming why. Callers that have somewhere to report should always pass it: a
// dropped route is the difference between "the search explored and found
// nothing" and "the search never ran", and those need different responses.
const qdRouteCells = (ver, rejects) => {
  const routes = Array.isArray(ver && ver.route_descriptors) ? ver.route_descriptors : [];
  const out = [];
  // Finding (59). One (cell, context) may be produced only once per candidate.
  // Two routes with the same mechanism tuple covering the same context is a
  // normal thing for a verifier to report -- and without this the admission
  // loop visited the cell twice, wrote the entry on the first pass, then
  // compared the SAME measurement against the entry it had just written. Equal
  // intervals never satisfy `lower > upper`, so the second pass fell through to
  // the near-boundary branch and filed the candidate as a challenger to itself.
  const claimed = new Map();
  const note = (route, reason) => {
    if (rejects) rejects.push({ route_id: (route && route.route_id) || '<unnamed>', reason });
  };
  for (const route of routes) {
    if (!route) { note(route, 'route:absent'); continue; }
    if (route.classification_status !== 'classified') {
      note(route, `route:classification_status=${JSON.stringify(route.classification_status)}`);
      continue;
    }
    const reason = qdDescriptorReject(route.descriptor);
    if (reason) { note(route, reason); continue; }
    const before = out.length;
    for (const contextId of route.case_ids || []) {
      const cell = qdCellId(contextId, route.descriptor);
      if (!cell) { note(route, `context:${JSON.stringify(contextId)}_not_a_known_harness_case`); continue; }
      if (claimed.has(cell)) {
        // Not a silent drop: the first route keeps the cell, and the collision
        // is reported so a verifier splitting one mechanism across two route
        // ids is visible rather than looking like a cell that never appeared.
        note(route, `cell:${cell}_already_claimed_by_${claimed.get(cell)}`);
        continue;
      }
      claimed.set(cell, (route.route_id || '<unnamed>'));
      out.push({ cell, context_id: contextId, route });
    }
    // A legal descriptor with no usable context is still a silent drop.
    if (out.length === before && !(route.case_ids || []).length) note(route, 'route:no_case_ids');
  }
  return out;
};
// Finding (59), third defect. This used to read
//   e.robust ? e.robust.median : e.geomean || 0
// which is not a default for a missing value -- it is a silently different
// QUANTITY. `robust.median` is the elite's speedup on THIS context; `geomean`
// is its speedup over the whole suite. Substituting one for the other yields a
// plausible number in the right units and the wrong scope, which is exactly the
// failure (55) named. Every one of the four sites that writes `qdArchive.cells`
// (seed, warm import, main-round admission, rollback-to-previous) refuses the
// entry unless `robust && robust.score`, so the branch was also unreachable --
// and (56): an unreachable branch is an unexercised one. Per (55), the fix for a
// wrong default is to remove it and let the absence be loud.
const qdContextScore = (cells) => {
  const totals = {}, counts = {};
  for (const e of Object.values(cells)) {
    if (!e.robust || typeof e.robust.median !== 'number') {
      throw new Error(`qd_archive cell ${e.cell || '<unnamed>'} (elite ${e.elite_id || '?'}) has no `
        + `robust.median; it reached the archive by a path that skipped the admission guard, and `
        + `scoring it against any other number would mix per-context and suite-wide speedups`);
    }
    totals[e.context_id] = (totals[e.context_id] || 0) + Math.max(0, e.robust.median);
    counts[e.context_id] = (counts[e.context_id] || 0) + 1;
  }
  const contexts = Object.keys(totals);
  return contexts.length ? contexts.reduce((s, c) => s + totals[c] / counts[c], 0) / contexts.length : 0;
};
const qdArchive = {
  version: 2, classifier_version: QD_CLASSIFIER_VERSION, generation: 0,
  cells: {}, challengers: {}, artifacts: {}, transitions: [], lineage: {}, visits: {}, capsules: {},
  preconditions: {},   // (86): arch -> [record]; flat, not descriptor-keyed

  global_best: null, coverage_stall: 0, qd_score_stall: 0, global_stall: 0,
  coverage: 0, context_coverage: 0, qd_score: 0, cost_used: 0, gpu_measurements: 0,
  archive_dir: `${EVAL_DIR}/qd_archive`,
};
// (86) writer. Arch-keyed, append-only within a run, and deliberately picky:
//
//  - `evidence` must be non-empty. A precondition with no evidence is a rumour
//    that every future run inherits and none can check, which is strictly worse
//    than rediscovering the fact -- rediscovery at least costs something
//    visible.
//  - `kind` is a closed vocabulary for the same reason the descriptor axes are:
//    a free-text kind makes the namespace unreadable at exactly the size where
//    reading it starts to matter.
//  - a second record under an existing `id` with a DIFFERENT statement does not
//    overwrite. Preconditions are supposed to be stable facts about the arch;
//    one that flips inside a single run is a contradiction to be surfaced, not
//    a value to be updated. First writer wins and the conflict is logged.
//
// Refusals are logged individually rather than folded into a count: (54), a
// labelled hole is worth as much as a pass, and a silently dropped precondition
// looks identical to an arch that needed none.
const qdRecordPreconditions = (arch, list, source) => {
  if (!Array.isArray(list) || !list.length) return { recorded: 0, refused: 0, conflicts: 0 };
  const bucket = qdArchive.preconditions[arch] || (qdArchive.preconditions[arch] = []);
  let recorded = 0, refused = 0, conflicts = 0;
  for (const raw of list) {
    const r = raw && typeof raw === 'object' ? raw : {};
    const id = String(r.id || '').trim();
    const kind = String(r.kind || '').trim();
    const statement = String(r.statement || '').trim();
    const evidence = String(r.evidence || '').trim();
    if (!id || !statement) {
      refused++; log(`  [qd precondition] refused from ${source}: record has no id or no statement`);
      continue;
    }
    if (!PRECONDITION_KINDS.includes(kind)) {
      refused++; log(`  [qd precondition] refused "${id}" from ${source}: kind "${kind}" is not one of `
        + `${PRECONDITION_KINDS.join('|')}`);
      continue;
    }
    if (!evidence) {
      refused++; log(`  [qd precondition] refused "${id}" from ${source}: no evidence. An unevidenced `
        + `precondition is inherited by every later run and checkable by none.`);
      continue;
    }
    const prior = bucket.find(x => x.id === id);
    if (prior) {
      if (prior.statement !== statement) {
        conflicts++;
        log(`  [qd precondition] CONFLICT on "${id}" from ${source}: kept "${prior.statement}", `
          + `refused "${statement}". A precondition that changes inside one run is a contradiction.`);
      }
      continue;
    }
    bucket.push({ id, kind, statement, evidence,
      established_by: String(r.established_by || source), arch, generation: qdArchive.generation });
    recorded++;
  }
  if (recorded) log(`  [qd precondition] ${recorded} recorded for ${arch} from ${source}`);
  return { recorded, refused, conflicts };
};

// (71). Three sites write `qdArchive.artifacts[hash]` -- the canonical seed, the
// warm import, and round admission -- and every one of them wrote with `=`.
// The key is a CONTENT address, so two writes under one key are by definition
// the same bytes: the second write cannot be new information about the tree,
// only a different *story* about where those bytes came from. Last-write-wins
// therefore does not update provenance, it replaces true provenance with later
// provenance. The collision is not hypothetical: a warm import of an archive
// produced from the same canonical source hashes to exactly the seed's hash,
// and the import record carries `generation: 0` with no `parent_workspace`, so
// it silently erases `parent_workspace: CANONICAL` from the root of the tree.
//
// First write wins, and a disagreement is logged rather than swallowed -- an
// archive that outlives the run (67) is read later by someone who cannot see
// which write got there last.
const qdRecordArtifact = (hash, record, where) => {
  const existing = qdArchive.artifacts[hash];
  if (!existing) {
    qdArchive.artifacts[hash] = record;
    return record;
  }
  const differs = existing.snapshot !== record.snapshot ||
    existing.generation !== record.generation ||
    (existing.parent_workspace || null) !== (record.parent_workspace || null);
  if (differs) {
    log(`  [qd] artifact ${String(hash).slice(0, 12)} keeps its first provenance ` +
        `(generation ${existing.generation}, from ${existing.parent_workspace || 'unrecorded'}); ` +
        `${where} would have rewritten it to generation ${record.generation} — same content ` +
        'address, so the later story describes the same bytes and is not an update');
  }
  return existing;
};
// How many DISTINCT pieces of work the archive holds, as opposed to how many
// cells it fills. Finding (108): an elite that wins K cells is filed K times,
// so `structural_coverage` counts occupancy and says nothing about diversity.
// The first real archive audited read as 12 cells and was 2 variants -- which
// is the literal definition of "repeating local parameter search", the failure
// this QD redesign exists to end, and it was invisible in the only window the
// planner gets. Coverage 12 with 2 variants and coverage 12 with 12 variants
// are opposite states of the search.
//
// Not a gate. A high concentration is CORRECT immediately after seed import,
// where one artifact is deliberately referenced by every route-context cell,
// and it is correct again whenever one mechanism genuinely wins broadly. It is
// reported so the planner can tell the two apart from the trend, not so
// anything can fail on it.
const qdVariantSpread = (cells) => {
  const elites = Object.values(cells);
  const perHash = new Map();
  for (const e of elites) {
    if (!e || !e.source_hash) continue;
    perHash.set(e.source_hash, (perHash.get(e.source_hash) || 0) + 1);
  }
  const counts = [...perHash.values()].sort((a, b) => b - a);
  return {
    distinct_variants: perHash.size,
    cells_filled: elites.length,
    // Share of filled cells held by the single most widely-copied variant.
    // 1.0 means every cell is the same piece of work.
    top_variant_share: elites.length ? counts[0] / elites.length : 0,
    cells_per_variant: perHash.size ? elites.length / perHash.size : 0,
  };
};
const qdSummary = () => {
  const contexts = new Set(Object.values(qdArchive.cells).map(e => e.context_id));
  return {
    version: qdArchive.version, classifier_version: qdArchive.classifier_version,
    generation: qdArchive.generation, structural_coverage: Object.keys(qdArchive.cells).length,
    variant_spread: qdVariantSpread(qdArchive.cells),
    context_coverage: QD_CONTEXT_IDS.size ? contexts.size / QD_CONTEXT_IDS.size : 0,
    qd_score: qdContextScore(qdArchive.cells), global_best: qdArchive.global_best,
    stalls: { coverage: qdArchive.coverage_stall, qd_score: qdArchive.qd_score_stall, global: qdArchive.global_stall },
    cells: Object.fromEntries(Object.entries(qdArchive.cells).map(([k, e]) => [k, {
      elite_id: e.elite_id, context_id: e.context_id, route_id: e.route_id,
      source_hash: e.source_hash, artifact: e.artifact, descriptor: e.descriptor,
      descriptor_evidence: e.descriptor_evidence || [], predicate_evidence: e.predicate_evidence || [],
      kernel_symbols: e.kernel_symbols || [], resource_signature: e.resource_signature || {},
      geomean: e.geomean, robust: e.robust, min_case: e.min_case,
      snapshot: e.snapshot, per_case: e.per_case, visits: qdArchive.visits[k] || 0,
      generation: e.generation, operator: e.operator, sol_card: e.sol_card || null,
      // The elite's own gap, not its parent's. Included here and not only in
      // the stored record because a field the archive holds but never shows the
      // planner cannot steer anything -- that unreachability, not the missing
      // value, was the expensive half of finding (44).
      sol_resolved: e.sol_resolved || null,
      // (76). Everything below here is stored on the elite at admission and,
      // before this, appeared in no agent's input ever again. `qdSummary()` is
      // the only archive window the tech lead, the round planner, the engineer
      // and the integrator get, so a field missing from this projection is
      // write-only -- durable, content-addressed, atomically persisted, and
      // incapable of influencing a decision. `strategy_capsule` is the worst
      // case of it: it is the mechanism record, i.e. the thing the archive
      // exists to carry across generations, and nothing could read it back.
      strategy_capsule: e.strategy_capsule || {},
      parent_elite_id: e.parent_elite_id || null,
      parent_elite_ids: e.parent_elite_ids || [],
      cost: e.cost == null ? null : e.cost,
    }])),
    challengers: Object.fromEntries(Object.entries(qdArchive.challengers).map(([k, e]) => [k, {
      elite_id: e.elite_id, context_id: e.context_id, source_hash: e.source_hash,
      robust: e.robust, generation: e.generation, remeasurement_required: true,
    }])),
    // (76). The capsule ledger was read in exactly one place -- as a *count*,
    // to block a third identical local mutation -- and was surfaced to no
    // agent. Exposing it is what turns the archive from a search-state store
    // into a mechanism memory: the planner can now see that
    // `route|context|mechanism` was tried, by which operator, what was
    // expected, and what was actually measured. Capped per key so a long run
    // cannot crowd the rest of the summary out of the prompt; the count is
    // reported separately so a truncated list never reads as a complete one.
    capsules: Object.fromEntries(Object.entries(qdArchive.capsules).map(([k, v]) => [k, {
      attempts: v.length, unimproved_local: v.filter(x =>
        x.operator === 'local_mutation' && x.improved !== true && x.regime_changed !== true).length,
      recent: v.slice(-4),
    }])),
    // (86). Same block as the capsule ledger, because that is the read path
    // (78)/(83) confirmed is actually live. `arch` is repeated on the outside so
    // a reader cannot mistake these for facts about the route they are planning.
    preconditions: { arch: QD_ARCH, records: qdArchive.preconditions[QD_ARCH] || [] },
    // (76). Named in tech_lead.md, and until now absent from the only input
    // that could have carried it.
    lineage: qdArchive.lineage,
    recent_transitions: qdArchive.transitions.slice(-32),
  };
};
const qdRecompute = () => {
  const elites = Object.values(qdArchive.cells);
  const contexts = new Set(elites.map(e => e.context_id));
  qdArchive.coverage = elites.filter(e => qdCoverageEligible(e.descriptor)).length;
  qdArchive.context_coverage = QD_CONTEXT_IDS.size ? contexts.size / QD_CONTEXT_IDS.size : 0;
  qdArchive.qd_score = qdContextScore(qdArchive.cells);
  qdArchive.global_best = elites.sort((a, b) => b.geomean - a.geomean)[0] || null;
};

// A sparse QD archive still needs real executable parents on a fresh run. Classify and hash the frozen
// canonical seed deterministically, then materialize one content-addressed artifact referenced by all of
// its route-context cells. This is not an optimization result and all baseline scores are exactly 1.
if (QD_ENABLED) {
  phase('QD Import');
  const seed = await agentT(
    roleAgent('verify_engineer', 'classify_qd_seed',
      'Policy-scan, hash, and classify the frozen canonical seed by observed routes; do not optimize it.', {
        CANONICAL, VERIFY_DIR: `${EVAL_DIR}/qd_archive/seed_verify`, SKILL_DIR: WORKFLOW_DIR,
        TASK_DIR: KERNEL_PATH_ORIG,
        COMMANDMENT, BASELINE_PER_CASE, QD_ARCH, QD_DTYPE,
        QD_EVIDENCE_HELPER: `${WORKFLOW_DIR}/scripts/qd_v2.py`,
      }),
    { phase: 'QD Import', label: 'classify canonical QD seed', schema: QD_SEED_SCHEMA });
  if (!seed || seed.policy_pass !== true || !says(seed.status, 'verified') ||
      !qdHashValid(seed.source_hash)) {
    throw new Error('QD bootstrap failed closed: canonical seed lacks a verified policy-clean source hash');
  }
  // (69). Deliberately a throw and not a `return null`: this is the root of the
  // archive, so there is no later stage that would catch it and no partial run
  // worth continuing with an unbacked policy claim underneath it.
  const seedPolicyReason = qdPolicyReject(seed, 'canonical seed');
  if (seedPolicyReason) {
    throw new Error(`QD bootstrap failed closed: ${seedPolicyReason}`);
  }
  // (86). Recorded before the cells, because a precondition is what made the
  // seed buildable at all -- it is not a result of the seed, it is upstream of
  // it. Recording it after admission would put it on the wrong side of the
  // throw above.
  qdRecordPreconditions(QD_ARCH, seed.preconditions, 'canonical seed bootstrap');
  const artifactSnapshot = `${qdArchive.archive_dir}/artifacts/${seed.source_hash}/workspace`;
  qdRecordArtifact(seed.source_hash, {
    source_hash: seed.source_hash, snapshot: artifactSnapshot, parent_workspace: CANONICAL, generation: 0,
  }, 'canonical seed');
  const seedAdmissions = [];
  const baselineRows = (BASELINE_PER_CASE || []).map(c => {
    const name = c.name || c.test_case_id;
    const latency = QD_BASELINE_MS.get(name);
    return { ...c, name, baseline_ms: latency, optimized_ms: latency, speedup: 1 };
  });
  const seedRejects = [];
  for (const rc of qdRouteCells({ route_descriptors: seed.route_descriptors }, seedRejects)) {
    const eliteId = `seed_${rc.context_id}_${seed.source_hash.slice(0, 12)}`;
    const entry = {
      elite_id: eliteId, cell: rc.cell, context_id: rc.context_id, route_id: rc.route.route_id,
      descriptor: rc.route.descriptor, descriptor_evidence: rc.route.descriptor_evidence || [],
      predicate_evidence: rc.route.predicate_evidence || [], kernel_symbols: rc.route.kernel_symbols || [],
      resource_signature: rc.route.resource_signature || {}, source_hash: seed.source_hash,
      artifact: seed.source_hash, geomean: 1, geomean_unweighted: 1, weighted: HAS_WORKLOAD ? 1 : null,
      arithmetic: 1, robust: qdSeedRobust(rc.context_id), suite_robust: qdSeedSuiteRobust(), min_case: 1,
      per_case: baselineRows, robust_baseline_frame: QD_ROBUST_BASELINE_FRAME,
      parent_workspace: CANONICAL, snapshot: artifactSnapshot,
      generation: 0, cost: 0, parent_elite_id: 'canonical_seed', parent_elite_ids: [],
      operator: 'bootstrap', materialization: 'source_snapshot', strategy_capsule: {}, previous_elite: null,
    };
    qdArchive.cells[rc.cell] = entry;
    qdArchive.lineage[eliteId] = ['canonical_seed'];
    seedAdmissions.push(entry);
  }
  // Report dropped routes even when enough survived to bootstrap: a partially
  // filled seed is the shape finding (44) wore, and it is indistinguishable
  // from a healthy one unless the drops are named.
  if (seedRejects.length) {
    log(`  [qd seed] ${seedRejects.length} route(s) produced no cell: ` +
        seedRejects.map(r => `${r.route_id} -> ${r.reason}`).join('; '));
  }
  if (!seedAdmissions.length) {
    // Finding (44): this used to say only "no classifiable route-context
    // cells", which is true of a vocabulary bug, a mis-classifying verifier and
    // an genuinely unclassifiable kernel alike -- and the vocabulary bug went
    // undiagnosed for weeks behind exactly this sentence. The reasons are the
    // difference between a five-minute fix and a search that looks empty.
    const why = seedRejects.length
      ? seedRejects.map(r => `${r.route_id} -> ${r.reason}`).join('; ')
      : `the verifier returned ${(seed.route_descriptors || []).length} route descriptor(s) and none was even considered`;
    throw new Error(`QD bootstrap failed closed: canonical seed had no classifiable route-context cells (${why})`);
  }
  qdRecompute();
  // The seed has no patch: it IS the baseline, so `qd_persist_manifest.py`
  // materializes it by copying the frozen tree and hashing it, and there is
  // nothing to reapply. Everything else about the write is the same code path
  // the rounds use, which is the point -- a bootstrap-only serializer is a
  // second implementation, and (63) is what a second implementation costs.
  const seedPersist = qdPersistPrompt(qdPersistPayload(seedAdmissions), 'seed');
  const persistedSeed = await agentT(
    roleAgent('tech_lead', 'update_qd_archive', seedPersist.prompt, {
      EVAL_DIR, CANONICAL, QD_ARCHIVE_DIR: qdArchive.archive_dir,
      QD_PERSIST_SCRIPT: `${WORKFLOW_DIR}/scripts/qd_persist_manifest.py`,
    }),
    { phase: 'QD Import', label: 'persist canonical QD seed', schema: QD_ARCHIVE_SCHEMA });
  const seedPersistedIds = new Set(persistedSeed && persistedSeed.persisted_elite_ids || []);
  if (!qdVerifyPersisted(persistedSeed, seedPersist.expect, 'canonical seed') ||
      !seedAdmissions.every(e => seedPersistedIds.has(e.elite_id))) {
    throw new Error('QD bootstrap failed closed: canonical artifact/cell manifest was not durably persisted');
  }
}

// QD warm-start is re-verification, never deserialization of trusted fitness. The Director has already
// copied and policy-scanned source-only snapshots into this run. Verify each one against the current
// frozen baseline/harness before it can enter a live cell or be selected as a parent.
if (QD_ENABLED && QD_STATE_DIR) {
  const importRoot = `${EVAL_DIR}/qd_archive/imported/`;
  const withinImportRoot = (p) => typeof p === 'string' && p.startsWith(importRoot) &&
    !p.slice(importRoot.length).split('/').includes('..');
  const importSourceOK = setup.qd_import_status === 'ready' && setup.qd_import_source === QD_STATE_DIR &&
    setup.qd_import_manifest;
  const classifierMatches = importSourceOK &&
    setup.qd_import_manifest.classifier_version === QD_CLASSIFIER_VERSION;
  const reclassifyMode = !!(importSourceOK && QD_RECLASSIFY);
  const manifestOK = !!(importSourceOK && (classifierMatches || reclassifyMode));
  const imported = manifestOK && Array.isArray(setup.qd_import_candidates)
    ? setup.qd_import_candidates.filter(c => {
        if (!c || c.policy_pass !== true || !qdHashValid(c.source_hash) || !withinImportRoot(c.snapshot)) return false;
        if (reclassifyMode || c.needs_reclassification === true) return true;
        const routes = Array.isArray(c.route_descriptors) ? c.route_descriptors : [];
        return routes.length > 0 && routes.every(route => route && (route.case_ids || []).every(contextId =>
          qdCellId(contextId, route.descriptor)));
      }) : [];
  if (!manifestOK) {
    log(`QD warm-start rejected by deterministic manifest/source/classifier checks; keeping canonical bootstrap only (${QD_STATE_DIR}).`);
  } else if (!imported.length) {
    log(`QD warm-start supplied no policy-clean source artifacts; keeping canonical bootstrap only.`);
  } else {
    phase('QD Import');
    const checked = (await parallel(imported.map((candidate, i) => () => agentT(
      roleAgent('verify_engineer', 'verify_imported_qd',
        'Independently re-verify and classify this imported source artifact under the current frozen harness.', {
          TASK_DIR: KERNEL_PATH_ORIG,
          SEARCH_STRATEGY, IMPORTED_SNAPSHOT: candidate.snapshot,
          IMPORTED_ELITE_ID: candidate.elite_id || `import_${candidate.source_hash.slice(0, 12)}`,
          IMPORTED_SOURCE_HASH: candidate.source_hash,
          SOURCE_CELL: reclassifyMode || candidate.needs_reclassification ? '' : (candidate.cell || ''),
          QD_ROUTE_DESCRIPTOR_PROPOSALS: reclassifyMode || candidate.needs_reclassification
            ? [] : (candidate.route_descriptors || []),
          QD_RECLASSIFY: reclassifyMode ? '1' : '0', QD_REPEAT_MEASUREMENTS: 3,
          EXPECTED_PARENT_SOURCE_HASH: candidate.source_hash,
          QD_EVIDENCE_HELPER: `${WORKFLOW_DIR}/scripts/qd_v2.py`, QD_ARCH, QD_DTYPE,
          QD_CELL_GUARDRAIL, QD_CANONICAL_GUARDRAIL,
          VERIFY_DIR: `${EVAL_DIR}/qd_archive/import_verify/${i}_${candidate.source_hash.slice(0, 12)}`,
          GPU_ID: GPU_MODE === 'pin' ? GPU_LIST[i % GPU_LIST.length] : GPU_POOL,
          ...GPU_LOCK_INPUT, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
          REQUIRE_CURRENT_HARNESS_REVERIFY: '1', HISTORICAL_SCORE_TRUSTED: 'false',
          ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
          ...(REQUIRE_GRAPH_CAPTURE ? { REQUIRE_GRAPH_CAPTURE: '1' } : {}),
        }),
      { phase: 'QD Import', label: `verify import ${candidate.source_hash.slice(0, 12)}`, schema: QD_IMPORT_VERIFY_SCHEMA }
    )))).filter(Boolean);

    const admissions = [];
    for (const ver of checked) {
      const proposed = imported.find(c => c.source_hash === (ver.imported_source_hash || ver.seed_source_hash));
      const rejects = [];
      const routeCells = qdRouteCells(ver, rejects);
      const expectedCells = new Set((proposed && proposed.route_descriptors || []).flatMap(route =>
        (route.case_ids || []).map(contextId => qdCellId(contextId, route.descriptor)).filter(Boolean)));
      // An import that re-verified clean but classified into nothing is a
      // vocabulary problem on the importing side, not a bad snapshot; without
      // this line it looks identical to a snapshot that failed its gates.
      if (rejects.length && !routeCells.length) {
        log(`  [qd import] ${(ver.imported_elite_id || ver.source_hash || 'snapshot')} yielded no cell: ` +
            rejects.map(x => `${x.route_id} -> ${x.reason}`).join('; '));
      }
      // Finding (69). The imported snapshot is explicitly untrusted and is being
      // re-verified from scratch, so its post-build receipt is checked like any
      // other; `continue` matches the surrounding style but the reason is named
      // rather than swallowed (60).
      const importPolicyReason = ver
        ? (qdPolicyReject(ver, 'qd import verify') || qdTwinReject(ver, 'qd import verify'))
        : null;
      if (importPolicyReason) {
        log(`  QD import ${proposed ? proposed.elite_id : '?'} rejected: ${importPolicyReason}`);
        continue;
      }
      if (!proposed || ver.policy_pass !== true || !says(ver.status, 'verified') ||
          !says(ver.correctness, 'pass') || ver.seed_source_hash !== proposed.source_hash ||
          !qdHashValid(ver.source_hash) || !routeCells.length ||
          (!reclassifyMode && proposed.needs_reclassification !== true && expectedCells.size &&
            !routeCells.some(x => expectedCells.has(x.cell)))) continue;
      const admission = qdAdmissionCheck(ver, routeCells);
      if (admission.reason) {
        log(`  [qd import] ${(ver.imported_elite_id || ver.source_hash || 'snapshot')} ` +
            `re-verified but not admitted: ${admission.reason}`);
        continue;
      }
      const suiteRobust = admission.suiteRobust;
      for (const rc of routeCells) {
        const robust = qdCaseRobust(ver, rc.context_id);
        const incumbent = qdArchive.cells[rc.cell];
        if (!robust || !robust.score) continue;
        const replacement = !incumbent || robust.lower > incumbent.robust.upper;
        const nearBoundary = !!(incumbent && !replacement && robust.upper > incumbent.robust.lower);
        const entry = {
          elite_id: `import_${proposed.source_hash.slice(0, 12)}_${rc.context_id}`, cell: rc.cell,
          context_id: rc.context_id, route_id: rc.route.route_id,
          descriptor: rc.route.descriptor, descriptor_evidence: rc.route.descriptor_evidence || [],
          predicate_evidence: rc.route.predicate_evidence || [], kernel_symbols: rc.route.kernel_symbols || [],
          resource_signature: rc.route.resource_signature || {},
          source_hash: ver.source_hash, artifact: ver.source_hash,
          geomean: primSpeedup(ver), geomean_unweighted: ver.verified_geomean,
          weighted: ver.verified_weighted != null ? ver.verified_weighted : null,
          arithmetic: ver.verified_arithmetic || ver.verified_geomean,
          robust, suite_robust: suiteRobust, min_case: admission.minCase, per_case: ver.per_case || [],
          robust_baseline_frame: QD_ROBUST_BASELINE_FRAME,
          snapshot: proposed.snapshot, parent_workspace: proposed.snapshot, patch: null,
          generation: 0, cost: 0, operator: 'historical_import',
          parent_elite_id: 'imported_archive', parent_elite_ids: [], previous_elite: incumbent || null,
          materialization: 'source_snapshot', historical_geomean: proposed.historical_geomean,
          historical_robust: proposed.historical_robust || null,
        };
        qdArchive.transitions.push({ generation: 0, candidate_id: entry.elite_id,
          from_cell: null, target_cell: proposed.cell || null, actual_cell: rc.cell,
          target_hit: !!proposed.cell && proposed.cell === rc.cell, operator: 'historical_import',
          fitness_delta: robust.score - (incumbent ? incumbent.robust.score : 0),
          accepted: replacement, cost: 0 });
        if (replacement) {
          qdArchive.cells[rc.cell] = entry;
          qdRecordArtifact(ver.source_hash, {
            source_hash: ver.source_hash, snapshot: proposed.snapshot, generation: 0,
          }, 'warm import');
          qdArchive.lineage[entry.elite_id] = ['imported_archive'];
          admissions.push(entry);
        } else if (nearBoundary) {
          const current = qdArchive.challengers[rc.cell];
          if (!current || robust.upper > current.robust.upper) qdArchive.challengers[rc.cell] = entry;
        }
      }
    }
    let persistedImportIds = [];
    if (admissions.length) {
      const importPersist = qdPersistPrompt(qdPersistPayload(admissions), 'import');
      const persistedImports = await agentT(
        roleAgent('tech_lead', 'update_qd_archive', importPersist.prompt, {
          EVAL_DIR, CANONICAL, QD_ARCHIVE_DIR: qdArchive.archive_dir,
          QD_PERSIST_SCRIPT: `${WORKFLOW_DIR}/scripts/qd_persist_manifest.py`,
        }),
        { phase: 'QD Import', label: 'persist verified QD imports', schema: QD_ARCHIVE_SCHEMA });
      // An unverified write rolls the whole batch back rather than part of it:
      // if the manifest on disk is not this archive, no individual receipt line
      // from the same run is worth honouring either.
      persistedImportIds = qdVerifyPersisted(persistedImports, importPersist.expect, 'warm import')
        ? qdPersistenceReceipt(persistedImports, admissions, 'warm import') : [];
      for (const entry of admissions) {
        if (persistedImportIds.includes(entry.elite_id)) {
          entry.snapshot = `${qdArchive.archive_dir}/artifacts/${entry.source_hash}/workspace`;
          const artifact = qdArchive.artifacts[entry.source_hash];
          if (artifact) artifact.snapshot = entry.snapshot;
          continue;
        }
        qdLogRollback(entry, 'warm import');
        if (qdArchive.cells[entry.cell] && qdArchive.cells[entry.cell].elite_id === entry.elite_id) {
          if (entry.previous_elite) qdArchive.cells[entry.cell] = entry.previous_elite;
          else delete qdArchive.cells[entry.cell];
        }
        if (!Object.values(qdArchive.cells).some(e => e.source_hash === entry.source_hash)) {
          delete qdArchive.artifacts[entry.source_hash];
        }
      }
    }
    qdRecompute();
    log(`QD warm-start: ${persistedImportIds.length} route-context cell(s) from ${imported.length} imported candidate(s) passed current verification and durable persistence.`);
  }
}

// DEEP-MODE resume: restore cumulative speedup + insight/ledger history from the prior wave so this
// continuation builds ON the cumulative best (canonical was already seeded from STATE_DIR/best by the
// director) and does not re-explore dead directions. No-op on a fresh run (prior_state undefined).
if (setup.resumed && setup.prior_state) {
  const ps = setup.prior_state;
  if (Number.isFinite(ps.cumulative) && ps.cumulative > cumulative) cumulative = ps.cumulative;
  if (Array.isArray(ps.insights)) history.insights = ps.insights;
  if (Array.isArray(ps.ledger)) history.ledger = ps.ledger;
  if (ps.bottleneck_now) history.bottleneck_now = ps.bottleneck_now;
  if (Array.isArray(ps.best_per_case) && ps.best_per_case.length) bestPerCase = ps.best_per_case;
  log(`RESUMED from STATE_DIR: cumulative=${cumulative.toFixed(3)}x, ${history.insights.length} insights, ${history.ledger.length} ledger entries carried forward.`);
}

while (dispatched < BUDGET && (QD_ENABLED || noImprove < MAX_NO_IMPROVE) &&
  (!QD_ENABLED || Math.min(qdArchive.coverage_stall, qdArchive.qd_score_stall, qdArchive.global_stall) < MAX_NO_IMPROVE)) {
  round++;
  if (QD_ENABLED) { qdArchive.generation = round; }
  const remaining = BUDGET - dispatched;
  phase('Optimize');

  // --- (a) Plan the round ------------------------------------------------
  // QD selection is deliberately isolated from SOL: archive fitness/frontier/UCB/exploration choose
  // route cells first. Only those frozen selections are profiled and handed to mutation planning.
  let plan;
  let qdSelections = [];
  let qdSolCards = [];
  const qdCommonInputs = QD_ENABLED ? {
    SEARCH_STRATEGY, QD_ARCHIVE: qdSummary(), QD_ARCHIVE_DIR: qdArchive.archive_dir,
    QD_ARCH: QD_ARCH, QD_DTYPE: QD_DTYPE,
    QD_EVIDENCE_HELPER: `${WORKFLOW_DIR}/scripts/qd_v2.py`,
    QD_OPERATOR_COSTS: { local_mutation: 1, directed_transition: 1, parameter_tuning: 1,
      deep_mutation: DEEP_COST, semantic_crossover: DEEP_COST },
    QD_DESCRIPTOR_AXES: {
      context_id: 'exact stable case_id from BASELINE_PER_CASE; predicates are metadata, never new contexts',
      compute_primitive: ['valu', 'rocwmma', 'native_mfma'],
      wave_schedule: ['independent', 'symmetric_interleave', 'symmetric_pingpong', 'asymmetric_producer_consumer'],
      k_pipeline: ['direct_global', 'lds_single', 'lds_reg_prefetch', 'lds_pingpong', 'lds_deep_single', 'lds_multistage'],
      decomposition: ['tile_grid', 'persistent_output', 'split_k', 'stream_k'],
      output_path: ['direct_store', 'lds_staged_store', 'atomic_fixup', 'workspace_fixup'],
      rasterization: ['linear', 'grouped_m', 'xcd_remapped_grouped'],
      plan_binding: ['static', 'runtime_tuned'],
    },
  } : {};
  const planningInputs = {
    EVAL_DIR, ROUND: round, BUDGET_REMAINING: remaining, CUMULATIVE_SPEEDUP: cumulative,
    BASELINE_GEOMEAN_MS, SKILL_DIR: WORKFLOW_DIR, PROFILE_SUMMARY: profileSummary,
    CURRENT_BEST_PER_CASE: bestPerCase, HISTORY: history,
    KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE, KK_REFS, ...qdCommonInputs, ...KB_INPUTS,
  };
  if (QD_ENABLED) {
    const selection = await agentT(
      roleAgent('tech_lead', 'select_qd_parents',
        'Select route-cell parents using QD signals only. SOL is unavailable in this phase.', planningInputs),
      { phase: 'Optimize', label: `tech_lead:qd-select r${round}`, schema: QD_SELECTION_SCHEMA });
    if (selection && !selection.stop) {
      qdSelections = (selection.selections || []).map(s => {
        const parent = qdArchive.cells[s.selected_cell];
        return parent && parent.elite_id === s.parent_elite_id && parent.context_id === s.context_id &&
          parent.source_hash === s.parent_source_hash && parent.snapshot && parent.artifact
          ? { ...s, parent } : null;
      }).filter(Boolean);
    }
    if (qdSelections.length) {
      const cardResults = await parallel(qdSelections.map((s, i) => () => agentT(
        roleAgent('profile_engineer', 'selected_cell_sol',
          'Hash-check and build a minimal SOL card for this already-selected route cell. Do not rank or replace parents.', {
            WORKSPACE: s.parent.snapshot, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR,
            GPU_ID: GPU_MODE === 'pin' ? GPU_LIST[i % GPU_LIST.length] : GPU_POOL,
            ...GPU_LOCK_INPUT, COMMANDMENT, BASELINE_PER_CASE,
            SELECTED_CELL: s.selected_cell, CONTEXT_ID: s.context_id,
            PARENT_ELITE_ID: s.parent_elite_id, EXPECTED_SOURCE_HASH: s.parent_source_hash,
            PARENT_PER_CASE: s.parent.per_case, ROUTE_DESCRIPTOR: s.parent.descriptor,
            QD_ARCH, QD_DTYPE, QD_SOL_CALIBRATION, QD_SOL_CALIBRATION_VERSION,
            QD_EVIDENCE_HELPER: `${WORKFLOW_DIR}/scripts/qd_v2.py`,
          }),
        { phase: 'Optimize', label: `sol ${s.context_id}`, schema: QD_SOL_CARD_SCHEMA }
      ).then(card => {
        if (!card || card.selected_cell !== s.selected_cell || card.context_id !== s.context_id ||
            card.parent_elite_id !== s.parent_elite_id ||
            card.parent_source_hash !== s.parent_source_hash ||
            card.calibration_version !== QD_SOL_CALIBRATION_VERSION) return null;
        const selfConsistent = (card.cases || []).filter(c => qdSolCaseValid(c, s.context_id));
        if (!selfConsistent.length) return null;
        // (70). First of the two sites. Cases whose ceiling does not back their
        // denominator are dropped individually, not silently averaged in, and
        // the card is refused outright only when none survives -- so a refusal
        // here names itself (60) instead of showing up a round later as a cell
        // that mysteriously never filled (44).
        const cases = selfConsistent.filter(c => {
          const why = qdSolCeilingReject(c, s.context_id);
          if (why) log(`  [qd] sol case dropped: ${why}`);
          return !why;
        });
        return cases.length ? { ...card, cases } : null;
      })));
      qdSolCards = cardResults.filter(Boolean);
      const cardKeys = new Set(qdSolCards.map(c => `${c.selected_cell}|${c.context_id}|${c.parent_source_hash}`));
      qdSelections = qdSelections.filter(s => cardKeys.has(
        `${s.selected_cell}|${s.context_id}|${s.parent_source_hash}`));
    }
    plan = qdSelections.length ? await agentT(
      roleAgent('tech_lead', 'plan_qd_mutations',
        'Plan mutations only for the frozen QD selections, using SOL for within-cell steering only.', {
          ...planningInputs, QD_SELECTIONS: qdSelections.map(s => ({
            id: s.id, parent_elite_id: s.parent_elite_id, selected_cell: s.selected_cell,
            context_id: s.context_id, parent_source_hash: s.parent_source_hash,
          })), CELL_SOL_CARDS: qdSolCards,
        }),
      { phase: 'Optimize', label: `tech_lead:qd-mutate r${round}`, schema: PLAN_SCHEMA })
      : { stop: true, reasoning: selection ? selection.reasoning : 'no valid hash-checked QD parent', directions: [] };
  } else {
    plan = await agentT(
      roleAgent('tech_lead', 'plan_round', 'Decide this round\'s orthogonal directions (or stop).', planningInputs),
      { phase: 'Optimize', label: `tech_lead:plan r${round}`, schema: PLAN_SCHEMA });
  }

  if (!plan || plan.stop || !plan.directions || plan.directions.length === 0) {
    log(`Round ${round}: TechLead chose to stop. ${plan ? plan.reasoning || '' : ''}`);
    break;
  }

  let directions = plan.directions.map((d, i) => {
    const gate = QD_ENABLED
      ? qdParentReject(d, qdSelections, qdSolCards, qdArchive.cells)
      : { selection: null, parent: null, card: null, donors: [] };
    if (gate.reason) {
      log(`Round ${round} dir ${d.id || i}: rejected stale/missing/same-source/illegal-transition `
        + `QD parent; no canonical fallback -- ${gate.reason}.`);
      return null;
    }
    const { selection, parent, card, donors } = gate;
    // The rounds-law gate, re-checked here rather than trusted from the prompt.
    // Dropping the direction rather than warning: (24) priced both crossings at
    // 27-43%, so building one spends a full build-and-verify cycle to re-learn
    // a closed result. The reason is named because a direction that vanishes
    // without one is indistinguishable from a planner that returned nothing.
    const residencyReason = QD_ENABLED ? qdResidencyReject(d.residency_receipt) : null;
    if (residencyReason) {
      log(`Round ${round} dir ${d.id || i}: dropped by the residency gate -- ${residencyReason}.`);
      return null;
    }
    // Finding (68). The same treatment for the route-priority gate, except that
    // this one is per-entry: a mixed list is not a refusal, so the closed
    // entries are removed and the direction proceeds against the rest. Only an
    // absent/inconsistent receipt or a wholly-closed target list costs the
    // direction. `|| [d.context_id]` mirrors the TARGET_CASES fallback the
    // engineer prompt is built with, so the set checked is the set built.
    let priorityCases = d.target_cases;
    if (QD_ENABLED) {
      const priority = qdPriorityFilter(d.priority_receipt, d.target_cases || [d.context_id]);
      if (priority.reason) {
        log(`Round ${round} dir ${d.id || i}: dropped by the route-priority gate -- ${priority.reason}.`);
        return null;
      }
      const dropped = (d.target_cases || [d.context_id]).filter(c => !priority.cases.includes(c));
      if (dropped.length) {
        log(`Round ${round} dir ${d.id || i}: route-priority dropped closed target(s) ${dropped.join(', ')}; ` +
            `aiming at ${priority.cases.join(', ')}.`);
      }
      priorityCases = priority.cases;
    }
    const op = QD_ENABLED ? (d.operator || (d.specialty === 'deep_explore' ? 'deep_mutation' : 'local_mutation')) : '';
    // (80): the capsule ledger is keyed on the mechanism, so an order-sensitive
    // key would file one mechanism under two names and quietly defeat the
    // "already failed twice on this route" block that (76) exists to create.
    const mechanism = QD_ENABLED && d.target_descriptor
      ? (qdDescriptorKey(d.target_descriptor) || '') : '';
    // (76). This used to be keyed on `parent.source_hash`. The key is the
    // identity of the thing being repeated, and what gets repeated is a
    // *direction on a route*, not a direction on one particular parent build.
    // Keying on the parent hash meant the record of what had already failed
    // twice on a route was discarded the instant that cell's elite was
    // replaced -- which is precisely the moment the planner, now looking at a
    // fresh parent, proposes the same direction again. Route + context +
    // mechanism survives elite replacement, which is the whole point of
    // keeping the negative result.
    const capsuleKey = QD_ENABLED
      ? `${parent.route_id || parent.context_id}|${parent.context_id}|${mechanism}` : '';
    const priorLocal = QD_ENABLED ? (qdArchive.capsules[capsuleKey] || []).filter(x =>
      x.operator === 'local_mutation' && x.improved !== true && x.regime_changed !== true).length : 0;
    if (QD_ENABLED && priorLocal >= 2 && (op === 'local_mutation' || op === 'parameter_tuning')) {
      log(`Round ${round} dir ${d.id || i}: blocked repeated local mutation with no fitness/regime shift.`);
      return null;
    }
    return {
      // (68): the filtered list, so the engineer is aimed at it rather than
      // merely told about it. A gate whose result is logged and then discarded
      // is a comment.
      ...d, target_cases: priorityCases, operator: op,
      selected_cell: selection ? selection.selected_cell : d.selected_cell,
      context_id: selection ? selection.context_id : d.context_id,
      parent_source_hash: parent ? parent.source_hash : d.parent_source_hash,
      sol_card: card, capsule_key: capsuleKey, donors,
      idx: i, id: d.id || `r${round}_d${i}`,
      gpu_id: GPU_MODE === 'pin' ? GPU_LIST[i % GPU_LIST.length] : GPU_POOL,
      out_dir: `${EVAL_DIR}/round_${round}/engineer_${i}`,
      seed_dir: parent ? parent.snapshot : CANONICAL,
      parent_elite: parent,
      cost: QD_ENABLED && (op === 'deep_mutation' || op === 'semantic_crossover') ? DEEP_COST :
        (d.specialty === 'deep_explore' ? DEEP_COST : 1),
    };
  }).filter(Boolean);
  // deep_explore and semantic crossover are DEDICATED-ROUND heavyweight mandates.
  const deepDir = directions.find(d => d.specialty === 'deep_explore' || d.operator === 'deep_mutation' || d.operator === 'semantic_crossover');
  if (deepDir) {
    // The four gates above each name the direction they drop and why. These
    // last two did not, and the comment at the residency gate -- "a direction
    // that vanishes without one is indistinguishable from a planner that
    // returned nothing" -- applies to them verbatim. Nothing downstream can
    // recover the difference either: the lane has no filesystem, `round_N/`
    // only ever contains the engineers that were dispatched, and the summary
    // line below prints the surviving count. So a round that planned four and
    // ran one reads on disk exactly like a round that planned one.
    const shed = directions.filter(d => d !== deepDir);
    if (shed.length) {
      log(`Round ${round}: dedicated ${deepDir.operator || deepDir.specialty} round -- `
        + `shedding ${shed.length} co-planned direction(s) ${shed.map(d => d.id).join(', ')}. `
        + `Not a rejection: they were never evaluated.`);
    }
    directions = [deepDir];
  }
  const affordable = [];
  let plannedCost = 0;
  for (const d of directions) {
    if (plannedCost + d.cost > remaining) {
      log(`Round ${round} dir ${d.id}: unaffordable -- cost ${d.cost}, `
        + `${remaining - plannedCost} of ${remaining} left this round. Not a rejection.`);
      continue;
    }
    affordable.push(d); plannedCost += d.cost;
  }
  directions = affordable;
  if (!directions.length) {
    log(`Round ${round}: no planned QD direction fits remaining budget ${remaining}.`);
    break;
  }
  const roundCost = directions.reduce((s, d) => s + (d.specialty === 'deep_explore' ? DEEP_COST : 1), 0);
  const chargedRoundCost = QD_ENABLED ? directions.reduce((s, d) => s + d.cost, 0) : roundCost;
  dispatched += chargedRoundCost;
  if (QD_ENABLED) qdArchive.cost_used = dispatched;
  // Planned-vs-dispatched on one line. `directions.length` alone is the count
  // that survived six independent drop sites, and reading it as "what the
  // planner decided to do" overstates the planner and hides the gates.
  log(`Round ${round}: planned ${plan.directions.length} -> dispatched ${directions.length} `
    + `direction(s) [${directions.map(d => d.specialty).join(', ')}], cost ${chargedRoundCost}, `
    + `budget ${dispatched}/${BUDGET}`);

  // --- (b,c) Optimize -> Verify, pipelined per direction ----------------
  const results = await pipeline(
    directions,
    (d) => {
      const isCrossover = QD_ENABLED && d.operator === 'semantic_crossover';
      const isDeep = d.specialty === 'deep_explore' || (QD_ENABLED && d.operator === 'deep_mutation') || isCrossover;
      // deep/crossover reads broad-authority contracts; specialists read engineer.md.
      const readLine = isCrossover
        ? `Then Read ${WORKFLOW_DIR}/roles/integrator.md (QD semantic crossover) and ${WORKFLOW_DIR}/roles/deep_engineer.md. ` +
          `Use the two parent elite snapshots named in PARENT_ELITE/PARENT_ELITE_IDS as semantic references; ` +
          `do not stack their ancestry-relative patches. Reconstruct one coherent implementation, fully measure it, ` +
          `and return the engineer result schema with its actual route_descriptors.`
        : isDeep
        ? `Then Read ${WORKFLOW_DIR}/roles/deep_engineer.md and ALL knowledge files under ${WORKFLOW_DIR}/knowledge/ ` +
          `(you have broad authority — combine algorithm + memory + compute + host_runtime levers in one ` +
          `coherent rewrite), and follow them. You MAY edit ANY modifiable source (kernel + Python wrapper ` +
          `+ C++ binding), not just focus_files. Run your OWN multi-iteration measure→(self-)profile→rewrite ` +
          `loop and push to the TARGET; keep the best correct version.`
        : `Then Read ${WORKFLOW_DIR}/roles/engineer.md and ${WORKFLOW_DIR}/knowledge/self_monitoring.md and the ` +
          `knowledge files for your specialty, and follow them.`;
      return agentT(
      `You are Engineer ${d.id} (specialty=${d.specialty}) for round ${round}.
First create YOUR private workspace, then optimize.
\`\`\`bash
# Fresh, ISOLATED workspace via tar-copy that EXCLUDES build artifacts (.git/build/__pycache__/.torch_ext/
# *.so/*.o) — no 'rm' anywhere. Each engineer's out_dir is unique per (round,engineer), so the workspace
# is clean on creation; the tar excludes mean no stale build cache is ever inherited (torch .torch_ext
# stores ABSOLUTE paths, so excluding it forces each workspace to build its own fresh). The big immutable
# golden (reference_io.pt, when present) lives in the selected parent as an absolute symlink; this tar
# carries the symlink verbatim, so every workspace shares the one physical file — NEVER add -h/--dereference here.
mkdir -p ${d.out_dir}/workspace
( cd ${d.seed_dir} && tar --exclude=./.git --exclude='*/.git' --exclude=./build --exclude='*/build' \\
    --exclude=./__pycache__ --exclude='*/__pycache__' --exclude=./.torch_ext --exclude='*/.torch_ext' \\
    --exclude='*.so' --exclude='*.o' -cf - . ) | ( cd ${d.out_dir}/workspace && tar -xf - )
( cd ${d.out_dir}/workspace && git init -q && \\
    git -c user.email=team@workflow -c user.name=team add -A && \\
    git -c user.email=team@workflow -c user.name=team commit -q --allow-empty -m workspace_baseline ) || exit 1
\`\`\`
${readLine} If KK_OPERATOR is non-empty, also consult the operator/language SOTA cards under
KERNEL_KNOWLEDGE_DIR per your role's "operator/language SOTA knowledge (REFERENCE ONLY)" section
(facts/how-to only; measure everything; never go below baseline).
Save best_patch.diff via \`cd <KERNEL_PATH> && git add -A && git diff --cached --binary > ${d.out_dir}/best_patch.diff\` when geomean>${CANDIDATE_FLOOR_TXT}.
${QD_ENABLED ? `QD OVERRIDE: save best_patch.diff for every correct executable candidate, even below the candidate floor; archive admission and canonical promotion are separate gates. Before editing, compute the candidate-owned source hash using python3 ${WORKFLOW_DIR}/scripts/qd_v2.py hash-tree ${d.seed_dir} and require it to equal PARENT_SOURCE_HASH; if it differs, return seed_mismatch without editing or benchmarking.` : ''}

## Inputs
${cfg({
        SPECIALTY: d.specialty,
        DIRECTION: { id: d.id, title: d.title, focus_files: d.focus_files || [], expected_speedup: d.expected_speedup, prompt: d.prompt },
        ...(isDeep ? { TARGET: d.expected_speedup ? `reach ${d.expected_speedup}x (or ~90% of the roofline ceiling), whichever is the harder bar` : 'reach ~90% of the roofline ceiling' } : {}),
        KERNEL_PATH: `${d.out_dir}/workspace`,
        OUTPUT_DIR: d.out_dir,
        CANONICAL, GPU_ID: d.gpu_id, ...GPU_LOCK_INPUT, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT,
        ...(QD_ENABLED ? {
          SEARCH_STRATEGY, PARENT_WORKSPACE: d.seed_dir,
          PARENT_ELITE: d.parent_elite,
          PARENT_ELITE_IDS: d.parent_elite_ids || [],
          PARENT_ELITE_WORKSPACES: (d.donors || []).map(e => e.snapshot),
          PARENT_SOURCE_HASH: d.parent_source_hash,
          QD_ARCHIVE: qdSummary(), QD_OPERATOR: d.operator,
          SELECTED_CELL: d.selected_cell, SELECTED_CONTEXT: d.context_id,
          TARGET_CELL: d.target_cell || '', TARGET_DESCRIPTOR: d.target_descriptor || null,
          TARGET_CASES: d.target_cases || [d.context_id], CELL_SOL_CARD: d.sol_card,
          MUTATION_SCALE: d.mutation_scale || '', TARGET_REGIME: d.target_regime || '',
          EXPECTED_EFFECT: d.expected_effect || '', REQUIRED_EVIDENCE: d.required_evidence || [],
          CHANGED_DIMENSIONS: d.changed_dimensions || [], PRESERVED_DIMENSIONS: d.preserved_dimensions || [],
          STRATEGY_CAPSULE: d.strategy_capsule || {}, QD_ROUTE_DESCRIPTORS_REQUIRED: '1',
          QD_EVIDENCE_HELPER: `${WORKFLOW_DIR}/scripts/qd_v2.py`,
        } : {}),
        codebase_context: `${EVAL_DIR}/codebase_context.md`,
        profiling_summary: profileSummary ? profileSummary.summary_path : '',
        baseline_per_case: BASELINE_PER_CASE,
        INSIGHTS: history.insights,
        KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE,
        KK_REFS: (d.kk_refs && d.kk_refs.length ? d.kk_refs : KK_REFS),
        ...KB_INPUTS,
      })}

Return ONLY the worker_result.json structure as StructuredOutput.` +
      // Built inline, not via roleAgent(), so the injection has to be appended here too.
      expertSkillsBlock(isDeep ? 'deep_engineer' : 'engineer'),
      { phase: 'Optimize', label: `${isDeep ? 'deep' : 'eng'} ${d.id}:${d.specialty}`, schema: ENG_SCHEMA }
    ).then((eng) => ({ d, eng }));
    },

    (prev) => {
      const { d, eng } = prev;
      const patch = `${d.out_dir}/best_patch.diff`;
      // Harvest is MEASUREMENT-anchored, not return-value-anchored. An engineer only writes
      // best_patch.diff when it beat the floor (the Optimize prompt: "Save best_patch.diff ... when
      // geomean>CANDIDATE_FLOOR"), so a lost/failed StructuredOutput does NOT imply there is no winning patch: an
      // engineer that died, timed out, or mis-returned can still have left an applies-clean above-floor diff
      // on disk (observed in a bake-off: a 1.56x Triton patch was silently dropped because its
      // worker_result.json/StructuredOutput never came back). The only return we can TRUST to suppress
      // a patch is a clean below-floor one — the engineer ran, measured, honestly reported <=the floor, and
      // therefore wrote no patch. In every other case (null/failed return, OR a claimed above-floor one) a patch
      // MIGHT be on disk, so we hand the path to verify and let the oracle be the source of truth. We
      // cannot stat the file from the workflow sandbox, so the "is there actually a patch" decision is
      // delegated to verify, which returns apply_failed on an absent/empty patch — dropped by the
      // `verified` filter below, i.e. the same outcome as skipping, but with no false loss.
      const trustworthyBelowBaseline = eng && eng.status !== 'failed' && !(primSpeedup(eng) > CANDIDATE_FLOOR);
      if (trustworthyBelowBaseline && !QD_ENABLED) {
        return { d, eng, ver: null };
      }
      const recovered = !eng || eng.status === 'failed';
      // Two different events wear the same `recovered` flag and used to print
      // nearly the same line. A MISSING return means the engineer died, timed
      // out, or mis-returned, and the patch on disk is unexamined -- the case
      // the recovery path was built for. A FAILED return means the engineer ran
      // to completion, measured, and reported a loss; the patch is examined and
      // the numbers are right there. Verifying it anyway is still correct, and
      // deliberately so: the whole design refuses to let a self-report suppress
      // an independent measurement, and an engineer's own number has been wrong
      // before. But the two cost the same full verify cycle for very different
      // reasons, and the round log is the only place that distinction can be
      // counted later. So the line carries the reported geomean and worst case
      // when there is one -- if the worst case is already below
      // QD_CELL_GUARDRAIL, `qdAdmissionCheck` will refuse on exactly that
      // number, and knowing how often that happens is the input to deciding
      // whether this path deserves a cheaper pre-check.
      if (recovered && !eng) {
        log(`Round ${round} dir ${d.id}: engineer return MISSING (died/timed out/mis-returned) — `
          + `best_patch.diff on disk is unexamined; sending to verify (oracle decides).`);
      } else if (recovered) {
        // Derived here rather than through the worst-context helper the gate
        // uses. That helper reads per-SAMPLE arrays via qdCaseRobust and yields
        // 0 when any context lacks them; an engineer result carries scalar
        // per-case speedups and no samples, so it would print a confident
        // "0.0000" for a candidate whose reported worst case is 0.2719. A wrong
        // number is worse than an absent one. (The helper also has a
        // single-call-site invariant in test_qd_archive.js, which counts
        // occurrences lexically -- so it must not be named here either.)
        const ratios = ((eng.per_case || []).map(c => c && c.speedup)
          .filter(v => typeof v === 'number' && Number.isFinite(v) && v > 0));
        const worst = ratios.length ? Math.min(...ratios) : null;
        log(`Round ${round} dir ${d.id}: engineer return FAILED but complete — self-reported `
          + `geomean ${primSpeedup(eng)}, worst case `
          + `${worst === null ? 'not reported' : worst.toFixed(4)} over ${ratios.length} case(s) `
          + `(scalar self-reports, not robust medians; cell guardrail ${QD_CELL_GUARDRAIL}). `
          + `Self-reports do not suppress an independent measurement; sending to verify `
          + `(oracle decides).`);
      }
      return agentT(
        roleAgent('verify_engineer', 'verify', 'Independently re-measure this candidate patch.', {
          CANONICAL: QD_ENABLED ? d.seed_dir : CANONICAL, PATCH: patch, VERIFY_DIR: `${d.out_dir}/verify`,
          TASK_DIR: KERNEL_PATH_ORIG,
          GPU_ID: d.gpu_id, ...GPU_LOCK_INPUT, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
          ...(QD_ENABLED ? { SEARCH_STRATEGY, QD_REPEAT_MEASUREMENTS: 3,
            QD_ROUTE_DESCRIPTOR_PROPOSALS: eng && eng.route_descriptors ? eng.route_descriptors : [],
            QD_EVIDENCE_HELPER: `${WORKFLOW_DIR}/scripts/qd_v2.py`,
            EXPECTED_PARENT_SOURCE_HASH: d.parent_source_hash,
            SELECTED_CELL: d.selected_cell, SELECTED_CONTEXT: d.context_id,
            CELL_SOL_CARD: d.sol_card, QD_ARCH, QD_DTYPE,
            QD_CELL_GUARDRAIL, QD_CANONICAL_GUARDRAIL } : {}),
          ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
          ...(REQUIRE_GRAPH_CAPTURE ? { REQUIRE_GRAPH_CAPTURE: '1' } : {}),
        }),
        { phase: 'Verify', label: `verify ${d.id}${recovered ? ' (recovered)' : ''}`, schema: VERIFY_SCHEMA }
      ).then((ver) => ({ d, eng, ver, patch }));
    }
  );

  const clean = results.filter(Boolean);
  const verified = clean.filter(r => {
    if (!(r.ver && r.ver.policy_pass === true &&
          says(r.ver.status, 'verified') && says(r.ver.correctness, 'pass'))) return false;
    // Finding (62): a metric fault and a slow kernel are different verdicts (60).
    // Finding (67): so is a measurement that cannot be tied to the pinned oracle.
    // Finding (69): so is a policy pass with nothing behind it.
    // Finding (87): so is a measurement of a tree whose twin was never checked.
    const metricReason = primMetricReason(r.ver) || oracleDrift(r.ver) || qdPolicyReject(r.ver)
      || qdTwinReject(r.ver);
    if (metricReason) {
      log(`  ${r.d && r.d.id ? r.d.id : 'candidate'} verified but not a candidate: ${metricReason}`);
      return false;
    }
    return primSpeedup(r.ver) > CANDIDATE_FLOOR;
  });
  // QD deliberately admits correct executable stepping stones below the greedy candidate floor. They
  // remain in archive cells and can parent later offspring, but still cannot pass canonical promotion.
  const qdVerified = QD_ENABLED ? clean.filter(r => {
    // Finding (69). The archive outlives the run (67): a candidate admitted on an
    // unbacked policy pass becomes a future parent and a future warm start.
    const policyReason = r.ver ? (qdPolicyReject(r.ver) || qdTwinReject(r.ver)) : null;
    if (policyReason) {
      log(`  ${r.d && r.d.id ? r.d.id : 'candidate'} not admitted to the archive: ${policyReason}`);
      return false;
    }
    // (71). `seed_source_hash` is the tree BEFORE the patch and `source_hash` is
    // the tree AFTER it, so equality means the patch changed nothing that the
    // hash can see. Nothing above catches it: such a candidate is correct (it is
    // its parent), verified, policy-clean, and measures within noise of the
    // incumbent. Admitting it costs three ways -- it overwrites its own parent's
    // content-address record, it credits an operator with an effect it did not
    // have, and it consumes budget as though the cell were explored. This is
    // (48) at the archive boundary: a mutation that silently did nothing must
    // not be indistinguishable from one that tried.
    if (r.ver && qdHashValid(r.ver.source_hash) && r.ver.source_hash === r.ver.seed_source_hash) {
      log(`  [qd] ${r.d && r.d.id ? r.d.id : 'candidate'} not admitted to the archive: ` +
          `hash:noop_patch(post-patch source_hash equals the pre-patch seed hash ` +
          `${String(r.ver.source_hash).slice(0, 12)}; the tree is byte-identical to its parent, so ` +
          'there is no mutation here to credit, measure, or store)');
      return false;
    }
    if (!r.ver || r.ver.policy_pass !== true || !says(r.ver.status, 'verified') ||
        !says(r.ver.correctness, 'pass') || r.ver.seed_source_hash !== r.d.parent_source_hash ||
        !qdHashValid(r.ver.source_hash)) {
      // Finding (62). A report whose leading word claims success while its own text
      // contradicts it ("PASS - 10/11 cases") is not the same event as an honest FAIL,
      // and it is the one the old prefix match let through. Name it (60).
      if (r.ver && ((/^pass/i.test(String(r.ver.correctness || '').trim()) && saysContradicted(r.ver.correctness)) ||
                    (/^verified/i.test(String(r.ver.status || '').trim()) && saysContradicted(r.ver.status)))) {
        log(`  [qd] ${r.d && r.d.id ? r.d.id : 'candidate'} rejected: the verify report claims ` +
            `status="${r.ver.status}" correctness="${r.ver.correctness}", but that text contradicts ` +
            `its own leading word — treated as NOT verified`);
      }
      return false;
    }
    const rejects = [];
    const routeCells = qdRouteCells(r.ver, rejects);
    if (!routeCells.length) {
      // A candidate that measured correct and fast is being discarded for a
      // classification fault, which is a different problem from a slow kernel
      // and must not read like one in the round log.
      log(`  [qd] ${r.d && r.d.id ? r.d.id : 'candidate'} verified but yielded no archive cell: ` +
          (rejects.map(x => `${x.route_id} -> ${x.reason}`).join('; ') || 'no route descriptors returned'));
      return false;
    }
    const admission = qdAdmissionCheck(r.ver, routeCells);
    if (admission.reason) {
      log(`  [qd] ${r.d && r.d.id ? r.d.id : 'candidate'} verified but not admitted: ${admission.reason}`);
      return false;
    }
    // Finding (63). The archive-writing loop below used to recompute `suiteRobust`,
    // `routeCells` and `minCase` from the same `ver` and carry its own bare
    // `if (!suiteRobust) continue;` -- a third, silent, weaker copy of the admission
    // rule that (60) did not reach because it is not spelled `qdAdmissionCheck`.
    // Carrying the decision forward makes this filter the single authority: there is
    // no second computation to diverge from it, and no second place to drop a
    // candidate without a reason.
    r.qd_admission = { suiteRobust: admission.suiteRobust, routeCells, minCase: admission.minCase };
    return true;
  }) : verified;

  // --- (d) Build candidate list; integrate if >=2 verified --------------
  // `geomean` here is the PRIMARY metric used for sorting/gating/cumulative: the time-weighted
  // ratio-of-sums when a workload spec is active, else the unweighted geomean (unchanged behavior).
  // The raw unweighted geomean is retained separately for the report.
  let candidates = verified.map(r => ({
    source: `engineer ${r.d.id}`, id: r.d.id, title: r.d.title, specialty: r.d.specialty,
    geomean: primSpeedup(r.ver), geomean_unweighted: r.ver.verified_geomean,
    weighted: r.ver.verified_weighted != null ? r.ver.verified_weighted : null,
    arithmetic: r.ver.verified_arithmetic || r.ver.verified_geomean,
    per_case: r.ver.per_case || [], patch: r.patch, seed_dir: r.d.seed_dir,
    operator: r.d.operator, parent_elite_id: r.d.parent_elite_id || 'canonical',
  }));

  let qdCellChanged = false;
  let qdGlobalImproved = false;
  const qdAdmissions = [];
  if (QD_ENABLED) {
    const previousCoverage = qdArchive.coverage;
    const previousScore = qdArchive.qd_score;
    const previousGlobal = qdArchive.global_best ? qdArchive.global_best.geomean : 0;
    for (const r of qdVerified) {
      // Finding (63): consume the admission decision, never re-derive it. An entry here
      // without one reached the loop past the filter that is supposed to be the only way
      // in, which is a control-flow break and not a candidate to skip quietly (55).
      if (!r.qd_admission) {
        throw new Error(`qd admission invariant broken: ${r.d && r.d.id ? r.d.id : 'a candidate'} `
          + `reached archive admission without passing the qdVerified filter that decides it`);
      }
      const { suiteRobust, routeCells, minCase } = r.qd_admission;
      const sourceHash = r.ver.source_hash;
      const artifactSnapshot = `${qdArchive.archive_dir}/artifacts/${sourceHash}/workspace`;
      qdRecordArtifact(sourceHash, {
        source_hash: sourceHash, snapshot: artifactSnapshot, parent_workspace: r.d.seed_dir,
        patch: r.patch, generation: round,
      }, `round ${round} admission`);
      for (const rc of routeCells) {
        const robust = qdCaseRobust(r.ver, rc.context_id);
        if (!robust || !robust.score) continue;
        const incumbent = qdArchive.cells[rc.cell];
        const eliteId = `${r.d.id}_${rc.context_id}_${sourceHash.slice(0, 12)}`;
        const replacement = !incumbent || robust.lower > incumbent.robust.upper;
        const nearBoundary = !!(incumbent && !replacement && robust.upper > incumbent.robust.lower);
        const beforeCase = r.d.sol_card && (r.d.sol_card.cases || []).find(c =>
          c && c.name === rc.context_id);
        const beforeGap = beforeCase ? beforeCase.sol_gap : null;
        const beforeRegime = beforeCase ? beforeCase.profile_regime : 'unknown';
        // (70), second site. The card that reaches here normally passed the
        // admission gate above -- but not always: a warm-started archive holds
        // elites whose cards were written before that gate existed, and their
        // `sol_ms` becomes this elite's denominator by inheritance. The
        // transition record below is deliberately left carrying the raw
        // `sol_gap_before` regardless: it records what the planner was told,
        // and rewriting it would erase the evidence rather than the error.
        const beforeCeilingReason = beforeCase
          ? qdSolCeilingReject(beforeCase, rc.context_id) : 'sol:ceiling_absent(no parent card)';
        if (beforeCeilingReason && beforeCase) {
          log(`  [qd] ${eliteId} sol_resolved withheld: ${beforeCeilingReason}`);
        }
        // Resolve the card against what the mutation actually measured.
        //
        // `sol_ms` is a denominator the mutation cannot move: the profile role
        // builds it from `2MNK` and the shape's compulsory bytes over the
        // calibrated peaks, all of which are properties of the *context*, not of
        // the kernel. So dividing the candidate's own verified latency by the
        // parent card's `sol_ms` is the same measurement on the same scale, and
        // `sol_gap_before -> sol_gap_after` is a headroom statement rather than
        // another copy of the speedup. Guarded on a positive `sol_ms` because a
        // card that failed to compute one would otherwise emit Infinity and
        // poison every later read of this transition.
        const afterCase = (r.ver.per_case || []).find(c => c && c.name === rc.context_id);
        const afterMs = afterCase ? afterCase.optimized_ms : null;
        const solMs = beforeCase && beforeCase.sol_ms;
        const afterGap = (typeof afterMs === 'number' && afterMs > 0 && typeof solMs === 'number' && solMs > 0)
          ? afterMs / solMs : null;
        // The observed effect uses the SAME non-overlap rule as admission, so a
        // transition can never read "improved" while the cell refused it. An
        // overlap is recorded as no readable change rather than as a small win:
        // treating a sub-interval delta as a direction is how a search learns
        // from noise. `unknown` is reserved for the first visit to a cell, where
        // there is nothing to compare against.
        //
        // `profile_regime_after` stays unknown by construction -- a regime is a
        // profiling verdict, not something derivable from a latency, and
        // inventing one here would put an unmeasured label in the archive.
        const effect = !incumbent ? 'unknown'
          : robust.lower > incumbent.robust.upper ? 'improved'
          : robust.upper < incumbent.robust.lower ? 'regressed'
          : 'no_readable_change';
        const entry = {
          elite_id: eliteId, cell: rc.cell, context_id: rc.context_id, route_id: rc.route.route_id,
          descriptor: rc.route.descriptor,
          descriptor_evidence: rc.route.descriptor_evidence || [],
          predicate_evidence: rc.route.predicate_evidence || [], kernel_symbols: rc.route.kernel_symbols || [],
          resource_signature: rc.route.resource_signature || {}, source_hash: sourceHash,
          artifact: sourceHash, geomean: primSpeedup(r.ver),
          geomean_unweighted: r.ver.verified_geomean,
          weighted: r.ver.verified_weighted != null ? r.ver.verified_weighted : null,
          arithmetic: r.ver.verified_arithmetic || r.ver.verified_geomean,
          robust, suite_robust: suiteRobust, min_case: minCase, per_case: r.ver.per_case || [],
          robust_baseline_frame: QD_ROBUST_BASELINE_FRAME,
          patch: r.patch, parent_workspace: r.d.seed_dir, snapshot: artifactSnapshot,
          generation: round, cost: r.d.cost, parent_elite_id: r.d.parent_elite_id,
          parent_elite_ids: r.d.parent_elite_ids || [], operator: r.d.operator,
          strategy_capsule: r.d.strategy_capsule || {}, sol_card: r.d.sol_card,
          // `sol_card` above is the PARENT's, which is what the mutation was
          // planned against and must stay verbatim for provenance. This is the
          // elite's own position on that same scale, so the next round selects
          // against where this candidate actually landed instead of against its
          // parent's headroom. `derived: true` and the unknown regime are not
          // hedging: nothing profiled this build, and a resolved gap next to a
          // measured `profile_regime` would read as though something had.
          sol_resolved: (afterGap == null || beforeCeilingReason) ? null : {
            context_id: rc.context_id, measured_ms: afterMs, sol_ms: solMs, sol_gap: afterGap,
            remaining_headroom: 1 - 1 / afterGap, profile_regime: 'unknown', derived: true,
            derived_from: r.d.sol_card ? r.d.sol_card.parent_elite_id : null,
            calibration_version: r.d.sol_card ? r.d.sol_card.calibration_version : null,
            // (70). The ceiling travels with the number it produced. `sol_ms`
            // here is the PARENT's denominator reused verbatim, so this elite's
            // headroom is only as backed as that parent's ceiling was -- and
            // (51) is the record of what happens when a card is inherited
            // without its provenance.
            ceiling: beforeCase.ceiling,
          },
          previous_elite: incumbent || null,
        };
        qdArchive.visits[rc.cell] = (qdArchive.visits[rc.cell] || 0) + 1;
        qdArchive.lineage[eliteId] = [entry.parent_elite_id].concat(entry.parent_elite_ids).filter(Boolean);
        const transition = { generation: round, candidate_id: eliteId,
          from_cell: r.d.selected_cell, target_cell: r.d.target_cell || null,
          actual_cell: rc.cell, target_hit: !!r.d.target_cell && r.d.target_cell === rc.cell,
          operator: r.d.operator, fitness_delta: robust.score - (incumbent ? incumbent.robust.score : 0),
          accepted: replacement, cost: r.d.cost, sol_gap_before: beforeGap,
          sol_gap_after: afterGap,
          headroom_closed: (typeof beforeGap === 'number' && beforeGap > 0 && afterGap)
            ? (beforeGap - afterGap) / beforeGap : null,
          expected_effect: r.d.expected_effect || '',
          profile_regime_before: beforeRegime,
          profile_regime_after: 'unknown', observed_effect: effect };
        qdArchive.transitions.push(transition);
        if (r.d.capsule_key) {
          if (!qdArchive.capsules[r.d.capsule_key]) qdArchive.capsules[r.d.capsule_key] = [];
          qdArchive.capsules[r.d.capsule_key].push({ operator: r.d.operator,
            improved: replacement && !!incumbent && robust.score > incumbent.robust.score,
            // (99). `improved` requires an incumbent to have beaten, so a
            // mechanism that fills EMPTY cells scores false on every one of
            // them however large the effect. Round 1's `output_path ->
            // direct_store` opened eight cells at 1.72x-2.29x and was filed
            // `improved: false` eight times, by this rule, correctly.
            //
            // That is the field the next planner reads to decide whether a
            // mechanism is worth repeating (76), so the archive would have
            // steered the search away from the best thing it had found -- not
            // through a bug this time but through the definition. Widening
            // `improved` to cover openings would be the wrong repair: "beat
            // the incumbent" and "reached somewhere nothing had reached" are
            // different events and a planner needs to tell them apart. So it
            // is a second flag, and coverage growth stops being invisible.
            opened_empty_cell: replacement && !incumbent,
            regime_changed: false, generation: round,
            // (76). The outcome flags alone make this a counter, and a counter
            // can only ever block a repeat -- it cannot tell the next planner
            // *what* was tried or what happened, so the same idea returns in a
            // slightly different costume and the counter does not recognise it.
            // These three fields are what makes the entry readable as a
            // mechanism rather than a tally. `observed_effect` is the measured
            // verdict, not the engineer's claim.
            elite_id: eliteId, expected_effect: r.d.expected_effect || '',
            observed_effect: effect,
            capsule: r.d.strategy_capsule || {},
          });
        }
        if (replacement) {
          qdArchive.cells[rc.cell] = entry;
          qdAdmissions.push(entry);
        } else if (nearBoundary) {
          const current = qdArchive.challengers[rc.cell];
          if (!current || robust.upper > current.robust.upper) qdArchive.challengers[rc.cell] = entry;
        }
      }
    }
    let persistedIds = [];
    // Finding (124b). This used to be `if (qdAdmissions.length)`, and every
    // other persist site in the file is guarded the same way, so the archive
    // was written ONLY by rounds that admitted something. Everything a round
    // learns by REFUTING -- the capsule ledger (76), the transition edges, the
    // three stall counters, the generation number -- lives in the top-level
    // fields of the same payload, which are computed and then thrown away at
    // process exit on exactly the rounds that fill them.
    //
    // Run 16 is the demonstration: two generations, two mechanisms measured to
    // destruction over ~2.5 GPU-hours, and a `manifest.json` still reading
    // `generation 0`, `capsules {}`, `recent_transitions []`, `stalls 0/0/0`.
    // A warm start from it is indistinguishable from a pristine seed import and
    // is free to re-propose both dead ends; the two-strike repeated-local-
    // mutation gate below `plan_qd_mutations` can never fire across runs; and
    // `MAX_NO_IMPROVE` resets. A quality-diversity search that
    // only records its successes is not keeping the cheaper half of its results.
    //
    // So the round persists unconditionally. With no admissions the payload is
    // the ledger and nothing else -- `cell_updates: {}` and `admissions: []` --
    // which `qd_persist_manifest.py` handles on the path it already had:
    // nothing to materialize, the merge leaves `cells` exactly as found, and
    // `verify_written` recomputes the same two digests off disk, so
    // `qdVerifyPersisted` still checks a real equality rather than a vacuous
    // one. See `test_qd_persist_manifest.py::LedgerOnlyWriteTest`.
    {
      const ledgerOnly = !qdAdmissions.length;
      const roundPersist = qdPersistPrompt(qdPersistPayload(qdAdmissions), `gen${round}`);
      const persisted = await agentT(
        roleAgent('tech_lead', 'update_qd_archive', roundPersist.prompt, {
          EVAL_DIR, CANONICAL, QD_ARCHIVE_DIR: qdArchive.archive_dir,
          QD_PERSIST_SCRIPT: `${WORKFLOW_DIR}/scripts/qd_persist_manifest.py`,
        }),
        { phase: 'Optimize', label: `tech_lead:qd-archive r${round}${ledgerOnly ? ' (ledger)' : ''}`,
          schema: QD_ARCHIVE_SCHEMA });
      const written = qdVerifyPersisted(persisted, roundPersist.expect, `round ${round}`);
      persistedIds = written ? qdPersistenceReceipt(persisted, qdAdmissions, `round ${round}`) : [];
      if (ledgerOnly) {
        // Nothing to roll back -- no cell moved -- but a failed ledger write is
        // not a non-event either: it means the next run starts blind to this
        // round's refutations, and silence here is what let that go unnoticed
        // for sixteen runs.
        log(written
          ? `  [qd] round ${round} admitted nothing; ledger persisted (generation ` +
            `${roundPersist.expect.generation}, ${Object.keys(qdArchive.capsules || {}).length} ` +
            `capsule(s), stalls ${qdArchive.coverage_stall}/${qdArchive.qd_score_stall}/` +
            `${qdArchive.global_stall})`
          : `  [qd] round ${round} admitted nothing AND its ledger write failed: this round's ` +
            `refutations are in memory only and will be lost at exit. A warm start from this ` +
            `archive may re-propose what this round refuted.`);
      }
      for (const entry of qdAdmissions) {
        if (persistedIds.includes(entry.elite_id)) continue;
        qdLogRollback(entry, `round ${round}`);
        if (qdArchive.cells[entry.cell] && qdArchive.cells[entry.cell].elite_id === entry.elite_id) {
          if (entry.previous_elite) qdArchive.cells[entry.cell] = entry.previous_elite;
          else delete qdArchive.cells[entry.cell];
        }
        if (!Object.values(qdArchive.cells).some(e => e.source_hash === entry.source_hash)) {
          delete qdArchive.artifacts[entry.source_hash];
        }
      }
    }
    qdRecompute();
    qdCellChanged = qdArchive.coverage > previousCoverage || persistedIds.some(id =>
      qdAdmissions.some(e => e.elite_id === id && !!e.previous_elite));
    qdGlobalImproved = !!(qdArchive.global_best && qdArchive.global_best.geomean > previousGlobal);
    qdArchive.coverage_stall = qdArchive.coverage > previousCoverage ? 0 : qdArchive.coverage_stall + 1;
    qdArchive.qd_score_stall = qdArchive.qd_score > previousScore ? 0 : qdArchive.qd_score_stall + 1;
    qdArchive.global_stall = qdGlobalImproved ? 0 : qdArchive.global_stall + 1;
    const durableElites = Object.values(qdArchive.cells).sort((a, b) => b.geomean - a.geomean);
    const canonicalByArtifact = new Map();
    for (const e of durableElites) {
      if (e.geomean <= CANDIDATE_FLOOR || e.min_case < QD_CANONICAL_GUARDRAIL) continue;
      const prior = canonicalByArtifact.get(e.source_hash);
      if (!prior || e.geomean > prior.geomean) canonicalByArtifact.set(e.source_hash, e);
    }
    candidates = [...canonicalByArtifact.values()].map(e => ({
      source: `archive ${e.elite_id}`, id: e.elite_id, title: e.cell,
      specialty: e.operator, geomean: e.geomean, geomean_unweighted: e.geomean_unweighted,
      weighted: e.weighted, arithmetic: e.arithmetic, per_case: e.per_case,
      patch: `${qdArchive.archive_dir}/artifacts/${e.source_hash}/baseline.patch`,
      seed_dir: e.parent_workspace, archive_snapshot: e.snapshot, operator: e.operator,
    }));
  }

  let integrate = null;
  if (!QD_ENABLED && verified.length >= 2) {
    phase('Merge');
    integrate = await agentT(
      roleAgent('integrator', 'integrate', 'Combine this round\'s verified patches into one best implementation.', {
        CANONICAL, INTEGRATE_DIR: `${EVAL_DIR}/round_${round}/integrate`,
        GPU_ID: GPU_POOL, ...GPU_LOCK_INPUT, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
        BEST_INDIVIDUAL: Math.max(...candidates.map(c => c.geomean)),
        PATCHES: verified.map(r => ({ id: r.d.id, specialty: r.d.specialty, title: r.d.title,
          strategy: r.eng ? r.eng.strategy : '', verified_geomean: r.ver.verified_geomean,
          files: r.d.focus_files || [], patch: r.patch })),
        INSIGHTS: history.insights,
      }),
      { phase: 'Merge', label: `integrate r${round}`, schema: INTEGRATE_SCHEMA });
    // Finding (62): the integrated result competes against `candidates[].geomean`, which
    // on a workload-aligned run are weighted numbers. An Integrator that reports only a
    // geomean must not be compared against them -- it would win or lose on the unit.
    const integMetric = integrate && integrate.best
      ? { verified_weighted: integrate.best.weighted, verified_geomean: integrate.best.geomean } : null;
    const integMetricReason = integMetric ? primMetricReason(integMetric) : null;
    if (integMetricReason) log(`  integrate r${round} result discarded: ${integMetricReason}`);
    const integPrim = integMetric && !integMetricReason ? primSpeedup(integMetric) : 0;
    // Finding (69). integrator.md: "Every combined or hand-written result is a new
    // candidate and must be independently scanned even when all parents previously
    // passed." A merge of two clean parents is exactly where a link line gets
    // re-written, so this site is not a weaker one than the others.
    const integPolicyReason = integrate && integrate.policy_pass === true
      ? (qdPolicyReject(integrate, 'integrate') || qdTwinReject(integrate, 'integrate')) : null;
    if (integPolicyReason) log(`  integrate r${round} result discarded: ${integPolicyReason}`);
    if (integrate && integrate.policy_pass === true && !integPolicyReason && integrate.conclusion === 'improved' &&
      integrate.best && integPrim > Math.max(...candidates.map(c => c.geomean))) {
      candidates.push({
        source: 'integrated', id: `r${round}_integrated`, title: 'integrated', specialty: 'integrate',
        geomean: integPrim, geomean_unweighted: integrate.best.geomean,
        weighted: integrate.best.weighted != null ? integrate.best.weighted : null,
        arithmetic: integrate.best.arithmetic || integrate.best.geomean,
        per_case: integrate.best.per_case || [], patch: integrate.best.patch_file,
      });
    }
  }

  candidates.sort((a, b) => b.geomean - a.geomean);
  const winner = candidates[0] || null;
  const improved = !!(winner && winner.geomean > cumulative * (1 + MIN_IMPROVE));
  // Separate question from `improved`: is the SEARCH advancing, not did it beat the incumbent. The
  // `bestSeen > 0` guard keeps round 1 deciding on `improved` alone; from then on bestSeen >= cumulative
  // at the default floor, so at the default PROGRESS_DELTA this implies `improved` and changes nothing.
  // A round with NO candidate is never progress, so a dead round still counts against MAX_NO_IMPROVE.
  const madeProgress = !!(winner && bestSeen > 0 && winner.geomean > bestSeen * (1 + PROGRESS_DELTA));

  // --- (e) Commit the winner into the canonical workspace ---------------
  if (improved) {
    await agentT(
      `You are the TechLead committing round ${round}'s winning patch into the canonical workspace.
\`\`\`bash
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
cd ${CANONICAL}
${QD_ENABLED ? `# QD elite patches are relative to the immutable run baseline, not to the current canonical ancestry.
ROOT=$(git rev-list --max-parents=0 HEAD)
git reset --hard "$ROOT"
git apply --check ${winner.patch}
git apply ${winner.patch}` : `git checkout -- .
# Try a plain apply first, then a 3-way apply (auto-reconciles context-line drift against the blobs)
# before falling back to a manual reconstruction. --3way resolves most "patch does not apply" cases
# that are just context offsets, so the manual path is only hit on a genuine semantic conflict.
git apply ${winner.patch} || git apply --3way ${winner.patch}`}
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "round ${round} winner: ${winner.source} (${winner.geomean.toFixed(2)}x)"
git --no-pager diff --binary "$(git rev-list --max-parents=0 HEAD)..HEAD" > ${EVAL_DIR}/current_best.diff
\`\`\`
If BOTH \`git apply\` and \`git apply --3way\` fail, inspect the patch and apply it manually (edit the
files to match the patch's intent), then \`add -A\` + commit. Before executing correctness, run
\`python3 ${WORKFLOW_DIR}/scripts/candidate_policy_scan.py\` over every candidate-owned source/build path
and candidate ELF, exempting only separately frozen immutable baseline/oracle paths. A finding, inspection
failure, or absent passing receipt means committed=false. The applied source is NOT guaranteed to match
the patch verbatim after a hand-merge, so after committing, RE-RUN both the policy gate and COMMANDMENT
correctness (cd ${CANONICAL} and use gpu_lock); only report committed=true if both pass. Even after a clean
apply, the policy receipt is mandatory because final materialization is a new trust boundary. Return JSON
{committed, current_best_diff, note}.`,
      { phase: 'Merge', label: `commit r${round}`, schema: COMMIT_SCHEMA });
    cumulative = winner.geomean;
    bestPerCase = winner.per_case && winner.per_case.length ? winner.per_case : bestPerCase;
    finalWinner = winner;

    // --- (f) Re-profile the new best ------------------------------------
    profileSummary = await agentT(
      roleAgent('profile_engineer', 'reprofile', 'Re-profile the new best and explain the bottleneck shift.', {
        WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL, ROUND: round,
        COMMANDMENT, PREVIOUS_METRICS: profileSummary,
      }),
      { phase: 'Optimize', label: `reprofile r${round}`, schema: PROFILE_SCHEMA });
    profileSummary = profileSolStrip(profileSummary, `reprofile r${round}`);
  }

  if (winner && winner.geomean > bestSeen) bestSeen = winner.geomean;
  if (QD_ENABLED) {
    // qdGlobalImproved was computed from the durable pre/post-persistence archive above. Do not compare
    // it again after bestSeen has already absorbed the same winner: that ordering erased real progress.
    noImprove = (qdCellChanged || qdGlobalImproved || improved) ? 0 : noImprove + 1;
  } else if (madeProgress || improved) { noImprove = 0; } else { noImprove++; }

  // --- update cross-round memory (insight blackboard + hypothesis ledger)
  const mem = await agentT(
    roleAgent('tech_lead', QD_ENABLED ? 'update_qd_memory' : 'update_memory',
      // (76). This instruction asked for strategy-capsule outcomes from an
      // input that contained no capsules -- unsatisfiable as written, and
      // unsatisfiable in the way that produces confident text about an absent
      // field rather than an error. `qdSummary()` now carries `capsules` and
      // `recent_transitions`, so the task names them and is answerable.
      QD_ENABLED ? 'Distill insights, transition lessons from QD_ARCHIVE.recent_transitions, and ' +
        'strategy-capsule outcomes from QD_ARCHIVE.capsules.' :
        'Distill durable insights + update the hypothesis ledger.', {
      EVAL_DIR, ROUND: round, SKILL_DIR: WORKFLOW_DIR,
      ROUND_RESULTS: clean.map(r => ({ id: r.d.id, title: r.d.title, specialty: r.d.specialty,
        expected: r.d.expected_speedup, claimed: r.eng ? r.eng.speedup_geomean : 0,
        verified: r.ver ? r.ver.verified_geomean : 0, status: r.ver ? r.ver.status : (r.eng ? r.eng.status : 'none'),
        notes: r.eng ? r.eng.notes : '',
        verifier_notes: r.ver ? (r.ver.notes || '') : '',
        verifier_correctness: r.ver ? (r.ver.correctness || '') : '',
        verifier_variance_note: r.ver ? (r.ver.variance_note || '') : '',
        verifier_graph_safe: r.ver ? (r.ver.graph_safe || '') : '' })),
      INTEGRATE: integrate, WINNER: winner ? { source: winner.source, geomean: winner.geomean } : null,
      IMPROVED: improved, REPROFILE_SHIFT: profileSummary ? profileSummary.shift_note : '',
      PRIOR_HISTORY: history,
      ...(QD_ENABLED ? { SEARCH_STRATEGY, QD_ARCHIVE: qdSummary(), QD_ARCHIVE_DIR: qdArchive.archive_dir,
        QD_CELL_CHANGED: qdCellChanged, QD_GLOBAL_IMPROVED: qdGlobalImproved } : {}),
      ...(STATE_DIR ? { STATE_DIR, CANONICAL, CUMULATIVE_SPEEDUP: cumulative, BEST_PER_CASE: bestPerCase } : {}),
      ...(SHARED_KB ? { SHARED_KB, TARGET_LANGUAGE } : {}),
    }),
    { phase: 'Optimize', label: `tech_lead:memory r${round}`, schema: MEMORY_SCHEMA });
  if (mem) {
    if (mem.insights) history.insights = mem.insights;
    if (mem.ledger) history.ledger = history.ledger.concat(mem.ledger);
    if (mem.bottleneck_now) history.bottleneck_now = mem.bottleneck_now;
    if (mem.suggest_next) history.suggest_next = mem.suggest_next;
  }
  history.rounds.push({
    round,
    directions: directions.map(d => ({ id: d.id, title: d.title, specialty: d.specialty })),
    results: clean.map(r => ({ id: r.d.id, claimed: r.eng ? r.eng.speedup_geomean : 0,
      verified: r.ver ? r.ver.verified_geomean : 0, status: r.ver ? r.ver.status : (r.eng ? r.eng.status : 'none') })),
    integrate: integrate ? { conclusion: integrate.conclusion, geomean: integrate.best ? integrate.best.geomean : 0 } : null,
    winner: winner ? { source: winner.source, geomean: winner.geomean } : null,
    improved, cumulative,
  });
  log(`Round ${round} done. winner=${winner ? winner.source + ' ' + winner.geomean.toFixed(2) + 'x' : 'none'}, cumulative=${cumulative.toFixed(2)}x, noImprove=${noImprove}`);
}

// ===========================================================================
// PHASE: Final report (TechLead)
// ===========================================================================
phase('Report');
const report = await agentT(
  roleAgent('tech_lead', 'report', 'Write the final report and the cumulative final patch.', {
    EVAL_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR,
    HISTORY: history, FINAL_WINNER: finalWinner, BASELINE_PER_CASE,
    BASELINE_GEOMEAN_MS, CUMULATIVE_SPEEDUP: cumulative,
  }),
  { phase: 'Report', label: 'tech_lead:report', schema: REPORT_SCHEMA });

// ===========================================================================
// PHASE: Director validation + arbitration
// ===========================================================================
phase('Validate');
const validation = await agentT(
  roleAgent('director', 'validate', 'Independently validate the final patch vs the TRUE baseline.', {
    KERNEL_PATH_ORIG, TASK_DIR: KERNEL_PATH_ORIG,
    EVAL_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL,
    APPLY_TO_ORIGINAL, COMMANDMENT,
    FINAL_PATCH: report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
    TECH_LEAD_REPORTED_GEOMEAN: report ? report.final_speedup_geomean : cumulative,
    ...(HAS_WORKLOAD && report && report.final_speedup_weighted != null
        ? { TECH_LEAD_REPORTED_WEIGHTED: report.final_speedup_weighted } : {}),
    BASELINE_TIMING: BASELINE_PER_CASE,
  }),
  { phase: 'Validate', label: 'director:validate', schema: VALIDATE_SCHEMA });

// Finding (65). The Director validate phase is the run's ONLY independent arbitration, and every one
// of its verdict fields was decorative: `correctness` was declared in VALIDATE_SCHEMA and read by
// nothing, `validation_status` was passed through verbatim with no vocabulary and no gate, and when
// the agent returned null (the hung-agent guard resolves a dead/hung agent to exactly that) the lane
// fell back to `cumulative` -- the TechLead's own self-report -- and published it as `final_speedup`.
// That number then competes in kernel_workflow.js's bakeoff, which ranks purely on speedup and carries
// validation_status as a table column only. So a lane whose arbiter DIED could out-rank a lane whose
// arbiter ran: the independence was removed precisely when it failed. (48): a refusal that cannot name
// itself is indistinguishable from a search that ran and found nothing.
// The verdict below is one place that decides, and it names each refusal distinctly (60).
const validationVerdict = (v, oraclePinned) => {
  // Finding (67). An unpinned run has no evidence that its measurements, its correctness
  // verdicts and its archive writes all used one oracle. That is not a property of the
  // final arbitration, so it is checked before it: the arbiter can be flawless and the
  // denominator still unknown. Reported, never a winner.
  if (!oraclePinned) {
    return { trust: 'flagged', basis: 'unknown',
      reason: 'oracle:unpinned(setup recorded no oracle digest, so no measurement in this run can be '
        + 'shown to share a denominator with any other, in this run or in the archive it writes to)' };
  }
  // Absent arbitration is not a slow kernel and not a fast one -- it is no measurement at all.
  if (!v) {
    return { trust: 'unverified', basis: 'unknown',
      reason: 'provenance:director_validate_returned_nothing(the independent arbitration agent produced '
        + 'no verdict, so the only remaining number is the TechLead self-report; it is kept as '
        + 'tech_lead_reported_geomean and is NOT published as a verified speedup)' };
  }
  // director.md names three unclean bases (host_bound / unprimed / unknown) and, by omission, no clean
  // one; `device` is that value. Absence defaults to `unknown` -- "Absence is not evidence of priming."
  const basis = typeof v.timing_basis === 'string' && v.timing_basis.trim()
    ? v.timing_basis.trim().toLowerCase() : 'unknown';
  if (v.correctness != null && String(v.correctness).trim() !== '' && !says(v.correctness, 'pass')) {
    return { trust: 'refused', basis,
      reason: `correctness:director_validation_reports_"${String(v.correctness).trim()}"(a speedup over a `
        + 'kernel that computes the wrong answer is not a speedup)' };
  }
  if (!says(v.validation_status, 'accept') && !says(v.validation_status, 'validated')) {
    return { trust: 'flagged', basis,
      reason: `validation:director_status_"${String(v.validation_status || '').trim()}"(not accepted; the `
        + 'number is reported for transparency but must not win a bakeoff against accepted lanes)' };
  }
  if (basis !== 'device' && basis !== 'device_time') {
    return { trust: 'flagged', basis,
      reason: `timing:basis_"${basis}"(the ratio is not established to be a device-time ratio; an `
        + 'unlabelled or host-bound number is read as a clean device-time win and must not be one)' };
  }
  return { trust: 'verified', basis, reason: null };
};
// (67): the final arbitration is measured too, so it is held to the same denominator as
// everything it arbitrates. A drifted digest throws inside oracleDrift; a missing one flags.
const oracleFinal = validation ? oracleDrift(validation) : null;
const verdict = oracleFinal
  ? { trust: 'flagged', basis: 'unknown', reason: oracleFinal }
  : validationVerdict(validation, ORACLE_DIGEST != null);
// A refused verdict publishes NO speedup. Leaving `final_weighted`/`final_geomean` populated would be
// pointless: kernel_workflow.js's primSpeedup falls through to them, resurrecting the exact number
// this verdict declined to stand behind.
const trusted = verdict.trust !== 'unverified' && verdict.trust !== 'refused';
const finalGeomean = trusted ? validation.director_verified_speedup_geomean : null;
// PRIMARY headline: the time-weighted speedup when workload-aligned, else the geomean (unchanged).
const finalWeighted = trusted && validation.director_verified_speedup_weighted != null
  ? validation.director_verified_speedup_weighted : null;
const finalPrimary = trusted
  ? (HAS_WORKLOAD && Number.isFinite(finalWeighted) ? finalWeighted : finalGeomean) : null;
if (verdict.reason) log(`  validation verdict ${verdict.trust}: ${verdict.reason}`);
log(`COMPLETE. ${KERNEL_NAME}: ${finalPrimary != null
      ? `${verdict.trust} ${HAS_WORKLOAD ? 'time-weighted' : 'geomean'} ${finalPrimary.toFixed(2)}x`
        + `${HAS_WORKLOAD && Number.isFinite(finalGeomean) ? ` (unweighted geomean ${finalGeomean.toFixed(2)}x)` : ''}`
        + ` [timing_basis ${verdict.basis}]`
      : `NO published speedup (${verdict.trust})`}` +
    ` (status ${validation ? validation.validation_status : 'none'}). Results in ${EVAL_DIR}`);

return {
  mode: MODE,
  target_language: MODE === 'author' ? TARGET_LANGUAGE : undefined,
  authored: MODE === 'author' ? true : undefined,
  eval_dir: EVAL_DIR,
  kernel_name: KERNEL_NAME,
  workload_aligned: HAS_WORKLOAD,
  final_speedup: finalPrimary,                 // PRIMARY metric (weighted when workload-aligned)
  final_weighted: finalWeighted,
  final_geomean: finalGeomean,
  final_arithmetic: trusted ? validation.director_verified_speedup_arithmetic : null,
  tech_lead_reported_geomean: report ? report.final_speedup_geomean : cumulative,
  validation_status: validation ? validation.validation_status : 'unknown',
  // Finding (65): the consumer needs the verdict, not just the arbiter's free-text status. `trust` is
  // the field the bakeoff ranks on; `timing_basis` is the label director.md requires to ride along with
  // any quoted speedup; `validation_reason` names the refusal so it is not read as "found nothing".
  validation_trust: verdict.trust,
  timing_basis: verdict.basis,
  validation_reason: verdict.reason,
  rounds: report ? report.rounds : round,
  budget_used: dispatched,
  budget_total: BUDGET,
  report_path: report ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`,
  final_patch: report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
};
