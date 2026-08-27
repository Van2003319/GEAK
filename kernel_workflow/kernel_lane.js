export const meta = {
  name: 'kernel-lane',
  description: 'SINGLE-LANGUAGE kernel optimization worker (Director/TechLead/specialist Engineers) with budget-controlled rounds, independent verification, integration, and greedy hill-climbing search. Optimizes ONE kernel in ONE language (mode=optimize) or authors a fresh seed then optimizes it (mode=author). This is the worker invoked per lane by the kernel-workflow dispatcher (kernel_workflow.js) and by e2e_workflow; prefer calling kernel-workflow directly unless you specifically want one unchanged lane. Target: AMD Instinct MI-series GPUs; the card is auto-detected on-box.',
  whenToUse: 'Internal single-language worker. Prefer the kernel-workflow dispatcher (kernel_workflow.js) as the entry point; invoke this directly only to run one unchanged lane. Pass args.kernel_path (required), args.mode, args.target_language, args.budget, args.gpu_ids, args.gpu_mode, args.task.',
  phases: [
    { title: 'Setup', detail: 'director builds the isolated eval dir + canonical workspace' },
    { title: 'Author', detail: 'author_engineer writes a fresh optimize-loop seed (only when mode=author); speedup denominator stays the frozen online kernel' },
    { title: 'Analyze', detail: 'tech_lead analyzes kernel + writes roadmap' },
    { title: 'Benchmark', detail: 'benchmark_engineer builds the COMMANDMENT + baseline' },
    { title: 'Profile', detail: 'profile_engineer classifies the bottleneck' },
    { title: 'Research', detail: 'OPT-IN (args.dra_enabled): researcher fans research questions out in parallel via native WebSearch/WebFetch, writes a ranked-directions brief the planner seeds from' },
    { title: 'WarmStart', detail: 'search the experience KB (remote geak:kernel:* when credentialed, else local kb_artifacts/) for the best curated patch per optimization direction for this (kernel,language,gfx), validate each through the verify gate, adopt the first that passes [warm_start!=off]' },
    { title: 'Optimize', detail: 'budget loop: tech_lead plans, specialist OR deep_explore engineers optimize, reprofile' },
    { title: 'Verify', detail: 'each candidate patch independently re-benchmarked' },
    { title: 'Merge', detail: 'integrator combines the round winners' },
    { title: 'Report', detail: 'tech_lead writes the final report + patch' },
    { title: 'Validate', detail: 'director independently validates vs the true baseline, then the TechLead curates ONE distilled card into knowledge/learned/ on a measured win [update_experience!=off]' },
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
// Every argument this worker reads, plus the four the DISPATCHER owns and forwards verbatim through
// its `{...A}` optimize path (`backends`, `enable_fp8`, `e2e_workflow_dir`, `kernel_lane_script`) --
// accepted here and unused, because refusing them would break `mode=optimize` for any caller that
// also named a bake-off option.
//
// Why an unknown key is a HARD ERROR rather than a shrug. Every knob below changes what the run
// admits, and a knob that does not arrive is indistinguishable from a knob whose default was
// intended. `search_strategy: "greedy"` sat in this lane's canonical invocation for six waves after
// the QD search was deleted, matching nothing, doing nothing, and reported by nothing -- so the
// invocation looked like it was selecting a strategy while the argument was inert. That is the same
// shape as the `.replace()` whose old-string was a guess (finding W-3): an edit that can quietly not
// happen. This throw costs nothing, because it fires before the first agent call and before any GPU
// work, and it is the only point in the run where a typo is still cheap.
//
// It cannot catch an OMITTED key -- nothing can, from inside. That is what the effective-config echo
// at the Setup phase is for, and why the launch arguments belong in a committed file rather than in
// prose that gets retyped every wave.
const KNOWN_ARGS = new Set([
  'kernel_path', 'workflow_dir', 'exp_root', 'eval_dir', 'task', 'mode', 'target_language',
  'op_spec', 'budget', 'deep_cost', 'min_improve', 'candidate_floor', 'progress_delta',
  'max_no_improve', 'route_bands', 'gpu_ids', 'gpu_mode', 'gpu_lock_env', 'apply_to_original',
  'state_dir', 'incremental_analyze', 'isa_evidence', 'ir_diagnostics', 'compiler_source_dir',
  'workload_spec_path', 'workload', 'perf_knowledge_dir', 'use_expert_skills', 'expert_skills_dir',
  'shared_kb', 'global_kb', 'e2e_feedback', 'harness_addendum',
  'dra_enabled', 'dra_max_questions', 'dra_blindspot', 'dra_max_blindspots',
  'agent_timeout_ms', 'agent_retries',
  // Learned-KB / warm-start, from upstream. This list is ours -- upstream has no KNOWN_ARGS -- so a
  // merge brings the code that READS these and nothing that ADMITS them, and every one of them
  // becomes the hard error above. That failure is silent in the worst way: the lane refuses at
  // startup, so a KB-on run and a typo look identical from the launcher.
  'use_learned_kb', 'kb_dir_cap', 'kb_cold_direction', 'update_experience',
  'warm_start', 'warm_start_match', 'warm_start_min_speedup', 'kb_artifacts_dir',
  'kb_mode', 'kb_store_dir', 'kb_remote', 'kb_framework_version',
  // The measuring instrument for the KB arms: `held_out` marks a run whose lessons must not be
  // learned from, `box_quiet` records whether the box was contended while it was measured.
  'held_out', 'box_quiet',
  // Wall-clock window. Removed here when the launcher was decoupled, and restored with the merge
  // that brought the code back: the only stop conditions this lane has left are budget and
  // deep_explore_failed, neither of which is a clock, so without these a fixed-duration run has to
  // be simulated from outside. Apostrophes are avoided inside this block on purpose -- the extractor
  // in scripts/lane_args.py reads it with a quote-delimited regex.
  'deadline_epoch', 'no_stop_s', 'max_forced_replans',
  // Dispatcher-owned, forwarded verbatim, unused by the worker.
  'backends', 'enable_fp8', 'e2e_workflow_dir', 'kernel_lane_script',
]);
{
  const unknown = Object.keys(A).filter(k => !KNOWN_ARGS.has(k));
  if (unknown.length) {
    // Levenshtein for ordinary typos, first-token match for a wrong NAME for the right idea
    // (`isa_mode` for `isa_evidence`) -- which edit distance alone ranks badly.
    const dist = (a, b) => {
      const prev = Array.from({ length: b.length + 1 }, (_, j) => j);
      for (let i = 1; i <= a.length; i++) {
        let diag = prev[0]; prev[0] = i;
        for (let j = 1; j <= b.length; j++) {
          const tmp = prev[j];
          prev[j] = Math.min(prev[j] + 1, prev[j - 1] + 1, diag + (a[i - 1] === b[j - 1] ? 0 : 1));
          diag = tmp;
        }
      }
      return prev[b.length];
    };
    const suggest = (k) => {
      const tok = k.split('_')[0];
      const near = [...KNOWN_ARGS]
        .map(c => ({ c, d: dist(k, c), tok: c.split('_')[0] === tok }))
        .filter(x => x.tok || x.d <= 3)
        .sort((x, y) => (x.tok === y.tok ? x.d - y.d : (x.tok ? -1 : 1)))
        .slice(0, 3).map(x => x.c);
      return near.length ? ` -- did you mean ${near.join(', ')}?` : '';
    };
    throw new Error(
      `args contains ${unknown.length} key(s) this workflow does not read: ` +
      unknown.map(k => `"${k}"${suggest(k)}`).join('; ') +
      '. An unrecognized argument is silently ignored, which makes a misspelled knob ' +
      'indistinguishable from an intended default -- so it is refused here, before any agent or GPU ' +
      'work, where it is still free to fix. Accepted arguments: ' +
      [...KNOWN_ARGS].sort().join(', ') + '.');
  }
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
// A candidate is banked when BOTH of these hold, measured against an incumbent arm re-run in the
// candidate's own session:
//
//   1. the suite geomean improved at all (> 1.0, no threshold), and
//   2. at least one route improved by more than max(MIN_ROUTE_WIN, that route's measured floor).
//
// ...unless some route regressed past CATASTROPHIC_REGRESSION, which refuses regardless.
//
// Why these two and not the previous pair. The suite average alone cannot see a single-route
// mechanism: an eleven-route geomean turns a real +7% route win into +0.62%, so a threshold on the
// average is close to "never commit". The per-route test alone cannot see whether the change was
// worth making at all. Requiring both means the average carries the "is this a net win" question
// and the route carries the "did anything actually happen" question, which is the split the
// evidence supports -- and neither has to answer the other's question badly.
//
// What was removed, and why it was wrong. The old rule was a UNION of a per-route test and a suite
// threshold, with a REGRESSION VETO outside the union: any route past its band in the wrong
// direction refused the candidate however good the average. Two costs, both paid on this lane:
//
//   * wave 1 round 2 -- ten of eleven routes +24%..+50%, refused because decode_m2_square gave back
//     0.0006 ms. The legacy gate would have committed it.
//   * wave 1 round 3 -- an integrated stack at +5.35% suite (the eight improved routes averaged
//     +8.4%), refused on three routes giving back 1.3%-2.8%.
//
// The veto optimised Pareto-improvement across routes. This lane is scored on the unweighted suite
// geomean. A gate that refuses a candidate which improves the objective is optimising something
// nobody is measuring, and those two refusals are what it cost. A defence of the veto also argued
// that a union "cannot make a run stricter"; the first round the gate was ever reachable falsified
// that, and a second argument for it -- that eleven routes at +0.4% is a "~4.4% suite win" the
// per-route test would miss -- was simply arithmetic error: the geomean of eleven 1.004s is 1.004.
//
// What survives from the veto is CATASTROPHIC_REGRESSION, which is not a noise judgement. It sits
// an order of magnitude outside the widest floor ever measured here, so it cannot refuse real work;
// it exists so "the average improved" cannot ship one shape a third slower.
//
// Per-route measured noise floors, as {route: fraction}. OPTIONAL, and now only ever a TIGHTENING:
// with no table every route is held to MIN_ROUTE_WIN.
//
// Why the per-route gate exists: COMMANDMENT's claim rule and the tech-lead close-out both say a
// single-route mechanism must be judged by that route's absolute microseconds, with the suite
// geomean used only to confirm nothing else regressed -- and the legacy gate did the opposite. An
// eleven-route geometric mean divides a single-route win by roughly eleven, so a real +7% route
// mechanism reaches the gate reading +0.6%, under the noise, and MIN_IMPROVE on top of that is
// equivalent to "never commit".
//
// Where the table comes from now, and why that changed. It was derived in-lane from the baseline's
// own `samples_ms` -- n=5, band = min-max range -- chosen because the only alternative then was a
// hand-maintained per-host JSON that had gone six epochs stale. That derivation is now deleted: at
// n=5 a min-max range is close to a Bernoulli draw on whether a flyer landed in the sample, and
// against a 24-repeat calibration of the same routes it came out 8.8x too TIGHT on decode_m2_square
// and 2.9x too LOOSE on prefill_m256_down. Wrong in both directions at once is exactly what a floor
// must not be.
//
// There is no producer for this table in the tree any more. The per-machine sweep that used to fill
// it is deleted: it keyed floors to hand-registered machine letters and hardcoded its route
// vocabulary to a single task, so for every other operator it could only refuse -- and it refused at
// launch, before the wave started. What is left is the ARGUMENT: a caller that has measured floors
// for its own routes may still pass them inline (this script cannot read files), and absent one,
// every route is held to MIN_ROUTE_WIN.
//
// A table, when supplied, can only TIGHTEN the gate. That matters because route noise is not
// uniform: on a suite where some routes measure wider than MIN_ROUTE_WIN, a flat bar reads their
// movement as a mechanism when it is noise. On a quiet suite the table changes nothing.
const ROUTE_BANDS_ARG = (() => {
  const raw = A.route_bands;
  if (raw == null) return null;
  let obj = raw;
  // This script cannot read the filesystem (see the header), so a path is not loadable here. It is
  // named explicitly rather than failing on `undefined.bands` later, because the band table lives in
  // a JSON file on disk and passing its path is the obvious first thing a caller tries.
  if (typeof raw === 'string') {
    try {
      obj = JSON.parse(raw);
    } catch (e) {
      throw new Error(`args.route_bands was a string that is not JSON ("${raw.slice(0, 80)}"). ` +
        'This script cannot read files; pass the table inline as an object of {route: floor}. ' +
        `(${e.message})`);
    }
  }
  if (obj && typeof obj === 'object' && obj.bands && typeof obj.bands === 'object') obj = obj.bands;
  if (obj && typeof obj === 'object' && obj.floors && typeof obj.floors === 'object') obj = obj.floors;
  if (!obj || typeof obj !== 'object') throw new Error('args.route_bands must be an object of {route: floor}');
  const out = {};
  for (const [route, band] of Object.entries(obj)) {
    const v = parseFloat(band);
    if (!Number.isFinite(v) || v < 0) {
      throw new Error(`args.route_bands["${route}"]=${band} is not a non-negative fraction`);
    }
    out[route] = v;
  }
  if (!Object.keys(out).length) throw new Error('args.route_bands is empty');
  return out;
})();
// The bar every route must clear before its movement counts as a mechanism rather than as luck.
// 2% is the operator's number and it is the DEFAULT, not the rule: a measured floor wider than this
// replaces it per route. It is comfortably above every floor measured on the three quiet epochs
// this lane has run (max 1.24%), and comfortably below the size of win this lane actually banks on
// its winning route (+12.9%, +30%, +66%).
const MIN_ROUTE_WIN = 0.02;
// Not a noise judgement and not tunable per epoch: an order of magnitude outside the widest floor
// ever measured on this task (7.64%). Its whole job is to stop "the average improved" from shipping
// one shape a third slower, which no suite geomean can see.
const CATASTROPHIC_REGRESSION = 0.10;
// Twin of `scripts/route_gate.py:decide()`. The two are NOT checked against each other by any
// parity test: the Python side is pinned by test_route_gate.py::DecisionTest and this side by
// test_lane_gates.js, so a divergence between them is exactly the failure neither guard sees.
// Returns {applicable, accepted, reason, improved, regressed, routes}. `applicable:false` means the
// evidence to run this gate is missing, and the caller falls back to the legacy test rather than
// refusing a round -- a missing band table must not be able to stall a lane, but it must also never
// look like a pass, which is why the reason is logged either way.
const routeGate = (candPerCase, incPerCase, bands, opts) => {
  const o = opts || {};
  const na = (reason) => ({ applicable: false, accepted: false, reason, improved: [], regressed: [],
    catastrophic: [], routes: [], suiteRatio: null });
  // A missing floor table no longer disables the gate: MIN_ROUTE_WIN covers every route, and the
  // table (when supplied) only raises the bar where a sweep measured the route to be noisier than
  // that. What the gate cannot do without is the paired per-case times, which is what `na` is for.
  bands = bands || {};
  const read = (rows, which) => {
    const out = {};
    for (const row of rows || []) {
      const name = row && (row.name || row.test_case_id);
      const ms = row && (row.optimized_ms != null ? row.optimized_ms : row.candidate_ms);
      if (!name || !(typeof ms === 'number' && ms > 0)) {
        return { err: `${which}: a per_case row has no route name or no positive time` };
      }
      if (out[name] != null) return { err: `${which}: route ${name} appears twice` };
      out[name] = ms;
    }
    if (!Object.keys(out).length) return { err: `${which}: per_case is empty` };
    return { rows: out };
  };
  const cand = read(candPerCase, 'candidate');
  if (cand.err) return na(cand.err);
  const inc = read(incPerCase, 'incumbent');
  if (inc.err) return na(inc.err);

  const missing = Object.keys(inc.rows).filter(r => cand.rows[r] == null).sort();
  if (missing.length) {
    return { applicable: true, accepted: false, improved: [], regressed: [], catastrophic: [],
      routes: [], suiteRatio: null,
      reason: `candidate did not measure ${missing.length} incumbent route(s): ${missing.join(', ')}; ` +
        'neither a suite average nor a non-regression claim can be made over routes that were ' +
        'not measured' };
  }
  // No per-route floor for a route is NOT a refusal any more. `MIN_ROUTE_WIN` is a floor under the
  // floor: a route we have not measured is held to 2%, which is above every floor measured on the
  // three quiet epochs this lane has run on and is the same number a reader would apply by hand.
  // The measured table only ever RAISES the bar, and only where a sweep says it must.
  const routes = [], improved = [], regressed = [], catastrophic = [];
  let logSum = 0;
  for (const route of Object.keys(inc.rows).sort()) {
    const floor = bands[route] == null ? 0 : bands[route];
    const need = Math.max(MIN_ROUTE_WIN, floor);
    const i = inc.rows[route], c = cand.rows[route];
    const delta = (i - c) / i;                       // > 0 means the candidate is faster
    logSum += Math.log(i / c);
    const status = delta > need ? 'won'
      : (-delta > CATASTROPHIC_REGRESSION ? 'catastrophic'
        : (-delta > Math.max(floor, 0) ? 'regressed' : 'flat'));
    routes.push({ route, incumbent_ms: i, candidate_ms: c, delta_frac: delta, floor, need, status });
    if (status === 'won') improved.push(route);
    if (status === 'regressed') regressed.push(route);
    if (status === 'catastrophic') { catastrophic.push(route); regressed.push(route); }
  }
  const suiteRatio = Math.exp(logSum / routes.length);
  const pct = (x) => `${(x * 100).toFixed(2)}%`;
  const done = (accepted, reason) => ({ applicable: true, accepted, reason, improved, regressed,
    catastrophic, routes, suiteRatio });

  // The "we would never ship this" fence, and the only refusal that survives a positive suite
  // number. It is deliberately an order of magnitude outside every floor ever measured here (the
  // widest is 7.64%), so it is not a noise judgement and cannot refuse real work: it exists so that
  // "the average improved" cannot ship one shape a third slower.
  if (catastrophic.length) {
    return done(false, 'a route regressed catastrophically (past ' + pct(CATASTROPHIC_REGRESSION) +
      ', an order of magnitude outside any measured floor): ' +
      routes.filter(r => r.status === 'catastrophic')
        .map(r => `${r.route} ${pct(-r.delta_frac)}`).join(', '));
  }
  // Condition 1: the suite average did not get worse. Computed HERE, over the paired per-route
  // times, rather than taken from `winner.geomean`: both arms then come from one invocation, and
  // the 1.5-3% an unchanged tree wanders between invocations cancels instead of deciding the round.
  if (!(suiteRatio > 1)) {
    return done(false, `the suite geomean did not improve (${suiteRatio.toFixed(5)} vs the ` +
      'incumbent arm measured in the same session)');
  }
  // Condition 2: something concrete moved. A suite average that improves while no single route
  // clears its own noise is the shape a round of pure measurement luck takes, and banking it is how
  // a lane ratchets upward on selection bias -- three candidates a round, and only the favourable
  // draw is ever kept.
  if (!improved.length) {
    const nearest = routes.slice().sort((a, b) => b.delta_frac - a.delta_frac)[0];
    return done(false, `the suite improved (${suiteRatio.toFixed(5)}) but no single route cleared ` +
      `its own bar; the best was ${nearest.route} ${pct(nearest.delta_frac)} against a required ` +
      `${pct(nearest.need)}${nearest.need > MIN_ROUTE_WIN
        ? ` (its measured floor, wider than the ${pct(MIN_ROUTE_WIN)} default)` : ''}`);
  }
  // Targeting is a NOTE, not a veto. It used to refuse a win that landed off the declared route;
  // `target_routes` came back on 1 of 26 engineer results, so in practice that rule refused work
  // over a missing label far more often than it caught an artefact. The caller logs the mismatch.
  return done(true, 'suite ' + suiteRatio.toFixed(5) + ' and cleared its own bar on: ' +
    routes.filter(r => r.status === 'won')
      .map(r => `${r.route} ${pct(r.delta_frac)} (needed ${pct(r.need)}${
        r.need > MIN_ROUTE_WIN ? ', its measured floor' : ''})`).join(', ') +
    (regressed.length ? `; gave ground within noise on ${regressed.join(', ')}` : ''));
};
// The n=5 band derivation that used to live here is DELETED, not disabled.
//
// It read a band off `baseline_per_case[].samples_ms` as (max-min)/median, and it was the thing
// that made the per-route gate reachable without a hand-maintained file -- which was the right
// problem to solve and the wrong statistic to solve it with. Against a 24-repeat calibration of the
// same routes it came back 8.8x too TIGHT on decode_m2_square (1.31% vs 11.55%) and 2.9x too LOOSE
// on prefill_m256_down. At n=5 a min-max range is close to a Bernoulli draw on whether a flyer
// landed in the sample, so it was not measuring the route, it was measuring the draw. It cost this
// lane a real refusal: wave 1 round 2 vetoed on decode_m2_square's 1.31% band.
//
// Nothing replaced it in-lane, and that is deliberate: a floor is a property of the box and the
// route, and this script has no honest way to measure one from a handful of baseline samples. A
// caller that has measured floors properly may pass them via `args.route_bands`; absent one, every
// route is held to MIN_ROUTE_WIN.

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
// gates on the unweighted number, beside weighted scores from other candidates. That
// is the (59c) shape: not a default for a missing value but a silently different
// QUANTITY. The fallback is kept here because this selector has a second, advisory
// caller (the engineer's own self-report at the skip-verify check, whose schema
// requires only `speedup_geomean`, and where being lenient errs toward sending the
// candidate to the oracle). Every AUTHORITATIVE caller now refuses the ambiguous case
// by name -- see the `verified` / integrate gates.
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
  // VERDICT vocabulary only. `error`, `nan` and `inf` were here and had to go: they are
  // MEASUREMENT vocabulary, and a correctness report quotes them when it PASSES --
  // "PASS (max error 3.1e-05, within rtol=1e-3)", "pass, no NaN or Inf in output",
  // "PASS - allclose within tolerance, no errors" were all vetoed. That is this veto
  // firing in the one direction it must never fire in: a correct, faster candidate
  // discarded, and the ledger recording it as a correctness failure -- which closes the
  // direction for the rest of the run, since the ledger is the search's only memory.
  // Matching the presence of a word cannot see a negation, so words that appear in
  // passing reports do not belong in the list. Do not re-add them: the fail-open cases
  // they were meant to catch ("contains NaN") are already caught by `status`, which the
  // caller gates through this same helper and which verify_engineer.md requires to be
  // `correctness_failed` on any correctness failure.
  if (/\b(fail\w*|mismatch\w*|incorrect|wrong|partial\w*|except)\b/i.test(s)) return true;
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
const EXPERT_SKILL_ROLES = new Set(['tech_lead', 'author_engineer', 'engineer', 'deep_engineer',
                                    'asm_engineer']);

// --- Deep Research Agent (DRA) -------------------------------------------------------------------
// OPT-IN: a v4-native research phase that runs AFTER Profile and BEFORE the optimize loop (so the
// COMMANDMENT + baseline profile + analysis exist). The `researcher` persona extracts facts and a
// ranked set of research QUESTIONS; the script fans those out in PARALLEL (each question = one
// hang-guarded agent using native WebSearch/WebFetch), then a synthesis pass writes a ranked
// directions portfolio (deep_search.md / deep_search_brief.md / deep_search.json) into EVAL_DIR that
// the TechLead's plan_round seeds from. DEFAULT OFF: when dra_enabled is not "true" NOTHING runs and
// behavior is byte-identical to a build without this feature (existing runs unchanged).
const DRA_ENABLED = String(A.dra_enabled != null ? A.dra_enabled : 'false') === 'true';
const DRA_MAX_QUESTIONS = (() => {
  const v = parseInt(A.dra_max_questions != null ? A.dra_max_questions : 8, 10);
  return Number.isFinite(v) && v >= 1 ? v : 8;
})();
// Optional Stage 5/6 blindspot-critique pass (one more parallel research wave). Default OFF (budget).
const DRA_BLINDSPOT = String(A.dra_blindspot != null ? A.dra_blindspot : 'false') === 'true';
const DRA_MAX_BLINDSPOTS = (() => {
  const v = parseInt(A.dra_max_blindspots != null ? A.dra_max_blindspots : 4, 10);
  return Number.isFinite(v) && v >= 1 ? v : 4;
})();

// ---------------------------------------------------------------------------
// LEARNED-KNOWLEDGE SINK. Derived from WORKFLOW_DIR and nothing else, so a lane spawned by
// e2e_workflow still curates into kernel_workflow/knowledge/learned/ — e2e's learned/ is a
// separate memory owned by its own system_architect step.
//   update_experience = on (default) | off | false | none
// The bake-off dispatcher passes `off` and curates once centrally instead (N parallel lanes would
// file N near-duplicate cards for one op).
const LEARNED_DIR = `${WORKFLOW_DIR}/knowledge/learned`;
// How much of ONE round the learned KB may steer. Measured, not guessed: an arm whose planner was
// handed at most 3 matched cards and could KB-seed at most one direction reached 4.45x geomean on 16
// kernels; the same 16 under a planner that reads the whole 84-line index and picks freely reached
// 3.44x — a 22.7% regression, with the losses concentrated on kernels where the bounded arm had found
// an unusual win (one went 25.77x -> 2.15x). Both arms averaged ~3 rounds, so it was not lost rounds.
// The index-reading design is better at RECALL; what it lost was the profile-driven direction that
// nothing from the KB had touched — the per-round control. `knowledge/learned/README.md` says the KB
// must be "an accelerant, NOT a cage", and this is the line that keeps that true.
//   KB_DIR_CAP        directions per round that may draw on cards (default 1)
//   KB_COLD_DIRECTION at least one direction per round must be profile-only (default on)
// Whether the planner is being pointed at the learned KB at all. Defined HERE because this lane is
// where it is USED: the constant of this name lives in the dispatcher, and referencing it from the
// lane threw `USE_LEARNED_READ is not defined` at the first plan step of every kernel — 13 of them
// before the run was stopped. A parse test cannot catch that: the file compiles, and the reference
// only resolves when the line runs. Default ON, matching this branch's design where the tech_lead
// role reads INDEX.md unconditionally; `use_learned_kb=false` turns the budget block off with it.
const USE_LEARNED_READ = String(A.use_learned_kb != null ? A.use_learned_kb : 'true') === 'true';
const KB_DIR_CAP = Math.max(0, parseInt(A.kb_dir_cap != null ? A.kb_dir_cap : 1, 10));
const KB_COLD_DIRECTION = String(A.kb_cold_direction != null ? A.kb_cold_direction : 'true') === 'true';
let kbCapBound = 0;      // rounds where the cap actually had to strip something
const UPDATE_EXPERIENCE = String(A.update_experience != null ? A.update_experience : 'on').trim().toLowerCase() || 'on';
const UPDATE_EXPERIENCE_ON = UPDATE_EXPERIENCE !== 'off' && UPDATE_EXPERIENCE !== 'false' && UPDATE_EXPERIENCE !== 'none';

// ---------------------------------------------------------------------------
// WARM-START (local experience KB). Before the optimize loop, search the machine-produced
// kb_artifacts/ store for the top-3 best patches for THIS (kernel, language, gfx), validate
// each through the SAME verify_engineer gate, and adopt the first that passes; after Validate,
// write this run's own win back.
//   on (default)      | read + validate top-3, ADOPT the first that passes.
//   reference         | read top-3 as prose only, never auto-apply.
//   return_after_read | adopt then RETURN before the optimize loop.
//   off/false/none, a STATE_DIR resume, or no arch => cold start (byte-identical to pre-feature).
const WARM_START = String(A.warm_start != null ? A.warm_start : 'on').trim().toLowerCase() || 'on';
const WARM_START_ON = WARM_START !== 'off' && WARM_START !== 'false' && WARM_START !== 'none';
const WARM_START_REF_ONLY = WARM_START === 'reference';
const WARM_START_RETURN_AFTER = WARM_START === 'return_after_read';
const KB_ARTIFACTS_DIR = String(A.kb_artifacts_dir ||
  (WORKFLOW_DIR ? WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/kb_artifacts' : '')).replace(/\/+$/, '');
const EXPERIENCE_STORE = `${WORKFLOW_DIR}/scripts/experience_store.py`;
// Every candidate costs a full on-box verify to reject, so a recorded near-tie is not worth reading.
const WARM_START_MIN_SPEEDUP = Number.isFinite(parseFloat(A.warm_start_min_speedup))
  ? parseFloat(A.warm_start_min_speedup) : 1.05;
// How hard the resolver may work to map this run's kernel name onto a stored page. The name is
// layout-derived (`fused_moe_kernel` standalone, `fused_moe_kernel_task` from an e2e head extraction),
// so `exact` alone would make the head path miss its own history; `fuzzy` also accepts an op_kind.
const WARM_START_MATCH = ['exact', 'normalized', 'fuzzy'].includes(String(A.warm_start_match || '').trim())
  ? String(A.warm_start_match).trim() : 'fuzzy';
// Which PLANE the experience comes from and goes back to. Same phases, same schemas, same verify
// gate either way — only the two command strings differ, because the store subcommands were built
// to print the same JSON as the directory ones.
//   local (default)  the curated kb_artifacts/ tree, keyed by slug.
//   store            a KB Store on disk in the shape the service uses, keyed by canonical id.
//                    This is the plane that later becomes the remote service, so a run in this
//                    mode is the rehearsal for it.
const KB_MODE = String(A.kb_mode || 'local').trim().toLowerCase() === 'store' ? 'store' : 'local';
const KB_STORE_DIR = String(A.kb_store_dir ||
  (KB_ARTIFACTS_DIR ? KB_ARTIFACTS_DIR.replace(/\/[^/]*$/, '') + '/kb_store_local' : '')).replace(/\/+$/, '');
// The key carries a rocm <major>.<minor>; on a box without /opt/rocm the measured stack is empty and
// every record would file under `unspecified`, which splits one kernel's history across two keys.
const KB_FRAMEWORK_VERSION = String(A.kb_framework_version || '').trim();
const KB_VERSION_FLAG = KB_FRAMEWORK_VERSION ? ` --framework-version ${JSON.stringify(KB_FRAMEWORK_VERSION)}` : '';
const KB_ROOT_OK = KB_MODE === 'store' ? !!KB_STORE_DIR : !!KB_ARTIFACTS_DIR;
// Whether this run may also talk to the KB Store SERVICE, on top of whichever local plane KB_MODE
// selected. `auto` (default) uses it when credentials are present and silently does not when they
// are not; `off` restores the directory-only behaviour byte for byte.
const KB_REMOTE = String(A.kb_remote || 'auto').trim().toLowerCase() === 'off' ? 'off' : 'auto';
// The credentials are NOT visible from this process. They live in the user's profile, and the
// shells these commands run in are non-interactive, so `process.env.KB_STORE_TOKEN` is empty here
// even on a box where the service is configured. Deciding the plane in JS would therefore mean
// deciding it wrong. Instead every emitted command exports its own credentials from the profile's
// two sources and then branches on what it actually got — the test happens where the answer is
// knowable. The token is read from a 0600 file into a variable, never passed in argv, because
// `ps` is world-readable on this box.
// The gateway's internal AMD CA is not in a stock container trust store, so a KB command run inside
// one fails TLS (the old workaround was `curl -k`). DETECT then heal: only when the caller has NOT
// already established trust (SSL_CERT_FILE unset) do we point urllib/requests/curl/node at the first
// readable AMD-root bundle we find. Most callers here run OUTSIDE the warm-start launcher (which sets
// these at `docker run`), so this self-heal is what keeps their KB reachable. Path-only, no CA
// content; overridable with KB_CA_BUNDLE; a no-op when SSL_CERT_FILE is already set or no bundle is
// readable (so CI and already-trusting images are byte-identical). DNS (the host has none
// in-container) is a launch concern, handled with `docker run --add-host`.
// GEAK_KB_STORE_* wins over the bare KB_STORE_* (so a host that also runs Hyperloom, which carries
// its own KB_STORE_*, can point GEAK at a distinct store); re-exported under the bare names every
// downstream reader here branches on.
const KB_ENV_PRELUDE =
  'export KB_STORE_URL="${GEAK_KB_STORE_URL:-${KB_STORE_URL:-https://global.primus-safe.amd.com/knowledge-base}}"; ' +
  'export KB_STORE_TOKEN="${GEAK_KB_STORE_TOKEN:-${KB_STORE_TOKEN:-$(cat ~/.geak_kb_token 2>/dev/null)}}"; ' +
  'if [ -z "${SSL_CERT_FILE:-}" ]; then for _ca in "${KB_CA_BUNDLE:-}" ' +
  '/shared_nfs/hyperloom/ca/amd-ca-combined.pem "$HOME/amd-extra-ca-bundle.pem"; do ' +
  '[ -n "$_ca" ] && [ -r "$_ca" ] && { export SSL_CERT_FILE="$_ca" REQUESTS_CA_BUNDLE="$_ca" ' +
  'CURL_CA_BUNDLE="$_ca" NODE_EXTRA_CA_CERTS="$_ca"; break; }; done; fi; ';
// Writing in store mode records BOTH planes in one call, so it needs both roots: the directory tree
// stays the source of truth a curation pass edits, and the store is derived from it.
const KB_WRITE_OK = KB_ROOT_OK && !!KB_ARTIFACTS_DIR;
const kebab = s => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60);

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
//   MAX_NO_IMPROVE   consecutive non-improving rounds a search PHASE gets before the ladder
//                    escalates to the next one (default 4). Spending them in the last phase is what
//                    ends the run.
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
// `max_no_improve` KEEPS ITS NAME AND CHANGES WHAT IT ENDS: a search PHASE, not the wave. The
// default moves 2 -> 4 with it, because 2 was chosen when the counter's only outcome was death.
//
// It is also no longer the PRIMARY escalation signal, only the backstop -- see `escalationReason`
// below. One knob rather than two on purpose: a separate `phase_escalate_after` beside a surviving
// `max_no_improve` would be two names for one quantity, and callers would set the one that no
// longer decides.
const MAX_NO_IMPROVE = Math.max(1, parseInt(A.max_no_improve != null ? A.max_no_improve : 4, 10));

// ---------------------------------------------------------------------------------------------
// THE SEARCH CYCLE
//
// Three states, and the third one closes a loop rather than ending the run:
//
//   phase1_source  L1 pattern triage + L2 profiling every round, L3 only when L2 or the TechLead
//                  asks for it. Rewrites are proposed against SOURCE.
//   phase2_asm     source analysis stops. L2 localises a KEYPOINT in the compiled assembly and the
//                  engineer edits that window of the `.s`.
//   deep_explore   EXACTLY ONE dedicated round. A ground-up rewrite with broad authority, which is
//                  a different route rather than a further step down this one.
//                    committed  -> back to phase1_source, and the cycle runs again ON THE NEW TREE
//                    not         -> the run ends
//
// WHY IT LOOPS, and why deep_explore is a round rather than a phase. `deep_engineer.md` rewrites
// the kernel from the ground up with authority over the kernel, the wrapper and the binding, so a
// committed deep round leaves a materially DIFFERENT program behind. Everything phase1 learned was
// about the old implementation and every keypoint phase2 found was in the old machine code; none of
// it describes the new tree. Treating deep_explore as a terminal phase would spend the search's
// most expensive move and then throw away the only thing it produced -- a fresh starting point with
// its own source-level and assembly-level headroom. So it resets the ladder instead.
//
// And it gets ONE round, not a stall allowance: it is a single rewrite whose whole job is to hand
// the next cycle a different route. Giving it four rounds would be four ground-up rewrites in a row,
// each discarding the last one's tree before anything had searched it.
//
// A deep round that does NOT commit is the terminal case: a ground-up rewrite starting from the
// current best failed to beat it, which is the strongest evidence available that the search is out
// of routes. Nothing caps the number of cycles beyond BUDGET, which already prices a deep round at
// DEEP_COST.
const PHASE_SOURCE = 'phase1_source';
const PHASE_ASM = 'phase2_asm';
const PHASE_DEEP = 'deep_explore';
// The order the states are visited within one cycle. `deep_explore` is last and wraps around.
const PHASE_ORDER = [PHASE_SOURCE, PHASE_ASM, PHASE_DEEP];

// A state with no implementation is SKIPPED rather than entered: burning an allowance running
// phase1's behaviour under phase2's name reads in the log as "the assembly phase found nothing"
// when nothing was tried. phase2 became live with `asm_roundtrip.py` and `isa_signals.py keypoint`;
// the flag stays because "can this state actually act" is a question the ladder should keep asking.
const PHASE_IMPLEMENTED = { [PHASE_SOURCE]: true, [PHASE_ASM]: true, [PHASE_DEEP]: true };

// The next state that can act. Wraps: after `deep_explore` the cycle restarts at the source phase.
// Never returns '' -- the loop is closed, and what ends a run is a deep round that banked nothing
// or the budget running out, not falling off the end of a list.
const nextLivePhase = (from) => {
  const n = PHASE_ORDER.length;
  const at = PHASE_ORDER.indexOf(from);
  for (let step = 1; step <= n; step++) {
    const candidate = PHASE_ORDER[(at + step) % n];
    if (PHASE_IMPLEMENTED[candidate]) return candidate;
  }
  return PHASE_SOURCE;   // unreachable while the source phase is implemented
};

// Why this round should hand the search on, or '' to keep going.
//
// TWO triggers, and the ORDER matters. The TechLead's `stop` is the reasoned one: its gate is "the
// launch floor has been attacked, no compute case is still far above it, and the last verified gain
// was under ~3%", which is a statement that THIS LEVEL is finished -- exactly the premise for
// moving to the next one. The counter is the backstop for a planner that keeps proposing directions
// none of which land, and never admits the level is done.
//
// It used to be the other way round: `stop` ended the whole run and the counter was the only
// escalation. That made the reasoned signal the destructive one and the crude signal the useful
// one, so a lane could reach the verdict "source search is finished" and go home instead of moving
// to the phase that exists for it.
const escalationReason = (plannerStopped, stall, max) => {
  if (plannerStopped) {
    return 'the TechLead reported this level finished (its stop gate: floor attacked, no compute '
      + 'case far above it, last verified gain under ~3%)';
  }
  if (stall >= max) return `${stall} consecutive non-improving round(s)`;
  return '';
};

// Finding (87), one question further along. `hip_twin_sync.py` proves the file
// that was EDITED is the file that was COMPILED. It cannot prove the MECHANISM
// that was written survived compilation: a widened staging load whose alignment
// the backend could not prove, a matrix-core builtin lowered back to the opcodes
// it replaced, a conversion burst re-materialized. All of those build, pass
// correctness, measure within noise of the parent, and get written into
// `history.ledger` as "tried X, no effect". That entry is false, and the planner
// reads the ledger as memory -- in greedy search it is the ONLY memory there is,
// so one false negative closes a direction for the rest of the run. This is the
// same cost (87) names: "not one wasted round, [but] a mechanism written off".
//
// `isa_capture.py` archives the AMDGCN out of the artifact that was measured and
// `isa_signals.py` diffs it against the parent's archive.
//   observe  capture + record + log. Never rejects a candidate. DEFAULT.
//   gate     additionally refuses a candidate whose own ISA receipt refutes the
//            mechanism it declared.
//   off      no capture, no prompt input, no record. An escape hatch, not a normal
//            setting -- for an ablation, or for a box where the read is impossible.
//
// `observe` IS THE DEFAULT, and that is the design rather than a convenience.
//
// The tempting rule is "read the machine code when the source does not explain the
// result". It cannot work here, because the case this layer exists to catch produces
// a result that IS fully explained by the source: the engineer widened a load, the
// compiler quietly declined to, the candidate builds, passes correctness, and
// measures within noise of its parent. That reads as a clean, self-consistent
// negative. NOTHING about it says "go look at the assembly". Escalating on a symptom
// works only for symptoms that exist, and the whole failure mode here is that there
// is no symptom -- so a conditional check is not a cheaper version of this check, it
// is a check that never fires on the case it was written for.
//
// Off-by-default would be the same mistake with a flag on it. Finding (87) is the
// standing example: `hip_twin_sync.py` "has existed and been correct for several
// rounds while nothing called it", and the fix was not a better tool, it was calling
// it. A layer that has to be switched on will be off during exactly the run whose
// ledger it was meant to keep honest.
//
// What default-on costs, stated plainly rather than hand-waved: the ISA inputs are
// now spread on every hip-shaped lane, so verify and engineer prompts grow by a
// bounded block, and each candidate pays two CPU-only subprocess calls over an
// artifact that already exists. It cannot rebuild, cannot touch the GPU, and cannot
// perturb a measurement. Where the read is impossible the capture returns its HOLE
// code, the verdict is `indeterminate`, and nothing is refused -- inert and visible.
//
// `gate` stays opt-in for a different reason: it changes outcomes, and there is no
// field data yet on how often a `refuted` verdict would be a false refusal. Observe
// first, on real kernels, then decide from the number.
const ISA_MODES = ['off', 'observe', 'gate'];
const ISA_MODE_RAW = String(A.isa_evidence != null ? A.isa_evidence : '').trim().toLowerCase();
const ISA_MODE = ISA_MODE_RAW === '' || !ISA_MODES.includes(ISA_MODE_RAW)
  ? 'observe' : ISA_MODE_RAW;
// A misspelled mode resolves to the default and SAYS SO. It deliberately does not
// resolve to `off`: `isa_evidence=gat` silently becoming "no ISA layer at all" is the
// same silent-off failure the TARGET_LANGUAGE whitelist had, and a typo is exactly
// how it would happen. Emitted with the effective-config echo at the Setup phase,
// because `log` is not available this early in the file.
//
// The unknown-argument check above cannot cover this case: `isa_evidence` is a KNOWN key here, so
// only its VALUE is wrong, and a value can be defaulted where a key cannot.
const ISA_MODE_WARNING = (ISA_MODE_RAW !== '' && !ISA_MODES.includes(ISA_MODE_RAW))
  ? `isa_evidence="${ISA_MODE_RAW}" is not one of ${ISA_MODES.join('|')}; using `
    + `"${ISA_MODE}". If you meant to enforce, pass "gate"; nothing is gated on this run.`
  : '';
// The mode is the ONLY precondition. This deliberately does NOT consult
// `TARGET_LANGUAGE`, and an earlier version that did was wrong in the worst
// direction.
//
// The real precondition is not "the lane declared a language this stack can read",
// it is "the build produced an AMDGPU code object in the tree that was scanned".
// `isa_capture.py` measures exactly that and says so out loud: no code object is
// its HOLE exit code, named in `manifest.json` under `holes`, and echoed per
// candidate by the round log. A language whitelist is a weaker, indirect proxy for
// that measurement, and it fails SILENTLY -- `TARGET_LANGUAGE` defaults to
// `'triton'` whenever the caller does not pass it (which the greedy pipeline does
// not), so the whitelist turned the whole layer off with no message at all. A loud
// HOLE beats a silent off: one is a gap you can see, the other is a gate everybody
// believes is running.
//
// What that costs on a lane this cannot read: the capture returns HOLE, the verdict
// is `indeterminate`, and `isaEvidenceReject` refuses nothing. It degrades to inert
// AND visible, which is the correct failure direction and is pinned by the
// non-refusal probes in `test_lane_gates.js`.
//
// The AMD-specific scope has not changed and is not a language check -- it is a
// property of the tools. `isa_capture.py` accepts an ELF only when `e_machine` is
// EM_AMDGPU (224), disassembles with `llvm-objdump --arch-name=amdgcn`, and reads
// register/scratch/LDS out of AMDGPU metadata (`amdhsa.kernels`, `.vgpr_count`,
// `.private_segment_fixed_size`); `isa_signals.py`'s whole op vocabulary is AMDGCN
// mnemonics. An NVIDIA cubin is EM_CUDA (190) and SASS spells everything
// differently (HMMA, LDS, LDG.E.128, BAR.SYNC), so a CUDA build yields a HOLE
// rather than a wrong answer. A Triton lane also yields a HOLE, because its code
// object is a `.hsaco` in TRITON_CACHE_DIR rather than in the scanned workspace;
// supporting it means passing a per-candidate cache dir as an extra --scan root AND
// forcing a fresh compile, or a cache hit hands over a PREVIOUS candidate's binary
// and the diff describes the wrong tree. `dump_ir.sh` already solves both and is
// the pattern to copy.
const ISA_ENABLED = ISA_MODE !== 'off';
// L3's capture RECOMPILES; `isa_capture.py` deliberately does not. That is a real
// difference in cost and in blast radius, so it gets its own switch rather than
// riding on `isa_evidence`: an operator who wants the verification receipt -- which
// reads the artifact that actually ran and cannot break a build -- may reasonably
// not want a lane that rebuilds a translation unit in order to look at it. Defaults
// ON when the ISA layer is on, because the ladder is the point of having one.
const IR_DIAGNOSTICS = String(A.ir_diagnostics != null ? A.ir_diagnostics : 'on').trim() !== 'off';
// The ladder needs both: L3 needs the trajectory, and it needs the ISA archive to
// tie that trajectory to the binary that was measured.
const LADDER_ENABLED = ISA_ENABLED && IR_DIAGNOSTICS;
// Optional read-only AMDGPU backend checkout for the compiler role's second tier.
// Empty by default and expected to stay empty: this image has no such sources, and
// the role is written to finish on runtime evidence or return inconclusive rather
// than reconstruct pass behaviour from memory.
const COMPILER_SOURCE_DIR = String(A.compiler_source_dir || '').trim();
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
// The INCUMBENT arm as re-measured in the candidate's OWN session. A deliberately narrower shape
// than `perCase`: a control arm has no speedup (it is the denominator measuring itself, so the
// number would be 1.0 by construction), and requiring a meaningless field is how a contract teaches
// an agent to omit the whole object.
//
// Why the lane needs this at all. `route_gate.py`'s header names the exposure and declines to guard
// it: "the same unchanged tree measures 1.5-3% differently between invocations, and the candidate
// and the incumbent it is compared against come from different invocations on whichever pool GPU was
// free. That exposure is real and larger than the per-round gains being judged... the tighter fix is
// not a device check but comparing against a control measured in the candidate's own session."
// The verifiers on this lane ALREADY build that arm -- interleaved, independently rebuilt -- and one
// of them caught a session-level -6.8% shift present in both arms. They just had nowhere to put it,
// so the gate went on comparing this round's candidate against a table measured in an earlier round.
const controlPerCase = {
  type: 'array',
  items: {
    type: 'object',
    properties: {
      name: { type: 'string' },
      // The time THIS arm measured. Named `optimized_ms` so `routeGate`/`route_gate.py` read a
      // control row with the same accessor they use for a candidate row -- one reader, not two.
      optimized_ms: { type: 'number' },
      samples_ms: { type: 'array', items: { type: 'number' } },
    },
    required: ['name', 'optimized_ms'],
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
    // (127). `cumulative` here is the LANE TOTAL VS THE ORIGINAL SEED, read back out of STATE.json.
    // It is NOT in the same frame as anything this run measures: this run's baseline is the tree
    // STATE_DIR/best just seeded, so this run's own speedups start at 1.0. Consumed only via
    // `priorCumulativeVsSeed`; never compare it to a verified_geomean.
    cumulative: { type: 'number' }, insights: { type: 'array', items: { type: 'string' } },
    ledger: { type: 'array', items: { type: 'object', additionalProperties: true } },
    bottleneck_now: { type: 'string' }, best_per_case: perCase,
  }, []),
  // Finding (67). sha256 over the immutable oracle (unittest.py, meta.json, reference_io.pt,
  // baseline_src/) recorded BEFORE any engineer runs. Candidates carry `source_hash` provenance;
  // the denominator and correctness reference carried none, so a write through the golden's
  // absolute symlink would silently rescore every measurement AND every archived elite.
  oracle_digest: { type: 'string' },
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
  // L3 IS NOW REQUESTED, NOT TRIGGERED. It used to fire on a stall counter, which meant the lane
  // paid for a lowering trajectory on the strength of "the last round did not help" -- a fact that
  // says nothing about whether the counters left a question a trajectory could answer. The role
  // that read the counters is the one that knows, so it asks.
  request_l3: { type: 'boolean' },
  request_l3_reason: { type: 'string' },
  // phase2 only. WHERE in the compiled assembly an edit is most likely to pay, localised by
  // `isa_signals.py keypoint` and ranked against the bottleneck this role just classified. This is
  // the one place the profiler hands the engineer a location rather than a direction, and it is why
  // phase2 is a local search instead of "here is the whole kernel, good luck".
  keypoints: { type: 'array', items: { type: 'object', additionalProperties: true } },
}, ['bottleneck', 'top_opportunities']);

// Finding (89). The baseline/reprofile profile branch has no SOL contract: no
// `sol_card.py` call, no ceiling provenance, no gate. It nevertheless
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
      + `has no calibrated roofline behind it, so it has no ceiling provenance to publish (89). `
      + `The measured counters are kept.`);
    rep.sol_note = 'SOL-shaped fields were removed: the roadmap profile has no ceiling '
      + 'provenance. Any headroom claim needs a measured ceiling, not a nameplate one.';
  }
  return rep;
};

// --- Deep Research Agent (DRA) schemas (opt-in via args.dra_enabled) -------
// Stage 0+1/2: the researcher extracts facts and returns a ranked set of research QUESTIONS the
// script then fans out in parallel. Stages 3/4: one answer per question (run concurrently). Stage 5:
// optional blindspot critique. Stage 7: the ranked-directions portfolio + brief the planner consumes.
const RESEARCH_PLAN_SCHEMA = obj({
  facts: { type: 'object', additionalProperties: true },
  questions: {
    type: 'array',
    items: obj({
      id: { type: 'string' }, question: { type: 'string' },
      search_queries: { type: 'array', items: { type: 'string' } },
      rationale: { type: 'string' }, tests_hypothesis: { type: 'string' },
      mode: { type: 'string', enum: ['bottleneck', 'design_space'] },
      rank_score: { type: 'number' },
    }, ['question']),
  },
  notes: { type: 'string' },
}, ['questions']);

const RESEARCH_QUESTION_SCHEMA = obj({
  question_id: { type: 'string' }, question: { type: 'string' },
  mode: { type: 'string' }, tests_hypothesis: { type: 'string' },
  answer: { type: 'string' },
  status: { type: 'string', enum: ['prefer', 'deprioritize', 'reject', 'open'] },
  affected: { type: 'array', items: { type: 'string' } },
  evidence: { type: 'array', items: { type: 'object', additionalProperties: true } },
  taskgen_implications: { type: 'string' }, notes: { type: 'string' },
}, ['answer', 'status']);

const RESEARCH_BLINDSPOT_SCHEMA = obj({
  blindspots: {
    type: 'array',
    items: obj({
      description: { type: 'string' }, why_it_matters: { type: 'string' },
      follow_up_question: { type: 'string' },
    }, ['description']),
  },
}, ['blindspots']);

// Stage 7 final portfolio. `directions` is kept COMPACT here (mirrors deep_search_brief.md): the
// planner reads the brief on disk, so this structured echo is just for logging/validation.
const RESEARCH_SCHEMA = obj({
  num_questions: { type: 'number' }, num_directions: { type: 'number' },
  brief_path: { type: 'string' }, full_path: { type: 'string' }, json_path: { type: 'string' },
  directions: {
    type: 'array',
    items: obj({
      id: { type: 'string' }, title: { type: 'string' },
      specialty: { type: 'string', enum: ['algorithm', 'memory', 'compute', 'host_runtime', 'deep_explore'] },
      mechanism: { type: 'string' }, expected_upside: { type: 'string' },
      confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
      rank_score: { type: 'number' },
    }, ['title']),
  },
  notes: { type: 'string' },
}, ['num_directions', 'brief_path']);

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
      // Learned cards that SEEDED this direction, by filename. Structural attribution: the planner
      // declares what it opened, the script joins that against what the VERIFIER independently
      // measured. Declared here rather than inferred because the read path is semantic — the planner
      // reads INDEX.md and judges by meaning, so nothing downstream can reconstruct which card it
      // acted on. A citation is not a causal claim (the planner saw the profile too, and one run
      // holds no counterfactual); it is the only way a card can ever LOSE standing.
      learned_refs: { type: 'array', items: { type: 'string' } },
    }, ['id', 'title', 'specialty', 'prompt']),
  },
  // The planner's L3 request, which necessarily applies to the NEXT round: the ladder runs before
  // planning so an attribution can inform the plan, and a request made during planning is therefore
  // one round late by construction. That is the honest shape of it -- "I cannot choose the next
  // direction until I know which pass ate the last one" is a statement about the round after this.
  request_l3: { type: 'boolean' },
  request_l3_reason: { type: 'string' },
}, ['stop', 'directions']);

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
// `files` is REQUIRED and `inspected` is not the same question: `policyReject` gates on
// `files`, because `inspected` counts directory entries and so reads 1 for an empty tree.
// Declaring only `inspected` is what made a receipt carrying every declared field, all
// healthy, come back `policy:inspected_nothing(files=undefined)` -- a refusal that reads as
// "the agent scanned nothing" when the agent was never asked for the field being gated on.
const POLICY_SUMMARY_SCHEMA = obj({
  schema: { type: 'string' }, passed: { type: 'boolean' },
  findings: { type: 'number' }, advisory: { type: 'number' },
  inspected: { type: 'number' }, files: { type: 'number' },
  elf: { type: 'number' }, unreadable: { type: 'number' },
}, ['passed', 'findings', 'inspected', 'files', 'elf']);
// (87). The exit code is the field with teeth, and it has three meanings, not
// two: 0 in lockstep, 1 drifted, 2 nothing was checked. `pairs` and `drifted`
// are carried because they let the orchestrator refuse a receipt that could not
// have come from running the tool -- exit 0 with zero pairs, or exit 0 with
// drift. See `twinReject`.
const HIP_TWIN_SCHEMA = obj({
  exit_code: { type: 'number' }, pairs: { type: 'number' }, drifted: { type: 'number' },
  checked: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
}, ['exit_code', 'pairs']);
// The ISA receipt. `mechanism_verdict` is a three-valued ENUM rather than a
// boolean because the third value is the load-bearing one: `indeterminate` means
// the archive did not carry the evidence needed to judge the claim, and a schema
// that could only say true or false would force that case into `false` -- which
// manufactures the very false negative this receipt exists to prevent, now with a
// receipt behind it. `isa_signals.py::mechanism_verdict` owns the truth table.
const ISA_EVIDENCE_SCHEMA = obj({
  exit_code: { type: 'number' },
  archive: { type: 'string' }, parent_archive: { type: 'string' },
  source_hash: { type: 'string' }, parent_source_hash: { type: 'string' },
  mechanism_claims: { type: 'array', items: { type: 'string' } },
  mechanism_verdict: { type: 'string', enum: ['realized', 'refuted', 'indeterminate'] },
  unchanged_machine_code: { type: 'boolean' },
  claims_refuted: { type: 'array', items: { type: 'string' } },
  claims_indeterminate: { type: 'array', items: { type: 'string' } },
  high_findings: { type: 'number' }, notes: { type: 'string' },
}, ['exit_code', 'mechanism_verdict']);
// The deep-analysis return, shared by L3 (`ir`) and L4 (`compiler`).
//
// `status` has four values and each one is load-bearing:
//   attributed     a pass and a structure explain it, and they imply ONE rewrite
//                  family. Escalation stops here.
//   needs_compiler the pass and structure are named but not WHY, and the residue
//                  is a legality/pass-constraint question. This is the only value
//                  that routes to L4, and it routes there IN THIS ROUND.
//   inconclusive   the evidence was read and did not answer. A real finding.
//   unavailable    there was no evidence to read (capture failed, or the trajectory
//                  could not be tied to the measured binary). Distinct from
//                  `inconclusive` on purpose -- one tells the next round not to
//                  bother asking again, the other tells it to retry.
//
// `source_change_required` is what makes the analysis worth its cost: an attribution
// that does not end in a condition the next edit must satisfy is a compiler note,
// not a diagnosis.
//
// `attributed_pass` and `stage_transition` are the fields that keep this layer from
// decaying back into disassembly reading. The previous version of L3 returned
// `vgpr=152, scratch=0, global_load_bytes.max=4` -- true numbers naming no pass, from
// which no narrowed compiler question can be built, which is why L4 could only ever
// guess at the whole backend. `admissibleAttribution` below refuses a positive status
// that does not carry them.
const IR_ATTRIBUTION_SCHEMA = obj({
  status: { type: 'string', enum: ['attributed', 'needs_compiler', 'inconclusive', 'unavailable'] },
  depth: { type: 'string', enum: ['ir', 'compiler'] },
  archive: { type: 'string' }, source_hash: { type: 'string' },
  summary_path: { type: 'string' },
  // L3's core product: which pass, between which two stages, changed what.
  attributed_pass: { type: 'string' },
  stage_transition: { type: 'string' },
  structural_signature: { type: 'string' },
  // Set by L3 only when status is `needs_compiler`; both are required before L4 runs.
  suspected_passes: { type: 'array', items: { type: 'string' } },
  compiler_question: { type: 'string' },
  // Did the recompiled trajectory match the binary that was actually measured?
  provenance_ok: { type: 'boolean' },
  // L4 only: which tier produced the finding (0 knowledge, 1 remarks, 2 source).
  tier_used: { type: 'number' },
  signals_cited: { type: 'array', items: { type: 'string' } },
  diagnosis: { type: 'string' },
  source_change_required: { type: 'string' },
  ruled_out: { type: 'array', items: { type: 'string' } },
  confidence: { type: 'string' },
  gaps: { type: 'array', items: { type: 'string' } },
}, ['status', 'diagnosis']);
// The retrospective pass. `reason` is required alongside a zero count on purpose:
// "nothing qualified" is the normal outcome of a run, and a bare 0 is
// indistinguishable from an agent that skipped the work.
const ISA_SYNTHESIS_SCHEMA = obj({
  promoted: { type: 'number' }, anti_signals: { type: 'number' },
  path: { type: 'string' }, reason: { type: 'string' },
  entries: { type: 'array', items: { type: 'string' } },
}, ['promoted']);
const ENG_SCHEMA = obj({
  engineer_id: { type: 'string' }, specialty: { type: 'string' }, task: { type: 'string' },
  strategy: { type: 'string' }, speedup_geomean: { type: 'number' }, speedup_arithmetic: { type: 'number' },
  // Time-weighted ratio-of-sums vs the TRUE baseline (PRIMARY metric when workload_aligned).
  // = Σ weight_i / Σ (weight_i / speedup_i). Omitted on unweighted runs.
  speedup_weighted: { type: 'number' },
  per_case: perCase, status: { type: 'string' }, patch_file: { type: 'string' },
  strategies_tried: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
  // The routes this direction claims a mechanism on. Feeds the per-route commit gate: without it an
  // incidental gain on an unrelated route can be banked as the declared mechanism.
  target_routes: { type: 'array', items: { type: 'string' } },
  // What this edit should have done to the MACHINE CODE, in `isa_signals.py`'s
  // closed vocabulary. Declared by the engineer because the engineer is the only
  // party that knows what it changed, and declared BEFORE verify measures anything
  // so the claim cannot be fitted to the result. Present only when the lane runs
  // with isa_evidence=observe|gate.
  mechanism_claims: { type: 'array', items: { type: 'string' } },
}, ['status', 'speedup_geomean']);

const VERIFY_SCHEMA = obj({
  status: { type: 'string' }, correctness: { type: 'string' },
  verified_geomean: { type: 'number' }, verified_arithmetic: { type: 'number' },
  verified_weighted: { type: 'number' }, // time-weighted ratio-of-sums (PRIMARY when workload_aligned)
  per_case: perCase, variance_note: { type: 'string' }, notes: { type: 'string' },
  graph_safe: { type: 'string' },
  // The incumbent arm re-measured in THIS verification's session (see controlPerCase). Declared,
  // never required: absent it, the gate falls back to the stored table from an earlier round and
  // says which one it used, so a verifier that cannot afford the arm still produces a usable result.
  control_per_case: controlPerCase,
  // See ENG_SCHEMA. The verifier's value is the one the commit gate uses, because the verifier's
  // per_case is what becomes `bestPerCase`.
  target_routes: { type: 'array', items: { type: 'string' } },
  // Finding (67): the setup digest, recomputed independently at verify time. Two agents at two
  // different points in the run reporting the same oracle is materially stronger evidence than
  // one agent asserting it once -- and it is the only evidence available to a script with no FS.
  oracle_digest: { type: 'string' },
  // Declared, not just listed in `required` below. It is the strictest gate in the admission
  // filter (`r.ver.policy_pass === true`), and a required-but-undeclared property is a field the
  // agent is obliged to return and never told about. INTEGRATE_SCHEMA has always declared it.
  policy_pass: { type: 'boolean' },
  policy_receipts: { type: 'object', additionalProperties: true },
  // (69): the summary of `$VERIFY_DIR/policy_postbuild.json`, the scan that sees the ELFs.
  policy_postbuild: POLICY_SUMMARY_SCHEMA,
  // (87): `hip_twin_sync.py` over the measured tree -- did ninja compile what was edited?
  hip_twin_sync: HIP_TWIN_SCHEMA,
  // The next question: did the mechanism survive the compiler? Present only when
  // the lane runs with isa_evidence=observe|gate.
  isa_evidence: ISA_EVIDENCE_SCHEMA,
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

const COMMIT_SCHEMA = obj({
  // (127). `head_sha_before`/`head_sha_after` make the one claim that matters checkable by the
  // script: a commit that reports success while HEAD never moved is the exact shape in which two
  // consecutive rounds silently kept building on the previous round's tree.
  committed: { type: 'boolean' }, current_best_diff: { type: 'string' }, note: { type: 'string' },
  head_sha_before: { type: 'string' }, head_sha_after: { type: 'string' },
}, ['committed']);

const REPORT_SCHEMA = obj({
  final_speedup_geomean: { type: 'number' }, final_speedup_arithmetic: { type: 'number' },
  final_speedup_weighted: { type: 'number' }, // time-weighted ratio-of-sums (PRIMARY when workload_aligned)
  rounds: { type: 'number' }, budget_used: { type: 'number' },
  report_path: { type: 'string' }, final_patch: { type: 'string' }, per_case: perCase,
}, ['final_speedup_geomean', 'report_path', 'final_patch']);

const UPDATE_EXPERIENCE_SCHEMA = obj({
  action: { type: 'string' },      // created | merged | skipped
  card_path: { type: 'string' },   // path under knowledge/learned/, or "" if nothing distilled
  key: { type: 'string' }, note: { type: 'string' },
}, []);

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

// Warm-start resolver output = experience_store.py `resolve` JSON, verbatim.
const WARMSTART_RESOLVE_SCHEMA = obj({
  read_reason: { type: 'string' }, slug: { type: 'string' },
  // What this run's name derived to vs the page actually served, so a surprising match is visible.
  requested_slug: { type: 'string' }, match_tier: { type: 'string' },
  // store mode only: the key the candidates came from. Declared rather than left to
  // additionalProperties so the agent relaying this JSON has no reason to drop it.
  canonical_id: { type: 'string' },
  // Same kernel, another language: the wrong target_language was passed, not an empty store.
  other_language_pages: { type: 'array', items: { type: 'string' } },
  filtered: obj({
    total: { type: 'number' }, retired: { type: 'number' }, below_min_speedup: { type: 'number' },
    same_direction_collapsed: { type: 'number' },
  }),
  candidates: {
    type: 'array',
    items: obj({
      rank: { type: 'number' }, slug: { type: 'string' }, speedup: { type: 'number' },
      exp_dir: { type: 'string' }, arch: { type: 'string' },
      patch_path: { type: 'string' }, prose_path: { type: 'string' },
      strategy: { type: 'string' }, status: { type: 'string' },
      // direction = the IDEA (one rank each). comparable=false: recorded on a different bench key
      // than rank 1's, so the order between them is a prior only.
      direction: { type: 'string' }, bench_key: { type: 'string' }, comparable: { type: 'boolean' },
    }, ['rank', 'patch_path']),
  },
}, ['read_reason']);
// Warm-start writer output = experience_store.py `write` JSON, verbatim. written=false with reason
// "duplicate_impl" is a success: the store already held this code and counted a reproduction.
const WARMSTART_WRITE_SCHEMA = obj({
  written: { type: 'boolean' }, reason: { type: 'string' }, slug: { type: 'string' },
  dir: { type: 'string' }, speedup: { type: 'number' },
  reproduced: { type: 'string' }, reproductions: { type: 'number' },
  // store mode only: where the same result landed on the KB plane. `replaced` says which of the
  // two outcomes it was — a new candidate under the key, or this patch's own one measured again.
  remote: obj({
    written: { type: 'boolean' }, reason: { type: 'string' }, canonical_id: { type: 'string' },
    session_id: { type: 'string' }, champion: { type: 'boolean' }, replaced: { type: 'boolean' },
  }),
}, ['written']);

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

// ---------------------------------------------------------------------------
// WALL-CLOCK DEADLINE (opt-in; absent => byte-identical to a build without it).
//   DEADLINE_EPOCH  unix seconds after which NO new optimization round may start.
//   NO_STOP_S       while more than this many seconds remain, the TechLead may not end the run on
//                   its own convergence gates; it gets exactly one forced re-plan.
// In CODE and not in the caller's prose because the prose version was measured: a "fixed 4h window
// per kernel" comparison whose window had no implementation anywhere realised 1.55h to 15.22h, a
// 9.81x spread, and the comparison it existed to make was void. The loop below is
// `while (dispatched < BUDGET && noImprove < MAX_NO_IMPROVE)` — no time term. Emphatic prompt text
// is not a constraint, it is one more input to a judgement call.
// A workflow script has no clock (Date.now() is unavailable — it would break resume), so the time is
// read by a tiny low-effort agent. If that read FAILS the run is NOT killed: an unreadable clock
// returns Infinity, degrading to the old budget-only behaviour rather than truncating a healthy run.
const DEADLINE_EPOCH = (() => {
  const v = parseInt(A.deadline_epoch != null ? A.deadline_epoch : 0, 10);
  return Number.isFinite(v) && v > 0 ? v : 0;
})();
const NO_STOP_S = Math.max(0, parseInt(A.no_stop_s != null ? A.no_stop_s : 900, 10));
// Run-level cap on how many TechLead stops may be refused (forcedReplans is not reset per round).
// Not 1: the refusal must scale with the window, or a long budget is handed back early.
// Not unbounded: a role that truly cannot name a direction would spend the window planning.
const MAX_FORCED_REPLANS = Math.max(1, parseInt(A.max_forced_replans != null ? A.max_forced_replans : 6, 10));
const CLOCK_SCHEMA = { type: 'object', properties: { epoch: { type: 'number' } },
                       required: ['epoch'], additionalProperties: true };
let deadlineHit = false;
let forcedReplans = 0;

async function secondsLeft(tag) {
  if (!DEADLINE_EPOCH) return Infinity;
  const r = await agentT(
    `Run EXACTLY this command and nothing else:
\`\`\`bash
date +%s
\`\`\`
Return {"epoch": <the integer it printed>}. Do NOT modify any file and do NOT run anything else.`,
    { phase: 'Optimize', label: `clock ${tag}`, effort: 'low', schema: CLOCK_SCHEMA });
  if (!r || !Number.isFinite(r.epoch)) {
    log(`  [deadline] clock read failed at ${tag} — treating time as unlimited for this check.`);
    return Infinity;
  }
  return DEADLINE_EPOCH - r.epoch;
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
// The EFFECTIVE value of every knob that decides what this run admits, with `(default)` marked, and
// emitted before the first agent call.
//
// This exists because of a specific, expensive failure that no whitelist can catch. This lane's
// protocol pins `min_improve: 0.005`; one wave's invocation was retyped without that key, so the
// wave ran at the 0.02 default and refused a verified, correctness-passing, policy-clean +1.58%
// integrated stack -- the largest result of two days' work. Nothing in the run said which threshold
// was live, so the omission was only recoverable afterwards by reading the refusal arithmetic
// backwards. An argument that did not arrive looks exactly like one whose default was intended, and
// the only cure is to print what is actually in force.
//
// Cheap, unconditional, and every value here is one a reader can check against the protocol they
// meant to run -- which is the whole point: the log has to be falsifiable against the intent.
{
  const shown = (key, value, extra) =>
    `${key}=${value}${A[key] == null ? ' (default)' : ''}${extra ? ` ${extra}` : ''}`;
  log('Effective run configuration (the knobs that decide what this run ADMITS):');
  log(`  ${shown('budget', BUDGET)}  ${shown('deep_cost', DEEP_COST)}  ` +
      `${shown('max_no_improve', MAX_NO_IMPROVE, '(per PHASE, then escalate; last phase ends the run)')}`);
  // The ladder is the run's shape, so it is stated before the run rather than inferred from the
  // escalation lines afterwards. Phases with no implementation are named as skipped HERE, so nobody
  // waits for an assembly phase this build cannot enter.
  log(`  search_phases=${PHASE_ORDER.map(p =>
        `${p}${PHASE_IMPLEMENTED[p] ? '' : ' (SKIPPED: not implemented)'}`).join(' -> ')}` +
      `  one-way, ${MAX_NO_IMPROVE} stalled round(s) each`);
  log(`  ${shown('min_improve', MIN_IMPROVE)}  ${shown('candidate_floor', CANDIDATE_FLOOR)}  ` +
      `${shown('progress_delta', PROGRESS_DELTA)}` +
      `${A.progress_delta == null ? ' (= min_improve)' : ''}`);
  log(`  ${shown('isa_evidence', ISA_MODE)}  ${shown('mode', MODE)}  ` +
      `${shown('gpu_mode', GPU_MODE)}  gpu_ids=${GPU_IDS}`);
  log(`  ${shown('ir_diagnostics', IR_DIAGNOSTICS ? 'on' : 'off')}` +
      `  evidence_ladder=${LADDER_ENABLED ? 'L1..L4' : 'L1..L2 (no IR trajectory, so no L3/L4)'}`);
  // Stale text is worse here than no text: this echo exists so a reader can see what will decide
  // the round before any GPU time is spent, and it went on promising a derivation that had been
  // deleted. `not supplied` is not a degraded state any more -- it means every route is held to
  // MIN_ROUTE_WIN -- so the line has to say which of the two is in force.
  log(`  route_bands=${ROUTE_BANDS_ARG
    ? `${Object.keys(ROUTE_BANDS_ARG).length} measured floor(s) supplied; a route is held to `
      + `max(${(MIN_ROUTE_WIN * 100).toFixed(2)}%, its own floor)`
    : `not supplied -- every route is held to the ${(MIN_ROUTE_WIN * 100).toFixed(2)}% default. `
      + 'On a box where a route measures noisier than that, its movement will read as a mechanism '
      + 'when it is noise; pass a measured {route: floor} table inline to fix that'}`);
  log(`  state_dir=${STATE_DIR || '(none -- this is a COLD start, not a continuation)'}`);
  log(`  workload_aligned=${HAS_WORKLOAD ? 'yes (PRIMARY metric is the time-weighted ratio-of-sums)'
    : 'no (PRIMARY metric is the unweighted geomean)'}`);
  if (ISA_MODE_WARNING) log(`  [isa] WARNING: ${ISA_MODE_WARNING}`);
}
const setup = await agentT(
  roleAgent('director', 'setup', 'Build the isolated evaluation environment.', {
    KERNEL_PATH_ORIG, TASK_DIR: KERNEL_PATH_ORIG,
    EXP_ROOT, EVAL_DIR_OVERRIDE, KERNEL_NAME_HINT, TASK, SKILL_DIR: WORKFLOW_DIR,
    MODE, TARGET_LANGUAGE, OP_SPEC,
    ...(STATE_DIR ? { STATE_DIR } : {}),
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
      // The author engineer reads the learned index too, so the switch has to reach it. It is the
      // second reader; a switch that covers one of two readers is not a switch.
      LEARNED_KB: USE_LEARNED_READ ? 'on' : 'off',
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

// --- What the commit gate will hold each route to ---------------------------
// The gate always runs now. With no measured table every route is held to MIN_ROUTE_WIN; a table
// raises the bar on the routes a sweep found noisier than that, and on nothing else. There is no
// longer a "the gate is off" state to fall back from, which is the point: for seven waves the
// per-route gate logged nothing at all while looking like a shipped feature, because the only way
// to feed it was a file nobody passed.
const ROUTE_BANDS = ROUTE_BANDS_ARG;
{
  const bar = (r) => Math.max(MIN_ROUTE_WIN, (ROUTE_BANDS && ROUTE_BANDS[r]) || 0);
  const names = (BASELINE_PER_CASE || []).map(r => r && (r.name || r.test_case_id)).filter(Boolean);
  const raised = names.filter(r => bar(r) > MIN_ROUTE_WIN).sort();
  log(`Commit gate: suite geomean must improve at all AND >=1 route must clear ` +
      `${(MIN_ROUTE_WIN * 100).toFixed(2)}% (or its own measured floor, whichever is larger); ` +
      `any route past -${(CATASTROPHIC_REGRESSION * 100).toFixed(0)}% refuses regardless. ` +
      (ROUTE_BANDS
        ? `Floor table supplied for ${Object.keys(ROUTE_BANDS).length} route(s); it raises the bar on ` +
          (raised.length
            ? `${raised.length}: ${raised.map(r => `${r} ${(bar(r) * 100).toFixed(2)}%`).join(', ')}.`
            : 'none of this suite\'s routes (every measured floor is below the default).')
        : 'No floor table supplied, so every route is held to the default. On a box where a route ' +
          'measures noisier than the default, that route\'s movement is being read as a mechanism ' +
          'when it is noise; pass a measured {route: floor} table inline via args.route_bands.'));
}

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
// PHASE: Research (Deep Research Agent — OPT-IN via args.dra_enabled)
// Runs AFTER Profile / BEFORE the optimize loop: profile + COMMANDMENT + analysis exist by now, so
// the researcher has the facts it needs. It produces EVAL_DIR/deep_search_brief.md (compact, ranked
// directions) which the TechLead's plan_round seeds directions from. The per-question research is
// fanned out with parallel() and EVERY research agent is wrapped in the agentT() hang-guard, so a
// hung research agent resolves to null and the parallel round-barrier still proceeds (it never wedges
// the run — the known v4 failure mode the hang-guard was built for). DEFAULT OFF → no behavior change.
// ===========================================================================
let researchBriefPath = '';   // EVAL_DIR/deep_search_brief.md when the DRA produced one; '' otherwise
if (DRA_ENABLED) {
  phase('Research');
  const RESEARCH_DIR = `${EVAL_DIR}/research`;

  // --- Stage 0 + 1/2: extract facts, generate + rank research questions (one agent) -------------
  const plan = await agentT(
    roleAgent('researcher', 'research_plan', 'Extract facts and propose ranked research questions.', {
      WORKSPACE: CANONICAL, EVAL_DIR, RESEARCH_DIR, COMMANDMENT, SKILL_DIR: WORKFLOW_DIR,
      ANALYSIS_JSON: `${EVAL_DIR}/analysis.json`, CODEBASE_CONTEXT: `${EVAL_DIR}/codebase_context.md`,
      PROFILING_SUMMARY: profileSummary ? profileSummary.summary_path : '',
      BOTTLENECK: profileSummary ? profileSummary.bottleneck : 'unknown',
      BASELINE_PER_CASE, MAX_QUESTIONS: DRA_MAX_QUESTIONS, TASK,
      KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE, KK_REFS,
    }),
    { phase: 'Research', label: 'researcher:plan', schema: RESEARCH_PLAN_SCHEMA });

  const facts = (plan && plan.facts) || {};
  let questions = (plan && Array.isArray(plan.questions) ? plan.questions : [])
    .slice(0, DRA_MAX_QUESTIONS)
    .map((q, i) => ({ ...q, id: q.id || `q${i}` }));
  log(`Research: ${questions.length} question(s) planned, bottleneck=${facts.bottleneck_type || '?'}`);

  // --- Stages 3/4: research each question IN PARALLEL (native WebSearch/WebFetch), hang-guarded ---
  // parallel() takes an array of zero-arg async thunks and runs them concurrently; each thunk's
  // agentT() bounds the research agent so a hang resolves null instead of blocking the barrier.
  const researchQuestion = (q, stageLabel) => agentT(
    roleAgent('researcher', 'research_question',
      'Research THIS ONE question on the live web and synthesize one judgment.', {
        QUESTION: q, FACTS: facts, RESEARCH_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR,
        ANSWER_OUT: `${RESEARCH_DIR}/answers/${q.id}.json`,
        CODEBASE_CONTEXT: `${EVAL_DIR}/codebase_context.md`,
        PROFILING_SUMMARY: profileSummary ? profileSummary.summary_path : '',
        KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE, KK_REFS,
      }),
    { phase: 'Research', label: `researcher:q ${q.id}${stageLabel ? ' ' + stageLabel : ''}`, schema: RESEARCH_QUESTION_SCHEMA });

  let answers = (await parallel(questions.map((q) => () => researchQuestion(q))))
    .filter(Boolean);
  log(`Research: ${answers.length}/${questions.length} first-pass answers returned`);

  // --- Stages 5/6 (optional): blindspot critique + a second parallel research wave --------------
  if (DRA_BLINDSPOT && answers.length) {
    const crit = await agentT(
      roleAgent('researcher', 'research_blindspot', 'Critique the research and surface new blindspots.', {
        FACTS: facts, RESEARCH_DIR, ANSWERS: answers, MAX_BLINDSPOTS: DRA_MAX_BLINDSPOTS,
        SKILL_DIR: WORKFLOW_DIR,
      }),
      { phase: 'Research', label: 'researcher:blindspot', schema: RESEARCH_BLINDSPOT_SCHEMA });
    const followups = (crit && Array.isArray(crit.blindspots) ? crit.blindspots : [])
      .filter(b => b && b.follow_up_question)
      .slice(0, DRA_MAX_BLINDSPOTS)
      .map((b, i) => ({ id: `b${i}`, question: b.follow_up_question, mode: 'bottleneck',
        tests_hypothesis: '', search_queries: [] }));
    if (followups.length) {
      const more = (await parallel(followups.map((q) => () => researchQuestion(q, 'blindspot'))))
        .filter(Boolean);
      answers = answers.concat(more);
      log(`Research: blindspot pass added ${more.length} answer(s) from ${followups.length} follow-up(s)`);
    }
  }

  // --- Stage 7: synthesize the ranked-directions portfolio + the compact planner brief -----------
  const synth = await agentT(
    roleAgent('researcher', 'research_synthesize',
      'Synthesize the ranked portfolio of optimization directions and the compact brief.', {
        FACTS: facts, RESEARCH_DIR, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR,
        BRIEF_OUT: `${EVAL_DIR}/deep_search_brief.md`, FULL_OUT: `${EVAL_DIR}/deep_search.md`,
        JSON_OUT: `${EVAL_DIR}/deep_search.json`,
      }),
    { phase: 'Research', label: 'researcher:synthesize', schema: RESEARCH_SCHEMA });

  if (synth && synth.brief_path) {
    researchBriefPath = synth.brief_path;
    log(`Research done. ${synth.num_directions || (synth.directions ? synth.directions.length : 0)} ranked direction(s) → ${researchBriefPath}`);
  } else {
    log('Research produced no brief (degraded) — plan_round proceeds without a DRA brief.');
  }
}

// ===========================================================================
// PHASE: Optimization loop (budget-controlled)
// ===========================================================================
let dispatched = 0;          // counts ONLY optimization-direction engineers (the budget)
let round = 0;
// `cumulative` is in THIS WAVE'S FRAME: speedup vs BASELINE_PER_CASE, which the benchmark
// engineer measured on CANONICAL at the top of this run. On a resumed wave CANONICAL already
// IS the prior waves' cumulative best, so this correctly starts at 1.0 on every wave.
let cumulative = 1.0;        // best verified geomean speedup vs the TRUE baseline
// Finding (127). The OTHER frame, kept strictly separate: total speedup vs the seed the lane
// started from, several waves back. It is a REPORTING number and must never be compared against
// a verifier's `verified_geomean`, which is always wave-local. Mixing the two is what made a
// resumed wave compare 1.01 (vs-incumbent) against 4.35 (vs-seed) and conclude IMPROVED=false
// on every round of a wave that in fact improved every round -- see the resume block below.
let priorCumulativeVsSeed = 1.0;
// Which frame `cumulative` is actually in -- DECIDED FROM DATA after the baseline is measured, not
// assumed. See `resolveVsSeedFrame` below the resume block for why the assumption above is not safe
// to make: on a task whose harness times against a frozen external oracle, `verified_geomean` is an
// ABSOLUTE number and chaining it onto the prior wave's absolute number multiplies two of the same
// thing. 'chained' is the historical behaviour and stays the default for the case where the two
// hypotheses agree anyway (a fresh wave, where prior is 1.0).
let vsSeedFrame = 'chained';
const cumulativeVsSeed = () => (vsSeedFrame === 'absolute'
  ? cumulative                       // `cumulative` already IS the vs-seed total
  : priorCumulativeVsSeed * cumulative);
let bestSeen = 0;            // best verified geomean of any candidate, committed or not
// Consecutive non-improving rounds IN THE CURRENT PHASE. Reset on progress and on every phase
// change, so each phase is judged on its own rounds rather than inheriting the stall that promoted
// it -- a phase that started life already one round from its own exhaustion would get less search
// than the one it replaced, which is backwards.
let phaseStall = 0;
let searchPhase = PHASE_SOURCE;
// How many times the ladder has come back around to the source phase. Each cycle after the first
// runs on a tree a deep round rewrote, so "round 12" alone does not say what was being searched.
let cycle = 1;
// Why the run ended, set once. '' while running. Distinguishing these downstream matters: a ladder
// that ran out of phases and a lane that ran out of budget look identical in a report that only
// says "stopped".
let stopReason = '';
const phaseLog = [{ phase: PHASE_SOURCE, entered_round: 1, cycle: 1, reason: 'start' }];

// Hand the search to the next state. Used from BOTH triggers -- the planner reporting the level
// finished, and the stall backstop -- so the two cannot drift into doing different things.
const advancePhase = (why, atRound) => {
  const next = nextLivePhase(searchPhase);
  if (next === PHASE_SOURCE) cycle++;
  log(`PHASE ${searchPhase} -> ${next}${next === PHASE_SOURCE ? ` (cycle ${cycle})` : ''}: ${why}. `
    + (next === PHASE_SOURCE
      ? 'The deep round left a different program behind, so the ladder restarts on it: the source '
        + 'findings and keypoints from the last cycle describe a tree that no longer exists.'
      : 'The stall counter resets; the new state is judged on its own rounds.'));
  phaseLog.push({ phase: next, entered_round: atRound + 1, cycle, reason: why });
  searchPhase = next;
  phaseStall = 0;
};
// The ISA archive of the tree that is currently canonical, i.e. the parent every
// candidate in the next round is a mutation of. Null until a committed winner
// supplies one, and reset to null -- never left stale -- whenever the new canonical
// has no archive. A stale value here is the one failure mode worth designing
// against: it would diff a candidate against a tree that is not its parent and
// report confident, wrong claim verdicts, which is strictly worse than the honest
// `indeterminate` a null produces.
let isaCanonicalArchive = null;
// The source hash of the tree `isaCanonicalArchive` actually describes, tracked BESIDE the path so
// the pairing can be checked rather than assumed. Without it the archive is a path with a claim
// attached to it by position in the code, and the stale-parent failure the comment above warns about
// is invisible; with it, a verifier reporting a different parent hash makes the disagreement loud.
let isaCanonicalSourceHash = null;
let bestPerCase = BASELINE_PER_CASE;
let finalWinner = null;      // {geomean, arithmetic, per_case, patch, source} — also set by a warm-start adopt
let roundsCommitted = 0;     // rounds this run actually landed; a warm-start adopt is NOT one of them
const history = { insights: [], ledger: [], rounds: [], bottleneck_now: profileSummary ? profileSummary.bottleneck : 'unknown', suggest_next: '' };
// One row per KB-seeded direction, joined against what the verifier measured (see the push site in
// the round loop). Fed to update_experience and returned to the caller: a driver aggregating these
// is how anyone notices the KB has been cited fifty times and never once carried a round.
const citations = [];

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
const policyReject = (rep, label) => {
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
// Scope, and it is narrow on purpose. The twin only exists because torch's
// cpp_extension hipifies `.hip` SOURCES; a Triton lane -- which is the default
// `target_language` -- has no `X.hip`/`X_hip.hip` pair to compare and would
// return exit 2 forever. Requiring the receipt there would refuse every
// candidate in the lane for a hazard that cannot occur in it, which is not
// fail-closed, it is fail-shut. So the gate is armed by language, and the
// arming condition is a literal the parity test can read.
// The ISA gate, the layer above `twinReject`. Twin-sync answers "was the edited
// file compiled"; this answers "did the mechanism survive the compiler". Returns a
// refusal string, or null when there is nothing to refuse.
//
// It refuses on exactly ONE thing: the candidate's own receipt contradicting the
// candidate's own declared mechanism. It deliberately does NOT refuse on a HOLE, a
// missing archive, an indeterminate verdict, or a `checks` finding:
//
//   - A HOLE or a missing tool is a gap in OUR evidence, not a fault in the
//     candidate. Refusing there would discard correct fast kernels on boxes where
//     ROCm sits somewhere `isa_capture.py` did not look, and it would do it
//     silently in the direction that destroys good work. That is the opposite
//     asymmetry from (87), where exit 2 MUST refuse -- because there, the missing
//     evidence is evidence about the measurement itself, and an unbacked
//     measurement manufactures null results. Here the measurement is already
//     backed by twin-sync, policy and the oracle; the ISA layer only explains it.
//   - `checks` findings are advisory by construction (a narrow load is correct for
//     a genuine gather). A rule that refused its own false positives would be
//     worse than no rule.
//
// So a run with `isa_evidence=gate` can only ever LOSE candidates that are
// provably not doing what they said. Everything else logs and proceeds.
const isaEvidenceReject = (rep, label) => {
  if (!ISA_ENABLED || ISA_MODE !== 'gate') return null;
  const where = label || 'verify';
  // Same scoping rule as `policyReject` and `twinReject`: only reports
  // claiming a pass. A verifier that already failed is refused on its own terms,
  // and stacking a paperwork complaint on top renames the refusal (60).
  if (!rep || rep.status === 'failed' || rep.correctness === 'FAIL') return null;
  // Named `receipt` rather than `s` for the same reason as the log site further
  // down: that variable name plus this shape of guard is what `test_js_suite.py`'s
  // mutation probes anchor on, and an extra copy of it moves the anchor off the
  // gate it is meant to watch.
  const receipt = rep.isa_evidence;
  // A missing receipt is NOT refused here, for the reason above: the gate is about
  // the candidate, and an absent receipt is about us. It is logged by the caller.
  if (!receipt || typeof receipt !== 'object') return null;
  if (receipt.mechanism_verdict === 'refuted') {
    const why = receipt.unchanged_machine_code
      ? 'the machine code is byte-identical to its parent, so the edit changed nothing that runs'
      : `the ISA contradicts the declared mechanism (${(receipt.claims_refuted || []).join(', ') || 'no claim survived'})`;
    return `isa:mechanism_refuted(${where}: ${why}. This is NOT a slow kernel and must not be `
      + 'recorded as one: the mechanism was never actually tested, so the ledger must say '
      + '"not realized", not "no effect")';
  }
  return null;
};

// The escalation ladder's decision, as a named function so the JS suite can EXECUTE
// it against fabricated histories. The two gates above were extracted for exactly
// this reason; the one that stayed inline was defended only by a regex asserting its
// log message existed, and `audit_pin_coverage.py` is what surfaced that. A rule
// about when to spend the run's most expensive evidence is not a good candidate for
// being the third.
//
// The four rungs, named after the paper's levels so the record can be read against
// it. L1 and L2 are RECORDED but never requested: pattern triage and profiling
// happen every round regardless, and a ladder that pretended to gate them would be
// describing a decision it does not make.
const STAGE_L1 = 'L1_pattern';
const STAGE_L2 = 'L2_profile';
const STAGE_L3 = 'L3_ir';
const STAGE_L4 = 'L4_compiler_source';
const STAGE_FAILED = 'requested_but_unreached';
// Rounds banked before this file was versioned carry `evidence_depth` with the OLD
// vocabulary, where `isa` meant "read the disassembly" and `compiler` meant "asked
// why the machine code looked like that". Neither is L3 or L4 in the current sense,
// and a resumed lane that read them as such would believe it already holds evidence
// nobody ever produced. They get their own value, which satisfies no rung.
const STAGE_LEGACY = 'legacy_machine_code';
const EVIDENCE_MODEL = 'geak.evidence-ladder/v2';

// What a prior round actually reached, tolerant of both record formats.
const reachedStageOf = (round) => {
  if (!round) return STAGE_L1;
  const ev = round.evidence;
  if (ev && ev.model === EVIDENCE_MODEL) return ev.reached_stage || STAGE_L1;
  if (round.evidence_depth && round.evidence_depth !== 'pattern') return STAGE_LEGACY;
  return STAGE_L2;
};

// Whether L2 handed up something specific enough to attribute. The ladder's first
// test in the paper is "has the current level localised a dominant bottleneck" --
// not "is it still slow" -- and L3 needs a named kernel to trace, since a trajectory
// is captured per function. Without one the escalation would spend a recompile to
// produce a trajectory of whatever translation unit was guessed at.
const hasDominantHotKernel = (profile) => {
  if (!profile || typeof profile !== 'object') return false;
  const kernels = Array.isArray(profile.top_kernels) ? profile.top_kernels : [];
  return kernels.some(k => k && typeof k.name === 'string' && k.name.trim().length > 0);
};

// The escalation decision, as a named function so the JS suite can EXECUTE it
// against fabricated histories. Returns {requested, from, reason, skip_reason}
// where `requested` is null or STAGE_L3.
//
// L4 IS NOT REQUESTABLE HERE, and that is the fix rather than an omission. The
// previous ladder could only reach `compiler` from a round whose prior depth was
// already `isa`, which needed one stalled round on the L3 round and then survival
// into the next -- but back when the stall counter ended the WAVE at 2, a
// non-improving L3 round took it to 2 and the loop exited first. The L3 -> L4 chain
// was unreachable under the default stop budget, and the only live path to L4 was
// `priorRefuted`, which SKIPPED L3 entirely. So the ladder's deepest rung was
// reached exclusively by the route that gave it nothing to work with.
//
// L4 is now entered from within the same round, immediately after L3 returns
// `needs_compiler` with a pass and a question. That is also what the paper
// describes: escalation to compiler source is a continuation of the attribution,
// not a separate round's decision.
//
// WHO MAY ASK, and why it is asking rather than a counter.
//
// L3 used to fire on the stall counter: one non-improving round bought a lowering trajectory. That
// conflates two different things. "The last round did not help" is a fact about the SEARCH; "the
// counters have left a question only the lowering trajectory can answer" is a fact about the
// EVIDENCE, and only the second is a reason to pay for L3. The roles that read the evidence are the
// ones that can tell, so the default path through a phase-1 round is L1 -> L2, and L3 runs when
// somebody asks for it by name.
//
// Three requesters, and the third is the one to argue about:
//   profile_engineer  asks in the same round -- it has just read the counters.
//   tech_lead         asks during planning, which lands one round later by construction: the ladder
//                     runs BEFORE planning so an attribution can inform the plan.
//   the verifier      does not "ask" in words, but a REFUTED mechanism is the machine code already
//                     answering "did my edit land" with no, and the open question -- which pass
//                     undid it -- is exactly an L3 question. Kept as an automatic requester because
//                     neither role above necessarily sees the refutation.
const l3Requesters = (prior, profile) => {
  const out = [];
  if (profile && profile.request_l3 === true) {
    out.push({ by: 'profile_engineer', reason: String(profile.request_l3_reason || '').trim() });
  }
  if (prior && prior.plan_request_l3 === true) {
    out.push({ by: 'tech_lead', reason: String(prior.plan_request_l3_reason || '').trim() });
  }
  // Routed TO L3, not past it. "My edit did not reach the binary" is first of all a question about
  // WHICH PASS undid it; only if L3 names the pass and cannot say why does it become a
  // compiler-source question. The old code jumped straight to the compiler here, forced by L3
  // having no pass-level capability to jump to.
  if (prior && Array.isArray(prior.results) && prior.results.some(r => r && r.mechanism === 'refuted')) {
    out.push({
      by: 'verifier',
      reason: 'the previous round\'s machine code refuted the mechanism it declared, so the open '
        + 'question is which lowering stage undid it -- a pass-level question, not yet a '
        + 'compiler-source one',
    });
  }
  return out;
};

// `phase` is REQUIRED. L3 recompiles from source in a scratch tree and L4 reads the compiler's own
// source, so both answer the question "how does this SOURCE become this machine code" -- and only
// one of the three states is searching source. The gate is not a style preference, it is the same
// provenance rule the ISA archive already enforces one level down:
//
//   phase2_asm    the measured binary came out of asm_roundtrip.py, i.e. a hand-edited .s. A
//                 trajectory recompiled from source describes a DIFFERENT binary, and an
//                 attribution to a neighbouring binary reads exactly like a real one. The phase
//                 has its own instrument for this -- isa_signals.py keypoint, which reads the
//                 artifact rather than rebuilding one.
//   deep_explore  the round exists to replace the tree. Evidence bought here describes the
//                 implementation that is about to stop existing.
//
// A request made in the wrong phase is SKIPPED and named, not deferred. Carrying it into the next
// phase-1 round would re-ask it about a tree that a deep rewrite had meanwhile replaced, which is
// the same stale-evidence failure one cycle further out.
const evidenceLadder = (rounds, profile, enabled, phase) => {
  if (!enabled) return { requested: null, from: '', reason: '', skip_reason: '', requested_by: [] };
  const list = Array.isArray(rounds) ? rounds : [];
  const prior = list.length ? list[list.length - 1] : null;
  const from = reachedStageOf(prior);
  const asks = l3Requesters(prior, profile);
  if (!asks.length) return { requested: null, from, reason: '', skip_reason: '', requested_by: [] };
  if (phase !== PHASE_SOURCE) {
    return {
      requested: null, from, reason: '', requested_by: asks.map(a => a.by),
      skip_reason: `${asks.map(a => a.by).join(' and ')} asked for L3, but this is a ${phase} round `
        + (phase === PHASE_ASM
          ? 'and the binary that was measured came from an edited assembly, so a trajectory '
            + 'recompiled from source would describe a different program -- the question this phase '
            + 'can answer is a KEYPOINT in the artifact that ran'
          : 'and the round exists to replace this tree, so a lowering trajectory would attribute '
            + 'the plateau of an implementation that is about to stop existing'),
    };
  }
  if (!hasDominantHotKernel(profile)) {
    return {
      requested: null, from, reason: '', requested_by: asks.map(a => a.by),
      skip_reason: `${asks.map(a => a.by).join(' and ')} asked for L3, but L2 has not named a `
        + 'dominant hot kernel, and a trajectory is captured per function -- without one, L3 would '
        + 'trace whichever translation unit was guessed at and attribute this plateau to it',
    };
  }
  return {
    requested: STAGE_L3, from,
    requested_by: asks.map(a => a.by),
    // Every requester's stated reason travels, not just the first. Two roles asking for different
    // reasons is the case where L3 is most clearly worth its cost, and an escalation record that
    // kept one of them would misreport exactly that round.
    reason: asks.map(a => `${a.by}: ${a.reason || '(no reason given)'}`).join(' | '),
    skip_reason: '',
  };
};

// A positive attribution is only admissible if it carries the two fields that make
// it one. This is the schema-level half of the rule the L3 role states in prose: the
// ISA archive may corroborate a conclusion, never be its source. An `attributed`
// return citing machine-code statistics and naming no pass is precisely the old
// behaviour, and it would sail through a check that only looked at `status`.
const admissibleAttribution = (attribution) => {
  if (!attribution || !attribution.diagnosis) return '';
  const status = String(attribution.status || '');
  if (status !== 'attributed' && status !== 'needs_compiler') return '';
  if (!String(attribution.attributed_pass || '').trim()) {
    return `status "${status}" with no attributed_pass: an attribution that names no pass cannot `
      + 'be escalated and cannot be acted on -- it is a machine-code statistic wearing a '
      + 'diagnosis\'s label';
  }
  if (!String(attribution.stage_transition || '').trim()) {
    return `status "${status}" with no stage_transition: without the two stages a reader cannot `
      + 're-run the finding, and an unverifiable attribution is the one output this layer must '
      + 'not produce';
  }
  return '';
};

// Whether L3's return actually earns an L4 escalation. Both fields required: L4's
// whole cost control is that it is handed one question with a stopping condition.
const wantsCompilerEscalation = (attribution) => (
  !!attribution && attribution.status === 'needs_compiler'
  && String(attribution.compiler_question || '').trim().length > 0
  && Array.isArray(attribution.suspected_passes) && attribution.suspected_passes.length > 0
);

// The other half of the round's evidence contract: what the round REACHED, given
// what it asked for and what came back. Named, for the same reason as the ladder
// above -- the first version of this lived inline and its guard matched a string
// that also appeared in the comment beside it, so the mutation that broke the rule
// sailed through a check that looked specific.
//
//   L2_profile                 nothing deeper was asked for (the normal round)
//   L3_ir | L4_compiler_source an analysis at that rung returned a diagnosis
//   requested_but_unreached    it was asked for and produced nothing
//
// An `inconclusive` attribution COUNTS as reaching the rung: a plateau the evidence
// genuinely cannot explain is a real finding, and an analyst forced to produce a
// mechanism instead of admitting that is an analyst inventing one. `unavailable`
// does NOT count -- there was nothing to read, which is the failed-escalation case.
//
// The third value exists because neither of the others is true of a failed
// escalation. Recording L3 says "we reconstructed the trajectory and it did not
// help"; recording L2 says "we never looked". A later planner acting on either
// would be acting on a false negative one level above the one this layer catches.
const reachedStage = (requested, irAttribution, compilerAttribution, enabled) => {
  if (!enabled || !requested) return STAGE_L2;
  if (compilerAttribution && compilerAttribution.status !== 'unavailable') return STAGE_L4;
  if (irAttribution && irAttribution.status !== 'unavailable') return STAGE_L3;
  return STAGE_FAILED;
};

const TWIN_LANGUAGES = ['hip', 'cuda'];
const TWIN_APPLICABLE = TWIN_LANGUAGES.includes(String(TARGET_LANGUAGE).toLowerCase());
const twinReject = (rep, label) => {
  if (!TWIN_APPLICABLE) return null;
  const where = label || 'verify';
  // Same scoping rule as `policyReject`: only reports claiming a pass. A
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

// DEEP-MODE resume: restore cumulative speedup + insight/ledger history from the prior wave so this
// continuation builds ON the cumulative best (canonical was already seeded from STATE_DIR/best by the
// director) and does not re-explore dead directions. No-op on a fresh run (prior_state undefined).
if (setup.resumed && setup.prior_state) {
  const ps = setup.prior_state;
  // Finding (127) -- the single line-pair that produced three separate symptoms across two waves.
  //
  // It used to read `if (ps.cumulative > cumulative) cumulative = ps.cumulative`. `ps.cumulative`
  // is vs the ORIGINAL SEED (4.35x after wave 2); `cumulative` is vs THIS wave's BASELINE_PER_CASE,
  // which is CANONICAL == that same cumulative best. Different denominators, same variable. Every
  // downstream comparison then mixed frames:
  //
  //   (a) `improved = winner.geomean > cumulative * (1 + MIN_IMPROVE)` compared a verifier's
  //       wave-local ~1.01 against 4.35 and was FALSE for every round of wave 2 -- including the
  //       three rounds that each really did improve (1.3524 -> 1.37928 -> 1.39884).
  //   (c) the canonical-commit block is `if (improved)`, so the winner was never committed and
  //       CANONICAL stayed on the previous round's tree. Rounds 6 and 7 both had to be repaired
  //       by hand afterwards; round 6's parent was lost once because of it.
  //   (b) `bestPerCase = ps.best_per_case` carried the PRIOR wave's per-case candidate times
  //       forward as if they described the current tree. They were threaded into planning and
  //       into verifiers as the incumbent's timings while the incumbent had moved on by ~37%,
  //       and were read at face value three separate times (it implied an incumbent geomean of
  //       1.019 against a measured ~1.39, which would have priced one patch at +34%).
  //
  // So: the vs-seed number is carried in its own variable, for reporting only, and the per-case
  // table is NOT imported. BASELINE_PER_CASE was just measured on CANONICAL this run, which is
  // exactly "the incumbent's per-case times" -- fresher than anything STATE_DIR can offer.
  if (Number.isFinite(ps.cumulative) && ps.cumulative > 0) priorCumulativeVsSeed = ps.cumulative;
  if (Array.isArray(ps.insights)) history.insights = ps.insights;
  if (Array.isArray(ps.ledger)) history.ledger = ps.ledger;
  if (ps.bottleneck_now) history.bottleneck_now = ps.bottleneck_now;
  if (Array.isArray(ps.best_per_case) && ps.best_per_case.length) {
    log(`  [resume] NOT importing prior_state.best_per_case (${ps.best_per_case.length} rows): those are ` +
        'the previous wave\'s numbers in the previous wave\'s frame. This wave\'s BASELINE_PER_CASE was ' +
        'measured on the same tree they claim to describe, and supersedes them.');
  }
  log(`RESUMED from STATE_DIR: prior cumulative vs seed=${priorCumulativeVsSeed.toFixed(3)}x, ` +
      `this wave restarts at 1.000x vs the incumbent it was seeded from (same tree, different frame), ` +
      `${history.insights.length} insights, ${history.ledger.length} ledger entries carried forward.`);
}

// --- which frame is `cumulative` actually in? -------------------------------
// The comment on `cumulative` above says "vs BASELINE_PER_CASE, so it starts at 1.0 every wave".
// That is true only when the harness times a candidate against THE WAVE'S OWN TREE. When the harness
// times against a FROZEN EXTERNAL ORACLE -- as dense_bf16_gemm_fused does, against direct
// rocblas_gemm_ex -- the verifier's `verified_geomean` is an ABSOLUTE score, `cumulative` inherits it
// on commit, and `priorCumulativeVsSeed * cumulative` multiplies two absolutes.
//
// Measured, not assumed: 1.2054 x 1.2707 = 1.5317 was emitted as CUMULATIVE_VS_SEED on wave 4, and
// 1.31581464 on the wave-2/3 boundary. The first was caught by the TechLead, the second by a human
// two waves later. A number whose correctness depends on who happens to be reading is a defect.
//
// The test is free and direct, because the wave's own baseline was just measured on the incumbent
// tree -- the SAME tree the prior wave's `cumulative` describes. So:
//   * self-anchored harness -> the baseline scores ~1.0 against its own denominator
//   * oracle-anchored harness -> it scores what the incumbent scores vs the oracle, i.e. ~prior
// Whichever it is closer to, in log space, is the frame. No task-specific knowledge, no flag.
const resolveVsSeedFrame = (perCase, prior) => {
  const sp = (Array.isArray(perCase) ? perCase : [])
    .map(r => r && Number(r.speedup)).filter(v => Number.isFinite(v) && v > 0);
  if (!sp.length) {
    return { frame: 'chained', baselineSuite: null,
      why: 'the baseline rows carry no speedup, so the frame cannot be read off the data; keeping '
        + 'the chained form, which is correct on a self-anchored harness and DOUBLE-COUNTS on an '
        + 'oracle-anchored one -- treat any vs-seed number from this run as unverified' };
  }
  const baselineSuite = Math.exp(sp.reduce((a, v) => a + Math.log(v), 0) / sp.length);
  if (!(prior > 0) || Math.abs(Math.log(prior)) < 1e-9) {
    return { frame: 'chained', baselineSuite,
      why: 'no prior wave to chain onto (prior is 1.0), so both readings give the same number' };
  }
  const toSelf = Math.abs(Math.log(baselineSuite));
  const toPrior = Math.abs(Math.log(baselineSuite / prior));
  if (toPrior < toSelf) {
    return { frame: 'absolute', baselineSuite,
      why: `this wave's baseline scores ${baselineSuite.toFixed(4)} against the harness denominator, `
        + `which is the prior wave's ${prior.toFixed(4)} and not 1.0 -- the harness times against a `
        + 'fixed external reference, so verified_geomean is already a vs-seed total and multiplying '
        + 'it by the prior total would count the same speedup twice' };
  }
  return { frame: 'chained', baselineSuite,
    why: `this wave's baseline scores ${baselineSuite.toFixed(4)} against its own denominator, i.e. `
      + 'the harness re-anchors on the incumbent each wave, so each wave contributes a factor' };
};
{
  const r = resolveVsSeedFrame(BASELINE_PER_CASE, priorCumulativeVsSeed);
  vsSeedFrame = r.frame;
  log(`vs-seed frame: ${vsSeedFrame.toUpperCase()} -- ${r.why}`);
}

// ===========================================================================
// PHASE: WarmStart. The recorded speedup only RANKS; adoption is decided by a fresh
// measurement through the verify gate here. gfx comes from the baseline profile's
// on-box `device` string (no extra probe).
// ===========================================================================
const GFX = (String((profileSummary && profileSummary.device) || '').match(/gfx\d+/i) || [''])[0].toLowerCase();
let warm_start = { adopted: false, read_reason: WARM_START_ON ? 'read' : 'disabled', candidates: [] };
let skipLoop = false;
if (WARM_START_ON && !setup.resumed && KB_ROOT_OK) {
  phase('WarmStart');
  if (!GFX) {
    warm_start.read_reason = 'missing_arch';
    log('[kb] warm-start skipped: no gfx detected from the baseline profile device string.');
  } else {
    // The two planes take a different root flag and a different name→page rule (the store addresses
    // by canonical id, so there is nothing to match fuzzily), and print the same JSON.
    const localResolveCmd = KB_MODE === 'store'
      ? `resolve-remote --plane local --store ${JSON.stringify(KB_STORE_DIR)}${KB_VERSION_FLAG}`
      : `resolve --root ${JSON.stringify(KB_ARTIFACTS_DIR)} --match ${WARM_START_MATCH}`;
    const commonArgs =
      `--kernel-name ${JSON.stringify(KERNEL_NAME)} --language ${JSON.stringify(TARGET_LANGUAGE)} \\\n` +
      `  --gfx ${GFX} --top-n 3 --min-speedup ${WARM_START_MIN_SPEEDUP} \\\n` +
      `  --refs-dir ${JSON.stringify(EVAL_DIR + '/kb_references')}`;
    // Remote first, local curated tree as the fallback. The service is the shared plane and should
    // win when it has an answer, but it is still filling up, while `kb_artifacts/` holds a hand-
    // curated history (retired entries, one entry per direction) that a thin remote page must not
    // shadow. An empty remote answer is indistinguishable from a 404 on this scheme, so "no
    // candidates" — not "no error" — is what triggers the second read. Both reads are seconds and
    // no GPU; the thing they protect against is a cold start that costs hours.
    const resolveScript = KB_REMOTE === 'off'
      ? `python3 ${JSON.stringify(EXPERIENCE_STORE)} ${localResolveCmd} \\\n  ${commonArgs}`
      : `${KB_ENV_PRELUDE}
REMOTE_OUT=''
if [ -n "$KB_STORE_TOKEN" ]; then
  REMOTE_OUT=$(python3 ${JSON.stringify(EXPERIENCE_STORE)} resolve-remote --plane remote${KB_VERSION_FLAG} \\
  ${commonArgs} 2>/dev/null || true)
  if printf '%s' "$REMOTE_OUT" | python3 -c 'import json,sys; sys.exit(0 if (json.load(sys.stdin).get("candidates") or []) else 1)' 2>/dev/null; then
    printf '%s\\n' "$REMOTE_OUT"; exit 0
  fi
fi
python3 ${JSON.stringify(EXPERIENCE_STORE)} ${localResolveCmd} \\
  ${commonArgs}`;
    const resolved = await agentT(
      `You are the warm-start resolver. Run EXACTLY this ${KB_REMOTE === 'off' ? 'command' : 'script'} ` +
      `and return its single-line JSON stdout verbatim as StructuredOutput — do not add, drop, reorder, ` +
      `or reinterpret any field. ` +
      (KB_REMOTE === 'off' ? '' :
        `It tries the shared KB Store service first and falls back to the on-disk knowledge base by ` +
        `itself; run it as one script, do not split it into separate commands. `) +
      `An empty candidate list is a VALID answer meaning "nothing recorded for this kernel yet". Do not ` +
      `retry with different flags and do not invent candidates:
\`\`\`bash
${resolveScript}
\`\`\``,
      { phase: 'WarmStart', label: 'warm_start:resolve', schema: WARMSTART_RESOLVE_SCHEMA }) || {};
    warm_start.read_reason = resolved.read_reason || 'read';
    warm_start.slug = resolved.slug || '';
    warm_start.match_tier = resolved.match_tier || '';
    warm_start.filtered = resolved.filtered || null;
    const cands = Array.isArray(resolved.candidates) ? resolved.candidates : [];
    warm_start.candidates = cands.map(c => ({
      rank: c.rank, slug: c.slug, speedup: c.speedup, direction: c.direction || '', status: 'read',
    }));
    const f = resolved.filtered || {};
    // Which plane actually answered. Only the key-addressed subcommand emits `canonical_id`, so its
    // presence separates a store read from a slug-tree read; in `local` KB_MODE the only key-
    // addressed reader in the script is the remote one, so that is also the remote/fallback tell.
    // Worth logging either way: the canonical id is the address, and on a scheme with no search a
    // thin answer and a mis-keyed question look identical from the outside.
    warm_start.plane = resolved.canonical_id ? (KB_MODE === 'store' ? 'store' : 'remote') : 'local';
    if (resolved.canonical_id) log(`[kb] plane=${warm_start.plane} key=${resolved.canonical_id}`);
    else if (KB_REMOTE !== 'off') log('[kb] plane=local (service had no candidates, or no credentials)');
    log(`[kb] experience read: slug=${resolved.slug || '?'} (${resolved.match_tier || 'exact'} match of ` +
      `${resolved.requested_slug || KERNEL_NAME}) reason=${warm_start.read_reason} ` +
      `candidates=${cands.length}${f.total ? ` of ${f.total} recorded [${f.retired || 0} retired, ` +
      `${f.below_min_speedup || 0} below ${WARM_START_MIN_SPEEDUP}x, ` +
      `${f.same_direction_collapsed || 0} same-direction]` : ''}`);
    const otherLangs = Array.isArray(resolved.other_language_pages) ? resolved.other_language_pages : [];
    if (!cands.length && otherLangs.length) {   // wrong target_language, not an empty store
      log(`[kb] NOTE: no ${TARGET_LANGUAGE} page, but the store holds ${otherLangs.join(', ')} — ` +
        `check the target_language this lane was invoked with.`);
    }
    // reference-only: prose is already mirrored to EVAL_DIR/kb_references by the resolver; do not apply.
    if (!WARM_START_REF_ONLY) {
      const editableCsv = ((analysis && analysis.modifiable_files) || []).join(',');
      for (const c of cands) {                              // already rank-ordered (fastest first)
        const rec = warm_start.candidates.find(x => x.rank === c.rank);
        const remapOut = `${EVAL_DIR}/warm_start/cand_${c.rank}/remapped.diff`;
        const ver = await agentT(
          roleAgent('verify_engineer', 'verify',
            'Validate a HISTORICAL warm-start patch — the SAME gate as any round candidate. The patch was ' +
            'won in the workspace that produced it, whose paths need not exist here, so FIRST run EXACTLY:\n' +
            '```bash\n' +
            `mkdir -p ${JSON.stringify(`${EVAL_DIR}/warm_start/cand_${c.rank}`)}\n` +
            `python3 ${JSON.stringify(EXPERIENCE_STORE)} remap --patch ${JSON.stringify(c.patch_path)} \\\n` +
            `  --out ${JSON.stringify(remapOut)} --editable ${JSON.stringify(editableCsv)} \\\n` +
            `  --workspace ${JSON.stringify(CANONICAL)}\n` +
            '```\n' +
            'It prints one line of JSON. remapped=true: apply the REMAPPED patch (its paths were rewritten ' +
            'onto this workspace). reason "no_change_needed": apply the original PATCH. Any other reason ' +
            '(notably "unmapped_paths" — the patch touches files this workspace does not have): return ' +
            'status:"apply_failed", notes "patch_outside_editable_set", and DO NOT apply or benchmark. ' +
            'Apply with `git apply "$P" || git apply --3way "$P"`. On ANY failure restore the working copy ' +
            'fully (git checkout -- . and delete untracked files) before returning so the next candidate ' +
            'starts from a clean tree.', {
            CANONICAL, PATCH: c.patch_path, REMAP_OUT: remapOut,
            // Same gate as a round candidate means the same INPUTS as a round candidate.
            // verify_engineer.md runs the oracle digest, which needs the task dir; upstream's
            // warm-start call omitted it, so the historical patch would have been verified against
            // whatever the agent guessed the oracle was. Caught by test_oracle_digest.
            TASK_DIR: KERNEL_PATH_ORIG,
            VERIFY_DIR: `${EVAL_DIR}/warm_start/cand_${c.rank}`,
            EDITABLE_SET: (analysis && analysis.modifiable_files) || [],
            GPU_ID: GPU_LIST[0], SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
            ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
            ...(REQUIRE_GRAPH_CAPTURE ? { REQUIRE_GRAPH_CAPTURE: '1' } : {}),
          }),
          { phase: 'WarmStart', label: `warm_start:verify c${c.rank}`, schema: VERIFY_SCHEMA });
        const sp = primSpeedup(ver);
        if (ver && says(ver.status, 'verified') && says(ver.correctness, 'pass') && sp > 1.0) {
          const adopt = await agentT(
            `You are the TechLead adopting a validated warm-start patch into the canonical workspace.
\`\`\`bash
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
cd ${CANONICAL}
git checkout -- .
# the remapped patch when verify produced one (its paths fit THIS workspace), else the stored one
P=${JSON.stringify(remapOut)}; [ -f "$P" ] || P=${JSON.stringify(c.patch_path)}
git apply "$P" || git apply --3way "$P"
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "warm-start adopt: ${c.slug} (${sp.toFixed(2)}x)"
git --no-pager diff "$(git rev-list --max-parents=0 HEAD)..HEAD" > ${EVAL_DIR}/current_best.diff
\`\`\`
If BOTH applies fail, apply manually to match intent, then add -A + commit and RE-RUN the COMMANDMENT
correctness check; only report committed=true if it still passes. Return JSON {committed, current_best_diff, note}.`,
            { phase: 'WarmStart', label: `warm_start:adopt c${c.rank}`, schema: COMMIT_SCHEMA });
          if (adopt && adopt.committed) {
            // The optimize loop now builds ON this patch: cumulative starts at the adopted Nx, so this
            // run's own rounds only earn the delta above it (split out as incremental_speedup below).
            cumulative = sp;
            bestPerCase = (ver.per_case && ver.per_case.length) ? ver.per_case : bestPerCase;
            finalWinner = { source: `warm_start:${c.slug}`, geomean: sp,
              arithmetic: ver.verified_arithmetic || sp, per_case: bestPerCase, patch: c.patch_path };
            if (bestSeen < sp) bestSeen = sp;
            warm_start.adopted = true;
            warm_start.adopted_speedup = sp;
            warm_start.slug = c.slug;
            // Carried into the write-back: the idea this run started from, and the entry it
            // descends from — lineage a later curation pass cannot recover from the diff alone.
            warm_start.direction = c.direction || '';
            warm_start.exp_dir = c.exp_dir || '';
            if (rec) rec.status = 'adopted';
            log(`[kb] warm-start ADOPTED ${c.slug} @ ${sp.toFixed(2)}x — optimizing from the patched state.`);
            profileSummary = await agentT(
              roleAgent('profile_engineer', 'reprofile',
                'Re-profile the adopted warm-start state and classify the new bottleneck.', {
                WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_LIST[0], ROUND: 0,
                COMMANDMENT, PREVIOUS_METRICS: profileSummary,
              }),
              { phase: 'WarmStart', label: 'warm_start:reprofile', schema: PROFILE_SCHEMA }) || profileSummary;
            if (history) history.bottleneck_now = profileSummary ? profileSummary.bottleneck : history.bottleneck_now;
            if (WARM_START_RETURN_AFTER) { skipLoop = true; warm_start.returned_after_read_kb = true; }
            break;
          }
        }
        if (rec && rec.status !== 'adopted') rec.status = (ver && ver.status) ? `rejected:${ver.status}` : 'rejected:apply_failed';
        log(`[kb] warm-start candidate c${c.rank} ${rec ? rec.status : 'rejected'} (${sp ? sp.toFixed(2) + 'x' : 'no measure'}).`);
      }
    }
  }
}

// `noImprove < MAX_NO_IMPROVE` used to be the third clause. It is `!stopReason` now because that
// counter no longer ends a run: a stalled phase ESCALATES (see `advancePhase`), and only the last
// phase of a cycle turns a stall into a stop. Leaving the old clause in would end the run at the
// first stalled phase-1, which is exactly the level the ladder exists to move past.
while (!skipLoop && dispatched < BUDGET && !stopReason) {
  // --- (0) HARD STOP: no new round may START past the deadline ----------
  // Checked BEFORE round++ so an expired check does not inflate the round count. The in-flight round
  // is never interrupted — a killed round leaves a half-verified patch and no report.
  let left = await secondsLeft(`pre-r${round + 1}`);
  if (left <= 0) {
    deadlineHit = true;
    log(`WALL-CLOCK DEADLINE reached after ${round} round(s) (budget used ${dispatched}/${BUDGET}) — ` +
        `starting no further rounds; going straight to report + validation.`);
    break;
  }
  if (DEADLINE_EPOCH) log(`[deadline] ${(left / 60).toFixed(0)} min remain before the hard stop.`);

  round++;
  // Pinned at the TOP of the round. The ladder can escalate at the bottom of this same iteration,
  // and a record written from the live variable afterwards would file the round that exhausted a
  // phase under the phase that replaced it -- which is the one reading that would make the new
  // phase look responsible for the stall that promoted it.
  const roundPhase = searchPhase;
  // (127). Distinct from `improved` on purpose: `improved` is "the search found something better",
  // `committedThisRound` is "the canonical tree actually moved". They came apart twice in wave 2 and
  // the history said only the first.
  let committedThisRound = false;
  const remaining = BUDGET - dispatched;
  phase('Optimize');

  // --- (a0) How deep should this round's evidence go? --------------------
  // The escalation ladder, decided from the history that already exists rather
  // than from an agent's sense of being stuck. Depths: `pattern` (the default --
  // code structure plus the baseline profile), `isa` (capture the machine code and
  // attribute the plateau to it), `compiler` (ask why the backend refused).
  //
  // Why it is computed here and in JS. `local_optimum.py`-style stagnation detection
  // needs the per-round geomean series and the metric basis, both of which are in
  // `history` already; shelling out to a Python helper would put a second copy of
  // the rule in the run, and this file's own ledger is explicit that two
  // implementations of one verdict is one more than can be trusted. It is pinned by
  // an executable guard instead.
  //
  // Nothing here counts rounds any more. L3 runs when profile_engineer asked for it in this round's
  // profile, when the TechLead asked for it while planning the last one, or when the last round's
  // machine code refuted the mechanism it claimed. Absent all three, a phase-1 round is L1 -> L2.
  const ladderState = evidenceLadder(history.rounds, profileSummary, LADDER_ENABLED, roundPhase);
  // Read once per round from the same profile the ladder consults, so the ISA
  // receipt and the escalation decision cannot disagree about which kernel is hot.
  const hotKernelNames = (profileSummary && Array.isArray(profileSummary.top_kernels)
    ? profileSummary.top_kernels : [])
    .map(k => k && typeof k.name === 'string' ? k.name.trim() : '')
    .filter(Boolean);
  if (ladderState.requested) {
    log(`  [ladder] round ${round}: ${ladderState.from} -> ${ladderState.requested}, asked for by `
      + `${(ladderState.requested_by || []).join(', ') || 'nobody named'}. ${ladderState.reason}`);
  } else if (ladderState.skip_reason) {
    log(`  [ladder] round ${round}: escalation NOT requested -- ${ladderState.skip_reason}`);
  }

  // Run the deep analysis BEFORE planning, so the plan is shaped by it. Running it
  // after would produce a document nobody acted on this round, which is how a
  // "hierarchy" becomes paperwork.
  let irAttribution = null;
  let compilerAttribution = null;
  let isaAnalysisSkipped = ladderState.skip_reason || '';
  if (ladderState.requested === STAGE_L3) {
    if (!isaCanonicalArchive) {
      // Named, not silent. The ISA archive is what ties a recompiled trajectory to
      // the binary that was actually measured; without it `ir_capture.py` can still
      // produce stages, but nothing can say they belong to the incumbent, and an
      // attribution to a neighbouring binary reads exactly like a real one.
      isaAnalysisSkipped = 'L3 was requested but the canonical tree has no ISA archive yet '
        + '(normal before the first committed round, and after any canonical that carried none), '
        + 'so a captured trajectory could not be tied to the binary that was measured';
      log(`  [ladder] L3 SKIPPED: ${isaAnalysisSkipped}`);
    } else {
      irAttribution = await agentT(
        roleAgent('ir_engineer', 'ir_attribution',
          'Name the lowering stage and pass where the source intent stopped surviving, and the '
          + 'structural condition the next rewrite must satisfy.', {
            EVAL_DIR, ROUND: round, SKILL_DIR: WORKFLOW_DIR, WORKSPACE: CANONICAL,
            KERNEL_KNOWLEDGE_DIR,
            IR_CAPTURE_HELPER: `${WORKFLOW_DIR}/scripts/ir_capture.py`,
            IR_SIGNALS_HELPER: `${WORKFLOW_DIR}/scripts/ir_signals.py`,
            // Supplied for provenance and for an optional closing corroboration --
            // NOT as the evidence. See ir_engineer.md "The ISA rule", and
            // `admissibleAttribution` below, which refuses a positive status that
            // names no pass precisely so this input cannot quietly become the source.
            ISA_ARCHIVE: isaCanonicalArchive,
            ISA_SIGNALS_HELPER: `${WORKFLOW_DIR}/scripts/isa_signals.py`,
            ESCALATION_FROM: ladderState.from,
            ESCALATION_REASON: ladderState.reason,
            PROFILE_SUMMARY: profileSummary,
            HISTORY: history,
            OUTPUT_PATH: `${EVAL_DIR}/round_${round}_ir_attribution.md`,
          }),
        { phase: 'Optimize', label: `ir_engineer:L3 r${round}`, schema: IR_ATTRIBUTION_SCHEMA });

      const inadmissible = admissibleAttribution(irAttribution);
      if (inadmissible) {
        isaAnalysisSkipped = `L3 returned an inadmissible attribution -- ${inadmissible}`;
        log(`  [ladder] L3 REFUSED: ${inadmissible}`);
        irAttribution = null;
      } else if (!irAttribution || !irAttribution.diagnosis) {
        isaAnalysisSkipped = 'L3 returned no usable diagnosis';
        irAttribution = null;
        log('  [ladder] L3 produced nothing usable; planning proceeds on profiling evidence');
      } else {
        log(`  [ladder] L3 attribution: status=${irAttribution.status}`
          + (irAttribution.attributed_pass ? ` pass=${irAttribution.attributed_pass}` : '')
          + (irAttribution.stage_transition ? ` (${irAttribution.stage_transition})` : '')
          + ` -- ${String(irAttribution.status === 'attributed'
            ? `next edit must satisfy: ${irAttribution.source_change_required || '(unstated)'}`
            : irAttribution.diagnosis).slice(0, 160)}`);
      }

      // L4, in THIS round, and only on a question L3 actually formed. This is the
      // whole reachability fix: waiting for the next round would put the decision
      // behind a stop budget that ends the lane first.
      if (wantsCompilerEscalation(irAttribution)) {
        log(`  [ladder] L3 -> L4: ${String(irAttribution.compiler_question).slice(0, 180)}`);
        compilerAttribution = await agentT(
          roleAgent('compiler_engineer', 'compiler_attribution',
            'Recover the constraint behind the pass L3 named, and the source condition that would '
            + 'satisfy it.', {
              EVAL_DIR, ROUND: round, SKILL_DIR: WORKFLOW_DIR, WORKSPACE: CANONICAL,
              KERNEL_KNOWLEDGE_DIR,
              IR_ARCHIVE: irAttribution.archive || '',
              IR_ATTRIBUTION: irAttribution,
              IR_SIGNALS_HELPER: `${WORKFLOW_DIR}/scripts/ir_signals.py`,
              IR_CAPTURE_HELPER: `${WORKFLOW_DIR}/scripts/ir_capture.py`,
              // Tier 1's provenance check needs both of these.
              ISA_ARCHIVE: isaCanonicalArchive,
              ISA_CAPTURE_HELPER: `${WORKFLOW_DIR}/scripts/isa_capture.py`,
              ISA_SIGNALS_HELPER: `${WORKFLOW_DIR}/scripts/isa_signals.py`,
              // Tier 2, and unset on this image: /opt/rocm/llvm ships the built
              // toolchain, not llvm/lib/Target/AMDGPU. Spread only when an operator
              // actually supplies a checkout, so the role sees its absence and
              // finishes on Tier 0/1 -- rather than being handed an empty string it
              // might read as "look somewhere".
              ...(COMPILER_SOURCE_DIR ? { COMPILER_SOURCE_DIR } : {}),
              ESCALATION_FROM: STAGE_L3,
              ESCALATION_REASON: irAttribution.compiler_question,
              PROFILE_SUMMARY: profileSummary,
              HISTORY: history,
              OUTPUT_PATH: `${EVAL_DIR}/round_${round}_compiler_attribution.md`,
            }),
          { phase: 'Optimize', label: `compiler_engineer:L4 r${round}`,
            schema: IR_ATTRIBUTION_SCHEMA });
        if (!compilerAttribution || !compilerAttribution.diagnosis) {
          compilerAttribution = null;
          log('  [ladder] L4 produced nothing usable; the round keeps L3\'s attribution');
        } else {
          log(`  [ladder] L4 attribution: status=${compilerAttribution.status} `
            + `tier=${compilerAttribution.tier_used != null ? compilerAttribution.tier_used : '?'} `
            + `-- ${String(compilerAttribution.source_change_required
              || compilerAttribution.diagnosis).slice(0, 160)}`);
        }
      } else if (irAttribution && irAttribution.status === 'needs_compiler') {
        // Named rather than silently downgraded: `needs_compiler` without a pass
        // list AND a question is a request for an unbounded investigation, and the
        // reason it is refused belongs in the record.
        log('  [ladder] L3 asked for L4 without both suspected_passes and a compiler_question; '
          + 'refused, because an unnarrowed compiler investigation has no stopping condition');
      }
    }
  }
  // The attribution the planner acts on: L4's when it produced one, else L3's. They
  // are not merged -- a compiler-level constraint supersedes the IR-level statement
  // of the same problem, and carrying both would let a planner satisfy the weaker one.
  const isaAttribution = compilerAttribution || irAttribution;

  // What this round actually REACHED, as distinct from what it asked for. See
  // `reachedStage`: the `requested_but_unreached` value exists because collapsing it
  // into plain L2 loses the one thing the next planner needs. Writing L3 on a round
  // whose analysis never ran says "we reconstructed the trajectory and it did not
  // help"; writing L2 says "we never looked". Neither is true of a failed
  // escalation, and the first is a false negative one level up from the one this
  // whole layer exists to catch.
  const stageReached = reachedStage(ladderState.requested, irAttribution, compilerAttribution,
    LADDER_ENABLED);
  if (ISA_ENABLED && ladderState.requested && stageReached !== ladderState.requested
      && stageReached !== STAGE_L4) {
    log(`  [ladder] round ${round} asked for "${ladderState.requested}" and reached `
      + `"${stageReached}". The record states what it reached, never what it wanted, `
      + 'so no later planner reads an unbacked escalation as evidence already spent.');
  }

  // --- (a0b) phase2 needs a LOCATION, and the profiler is what produces one ---
  // The reprofile that would normally refresh this only runs after an IMPROVING round, so on the
  // first phase-2 round -- and on every phase-2 round that banks nothing, which is most of them --
  // the lane would otherwise hand the engineer a whole kernel of assembly and no reason to prefer
  // any part of it. Localisation is therefore its own step, run at the top of the round against the
  // canonical that is actually about to be edited.
  if (roundPhase === PHASE_ASM && ISA_ENABLED) {
    const located = await agentT(
      roleAgent('profile_engineer', 'keypoint',
        'Localise the instruction windows where an assembly edit is most likely to pay.', {
          WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL, ROUND: round,
          COMMANDMENT,
          ISA_SIGNALS_HELPER: `${WORKFLOW_DIR}/scripts/isa_signals.py`,
          ASM_ROUNDTRIP_HELPER: `${WORKFLOW_DIR}/scripts/asm_roundtrip.py`,
          // The archive of the tree about to be edited. Empty is not fatal: the role re-emits the
          // assembly with asm_roundtrip and localises on that instead, which is the same text.
          ISA_ARCHIVE: isaCanonicalArchive || '',
          BOTTLENECK: (profileSummary && profileSummary.bottleneck) || '',
          TOP_KERNELS: (profileSummary && profileSummary.top_kernels) || [],
        }),
      { phase: 'Optimize', label: `profile_engineer:keypoint r${round}`, schema: PROFILE_SCHEMA });
    const found = located && Array.isArray(located.keypoints) ? located.keypoints : [];
    profileSummary = { ...(profileSummary || {}), keypoints: found };
    if (found.length) {
      log(`  [phase2] ${found.length} keypoint(s) localised: `
        + found.map(k => `${k.kernel || '?'}@${k.block || '?'}+${k.ordinal_in_block}`
          + ` (score ${k.score})`).join(', '));
    } else {
      // Not a refusal. An engineer with no keypoint still has the assembly and the bottleneck; what
      // it loses is the reason to edit one window rather than another, and saying so is better than
      // letting the round look localised when it was not.
      log('  [phase2] no keypoint localised this round -- the engineer works from the bottleneck '
        + 'alone, so this round is NOT a local search and should not be read as one.');
    }
  }

  // --- (a) Plan the round ------------------------------------------------
  // A FUNCTION of `forbidStop`, not an object, so the forced re-plan below can rebuild the same
  // inputs with one key added. Building it twice by hand is how a replan ends up planning against a
  // different input set than the round it replaces.
  const planInputs = (forbidStop) => ({
    // FIRST, deliberately. Stated UNCONDITIONALLY, in both directions: `use_learned_kb=false` used
    // to omit the budget and nothing else, while roles/tech_lead.md still listed
    // knowledge/learned/INDEX.md among the files to read -- so the off switch removed the budget
    // that CONSTRAINS the KB and left the instruction that USES it, and the KB-off control arm was
    // never actually KB-off. The role gates on this value, so it has to be present whichever way it
    // points. Kept at the top because test_kb_switch.py follows one level of helper indirection and
    // reads a bounded window of the body; a switch buried below that window is one the guard cannot
    // see, which is the same class of lapse the guard exists to catch.
    LEARNED_KB: USE_LEARNED_READ ? 'on' : 'off',
    ...(USE_LEARNED_READ ? { LEARNED_KB_BUDGET:
      `At most ${KB_DIR_CAP} of this round's directions may draw on learned cards, and you must cite ` +
      `the cards you used in that direction's \`learned_refs\`. ` +
      (KB_COLD_DIRECTION ? `At least ONE direction must be planned from the profile alone, with an ` +
        `empty \`learned_refs\` — it is this round's control, and without it a round cannot tell you ` +
        `whether the cards helped or merely crowded out what you would have tried anyway. ` : '') +
      `Read as much of the index as you like; the budget is on how much of the ROUND the KB steers, ` +
      `not on what you may look at.` } : {}),
    // The escalation ladder's output, spread only when it produced something. An
    // `attributed` block names a constraint the next edit must satisfy AND the
    // directions it ruled out -- the paper's point being that deep evidence earns
    // its cost mostly by REJECTING plausible directions, not by discovering winners.
    ...(ISA_ENABLED ? {
      // The stage REACHED, not the one requested. Handing the planner the request
      // would tell it evidence exists that does not.
      ISA_EVIDENCE_DEPTH: stageReached,
      ...(ladderState.reason ? { ISA_ESCALATION_REASON: ladderState.reason } : {}),
      ...(isaAttribution ? { ISA_ATTRIBUTION: isaAttribution } : {}),
      ...(isaAnalysisSkipped ? { ISA_ANALYSIS_SKIPPED: isaAnalysisSkipped } : {}),
      // Both rungs, when both ran. The planner acts on `ISA_ATTRIBUTION` (L4's if it
      // exists); this pair is here so a reader of the round can see that L4's
      // constraint came from a pass L3 named, rather than from a fresh guess.
      ...(irAttribution ? { IR_ATTRIBUTION: irAttribution } : {}),
      ...(compilerAttribution ? { COMPILER_ATTRIBUTION: compilerAttribution } : {}),
    } : {}),
    EVAL_DIR, ROUND: round, BUDGET_REMAINING: remaining, CUMULATIVE_SPEEDUP: cumulative,
    // (127). Both frames, each labelled, so a planner cannot silently read one as the other.
    // CUMULATIVE_SPEEDUP is vs THIS wave's baseline (the tree you are editing, 1.000 at wave start);
    // CUMULATIVE_VS_SEED is the lane total vs the original seed and is context, not a target.
    CUMULATIVE_SPEEDUP_FRAME: 'vs this wave\'s BASELINE_PER_CASE == the incumbent tree at wave start',
    CUMULATIVE_VS_SEED: cumulativeVsSeed(),
    BASELINE_GEOMEAN_MS, SKILL_DIR: WORKFLOW_DIR, PROFILE_SUMMARY: profileSummary,
    CURRENT_BEST_PER_CASE: bestPerCase, HISTORY: history,
    KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE, KK_REFS, ...KB_INPUTS,
    // DRA brief (REFERENCE). plan_round Reads it and seeds directions[] from the ranked DRA
    // directions — see tech_lead.md plan_round. Spread conditionally, so when dra_enabled was off
    // (or the research degraded) the key is absent and the prompt is byte-identical to a build
    // without this feature.
    ...(researchBriefPath ? { DEEP_SEARCH_BRIEF: researchBriefPath } : {}),
    // Which phase this round is, and how close it is to handing the search on. Without it the
    // planner would keep proposing source directions in a phase whose engineers cannot edit source,
    // and would read a phase-2 plateau as a failure of its own ideas rather than as the ladder
    // working. `SEARCH_PHASE_STALL` is deliberately the raw pair, not a percentage: "3 of 4" is a
    // fact the planner can act on, a "75% exhausted" is a number it would have to interpret.
    SEARCH_PHASE: roundPhase,
    SEARCH_PHASE_STALL: `${phaseStall} of ${MAX_NO_IMPROVE} non-improving round(s) in this phase`,
    CYCLE: cycle,
    // Fires on the FIRST round of a new cycle only. The insight blackboard needs no expiry -- its
    // contract is "durable, transferable findings", and a rewrite does not falsify "L2 is already
    // 99%". The LEDGER is the tree-specific memory, and it already handles this correctly per row:
    // every closure carries a `reopens_when` condition, so a closure holds only while that condition
    // does. What was missing is the prompt to CHECK them, and a ground-up rewrite is the single most
    // likely event to satisfy several at once -- those conditions name tiles, LDS, residency and
    // CTA/CU, which is exactly what a deep round changes. Without this the run keeps avoiding
    // dead-ends that its own last move reopened.
    ...(cycle > 1 && roundPhase === PHASE_SOURCE && phaseStall === 0 && history.ledger.length ? {
      RECHECK_CLOSED_MECHANISMS:
        'A deep round rewrote the kernel, so this is a NEW implementation. Before planning, re-read '
        + 'the LEDGER\'s closed rows and check each `reopens_when` against THIS tree: a ground-up '
        + 'rewrite changes tiles, LDS footprint, residency and CTA/CU, which is what most of those '
        + 'conditions are written about. Name any you find newly satisfied in `reasoning` — a '
        + 'closure that the rewrite reopened is the cheapest direction available, because it is '
        + 'already measured and already understood.',
    } : {}),
    ...(roundPhase === PHASE_ASM
      ? { KEYPOINTS: (profileSummary && profileSummary.keypoints) || [] } : {}),
    // Both keys absent unless a deadline was passed => byte-identical prompt when OFF.
    ...(DEADLINE_EPOCH ? { MINUTES_REMAINING: Math.round(left / 60) } : {}),
    ...(forbidStop ? { STOP_NOT_PERMITTED:
      `You returned stop=true with ${Math.round(left / 60)} min of the wall-clock window still ` +
      `unspent, and this is the LAST phase of the cycle, so there is no level left to escalate to — ` +
      `a stop here ends the run. This is a FIXED-WINDOW measurement: ending early is an invalid data ` +
      `point no matter how good the number or how well argued the rationale, and your role's own ` +
      `convergence gates (launch floor closed, gain under the promotion bar, roofline %, "only a ` +
      `different tech stack is left") do NOT authorise stopping here. Plan at least one direction. ` +
      `Switch lever rather than re-running an exhausted one — algorithm / memory / compute / ` +
      `host_runtime, or a dedicated deep_explore ground-up rewrite. If every direction really is ` +
      `exhausted, spend the round RE-MEASURING the committed best to tighten its error bars and say ` +
      `so. Sizing: keep the round finishable inside the remaining ${Math.round(left / 60)} min.` } : {}),
  });
  let plan = await agentT(
    roleAgent('tech_lead', 'plan_round', 'Decide this round\'s orthogonal directions (or stop).',
      planInputs(false)),
    { phase: 'Optimize', label: `tech_lead:plan r${round}`, schema: PLAN_SCHEMA });

  // NO EARLY STOP, but only where the ladder has nowhere left to go. Upstream refuses EVERY stop
  // while the window is unspent, because there a stop ends the run outright. Here a stop in
  // phase1/phase2 is the reasoned signal that THIS LEVEL is finished and it escalates instead --
  // refusing it would keep the planner grinding a level it just declared done, which is the opposite
  // of what the window should buy. The deep round is the last state of the cycle, so a stop there is
  // the one that can still hand hours back, and that is the one worth refusing. Bounded by
  // MAX_FORCED_REPLANS so a role that genuinely cannot name a direction degrades to stopping rather
  // than spending the window on planning.
  while (roundPhase === PHASE_DEEP && DEADLINE_EPOCH && left > NO_STOP_S &&
         forcedReplans < MAX_FORCED_REPLANS &&
         (!plan || plan.stop || !plan.directions || plan.directions.length === 0)) {
    forcedReplans++;
    log(`Round ${round}: stop=true in the last phase with ${(left / 60).toFixed(0)} min left — ` +
        `refusing (#${forcedReplans}), re-planning.`);
    plan = await agentT(
      roleAgent('tech_lead', 'plan_round', 'Your stop was refused: plan at least one direction.',
        planInputs(true)),
      { phase: 'Optimize', label: `tech_lead:replan r${round}#${forcedReplans}`, schema: PLAN_SCHEMA });
    left = await secondsLeft(`replan-r${round}`);
  }

  // The planner reporting this level finished ESCALATES; it does not end the run. Its stop gate is
  // a statement about the current level -- floor attacked, no compute case far above it, last gain
  // under ~3% -- and that is the premise for moving to the next level, not for going home. It used
  // to `break`, which made the reasoned signal destructive and left the crude counter as the only
  // way to reach a phase that exists precisely for "source search is finished".
  const plannerStopped = !plan || plan.stop || !plan.directions || plan.directions.length === 0;
  if (plannerStopped && roundPhase !== PHASE_DEEP) {
    log(`Round ${round}: TechLead reports ${roundPhase} finished. `
      + `${plan ? plan.reasoning || '' : '(no plan returned)'}`);
    advancePhase(escalationReason(true, phaseStall, MAX_NO_IMPROVE), round);
    // No directions were dispatched, so nothing is scored: the round is spent moving the ladder.
    // It cannot loop -- the next state differs, and a deep round always produces a direction.
    continue;
  }

  let planned = (plan && plan.directions) || [];
  // A deep round IS the deep engineer. Leaving it discretionary would make the round that closes a
  // cycle indistinguishable from a phase-1 round, and the ladder would have spent two escalations to
  // arrive somewhere that runs specialist directions again. The planner's own direction becomes the
  // BRIEF, so a promoted round redirects the thinking rather than discarding it -- and a planner
  // that returned `stop` here is overridden outright, because this round's whole purpose is to
  // produce one rewrite before the cycle restarts.
  if (roundPhase === PHASE_DEEP && !planned.some(d => d.specialty === 'deep_explore')) {
    const seed = planned[0] || {};
    log(`Round ${round}: deep round -- ${plannerStopped
      ? 'the planner returned no plan, so the round proceeds on its own mandate'
      : `the planner returned no deep_explore direction, so ${planned.length} specialist `
        + 'direction(s) fold into its brief'}.`);
    planned = [{
      ...seed,
      id: seed.id || `r${round}_d0`,
      title: seed.title ? `deep: ${seed.title}` : 'deep: open-ended rewrite',
      specialty: 'deep_explore',
      prompt: (seed.prompt ? `${seed.prompt}\n\n` : '')
        + 'Source-level directions and assembly edits are both exhausted on the CURRENT tree. Your '
        + 'job is a DIFFERENT ROUTE, not a further step down this one: a ground-up rewrite that the '
        + 'next cycle will then search at source and assembly level in turn. Treat any text above '
        + 'as a starting hypothesis rather than as the assignment.',
    }];
  }

  let directions = planned.map((d, i) => ({
    ...d,
    idx: i, id: d.id || `r${round}_d${i}`,
    gpu_id: GPU_MODE === 'pin' ? GPU_LIST[i % GPU_LIST.length] : GPU_POOL,
    out_dir: `${EVAL_DIR}/round_${round}/engineer_${i}`,
    seed_dir: CANONICAL,
    cost: d.specialty === 'deep_explore' ? DEEP_COST : 1,
  }));
  // deep_explore is a DEDICATED-ROUND heavyweight mandate.
  const deepDir = directions.find(d => d.specialty === 'deep_explore');
  if (deepDir) {
    // A direction that vanishes without a named reason is indistinguishable
    // from a planner that returned nothing, and nothing downstream can recover
    // the difference: the lane has no filesystem, `round_N/` only ever contains
    // the engineers that were dispatched, and the summary line below prints the
    // surviving count. So a round that planned four and ran one reads on disk
    // exactly like a round that planned one.
    const shed = directions.filter(d => d !== deepDir);
    if (shed.length) {
      log(`Round ${round}: dedicated ${deepDir.specialty} round -- `
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
    stopReason = 'budget_exhausted';
    log(`Round ${round}: no planned direction fits remaining budget ${remaining}.`);
    break;
  }
  const roundCost = directions.reduce((s, d) => s + (d.specialty === 'deep_explore' ? DEEP_COST : 1), 0);
  dispatched += roundCost;
  // Planned-vs-dispatched on one line. `directions.length` alone is the count
  // that survived the dedicated-round shed and the affordability check, and
  // reading it as "what the planner decided to do" hides both.
  log(`Round ${round}: planned ${plan.directions.length} -> dispatched ${directions.length} `
    + `direction(s) [${directions.map(d => d.specialty).join(', ')}], cost ${roundCost}, `
    + `budget ${dispatched}/${BUDGET}`);

  // --- (b,c) Optimize -> Verify, pipelined per direction ----------------
  const results = await pipeline(
    directions,
    (d) => {
      // The PHASE decides which contract an engineer reads, and it outranks the specialty: in
      // phase2 there is no source direction to implement, so a specialty label would send the
      // engineer to a role that edits the wrong artifact.
      const isAsm = roundPhase === PHASE_ASM;
      const isDeep = !isAsm && d.specialty === 'deep_explore';
      const readLine = isAsm
        ? `Then Read ${WORKFLOW_DIR}/roles/asm_engineer.md and follow it. You are editing the device ` +
          `ASSEMBLY, not the source: emit it with asm_roundtrip.py, change ONLY the window a KEYPOINT ` +
          `names, rebuild with asm_roundtrip.py, and measure. The source files are read-only to you ` +
          `this round — an edit to them would be a phase-1 direction wearing a phase-2 label, and the ` +
          `round would attribute a source result to the assembly phase.`
        : isDeep
        ? `Then Read ${WORKFLOW_DIR}/roles/deep_engineer.md and ALL knowledge files under ${WORKFLOW_DIR}/knowledge/ ` +
          // "ALL" is what makes this exclusion necessary: without it a KB-off run still sent the deep
          // engineer into knowledge/learned/, so the switch covered the planner and not the one role
          // with the broadest licence to act on a card.
          (USE_LEARNED_READ ? '' : `— EXCLUDING knowledge/learned/, which is off for this run: do not open ` +
            `INDEX.md or any card there. `) +
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

## Inputs
${cfg({
        SPECIALTY: d.specialty,
        DIRECTION: { id: d.id, title: d.title, focus_files: d.focus_files || [], expected_speedup: d.expected_speedup, prompt: d.prompt },
        ...(isDeep ? { TARGET: d.expected_speedup ? `reach ${d.expected_speedup}x (or ~90% of the roofline ceiling), whichever is the harder bar` : 'reach ~90% of the roofline ceiling' } : {}),
        KERNEL_PATH: `${d.out_dir}/workspace`,
        OUTPUT_DIR: d.out_dir,
        CANONICAL, GPU_ID: d.gpu_id, ...GPU_LOCK_INPUT, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT,
        ...(ISA_ENABLED ? {
          ISA_SIGNALS_HELPER: `${WORKFLOW_DIR}/scripts/isa_signals.py`,
          ISA_MODE,
        } : {}),
        // Absent outside phase2, so the phase-1 prompt is byte-identical to the build before the
        // ladder existed. KEYPOINTS is what makes this phase local: without it the engineer has a
        // whole kernel of assembly and no reason to prefer one part of it.
        ...(isAsm ? {
          ASM_ROUNDTRIP_HELPER: `${WORKFLOW_DIR}/scripts/asm_roundtrip.py`,
          KEYPOINTS: (profileSummary && profileSummary.keypoints) || [],
          SEARCH_PHASE: roundPhase,
        } : {}),
        codebase_context: `${EVAL_DIR}/codebase_context.md`,
        profiling_summary: profileSummary ? profileSummary.summary_path : '',
        baseline_per_case: BASELINE_PER_CASE,
        INSIGHTS: history.insights,
        // The ledger, which engineers never used to receive. `deep_engineer.md` has always been
        // told "do not re-walk confirmed dead-ends" -- and the dead-end verdicts live HERE, not in
        // the insight blackboard, which holds transferable findings rather than closed directions.
        // So the one role most able to re-walk a dead end was the one asked to avoid dead ends it
        // could not see. Every engineer gets it: a specialist can be handed a direction adjacent to
        // a closed one, and an assembly engineer can pick a keypoint whose mechanism was already
        // measured shut at source level.
        LEDGER: history.ledger,
        KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE,
        KK_REFS: (d.kk_refs && d.kk_refs.length ? d.kk_refs : KK_REFS),
        ...KB_INPUTS,
      })}

Return ONLY the worker_result.json structure as StructuredOutput.` +
      // Built inline, not via roleAgent(), so the injection has to be appended here too.
      expertSkillsBlock(isAsm ? 'asm_engineer' : isDeep ? 'deep_engineer' : 'engineer'),
      { phase: 'Optimize',
        label: `${isAsm ? 'asm' : isDeep ? 'deep' : 'eng'} ${d.id}:${isAsm ? 'assembly' : d.specialty}`,
        schema: ENG_SCHEMA }
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
      if (trustworthyBelowBaseline) {
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
      // when there is one, which is the input to deciding whether this path
      // deserves a cheaper pre-check.
      if (recovered && !eng) {
        log(`Round ${round} dir ${d.id}: engineer return MISSING (died/timed out/mis-returned) — `
          + `best_patch.diff on disk is unexamined; sending to verify (oracle decides).`);
      } else if (recovered) {
        const ratios = ((eng.per_case || []).map(c => c && c.speedup)
          .filter(v => typeof v === 'number' && Number.isFinite(v) && v > 0));
        const worst = ratios.length ? Math.min(...ratios) : null;
        log(`Round ${round} dir ${d.id}: engineer return FAILED but complete — self-reported `
          + `geomean ${primSpeedup(eng)}, worst case `
          + `${worst === null ? 'not reported' : worst.toFixed(4)} over ${ratios.length} case(s). `
          + `Self-reports do not suppress an independent measurement; sending to verify `
          + `(oracle decides).`);
      }
      return agentT(
        roleAgent('verify_engineer', 'verify', 'Independently re-measure this candidate patch.', {
          CANONICAL, PATCH: patch, VERIFY_DIR: `${d.out_dir}/verify`,
          TASK_DIR: KERNEL_PATH_ORIG,
          GPU_ID: d.gpu_id, ...GPU_LOCK_INPUT, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
          ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
          ...(REQUIRE_GRAPH_CAPTURE ? { REQUIRE_GRAPH_CAPTURE: '1' } : {}),
          ...(ISA_ENABLED ? {
            ISA_MODE,
            ISA_CAPTURE_HELPER: `${WORKFLOW_DIR}/scripts/isa_capture.py`,
            ISA_SIGNALS_HELPER: `${WORKFLOW_DIR}/scripts/isa_signals.py`,
            ISA_ARCHIVE_DIR: `${d.out_dir}/verify/isa`,
            // The engineer's own declaration, forwarded unchanged. Verify must not
            // re-derive it from the candidate's source: the claim is what is being
            // tested, and a verifier that rewrites the claim to fit the evidence is
            // not an independent check. Absent when the engineer declared none.
            ISA_MECHANISM_CLAIMS: eng && Array.isArray(eng.mechanism_claims)
              ? eng.mechanism_claims : [],
            // ...and the same claim's ON-DISK copy, for the recovery path only.
            // A lost StructuredOutput used to take the mechanism claim with it: `eng`
            // is null, this list renders as `[]`, and the verifier -- correctly
            // refusing to substitute a claim of its own -- reports `indeterminate`.
            // Observed on this task the first time the lane ever committed: the
            // engineer had declared ['reduce_lds'] and written it to
            // <out_dir>/worker_result.json at 13:32; the recovered verify at 15:02
            // was handed [] and the ISA layer answered nothing about the one patch
            // that got banked. That is the identical failure the patch harvest above
            // already refuses to make -- an engineer that died can still have left a
            // good artefact on disk -- and the identical remedy applies, because
            // worker_result.json is written by the engineer BEFORE verify measures
            // anything, so reading it is not fitting the claim to the evidence. We
            // cannot stat it from the workflow sandbox, so, exactly as with the
            // patch, the file is named and the decision is delegated. Sent ONLY when
            // the return was lost: a live `eng` is the authority and a file must
            // never get to override it.
            ...(recovered ? { ISA_MECHANISM_CLAIMS_FILE: `${d.out_dir}/worker_result.json` } : {}),
            // Null on round 1, and null again after any canonical with no archive.
            // Both mean the same thing to the verifier: diff-based claims are
            // indeterminate this round; report that, do not substitute another tree.
            ...(isaCanonicalArchive ? { ISA_PARENT_ARCHIVE: isaCanonicalArchive } : {}),
            // The kernels L2 said the time is in. A claim holds if ANY shared kernel
            // moved in the claimed direction -- which is the right rule and stays --
            // but it means a mechanism can land on a cold sibling and the receipt
            // read exactly like one that landed on the route being measured. Passed
            // so the diff can SAY which kernel satisfied it. It never causes a
            // refusal; see isa_signals.py `--hot-kernel`.
            ...(hotKernelNames.length ? { ISA_HOT_KERNELS: hotKernelNames } : {}),
          } : {}),
        }),
        { phase: 'Verify', label: `verify ${d.id}${recovered ? ' (recovered)' : ''}`, schema: VERIFY_SCHEMA }
      ).then((ver) => ({ d, eng, ver, patch }));
    }
  );

  const clean = results.filter(Boolean);
  // The ISA receipt is LOGGED for every candidate in both observe and gate mode;
  // only `gate` lets it change an outcome. Logging it under `observe` is the entire
  // point of that mode -- it is how you find out, on your own kernels and before
  // anything is gated, how often a round that reads "no effect" was actually a
  // round whose mechanism never reached the machine code. That number is what
  // should decide whether `gate` is worth turning on.
  if (ISA_ENABLED) {
    for (const r of clean) {
      const id = r.d && r.d.id ? r.d.id : 'candidate';
      // Named `receipt`, not `s`, and deliberately not spelled the way the two
      // receipt-absence gates above spell their own guard: `test_js_suite.py`'s
      // MUTANTS anchor on that exact line of text, and a third copy here would
      // take a slot the probe means for `policyReject` and `twinReject`.
      const receipt = r.ver && r.ver.isa_evidence;
      if (!receipt || typeof receipt !== 'object') {
        log(`  [isa] ${id}: no ISA receipt returned -- mechanism unverified. This is a gap in `
          + 'OUR evidence, not a fault in the candidate, so it does not gate.');
        continue;
      }
      const parts = [`verdict=${receipt.mechanism_verdict}`,
        `claims=[${(receipt.mechanism_claims || []).join(',') || 'none declared'}]`];
      if (receipt.unchanged_machine_code) parts.push('UNCHANGED-MACHINE-CODE');
      if ((receipt.claims_refuted || []).length) {
        parts.push(`refuted=[${receipt.claims_refuted.join(',')}]`);
      }
      if ((receipt.claims_indeterminate || []).length) {
        parts.push(`indeterminate=[${receipt.claims_indeterminate.join(',')}]`);
      }
      if (Number.isFinite(receipt.high_findings) && receipt.high_findings > 0) {
        parts.push(`high_findings=${receipt.high_findings}`);
      }
      if (receipt.exit_code === 2) parts.push('(archive HOLE: nothing was captured)');
      log(`  [isa] ${id}: ${parts.join(' ')}`);
    }
    // Adopt a parent archive the verifier captured itself, when we have none.
    //
    // This is what turns the ISA layer from decorative into live. `isaCanonicalArchive` used to be
    // assigned in exactly ONE place -- inside the committed-winner branch -- so it stayed null until
    // the first commit landed, and on a lane where most rounds commit nothing it stayed null for the
    // whole run. No parent archive means no diff, which means every `mechanism_verdict` is
    // `indeterminate`, which means the layer answers nothing in precisely the situation it was built
    // for: a plateau where you need to know whether the mechanism reached the machine code. Across
    // seven waves of the greedy lane, 11 clean captures produced 0 machine-readable verdicts.
    //
    // The verifier is the right source because it already builds the parent tree in its own session
    // for the control arm, so capturing its ISA is nearly free -- and one verifier on this lane did
    // exactly that unprompted, leaving a `verify/isa_parent_canonical/` directory behind.
    //
    // The safety property is unchanged and is why this keys on the hash: a WRONG parent is strictly
    // worse than none, because it yields confident, wrong claim verdicts instead of an honest
    // `indeterminate`. So an archive is adopted only when the verifier says which tree it describes,
    // and any later disagreement drops the pointer rather than picking a side.
    if (!isaCanonicalArchive) {
      for (const r of clean) {
        const receipt = r.ver && r.ver.isa_evidence;
        if (!receipt || typeof receipt !== 'object') continue;
        if (typeof receipt.parent_archive !== 'string' || !receipt.parent_archive) continue;
        if (typeof receipt.parent_source_hash !== 'string' || !receipt.parent_source_hash) {
          log('  [isa] a verifier returned a parent_archive with no parent_source_hash; NOT adopting '
            + 'it as the canonical parent. An archive whose tree is unnamed cannot be checked against '
            + 'the next round\'s parent, and an unnoticed stale parent produces confident wrong '
            + 'verdicts -- worse than the indeterminate a missing parent produces.');
          continue;
        }
        isaCanonicalArchive = receipt.parent_archive;
        isaCanonicalSourceHash = receipt.parent_source_hash;
        log(`  [isa] adopted the parent archive captured by ${r.d && r.d.id ? r.d.id : 'a verifier'} `
          + `(tree ${isaCanonicalSourceHash.slice(0, 12)}). Mechanism claims can now be DIFFED from `
          + 'round 1 instead of reported indeterminate until something commits.');
        break;
      }
    } else {
      // Disagreement check. Every candidate in a round forks from the same canonical tree, so every
      // verifier that names a parent must name the same one. If one does not, either canonical moved
      // under the round or an archive belongs to another tree -- and both make the pointer unsafe.
      const conflict = clean.find((r) => {
        const rec = r.ver && r.ver.isa_evidence;
        return rec && typeof rec.parent_source_hash === 'string' && rec.parent_source_hash
          && isaCanonicalSourceHash && rec.parent_source_hash !== isaCanonicalSourceHash;
      });
      if (conflict) {
        const rec = conflict.ver.isa_evidence;
        log(`  [isa] DROPPING the canonical parent archive: ${conflict.d && conflict.d.id ? conflict.d.id : 'a verifier'} `
          + `reports its parent tree as ${rec.parent_source_hash.slice(0, 12)} while the stored parent `
          + `is ${isaCanonicalSourceHash.slice(0, 12)}. Two trees cannot both be the parent of one `
          + 'round, so the pointer is not trustworthy and is cleared rather than guessed at. Verdicts '
          + 'go back to indeterminate until a parent is re-established.');
        isaCanonicalArchive = null;
        isaCanonicalSourceHash = null;
      }
    }
  }
  const verified = clean.filter(r => {
    if (!(r.ver && r.ver.policy_pass === true &&
          says(r.ver.status, 'verified') && says(r.ver.correctness, 'pass'))) return false;
    // Finding (62): a metric fault and a slow kernel are different verdicts (60).
    // Finding (67): so is a measurement that cannot be tied to the pinned oracle.
    // Finding (69): so is a policy pass with nothing behind it.
    // Finding (87): so is a measurement of a tree whose twin was never checked.
    const metricReason = primMetricReason(r.ver) || oracleDrift(r.ver) || policyReject(r.ver)
      || twinReject(r.ver);
    // And so is a candidate whose own ISA receipt says the mechanism it declared
    // never reached the machine code -- that is not a slow kernel, it is an
    // untested one (isa_evidence=gate only).
    //
    // Held as its OWN expression rather than appended to the chain above, because
    // that chain's exact text is pinned by a guard proving the twin gate is wired
    // into the admission path. Widening that regex to make room for a new clause
    // would weaken the check finding (87) exists to keep, which is a bad trade for
    // one line of syntax.
    const isaReason = isaEvidenceReject(r.ver);
    if (metricReason || isaReason) {
      log(`  ${r.d && r.d.id ? r.d.id : 'candidate'} verified but not a candidate: `
        + `${metricReason || isaReason}`);
      return false;
    }
    return primSpeedup(r.ver) > CANDIDATE_FLOOR;
  });
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
    // The incumbent as re-measured beside THIS candidate. Preferred over the stored `bestPerCase`
    // at the gate below, because both arms then come from one session and the drift between
    // sessions -- which is larger than the gains being judged -- cancels instead of being scored.
    control_per_case: Array.isArray(r.ver.control_per_case) && r.ver.control_per_case.length
      ? r.ver.control_per_case : null,
    // Carried for the per-route commit gate. The VERIFIER's value wins over the engineer's: its
    // per_case is the table the gate compares and the one that becomes `bestPerCase`.
    target_routes: Array.isArray(r.ver.target_routes) ? r.ver.target_routes
      : (Array.isArray(r.d.target_routes) ? r.d.target_routes : undefined),
    // Carried on the candidate itself rather than re-looked-up after the sort, so
    // the archive cannot be matched to the wrong winner. An `integrated` candidate
    // never gets this field: the integrator builds a fresh tree that nothing
    // captured, and inheriting a predecessor's archive there is exactly the stale
    // attribution this field exists to make impossible.
    isa_archive: r.ver.isa_evidence && typeof r.ver.isa_evidence.archive === 'string'
      ? r.ver.isa_evidence.archive : null,
    // The tree that archive describes. Carried so that when this candidate becomes canonical, the
    // parent pointer records WHICH tree it is the parent of, and the next round can check rather
    // than trust the pairing.
    isa_source_hash: r.ver.isa_evidence && typeof r.ver.isa_evidence.source_hash === 'string'
      ? r.ver.isa_evidence.source_hash : null,
  }));

  let integrate = null;
  if (verified.length >= 2) {
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
      ? (policyReject(integrate, 'integrate') || twinReject(integrate, 'integrate')) : null;
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

  // The commit decision for ONE candidate, as a pure function. SELECTION below and the logged
  // verdict further down both call it, so they cannot answer differently -- two implementations of
  // "does this pass" is one more than can be trusted, and this particular pair is where the
  // divergence fixed here actually lived.
  const judgeCandidate = (cand) => {
    const sameSession = !!cand.control_per_case;
    const legacyImproved = cand.geomean > cumulative * (1 + MIN_IMPROVE);
    // Always called, table or no table. This used to be conditional on ROUTE_BANDS, which was
    // correct while an absent table meant the gate had nothing to judge with; it now means every
    // route is held to MIN_ROUTE_WIN, so skipping the call would silently put the run back on the
    // suite threshold -- the exact "shipped feature that decides nothing" state this replaced.
    const rv = routeGate(cand.per_case, sameSession ? cand.control_per_case : bestPerCase,
      ROUTE_BANDS, { targetRoutes: cand.target_routes });
    if (!rv || !rv.applicable) {
      return { sameSession, legacyImproved, routeVerdict: rv, suiteSaysYes: legacyImproved,
        improved: legacyImproved };
    }
    // No union any more: the gate's own two conditions ARE the decision. `legacyImproved` is still
    // computed, and still logged whenever the two disagree, because it is the number every prior
    // round of this lane was judged on and a change of gate that cannot be audited against the old
    // one is a change nobody can check.
    return { sameSession, legacyImproved, routeVerdict: rv, suiteSaysYes: legacyImproved,
      improved: rv.accepted };
  };

  candidates.sort((a, b) => b.geomean - a.geomean);
  // The highest geomean OFFERED this round, kept before any reordering below, because `bestSeen`
  // means "the best any candidate measured, committed or not" and that must not change just
  // because a different candidate was the one that could be banked.
  const topOffered = candidates.length ? candidates[0].geomean : 0;
  // SELECTION is "the best candidate that PASSES the gate", not "the best candidate".
  //
  // It used to be `candidates[0]`, ranked on suite geomean, while the ACCEPTANCE test below is
  // per-route -- two criteria with nothing in between. So a round could be holding a candidate the
  // gate would have accepted and still bank nothing. That is not hypothetical: on lane
  // coldstart_newgate_20260819 wave 1 round 2 the top candidate (0.924) was refused for giving back
  // 0.0006 ms on decode_m2_square, while the second (0.632) was ACCEPT with zero banded regressions
  // and was never offered to the gate at all. The hole did not exist while both halves ranked on the
  // same geomean; it was introduced by changing only the acceptance test, and it can only ever cost
  // a round work it had already paid for.
  //
  // Deliberately NOT added: a floor requiring the chosen candidate to also beat `cumulative`.
  // `cumulative` is an absolute from an earlier session, and refusing a candidate that is
  // flat-or-better on every route against its OWN session's control, because a stale absolute
  // drifted underneath it, would reintroduce exactly the cross-session comparison the control arm
  // exists to remove. The disagreement is logged instead: it is a fact about the round, not a reason
  // to discard the round's only bankable result.
  const passIdx = candidates.findIndex(c => judgeCandidate(c).improved);
  if (passIdx > 0) {
    const chosen = candidates.splice(passIdx, 1)[0];
    const refused = candidates[0];
    log(`  [gate r${round}] the highest-geomean candidate (${refused.id}, ${refused.geomean.toFixed(5)}) ` +
        `does not pass the commit gate, but ${chosen.id} (${chosen.geomean.toFixed(5)}) does; ` +
        'selecting the best candidate that PASSES rather than banking nothing. Ranking is by suite ' +
        'geomean and acceptance is per-route, so the top of the ranking is not always the one the ' +
        'gate can take.');
    if (chosen.geomean <= cumulative) {
      log(`  [gate r${round}] NOTE: ${chosen.id}'s suite geomean ${chosen.geomean.toFixed(5)} does not ` +
          `exceed the incumbent's stored ${cumulative.toFixed(5)}. Those two numbers come from ` +
          'different sessions; the per-route verdict that accepted it is paired inside one. Committing ' +
          'it will move `cumulative` DOWN even though no route regressed past its band.');
    }
    candidates.unshift(chosen);
  }
  const winner = candidates[0] || null;
  const verdict = winner ? judgeCandidate(winner) : null;
  const legacyImproved = !!(verdict && verdict.legacyImproved);
  let improved = legacyImproved;
  let routeVerdict = null;
  if (winner) {
    // Prefer the incumbent arm re-measured in the candidate's OWN session. Both arms then come from
    // one invocation, so the 1.5-3% a single unchanged tree wanders BETWEEN invocations cancels
    // instead of being scored as the patch's effect -- and that drift is larger than the per-round
    // gains being judged, which makes this the difference between a gate and a coin flip on a
    // marginal round. Falling back to `bestPerCase` is still much better than the suite geomean, so
    // its absence degrades the gate rather than disabling it; which side was used is always logged,
    // because "compared across sessions" is a caveat a reader of the verdict has to be handed.
    const sameSession = verdict.sameSession;
    routeVerdict = verdict.routeVerdict;
    if (routeVerdict.applicable) {
      // The decision is the gate's own conjunction: suite improved AND one route cleared its bar,
      // minus a catastrophic-regression fence. See the header above `routeGate` for what this
      // replaced and what the previous rule cost.
      improved = verdict.improved;
      log(`  [gate r${round}] ${routeVerdict.accepted ? 'ACCEPT' : 'REFUSE'} -- ${routeVerdict.reason}`);
      if (routeVerdict.regressed.length && routeVerdict.accepted) {
        // Banked WITH ground given up somewhere. Not a refusal -- the objective is the suite
        // average and the average went up -- but the routes that paid for it are named, because a
        // ledger that records only the win cannot answer "which shape got slower" three waves later.
        log(`  [gate r${round}] banked while giving ground on ${routeVerdict.regressed.length} ` +
            `route(s): ${routeVerdict.routes.filter(r => r.status === 'regressed')
              .map(r => `${r.route} ${(-r.delta_frac * 100).toFixed(2)}% (floor ` +
                        `${(r.floor * 100).toFixed(2)}%)`).join(', ')}. The suite average is the ` +
            'objective, so this is accepted; it is named so the cost is on the record.');
      }
      log(`  [gate r${round}] incumbent side: ${sameSession
        ? 'the verifier\'s SAME-SESSION control arm (both arms from one invocation)'
        : 'the stored best_per_case from an earlier round -- NO same-session control was returned, so '
          + 'this verdict carries the between-invocation drift (1.5-3% on an unchanged tree) that a '
          + 'control arm would have cancelled. Read a marginal verdict with that in mind.'}`);
      // A win banked without a declared target route is a win nobody attributed. Not refused --
      // refusing correct work over a missing label is the defect this codebase keeps having to fix --
      // but never silent either, because "the mechanism landed" and "something moved" are different
      // claims and only one of them is evidence.
      if (routeVerdict.accepted && !(Array.isArray(winner.target_routes) && winner.target_routes.length)) {
        log(`  [gate r${round}] NOTE: accepted with no declared target_routes, so the improvement on ` +
            `${routeVerdict.improved.join(', ') || '(none named)'} was not checked against a claimed ` +
            'mechanism. An incidental gain and a realized mechanism are indistinguishable here.');
      }
      if (improved !== legacyImproved) {
        // Logged on every disagreement, in both directions, so the change of gate is auditable
        // against the number every prior round was judged on rather than silently replacing it.
        log(`  [gate r${round}] this OVERTURNS the legacy suite-geomean gate ` +
            `(${legacyImproved ? 'it would have committed' : 'it would have refused'}: ` +
            `winner ${winner.geomean.toFixed(5)} vs cumulative ${cumulative.toFixed(5)} ` +
            `at MIN_IMPROVE=${MIN_IMPROVE}). ${improved
              ? 'A suite geomean divides a single-route win by the route count, and it is measured '
                + 'against a stored absolute from an earlier session; this gate pairs both arms '
                + 'inside one invocation and asks the route directly.'
              : 'The suite average moved but nothing cleared its own noise, or a route regressed '
                + 'catastrophically -- neither of which an averaged number can distinguish.'}`);
      }
    } else {
      log(`  [gate r${round}] commit gate NOT APPLICABLE (${routeVerdict.reason}); ` +
          `falling back to the legacy suite-geomean gate at MIN_IMPROVE=${MIN_IMPROVE}. This is not ` +
          'a missing floor table -- the gate no longer needs one -- it is missing paired per-case ' +
          'times, so nothing can be compared route by route. Have the verifier return per_case and ' +
          'control_per_case.');
    }
  }
  // Separate question from `improved`: is the SEARCH advancing, not did it beat the incumbent. The
  // `bestSeen > 0` guard keeps round 1 deciding on `improved` alone; from then on bestSeen >= cumulative
  // at the default floor, so at the default PROGRESS_DELTA this implies `improved` and changes nothing.
  // A round with NO candidate is never progress, so a dead round still counts against MAX_NO_IMPROVE.
  const suiteProgress = !!(winner && bestSeen > 0 && winner.geomean > bestSeen * (1 + PROGRESS_DELTA));
  // ...and the same unit error, in the stall counter. This one does not refuse a candidate, it ends
  // the WAVE: `MAX_NO_IMPROVE` defaults to 2, so two rounds scored as stalls stop the loop with
  // budget unspent. Measured cost on the greedy lane: wave 6 stopped after 3 rounds having used 8 of
  // 12 directions.
  //
  // Why it misfires. `PROGRESS_DELTA` defaults to `MIN_IMPROVE`, and it is applied to the SUITE
  // geomean against the best suite number ever seen. A single route improving 7% moves an unweighted
  // eleven-route geomean by roughly 0.6%, so a round that produced a real, banded, independently
  // verified route win is scored as "not advancing" and counted toward stopping the search -- while
  // the same round's candidate may be getting committed by the gate above. A loop that commits a
  // round and simultaneously counts it as a stall is measuring two different things and calling both
  // progress.
  //
  // So a round in which any route cleared its own bar counts as progress. This is strictly more
  // permissive, which is the correct direction for a STOPPING rule: the cost of a false "advancing"
  // is one more round of a budget the caller already authorised, while the cost of a false
  // "stalled" is every remaining round of the wave.
  //
  // Deliberately broader than the commit gate in TWO ways now. It ignores the suite condition -- a
  // route that genuinely moved is evidence the search found something, whether or not the average
  // followed -- and it ignores routes that gave ground within noise. It still respects the
  // catastrophic fence, because a round that made one shape a third slower has not advanced
  // anything.
  const routeProgress = !!(routeVerdict && routeVerdict.applicable && routeVerdict.improved.length
    && !routeVerdict.catastrophic.length);
  const madeProgress = suiteProgress || routeProgress;
  if (routeProgress && !suiteProgress) {
    log(`  [gate r${round}] the search ADVANCED on route evidence (${routeVerdict.improved.join(', ')}) ` +
        `even though the suite geomean ${winner.geomean.toFixed(5)} did not clear bestSeen ` +
        `${bestSeen.toFixed(5)} by PROGRESS_DELTA=${PROGRESS_DELTA}. Scored as progress: a stalled ` +
        'round ends the wave, and one route of many moving cannot shift an unweighted geomean far ' +
        'enough to register.');
  }

  // --- (e) Commit the winner into the canonical workspace ---------------
  if (improved) {
    const commit = await agentT(
      `You are the TechLead committing round ${round}'s winning patch into the canonical workspace.
\`\`\`bash
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
cd ${CANONICAL}
git rev-parse HEAD    # report this as head_sha_before
git checkout -- .
# Try a plain apply first, then a 3-way apply (auto-reconciles context-line drift against the blobs)
# before falling back to a manual reconstruction. --3way resolves most "patch does not apply" cases
# that are just context offsets, so the manual path is only hit on a genuine semantic conflict.
git apply ${winner.patch} || git apply --3way ${winner.patch}
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "round ${round} winner: ${winner.source} (${winner.geomean.toFixed(2)}x)"
git --no-pager diff --binary "$(git rev-list --max-parents=0 HEAD)..HEAD" > ${EVAL_DIR}/current_best.diff
git rev-parse HEAD    # report this as head_sha_after
git status --porcelain  # must be EMPTY: a dirty tree here means the commit did not capture everything
\`\`\`
\`git apply\` is only safe here because ${CANONICAL} is its own git repository. Inside a directory that
merely sits UNDER some other repository's working tree, \`git apply\` prints \`Skipped patch\` and **exits
0** — a clean-looking receipt for a patch that was never applied. If you ever materialize a tree outside
its own repo, use \`patch -p1 --forward\` and re-hash the files afterwards; judge by "is this directory
its own repo?", never by "did the command return non-zero?".
If BOTH \`git apply\` and \`git apply --3way\` fail, inspect the patch and apply it manually (edit the
files to match the patch's intent), then \`add -A\` + commit. Before executing correctness, run
\`python3 ${WORKFLOW_DIR}/scripts/candidate_policy_scan.py\` over every candidate-owned source/build path
and candidate ELF, exempting only separately frozen immutable baseline/oracle paths. A finding, inspection
failure, or absent passing receipt means committed=false. The applied source is NOT guaranteed to match
the patch verbatim after a hand-merge, so after committing, RE-RUN both the policy gate and COMMANDMENT
correctness (cd ${CANONICAL} and use gpu_lock); only report committed=true if both pass. Even after a clean
apply, the policy receipt is mandatory because final materialization is a new trust boundary. Return JSON
{committed, current_best_diff, note, head_sha_before, head_sha_after}. Report the two SHAs verbatim from
\`git rev-parse HEAD\`; if they are equal, the commit did not happen and \`committed\` MUST be false no
matter how the rest of the run looked.`,
      { phase: 'Merge', label: `commit r${round}`, schema: COMMIT_SCHEMA });

    // (127). The result of this call used to be discarded, so `committed: false` -- and the empty
    // result of a dead agent -- advanced the ledger exactly like a success. From here on the round's
    // bookkeeping only moves if the canonical tree demonstrably moved with it. The alternative
    // (advance anyway) is the strictly worse failure: subsequent rounds fork engineers off a tree the
    // ledger no longer describes, and every measurement after that is against the wrong parent.
    const headMoved = !!(commit && commit.head_sha_before && commit.head_sha_after &&
      commit.head_sha_before.trim() !== commit.head_sha_after.trim());
    const commitOK = !!(commit && commit.committed === true) &&
      // Absent SHAs are tolerated (an older TechLead prompt may not report them) but equal ones are
      // not: that is a positive claim that nothing changed, contradicting `committed: true`.
      !(commit.head_sha_before && commit.head_sha_after && !headMoved);
    if (commitOK) {
      cumulative = winner.geomean;
      bestPerCase = winner.per_case && winner.per_case.length ? winner.per_case : bestPerCase;
      finalWinner = winner;
      // Inside the (127) guard on purpose. This counter is what the curation step reads to decide
      // whether the run is entitled to teach; a refused commit that still incremented it would let a
      // run that moved nothing file a card.
      roundsCommitted += 1;
      // Inside the (127) guard, and deliberately absent from the `else`. This pointer
      // names the tree the NEXT round's candidates are mutations of, so it may only
      // move when the canonical tree demonstrably moved. On a refused commit canonical
      // is unchanged, which means the archive it already points at is still the correct
      // parent -- leaving it alone is the same rule (127) states for `cumulative` and
      // `bestPerCase`, applied to the evidence pointer.
      if (ISA_ENABLED) {
        isaCanonicalArchive = winner.isa_archive || null;
        // Moved together with the path, always, including when both go null. A hash left behind by a
        // cleared archive would make the next round's disagreement check compare a fresh parent
        // against a retired tree and clear a pointer that was correct.
        isaCanonicalSourceHash = isaCanonicalArchive ? (winner.isa_source_hash || null) : null;
        if (!isaCanonicalArchive) {
          log(`  [isa] the new canonical (${winner.source}) carries no ISA archive, so next round's `
            + 'mechanism claims will be reported indeterminate rather than diffed against a stale '
            + 'parent. Expected for an integrated winner.');
        } else if (!isaCanonicalSourceHash) {
          log(`  [isa] the new canonical (${winner.source}) carries an ISA archive but no source hash, `
            + 'so the next round cannot check that its parent is this tree. The archive is still used '
            + '-- it was captured from the tree that just became canonical -- but the disagreement '
            + 'check is inert until a verifier names a parent tree.');
        }
      }
    } else {
      log(`  [commit r${round}] REFUSED to advance the ledger: winner ${winner.source} ` +
          `(${winner.geomean.toFixed(4)}x) was NOT committed into the canonical workspace ` +
          `(committed=${commit ? commit.committed : 'no result'}` +
          `${commit && commit.head_sha_before && commit.head_sha_after
            ? `, HEAD ${commit.head_sha_before.trim().slice(0, 8)} -> ${commit.head_sha_after.trim().slice(0, 8)}` +
              `${headMoved ? '' : ' UNCHANGED'}` : ''}). ` +
          `${commit && commit.note ? `note: ${commit.note} ` : ''}` +
          'cumulative, best_per_case and final_winner are left where they were, so the next round ' +
          'plans against the tree that actually exists. The patch is not lost: it is still on disk ' +
          `at ${winner.patch} and can be re-competed or applied by hand.`);
    }

    // --- (f) Re-profile the new best ------------------------------------
    // Only when there IS a new best in the tree. Re-profiling an unchanged canonical and filing it
    // as "the new best's bottleneck shift" manufactures a shift out of run-to-run profiler drift
    // (measured at up to 17% between rocprofv3 invocations on this lane) and steers the next round
    // off it.
    if (commitOK) {
      profileSummary = await agentT(
        roleAgent('profile_engineer', 'reprofile', 'Re-profile the new best and explain the bottleneck shift.', {
          WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL, ROUND: round,
          COMMANDMENT, PREVIOUS_METRICS: profileSummary,
        }),
        { phase: 'Optimize', label: `reprofile r${round}`, schema: PROFILE_SCHEMA });
      profileSummary = profileSolStrip(profileSummary, `reprofile r${round}`);
    }
    committedThisRound = commitOK;
  }

  // `topOffered`, not `winner.geomean`: the winner may be a lower-ranked candidate that the gate
  // could take while the top one could not, and "the best any candidate measured" must not fall
  // just because a different one was the one banked.
  if (topOffered > bestSeen) bestSeen = topOffered;
  // `committedThisRound`, never `improved`. `improved` is decided BEFORE the commit is attempted,
  // so a winner that repeatedly fails to LAND (patch will not apply, hand-merge conflict, commit
  // agent dead or honestly reporting committed:false) reset this counter every round while
  // `cumulative` never moved -- and because the next round's winner then clears that same
  // unchanged threshold, the stall counter could never fire and the lane spent its ENTIRE budget
  // re-deriving one patch it could not bank. That is (127)'s rule applied to the stall counter:
  // bookkeeping may only advance when the canonical tree demonstrably moved. It also restores
  // what the pre-knob loop did (`legacy()` in test_candidate_floor.js: "reset ONLY on a commit")
  // while keeping `madeProgress`, the knob that lets a sub-baseline climb keep its budget.
  if (madeProgress || committedThisRound) { phaseStall = 0; } else { phaseStall++; }

  // --- close the cycle, or hand the search on -------------------------------
  if (roundPhase === PHASE_DEEP) {
    // A deep round gets exactly one attempt, and what it did decides everything. Committed means a
    // different program is now canonical, so the ladder restarts and searches THAT. Not committed
    // means a ground-up rewrite starting from the current best failed to beat it, which is the
    // strongest evidence available that there are no routes left -- stronger than any stall count,
    // because it is a negative result from the broadest move the search has.
    if (committedThisRound) {
      advancePhase('the deep round banked a different implementation, so the ladder restarts on it',
        round);
    } else {
      stopReason = 'deep_explore_failed';
      log(`STOP: the deep round banked nothing. A ground-up rewrite from the current best did not `
        + `beat it, so the search is out of routes. Budget used ${dispatched}/${BUDGET} over `
        + `${round} round(s) across ${cycle} cycle(s).`);
    }
  } else if (phaseStall >= MAX_NO_IMPROVE) {
    advancePhase(escalationReason(false, phaseStall, MAX_NO_IMPROVE), round);
  }

  // --- update cross-round memory (insight blackboard + hypothesis ledger)
  const mem = await agentT(
    roleAgent('tech_lead', 'update_memory',
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
      // (127). STATE.json's `cumulative` is the LANE total vs the original seed, so it must be
      // written from CUMULATIVE_VS_SEED -- never from CUMULATIVE_SPEEDUP, which is wave-local and
      // would overwrite a 4.35x lane history with a 1.01x round number on the next resume.
      ...(STATE_DIR ? { STATE_DIR, CANONICAL, CUMULATIVE_SPEEDUP: cumulative,
        CUMULATIVE_VS_SEED: cumulativeVsSeed(), PRIOR_CUMULATIVE_VS_SEED: priorCumulativeVsSeed,
        // The frame, handed over WITH the number. `update_memory` writes STATE.cumulative from
        // CUMULATIVE_VS_SEED and the next wave reads it back as PRIOR -- so if the agent multiplies
        // once more the error compounds per wave. On an oracle-anchored harness CUMULATIVE_SPEEDUP
        // and CUMULATIVE_VS_SEED are the SAME number, which is exactly the shape that invites one
        // more multiplication; saying so is cheaper than catching it a third time.
        VS_SEED_FRAME: vsSeedFrame === 'absolute'
          ? 'ABSOLUTE: the harness times against a fixed external reference, so CUMULATIVE_VS_SEED '
            + 'and CUMULATIVE_SPEEDUP are the same number and neither may be multiplied by '
            + 'PRIOR_CUMULATIVE_VS_SEED. Write CUMULATIVE_VS_SEED into STATE.cumulative verbatim.'
          : 'CHAINED: each wave re-anchors on its incumbent, so CUMULATIVE_VS_SEED is already '
            + 'PRIOR_CUMULATIVE_VS_SEED x CUMULATIVE_SPEEDUP. Do not multiply it again.',
        BEST_PER_CASE: bestPerCase } : {}),
      ...(SHARED_KB ? { SHARED_KB, TARGET_LANGUAGE } : {}),
    }),
    { phase: 'Optimize', label: `tech_lead:memory r${round}`, schema: MEMORY_SCHEMA });
  if (mem) {
    if (mem.insights) history.insights = mem.insights;
    if (mem.ledger) history.ledger = history.ledger.concat(mem.ledger);
    if (mem.bottleneck_now) history.bottleneck_now = mem.bottleneck_now;
    if (mem.suggest_next) history.suggest_next = mem.suggest_next;
  }
  // Enforce the round's KB budget. Honest about what this can and cannot do: stripping a citation
  // does NOT un-influence a direction that was already planned with a card in mind — only the prompt
  // above can do that. What the strip buys is OBSERVABILITY: `kbCapBound` counts the rounds where the
  // planner overran its budget, so "the KB quietly took over planning" becomes a number in the run's
  // return value instead of something you infer from a speedup that got worse.
  if (USE_LEARNED_READ) {
    let seeded = 0; const stripped = [];
    for (const d of directions) {
      if (!d.learned_refs || !d.learned_refs.length) continue;
      if (seeded < KB_DIR_CAP) { seeded++; continue; }
      stripped.push(d.id); d.learned_refs = [];
    }
    // A round where EVERY direction is KB-seeded has no control in it at all.
    if (KB_COLD_DIRECTION && directions.length && directions.every(d => (d.learned_refs || []).length)) {
      const last = directions[directions.length - 1];
      stripped.push(`${last.id}(forced cold)`); last.learned_refs = [];
    }
    if (stripped.length) {
      kbCapBound++;
      log(`Round ${round}: KB budget ${KB_DIR_CAP} exceeded — un-cited ${stripped.join(', ')} ` +
          `(the directions still run). Rounds where the cap bound so far: ${kbCapBound}.`);
    }
  }

  // Join the planner's declared citations against what the verifier independently measured. Derived
  // here rather than self-reported: the planner said which card seeded which direction, the verifier
  // re-measured that direction without knowing, and their join is a fact neither one could fake.
  // `became_winner` and not `verified > 1.0` is the standing test — verified_geomean is measured
  // against the FROZEN baseline, so once a kernel sits at 2.5x cumulative every non-regressing
  // direction clears 1.0 and a card would accrue credit for advancing nothing.
  for (const r of clean) {
    for (const cardRef of (r.d.learned_refs || [])) {
      citations.push({
        card: cardRef, round, direction: r.d.id, specialty: r.d.specialty,
        // null, NOT 0, when the verifier produced nothing. `kb.py` scores <=1.0 as a real loss, so
        // encoding "no evidence" as 0 charged a card for a verifier that crashed or timed out.
        cited_then_verified: r.ver && Number.isFinite(r.ver.verified_geomean)
          ? r.ver.verified_geomean : null,
        // `winner` is only the best candidate OF THIS ROUND; `improved` is whether it beat the
        // incumbent by MIN_IMPROVE. Crediting on `winner` alone let a 1.5x round winner confirm a
        // card while the committed best already sat at 2x — a confirmation for advancing nothing,
        // which is precisely what the three-state scoring was written to avoid. Require both.
        became_winner: !!(winner && winner.id === r.d.id && improved),
      });
    }
  }
  history.rounds.push({
    round,
    directions: directions.map(d => ({ id: d.id, title: d.title, specialty: d.specialty,
      learned_refs: d.learned_refs || [] })),
    results: clean.map(r => ({ id: r.d.id, claimed: r.eng ? r.eng.speedup_geomean : 0,
      verified: r.ver ? r.ver.verified_geomean : 0, status: r.ver ? r.ver.status : (r.eng ? r.eng.status : 'none'),
      // Carried into HISTORY so the next round's planner can tell "this mechanism
      // was measured and did not help" from "this mechanism never reached the
      // machine code". Those two justify opposite next moves, and without this
      // field they are the same ledger entry. Absent when the lane runs isa off.
      ...(ISA_ENABLED ? { mechanism: r.ver && r.ver.isa_evidence
        ? r.ver.isa_evidence.mechanism_verdict : 'no_receipt' } : {}) })),
    // The round's evidence contract. Recorded together so the three cannot drift:
    // the depth that was asked for, whether the artifact behind it exists, and the
    // named reason when it does not. `cannbot`'s round-state rule is the model --
    // "do not rely on the presence of `profile/` or `ir/` to imply the current
    // level; state it directly" -- and the consistency check below is what stops a
    // depth from being claimed without evidence.
    ...(ISA_ENABLED ? {
      // Versioned, and separated into request / reach / status / artifact. The
      // previous record collapsed these into one `evidence_depth` string, which
      // meant "we asked for machine-code evidence" and "we have machine-code
      // evidence" were the same field on a resumed lane. `model` is what lets
      // `reachedStageOf` tell a v2 record from one written before this file
      // existed -- an older round says `isa`, which is neither L3 nor L4, and
      // reading it as either would hand the next planner evidence nobody produced.
      evidence: {
        model: EVIDENCE_MODEL,
        requested_stage: ladderState.requested || null,
        // WHO asked. With L3 on request rather than on a counter, "was this escalation worth it"
        // is only answerable against the role that wanted it -- and a requester whose requests
        // never pay off is a thing to notice.
        requested_by: ladderState.requested_by || [],
        reached_stage: stageReached,
        status: (isaAttribution && isaAttribution.status) || (ladderState.requested
          ? 'unavailable' : 'not_requested'),
        attributed_pass: (irAttribution && irAttribution.attributed_pass) || '',
        stage_transition: (irAttribution && irAttribution.stage_transition) || '',
        tier_used: compilerAttribution && compilerAttribution.tier_used != null
          ? compilerAttribution.tier_used : null,
        ir_artifact_path: (irAttribution && irAttribution.summary_path) || '',
        compiler_artifact_path: (compilerAttribution && compilerAttribution.summary_path) || '',
        skip_reason: isaAnalysisSkipped || '',
      },
      // Retained under their old names for one release: `test_lane_gates.js`, the
      // report phase and any STATE written by a previous build read these, and
      // renaming them in the same change that renamed the vocabulary would make a
      // resume failure indistinguishable from a ladder bug.
      evidence_depth: stageReached,
      escalation_from: ladderState.from || '',
      escalation_reason: ladderState.reason || '',
      isa_attribution_path: isaAttribution && isaAttribution.summary_path
        ? isaAttribution.summary_path : '',
      analysis_skipped_reason: isaAnalysisSkipped || '',
    } : {}),
    integrate: integrate ? { conclusion: integrate.conclusion, geomean: integrate.best ? integrate.best.geomean : 0 } : null,
    winner: winner ? { source: winner.source, geomean: winner.geomean } : null,
    improved, committed: committedThisRound, cumulative, cumulative_vs_seed: cumulativeVsSeed(),
    search_phase: roundPhase,
    // Carried so NEXT round's ladder can see it. The planner's request cannot be honoured in the
    // round it was made -- the ladder had already run by then -- so the record is the only place it
    // can survive to the round that can act on it.
    ...(plan && plan.request_l3 === true
      ? { plan_request_l3: true, plan_request_l3_reason: String(plan.request_l3_reason || '').trim() }
      : {}),
  });
  log(`Round ${round} done. winner=${winner ? winner.source + ' ' + winner.geomean.toFixed(2) + 'x' : 'none'}` +
      `${improved && !committedThisRound ? ' (WINNER NOT COMMITTED -- canonical unchanged)' : ''}` +
      `, cumulative=${cumulative.toFixed(2)}x vs this wave's baseline` +
      `${priorCumulativeVsSeed !== 1.0 ? ` (${cumulativeVsSeed().toFixed(2)}x vs seed)` : ''}` +
      `, phase=${roundPhase}, stall=${phaseStall}/${MAX_NO_IMPROVE}`);
}
// Falling out of the `while` with nothing set means the budget condition failed. Named rather than
// inferred: "a ground-up rewrite could not beat the incumbent" and "the caller's budget ran out" are
// very different verdicts about the search, and a report that says only "stopped" invites the wrong
// one -- the first says the kernel is done, the second says the run was.
if (!stopReason) stopReason = 'budget_exhausted';
log(`Search ended: ${stopReason}. Final state ${searchPhase}, ${cycle} cycle(s). Ladder: ` +
    phaseLog.map(p => `${p.phase}@r${p.entered_round}`).join(' -> ') +
    `. Budget used ${dispatched}/${BUDGET} over ${round} round(s).`);

// ===========================================================================
// PHASE: Final report (TechLead)
// ===========================================================================
phase('Report');
// The slow half of the loop, run once. Per-round work SPENDS machine-code evidence;
// this turns what was spent into something the cheap layer knows next time, so the
// deep levels get rarer instead of becoming a fixed tax on every run. Dispatched
// only when there is something to synthesise -- a run whose rounds all stayed at
// pattern depth and refuted nothing has produced no durable lesson, and asking for
// one anyway is how a knowledge base fills with rules nobody validated.
let isaSynthesis = null;
if (ISA_ENABLED) {
  const deepRounds = (history.rounds || []).filter(r => {
    const stage = reachedStageOf(r);
    return stage === STAGE_L3 || stage === STAGE_L4 || stage === STAGE_LEGACY
      || (Array.isArray(r.results) && r.results.some(x => x && x.mechanism === 'refuted'));
  });
  if (deepRounds.length) {
    isaSynthesis = await agentT(
      roleAgent('tech_lead', 'synthesize_isa_lessons',
        'Distil this run\'s machine-code evidence into reusable rules, or report that none qualified.', {
          EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, KERNEL_KNOWLEDGE_DIR, HISTORY: history,
          ISA_ATTRIBUTIONS: deepRounds.map(r => ({
            round: r.round, evidence_depth: reachedStageOf(r),
            escalation_reason: r.escalation_reason || '',
            // The pass, when the round has a v2 record. This is what makes a
            // synthesised rule reusable at L3 rather than only at the ISA layer:
            // "signal X means direction Y" is a machine-code rule, "pass P drops
            // structure S unless condition C" is a lowering one.
            attributed_pass: (r.evidence && r.evidence.attributed_pass) || '',
            stage_transition: (r.evidence && r.evidence.stage_transition) || '',
            attribution_path: (r.evidence && r.evidence.ir_artifact_path)
              || r.isa_attribution_path || '',
            compiler_attribution_path: (r.evidence && r.evidence.compiler_artifact_path) || '',
            analysis_skipped_reason: (r.evidence && r.evidence.skip_reason)
              || r.analysis_skipped_reason || '',
            mechanisms: (r.results || []).map(x => x && x.mechanism).filter(Boolean),
          })),
          OUTPUT_PATH: `${KERNEL_KNOWLEDGE_DIR}/isa_signals/learned_rules.md`,
        }),
      { phase: 'Report', label: 'tech_lead:synthesize_isa_lessons', schema: ISA_SYNTHESIS_SCHEMA });
    if (isaSynthesis) {
      log(`  [isa] synthesis: promoted=${isaSynthesis.promoted || 0} `
        + `anti_signals=${isaSynthesis.anti_signals || 0}`
        + (isaSynthesis.promoted ? ` -> ${isaSynthesis.path}` : ` (${isaSynthesis.reason || 'no reason given'})`));
    }
  } else {
    log('  [isa] synthesis skipped: no round reached machine-code depth and no mechanism was '
      + 'refuted, so this run produced nothing durable to promote.');
  }
}
const report = await agentT(
  roleAgent('tech_lead', 'report', 'Write the final report and the cumulative final patch.', {
    ...(isaSynthesis ? { ISA_SYNTHESIS: isaSynthesis } : {}),
    EVAL_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR,
    HISTORY: history, FINAL_WINNER: finalWinner, BASELINE_PER_CASE,
    BASELINE_GEOMEAN_MS, CUMULATIVE_SPEEDUP: cumulative,
    // (127). The report's headline number is wave-local by construction (the director validates the
    // final patch against BASELINE_PER_CASE). The lane total is a separate, labelled figure.
    CUMULATIVE_SPEEDUP_FRAME: 'vs this wave\'s BASELINE_PER_CASE == the incumbent tree at wave start',
    CUMULATIVE_VS_SEED: cumulativeVsSeed(),
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

// ===========================================================================
// PHASE: UpdateExperience — distill ONE reusable card into this workflow's knowledge/learned/
// (see knowledge/learned/README.md). Runs ONCE, at the very end: never per round, because only
// the final director-verified speedup is evidence a card may cite. `validation` is required for
// the same reason — without it `finalPrimary` falls back to the TechLead's own unverified
// `cumulative`. ADD-only, so a skipped or failing step leaves the run byte-neutral.
// ===========================================================================
let learned_card = null;
// Two runs must never become a card, and neither is a failure. A run that shared its GPU measured
// contention, not the kernel. A run on a HELD-OUT kernel is the instrument for measuring whether the
// KB helps at all — distil it and the next A/B over that kernel reads back its own answer and looks
// spectacular. Defaults are permissive, so the CALLER is the one who has to declare a busy box.
const BOX_QUIET = String(A.box_quiet != null ? A.box_quiet : 'true') === 'true';
const HELD_OUT = String(A.held_out != null ? A.held_out : 'false') === 'true';
const kbGate = !BOX_QUIET ? 'the box was contended — the number measured neighbours, not the kernel'
  : HELD_OUT ? 'this kernel is in the held-out split and is the measuring instrument'
  : '';
if (kbGate) log(`[kb] not distilling: ${kbGate}.`);
// `validation.validation_status === 'accepted'` and not merely "a validation object exists": a
// `flagged` result means the director found a reason not to believe the number (patch install
// no-op, correctness fail, contended box), and curating from it teaches the next run a lesson this
// run did not earn. Reported in review of #411.
const kbAccepted = String((validation && validation.validation_status) || '').toLowerCase() === 'accepted';
if (!kbGate && UPDATE_EXPERIENCE_ON && kbAccepted && Number.isFinite(finalPrimary) && finalPrimary > 1.0) {
  try {
    learned_card = await agentT(
      roleAgent('update_experience', 'Validate',
        'Curate one distilled learned card from this lane\'s verified win (ADD-only, measured evidence, ' +
        'ratios not wall-clock; record the pitfalls hit; total-then-per-direction for a stacked win).', {
          SCOPE: 'lane', LEARNED_DIR, SKILL_DIR: WORKFLOW_DIR, EVAL_DIR,
          PERF_KNOWLEDGE_DIR: KERNEL_KNOWLEDGE_DIR,
          WINNER: {
            kernel: KERNEL_NAME, language: TARGET_LANGUAGE, mode: MODE, gfx: GFX,
            kernel_class: (analysis && analysis.kernel_type) || '',
            speedup: finalPrimary, validation_status: validation ? validation.validation_status : '',
            bottleneck: profileSummary ? profileSummary.bottleneck : '',
          },
          // `rounds[]` (per-round directions + per-candidate claimed/verified/status + running
          // `cumulative`) is what makes the card's `stack:` attribution and `pitfall:` lines derivable.
          HISTORY: history,
          // What this run's plan cited and what the verifier then measured. CONTEXT ONLY — the
          // counters are applied by `kb.py drain` from the ledger filed below, never by this agent.
          // The comment that used to sit here said the curator merges them, which stopped being
          // true when that arithmetic moved into code.
          CITATIONS: citations,
          PROFILE: profileSummary,
          REPORT_PATH: report && report.report_path ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`,
        }),
      { phase: 'Validate', label: 'update_experience', schema: UPDATE_EXPERIENCE_SCHEMA });
    if (learned_card && learned_card.card_path) {
      log(`[kb] learned card ${learned_card.action || 'written'}: ${learned_card.card_path}`);
    }
  } catch (e) {
    log(`[kb] update_experience skipped: ${e && e.message ? e.message : e}`);
  }
}

// FILE THE CITATION LEDGER — OUTSIDE the curation gate, and deliberately so.
// This is the loss half of the loop. It first shipped INSIDE the `finalPrimary > 1.0` branch above,
// which is the exact bias it exists to remove: the runs that produce losses are the ones that did
// not beat the baseline, were flagged, or ran on a contended box, and every one of them was
// filtered out before it could file anything. Selecting successful runs into the feedback data is
// worse than no feedback, because the resulting counters look earned. Reported in review of #411.
// Curation (writing a NEW card) still requires a verified win — a run with nothing to teach must
// not teach — but the ledger of what the cards ALREADY in the tree did is owed for every run.
// `held_out` is the one exception: that split is the measuring instrument, so it neither writes
// cards nor votes on them.
// A workflow script has no filesystem, hence the one-line agent; the JSON is assembled by
// `kb.py cite` rather than by the agent, because a proposal hand-written by a model is one more
// place for the schema to drift.
if (citations.length && !HELD_OUT) {
  try {
    await agentT(
      `Run EXACTLY this command and nothing else. Do NOT edit any file.
\`\`\`bash
cat <<'CITEJSON' | python3 ${WORKFLOW_DIR}/scripts/kb.py --kb-dir ${LEARNED_DIR} cite \\
--run-id ${JSON.stringify(KERNEL_NAME)} \\
--kernel ${JSON.stringify(KERNEL_NAME)} --citations -
${JSON.stringify(citations)}
CITEJSON
\`\`\`
Return {"filed": <the "citations" number the command printed, or 0>}.`,
      { phase: 'Validate', label: 'kb:cite', effort: 'low',
        schema: { type: 'object', properties: { filed: { type: 'number' } },
                  required: ['filed'], additionalProperties: true } });
    log(`[kb] citation ledger filed: ${citations.length} row(s) for the next drain`);
  } catch (e) {
    log(`[kb] citation ledger NOT filed: ${e && e.message ? e.message : e}`);
  }
}

// ===========================================================================
// Write this run's outcome back to kb_artifacts/ — the producer half of the loop.
// The script applies its own gate and never fails the run; the `finalPrimary > 1.0`
// pre-check just avoids spending an agent on a run that cannot pass the gate anyway.
// ===========================================================================
let kb_written = null;
if (KB_WRITE_OK && GFX && Number.isFinite(finalPrimary) && finalPrimary > 1.0) {
  const kernelClass = (analysis && analysis.kernel_type) || 'unknown';
  const finalPatch = report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`;
  const reportPath = report && report.report_path ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`;
  // What the store cannot infer from a diff but needs to curate this entry later. `direction` is the
  // IDEA, and only a warm start supplies a real one: a round title is free-form per run, so kebabbing
  // it would mint a unique label every time and group nothing. Unlabeled is honest — resolve treats
  // such an entry as its own direction, and a curation pass assigns the shared label later.
  const winnerDirection = kebab(warm_start.direction || '');
  const metricKind = HAS_WORKLOAD ? 'time_weighted' : 'geomean';
  // Commas separate the list, so a case name may not contain one.
  const caseNames = (bestPerCase || []).map(c => c && String(c.name || '').replace(/,/g, ';'))
    .filter(Boolean).join(',');
  // write-remote runs the directory write first and files the same entry in the store, so the two
  // planes cannot drift; its extra `remote` field rides along under the schema's open object.
  // `--plane both` extends that ordering one hop further: directory tree, then local store, then the
  // service. Unlike the read there is no branch on credentials here, because `_open_plane` already
  // makes the right call — a `both` without them degrades to the local planes and reports
  // `remote_unavailable` rather than failing, which is the behaviour we would have written by hand.
  const remoteWriteOn = KB_REMOTE !== 'off' && !!KB_STORE_DIR;
  const writeCmd = remoteWriteOn
    ? `write-remote --plane both --store ${JSON.stringify(KB_STORE_DIR)}${KB_VERSION_FLAG}`
    : KB_MODE === 'store'
      ? `write-remote --plane local --store ${JSON.stringify(KB_STORE_DIR)}${KB_VERSION_FLAG}`
      : 'write';
  kb_written = await agentT(
    `You are the experience writer. Run EXACTLY this command (it applies its own gates and prints a ` +
    `single-line JSON) and return that JSON verbatim as StructuredOutput. If the command errors, return ` +
    `{"written": false, "reason": "io_error"}` +
    (remoteWriteOn
      ? ` — do NOT retry it and do NOT adjust its arguments. It may reach the shared KB Store service, ` +
        `and every write that service accepts is PERMANENT (it exposes no delete), so a second attempt ` +
        `creates a second permanent record instead of fixing the first.`
      : `.`) + `
\`\`\`bash
${remoteWriteOn ? KB_ENV_PRELUDE + '\n' : ''}python3 ${JSON.stringify(EXPERIENCE_STORE)} ${writeCmd} --root ${JSON.stringify(KB_ARTIFACTS_DIR)} \\
  --kernel-name ${JSON.stringify(KERNEL_NAME)} --language ${JSON.stringify(TARGET_LANGUAGE)} \\
  --gfx ${GFX} --kernel-class ${JSON.stringify(kernelClass)} \\
  --speedup ${finalPrimary} --baseline-wall-ms ${BASELINE_GEOMEAN_MS} \\
  --patch ${JSON.stringify(finalPatch)} --eval-dir ${JSON.stringify(EVAL_DIR)} \\
  --report ${JSON.stringify(reportPath)} --metric-kind ${metricKind} \\
  --direction ${JSON.stringify(winnerDirection)} --case-names ${JSON.stringify(caseNames)}\
${warm_start.exp_dir ? ` \\\n  --parent ${JSON.stringify(warm_start.exp_dir)}` : ''}
\`\`\``,
    { phase: 'Validate', label: 'kb:write', schema: WARMSTART_WRITE_SCHEMA });
  const remoteWrite = (kb_written && kb_written.remote) || null;
  if (remoteWrite) {
    log(remoteWrite.written
      // `replaced` distinguishes the two write outcomes the store has: a NEW patch appends a session
      // under the same key, the SAME patch remeasured lands back on its own content-addressed one.
      ? `[kb] plane=store ${remoteWrite.replaced ? 'updated' : 'appended'} ` +
        `${remoteWrite.canonical_id}/${remoteWrite.session_id}${remoteWrite.champion ? ' (champion)' : ''}`
      : `[kb] plane=store not written: ${remoteWrite.reason || 'unknown'}`);
  }
  log(kb_written && kb_written.written
    ? `[kb] experience written: ${kb_written.slug} (speedup ${finalPrimary.toFixed(2)}, direction ` +
      `${winnerDirection || 'unlabeled'})`
    : kb_written && kb_written.reason === 'duplicate_impl'
      // Not a failure — the expected outcome of adopting a warm start and not beating it.
      ? `[kb] experience already known — counted as reproduction #${kb_written.reproductions || '?'} of ` +
        `${kb_written.reproduced || kb_written.dir}`
      : `[kb] experience not written: ${kb_written ? kb_written.reason : 'writer returned nothing'}`);
}

// finalPrimary is the total vs the pristine baseline; when a warm-start patch was adopted, split out
// the delta ABOVE it so a KB-derived gain is never reported as this run's own work.
//
// With no round committed after the adopt, the workspace still holds exactly the adopted code, and
// the two numbers are two measurements of the SAME thing taken at different times — the ratio is
// bench noise (~2% on this box; `_gemm_a16_w16_kernel` reported 0.98 and read as a regression).
// Report 1.0 and say why, rather than dressing the noise up as a delta.
const noRoundsAfterAdopt = !!(warm_start.adopted && roundsCommitted === 0);
const incrementalSpeedup = warm_start.adopted && warm_start.adopted_speedup
  ? (noRoundsAfterAdopt ? 1.0
     : Number.isFinite(finalPrimary) ? finalPrimary / warm_start.adopted_speedup : null)
  : finalPrimary;

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
  // How the search ended, and how far up the ladder it got. Without these a caller sees only that a
  // run finished, and "source search was exhausted and assembly found nothing either" is a very
  // different result from "the budget ran out in phase1" even when both report the same speedup.
  stopped_by: stopReason,
  final_phase: searchPhase,
  cycles: cycle,
  phases_entered: phaseLog,
  report_path: report ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`,
  final_patch: report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
  // null when the step was off, or the run earned nothing worth a card.
  // Which condition actually ended the loop. RECORDED, not inferred: on the previous campaign this
  // existed only in the driver's return value, which the caller sees after ALL its kernels finish —
  // so a parent-process death erased it for every kernel already done, and the realised window had
  // to be reconstructed from queue timestamps. It is the whole point of an enforced window that you
  // can tell a run that used its budget from one that gave up.
  stopped_by: deadlineHit ? 'deadline'
    : (dispatched >= BUDGET) ? 'budget'
    : (noImprove >= MAX_NO_IMPROVE) ? 'no_improve'
    : 'tech_lead_stop',
  deadline_hit: deadlineHit,
  forced_replans: forcedReplans,
  // What the plan cited and whether it carried its round. Returned ALWAYS, including when nothing was
  // cited (an empty array is the finding: the KB was read and nothing in it was worth acting on).
  learned_citations: citations,
  // Rounds where the planner overran the KB budget. A KB that binds every round is one that has taken
  // over planning, which is the regression this budget exists to catch.
  kb_cap_bound_rounds: kbCapBound,
  // The monoculture canary, and it costs no GPU time: how many DISTINCT (specialty, focus_files)
  // directions this run explored vs how many it issued. A KB that helps raises the verified speedup;
  // a KB that cages lowers this without raising that. Reported on KB-less runs too — those are the
  // baseline it has to be read against.
  direction_entropy: (() => {
    const seen = new Set(); let n = 0;
    for (const r of history.rounds) for (const d of (r.directions || [])) {
      n++; seen.add(`${d.specialty}|${(d.focus_files || []).slice().sort().join(',')}`);
    }
    return n ? { distinct: seen.size, issued: n, ratio: +(seen.size / n).toFixed(3) } : null;
  })(),
  learned_card: learned_card && learned_card.card_path
    ? { action: learned_card.action || '', card_path: learned_card.card_path, key: learned_card.key || '' }
    : null,
  // Warm-start (local experience KB) outcome. adopted=false + read_reason on a cold start.
  warm_start: {
    adopted: warm_start.adopted,
    read_reason: warm_start.read_reason,
    slug: warm_start.slug || '',
    // How the store was reached and what it filtered out: tells a genuine cold start from a
    // name/language mismatch that only looks like an empty KB.
    match_tier: warm_start.match_tier || '',
    filtered: warm_start.filtered || null,
    adopted_speedup: warm_start.adopted ? warm_start.adopted_speedup : null,
    total_speedup: Number.isFinite(finalPrimary) ? finalPrimary : null,
    incremental_speedup: incrementalSpeedup,
    // true = the run adopted a stored patch and never improved on it; incremental is 1.0 by
    // definition, not by measurement.
    no_rounds_after_adopt: noRoundsAfterAdopt,
    rounds_committed: roundsCommitted,
    incremental_improved: !!(warm_start.adopted && Number.isFinite(incrementalSpeedup) && incrementalSpeedup > 1 + MIN_IMPROVE),
    returned_after_read_kb: !!warm_start.returned_after_read_kb,
    candidates: warm_start.candidates,
  },
  kb_written: kb_written && kb_written.written ? { slug: kb_written.slug, dir: kb_written.dir } : null,
};
