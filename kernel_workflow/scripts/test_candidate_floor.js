#!/usr/bin/env node
// Regression guard for CANDIDATE_FLOOR (no GPU, no model needed).
//
// The round loop had `1.0` hard-coded in three places: the Optimize prompt that tells an engineer when
// to save best_patch.diff, the `trustworthyBelowBaseline` harvest shortcut, and the `verified` filter.
// Right for an ordinary optimization, but it makes a transcription (plain Triton -> Gluon/TileLang/HIP)
// inexpressible: it lands below the comparator by construction, so at 1.0 no patch is saved, no verify
// runs, and the loop stalls on a run that is working as designed. The floor is now a caller knob like
// `budget` / `min_improve` / `deep_cost`. Under test: the default is 1.0 (so existing runs are
// unchanged), all three sites read it, and the COMMIT gate is untouched.
//
// Run:  node kernel_workflow/scripts/test_candidate_floor.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
// kernel_workflow.js is a thin mode dispatcher; the round loop this knob governs lives in the
// single-language WORKER, kernel_lane.js.
const src = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'kernel_lane.js'), 'utf8');

let failures = 0;
const ok = (cond, msg, detail) => {
  if (!cond) { console.error('  FAIL:', msg, detail != null ? `-- ${detail}` : ''); failures++; }
  else console.log('  ok:', msg);
};

// Pull the real expressions out of the source rather than restating them here. Each `grab` must match
// the WHOLE statement: a `[^\n]*` against a multi-line const silently truncates it to its first line
// (ASI keeps the result valid JS), leaving the rest with no coverage at all.
function knobs(A) {
  const grab = (re, name) => {
    const m = src.match(re);
    if (!m) throw new Error(`could not extract ${name} from kernel_lane.js`);
    return m[0];
  };
  const body = [
    grab(/const BUDGET = parseInt\(A\.budget[^\n]*\n/, 'BUDGET'),
    grab(/const MIN_IMPROVE = \(\(\) => \{[\s\S]*?\n\}\)\(\);\n/, 'MIN_IMPROVE'),
    grab(/const CANDIDATE_FLOOR = \(\(\) => \{[\s\S]*?\n\}\)\(\);\n/, 'CANDIDATE_FLOOR'),
    grab(/const PROGRESS_DELTA = \(\(\) => \{[\s\S]*?\n\}\)\(\);\n/, 'PROGRESS_DELTA'),
    grab(/const DEEP_COST = \(\(\) => \{[\s\S]*?\n\}\)\(\);\n/, 'DEEP_COST'),
    grab(/const MAX_NO_IMPROVE = Math\.max\(1, parseInt\([^\n]*\n/, 'MAX_NO_IMPROVE'),
  ].join('\n');
  return new Function('A',
    `${body}\nreturn { BUDGET, MIN_IMPROVE, CANDIDATE_FLOOR, PROGRESS_DELTA, DEEP_COST, MAX_NO_IMPROVE };`)(A);
}

// The prompt renders CANDIDATE_FLOOR_TXT, not the number.
function txt(A) {
  const grab = (re, name) => {
    const m = src.match(re);
    if (!m) throw new Error(`could not extract ${name} from kernel_lane.js`);
    return m[0];
  };
  return new Function('A',
    grab(/const CANDIDATE_FLOOR = \(\(\) => \{[\s\S]*?\n\}\)\(\);\n/, 'CANDIDATE_FLOOR') +
    grab(/const CANDIDATE_FLOOR_TXT = [\s\S]*?;\n/, 'CANDIDATE_FLOOR_TXT') +
    'return CANDIDATE_FLOOR_TXT;')(A);
}

console.log('\n# the default is the historical behavior');
const o = knobs({});
ok(o.CANDIDATE_FLOOR === 1.0, 'CANDIDATE_FLOOR defaults to 1.0', o.CANDIDATE_FLOOR);
ok(o.BUDGET === 6, 'BUDGET default 6 is untouched', o.BUDGET);
ok(o.MAX_NO_IMPROVE === 2, 'MAX_NO_IMPROVE default 2 is untouched', o.MAX_NO_IMPROVE);
ok(o.MIN_IMPROVE === 0.02, 'MIN_IMPROVE default 0.02 is untouched', o.MIN_IMPROVE);
ok(o.DEEP_COST === 2, 'DEEP_COST default 2 is untouched', o.DEEP_COST);
ok(o.PROGRESS_DELTA === o.MIN_IMPROVE,
   'PROGRESS_DELTA defaults to MIN_IMPROVE, i.e. the historical progress test', o.PROGRESS_DELTA);

console.log('\n# it is a caller knob, like the ones next to it');
ok(knobs({ candidate_floor: 0.5 }).CANDIDATE_FLOOR === 0.5, 'an explicit floor wins');
ok(knobs({ candidate_floor: '0.5' }).CANDIDATE_FLOOR === 0.5, 'and parses from a CLI string');
ok(knobs({ candidate_floor: 'nonsense' }).CANDIDATE_FLOOR === 1.0,
   'garbage falls back to 1.0 rather than to NaN');
ok(knobs({ candidate_floor: 0 }).CANDIDATE_FLOOR === 1.0,
   'and so does a floor of 0, which would admit a broken kernel');
ok(knobs({ candidate_floor: -1 }).CANDIDATE_FLOOR === 1.0, 'and a negative floor');
ok(knobs({ progress_delta: -0.05 }).PROGRESS_DELTA === -0.05, 'an explicit progress_delta wins');
ok(knobs({ progress_delta: 'nonsense' }).PROGRESS_DELTA === o.MIN_IMPROVE,
   'and garbage falls back to MIN_IMPROVE');

console.log('\n# nothing was left behind on a hard-coded 1.0');
// A missed site means a lowered floor applies to only part of the pipeline -- e.g. the engineer saves
// a 0.7x patch that the `verified` filter then drops, which looks exactly like the stall it removes.
ok(/Save best_patch\.diff[^\n]*geomean>\$\{CANDIDATE_FLOOR_TXT\}/.test(src),
   'the Optimize prompt tells the engineer to save above CANDIDATE_FLOOR');
ok(/trustworthyBelowBaseline = eng && eng\.status !== 'failed' && !\(primSpeedup\(eng\) > CANDIDATE_FLOOR\)/.test(src),
   'the harvest shortcut suppresses against CANDIDATE_FLOOR');
// This clause used to be pinned as ONE line, `says(...correctness, 'pass') && primSpeedup(r.ver) >
// CANDIDATE_FLOOR`. The filter has since grown a metric/oracle/policy/twin arm and spans a block, so
// that pin stopped matching -- and nobody saw it, because this file was runnable only under node and
// the box has none. It is now hosted by run_js_tests.py and mutated by test_js_suite.py. Pinned as
// three clauses: the correctness gate is in the filter, the admission compares against the knob, and
// no site admits against a hard-coded literal.
ok(/if \(!\(r\.ver && r\.ver\.policy_pass === true &&\n\s*says\(r\.ver\.status, 'verified'\) && says\(r\.ver\.correctness, 'pass'\)\)\) return false;/.test(src) &&
   /return primSpeedup\(r\.ver\) > CANDIDATE_FLOOR;/.test(src) &&
   !/primSpeedup\(r\.ver\) > [0-9]/.test(src),
   'the verified filter admits against CANDIDATE_FLOOR, behind the correctness gate');

// `${1.0}` stringifies to "1", which would silently reword a prompt that has always said
// "geomean>1.0". The default run must render byte-identically.
ok(txt({}) === '1.0', 'the default renders as "1.0" in the prompt, exactly as before', txt({}));
ok(txt({ candidate_floor: 0.5 }) === '0.5', 'and a lowered floor renders faithfully', txt({ candidate_floor: 0.5 }));
ok(txt({ candidate_floor: 0.55 }) === '0.55', 'including a non-round one', txt({ candidate_floor: 0.55 }));

console.log('\n# the commit gate is NOT loosened by any of this');
ok(/const legacyImproved = cand\.geomean > cumulative \* \(1 \+ MIN_IMPROVE\);/.test(src),
   'banking still requires beating cumulative by MIN_IMPROVE');
// The suite threshold above is the FALLBACK and also one half of a union: a per-route band table
// (derived from the baseline repeats by default) can accept a candidate the suite test refuses, and
// the suite test can accept one no single band clears. Pinned here because this file's subject is
// "the commit gate is not loosened", and the thing that would loosen it is the REGRESSION VETO going
// missing -- not either test accepting.
ok(/let improved = legacyImproved;/.test(src) && /if \(winner\) \{/.test(src),
   'the suite threshold is the value the commit decision STARTS at, and the gate runs on every ' +
   'round that produced a candidate -- no longer only on rounds that happened to carry a table');
// The union is gone; the commit decision is now the gate's own conjunction (suite improved AND one
// route past its bar), with a catastrophic-regression fence outside it. What this file is about --
// "the candidate floor cannot loosen the commit gate" -- is unchanged and if anything easier to
// state: CANDIDATE_FLOOR appears in neither condition.
ok(/improved: rv\.accepted/.test(src) && !/rv\.accepted \|\| suiteSaysYes/.test(src),
   'the commit decision is the gate verdict alone; there is no second arm to loosen');
ok(/if \(catastrophic\.length\) \{/.test(src) && /CATASTROPHIC_REGRESSION = 0\.10;/.test(src),
   'and the one refusal that survives a good average is the catastrophic fence, not the floor');
// The veto now also decides SELECTION, not just acceptance of the top-ranked candidate: the
// selector walks candidates in geomean order and takes the first this same predicate passes. That
// direction can only bank a round that would otherwise have banked nothing, so it cannot loosen the
// gate -- but it does mean the veto is applied to every candidate rather than to one, which is a
// strictly wider application of the safety property this file is about.
ok(/const passIdx = candidates\.findIndex\(c => judgeCandidate\(c\)\.improved\);/.test(src),
   'selection uses the SAME predicate as acceptance, so a candidate cannot be selected by one rule ' +
   'and judged by another');
ok(/const suiteProgress = !!\(winner && bestSeen > 0 && winner\.geomean > bestSeen \* \(1 \+ PROGRESS_DELTA\)\)/.test(src),
   'the suite progress signal reads PROGRESS_DELTA against bestSeen, guarded on bestSeen > 0');
// A route win is progress too, because the suite geomean divides a single-route win by the route
// count and a round scored as a stall ends the WAVE at MAX_NO_IMPROVE. That is a STOPPING rule, so it
// errs toward continuing: a false "advancing" costs one authorised round, a false "stalled" costs all
// the remaining ones.
ok(/const madeProgress = suiteProgress \|\| routeProgress;/.test(src),
   'progress is the union, so a verified single-route win is not counted toward stopping the search');
// The other half of the stall counter, and the half that decides whether a lane can ever stop.
// `improved` is settled BEFORE the commit is attempted, so resetting on it let a winner that
// never LANDED clear the counter every round while `cumulative` stood still -- and the next
// round's winner then cleared the same unchanged threshold, so MAX_NO_IMPROVE could not fire
// and the whole budget went into re-deriving one unbankable patch.
ok(/if \(madeProgress \|\| committedThisRound\) \{ noImprove = 0; \} else \{ noImprove\+\+; \}/.test(src),
   'the stall counter resets on a COMMIT, not on an `improved` that may never have landed');
ok(!/cumulative \* \(1 \+ CANDIDATE_FLOOR\)|CANDIDATE_FLOOR\)\s*;?\s*\/\/ commit/.test(src),
   'and the floor is nowhere near the commit decision');

// --- behavioral replay -------------------------------------------------------
// Mirrors the real round loop: candidate filter, commit gate, progress signal, DEEP_COST, stall counter.
function replay({ budget, deepCost, maxNoImprove, floor, minImprove, progressDelta }, traj) {
  let dispatched = 0, round = 0, noImprove = 0, cumulative = 1.0, bestSeen = 0, commits = 0;
  const log = [];
  while (dispatched < budget && noImprove < maxNoImprove && round < traj.length) {
    round++; dispatched += deepCost;
    const raw = traj[round - 1];
    const winner = raw > floor ? raw : null;
    const improved = winner !== null && winner > cumulative * (1 + minImprove);
    const progress = winner !== null && bestSeen > 0 && winner > bestSeen * (1 + progressDelta);
    // This model's commit never fails, so `committed` tracks `improved` here. Named anyway,
    // because the lane resets the stall counter on the COMMIT and a model that reads `improved`
    // instead is one edit away from re-authorising the bug that change removed.
    const committed = improved;
    if (committed) { cumulative = winner; commits++; }
    if (winner !== null && winner > bestSeen) bestSeen = winner;
    noImprove = (progress || committed) ? 0 : noImprove + 1;
    log.push([round, noImprove, +cumulative.toFixed(4)]);
  }
  return { rounds: round, commits, final: +cumulative.toFixed(4), log };
}
// main's loop, before the knob existed: reset ONLY on a commit, floor pinned at 1.0.
function legacy({ budget, deepCost, maxNoImprove }, traj) {
  let dispatched = 0, round = 0, noImprove = 0, cumulative = 1.0, commits = 0;
  const log = [];
  while (dispatched < budget && noImprove < maxNoImprove && round < traj.length) {
    round++; dispatched += deepCost;
    const winner = traj[round - 1] > 1.0 ? traj[round - 1] : null;
    const improved = winner !== null && winner > cumulative * 1.02;
    if (improved) { cumulative = winner; commits++; noImprove = 0; } else { noImprove++; }
    log.push([round, noImprove, +cumulative.toFixed(4)]);
  }
  return { rounds: round, commits, final: +cumulative.toFixed(4), log };
}

console.log('\n# at the defaults the loop is EXHAUSTIVELY equivalent to the pre-knob one');
// The `bestSeen > 0` guard is what makes this hold: without it a first-round winner in (1.0, 1.02]
// makes `suiteProgress` trivially true (bestSeen is 0), resetting a stall counter main would advance.
// This replay models the SUITE arm only; `routeProgress` needs a band table, and the equivalence
// being pinned here is with the pre-knob loop, which had none.
// Values below straddle 1.0 and the 1.02 commit margin, and include no-candidate rounds.
{
  const VALS = [0.0, 0.5, 0.99, 1.0, 1.005, 1.01, 1.02, 1.021, 1.05, 1.1, 1.3];
  const cfg = { budget: o.BUDGET, deepCost: 1, maxNoImprove: o.MAX_NO_IMPROVE,
                floor: o.CANDIDATE_FLOOR, minImprove: o.MIN_IMPROVE, progressDelta: o.PROGRESS_DELTA };
  let n = 0, diverged = 0, example = null;
  for (const a of VALS) for (const b of VALS) for (const c of VALS) for (const d of VALS) {
    const t = [a, b, c, d]; n++;
    if (JSON.stringify(replay(cfg, t)) !== JSON.stringify(legacy(cfg, t))) {
      diverged++; if (!example) example = t;
    }
  }
  ok(diverged === 0, `all ${n} default-knob trajectories replay identically to the pre-knob loop`,
     example ? `first divergence ${JSON.stringify(example)}` : null);
}

console.log('\n# what each knob buys a port, separately');
// Each row is a real shape: fast recovery, a slow climb, the measured pa_decode run (five consecutive
// non-improving rounds), and a climb so slow no static stall counter could have been pre-guessed.
const TRAJ = {
  'fast recovery': [0.75, 1.02, 1.05, 1.01, 1.06, 1.08, 1.09, 1.10, 1.11, 1.12],
  'slow climb':    [0.60, 0.70, 0.78, 0.86, 0.93, 0.98, 1.03, 1.07, 1.10, 1.14],
  'pa_decode':     [1.04, 0.99, 1.01, 1.00, 1.02, 1.03, 1.05, 1.19, 1.22, 1.25],
  'very slow':     [0.55, 0.62, 0.69, 0.75, 0.82, 0.88, 0.94, 0.99, 1.04, 1.09],
};
const PORT = { budget: 20, deepCost: o.DEEP_COST, maxNoImprove: 6, floor: 0.5,
               minImprove: o.MIN_IMPROVE, progressDelta: -0.05 };
// Same run with the floor lowered but progress_delta left at its default. Measuring progress against
// bestSeen already rescues a MONOTONE climb at any delta -- each round beats the previous best. The
// NEGATIVE delta buys the other shape: a trajectory that oscillates, where a round gives ground while
// exploring. pa_decode is that shape, and it is the one that still dies here.
const FLOOR_ONLY = { ...PORT, progressDelta: o.PROGRESS_DELTA };
for (const [name, t] of Object.entries(TRAJ)) {
  const full = replay(PORT, t);
  const half = replay(FLOOR_ONLY, t);
  console.log(`    ${name.padEnd(15)} floor + negative delta: ${full.rounds} round(s) ${full.final}x` +
              `   default delta: ${half.rounds} round(s) ${half.final}x`);
  ok(full.rounds === t.length, `${name}: the port runs to the end of its budget`, full.rounds);
  ok(full.final > 1.0, `${name}: and closes above the comparator`, full.final);
}
ok(replay(FLOOR_ONLY, TRAJ['pa_decode']).rounds < TRAJ['pa_decode'].length,
   'an OSCILLATING port needs the negative delta too -- pa_decode ends early without it');
// And neither knob can be replaced by simply raising max_no_improve: that counter only resets on a
// COMMIT, so it would have to pre-guess how many rounds the climb spends under the comparator.
{
  const noBestSeen = (mni) => legacy({ budget: 20, deepCost: o.DEEP_COST, maxNoImprove: mni },
                                     TRAJ['very slow']);
  ok(noBestSeen(6).rounds < TRAJ['very slow'].length && noBestSeen(8).rounds < TRAJ['very slow'].length,
     'a bigger max_no_improve is NOT a substitute -- it resets only on a commit',
     `mni=6 -> ${noBestSeen(6).rounds} rounds, mni=8 -> ${noBestSeen(8).rounds} rounds`);
}

console.log('\n# a sub-baseline candidate is TRACKED but never BANKED');
// The whole trajectory stays under the comparator: the floor admits every round, the gate refuses all.
const doomed = replay(PORT, [0.60, 0.72, 0.81, 0.88, 0.95, 0.99]);
ok(doomed.commits === 0, 'nothing sub-baseline is ever committed', doomed.commits);
ok(doomed.final === 1.0, 'and the cumulative best stays at the baseline', doomed.final);

console.log('\n# the Optimize prompt is built inline, so it must append the expert-skills block itself');
// Every other consumer role gets the block from roleAgent(). This one prompt is assembled by hand, so
// dropping the append is invisible: the run succeeds and the engineers simply never see the skills
// index. Listing the roles in EXPERT_SKILL_ROLES is the other half -- without it the block is always ''.
{
  const declared = (src.match(/const EXPERT_SKILL_ROLES = new Set\(\[([^\]]*)\]\)/) || [, ''])[1];
  ok(/\+\s*(?:\/\/[^\n]*\n\s*)*expertSkillsBlock\(isDeep \? 'deep_engineer' : 'engineer'\)/.test(src),
     'the inline Optimize prompt appends expertSkillsBlock (additive)');
  for (const r of ['engineer', 'deep_engineer']) {
    ok(new RegExp(`'${r}'`).test(declared), `EXPERT_SKILL_ROLES includes '${r}' (the inline site is live)`);
  }
}

console.log(failures
  ? `\nFAIL: ${failures} check(s) failed.`
  : '\nPASS: CANDIDATE_FLOOR defaults to 1.0; a lowered floor tracks but never banks.');
process.exit(failures ? 1 : 0);
