#!/usr/bin/env node
// Deterministic regression guard for the kernel lane's admission gates (no GPU/model/network).
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const src = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'kernel_lane.js'), 'utf8');
// Finding (65): the lane produces a validation verdict and kernel_workflow.js consumes it to pick the
// bake-off winner. A guard that reads only the producer cannot see the consumer drop it on the floor.
const wfSrc = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'kernel_workflow.js'), 'utf8');
let failures = 0;
const ok = (cond, msg, detail) => {
  if (!cond) { console.error('  FAIL:', msg, detail == null ? '' : `-- ${detail}`); failures++; }
  else console.log('  ok:', msg);
};
const grab = (re, name) => {
  const m = src.match(re);
  if (!m) throw new Error(`could not extract ${name} from kernel_lane.js`);
  return m[0];
};
// The gate's thresholds, read out of the lane rather than restated here. A guard that hardcoded 2%
// would keep passing after someone changed the lane to 5%, which is the one thing a threshold guard
// must not do -- and this file's whole subject is thresholds.
const grabConst = (name) => {
  const m = src.match(new RegExp(`^const ${name} = ([0-9.]+);`, 'm'));
  if (!m) throw new Error(`could not extract ${name} from kernel_lane.js`);
  return parseFloat(m[1]);
};
const MIN_ROUTE_WIN = grabConst('MIN_ROUTE_WIN');
const CATASTROPHIC_REGRESSION = grabConst('CATASTROPHIC_REGRESSION');
// `routeGate` closes over both constants, so an extraction that did not supply them would throw at
// call time -- and a guard that throws reports a lane defect where there is only a harness gap.
const makeGate = () => new Function(
  `const MIN_ROUTE_WIN = ${MIN_ROUTE_WIN};
   const CATASTROPHIC_REGRESSION = ${CATASTROPHIC_REGRESSION};
   ${grab(/const routeGate = \(candPerCase, incPerCase, bands, opts\) => \{[\s\S]*?\n\};\n/, 'routeGate')}
   return routeGate;`)();

console.log('\n# the correctness gate and the primary-metric selector, executed');
{
  // Finding (62). Both of these were lexical-only until now, and both are gates:
  // `says` decides whether a candidate counts as correct, `primSpeedup` decides
  // what number the round gates on and what the ledger records as the winner.
  const gateBlock = grab(/const saysContradicted = \(v\) => \{[\s\S]*?\nconst says = [^;]*;\n/, 'says');
  const says = new Function(`${gateBlock}\nreturn {says, saysContradicted};`)();

  // The gate opens on genuine successes, including the phrasing it exists for.
  for (const s of ['PASS', 'pass', ' passed ', 'PASS - 15/15 draws', 'passes all 11 cases',
                   'PASS_WITH_WARNINGS']) {
    ok(says.says(s, 'pass') === true, `the correctness gate opens on ${JSON.stringify(s)}`);
  }
  // ...and on a PASS that quotes its MEASUREMENT vocabulary, which is what a numerical
  // correctness report normally does. `error\w*`, `nan` and `inf` were in the veto list and
  // vetoed every one of these: matching the presence of a word cannot see the negation in
  // "no NaN", and a max-error figure is evidence the check RAN, not that it failed. This is
  // the veto firing in the only direction it must never fire in -- a correct, faster
  // candidate discarded and written into the ledger as a correctness failure.
  for (const s of ['PASS (max error 3.1e-05, within rtol=1e-3)', 'pass, no NaN or Inf in output',
                   'PASS - allclose within tolerance, no errors',
                   'passed - relative error 1e-7 on all cases']) {
    ok(says.says(s, 'pass') === true,
      `the correctness gate opens on a PASS that quotes measurement vocabulary: ${JSON.stringify(s)}`);
  }
  // ...and stays shut on failures, INCLUDING the ones that used to open it.
  for (const s of ['FAIL', 'failed', 'did not pass', 'partially passes', '', null, undefined,
                   'passes 10/11 cases', 'pass rate 10/11', 'PASS except decode_m2',
                   'pass with 1 mismatch', 'passed but 2 cases are incorrect']) {
    ok(says.says(s, 'pass') === false, `the correctness gate stays shut on ${JSON.stringify(s)}`);
  }
  ok(says.says('verified', 'verified') === true &&
     says.says('verified: FAILED correctness', 'verified') === false &&
     says.says('not verified', 'verified') === false,
    'the status gate stays shut on a report that verifies its own failure');
  // The veto is what does the work -- without it the prefix alone says true.
  ok('passes 10/11 cases'.trim().toLowerCase().startsWith('pass') &&
     says.saysContradicted('passes 10/11 cases') === true &&
     says.saysContradicted('PASS - 15/15 draws') === false,
    'an n-of-m fraction is a contradiction only when it is not n/n');

  const metricBlock = grab(/const primWeighted = \(o\) => \{[\s\S]*?\n\};\n/, 'primWeighted') +
    grab(/const primMetricReason = \(o\) =>[\s\S]*?: null\);\n/, 'primMetricReason') +
    grab(/const primSpeedup = \(o\) => \{[\s\S]*?\n\};\n/, 'primSpeedup');
  const mk = (hasWorkload) => new Function('HAS_WORKLOAD',
    `${metricBlock}\nreturn {primSpeedup, primMetricReason};`)(hasWorkload);
  const off = mk(false), on = mk(true);
  const both = { verified_geomean: 1.1, verified_weighted: 1.3 };
  const onlyGeo = { verified_geomean: 1.1 };
  ok(off.primSpeedup(both) === 1.1 && off.primMetricReason(onlyGeo) === null,
    'an unweighted run is unchanged: the geomean is the primary metric and nothing is refused');
  ok(on.primSpeedup(both) === 1.3, 'a workload-aligned run gates on the weighted number');
  // The defect: the same call used to return 1.1 and say nothing.
  ok(/^metric:weighted_speedup_missing/.test(String(on.primMetricReason(onlyGeo))) &&
     /^metric:weighted_speedup_missing/.test(String(on.primMetricReason(
       { verified_geomean: 1.1, verified_weighted: null }))),
    'a workload-aligned run refuses, by name, a report with no weighted number');
  ok(on.primMetricReason({ verified_geomean: 1.1, speedup_weighted: 1.4 }) === null &&
     on.primSpeedup({ verified_geomean: 1.1, speedup_weighted: 1.4 }) === 1.4,
    'the legacy speedup_weighted field still satisfies the requirement');
  // (60): a metric fault must not be reported as, or silently share a path with,
  // a slow kernel. Both authoritative sites route it through the named check.
  ok((src.match(/primMetricReason\(/g) || []).length === 2 &&
     /primMetricReason\(r\.ver\)/.test(src) && /primMetricReason\(integMetric\)/.test(src),
    'the authoritative sites -- the candidate filter and integrate -- route an '
    + 'ambiguous primary metric through the named refusal');
}

console.log('\n# the oracle pin, executed');
{
  // Finding (67). Candidates carry `source_hash` provenance; the oracle -- the denominator of
  // every speedup, the correctness reference, and the yardstick the persisted archive is scored
  // on -- carried none, though four role prompts say "NEVER modify" it and the golden reaches
  // each workspace as an absolute symlink that writes through to the shared original.
  const driftBlock = grab(/const ORACLE_DEGENERATE = new Set\(\[[\s\S]*?\n\]\);\n/, 'ORACLE_DEGENERATE') +
    grab(/const oracleDrift = \(o\) => \{[\s\S]*?\n\};\n/, 'oracleDrift');
  const mkDrift = (pinned) => new Function('ORACLE_DIGEST',
    `${driftBlock}\nreturn oracleDrift;`)(pinned);
  const PIN = 'a'.repeat(64), OTHER = 'b'.repeat(64);
  const pinned = mkDrift(PIN), unpinned = mkDrift(null);

  ok(pinned({ oracle_digest: PIN }) === null && pinned({ oracle_digest: ` ${PIN} ` }) === null,
    'a report measuring against the pinned oracle passes, whitespace and all');
  ok(/^oracle:digest_missing/.test(String(pinned({ verified_geomean: 1.4 }))) &&
     /^oracle:digest_missing/.test(String(pinned({ oracle_digest: '' }))) &&
     /^oracle:digest_missing/.test(String(pinned(null))),
    'a report with no digest is refused by name, not read as a result');
  // The drift case is the one that must NOT be a per-candidate skip: results already admitted
  // to a persisted archive were scored against the denominator that just moved.
  let threw = null;
  try { pinned({ oracle_digest: OTHER }); } catch (e) { threw = e; }
  ok(threw && /oracle:digest_drift/.test(threw.message) &&
     /already admitted to the archive/.test(threw.message) &&
     /Refusing to continue or to persist/.test(threw.message),
    'a moved oracle fails the run closed and says why, rather than dropping one candidate');
  ok(unpinned({ oracle_digest: OTHER }) === null && unpinned({}) === null,
    'an unpinned run does not re-litigate every candidate -- it is named once, at setup');

  // The pin must reach all three gates, and the unpinned run must not be publishable.
  ok(/primMetricReason\(r\.ver\) \|\| oracleDrift\(r\.ver\)/.test(src) &&
     /const oracleFinal = validation \? oracleDrift\(validation\) : null;/.test(src),
    'the candidate filter and the final verdict both consult the pin');
  ok(/oracle_digest: \{ type: 'string' \}/.test(src) &&
     (src.match(/oracle_digest: \{ type: 'string' \}/g) || []).length === 3,
    'setup, verify and validate all declare the digest -- a field only one side reports is not a check');
  ok(/if \(!oraclePinned\) \{/.test(src) && /oracle:unpinned\(setup recorded no oracle digest/.test(src),
    'a run that never pinned the oracle is flagged, not quietly trusted');
}

console.log('\n# the final validation verdict, executed');
{
  // Finding (65). Every field of the run's only independent arbitration was decorative:
  // `correctness` was in the schema and read by nothing, `validation_status` was passed
  // through with no gate, `timing_basis` was required by director.md and absent from both
  // the schema and every consumer, and a null verdict published the TechLead's self-report.
  const gateBlock = grab(/const saysContradicted = \(v\) => \{[\s\S]*?\nconst says = [^;]*;\n/, 'says');
  const verdictBlock = grab(/const validationVerdict = \(v, oraclePinned\) => \{[\s\S]*?\n\};\n/, 'validationVerdict');
  const { validationVerdict: raw } =
    new Function(`${gateBlock}\n${verdictBlock}\nreturn {validationVerdict};`)();
  // (67) added the oracle pin ahead of every other clause; these cases are about the clauses
  // behind it, so they run pinned. The unpinned case is asserted separately just below.
  const vv = (v) => raw(v, true);

  const clean = { director_verified_speedup_geomean: 1.3, validation_status: 'accepted',
                  correctness: 'PASS - 11/11', timing_basis: 'device' };
  ok(vv(clean).trust === 'verified' && vv(clean).reason === null && vv(clean).basis === 'device',
    'a clean, accepted, device-timed, correct verdict is verified and needs no reason');

  // The four refusals, and -- (60) -- the requirement that they be four DIFFERENT refusals.
  const none = vv(null);
  const wrong = vv({ ...clean, correctness: 'FAIL: decode_m2 mismatch' });
  const notAccepted = vv({ ...clean, validation_status: 'flagged' });
  const noBasis = vv({ ...clean, timing_basis: undefined });
  ok(none.trust === 'unverified' && /^provenance:director_validate_returned_nothing/.test(none.reason),
    'a dead or hung arbiter is named as absent arbitration, not as a result');
  ok(wrong.trust === 'refused' && /^correctness:/.test(wrong.reason),
    'a speedup over a kernel that computes the wrong answer is refused by name');
  ok(notAccepted.trust === 'flagged' && /^validation:director_status_"flagged"/.test(notAccepted.reason),
    'a status the arbiter did not accept is flagged by name');
  ok(noBasis.trust === 'flagged' && noBasis.basis === 'unknown' && /^timing:basis_"unknown"/.test(noBasis.reason),
    'a missing timing basis defaults to unknown and is flagged -- absence is not evidence of priming');
  ok(vv({ ...clean, timing_basis: 'host_bound' }).trust === 'flagged' &&
     vv({ ...clean, timing_basis: 'unprimed' }).trust === 'flagged',
    'a host-bound or unprimed ratio is not a device-time speedup');
  // (67): a flawless arbiter over an unknown denominator is still an unknown result, so the pin
  // is checked ahead of every other clause and cannot be argued past by a clean verdict.
  const unpinned = raw(clean, false);
  ok(unpinned.trust === 'flagged' && /^oracle:unpinned/.test(unpinned.reason),
    'an unpinned oracle flags even an otherwise perfect verdict');
  ok(new Set([none.reason, wrong.reason, notAccepted.reason, noBasis.reason, unpinned.reason]).size === 5,
    'the five refusals are five distinguishable strings, not one');
  // The gate must not be so eager it refuses honest arbitration.
  ok(vv({ ...clean, validation_status: 'accept' }).trust === 'verified' &&
     vv({ ...clean, validation_status: 'validated' }).trust === 'verified' &&
     vv({ ...clean, correctness: '' }).trust === 'verified' &&
     vv({ ...clean, timing_basis: ' Device ' }).trust === 'verified',
    'accepted/validated, an unstated correctness, and a padded basis all still pass');
  // (62)'s veto reaches this gate too: a correctness string that contradicts its own word.
  ok(vv({ ...clean, correctness: 'passes 10/11 cases' }).trust === 'refused',
    'a partial pass is refused here as well, not read as a pass');

  // Structural: the producer must publish no number it refused, and the consumer must gate.
  ok(/const trusted = verdict\.trust !== 'unverified' && verdict\.trust !== 'refused';/.test(src) &&
     /const finalGeomean = trusted \? validation\.director_verified_speedup_geomean : null;/.test(src) &&
     !/const finalGeomean = validation \? validation\.director_verified_speedup_geomean : cumulative;/.test(src),
    'a refused verdict publishes no geomean -- the self-report is not resurrected as a verified number');
  ok(/validation_trust: verdict\.trust,/.test(src) && /timing_basis: verdict\.basis,/.test(src),
    'the lane hands the consumer its verdict and its timing basis, not just a free-text status');
  ok(/const rankable = \(c\) => c\.kind !== 'lane' \|\| c\.validation_trust === 'verified';/.test(wfSrc) &&
     /cands\.filter\(c => c\.speedup > 1\.0 && rankable\(c\)\)/.test(wfSrc) &&
     !/cands\.filter\(c => c\.speedup > 1\.0\)\.sort/.test(wfSrc),
    'the bake-off ranks only lanes whose own arbiter stands behind the number');
  ok(/cannot win: /.test(wfSrc),
    'a fast-but-unverified lane is excluded out loud, which is what makes it not "found nothing"');
  ok(/timing_basis: \{ type: 'string' \}/.test(src),
    'timing_basis is in the validate schema, so a Director that reports it is not dropped');
}

// Finding (68). The third mandatory planner gate had no orchestrator half at
// all, so unlike (61)'s helpers there was nothing here to execute. Executed
// from the start: every branch that can cost a direction, plus the mixed-list
// case, which is the one the prompt's rule is actually about.

console.log('\n# the post-build policy receipt gate, executed');
{
  // Finding (69). `policy_pass` is the boolean the whole "no rocBLAS" constraint
  // reduces to, and it was consumed at five sites as a bare self-report. This
  // block executes the gate rather than grepping for it, because (57): a regex
  // proves the function exists, never that it refuses anything.
  const policyBlock = grab(/const policyReject = \(rep, label\) => \{[\s\S]*?\n\};\n/, 'policyReject');
  const reject = new Function(`${policyBlock}\nreturn policyReject;`)();
  const good = { schema: 'candidate-policy-scan', passed: true, findings: 0, advisory: 2,
    inspected: 11, files: 9, elf: 2, unreadable: 0 };
  const rep = (s) => ({ policy_pass: true, policy_postbuild: s });

  ok(reject(rep(good)) === null, 'a pass backed by a post-build receipt that saw an ELF is admitted');
  ok(reject({ policy_pass: true }) !== null,
    'a bare policy_pass with no post-build summary is refused');
  ok(reject(rep({ ...good, elf: 0 })) !== null,
    'a post-build scan that inspected no binary is refused -- DT_NEEDED is only visible there');
  ok(reject(rep({ ...good, files: 0, elf: 0 })) !== null,
    'a scan that opened no file at all is refused, not counted as passing');
  // `inspected` counts the directory entry, so an empty tree reads 1 there and 0
  // in `files`. Gating on the wrong one of those two is a gate that passes on
  // nothing at all.
  ok(reject(rep({ ...good, inspected: 1, files: 0, elf: 0 })) !== null,
    'an empty tree, which reads inspected:1, is still refused -- the gate reads files');
  ok(reject(rep({ ...good, passed: false })) !== null,
    'a receipt whose own verdict contradicts policy_pass is refused');
  ok(reject(rep({ ...good, findings: 1 })) !== null,
    'passed:true beside nonzero findings is self-contradictory and refused');
  ok(reject(rep({ ...good, unreadable: 1 })) !== null,
    'an uninspectable artifact fails closed, matching verify_engineer.md 2b');
  ok(reject(rep({ ...good, schema: 'isa-signals-v1' })) !== null,
    'a summary from some other tool is refused rather than read as a policy receipt');
  ok(reject(rep({ ...good, advisory: 7 })) === null,
    'advisory findings do not fail the gate -- (53): a gate that cries wolf gets ignored');

  // (60)/(61): the honest failure must keep its own name. If this gate spoke about
  // reports that never claimed a pass, a real `policy_failed` would be renamed into
  // a paperwork complaint and the caller's own refusal would never be reached.
  ok(reject({ policy_pass: false }) === null,
    'an honest policy_failed report is left alone for the caller to refuse under its own name');
  ok(reject(null) === null && reject(undefined) === null,
    'an absent report is not this gate\'s refusal to make');

  const reasons = [reject({ policy_pass: true }), reject(rep({ ...good, elf: 0 })),
    reject(rep({ ...good, files: 0 })), reject(rep({ ...good, passed: false })),
    reject(rep({ ...good, findings: 1 })), reject(rep({ ...good, unreadable: 1 })),
    reject(rep({ ...good, schema: 'isa-signals-v1' }))];
  ok(new Set(reasons.map(r => String(r).split('(')[0])).size === reasons.length,
    'the seven policy refusals are seven distinguishable tokens', JSON.stringify(reasons));
  ok(reasons.every(r => /^policy:/.test(String(r))),
    'every one of them is namespaced, so a log line says which gate spoke');
  ok(reject(rep(good), 'integrate') === null && /integrate/.test(String(reject({ policy_pass: true }, 'integrate'))),
    'the label names which consumer asserted the unbacked pass');

  // Wiring: both consumers of policy_pass (63: enumerate the call sites of the
  // quantity, not of the helper).
  ok(/oracleDrift\(r\.ver\) \|\| policyReject\(r\.ver\)/.test(src),
    'the candidate admission path checks it beside the metric and oracle gates');
  ok(/policyReject\(integrate, 'integrate'\) \|\| twinReject\(integrate, 'integrate'\)/.test(src),
    'integrate discards a result whose policy claim has nothing behind it, and one '
    + 'whose hipified twin drifted -- a merge is the site where an edit is most '
    + 'likely to land in only one half of a hipify pair');
  ok(/policy_postbuild: POLICY_SUMMARY_SCHEMA/.test(src),
    'the summary is in the schemas, so an agent that returns it is not stripped');
  ok((src.match(/policy_postbuild: POLICY_SUMMARY_SCHEMA/g) || []).length === 2,
    'in both report schemas that carry a verdict, not just the one easiest to reach');

  // A gate may only refuse on a field the schema ASKED FOR. This one reads `files`, and the
  // schema declared only `inspected` -- so a receipt carrying every declared field, all healthy,
  // came back `policy:inspected_nothing(files=undefined)`: a refusal that names the AGENT for a
  // field nothing ever requested. The fixtures above could not see it, because they supply
  // `files` themselves, and that is the general trap -- EXECUTING a gate proves nothing about
  // the gate's reachability when the fixture is more generous than the contract the agent is
  // handed. So the schemas are evaluated too, and the fixture is checked against them.
  const schemaBlock = grab(/const obj = \(props, required\) => [^\n]*\n/, 'obj') +
    grab(/const perCase = \{[\s\S]*?\n\};\n/, 'perCase') +
    // VERIFY_SCHEMA references this, so evaluating the schema requires it. Listed rather than
    // inlined into the schema because that is what makes the omission LOUD: leaving it out makes
    // this block throw `controlPerCase is not defined` instead of quietly evaluating a schema with
    // one field missing, which is the difference between a caught mistake and a gate that reads a
    // contract the agent was never handed.
    grab(/const controlPerCase = \{[\s\S]*?\n\};\n/, 'controlPerCase') +
    grab(/const POLICY_SUMMARY_SCHEMA = obj\(\{[\s\S]*?\}, \[[^\]]*\]\);\n/, 'POLICY_SUMMARY_SCHEMA') +
    grab(/const HIP_TWIN_SCHEMA = obj\(\{[\s\S]*?\}, \[[^\]]*\]\);\n/, 'HIP_TWIN_SCHEMA') +
    grab(/const ISA_EVIDENCE_SCHEMA = obj\(\{[\s\S]*?\}, \[[^\]]*\]\);\n/, 'ISA_EVIDENCE_SCHEMA') +
    grab(/const VERIFY_SCHEMA = obj\(\{[\s\S]*?\}, \[[^\]]*\]\);\n/, 'VERIFY_SCHEMA');
  const schemas = new Function(`${schemaBlock}\nreturn { POLICY_SUMMARY_SCHEMA, VERIFY_SCHEMA };`)();
  const polSummary = schemas.POLICY_SUMMARY_SCHEMA;
  ok(polSummary.properties.files !== undefined && polSummary.required.includes('files'),
    'POLICY_SUMMARY_SCHEMA declares AND requires `files`, the count this gate refuses without');
  ok(Object.keys(good).every(k => polSummary.properties[k] !== undefined),
    'every field the passing fixture carries is one the schema asks the agent for',
    Object.keys(good).filter(k => polSummary.properties[k] === undefined).join(', '));
  // The same defect one level up: `policy_pass` is the boolean the admission filter reads, and
  // it sat in VERIFY_SCHEMA's `required` list with no property declared beside it -- a field the
  // agent is obliged to return and was never told about.
  ok(schemas.VERIFY_SCHEMA.properties.policy_pass !== undefined &&
     schemas.VERIFY_SCHEMA.required.includes('policy_pass'),
    'VERIFY_SCHEMA declares policy_pass rather than only listing it in `required`');
}

console.log('\n# the hipify twin-sync gate, executed');
{
  // Finding (87), wired. The tool has been correct and uncalled for several
  // rounds. Its three exit codes are the point: 0 lockstep, 1 drift, 2 nothing
  // checked -- and 2 is the one a boolean `twin_pass` destroys.
  const block = grab(/const TWIN_LANGUAGES = [\s\S]*?const twinReject = \(rep, label\) => \{[\s\S]*?\n\};\n/,
    'twinReject');
  const mk = (lang) => new Function('TARGET_LANGUAGE',
    `${block}\nreturn {twinReject, TWIN_APPLICABLE};`)(lang);
  const g = mk('hip');
  const t = (twin, extra) => ({ status: 'verified', correctness: 'pass',
    hip_twin_sync: twin, ...(extra || {}) });
  const good = { exit_code: 0, pairs: 1, drifted: 0 };
  const r = (rep, label) => g.twinReject(rep, label);

  ok(r(t(good)) === null, 'a tree in lockstep passes');
  ok(r(t({ ...good, pairs: 3 })) === null, 'more than one pair is fine');
  ok(/twin:drifted/.test(String(r(t({ exit_code: 1, pairs: 2, drifted: 1 })))),
    'exit 1 is refused: the edit is not in the binary that was measured');
  // THE case this gate exists for. Exit 2 means the tool found nothing to
  // compare, printed `HOLE:`, and proved nothing -- and a verifier that
  // measured a build in a directory with no twins either did not build there or
  // scanned before the build.
  const hole = String(r(t({ exit_code: 2, pairs: 0, drifted: 0 })));
  ok(/twin:nothing_checked/.test(hole) && /HOLE, not its pass/.test(hole),
    'exit 2 is refused and says why, rather than reading as a pass', hole);
  // Exit 3 is the shape most easily mistaken for a pass: every line the tool
  // could read matched, and only the launch half is unread. A pair that reports
  // it has NOT been shown to be in lockstep, and a launch-only edit -- a grid,
  // a block size, a stream -- is exactly what would hide there.
  const unread = String(r(t({ exit_code: 3, pairs: 1, drifted: 0 })));
  ok(/twin:launch_unreadable/.test(unread) && /UNCHECKED/.test(unread),
    'exit 3 is refused and says which half went unread', unread);
  ok(/twin:receipt_absent/.test(String(r({ status: 'verified', correctness: 'pass' }))),
    'a verifier that never ran the tool is refused: a check nobody runs is a comment');
  ok(/twin:no_exit_code/.test(String(r(t({ pairs: 1 })))) &&
     /twin:no_exit_code/.test(String(r(t({ exit_code: '0', pairs: 1 })))),
    'a receipt with no integer exit code cannot be read as any of the three outcomes');
  ok(/twin:unknown_exit/.test(String(r(t({ exit_code: 7, pairs: 1 })))),
    'an exit code the tool cannot emit is refused rather than defaulted to pass');
  // Exit 0 has two consequences the tool itself guarantees, so violating either
  // means the receipt was written rather than run.
  ok(/twin:zero_pairs_with_exit_zero/.test(String(r(t({ exit_code: 0, pairs: 0, drifted: 0 })))),
    'exit 0 with no pairs is impossible: the tool returns 2 in that case');
  ok(/twin:exit_zero_with_drift/.test(String(r(t({ exit_code: 0, pairs: 2, drifted: 1 })))),
    'exit 0 with drift is impossible: the tool returns 1 in that case');
  ok(r(null) === null && r(undefined) === null,
    'an absent report is not this gate\'s refusal to make');
  ok(r({ status: 'failed' }) === null && r({ correctness: 'FAIL' }) === null,
    'an honest failure is refused under its own name, not renamed into paperwork (60)');
  ok(/merge/.test(String(r({ status: 'verified' }, 'merge'))),
    'the label names which consumer measured without checking');
  const reasons = [r({ status: 'verified' }), r(t({ pairs: 1 })), r(t({ exit_code: 1, pairs: 1, drifted: 1 })),
    r(t({ exit_code: 2, pairs: 0 })), r(t({ exit_code: 3, pairs: 1 })),
    r(t({ exit_code: 7, pairs: 1 })),
    r(t({ exit_code: 0, pairs: 0 })), r(t({ exit_code: 0, pairs: 2, drifted: 1 }))];
  ok(new Set(reasons.map(x => String(x).split('(')[0])).size === reasons.length,
    'the eight twin refusals are eight distinguishable tokens', JSON.stringify(reasons));
  ok(reasons.every(x => /^twin:/.test(String(x))),
    'every one is namespaced, so a log line says which gate spoke');

  // Scope. The default `target_language` is triton, which has no .hip twin to
  // compare -- the tool would return 2 forever and the gate would refuse every
  // candidate in the lane for a hazard that cannot occur in it.
  const tri = mk('triton');
  ok(tri.TWIN_APPLICABLE === false && tri.twinReject({ status: 'verified' }) === null,
    'a triton lane is not held to a check about hipified .hip twins');
  ok(mk('HIP').TWIN_APPLICABLE === true && mk('cuda').TWIN_APPLICABLE === true,
    'hip and cuda arm it, case-insensitively -- those are the lanes ninja hipifies');
  ok(/twin:receipt_absent/.test(String(mk('hip').twinReject({ status: 'verified' }))),
    'and the hip lane, which is the one this project runs, is held to it');

  // Wiring (55): a gate whose result is computed and discarded is a comment,
  // and this gate's whole finding is that the tool was never called at all.
  ok(/policyReject\(r\.ver\)\n\s*\|\| twinReject\(r\.ver\);/.test(src),
    'the candidate admission path runs it beside the policy gate');
  ok(/twinReject\(integrate, 'integrate'\)/.test(src),
    'and so does the merge, where an edit is most likely to land in only one half of a pair');
  // Counted, not merely present. The sentence this asserts is plural -- the
  // receipt is declared everywhere a verdict carrying one arrives -- and a bare
  // presence match is satisfied by any single surviving declaration, so it stays
  // green while the schema that actually strips the field loses it. Two sites:
  // the verify verdict and the integrate verdict.
  ok(/hip_twin_sync: HIP_TWIN_SCHEMA/.test(src) &&
     (src.match(/hip_twin_sync: HIP_TWIN_SCHEMA/g) || []).length === 2,
    'the receipt is in every schema, so a verifier that returns it is not stripped');
}

console.log('\n# the ISA mechanism gate: language scope and what it may refuse, executed');
{
  // The layer above the twin gate. Twin-sync answers "was the edited file
  // compiled"; this answers "did the mechanism survive the compiler". Two things
  // are pinned here, and the first is a naming claim rather than a behaviour:
  //
  // The mode is the ONLY precondition: this must NOT consult TARGET_LANGUAGE. An
  // earlier version did, and it failed silently in the one way that matters --
  // TARGET_LANGUAGE defaults to 'triton' when the caller omits it (the greedy
  // pipeline omits it), so a whitelist turned the entire layer off with no message.
  // The real precondition is "did the build produce an AMDGPU code object", which
  // `isa_capture.py` measures directly and reports as its HOLE exit code. A loud
  // HOLE beats a silent off, so the protection lives in the non-refusal probes
  // below rather than in a language list.
  const block =
    grab(/const ISA_MODES = \['off', 'observe', 'gate'\][\s\S]*?const ISA_ENABLED = ISA_MODE !== 'off';\n/,
      'ISA_MODES..ISA_ENABLED') +
    grab(/const isaEvidenceReject = \(rep, label\) => \{[\s\S]*?\n\};\n/, 'isaEvidenceReject');
  const mk = (lang, mode) => new Function('A', 'TARGET_LANGUAGE',
    `${block}\nreturn {isaEvidenceReject, ISA_ENABLED, ISA_MODE, ISA_MODE_WARNING};`)(
    { isa_evidence: mode }, lang);

  ok(!/ISA_LANGUAGES/.test(src) && !/ISA_ENABLED[^\n]*TARGET_LANGUAGE/.test(src),
    'arming the ISA layer does not depend on TARGET_LANGUAGE -- a self-declared '
    + 'argument that defaults to triton is not a precondition, it is a silent off '
    + 'switch, and the real one (did a code object exist) is measured by the capture');
  ok(mk('triton', 'gate').ISA_ENABLED === true && mk('cuda', 'gate').ISA_ENABLED === true &&
     mk('hip', 'gate').ISA_ENABLED === true,
    'every language arms it identically; a lane whose build this cannot read gets a '
    + 'reported HOLE and an indeterminate verdict, not a gate that quietly did nothing');
  ok(mk('hip', 'gate').ISA_ENABLED === true && mk('HIP', 'observe').ISA_ENABLED === true,
    'both modes that do anything arm it');

  // Observation is ON by default, and that is load-bearing. The case this layer
  // catches -- a mechanism the compiler removed -- produces a round that reads as a
  // clean negative, so there is no symptom to escalate from. A conditional or
  // off-by-default check is therefore not a cheaper version of this check, it is one
  // that never fires on the case it was written for. Finding (87) is the standing
  // example of a correct tool nothing called.
  ok(mk('hip', undefined).ISA_MODE === 'observe' && mk('hip', undefined).ISA_ENABLED === true,
    'the default is observe: a lane that passes no isa_evidence arg still records '
    + 'whether each round\'s mechanism reached the machine code');
  ok(mk('hip', undefined).isaEvidenceReject({ status: 'verified', correctness: 'pass',
    isa_evidence: { exit_code: 0, mechanism_verdict: 'refuted' } }) === null,
    'and the default still refuses NOTHING -- default-on is an observation, not an '
    + 'enforcement, so turning it on by default cannot change a run\'s outcome');
  ok(mk('hip', 'off').ISA_ENABLED === false,
    '`off` remains a real escape hatch for an ablation or an unreadable box');
  ok(mk('hip', 'GATE ').ISA_MODE === 'gate',
    'the mode normalises case and whitespace');
  ok(mk('hip', 'nonsense').ISA_MODE === 'observe' &&
     /is not one of/.test(String(mk('hip', 'nonsense').ISA_MODE_WARNING)),
    'an unknown mode resolves to the default AND names itself -- resolving `gat` to '
    + '`off` in silence is the same silent-off failure the language whitelist had, and '
    + 'a typo is exactly how enforcement would get switched off unnoticed');
  ok(mk('hip', undefined).ISA_MODE_WARNING === '' && mk('hip', 'gate').ISA_MODE_WARNING === '',
    'and a legitimate mode warns about nothing, so the warning means what it says');

  const refuted = { status: 'verified', correctness: 'pass',
    isa_evidence: { exit_code: 0, mechanism_verdict: 'refuted',
      claims_refuted: ['widen_global_load'], unchanged_machine_code: false } };
  const gate = mk('hip', 'gate');
  ok(/isa:mechanism_refuted/.test(String(gate.isaEvidenceReject(refuted))),
    'in gate mode a refuted mechanism is refused, by name');
  ok(/not be recorded as one/.test(String(gate.isaEvidenceReject(refuted))),
    'and the refusal says the round must NOT be recorded as a slow kernel -- that '
    + 'distinction is the entire point: the mechanism was never tested, and a ledger '
    + 'row saying "no effect" closes the direction for the rest of a greedy run');
  ok(mk('hip', 'observe').isaEvidenceReject(refuted) === null,
    'observe mode records the same verdict and refuses nothing, so the signal can be '
    + 'measured on real kernels before anything is gated on it');

  // The asymmetry against (87), and the reason it is not an inconsistency. Exit 2
  // MUST refuse there, because the missing evidence is evidence about the
  // measurement itself and an unbacked measurement manufactures null results. Here
  // the measurement is already backed by twin-sync, policy and the oracle, so
  // missing ISA evidence is a gap on OUR side -- refusing on it would discard
  // correct fast kernels on any box where ROCm sits where the capture did not look.
  ok(gate.isaEvidenceReject({ status: 'verified', correctness: 'pass' }) === null,
    'a MISSING receipt does not refuse: that is our evidence gap, not a fault in the '
    + 'candidate -- the opposite of the twin gate, and deliberately so');
  ok(gate.isaEvidenceReject({ status: 'verified', correctness: 'pass',
    isa_evidence: { exit_code: 2, mechanism_verdict: 'indeterminate' } }) === null,
    'an archive HOLE does not refuse either, for the same reason');
  ok(gate.isaEvidenceReject({ status: 'verified', correctness: 'pass',
    isa_evidence: { exit_code: 0, mechanism_verdict: 'indeterminate' } }) === null,
    'nor does an indeterminate verdict -- reading absent evidence as a refutation '
    + 'would manufacture the very false negative this gate exists to prevent');
  ok(gate.isaEvidenceReject({ status: 'failed', correctness: 'FAIL',
    isa_evidence: { exit_code: 0, mechanism_verdict: 'refuted' } }) === null,
    'a report that already failed is refused on its own terms; stacking a mechanism '
    + 'complaint on top would rename the refusal (60)');

  // --- the escalation ladder, executed against fabricated histories -------
  // Extracted as a named function precisely so this block can run it. A rule about
  // when to spend the run's most expensive evidence, defended only by a regex on its
  // log message, is the shape `audit_pin_coverage.py` already caught once.
  // Extracted as one contiguous block, which also pins that the ladder's vocabulary,
  // its two admissibility predicates and its accounting stay together: they are one
  // rule split across four names, and a mutation that moved one away from the others
  // would be invisible to a per-function grab.
  const L = new Function(
    grab(/const STAGE_L1 = [\s\S]*?\nconst reachedStage = \([\s\S]*?\n\};\n/, 'the evidence ladder')
    + '\nreturn { STAGE_L1, STAGE_L2, STAGE_L3, STAGE_L4, STAGE_FAILED, STAGE_LEGACY,'
    + ' EVIDENCE_MODEL, reachedStageOf, hasDominantHotKernel, evidenceLadder,'
    + ' admissibleAttribution, wantsCompilerEscalation, reachedStage };')();
  const ladder = L.evidenceLadder;
  // A profile that named a hot kernel. Required for any escalation: a trajectory is
  // captured per function, so without one L3 would trace a guess.
  const hot = { top_kernels: [{ name: '_Z10hot_kernelPf', pct_of_total: 91.2 }] };
  const rnd = (stage, mechanisms) => ({
    evidence: { model: L.EVIDENCE_MODEL, reached_stage: stage },
    results: (mechanisms || []).map(m => ({ mechanism: m })),
  });

  ok(ladder([], 0, true, hot).requested === null
     && ladder([rnd(L.STAGE_L2)], 0, true, hot).requested === null,
    'a round that is still improving does not escalate -- the ladder does not spend a recompile '
    + 'on a search that is working');
  ok(ladder([rnd(L.STAGE_L2)], 1, true, hot).requested === L.STAGE_L3,
    'ONE non-improving round escalates to the IR trajectory: MAX_NO_IMPROVE defaults to 2, so a '
    + 'three-round stagnation window would never fire before the lane was already dead');
  ok(ladder([rnd(L.STAGE_L2)], 1, true, hot).from === L.STAGE_L2,
    'and it records what it escalated FROM');

  // THE INVERSION FIX. The previous ladder sent a refuted mechanism straight to the
  // compiler role, skipping IR attribution entirely -- so the deepest rung was
  // reached exclusively by the route that handed it nothing but disassembly
  // statistics, which is why it could only ever guess at the whole backend.
  ok(ladder([rnd(L.STAGE_L2, ['refuted'])], 0, true, hot).requested === L.STAGE_L3,
    'a REFUTED mechanism escalates to L3, NOT past it: "my edit did not reach the binary" is '
    + 'first of all a question about which pass undid it, and that is an IR question');
  ok(ladder([rnd(L.STAGE_L2, ['refuted'])], 0, true, hot).requested !== L.STAGE_L4,
    'and it specifically does not jump to the compiler, which is what the old ladder did');
  ok(ladder([rnd(L.STAGE_L3)], 1, true, hot).requested !== L.STAGE_L4
     && ladder([rnd(L.STAGE_L3)], 5, true, hot).requested !== L.STAGE_L4,
    'L4 is NEVER requested at the top of a round, at any stagnation count: it is entered from '
    + 'inside the round, on a question L3 formed. Requesting it here is what made it unreachable '
    + 'under the default stop budget');

  ok(ladder([rnd(L.STAGE_L2, ['realized'])], 0, true, hot).requested === null
     && ladder([rnd(L.STAGE_L2, ['indeterminate'])], 0, true, hot).requested === null,
    'neither a realized nor an indeterminate verdict triggers escalation -- only a positive '
    + 'contradiction does, so missing evidence never spends a deep round');
  ok(ladder([rnd(L.STAGE_L2)], 5, false, hot).requested === null,
    'with the layer off it never escalates at all, whatever the history looks like');

  // L2 must have localised something. The paper's first test is "has this level
  // found the dominant bottleneck", not "is it still slow".
  const noHot = { top_kernels: [] };
  ok(ladder([rnd(L.STAGE_L2)], 1, true, noHot).requested === null
     && /dominant hot kernel/.test(String(ladder([rnd(L.STAGE_L2)], 1, true, noHot).skip_reason)),
    'a stalled round whose profile named no dominant hot kernel does not escalate, and says why: '
    + 'a trajectory is per-function, so L3 would otherwise trace whichever translation unit was '
    + 'guessed at and attribute this plateau to it');
  ok(ladder([rnd(L.STAGE_L2)], 1, true, undefined).requested === null,
    'and a missing profile is treated the same way rather than as permission');

  // Legacy STATE. A resumed lane must not read a pre-v2 round as if it held the
  // evidence this ladder produces.
  ok(L.reachedStageOf({ evidence_depth: 'isa' }) === L.STAGE_LEGACY
     && L.reachedStageOf({ evidence_depth: 'compiler' }) === L.STAGE_LEGACY,
    'a round banked before this ladder existed reads as legacy machine-code evidence, never as '
    + 'L3 or L4: it names no pass, so it can discharge nothing');
  ok(ladder([{ evidence_depth: 'compiler' }], 1, true, hot).requested === L.STAGE_L3,
    'and a resumed lane whose last round says "compiler" still starts at L3 -- it does not '
    + 'inherit a rung nobody ever reached');

  // Admissibility: the schema-level half of "ISA may corroborate, never be the source".
  const full = { status: 'attributed', diagnosis: 'd', attributed_pass: 'si-load-store-opt',
    stage_transition: '58 -> 59' };
  ok(L.admissibleAttribution(full) === '',
    'an attribution naming a pass and a stage pair is admissible');
  // Each guard is exercised with the OTHER field present, so neither can be
  // deleted and stay green behind the one beside it. The first version of this
  // omitted `stage_transition` from the no-pass case, so both guards fired on one
  // input and the mutation suite proved the pass check itself was unpinned.
  ok(L.admissibleAttribution({ status: 'attributed', diagnosis: 'vgpr=152, scratch=0',
    stage_transition: '58 -> 59' }) !== '',
    'an `attributed` return that names NO pass is refused -- that is exactly the old L3 output, '
    + 'true numbers from which no narrowed compiler question can be built');
  ok(L.admissibleAttribution({ status: 'needs_compiler', diagnosis: 'd',
    attributed_pass: 'p' }) !== '',
    'and one with a pass but no stage transition is refused too: without the two stages the '
    + 'finding cannot be re-run, and an unverifiable attribution is the one output this layer '
    + 'must not produce');
  ok(L.admissibleAttribution({ status: 'inconclusive', diagnosis: 'the trajectory does not '
    + 'explain it' }) === '',
    'but `inconclusive` needs no pass -- admitting the evidence did not answer is a real finding '
    + 'and must stay reachable, or the analyst is pushed into inventing a mechanism');

  // The L3 -> L4 handoff, and its refusal.
  ok(L.wantsCompilerEscalation({ status: 'needs_compiler', suspected_passes: ['si-load-store-opt'],
    compiler_question: 'which legality condition blocks the merge?' }) === true,
    'a needs_compiler return carrying suspected passes AND one question escalates to L4');
  ok(L.wantsCompilerEscalation({ status: 'needs_compiler', suspected_passes: [],
    compiler_question: 'why is it slow?' }) === false
     && L.wantsCompilerEscalation({ status: 'needs_compiler',
       suspected_passes: ['p'], compiler_question: '  ' }) === false,
    'and one missing either is refused: L4\'s whole cost control is that it is handed a question '
    + 'with a stopping condition');
  ok(L.wantsCompilerEscalation({ status: 'attributed', suspected_passes: ['p'],
    compiler_question: 'q' }) === false,
    'an `attributed` L3 does NOT escalate even if it filled the fields -- it reached a single '
    + 'rewrite family, and spending L4 on top of that is the per-round tax this ladder avoids');

  // Reachability under the DEFAULT stop budget, which is the defect this change fixes.
  // MAX_NO_IMPROVE is 2, so noImprove can only ever be 1 when an escalation is
  // decided; the old chain needed a SECOND round after an L3 round to reach the
  // compiler, and the loop exits first.
  {
    // Read out of the lane rather than restated, for the reason `grabConst` exists:
    // a guard that hardcoded 2 would keep passing after someone raised the budget,
    // and the whole point of this check is that the budget is what made L4
    // unreachable. `grabConst` cannot be used because this one is computed, so the
    // default is taken from the expression that supplies it.
    const maxNoImprove = Number(grab(
      /const MAX_NO_IMPROVE = Math\.max\(1, parseInt\(A\.max_no_improve != null \? A\.max_no_improve : (\d+), 10\)\);/,
      'the MAX_NO_IMPROVE default').match(/: (\d+), 10/)[1]);
    let reachedL4 = false;
    const history = [];
    for (let noImprove = 0; noImprove < maxNoImprove; noImprove++) {
      const state = ladder(history, noImprove, true, hot);
      if (state.requested === L.STAGE_L3) {
        const l3 = { status: 'needs_compiler', diagnosis: 'd', attributed_pass: 'si-insert-waitcnts',
          stage_transition: '92 -> 93', suspected_passes: ['si-insert-waitcnts'],
          compiler_question: 'which dependency forces the wait?' };
        if (L.admissibleAttribution(l3) === '' && L.wantsCompilerEscalation(l3)) reachedL4 = true;
      }
      history.push(rnd(state.requested === L.STAGE_L3 ? L.STAGE_L3 : L.STAGE_L2));
    }
    ok(reachedL4,
      'L4 is reachable within the DEFAULT stop budget. It was not before: `compiler` could only '
      + 'be requested from a round whose prior depth was already `isa`, which needed the L3 round '
      + 'to survive into the next -- but a non-improving L3 round takes noImprove to '
      + `${maxNoImprove} and the loop exits. The only live path to the deepest rung was the one `
      + 'that skipped L3');
  }

  // Wiring (55): a gate whose result is computed and discarded is a comment.
  ok(/const ladderState = evidenceLadder\(history\.rounds, noImprove, LADDER_ENABLED, profileSummary\);/.test(src),
    'the round loop actually consults the ladder rather than deciding depth inline');
  ok(/if \(wantsCompilerEscalation\(irAttribution\)\) \{/.test(src)
     && src.indexOf('roleAgent(\'compiler_engineer\'') > src.indexOf('roleAgent(\'ir_engineer\''),
    'and L4 is dispatched from inside the L3 branch, after L3 returned -- not from the top of a '
    + 'later round');
  ok(/const inadmissible = admissibleAttribution\(irAttribution\);/.test(src),
    'the admissibility check runs on the real return, so an attribution naming no pass cannot '
    + 'reach the planner');
  // Executed, not regex-matched. The first version of this asserted that the string
  // `pattern_after_failed_escalation` appeared in the lane -- and it did, in the
  // COMMENT beside the code, so the mutation that deleted the rule left the check
  // green. A guard that a comment can satisfy is not a guard.
  const reached = L.reachedStage;
  const someAttribution = { status: 'attributed', diagnosis: 'x' };
  ok(reached(L.STAGE_L3, someAttribution, null, true) === L.STAGE_L3 &&
     reached(L.STAGE_L3, someAttribution, someAttribution, true) === L.STAGE_L4,
    'a rung whose analysis returned a diagnosis is recorded as reached, and L4 supersedes L3 '
    + 'when both ran');
  ok(reached(L.STAGE_L3, { status: 'inconclusive', diagnosis: 'the trajectory does not explain it' },
    null, true) === L.STAGE_L3,
    'an INCONCLUSIVE attribution still counts as reaching the rung -- a plateau the evidence '
    + 'cannot explain is a real finding, and an analyst forced to produce a mechanism instead of '
    + 'admitting that is an analyst inventing one');
  ok(reached(L.STAGE_L3, { status: 'unavailable', diagnosis: 'capture failed' }, null, true)
       === L.STAGE_FAILED,
    'but `unavailable` does NOT: there was nothing to read, which is the failed-escalation case, '
    + 'and collapsing it into `inconclusive` would tell the next round to stop asking');
  ok(reached(L.STAGE_L3, null, null, true) === L.STAGE_FAILED,
    'a requested rung with NO attribution behind it is neither L3 nor L2: the first says "we '
    + 'reconstructed the trajectory and it did not help", the second says "we never looked", and '
    + 'a failed escalation is a false negative one level above the one this layer catches');
  ok(reached(null, null, null, true) === L.STAGE_L2 &&
     reached(L.STAGE_L3, someAttribution, null, false) === L.STAGE_L2,
    'no escalation, and the layer off, both record plain L2');
  ok(/reached_stage: stageReached/.test(src) && /evidence_depth: stageReached/.test(src) &&
     /const stageReached = reachedStage\(ladderState\.requested, irAttribution, compilerAttribution,/.test(src),
    'and the round record actually stores that computed value rather than the requested rung');
  ok(/model: EVIDENCE_MODEL/.test(src) && /requested_stage: ladderState\.requested \|\| null/.test(src),
    'the record is versioned and keeps request and reach in SEPARATE fields -- collapsing them is '
    + 'what let "we asked for deep evidence" read as "we have deep evidence" on a resumed lane');
  ok(/const isaReason = isaEvidenceReject\(r\.ver\);/.test(src) &&
     /if \(metricReason \|\| isaReason\) \{/.test(src),
    'the candidate admission path actually consults it, and does so WITHOUT joining '
    + 'the twin/policy chain above, whose exact text another check in this file pins');
  ok(/isa_evidence: ISA_EVIDENCE_SCHEMA/.test(src),
    'the receipt is in the verify schema, so an agent that returns it is not stripped');
  ok(/mechanism_claims: \{ type: 'array'/.test(src),
    'and the engineer can declare the claim the receipt is tested against');
}

console.log('\n# the budget, the commit threshold and the correctness gate');
{
  ok(/plannedCost \+ d\.cost > remaining/.test(src),
    'the planner cannot overspend the remaining budget');
  // The suite threshold is unchanged and is still the FALLBACK -- the expression a run evaluates
  // when no band table could be built at all. It is no longer the normal path: bands are now
  // derived from the baseline's own repeats, so `ROUTE_BANDS` is populated on any run whose
  // benchmark engineer reported `samples_ms`. Both are pinned, because a lexical check on the
  // threshold alone would keep passing if the fallback were deleted and the per-route verdict
  // became unconditional, and a check on the per-route branch alone would keep passing if the
  // derivation were deleted and the gate silently went back to never running.
  // Anchored inside `judgeCandidate`, where the per-candidate decision now lives so that SELECTION
  // and the logged verdict cannot answer differently. The threshold itself is untouched.
  // The legacy suite threshold is no longer any part of the commit decision. It survives for two
  // narrower jobs, both pinned here: the degraded path when the paired per-case times are missing
  // entirely, and the audit line that reports whenever the live gate and the number every prior
  // round was judged on disagree.
  ok(/const legacyImproved = cand\.geomean > cumulative \* \(1 \+ MIN_IMPROVE\);/.test(src),
    'the suite-geomean threshold survives as the degraded fallback');
  ok(/if \(!rv \|\| !rv\.applicable\) \{/.test(src) &&
     /suiteSaysYes: legacyImproved,\n\s*improved: legacyImproved \};/.test(src),
    'and it is the WHOLE decision only when the gate cannot run at all');
  ok(/improved: rv\.accepted/.test(src),
    'when the gate can run, its own verdict is the decision -- no second arm can carry a candidate ' +
    'it refused, and no threshold can refuse one it accepted');
  ok(/OVERTURNS the legacy suite-geomean gate/.test(src),
    'a disagreement with the legacy number is logged in both directions, so the change of gate is ' +
    'auditable against what every prior round was judged on');
  // Finding (62) split this from one conjunction into a gate plus a named metric
  // refusal, so the shape changed; the threshold it enforces did not.
  ok(/says\(r\.ver\.correctness, 'pass'\)\)\) return false;/.test(src) &&
     /return primSpeedup\(r\.ver\) > CANDIDATE_FLOOR;/.test(src),
    'the verified filter still gates on correctness and the candidate floor');
}

console.log('\n# the commit gate needs no band file, and a measured table only TIGHTENS it, executed');
{
  // Why this section exists. The per-route gate and its Python twin were both written, both correct,
  // and both defended by tests -- and across seven waves the gate logged NOTHING, because its only
  // band source was an `args.route_bands` table nobody passed and the sole file on disk was six
  // epochs stale. A gate that cannot be fed is a gate that does not run.
  //
  // The first answer to that was to derive a table in-lane from the baseline's five repeats. That is
  // now deleted: at n=5 a min-max range measures whether a flyer landed in the sample, not the
  // route, and against a 24-repeat calibration it came out 8.8x too tight on one route and 2.9x too
  // loose on another. The second answer, pinned here, is that the gate needs NO table: every route
  // is held to MIN_ROUTE_WIN unless a measured floor says that route is noisier than that.
  const routeGate = makeGate();
  const inc = [{ name: 'a', optimized_ms: 0.100 }, { name: 'b', optimized_ms: 0.200 }];
  const cand = (aMs) => [{ name: 'a', optimized_ms: aMs }, { name: 'b', optimized_ms: 0.200 }];

  ok(MIN_ROUTE_WIN === 0.02,
    'the default bar is the operator\'s 2%; if this changes, every assertion below moves with it',
    String(MIN_ROUTE_WIN));

  // No table at all is the normal case, not a degraded one.
  {
    const v = routeGate(cand(0.097), inc, null);        // a is 3% faster
    ok(v.applicable && v.accepted && v.improved.join() === 'a',
      'with NO floor table the gate still runs and accepts a route past the default bar', v.reason);
  }
  {
    const v = routeGate(cand(0.099), inc, null);        // a is 1% faster: real, but under the bar
    ok(v.applicable && !v.accepted && /no single route cleared/.test(v.reason),
      'a 1% route gain is refused by the default bar even though the suite improved -- this is the ' +
      'deliberate cost of a fixed 2%, and it is the reason a measured table may only RAISE it',
      v.reason);
  }
  // A measured floor WIDER than the default raises that route's bar. This is the epoch Z case:
  // decode_m8_up measured 7.64%, so a 3% move there is noise being read as a mechanism.
  {
    const v = routeGate(cand(0.097), inc, { a: 0.0764 });
    ok(v.applicable && !v.accepted && /its measured floor/.test(v.reason),
      'a measured floor wider than the default raises that route\'s bar, so a 3% move on a route ' +
      'measured at 7.64% is not a mechanism', v.reason);
  }
  // A measured floor NARROWER than the default changes nothing. Every floor on epochs Y, A and B is
  // in this range, so on a quiet box the table is inert and 2% governs -- which is what makes the
  // operator's number the rule and the table the exception.
  {
    const v = routeGate(cand(0.097), inc, { a: 0.002, b: 0.002 });
    ok(v.applicable && v.accepted,
      'a measured floor narrower than the default does not lower the bar below it');
    const under = routeGate(cand(0.099), inc, { a: 0.002, b: 0.002 });
    ok(!under.accepted,
      'and a 1% gain stays refused even where the route measured 0.2% -- the table cannot open the ' +
      'gate wider than the operator set it');
  }
  // A partial table is no longer a refusal. It used to be ("no measured band for route(s) ..."),
  // which read as a candidate defect rather than as our own missing measurement.
  {
    const v = routeGate(cand(0.097), inc, { a: 0.001 });   // b has no entry
    ok(v.applicable && v.accepted,
      'a table missing a route falls back to the default bar for that route rather than refusing ' +
      'the whole candidate');
  }

  // The wiring, lexically.
  ok(/const ROUTE_BANDS = ROUTE_BANDS_ARG;/.test(src),
    'the effective table is the supplied one, with no in-lane derivation behind it');
  ok(!/bandsFromSamples/.test(src),
    'the n=5 derivation is deleted, not left dormant -- a dormant estimator is one someone re-enables');
  ok(/scripts\/route_floors\.py/.test(src),
    'the log names where a real table comes from; the previous supply failure was nobody knowing');
  ok(/Commit gate: suite geomean must improve at all AND/.test(src),
    'the rule in force is stated in the log at Setup, so the run says what it will accept');
  // The effective-config echo is the OTHER place the rule is described, and it was missed when the
  // derivation was deleted: it went on printing "will be DERIVED from the baseline repeats after
  // the Benchmark phase" for a derivation that no longer existed. An echo whose whole purpose is to
  // tell a reader what will decide the round, describing machinery that cannot run, is worse than
  // no echo. Both descriptions must name the bar that is actually applied.
  ok(!/will be DERIVED from the baseline repeats/.test(src),
    'the config echo no longer promises a derivation that was deleted');
  ok(/route_bands=\$\{ROUTE_BANDS_ARG[\s\S]{0,400}MIN_ROUTE_WIN \* 100/.test(src),
    'the config echo names the actual bar in BOTH branches -- supplied and not -- so "no table" ' +
    'reads as "held to the default" rather than as a missing feature');
}

console.log('\n# the gate compares against a SAME-SESSION control when the verifier returns one, executed');
{
  // route_gate.py's own header records this exposure and declines to guard it: "the same unchanged
  // tree measures 1.5-3% differently between invocations, and the candidate and the incumbent it is
  // compared against come from different invocations... the tighter fix is not a device check but
  // comparing against a control measured in the candidate's own session." The verifiers already
  // build that arm; until now there was no field to return it in, so the gate compared this round's
  // candidate against a table measured in an earlier round -- a drift larger than the gains judged.
  const routeGate = makeGate();
  const bands = { a: 0.02, b: 0.02 };

  // A control row is `{name, optimized_ms}` with NO speedup -- the denominator measuring itself, so
  // a speedup would be 1.0 by construction. The gate must read that shape with the same accessor it
  // uses for a candidate row, or the narrower schema would be unreadable by the thing it feeds.
  const control = [{ name: 'a', optimized_ms: 0.100 }, { name: 'b', optimized_ms: 0.200 }];
  const cand = [{ name: 'a', optimized_ms: 0.090, speedup: 1.11 },
                { name: 'b', optimized_ms: 0.200, speedup: 1.00 }];
  {
    const v = routeGate(cand, control, bands, { targetRoutes: ['a'] });
    ok(v.applicable && v.accepted && v.improved.join() === 'a',
      'a control arm carrying only {name, optimized_ms} is readable and decides the verdict', v.reason);
  }
  // The point of the arm: the same patch judged against a control measured in a session that ran
  // 5% slow reads as a win, and against one that ran 5% fast reads as flat. Same candidate, same
  // bands -- only the incumbent's session differs. That is the drift this field cancels.
  {
    const slowSession = [{ name: 'a', optimized_ms: 0.105 }, { name: 'b', optimized_ms: 0.210 }];
    const fastSession = [{ name: 'a', optimized_ms: 0.0905 }, { name: 'b', optimized_ms: 0.190 }];
    const vSlow = routeGate(cand, slowSession, bands, { targetRoutes: ['a'] });
    const vFast = routeGate(cand, fastSession, bands, { targetRoutes: ['a'] });
    ok(vSlow.accepted === true && vFast.accepted === false,
      'the SAME candidate flips verdict with the incumbent\'s session -- which is why the arm exists',
      `slow=${vSlow.accepted} fast=${vFast.accepted}`);
  }

  // The wiring: preferred when present, degraded-but-used when absent, and always stated.
  // Anchored inside `judgeCandidate` since the per-candidate decision moved there; the invariant is
  // unchanged, and it now holds for every candidate the selector considers rather than only the
  // top-ranked one.
  ok(/const sameSession = !!cand\.control_per_case;/.test(src),
    'the gate prefers the same-session control and falls back to the stored table');
  ok(/routeGate\(cand\.per_case, sameSession \? cand\.control_per_case : bestPerCase,\s*\n\s*ROUTE_BANDS/.test(src),
    'the chosen incumbent side is what the gate actually reads');
  ok(/incumbent side: /.test(src),
    'which incumbent side decided the verdict is logged every round, not inferred');
  ok(/control_per_case: controlPerCase,/.test(src),
    'control_per_case is DECLARED in VERIFY_SCHEMA -- a required-but-undeclared field is one the ' +
    'agent is obliged to return and never told about');
  ok(!/\}, \['status', 'verified_geomean', 'policy_pass', 'control_per_case'\]/.test(src),
    'control_per_case is NOT in required: a hard schema failure would discard a whole round of GPU ' +
    'work over a missing arm (the precedent timing_basis set)');
  ok(/control_per_case: Array\.isArray\(r\.ver\.control_per_case\) && r\.ver\.control_per_case\.length/.test(src),
    'an empty control array is treated as absent, not as a control that measured nothing');
  ok(/accepted with no declared target_routes/.test(src),
    'a win banked with no declared target route is NOTED rather than refused -- unattributed, but ' +
    'not thrown away');
}

console.log('\n# the commit rule is a CONJUNCTION with a catastrophic fence, executed');
{
  // What this replaced, and what the old rule cost. The gate used to be a UNION of a per-route test
  // and a suite threshold, with a REGRESSION VETO sitting outside the union: any route past its band
  // in the wrong direction refused the candidate however good the average. Two refusals on the
  // record, both of them the largest result of their day:
  //   * wave 1 round 2 -- ten of eleven routes +24%..+50%, refused for 0.0006 ms on decode_m2_square.
  //   * wave 1 round 3 -- +5.35% suite (the eight improved routes averaged +8.4%), refused on three
  //     routes giving back 1.3%-2.8%.
  // The veto optimised Pareto-improvement across routes; this lane is scored on the unweighted suite
  // geomean. Refusing a candidate that improves the objective is optimising something nobody
  // measures. Two arguments were made for the veto and neither survives: that a union "cannot make a
  // run stricter" (falsified by round 2, the first round the gate was reachable) and that eleven
  // routes at +0.4% is a "~4.4% suite win" the per-route test would miss -- which was arithmetic
  // error. The geomean of eleven 1.004s is 1.004, asserted below so it cannot be believed again.
  const routeGate = makeGate();
  const routes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'];
  const rows = (f) => routes.map(r => ({ name: r, optimized_ms: 0.100 * f(r), speedup: 1 / f(r) }));
  const inc = rows(() => 1);

  // The arithmetic that was got wrong, pinned as a number.
  {
    const v = routeGate(rows(() => 0.996), inc, null);       // every route 0.4% FASTER
    ok(Math.abs(v.suiteRatio - 1 / 0.996) < 1e-9 && v.suiteRatio < 1.005,
      'eleven routes each 0.4% faster is a 0.4% suite win, not 4.4% -- the geomean does not add ' +
      'across routes, and the claim that it did was an argument for the rule this replaced',
      String(v.suiteRatio));
    ok(v.applicable && !v.accepted && /no single route cleared/.test(v.reason),
      'a broad thin gain is REFUSED: the average moved but nothing cleared its own bar. This is the ' +
      'known cost of requiring one concrete route, and it is deliberate -- a suite average that ' +
      'improves while no route clears its noise is the shape a round of measurement luck takes',
      v.reason);
  }
  // Condition 2 alone is not enough either: a route can clear its bar while the suite goes backwards.
  {
    // b gives back 8%: past its own noise, but deliberately INSIDE the catastrophic fence, so the
    // refusal below is the suite condition doing the work and not the fence.
    const v = routeGate(rows(r => (r === 'a' ? 0.96 : (r === 'b' ? 1.08 : 1))), inc, null);
    ok(v.applicable && !v.accepted && /suite geomean did not improve/.test(v.reason),
      'a route clearing its bar cannot bank a candidate that made the suite worse -- both ' +
      'conditions are required, which is the whole difference from the union', v.reason);
  }
  // Both conditions met: the single-route win the suite geomean divides by eleven.
  {
    const v = routeGate(rows(r => (r === 'a' ? 0.93 : 1)), inc, null);
    ok(v.applicable && v.accepted && v.improved.join() === 'a',
      'a single-route win of 7% is banked, though it moves the eleven-route geomean only 0.62%',
      v.reason);
  }
  // THE BEHAVIOUR CHANGE, as arithmetic: wave 1 round 2's shape is now accepted.
  {
    const v = routeGate(rows(r => (r === 'a' ? 0.80 : (r === 'b' ? 1.05 : 1))), inc, null);
    ok(v.applicable && v.accepted && v.regressed.join() === 'b' && v.suiteRatio > 1,
      'a candidate that gives ground on one route while the suite improves is now BANKED, and the ' +
      'route it paid with is named rather than used to refuse it', v.reason);
  }
  // ...but not without limit. The fence is not a noise judgement: it sits an order of magnitude
  // outside the widest floor ever measured on this task (7.64%), so it cannot refuse real work.
  {
    const bad = 1 + CATASTROPHIC_REGRESSION + 0.03;
    const v = routeGate(rows(r => (r === 'a' ? 0.50 : (r === 'b' ? bad : 1))), inc, null);
    ok(v.applicable && !v.accepted && v.catastrophic.join() === 'b' && /catastrophic/.test(v.reason),
      'a route regressed past the fence refuses the candidate however good the average -- "the ' +
      'average improved" must not be able to ship one shape a third slower', v.reason);
    ok(v.suiteRatio > 1,
      'and the fence fires on a candidate whose suite number is GOOD, which is the only case where ' +
      'it does any work', String(v.suiteRatio));
  }
  {
    const edge = 1 + CATASTROPHIC_REGRESSION - 0.005;
    const v = routeGate(rows(r => (r === 'a' ? 0.50 : (r === 'b' ? edge : 1))), inc, null);
    ok(v.accepted && !v.catastrophic.length,
      'just inside the fence is accepted, so the fence is a cliff at a stated number and not a ' +
      'gradual tightening nobody can predict');
  }

  ok(/improved: rv\.accepted/.test(src) && !/rv\.accepted \|\| suiteSaysYes/.test(src),
    'the commit decision IS the gate verdict -- there is no second arm that can carry a candidate ' +
    'the gate refused');
  ok(/banked while giving ground on/.test(src),
    'a commit that cost ground on some route says which routes and by how much; the ledger has to ' +
    'be able to answer "which shape got slower" three waves later');

  // The stall counter.
  ok(/const suiteProgress = !!\(winner && bestSeen > 0 && winner\.geomean > bestSeen \* \(1 \+ PROGRESS_DELTA\)\)/.test(src),
    'the old suite progress test is preserved under its own name');
  ok(/const routeProgress = !!\(routeVerdict && routeVerdict\.applicable && routeVerdict\.improved\.length\s*&& !routeVerdict\.catastrophic\.length\)/.test(src),
    'a round in which any route cleared its bar counts as progress, and only the fence can veto that');
  ok(/const madeProgress = suiteProgress \|\| routeProgress;/.test(src),
    'progress is the union of the two, so a route win cannot be scored as a stall');
  ok(/if \(madeProgress \|\| committedThisRound\) \{ noImprove = 0; \} else \{ noImprove\+\+; \}/.test(src),
    'the stall counter still also resets on a landed commit -- (127) unchanged');
  ok(/the search ADVANCED on route evidence/.test(src),
    'a round rescued from being scored a stall says which routes rescued it');
  // routeProgress is deliberately BROADER than the commit gate in two ways: it ignores the suite
  // condition and it ignores routes that gave ground within noise, because the cost of a false
  // "stalled" is every remaining round of the wave while a false "advancing" costs one.
  ok(!/routeProgress = .*suiteRatio/.test(src),
    'progress does not require the suite to have followed -- a route that moved is evidence the ' +
    'search found something, and a stopping rule errs toward continuing');
}

console.log('\n# SELECTION is the best candidate that PASSES, not the best candidate, executed');
{
  // Ranking is by suite geomean; acceptance is per-route; and only `candidates[0]` was ever offered
  // to the gate. So a round could be holding a candidate the gate would accept and still bank
  // nothing. Observed, not hypothetical: coldstart_newgate_20260819 wave 1 round 2 refused the top
  // candidate (0.924) for giving back 0.0006 ms on decode_m2_square while the second (0.632) was
  // ACCEPT with zero banded regressions and was never asked. The hole did not exist while both
  // halves ranked on the same geomean -- it was introduced by changing only the acceptance test,
  // which is why the fix belongs to the same change and not to a later one.
  const gateBlock = grab(/const routeGate = \(candPerCase, incPerCase, bands, opts\) => \{[\s\S]*?\n\};\n/,
    'routeGate');
  const judgeBlock = grab(/ {2}const judgeCandidate = \(cand\) => \{[\s\S]*?\n {2}\};\n/, 'judgeCandidate');
  const selectBlock = grab(
    / {2}candidates\.sort\(\(a, b\) => b\.geomean - a\.geomean\);[\s\S]*?\n {2}const winner = candidates\[0\] \|\| null;\n/,
    'the selection block');

  // Executed against the lane's own source, not a paraphrase: a selector that reordered correctly
  // in a reimplementation would prove nothing about the one that runs.
  const select = (cands, o) => new Function('CANDS', 'O', `
    const MIN_ROUTE_WIN = ${MIN_ROUTE_WIN};
    const CATASTROPHIC_REGRESSION = ${CATASTROPHIC_REGRESSION};
    ${gateBlock}
    const log = () => {};
    const round = 1;
    const MIN_IMPROVE = O.MIN_IMPROVE, ROUTE_BANDS = O.ROUTE_BANDS;
    const cumulative = O.cumulative, bestPerCase = O.bestPerCase;
    const candidates = CANDS;
    ${judgeBlock}
    ${selectBlock}
    return { winnerId: winner && winner.id, topOffered, improved: judgeCandidate(winner).improved };
  `)(cands, o);

  const bands = { a: 0.02, b: 0.02, c: 0.02 };
  const base = [{ name: 'a', optimized_ms: 0.100 }, { name: 'b', optimized_ms: 0.200 },
                { name: 'c', optimized_ms: 0.300 }];
  const cand = (id, geomean, ms) => ({
    id, geomean, per_case: [{ name: 'a', optimized_ms: ms.a }, { name: 'b', optimized_ms: ms.b },
                            { name: 'c', optimized_ms: ms.c }],
    control_per_case: base,
  });
  const o = { MIN_IMPROVE: 0.005, ROUTE_BANDS: bands, cumulative: 0.5, bestPerCase: base };

  // Fixtures written against the CONJUNCTION, not the old veto. `top` gives back 8% on `a` -- past
  // its noise, deliberately inside the catastrophic fence -- and its 2% gain on `b` cannot pull the
  // suite back above 1.0, so it fails on the suite condition.
  const topRefused = cand('top', 0.924, { a: 0.108, b: 0.196, c: 0.300 });
  // `a` improves 10% and nothing gives ground: both conditions met.
  const lowerPasses = cand('lower', 0.632, { a: 0.090, b: 0.200, c: 0.300 });
  // Suite down and nothing clears its bar: neither condition met.
  const alsoRefused = cand('lower2', 0.600, { a: 0.108, b: 0.200, c: 0.300 });

  {
    const r = select([topRefused, lowerPasses], o);
    ok(r.winnerId === 'lower' && r.improved === true,
      'a lower-ranked candidate that PASSES is selected over a top-ranked one that does not, so a ' +
      'round holding a bankable result does not come away empty', `winner=${r.winnerId}`);
    ok(r.topOffered === 0.924,
      'bestSeen still tracks the highest geomean OFFERED, not the one banked -- otherwise selecting ' +
      'a lower candidate would quietly lower the bar the stall counter measures against',
      String(r.topOffered));
  }
  {
    const r = select([lowerPasses, topRefused].sort((x, y) => y.geomean - x.geomean), o);
    ok(r.winnerId === 'lower', 'the input order does not matter -- selection re-sorts first');
  }
  {
    // The top one passing must behave exactly as before: no reordering, no new logging path.
    const topPasses = cand('top', 0.924, { a: 0.090, b: 0.200, c: 0.300 });
    const r = select([topPasses, lowerPasses], o);
    ok(r.winnerId === 'top' && r.improved === true,
      'when the top candidate passes it is still the winner -- the fix adds a fallback, it does not ' +
      'change which candidate is preferred');
  }
  {
    const r = select([topRefused, alsoRefused], o);
    ok(r.winnerId === 'top' && r.improved === false,
      'when NOTHING passes the top candidate is still the winner and still refused, so the refusal ' +
      'is reported against the best measurement of the round rather than against a leftover');
  }
  {
    // No floor table no longer means "no gate" -- every route falls back to MIN_ROUTE_WIN -- so
    // selection must still reach past a failing top candidate. Under the previous design this case
    // read the opposite way, because an absent table disabled the gate entirely.
    const r = select([topRefused, lowerPasses], { ...o, ROUTE_BANDS: null });
    ok(r.winnerId === 'lower',
      'with no floor table the gate still runs on the default bar, so selection still prefers the ' +
      'candidate that passes');
  }
  ok(/selecting the best candidate that PASSES rather than banking nothing/.test(src),
    'a round that had to reach past the top-ranked candidate says so, because "the winner" and ' +
    '"the best measurement" stop being the same object at that point');
  ok(/will move `cumulative` DOWN even though no route regressed/.test(src),
    'selecting a candidate whose suite number is below the incumbent is allowed but never silent');
}

console.log('\n# the vs-seed frame is READ OFF THE BASELINE, not assumed, executed');
{
  // (127), third firing, and this time root-caused to the script rather than to an agent's
  // arithmetic. `cumulative` is documented as "vs BASELINE_PER_CASE, so it starts at 1.0 every
  // wave", and `cumulativeVsSeed()` multiplied it by the prior wave's total on that basis. The
  // assumption holds only when the harness times a candidate against THE WAVE'S OWN TREE. When it
  // times against a frozen external oracle -- dense_bf16_gemm_fused against direct rocblas_gemm_ex
  // -- `verified_geomean` is absolute, `cumulative` inherits it on commit, and the product counts
  // the same speedup twice: 1.2054 x 1.2707 = 1.5317 went out as CUMULATIVE_VS_SEED on wave 4, and
  // 1.31581464 on the wave-2/3 boundary. One was caught by the TechLead, one by a human two waves
  // later. Correctness that depends on who is reading is not correctness.
  const resolve = new Function(
    `${grab(/const resolveVsSeedFrame = \(perCase, prior\) => \{[\s\S]*?\n\};\n/, 'resolveVsSeedFrame')}
     return resolveVsSeedFrame;`)();
  const rows = (...sp) => sp.map((s, i) => ({ name: `r${i}`, speedup: s }));

  // The real numbers from wave 4: the incumbent scores 1.2142 against the harness denominator, and
  // the prior wave banked 1.2054. Not 1.0 -- so the harness is anchored on something outside the wave.
  {
    const r = resolve(rows(1.2142, 1.2142, 1.2142), 1.2054);
    ok(r.frame === 'absolute', 'a baseline that already scores the PRIOR total against the harness ' +
      'denominator means the harness is oracle-anchored and the product would double-count', r.why);
  }
  // The other harness shape, which is what the chained form was written for.
  {
    const r = resolve(rows(1.0, 1.001, 0.999), 1.2054);
    ok(r.frame === 'chained', 'a baseline that scores ~1.0 against its own denominator means each ' +
      'wave re-anchors, so each wave really does contribute a factor', r.why);
  }
  // A fresh wave: both readings coincide, so there is nothing to get wrong and nothing to warn about.
  {
    const r = resolve(rows(1.21, 1.21), 1.0);
    ok(r.frame === 'chained' && /prior is 1\.0/.test(r.why),
      'with no prior wave the two readings give the same number, so the detector stays out of it');
  }
  // No evidence must not read as evidence. `speedup` is not a required field on a baseline row.
  {
    const r = resolve([{ name: 'a', latency_ms: 0.1 }], 1.2054);
    ok(r.frame === 'chained' && /unverified/.test(r.why),
      'with no speedup on the baseline rows the frame cannot be read, so it keeps the old behaviour ' +
      'and says the vs-seed number is unverified rather than presenting a guess as a measurement');
  }
  // The arithmetic the whole thing is about, end to end.
  {
    const vsSeed = (frame, prior, cum) => (frame === 'absolute' ? cum : prior * cum);
    ok(vsSeed('absolute', 1.2054, 1.2707) === 1.2707,
      'on an oracle-anchored harness the vs-seed total IS the cumulative, not the product');
    ok(Math.abs(vsSeed('chained', 1.2054, 1.2707) - 1.5317) < 1e-3,
      'and the product is exactly the 1.5317 that went out on wave 4 -- pinned so the wrong branch ' +
      'is recognisable if it is ever taken again');
  }

  ok(/const cumulativeVsSeed = \(\) => \(vsSeedFrame === 'absolute'/.test(src),
    'the reporting function branches on the detected frame rather than always multiplying');
  ok(/log\(`vs-seed frame: \$\{vsSeedFrame\.toUpperCase\(\)\}/.test(src),
    'which frame was detected, and why, is logged once per wave -- a silent choice here is how this ' +
    'defect survived two waves');
  ok(/VS_SEED_FRAME: vsSeedFrame === 'absolute'/.test(src),
    'the frame travels WITH the number to update_memory, because STATE.cumulative is written from ' +
    'it and read back as PRIOR next wave, so one more multiplication compounds per wave');
}

console.log('\n# the ISA parent archive follows the TREE, not the commit event, executed');
{
  // `isaCanonicalArchive` was assigned in exactly one place: inside the committed-winner branch. On a
  // lane where most rounds commit nothing it therefore stayed null for the whole run, no diff was
  // possible, and every mechanism_verdict came back `indeterminate` -- 11 clean captures across seven
  // waves produced 0 machine-readable verdicts. The layer was self-disabling in exactly the situation
  // it exists for.
  ok(/let isaCanonicalSourceHash = null;/.test(src),
    'the parent archive now carries the hash of the tree it describes, so the pairing is checkable');
  ok(/if \(!isaCanonicalArchive\) \{\s*for \(const r of clean\) \{/.test(src),
    'a parent archive is adopted from a verifier when the lane has none, instead of waiting for a commit');
  ok(/typeof receipt\.parent_source_hash !== 'string' \|\| !receipt\.parent_source_hash/.test(src) &&
     /NOT adopting/.test(src),
    'an archive that does not name its tree is REFUSED -- a stale parent gives confident wrong ' +
    'verdicts, which is worse than the indeterminate a missing parent gives');
  ok(/DROPPING the canonical parent archive/.test(src),
    'two verifiers disagreeing about the parent tree clears the pointer rather than picking a side');
  ok(/isaCanonicalSourceHash = isaCanonicalArchive \? \(winner\.isa_source_hash \|\| null\) : null;/.test(src),
    'on a commit the hash moves WITH the path, including to null -- a hash left behind would make ' +
    'the next round clear a pointer that was correct');
  ok(/isa_source_hash: r\.ver\.isa_evidence && typeof r\.ver\.isa_evidence\.source_hash === 'string'/.test(src),
    'a candidate carries the tree its own archive describes, ready to become the next parent');
  // The safety property the original comment names, still true: nothing invents a parent.
  ok(/\.\.\.\(isaCanonicalArchive \? \{ ISA_PARENT_ARCHIVE: isaCanonicalArchive \} : \{\}\)/.test(src),
    'a null parent is still simply absent from the verifier prompt -- no substituted tree');
}

console.log('\n# an argument that does not arrive must not look like a chosen default, executed');
{
  // Two different failures, two different mechanisms, and neither covers the other:
  //  * a MISSPELLED knob is silently dropped -- `search_strategy: "greedy"` sat in this lane's
  //    canonical invocation for six waves after the QD search was deleted, matching nothing;
  //  * an OMITTED knob takes its default -- one wave was retyped without `min_improve: 0.005`, ran
  //    at the 0.02 default, and refused a verified +1.58% stack. No whitelist can see that one.
  // So: a throw for the first, an effective-config echo for the second.
  const block = grab(/const KNOWN_ARGS = new Set\(\[[\s\S]*?\n\]\);\n/, 'KNOWN_ARGS');
  const { KNOWN_ARGS } = new Function(`${block}\nreturn {KNOWN_ARGS};`)();

  // Completeness, derived from the source rather than restated: every `A.<key>` the worker actually
  // reads must be in the set, or a legitimate caller gets thrown at. This is the assertion that
  // keeps the list honest as the worker grows -- a restated list would just be a second thing to
  // forget to update.
  const readKeys = [...new Set((src.match(/\bA\.[a-z_][a-z0-9_]*/g) || [])
    .map(s => s.slice(2)))].sort();
  const notListed = readKeys.filter(k => !KNOWN_ARGS.has(k));
  ok(notListed.length === 0,
    'every argument the worker reads is in KNOWN_ARGS -- otherwise the check refuses a real caller',
    notListed.join(', '));

  // The dispatcher spreads its OWN args into the worker on the optimize path (`{...A}`), so its
  // four private keys reach this check and must be tolerated. Without them, `mode=optimize` throws
  // for anyone who also named a bake-off option.
  for (const k of ['backends', 'enable_fp8', 'e2e_workflow_dir', 'kernel_lane_script']) {
    ok(KNOWN_ARGS.has(k), `the dispatcher's forwarded key "${k}" is accepted, not refused`);
  }
  // Everything e2e_workflow.js passes when it calls this worker directly.
  for (const k of ['kernel_path', 'workflow_dir', 'mode', 'target_language', 'op_spec', 'budget',
                   'max_no_improve', 'gpu_ids', 'state_dir', 'shared_kb', 'global_kb',
                   'incremental_analyze', 'e2e_feedback', 'task', 'exp_root', 'apply_to_original',
                   'use_expert_skills', 'expert_skills_dir', 'perf_knowledge_dir']) {
    ok(KNOWN_ARGS.has(k), `e2e's argument "${k}" is accepted`);
  }
  // The dead argument that motivated this, pinned as still dead: if a `search_strategy` is ever
  // reintroduced it must be READ, not merely tolerated.
  ok(!KNOWN_ARGS.has('search_strategy') && !/A\.search_strategy/.test(src),
    'search_strategy is neither read nor silently accepted -- it now throws instead of doing nothing');

  ok(/An unrecognized argument is silently ignored/.test(src) &&
     /Accepted arguments: /.test(src),
    'the refusal explains WHY and lists the accepted set, so the fix does not need the source');
  ok(/did you mean \$\{near\.join\(', '\)\}\?/.test(src),
    'the refusal suggests near misses, which is what makes a typo self-correcting');

  // The echo: it has to name the knobs whose SILENT default costs a result.
  ok(/Effective run configuration \(the knobs that decide what this run ADMITS\):/.test(src),
    'the effective configuration is echoed before the first agent call');
  for (const k of ['min_improve', 'candidate_floor', 'progress_delta', 'max_no_improve', 'budget',
                   'isa_evidence']) {
    ok(new RegExp(`shown\\('${k}'`).test(src), `the echo names ${k} and its effective value`);
  }
  ok(/\$\{A\[key\] == null \? ' \(default\)' : ''\}/.test(src),
    'the echo marks which values came from a DEFAULT rather than from args -- the distinction the ' +
    'min_improve incident turned on');
  ok(/state_dir=\$\{STATE_DIR \|\| '\(none -- this is a COLD start, not a continuation\)'\}/.test(src),
    'the echo says out loud when a run is a cold start, which a resumed lane must never be silently');
}

console.log('\n# the roadmap profile may not publish a ceiling it did not earn -- (89)');
{
  const block = grab(/const SOL_SHAPED_KEY = [\s\S]*?const profileSolStrip = \(rep, where\) => \{[\s\S]*?\n\};\n/,
    'profileSolStrip');
  const lines = [];
  const { profileSolStrip } = new Function('log',
    `${block}\nreturn {profileSolStrip};`)((m) => lines.push(String(m)));

  const rep = {
    bottleneck: 'latency', dispatch_count: 1,
    sol_gap_x: 12, peak_pct: 8.36, compute_floor_us: 71.6, memory_floor_us: 34.6,
    remaining_headroom: 0.917, roofline_basis: 'bf16 MFMA peak ~1.29 PFLOP/s',
    key_metrics: { valu_pct: 61.2, hbm_gbps: 2680, sol_gap: 12, pct_of_peak: 8.36 },
    top_kernels: [{ name: 'gemm', pct_of_total: 98.74 }],
    top_opportunities: ['shorten the C1 dependency chain'],
  };
  const out = profileSolStrip(rep, 'baseline profile');

  ok(out.sol_gap_x === undefined && out.peak_pct === undefined
    && out.compute_floor_us === undefined && out.memory_floor_us === undefined
    && out.remaining_headroom === undefined && out.roofline_basis === undefined,
    'every SOL-shaped top-level field is removed from the roadmap profile');
  ok(out.key_metrics.sol_gap === undefined && out.key_metrics.pct_of_peak === undefined,
    'key_metrics is scrubbed too -- additionalProperties:true is exactly where the '
    + 'ungrounded number hides from the schema');
  ok(out.key_metrics.valu_pct === 61.2 && out.key_metrics.hbm_gbps === 2680
    && out.bottleneck === 'latency' && out.dispatch_count === 1
    && out.top_kernels[0].pct_of_total === 98.74
    && out.top_opportunities.length === 1,
    'the measured counters survive -- the counters were real and were adopted; only '
    + 'the denominators were ungrounded');
  ok(/pct_of_total/.test(JSON.stringify(out)),
    'pct_of_total is not a SOL field and the prefix match does not eat it');
  ok(lines.length === 1 && /dropped ungrounded SOL fields/.test(lines[0])
    && /sol_gap_x/.test(lines[0]) && /key_metrics\.sol_gap/.test(lines[0]),
    'the removal is logged by name and qualified path, not silent -- a strip nobody '
    + 'is told about is indistinguishable from an agent that never emitted it');
  ok(typeof out.sol_note === 'string' && /measured ceiling/.test(out.sol_note),
    'the note says what a headroom claim would need instead, so the gap reads as '
    + '"this number was never earned" rather than "no headroom analysis exists"');

  const clean = { bottleneck: 'compute', key_metrics: { valu_pct: 90 } };
  const untouched = profileSolStrip(clean, 'baseline profile');
  ok(untouched.sol_note === undefined && lines.length === 1,
    'a profile that published no ceiling is not annotated and logs nothing');
  ok(profileSolStrip(null, 'x') === null && profileSolStrip(undefined, 'x') === undefined,
    'a missing profile does not throw on the way through the strip');

  ok(/profileSolStrip\(profileSummary, 'baseline profile'\)/.test(src)
    && /profileSolStrip\(profileSummary, `reprofile r\$\{round\}`\)/.test(src),
    'both profile branches are stripped -- reprofile writes to the same variable the '
    + 'TechLead reads, so covering only the baseline would leak from round 1 on');
}

console.log(failures ? `\nFAIL: ${failures} check(s) failed.`
  : '\nPASS: the admission gates hold.');
process.exit(failures ? 1 : 0);
