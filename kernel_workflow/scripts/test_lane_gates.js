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
  const ladder = new Function(
    grab(/const isaEvidenceDepth = \(rounds, noImproveCount, enabled\) => \{[\s\S]*?\n\};\n/,
      'isaEvidenceDepth') + '\nreturn isaEvidenceDepth;')();
  const rnd = (depth, mechanisms) => ({
    evidence_depth: depth,
    results: (mechanisms || []).map(m => ({ mechanism: m })),
  });

  ok(ladder([], 0, true).depth === 'pattern' && ladder([rnd('pattern')], 0, true).depth === 'pattern',
    'a round that is still improving stays at pattern depth -- the ladder does not spend '
    + 'machine-code analysis on a search that is working');
  ok(ladder([rnd('pattern')], 1, true).depth === 'isa',
    'ONE non-improving round escalates to machine-code evidence: MAX_NO_IMPROVE defaults to 2, '
    + 'so a three-round stagnation window would never fire before the lane was already dead');
  ok(/actually realised/.test(String(ladder([rnd('pattern')], 1, true).reason)),
    'and the reason says what the evidence is for -- establishing whether the last direction '
    + 'was realised at all, which a flat benchmark cannot distinguish from a bad idea');
  ok(ladder([rnd('isa')], 1, true).depth === 'compiler' &&
     ladder([rnd('isa')], 1, true).from === 'isa',
    'a plateau that survived machine-code attribution escalates once more, to the compiler, '
    + 'and records what it escalated FROM');
  ok(ladder([rnd('pattern', ['refuted'])], 0, true).depth === 'compiler',
    'a REFUTED mechanism jumps straight to the compiler even on an improving round: the ISA '
    + 'has already answered "did it land" with no, so re-asking it re-derives what is known');
  ok(ladder([rnd('pattern', ['realized'])], 0, true).depth === 'pattern' &&
     ladder([rnd('pattern', ['indeterminate'])], 0, true).depth === 'pattern',
    'and neither a realized nor an indeterminate verdict triggers it -- only a positive '
    + 'contradiction does, so missing evidence never spends a deep round');
  ok(ladder([rnd('pattern')], 3, true).depth === 'isa',
    'the ladder climbs one step at a time; it does not skip to the compiler because the lane '
    + 'has been flat longer');
  ok(ladder([rnd('isa')], 5, false).depth === 'pattern',
    'and with the ISA layer off it never escalates at all, whatever the history looks like');

  // Wiring (55): a gate whose result is computed and discarded is a comment.
  ok(/const isaDepthState = isaEvidenceDepth\(history\.rounds, noImprove, ISA_ENABLED\);/.test(src),
    'the round loop actually consults the ladder rather than deciding depth inline');
  // Executed, not regex-matched. The first version of this asserted that the string
  // `pattern_after_failed_escalation` appeared in the lane -- and it did, in the
  // COMMENT beside the code, so the mutation that deleted the rule left the check
  // green. A guard that a comment can satisfy is not a guard.
  const reached = new Function(
    grab(/const isaReachedDepth = \(requestedDepth, attribution, enabled\) => \{[\s\S]*?\n\};\n/,
      'isaReachedDepth') + '\nreturn isaReachedDepth;')();
  const someAttribution = { status: 'attributed', diagnosis: 'x' };
  ok(reached('isa', someAttribution, true) === 'isa' &&
     reached('compiler', someAttribution, true) === 'compiler',
    'a depth whose analysis returned a diagnosis is recorded as reached');
  ok(reached('isa', { status: 'inconclusive', diagnosis: 'the ISA does not explain it' }, true) === 'isa',
    'an INCONCLUSIVE attribution still counts as reaching the depth -- a plateau the evidence '
    + 'cannot explain is a real finding, and an analyst forced to produce a mechanism instead of '
    + 'admitting that is an analyst inventing one');
  ok(reached('isa', null, true) === 'pattern_after_failed_escalation' &&
     reached('compiler', null, true) === 'pattern_after_failed_escalation',
    'but a requested depth with NO attribution behind it is neither `isa` nor `pattern`: the '
    + 'first says "we looked and it did not help", the second says "we never looked", and a '
    + 'failed escalation is a false negative one level above the one this layer catches');
  ok(reached('pattern', null, true) === 'pattern' && reached('isa', someAttribution, false) === 'pattern',
    'no escalation, and the layer off, both record plain pattern');
  ok(/evidence_depth: isaEffectiveDepth/.test(src) &&
     /const isaEffectiveDepth = isaReachedDepth\(isaDepthState\.depth, isaAttribution, ISA_ENABLED\);/.test(src),
    'and the round record actually stores that computed value rather than the requested depth');
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
  ok(/const legacyImproved = !!\(winner && winner\.geomean > cumulative \* \(1 \+ MIN_IMPROVE\)\)/.test(src),
    'the suite-geomean threshold survives as the no-band fallback');
  ok(/let improved = legacyImproved;/.test(src),
    'the legacy threshold is the starting commit decision, not one branch of two');
  ok(/if \(winner && ROUTE_BANDS\) \{/.test(src) &&
     /if \(routeVerdict\.applicable\) \{/.test(src) &&
     /improved = routeVerdict\.accepted \|\| suiteSaysYes;/.test(src),
    'the per-route verdict joins the suite threshold as a union when bands exist AND it is applicable');
  // Finding (62) split this from one conjunction into a gate plus a named metric
  // refusal, so the shape changed; the threshold it enforces did not.
  ok(/says\(r\.ver\.correctness, 'pass'\)\)\) return false;/.test(src) &&
     /return primSpeedup\(r\.ver\) > CANDIDATE_FLOOR;/.test(src),
    'the verified filter still gates on correctness and the candidate floor');
}

console.log('\n# the per-route gate must be REACHABLE without a hand-maintained band file, executed');
{
  // Why this section exists. The per-route gate and its Python twin were both written, both
  // correct, and both defended by tests -- and across seven waves of the greedy lane the gate
  // logged NOTHING, because its only band source was an `args.route_bands` table nobody passed and
  // the sole file on disk was six epochs stale. A gate that cannot be fed is a gate that does not
  // run, and the suite threshold it was written to overrule went on refusing verified single-route
  // wins the whole time. So what is pinned here is not the arithmetic (test_route_gate.py owns
  // that) but the SUPPLY: bands come from data every run already produces.
  const block = grab(/const BAND_MIN_REPEATS = 3;[\s\S]*?const bandsFromSamples = \(perCase\) => \{[\s\S]*?\n\};\n/,
    'bandsFromSamples');
  const { bandsFromSamples, BAND_MIN, BAND_MIN_REPEATS } = new Function(
    `${block}\nreturn {bandsFromSamples, BAND_MIN, BAND_MIN_REPEATS};`)();

  const row = (name, samples) => ({ name, latency_ms: samples[0], samples_ms: samples });

  // The statistic, and that it agrees with the Python twin's definition: full min-max spread over
  // the median. test_route_gate.py::BandsFromSamplesTest pins the identical numbers on the other
  // side, which is the only thing keeping two hand-written implementations of one rule together.
  {
    const r = bandsFromSamples([row('a', [0.100, 0.104, 0.096])]);
    ok(r.bands != null && Math.abs(r.bands.a - (0.104 - 0.096) / 0.100) < 1e-9,
      'the band is the full spread over the median', r.reason);
  }
  // A zero spread must not become a zero band. Three identical coarse-timer reads would otherwise
  // make every rounding difference read as `improved` -- the gate banking noise on exactly the
  // routes where it is quietest, which is the opposite of what it is for.
  {
    const r = bandsFromSamples([row('a', [0.2, 0.2, 0.2])]);
    ok(r.bands != null && r.bands.a === BAND_MIN,
      'a zero spread is clamped to BAND_MIN rather than left at zero', r.reason);
  }
  ok(BAND_MIN === 0.002 && BAND_MIN_REPEATS === 3,
    'the floor constants match measure_noise_floor.py (asserted on the Python side too)');

  // Every refusal returns a REASON and never throws. `samples_ms` is optional by contract
  // (benchmark_engineer.md: "Omitting the field is not an error and the run proceeds"), so a throw
  // here would abort a run over a field the agent is allowed to leave out.
  for (const [bad, why] of [
    [[row('a', [0.1, 0.1])], 'two samples cannot define a spread'],
    [[{ name: 'a', latency_ms: 0.1 }], 'a latency with no samples_ms'],
    [[row('a', [0.1, 0.11, 0.09]), { name: 'b', latency_ms: 0.2 }], 'one route missing samples'],
    [[row('a', [0.1, 0.11, -1])], 'a negative sample'],
    [[row('a', [0.1, 0.11, Infinity])], 'a non-finite sample'],
    [[{ samples_ms: [0.1, 0.11, 0.09] }], 'a row with no route name'],
    [[row('a', [0.1, 0.11, 0.09]), row('a', [0.1, 0.11, 0.09])], 'a duplicate route'],
    [[], 'an empty table'],
  ]) {
    const r = bandsFromSamples(bad);
    ok(r.bands === null && typeof r.reason === 'string' && r.reason.length > 0,
      `refused with a reason, not a throw and not a partial table: ${why}`,
      JSON.stringify(r.bands));
  }
  // A PARTIAL table is the failure worth naming separately: it would leave `routeGate` refusing the
  // unbanded route with "no measured band for ...", which reads as a candidate defect rather than
  // as our own missing measurement.
  {
    const r = bandsFromSamples([row('a', [0.1, 0.11, 0.09]), { name: 'b', latency_ms: 0.2 }]);
    ok(r.bands === null, 'one unusable route refuses the WHOLE table, never a partial one');
  }

  // The wiring, lexically: derived by default, arg as override, and the fallback says so out loud.
  ok(/const ROUTE_BANDS_ARG = \(\(\) => \{/.test(src),
    'args.route_bands is now the OVERRIDE (ROUTE_BANDS_ARG), not the only source');
  ok(/const ROUTE_BANDS = ROUTE_BANDS_ARG \|\| ROUTE_BAND_DERIVED\.bands;/.test(src),
    'the effective band table prefers an explicit table and otherwise uses the derived one');
  ok(/bandsFromSamples\(BASELINE_PER_CASE\)/.test(src),
    'the derived table is built from the baseline the benchmark engineer just measured');
  ok(/Commit gate: SUITE GEOMEAN at MIN_IMPROVE=/.test(src),
    'a run that falls back to the suite gate LOGS that it did -- the silent-off failure this ' +
    'whole section exists to prevent');
  ok(/Commit gate: PER-ROUTE, /.test(src),
    'a run on the per-route gate says so, with the band span, so the log shows which gate decided');
}

console.log('\n# the gate compares against a SAME-SESSION control when the verifier returns one, executed');
{
  // route_gate.py's own header records this exposure and declines to guard it: "the same unchanged
  // tree measures 1.5-3% differently between invocations, and the candidate and the incumbent it is
  // compared against come from different invocations... the tighter fix is not a device check but
  // comparing against a control measured in the candidate's own session." The verifiers already
  // build that arm; until now there was no field to return it in, so the gate compared this round's
  // candidate against a table measured in an earlier round -- a drift larger than the gains judged.
  const gateBlock = grab(/const routeGate = \(candPerCase, incPerCase, bands, opts\) => \{[\s\S]*?\n\};\n/,
    'routeGate');
  const { routeGate } = new Function(`${gateBlock}\nreturn {routeGate};`)();
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
  ok(/const sameSession = !!winner\.control_per_case;/.test(src) &&
     /const incumbentSide = sameSession \? winner\.control_per_case : bestPerCase;/.test(src),
    'the gate prefers the same-session control and falls back to the stored table');
  ok(/routeGate\(winner\.per_case, incumbentSide, ROUTE_BANDS/.test(src),
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

console.log('\n# the two gates are a UNION with a regression veto, and a route win is progress, executed');
{
  // Two separate unit errors, one cause: a suite geomean was being used to judge single-route work.
  //   * at the COMMIT gate it refused verified route wins (what the band table was added to fix);
  //   * at the STALL counter it ends the WAVE -- MAX_NO_IMPROVE defaults to 2, so two rounds scored
  //     as stalls stop the loop with budget unspent. Wave 6 stopped after 3 rounds on 8 of 12.
  // The commit gate had to become a union rather than a replacement, because the per-route test has
  // the mirror-image blind spot: eleven routes each +0.4% is a real ~4.4% suite win that clears no
  // single band.
  const gateBlock = grab(/const routeGate = \(candPerCase, incPerCase, bands, opts\) => \{[\s\S]*?\n\};\n/,
    'routeGate');
  const { routeGate } = new Function(`${gateBlock}\nreturn {routeGate};`)();
  const routes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'];
  const bands = Object.fromEntries(routes.map(r => [r, 0.02]));
  const rows = (f) => routes.map(r => ({ name: r, optimized_ms: 0.100 * f(r), speedup: 1 / f(r) }));
  const inc = rows(() => 1);

  // The union's reason for existing, as arithmetic rather than assertion.
  const broad = routeGate(rows(() => 0.996), inc, bands);      // every route -0.4%
  ok(broad.applicable && !broad.accepted && !broad.regressed.length,
    'a broad thin gain clears NO band, so the per-route test alone would refuse a real suite win',
    broad.reason);
  const single = routeGate(rows(r => (r === 'a' ? 0.93 : 1)), inc, bands);
  ok(single.applicable && single.accepted,
    'a single-route win clears its band, which the suite geomean divides by the route count');

  // The veto is what is NOT unioned.
  const mixed = routeGate(rows(r => (r === 'a' ? 0.80 : (r === 'b' ? 1.10 : 1))), inc, bands);
  ok(mixed.applicable && !mixed.accepted && mixed.regressed.join() === 'b',
    'a route regressed past its band is refused however good the average looks');

  ok(/const suiteSaysYes = legacyImproved && !routeVerdict\.regressed\.length;/.test(src),
    'the suite test survives as a second route to acceptance, minus any banded regression');
  ok(/improved = routeVerdict\.accepted \|\| suiteSaysYes;/.test(src),
    'the commit decision is the UNION -- so making the per-route gate the default cannot make any ' +
    'run STRICTER than it was, which was the entire complaint against the old gate');
  ok(/committing anyway on the SUITE test/.test(src),
    'a commit that only the suite test justifies says so, rather than looking like a route win');

  // The stall counter.
  ok(/const suiteProgress = !!\(winner && bestSeen > 0 && winner\.geomean > bestSeen \* \(1 \+ PROGRESS_DELTA\)\)/.test(src),
    'the old suite progress test is preserved under its own name');
  ok(/const routeProgress = !!\(routeVerdict && routeVerdict\.applicable && routeVerdict\.improved\.length\s*&& !routeVerdict\.regressed\.length\)/.test(src),
    'a round that cleared a route band counts as progress');
  ok(/const madeProgress = suiteProgress \|\| routeProgress;/.test(src),
    'progress is the union of the two, so a route win cannot be scored as a stall');
  ok(/if \(madeProgress \|\| committedThisRound\) \{ noImprove = 0; \} else \{ noImprove\+\+; \}/.test(src),
    'the stall counter still also resets on a landed commit -- (127) unchanged');
  ok(/the search ADVANCED on route evidence/.test(src),
    'a round rescued from being scored a stall says which routes rescued it');
  // routeProgress is deliberately BROADER than the commit gate: it ignores target narrowing, because
  // the cost of a false "stalled" is every remaining round of the wave.
  ok(!/routeProgress = .*targetRoutes/.test(src),
    'progress does not require the win to be on the DECLARED route -- a stopping rule errs toward ' +
    'continuing, since a false stall costs the whole remaining wave');
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
