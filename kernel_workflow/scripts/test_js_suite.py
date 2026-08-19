#!/usr/bin/env python3
"""Run the JS regression suite under pytest, and prove it is not vacuous.

Two tests live here.

`test_qd_archive_js_passes` runs `test_qd_archive.js` on the embedded V8 host in
`run_js_tests.py`. It skips -- loudly, with the install hint -- when no engine is
present, because a silently-absent JS engine is precisely how that file went
unexecuted for as long as it did.

`test_the_js_suite_catches_a_broken_invariant` is the more important one. The JS
suite is a *lexical* guard: it asserts regexes against `kernel_lane.js` source
text. A lexical guard has a specific failure mode -- it keeps passing after the
code it watches has been renamed, refactored, or deleted, because a regex that
matches nothing and a regex that matches something both look like "no error"
unless you check which. So each invariant is deliberately broken in an in-memory
copy of the lane and the suite must go red. A mutant that survives is a check
that was watching nothing.

The mutants are single-site on purpose. Several of these rules are enforced on
BOTH the warm-import and main-round admission paths, and the first version of
this probe found that breaking only one site left the suite green: `.test()`
proves at least one site has the rule, never that every site does. That hole is
now closed by counting sites, and these mutants are what keep it closed.

Nothing here touches the GPU, the network, a model, or any file on disk: the
mutated lane exists only as a Python string.
"""
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_js_tests", SCRIPTS_DIR / "run_js_tests.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RJT = _load_runner()

# (label, original source fragment, mutated replacement). Each must break exactly
# one invariant the JS suite claims to hold.
MUTANTS = [
    ("an ambiguous baseline frame is ranked by precedence instead of refused",
     "  const disagreeing = offered.filter(([, v]) => Math.abs(v - first) > 1e-3 * first);",
     "  const disagreeing = [];"),
    ("the ambiguity refusal stops naming the numbers it is refusing",
     "`disagree -- ${offered.map(([f, v]) => `${f}=${v}`).join(', ')}. These field names do not `",
     "`disagree. These field names do not `"),
    ("an elite is written without stating which denominator its statistics used",
     "robust_baseline_frame: QD_ROBUST_BASELINE_FRAME,",
     ""),
    ("an interval is recorded without the epoch it was measured on",
     "${QD_EPOCH_STAMP}",
     ""),
    # `audit_pin_coverage.py` reported six assertions as pinned by no mutant in
    # the corpus -- i.e. green whatever the lane says. Finding (128): an
    # unexercised pin does not fail, it goes quiet, and quiet reads as covered.
    # Four were legacy; two arrived with the frame label in section 39. The
    # seven mutants below close all six, and the audit now reports zero.
    ("the classifier version silently reverts to v1",
     "const QD_CLASSIFIER_VERSION = 'geak-qd-v2';",
     "const QD_CLASSIFIER_VERSION = 'geak-qd-v1';"),
    ("the parent-rejection log stops naming which of the four reasons fired",
     "+ `QD parent; no canonical fallback -- ${gate.reason}.`);",
     "+ `QD parent; no canonical fallback.`);"),
    # Two mutants for one assertion, because it is a CONJUNCTION: the persister
    # is named AND `REQUIRE_ATOMIC_MANIFEST` is absent. The second half breaks
    # under a single edit; the first did not, because the script name appears at
    # four sites and `.test()` stays true while three of them agree and one does
    # not -- which is not a typo, it is two writers, and it is exactly the state
    # the atomicity guarantee cannot survive. The suite now counts the sites, so
    # a single rename is caught.
    ("one call site is pointed at a second, differently-named persister",
     "${WORKFLOW_DIR}/scripts/qd_persist_manifest.py",
     "${WORKFLOW_DIR}/scripts/qd_persist_manifest_v2.py"),
    ("atomicity is asked of an agent instead of done by the persister script",
     "      QD_PERSIST_SCRIPT: `${WORKFLOW_DIR}/scripts/qd_persist_manifest.py`,",
     "      REQUIRE_ATOMIC_MANIFEST: true,"),
    ("a gate-dropped direction stops saying which gate dropped it",
     "dropped by the route-priority gate -- ${priority.reason}.`);",
     "dropped.`);"),
    ("the elite frame label is flipped to the denominator the lane does not use",
     "const QD_ROBUST_BASELINE_FRAME = 'oracle';",
     "const QD_ROBUST_BASELINE_FRAME = 'parent';"),
    # Semantically inert on purpose: the pin it exercises is a COUNT of elite
    # construction sites, and what must not silently change is how many there
    # are. A reordering that the counter stops recognising is exactly the drift
    # that would let a fourth, unlabelled site slip past the label check.
    ("an elite construction site is reordered past the site counter",
     "elite_id: eliteId, cell: rc.cell,",
     "cell: rc.cell, elite_id: eliteId,"),
    # The six below are INJECTION mutants, and they close the last bucket
    # `audit_pin_coverage.py` reported: converse-absence pins, which assert that
    # a forbidden form is *absent* from the lane. A deletion mutant can never
    # exercise one -- removing code cannot make a banned string appear -- so the
    # audit used to classify every negated pin as unexercisable by construction
    # and stopped consulting the corpus. That was wrong about its own corpus:
    # `REQUIRE_ATOMIC_MANIFEST` had already shown the shape. Each mutant here
    # replaces a correct line with the specific regression its pin forbids, so
    # the pin is now demonstrated to fire rather than assumed to.
    ("the seed's interval reverts to the hardcoded zero-width literal",
     "robust: qdSeedRobust(rc.context_id)",
     "robust: { score: 1, median: 1, mad: 0, lower: 1, upper: 1 }"),
    ("archive admission re-derives the suite interval instead of consuming it",
     "      const { suiteRobust, routeCells, minCase } = r.qd_admission;",
     "      const suiteRobust = qdRobust(r.ver);\n"
     "      const { routeCells, minCase } = r.qd_admission;"),
    ("the cell guardrail is open-coded beside the shared named-reason check",
     "  const minCase = qdMinCase(ver);\n  if (minCase < QD_CELL_GUARDRAIL) {",
     "  const minCase = qdMinCase(ver);\n  if (qdMinCase(ver) < QD_CELL_GUARDRAIL) {"),
    ("historical cells are deserialized straight into the live archive",
     "  const manifestOK = !!(importSourceOK && (classifierMatches || reclassifyMode));",
     "  const manifestOK = !!(importSourceOK && (classifierMatches || reclassifyMode));\n"
     "  if (manifestOK && setup.qd_import_manifest.cells) "
     "qdArchive.cells = setup.qd_import_manifest.cells;"),
    # (124b) again, as a mutant this time: the round persist site re-gated on
    # having admitted something, which is how run 16 produced two generations of
    # measurements and a manifest still reading `generation 0`.
    ("the round persist is gated back on qdAdmissions.length",
     "    let persistedIds = [];",
     "    let persistedIds = [];\n    if (qdAdmissions.length)"),
    ("a site writes the artifact map directly instead of through the recorder",
     "      qdRecordArtifact(sourceHash, {",
     "      qdArchive.artifacts[sourceHash] = { source_hash: sourceHash };\n"
     "      qdRecordArtifact(sourceHash, {"),
    ("seed-hash provenance guard removed",
     "r.ver.seed_source_hash !== r.d.parent_source_hash ||",
     "false ||"),
    ("per-context repeated-sample requirement removed",
     "const suiteRobust = qdRobust(ver);\n  if (!suiteRobust) {",
     "const suiteRobust = qdRobust(ver) || {score: 1};\n  if (false) {"),
    ("a measurement fault is reported as if it were a slow kernel",
     "return { reason: 'measurement:suite_interval_unavailable"
     "(a harness context has fewer than 3 usable samples)' };",
     "return { reason: `performance:worst_context_speedup_0_below_cell_guardrail"
     "_${QD_CELL_GUARDRAIL}` };"),
    ("one admission path stops using the shared named-reason check",
     "const admission = qdAdmissionCheck(",
     "const admission = ((v, rc) => ({suiteRobust: qdRobust(v)}))("),
    ("warm import trusts historical route descriptors",
     "const routeCells = qdRouteCells(ver, rejects);",
     "const routeCells = qdRouteCells(proposed);"),
    ("non-overlap replacement rule weakened to a bare score comparison",
     "const replacement = !incumbent || robust.lower > incumbent.robust.upper;",
     "const replacement = !incumbent || robust.score > incumbent.robust.score;"),
    ("near-boundary challengers stop being recorded",
     "if (!current || robust.upper > current.robust.upper) "
     "qdArchive.challengers[rc.cell] = entry;",
     "if (false) qdArchive.challengers[rc.cell] = entry;"),
    ("canonical promotion drops the stricter guardrail",
     "e.min_case < QD_CANONICAL_GUARDRAIL) continue",
     "false) continue"),
    ("admission noise floor removed, leaving a bare 2*MAD radius",
     "const radius = Math.max(2 * mad, Math.abs(median) * qdNoiseFloor(contextId));",
     "const radius = 2 * mad;"),
    ("noise floor silently defaults to zero for an unmeasured route",
     "const QD_DEFAULT_NOISE_FLOOR = Math.max(\n"
     "  ...[...QD_NOISE_FLOOR_BY_MACHINE.values()].flatMap((t) => [...t.values()]));",
     "const QD_DEFAULT_NOISE_FLOOR = 0;"),
    # The default is deliberately the widest floor measured on ANY machine. This
    # mutant narrows it to the widest on the CURRENT machine, which is the exact
    # slip the machine-keyed table invites: on machine N that is 0.0378 against a
    # true cross-machine 0.072, so an unmeasured route -- the one carrying the
    # least information about its own spread -- would get an interval half as
    # wide as the widest thing this project has actually measured.
    ("an unmeasured route borrows the current machine's spread instead of the widest",
     "const QD_DEFAULT_NOISE_FLOOR = Math.max(\n"
     "  ...[...QD_NOISE_FLOOR_BY_MACHINE.values()].flatMap((t) => [...t.values()]));",
     "const QD_DEFAULT_NOISE_FLOOR = Math.max(...QD_NOISE_FLOOR.values());"),
    # NOT here: "the lane applies a different machine's floors than the analysis
    # module". That mutant lived in this list and appeared to be caught, but the
    # JS suite only sees the lane, and every machine in the table is a valid lane
    # on its own terms -- what actually failed was `test_qd_archive.js` having
    # hard-coded machine N's floor values as literals. When the container moved
    # to machine O that third copy went red and named the wrong cause. Which
    # epoch is correct is a cross-FILE fact, so the probe now lives beside the
    # cross-file check, in
    # test_qd_lane_parity.py::NoiseFloorParityTest::
    # test_the_epoch_pin_check_actually_goes_red_on_a_mismatched_lane.
    # (108)/(110). `variant_spread` is the only window in which the planner can
    # tell "12 cells, 12 mechanisms" from "12 cells, 2 mechanisms copied around"
    # -- the second being the exact failure this QD redesign exists to end. It is
    # not a gate, so nothing downstream refuses anything on it; if the number is
    # silently wrong the planner reads a diverse archive that is not one, and no
    # other check ever disagrees. These four mutate its DECISION logic (which
    # variant is the top one, what divides what, what counts as a variant) and
    # not its packaging, per (110).
    ("the most-copied variant's share is reported as the least-copied one's",
     "const counts = [...perHash.values()].sort((a, b) => b - a);",
     "const counts = [...perHash.values()].sort((a, b) => a - b);"),
    ("concentration is reported as a share of variants rather than of cells",
     "top_variant_share: elites.length ? counts[0] / elites.length : 0,",
     "top_variant_share: perHash.size ? counts[0] / perHash.size : 0,"),
    ("cells-per-variant inverts into variants-per-cell",
     "cells_per_variant: perHash.size ? elites.length / perHash.size : 0,",
     "cells_per_variant: elites.length ? perHash.size / elites.length : 0,"),
    ("a cell with no source hash invents a variant of its own",
     "if (!e || !e.source_hash) continue;",
     "if (!e) continue;"),
    ("one (cell, context) may be claimed twice per candidate",
     "if (claimed.has(cell)) {",
     "if (false) {"),
    ("SOL cards claiming the kernel beat its own speed of light are accepted",
     "if (c.sol_gap < 1 - 1e-9) return false;",
     ""),
    ("qd_score silently falls back to the suite geomean for a robust-less cell",
     "if (!e.robust || typeof e.robust.median !== 'number') {",
     "if (false) {\n      e.robust = e.robust || {median: e.geomean || 0};"),
    ("the rounds law stops refusing a crossing from one round to two",
     "if (receipt.candidate.rounds > Math.max(1, receipt.current.rounds)) {",
     "if (false) {"),
    ("a residency receipt no longer has to agree with its own arithmetic",
     "if (rounds !== Math.ceil(ctas / slots)) {",
     "if (false) {"),
    ("the two directions of the reduction/fixup coupling share one token",
     "if (fixup && !reduction) return 'rule:fixup_without_reduction';",
     "if (fixup && !reduction) return 'rule:reduction_without_fixup';"),
    ("cell identity drops the workload context prefix",
     "const qdCellId = (contextId, d) => {",
     "const qdCellId = (contextId, d) => { contextId = 'x';"),
    # Finding (62). The correctness gate and the primary-metric selector.
    ("the correctness gate goes back to a bare leading-word match",
     "String(v == null ? '' : v).trim().toLowerCase().startsWith(w) && !saysContradicted(v);",
     "String(v == null ? '' : v).trim().toLowerCase().startsWith(w);"),
    ("a partial pass fraction stops contradicting the word PASS",
     "for (let m = frac.exec(s); m; m = frac.exec(s)) if (Number(m[1]) !== Number(m[2])) return true;",
     "for (let m = frac.exec(s); m; m = frac.exec(s)) if (false) return true;"),
    ("archive admission stops requiring the weighted primary metric",
     "const metricReason = primMetricReason(ver);\n  if (metricReason) return { reason: metricReason };",
     "const metricReason = null;\n  if (metricReason) return { reason: metricReason };"),
    # (67) added the oracle pin to this same line; the mutant still targets the metric
    # half, which is what it was written to defend.
    ("the greedy candidate filter stops checking which quantity it is gating on",
     "const metricReason = primMetricReason(r.ver) || oracleDrift(r.ver) || qdPolicyReject(r.ver)",
     "const metricReason = null || oracleDrift(r.ver) || qdPolicyReject(r.ver)"),
    ("the integrated result is compared against candidates in whatever unit it arrives in",
     "const integMetricReason = integMetric ? primMetricReason(integMetric) : null;",
     "const integMetricReason = null;"),
    # Finding (63). The third admission copy.
    ("the archive-writing loop re-derives admission instead of consuming it",
     "const { suiteRobust, routeCells, minCase } = r.qd_admission;",
     "const suiteRobust = qdRobust(r.ver), routeCells = qdRouteCells(r.ver),\n"
     "        minCase = qdMinCase(r.ver);"),
    ("a candidate may reach archive admission without passing the filter that decides it",
     "if (!r.qd_admission) {",
     "if (false) {"),
    ("the stored worst-case speedup stops being the one the gate ran on",
     "return { suiteRobust, minCase };",
     "return { suiteRobust, minCase: 1 };"),
    # Finding (64). The persistence receipt.
    ("a missing persistence receipt is silently treated as an empty one",
     "  if (reported === null) {",
     "  if (false) {"),
    ("the persistence receipt is trusted for ids that were never admitted",
     "return (reported || []).filter(id => admittedIds.has(id));",
     "return (reported || []);"),
    ("rolling a won cell back out of the archive stops being logged",
     "        qdLogRollback(entry, `round ${round}`);",
     ""),
    ("the warm-import path stops validating its persistence receipt",
     "qdPersistenceReceipt(persistedImports, admissions, 'warm import')",
     "(persistedImports && persistedImports.persisted_elite_ids) || []"),
    # Finding (65). The run's only independent arbitration.
    ("a dead arbiter's absence is reported as a result rather than as absent arbitration",
     "  if (!v) {\n    return { trust: 'unverified', basis: 'unknown',",
     "  if (!v) {\n    return { trust: 'verified', basis: 'device',"),
    ("the validation verdict stops reading the arbiter's correctness field",
     "if (v.correctness != null && String(v.correctness).trim() !== '' && !says(v.correctness, 'pass')) {",
     "if (false) {"),
    ("a status the arbiter did not accept stops being flagged",
     "if (!says(v.validation_status, 'accept') && !says(v.validation_status, 'validated')) {",
     "if (false) {"),
    ("an unlabelled or host-bound ratio is accepted as a device-time speedup",
     "if (basis !== 'device' && basis !== 'device_time') {",
     "if (false) {"),
    ("a missing timing basis defaults to clean instead of unknown",
     "? v.timing_basis.trim().toLowerCase() : 'unknown';",
     "? v.timing_basis.trim().toLowerCase() : 'device';"),
    ("a refused verdict publishes its number anyway",
     "const finalGeomean = trusted ? validation.director_verified_speedup_geomean : null;",
     "const finalGeomean = validation ? validation.director_verified_speedup_geomean : cumulative;"),
    ("the lane stops handing the bake-off its verdict",
     "  validation_trust: verdict.trust,",
     "  validation_trust: 'verified',"),
    ("timing_basis leaves the validate schema again",
     "  timing_basis: { type: 'string' },",
     ""),
    # Finding (67). The oracle pin: the denominator's provenance.
    ("a moved oracle becomes a per-candidate skip instead of failing the run closed",
     "    throw new Error(\n      `oracle:digest_drift",
     "    return (\n      `oracle:digest_drift"),
    ("a report with no oracle digest is treated as if it had matched",
     "  if (!got) {\n    return 'oracle:digest_missing",
     "  if (false) {\n    return 'oracle:digest_missing"),
    ("archive admission stops pinning the denominator of what it stores",
     "const oracleReason = oracleDrift(ver);",
     "const oracleReason = null;"),
    ("the greedy candidate filter stops pinning the denominator",
     "primMetricReason(r.ver) || oracleDrift(r.ver) || qdPolicyReject(r.ver)",
     "primMetricReason(r.ver) || null || qdPolicyReject(r.ver)"),
    ("a run that never pinned its oracle can still be published as verified",
     "  if (!oraclePinned) {",
     "  if (false) {"),
    ("the final arbitration stops being held to the denominator it arbitrates",
     "const oracleFinal = validation ? oracleDrift(validation) : null;",
     "const oracleFinal = null;"),
    # Finding (68). The route-priority gate: is this route worth a build at all?
    ("a closed route survives the filter and gets a build",
     "    if (unmeasured || derived !== 'closed' || !qdFloorIsMeasured(context)) "
     "kept.push(context);",
     "    kept.push(context);"),
    # Finding (92). The opposite mutation of the one above, and it needs its own
    # probe: dropping an UNMEASURED route is the failure that hides itself, so a
    # suite that only watches for closed routes getting through would call the
    # regression a tightening.
    ("an unmeasured route is dropped as though it had been measured and closed",
     "    if (unmeasured || derived !== 'closed' || !qdFloorIsMeasured(context)) "
     "kept.push(context);",
     "    if (derived !== 'closed') kept.push(context);"),
    # Finding (126). The third way to break the same line, and the one the
    # current epoch cannot see at all: a route whose floor is a fail-closed
    # placeholder is dropped as though the closure had been measured on this
    # box. On a provisional epoch this mutation removes the ONLY thing keeping
    # the branch alive, so the suite's cover for it has to come from the
    # measured-epoch instance in test_qd_archive.js rather than from the pin.
    ("a placeholder floor is treated as a measurement and closes the route",
     "    if (unmeasured || derived !== 'closed' || !qdFloorIsMeasured(context)) "
     "kept.push(context);",
     "    if (unmeasured || derived !== 'closed') kept.push(context);"),
    ("the filter refuses a mixed list instead of dropping just the closed entries",
     "  if (!kept.length) return { reason: `priority:all_targets_closed",
     "  if (kept.length !== wanted.length) return { reason: `priority:all_targets_closed"),
    ("the receipt's own verdict is trusted instead of re-derived",
     "    if (claimed !== derived) {",
     "    if (false) {"),
    ("a receipt may borrow a friendlier route's noise floor",
     "    if (Math.abs(floor - own) > 1e-6 * Math.max(1, own)) {",
     "    if (false) {"),
    ("the gate stops requiring the receipt to cover the direction's target cases",
     "  if (uncovered.length) return { reason: `priority:uncovered",
     "  if (false) return { reason: `priority:uncovered"),
    ("an absent route-priority receipt is treated as a pass",
     "  if (!receipt || !Array.isArray(receipt.per_context)) return { reason: 'priority:receipt_absent' };",
     "  if (!receipt) return { cases: wanted };"),
    ("the closed threshold is loosened off the definition",
     "const QD_CLOSED_RATIO = 1.0;",
     "const QD_CLOSED_RATIO = 0.5;"),
    ("the gate runs but its filtered list is discarded",
     "      ...d, target_cases: priorityCases, operator: op,",
     "      ...d, operator: op,"),
    ("the gate is never invoked on any direction",
     "      const priority = qdPriorityFilter(d.priority_receipt, d.target_cases || [d.context_id]);",
     "      const priority = { cases: d.target_cases || [d.context_id] };"),
    ("priority_receipt leaves the planning schema again",
     "      priority_receipt: obj({",
     "      unused_receipt_: obj({"),
    # Finding (69). The mutants below each restore one form of "take the boolean
    # at its word", which is the state the whole project was in until now.
    ("an unbacked policy_pass is admitted again",
     "  if (!s || typeof s !== 'object') {",
     "    if (false) {"),
    ("a scan that opened no file passes trivially again",
     "  if (!Number.isFinite(s.files) || s.files < 1) {",
     "  if (false) {"),
    ("a post-build scan that saw no binary counts as a post-build scan",
     "  if (!Number.isFinite(s.elf) || s.elf < 1) {",
     "    if (false) {"),
    ("a receipt is allowed to contradict the boolean beside it",
     "  if (s.passed !== true) {",
     "    if (false) {"),
    ("passing with findings is no longer a contradiction",
     "  if (!Number.isFinite(s.findings) || s.findings !== 0) {",
     "    if (false) {"),
    ("uninspectable artifacts stop failing closed",
     "  if (Number.isFinite(s.unreadable) && s.unreadable > 0) {",
     "    if (false) {"),
    ("any tool's summary is accepted as a policy receipt",
     "  if (typeof s.schema === 'string' && s.schema && !/policy/i.test(s.schema)) {",
     "    if (false) {"),
    ("the gate starts speaking about honest failures too, renaming them",
     "  if (!rep || rep.policy_pass !== true) return null;",
     "  if (!rep) return null;"),
    ("the canonical seed stops being checked",
     "  const seedPolicyReason = qdPolicyReject(seed, 'canonical seed');",
     "  const seedPolicyReason = null;"),
    ("the archive admission path stops checking it",
     "  const policyReason = r.ver ? (qdPolicyReject(r.ver) || qdTwinReject(r.ver)) : null;",
     "    const policyReason = null;"),
    # Finding (87). Its own probe, because the twin gate and the policy gate now
    # share a line: a mutation that removes only the twin half leaves the policy
    # probe above green.
    ("the archive admission path stops checking the hipify twin",
     "  const policyReason = r.ver ? (qdPolicyReject(r.ver) || qdTwinReject(r.ver)) : null;",
     "  const policyReason = r.ver ? qdPolicyReject(r.ver) : null;"),
    ("the greedy candidate filter stops checking the hipify twin",
     "      || qdTwinReject(r.ver);",
     "      || null;"),
    ("a re-verified imported snapshot stops being checked for twin drift",
     "        ? (qdPolicyReject(ver, 'qd import verify') || qdTwinReject(ver, 'qd import verify'))",
     "        ? qdPolicyReject(ver, 'qd import verify')"),
    ("integrate stops checking policy",
     "      ? (qdPolicyReject(integrate, 'integrate') || qdTwinReject(integrate, 'integrate')) : null;",
     "      ? qdTwinReject(integrate, 'integrate') : null;"),
    ("integrate stops checking the hipify twin",
     "      ? (qdPolicyReject(integrate, 'integrate') || qdTwinReject(integrate, 'integrate')) : null;",
     "      ? qdPolicyReject(integrate, 'integrate') : null;"),
    # Exit 3 is the newest of the tool's four codes, and the one a stale
    # consumer is likeliest to get wrong: it says the pair matched everywhere
    # the checker could read, which reads like a pass and is not one.
    # Exit 3 is the newest of the tool's four codes and the one a consumer is
    # likeliest to get wrong: it says the pair matched everywhere the checker
    # could read, which reads like a pass and is not one. Note the mutation has
    # to return null outright -- merely deleting the branch leaves the
    # `exit_code !== 0` fallback to refuse it, which is the fail-shut behaviour
    # working, so that weaker mutant would not be a defect to catch.
    ("an unreadable launch statement is accepted as lockstep",
     "  if (s.exit_code === 3) {",
     "  if (s.exit_code === 3) { return null; }\n  if (false) {"),
    ("an unevidenced precondition is accepted",
     "    if (!evidence) {",
     "    if (false) {"),
    ("the precondition kind vocabulary stops being closed",
     "    if (!PRECONDITION_KINDS.includes(kind)) {",
     "    if (false) {"),
    ("a precondition that flips mid-run overwrites the first",
     "      if (prior.statement !== statement) {",
     "      if (false) {"),
    ("the planner stops being shown the arch preconditions",
     "    preconditions: { arch: QD_ARCH, records: qdArchive.preconditions[QD_ARCH] || [] },",
     "    "),
    ("the bootstrap stops recording preconditions",
     "  qdRecordPreconditions(QD_ARCH, seed.preconditions, 'canonical seed bootstrap');",
     "  void 0;"),
    ("the roadmap profile keeps its ungrounded SOL fields",
     "profileSummary = profileSolStrip(profileSummary, 'baseline profile');",
     "profileSummary = profileSummary;"),
    ("the reprofile branch stops being stripped",
     "    profileSummary = profileSolStrip(profileSummary, `reprofile r${round}`);",
     "    profileSummary = profileSummary;"),
    ("the strip stops reaching into key_metrics",
     "  scrub(rep.key_metrics, 'key_metrics.');",
     "  void 0;"),
    ("the strip removes the fields but stops reporting them at all",
     "      if (SOL_SHAPED_KEY.test(k)) { dropped.push(prefix + k); delete o[k]; }",
     "      if (SOL_SHAPED_KEY.test(k)) { delete o[k]; }"),
    ("the strip stops saying what it removed",
     "    rep.sol_note = 'SOL-shaped fields were removed: the roadmap profile has no ceiling '",
     "    rep.sol_note = undefined && ("),
    ("the summary leaves the report schemas",
     "  policy_postbuild: POLICY_SUMMARY_SCHEMA,",
     "  unused_postbuild_: POLICY_SUMMARY_SCHEMA,"),
    # (70) -- the SOL bandwidth-ceiling provenance gate.
    ("a card with no ceiling provenance is admitted again",
     "  if (!k || typeof k !== 'object') {",
     "  if (false) {"),
    ("a clamped ceiling may set the denominator again",
     "  if (memoryBinding && QD_SOL_BACKED.indexOf(k.confidence) < 0) {",
     "  if (false) {"),
    ("the gate stops caring which floor actually binds, so it cries wolf",
     "    && c.memory_floor_ms >= c.compute_floor_ms;",
     "    && true;"),
    ("an unmeasured reference peak counts as measured",
     "const QD_SOL_BACKED = ['measured_interpolated', 'measured_scalar'];",
     "const QD_SOL_BACKED = ['measured_interpolated', 'measured_scalar', 'unmeasured', 'low'];"),
    ("a confidence word from another vocabulary is accepted",
     "  if (QD_SOL_CONFIDENCE.indexOf(k.confidence) < 0) {",
     "  if (false) {"),
    ("a clamped read may claim a measured confidence",
     "  if (k.extrapolated && k.confidence !== 'low') {",
     "  if (false) {"),
    ("a footprint_table ceiling stops needing a footprint",
     "    if (!Number.isFinite(k.footprint_bytes) || k.footprint_bytes <= 0) {",
     "    if (false) {"),
    ("the reported bracket stops having to contain the footprint",
     "    if (!k.extrapolated && (k.footprint_bytes < b[0] || k.footprint_bytes > b[1])) {",
     "    if (false) {"),
    # (84), converted from a counted backlog into probes. Each of these lines
    # was pinned in `test_qd_archive.js` by matching its text and by nothing
    # else, which is finding (82)'s shape: green while the expression is wrong,
    # red when you fix it. `audit_pin_coverage.py` is what found them -- a pin
    # is load-bearing exactly when some mutation of the source flips it, so the
    # unflippable pins are the list, computed rather than guessed. The wrong
    # form chosen for each is the plausible one, not an arbitrary break.
    ("a route that failed classification still occupies a cell",
     "    if (route.classification_status !== 'classified') {",
     "    if (false) {"),
    ("opening a cell is folded back into `improved`",
     "            improved: replacement && !!incumbent && robust.score > incumbent.robust.score,",
     "            improved: replacement,"),
    ("a beaten incumbent is reported as an opening",
     "            opened_empty_cell: replacement && !incumbent,",
     "            opened_empty_cell: replacement,"),
    ("a candidate that yields no archive cell is admitted anyway",
     "    if (!routeCells.length) {",
     "    if (false && !routeCells.length) {"),
    ("the route rejects are collected but never asked for",
     "    const routeCells = qdRouteCells(r.ver, rejects);",
     "    const routeCells = qdRouteCells(r.ver);"),
    ("the SOL card stops having to be about the selected context",
     "        if (!card || card.selected_cell !== s.selected_cell || card.context_id !== s.context_id ||",
     "        if (!card || card.selected_cell !== s.selected_cell || false ||"),
    ("a selected parent no longer has to be the live cell's source",
     "          parent.source_hash === s.parent_source_hash && parent.snapshot && parent.artifact",
     "          parent.snapshot && parent.artifact"),
    # These six sit in `qdParentReject`, which was inline in the direction map
    # until `audit_pin_coverage.py` reported that the only thing watching it was
    # a regex on its log message. Two of them are the old pair, re-pointed at
    # the extracted form; the other four became worth probing only once the
    # refusals were executed rather than grepped.
    ("a crossover may use the same source twice",
     "      new Set([parent.source_hash, ...donors.map(e => e.source_hash)]).size === donors.length + 1;",
     "      true;"),
    ("a directed transition may target any descriptor",
     "    if (!legal.some(n => qdDescriptorSame(n.descriptor, d.target_descriptor))) {",
     "    if (false) {"),
    ("a direction may name a parent build the cell no longer holds",
     "  if (d.parent_source_hash !== parent.source_hash) {",
     "  if (false) {"),
    ("any SOL card for the cell will do, whichever parent and context it measured",
     "    c.context_id === selection.context_id && c.parent_source_hash === selection.parent_source_hash);",
     "    true);"),
    ("a directed transition may target the parent's own descriptor",
     "    if (d.target_descriptor && qdDescriptorSame(d.target_descriptor, parent.descriptor)) {",
     "    if (false) {"),
    ("a selection that resolved to no elite is treated as unparented rather than refused",
     "  if (!parent) return { reason: `parent:selection_carries_no_parent(${selection.selected_cell})` };",
     "  if (!parent) return { selection, parent: null, card: null, donors: [] };"),
    ("the repeated-local-mutation block needs one more failure",
     "    if (QD_ENABLED && priorLocal >= 2 && (op === 'local_mutation' || op === 'parameter_tuning')) {",
     "    if (QD_ENABLED && priorLocal >= 3 && (op === 'local_mutation' || op === 'parameter_tuning')) {"),
    # Roadmap item 2b's acceptance criterion, mutated. Until the suite could be
    # run there was no point writing these; (78) had to hold the item open as a
    # labelled hole because the live budget-2 run put both parents in one round
    # and therefore had no generation N+2 to check against.
    ("the capsule key goes back to the parent's source hash, so the negative "
     "memory evaporates on every elite replacement",
     "      ? `${parent.route_id || parent.context_id}|${parent.context_id}|${mechanism}` : '';",
     "      ? `${parent.source_hash}|${mechanism}` : '';"),
    ("the repeat block stops covering the parameter-tuning relabel of the same direction",
     "    if (QD_ENABLED && priorLocal >= 2 && (op === 'local_mutation' || op === 'parameter_tuning')) {",
     "    if (QD_ENABLED && priorLocal >= 2 && op === 'local_mutation') {"),
    ("the repeat block widens to every operator, choking off deep exploration "
     "of a route a local mutation happened to fail on twice",
     "    if (QD_ENABLED && priorLocal >= 2 && (op === 'local_mutation' || op === 'parameter_tuning')) {",
     "    if (QD_ENABLED && priorLocal >= 2) {"),
    ("the ledger leaves the planner's window again, so a mechanism survives only "
     "as long as its cell's elite does",
     "    capsules: Object.fromEntries(Object.entries(qdArchive.capsules).map(([k, v]) => [k, {\n"
     "      attempts: v.length, unimproved_local: v.filter(x =>\n"
     "        x.operator === 'local_mutation' && x.improved !== true && "
     "x.regime_changed !== true).length,\n"
     "      recent: v.slice(-4),\n"
     "    }])),",
     "    capsules: Object.keys(qdArchive.capsules).length,"),
    ("the capsule count is capped along with the list, so a truncated history "
     "reads as a complete one",
     "      attempts: v.length, unimproved_local: v.filter(x =>",
     "      attempts: Math.min(v.length, 4), unimproved_local: v.slice(-4).filter(x =>"),
    ("a capsule entry keeps only its outcome flags, so the planner can see THAT "
     "a direction was tried and never WHAT it predicted",
     "            capsule: r.d.strategy_capsule || {},",
     "            capsule: {},"),
    # Four holes the pin-coverage audit surfaced only once it learned to score
    # pins written against a `grab`bed slice. Every one of them is a (76) read
    # path: the archive keeps the data and the projection stops handing it over,
    # which no other mutant here touches because the writers are all intact.
    ("qdSummary drops lineage, so the archive still records where each elite "
     "came from and no role can read it",
     "    lineage: qdArchive.lineage,\n", ""),
    # (108) exactly reversed: the swap the finding refused, as a mutant.
    ("qdSummary reports structural_coverage without variant_spread, so twelve "
     "mechanisms and two re-filed ones read as the same number",
     "    variant_spread: qdVariantSpread(qdArchive.cells),\n", ""),
    ("qdSummary reports variant_spread INSTEAD of structural_coverage -- (108) "
     "asked for a second number, not a different one",
     "structural_coverage: Object.keys(qdArchive.cells).length,\n", "\n"),
    ("the update_qd_memory task stops naming the archive fields it wants "
     "distilled, so the tech lead is asked to summarise a window it was never "
     "told the shape of",
     "transition lessons from QD_ARCHIVE.recent_transitions, and ' +\n"
     "        'strategy-capsule outcomes from QD_ARCHIVE.capsules.'",
     "transition lessons, and ' +\n        'strategy-capsule outcomes.'"),
    # An INJECTION mutant, and the reason the audit distinguishes those: a
    # corpus that only removes correct forms can never exercise a `ok(!/.../)`
    # pin, so the two converse-absence pins at test_qd_archive.js:2245 were
    # unexercised by construction until this existed. (71): filing preconditions
    # under a descriptor key makes one identity serve as two slots, and "failed
    # twice on this route" stops being readable.
    ("preconditions are filed under a descriptor key instead of the arch",
     "  const bucket = qdArchive.preconditions[arch] "
     "|| (qdArchive.preconditions[arch] = []);",
     "  const capsule_key = qdDescriptorKey(arch);\n"
     "  const bucket = qdArchive.preconditions[capsule_key] "
     "|| (qdArchive.preconditions[capsule_key] = []);"),
    ("the planner may overspend by the cost of one direction",
     "    if (plannedCost + d.cost > remaining) {",
     "    if (plannedCost > remaining) {"),
    # (124b). The regression this restores is the one the lane shipped with for
    # sixteen runs: gate the round persist on the admission list and every
    # refutation is lost at process exit, while the run itself looks fine.
    ("the round persist site goes back to being gated on admissions",
     "    {\n      const ledgerOnly = !qdAdmissions.length;",
     "    if (qdAdmissions.length) {\n      const ledgerOnly = false;"),
    ("a candidate wins the round without clearing MIN_IMPROVE",
     "  const legacyImproved = !!(winner && winner.geomean > cumulative * (1 + MIN_IMPROVE));",
     "  const legacyImproved = !!(winner && winner.geomean > cumulative);"),
    ("an incorrect candidate reaches the archive",
     "          says(r.ver.status, 'verified') && says(r.ver.correctness, 'pass'))) return false;",
     "          says(r.ver.status, 'verified'))) return false;"),
    ("explicit reclassification stops being explicit",
     "  const reclassifyMode = !!(importSourceOK && QD_RECLASSIFY);",
     "  const reclassifyMode = !!importSourceOK;"),
    ("the verifier is no longer told whether this is a reclassification",
     "          QD_RECLASSIFY: reclassifyMode ? '1' : '0', QD_REPEAT_MEASUREMENTS: 3,",
     "          QD_REPEAT_MEASUREMENTS: 3,"),
    ("a crossover is costed as a cheap local mutation",
     "      cost: QD_ENABLED && (op === 'deep_mutation' || op === 'semantic_crossover') ? DEEP_COST :",
     "      cost: QD_ENABLED && (op === 'deep_mutation') ? DEEP_COST :"),
    ("a refused validation verdict is treated as trusted",
     "const trusted = verdict.trust !== 'unverified' && verdict.trust !== 'refused';",
     "const trusted = verdict.trust !== 'unverified';"),
    ("the warm-import path stops running the policy scan",
     "        ? (qdPolicyReject(ver, 'qd import verify') || qdTwinReject(ver, 'qd import verify'))",
     "        ? qdTwinReject(ver, 'qd import verify')"),
    ("a second artifact write silently overwrites the first",
     "  const existing = qdArchive.artifacts[hash];",
     "  const existing = null;"),
    ("the parent SOL case is matched by position instead of by context",
     "        const beforeCase = r.d.sol_card && (r.d.sol_card.cases || []).find(c =>",
     "        const beforeCase = r.d.sol_card && (r.d.sol_card.cases || [])[0] && (c =>"),
    # (89) item 2 -- the compute-ceiling witness.
    ("an unwitnessed compute peak becomes a denominator again",
     "  } else if (computeBinding) {",
     "  } else if (false) {"),
    ("the witness gate arms on memory-bound cases too, failing a whole arch shut",
     "    && c.compute_floor_ms > c.memory_floor_ms;",
     "    && c.compute_floor_ms >= 0;"),
    ("an absent witness block is waved through",
     "  if (!cc || typeof cc !== 'object' || typeof cc.witnessed !== 'boolean') {",
     "  if (false) {"),
    ("witnessed:true stops having to name anybody",
     "    if (!String(cc.witness || '').trim()) {",
     "    if (false) {"),
    ("a witness that outran the ceiling stops being caught",
     "    if (!Number.isFinite(cc.attainment) || cc.attainment <= 0 || cc.attainment > 1 + 1e-9) {",
     "    if (false) {"),
    ("the witness block leaves the case schema",
     "  ceiling: SOL_CEILING_SCHEMA, compute_ceiling: SOL_COMPUTE_CEILING_SCHEMA,",
     "  ceiling: SOL_CEILING_SCHEMA,"),
    ("the witness block stops being required at the agent boundary",
     "'confidence', 'ceiling', 'compute_ceiling']);",
     "'confidence', 'ceiling']);"),
    ("the card admission site stops checking the denominator",
     "          const why = qdSolCeilingReject(c, s.context_id);",
     "          const why = null;"),
    ("an inherited denominator stops being re-checked",
     "          ? qdSolCeilingReject(beforeCase, rc.context_id) : 'sol:ceiling_absent(no parent card)';",
     "          ? null : null;"),
    ("sol_resolved is published on an unbacked ceiling again",
     "          sol_resolved: (afterGap == null || beforeCeilingReason) ? null : {",
     "          sol_resolved: (afterGap == null) ? null : {"),
    ("the ceiling stops travelling with the number it produced",
     "            ceiling: beforeCase.ceiling,",
     "            ceiling_unused_: beforeCase.ceiling,"),
    ("the ceiling block leaves the case schema",
     "  ceiling: SOL_CEILING_SCHEMA,",
     "  ceiling_unused_: SOL_CEILING_SCHEMA,"),
    # (71) -- content-addressed provenance and the no-op patch.
    ("a later write of the same content address wins again",
     "  const existing = qdArchive.artifacts[hash];\n  if (!existing) {",
     "  const existing = null;\n  if (!existing) {"),
    ("a provenance disagreement stops being visible",
     "  if (differs) {",
     "  if (false) {"),
    ("parent_workspace stops counting as provenance",
     "    (existing.parent_workspace || null) !== (record.parent_workspace || null);",
     "    false;"),
    # These three keep the JS valid on purpose -- a mutant killed by a syntax
    # error proves nothing about the check that was supposed to catch it. Each
    # one leaves the site working but stops it being the site the wiring check
    # enumerates, which is the regression that would actually happen.
    ("the canonical seed stops being an enumerated artifact writer",
     "  qdRecordArtifact(seed.source_hash, {",
     "  qdRecordArtifact(String(seed.source_hash), {"),
    ("the warm import stops being an enumerated artifact writer",
     "          qdRecordArtifact(ver.source_hash, {",
     "          qdRecordArtifact(String(ver.source_hash), {"),
    ("round admission stops being an enumerated artifact writer",
     "      qdRecordArtifact(sourceHash, {",
     "      qdRecordArtifact(String(sourceHash), {"),
    ("a patch that changed nothing is admitted to the archive again",
     "    if (r.ver && qdHashValid(r.ver.source_hash) && r.ver.source_hash === r.ver.seed_source_hash) {",
     "    if (false) {"),
    # (84), second pass. The first pass audited only the leading clause of each
    # `ok(a && b && !c)`; a third of the patterns in the suite live in the
    # trailing conjuncts and had never been exercised. These probe the ones
    # whose wrong form is a real defect rather than a renamed literal.
    ("a historical fitness score is handed to the re-verifier as trusted",
     "REQUIRE_CURRENT_HARNESS_REVERIFY: '1', HISTORICAL_SCORE_TRUSTED: 'false',",
     "REQUIRE_CURRENT_HARNESS_REVERIFY: '1', HISTORICAL_SCORE_TRUSTED: 'true',"),
    ("an imported elite is admitted without re-measuring on the current harness",
     "REQUIRE_CURRENT_HARNESS_REVERIFY: '1', HISTORICAL_SCORE_TRUSTED: 'false',",
     "REQUIRE_CURRENT_HARNESS_REVERIFY: '0', HISTORICAL_SCORE_TRUSTED: 'false',"),
    ("an imported elite's snapshot points at a path keyed by elite id, not content",
     "entry.snapshot = `${qdArchive.archive_dir}/artifacts/${entry.source_hash}/workspace`;",
     "entry.snapshot = `${qdArchive.archive_dir}/artifacts/${entry.elite_id}/workspace`;"),
    ("a round-admitted cell references its artifact by elite id instead of source hash",
     "artifact: sourceHash, geomean: primSpeedup(r.ver),",
     "artifact: eliteId, geomean: primSpeedup(r.ver),"),
    ("canonical promotion fetches the baseline patch from an id-keyed path",
     "patch: `${qdArchive.archive_dir}/artifacts/${e.source_hash}/baseline.patch`,",
     "patch: `${qdArchive.archive_dir}/artifacts/${e.elite_id}/baseline.patch`,"),
    ("only one side of the run declares the oracle digest, so nothing cross-checks it",
     "oracle_digest: { type: 'string' },   // (67): the arbiter's own recomputation of the pinned oracle",
     ""),
    ("the hipify twin receipt is stripped from one of the verdict schemas",
     "hip_twin_sync: HIP_TWIN_SCHEMA,            // (87)",
     ""),
    ("a round-admitted snapshot is keyed by elite id instead of source hash",
     "const artifactSnapshot = `${qdArchive.archive_dir}/artifacts/${sourceHash}/workspace`;",
     "const artifactSnapshot = `${qdArchive.archive_dir}/artifacts/${eliteId}/workspace`;"),
    # The ISA mechanism gate. The first probe replaces an earlier language-whitelist
    # one: arming this layer must depend on the mode ALONE. Re-introducing a
    # TARGET_LANGUAGE condition is the specific regression to catch, because it fails
    # silently -- that argument defaults to 'triton' whenever the caller omits it, and
    # the greedy pipeline omits it, so the whitelist switched the whole layer off with
    # no message. The condition that actually matters (did the build produce an AMDGPU
    # code object) is measured by `isa_capture.py` and reported as its HOLE exit code.
    ("arming the ISA layer is made conditional on TARGET_LANGUAGE again",
     "const ISA_ENABLED = ISA_MODE !== 'off';",
     "const ISA_ENABLED = ISA_MODE !== 'off' && "
     "String(TARGET_LANGUAGE).toLowerCase() === 'hip';"),
    # Both directions of the default. Flipping it to `gate` arms ENFORCEMENT nobody
    # asked for; flipping it to `off` is the subtler regression and the one worth a
    # probe of its own -- the case this layer catches has no symptom, so a check that
    # must be switched on will be off during exactly the run whose ledger it was
    # meant to keep honest. That is finding (87) restated, and (87)'s fix was not a
    # better tool, it was calling the tool.
    ("the ISA default is flipped to gate, enforcing without being asked",
     "const ISA_MODE = ISA_MODE_RAW === '' || !ISA_MODES.includes(ISA_MODE_RAW)\n"
     "  ? 'observe' : ISA_MODE_RAW;",
     "const ISA_MODE = ISA_MODE_RAW === '' || !ISA_MODES.includes(ISA_MODE_RAW)\n"
     "  ? 'gate' : ISA_MODE_RAW;"),
    ("the ISA default is flipped back to off, so the layer waits to be switched on",
     "const ISA_MODE = ISA_MODE_RAW === '' || !ISA_MODES.includes(ISA_MODE_RAW)\n"
     "  ? 'observe' : ISA_MODE_RAW;",
     "const ISA_MODE = ISA_MODE_RAW === '' || !ISA_MODES.includes(ISA_MODE_RAW)\n"
     "  ? 'off' : ISA_MODE_RAW;"),
    # The next two are the gate's asymmetry against (87), in both directions.
    # Missing ISA evidence is a gap on OUR side and must never refuse a candidate;
    # only a receipt that positively contradicts the candidate's own declared
    # mechanism may. Either mutation turns this into a gate that discards correct,
    # fast kernels for lacking evidence about them.
    ("a missing ISA receipt starts refusing candidates",
     "  if (!receipt || typeof receipt !== 'object') return null;",
     "  if (!receipt || typeof receipt !== 'object') return 'isa:receipt_absent';"),
    ("an indeterminate ISA verdict starts refusing candidates",
     "  if (receipt.mechanism_verdict === 'refuted') {",
     "  if (receipt.mechanism_verdict !== 'realized') {"),
    ("the greedy filter computes the ISA reason and discards it",
     "    if (metricReason || isaReason) {",
     "    if (metricReason) {"),
    # The escalation ladder. The first two are the thresholds that decide when the
    # run spends its most expensive evidence; the third is the contract that stops a
    # requested depth from being recorded as a reached one.
    ("the ladder stops escalating on a stagnating lane",
     "  if (!(noImproveCount >= 1) && !priorRefuted) {",
     "  if (true) {"),
    ("a refuted mechanism no longer escalates to the compiler",
     "  if (priorDepth === 'isa' || priorRefuted) {",
     "  if (priorDepth === 'isa') {"),
    ("a requested evidence depth is recorded as though it were reached",
     "  return attribution ? requestedDepth : 'pattern_after_failed_escalation';",
     "  return requestedDepth;"),
    ("an inconclusive attribution stops counting as reaching the depth",
     "  if (!enabled || requestedDepth === 'pattern') return 'pattern';",
     "  if (!enabled || requestedDepth === 'pattern') return 'pattern';\n"
     "  if (attribution && attribution.status !== 'attributed') return 'pattern';"),
]

# Finding (65) spans two files: the lane decides the verdict, kernel_workflow.js
# decides the winner. Mutating only the producer would leave every check on the
# consumer unproven -- which is (57) exactly, a guard asserted but never exercised.
WF_MUTANTS = [
    ("the bake-off ranks an unverified lane against verified ones",
     "const rankable = (c) => c.kind !== 'lane' || c.validation_trust === 'verified';",
     "const rankable = (c) => true;"),
    ("the winner filter stops consulting the verdict",
     "cands.filter(c => c.speedup > 1.0 && rankable(c))",
     "cands.filter(c => c.speedup > 1.0)"),
    ("a fast-but-unverified lane is dropped silently instead of excluded out loud",
     "scored ${c.speedup.toFixed(2)}x but cannot win: ",
     "scored ${c.speedup.toFixed(2)}x: "),
]

# `test_candidate_floor.js` is the third lexical guard on the lane, and until now
# it was the unmutated one: eight `.test(src)` pins and a `knobs()` extractor,
# none of which had ever been shown to go red. That is the same exposure finding
# (128) names -- a guard nobody has watched fail is a guard nobody knows is
# watching. These mutants are the regressions that file exists to catch: the
# knob's fallbacks, the three sites that must read it instead of a hard-coded
# 1.0, the prompt's byte-identical rendering, and the commit gate it must NOT
# reach.
FLOOR_MUTANTS = [
    ("a floor of 0 is accepted instead of falling back to 1.0",
     "  return Number.isFinite(v) && v > 0 ? v : 1.0;",
     "  return Number.isFinite(v) ? v : 1.0;"),
    ("the prompt renders the default floor as a bare '1' again",
     "const CANDIDATE_FLOOR_TXT = Number.isInteger(CANDIDATE_FLOOR)\n"
     "  ? CANDIDATE_FLOOR.toFixed(1) : String(CANDIDATE_FLOOR);",
     "const CANDIDATE_FLOOR_TXT = String(CANDIDATE_FLOOR);"),
    ("the Optimize prompt goes back to a hard-coded geomean>1.0",
     "when geomean>${CANDIDATE_FLOOR_TXT}.",
     "when geomean>1.0."),
    ("the harvest shortcut reverts to a hard-coded 1.0",
     "!(primSpeedup(eng) > CANDIDATE_FLOOR)",
     "!(primSpeedup(eng) > 1.0)"),
    ("the verified filter reverts to a hard-coded 1.0",
     "return primSpeedup(r.ver) > CANDIDATE_FLOOR;",
     "return primSpeedup(r.ver) > 1.0;"),
    # Injection, and the one that matters most: the floor is a CANDIDATE gate,
    # and the whole knob is only safe because the COMMIT gate never sees it. The
    # suite pins that as an absence, so only a mutant that introduces the
    # forbidden form can show the pin fires.
    ("the commit gate is loosened to the candidate floor",
     "const legacyImproved = !!(winner && winner.geomean > cumulative * (1 + MIN_IMPROVE));",
     "const legacyImproved = !!(winner && winner.geomean > cumulative * (1 + CANDIDATE_FLOOR));"),
    ("the progress signal drops its bestSeen > 0 guard",
     "const madeProgress = !!(winner && bestSeen > 0 && "
     "winner.geomean > bestSeen * (1 + PROGRESS_DELTA));",
     "const madeProgress = !!(winner && "
     "winner.geomean > bestSeen * (1 + PROGRESS_DELTA));"),
    # The one hole the first multi-suite audit reported: the Optimize prompt is
    # the only one assembled by hand rather than through `roleAgent()`, so
    # dropping the skills append is invisible -- the run succeeds and the
    # engineers simply never see the index.
    ("the hand-built Optimize prompt stops appending the expert-skills block",
     "      expertSkillsBlock(isDeep ? 'deep_engineer' : 'engineer'),",
     "      '',"),
    ("progress_delta stops defaulting to MIN_IMPROVE",
     "  const v = parseFloat(A.progress_delta != null ? A.progress_delta : MIN_IMPROVE);",
     "  const v = parseFloat(A.progress_delta != null ? A.progress_delta : 0.05);"),
]


# `test_mode_dispatch.js` is a different kind of guard from the three above: it
# does not pin source text, it EXECUTES kernel_workflow.js against stubbed
# globals and asserts on what comes out. So its corpus is behavioural -- each
# mutant here is a wrong dispatcher that a lexical pin would not notice, and the
# suite catches it by getting a different winner, a different lane set, or no
# throw where a throw was required.
#
# The suite had never run on this box (async IIFE, see run_body), and when it
# first did, section J was red for a reason worth recording: its `workflow` stub
# predated finding (65) and returned no `validation_trust`, so every lane was
# unrankable and the winner was null. A guard that cannot be executed cannot be
# kept honest with the code it guards.
DISPATCH_MUTANTS = [
    ("the bake-off ranks an unverified lane against verified ones",
     "const rankable = (c) => c.kind !== 'lane' || c.validation_trust === 'verified';",
     "const rankable = (c) => true;"),
    ("the winner filter stops consulting the verdict",
     "cands.filter(c => c.speedup > 1.0 && rankable(c))",
     "cands.filter(c => c.speedup > 1.0)"),
    ("a lane merely matching the frozen baseline is allowed to win",
     "cands.filter(c => c.speedup > 1.0 && rankable(c))",
     "cands.filter(c => c.speedup >= 1.0 && rankable(c))"),
    ("the ranking is sorted the wrong way, so the SLOWEST candidate wins",
     ".sort((a, b) => b.speedup - a.speedup)",
     ".sort((a, b) => a.speedup - b.speedup)"),
    ("the table reports only the candidates that could win",
     "const laneRows = cands;",
     "const laneRows = ranked;"),
    ("an unknown mode silently downgrades to the single-language pass-through",
     "if (MODE !== 'bakeoff') {",
     "if (false) {"),
    ("mode stops being trimmed and lowercased",
     "const MODE = String(A.mode != null ? A.mode : 'optimize').trim().toLowerCase() || 'optimize';",
     "const MODE = String(A.mode != null ? A.mode : 'optimize') || 'optimize';"),
    ("an explicit backends list is allowed to drop the incumbent language",
     "  .map(s => String(s == null ? '' : s).trim().toLowerCase()).filter(Boolean);",
     "  .map(s => String(s == null ? '' : s)).filter(Boolean);"),
]


# The expert-skills guard is the fourth file this runner learned to host. It
# asserts one thing that matters -- with the toggle OFF, nothing is injected into
# any prompt, so a run is byte-identical to a build without the feature -- and it
# does it behaviourally, by extracting the real `expertSkillsBlock` and probing
# it. These mutants are the ways that identity breaks.
SKILLS_MUTANTS = [
    ("the toggle defaults to ON, so an unaware caller silently gets injections",
     "const USE_EXPERT_SKILLS = String(A.use_expert_skills != null ? "
     "A.use_expert_skills : 'false') === 'true';",
     "const USE_EXPERT_SKILLS = String(A.use_expert_skills != null ? "
     "A.use_expert_skills : 'true') === 'true';"),
    ("the block is injected into EVERY role, not just the consumers",
     "  if (!USE_EXPERT_SKILLS || !EXPERT_SKILL_ROLES.has(role) || "
     "!EXPERT_SKILLS_DIR) return '';",
     "  if (!USE_EXPERT_SKILLS || !EXPERT_SKILLS_DIR) return '';"),
    ("the toggle stops being consulted at all, so OFF is no longer identity",
     "  if (!USE_EXPERT_SKILLS || !EXPERT_SKILL_ROLES.has(role) || "
     "!EXPERT_SKILLS_DIR) return '';",
     "  if (!EXPERT_SKILL_ROLES.has(role) || !EXPERT_SKILLS_DIR) return '';"),
    ("the injected block drops its ADVISORY framing and reads as an instruction",
     "## Expert skills (ADVISORY — opt-in, enabled this run)",
     "## Expert skills (enabled this run)"),
]


def _require_engine():
    try:
        RJT._make_context()
    except RJT.JSRuntimeUnavailable as exc:
        raise unittest.SkipTest(str(exc)) from exc


def _test_body(suite: str = "test_qd_archive.js", where: Path | None = None) -> str:
    body = ((where or SCRIPTS_DIR) / suite).read_text(encoding="utf-8")
    # V8, unlike node, does not strip a shebang. Blank it rather than drop it so
    # any reported line number still matches the file on disk.
    return "//" + body[2:] if body.startswith("#!") else body


def _run_against(lane_text: str, wf_text: str | None = None,
                 suite: str = "test_qd_archive.js") -> tuple[int, str]:
    # (66): these are distinct files to the shim now, so a mutant in one can no
    # longer masquerade as a change to the other. Whichever is not being mutated
    # is served verbatim from disk.
    #
    # Hosted through the runner's own `run_body` rather than through a private
    # copy of it. The copy had already drifted in the one place that matters:
    # it read a null exit code as 0, so a suite that never ran to the end
    # counted as a mutant SURVIVING here and as a clean pass over there. Two
    # implementations of "did the suite pass" is one more than the number that
    # can be trusted.
    return RJT.run_body({
        str(RJT.LANE): lane_text,
        str(RJT.WORKFLOW): (RJT.WORKFLOW.read_text(encoding="utf-8")
                            if wf_text is None else wf_text),
    }, _test_body(suite))


class JSSuiteTest(unittest.TestCase):
    def test_qd_archive_js_passes(self):
        _require_engine()
        code, output = RJT.run_js_test("test_qd_archive.js")
        self.assertEqual(code, 0, output)
        self.assertIn("PASS:", output)

    def test_the_js_suite_catches_a_broken_invariant(self):
        _require_engine()
        lane = RJT.LANE.read_text(encoding="utf-8")
        baseline, output = _run_against(lane)
        self.assertEqual(baseline, 0, f"unmutated lane must pass first:\n{output}")
        for label, old, new in MUTANTS:
            with self.subTest(mutant=label):
                self.assertIn(
                    old, lane,
                    f"mutant {label!r} no longer applies -- the lane was "
                    f"refactored and this probe is now watching nothing, which "
                    f"is the exact failure it exists to detect")
                # First and last occurrence separately: a rule enforced on two
                # admission paths must be defended on both, not on either.
                for which, mutated in (
                    ("first", lane.replace(old, new, 1)),
                    ("last", new.join(lane.rsplit(old, 1))),
                ):
                    code, out = _run_against(mutated)
                    self.assertNotEqual(
                        code, 0,
                        f"{label} at the {which} site survived the JS suite -- "
                        f"the check watching it is vacuous:\n{out}")

    def test_candidate_floor_js_passes(self):
        _require_engine()
        code, output = RJT.run_js_test("test_candidate_floor.js")
        self.assertEqual(code, 0, output)

    def test_the_candidate_floor_suite_catches_a_broken_invariant(self):
        """The third lexical guard on the lane, mutated like the other two."""
        _require_engine()
        lane = RJT.LANE.read_text(encoding="utf-8")
        baseline, output = _run_against(lane, suite="test_candidate_floor.js")
        self.assertEqual(baseline, 0, f"unmutated lane must pass first:\n{output}")
        for label, old, new in FLOOR_MUTANTS:
            with self.subTest(mutant=label):
                self.assertIn(
                    old, lane,
                    f"mutant {label!r} no longer applies -- the lane was "
                    f"refactored and this probe is now watching nothing")
                code, out = _run_against(lane.replace(old, new, 1),
                                         suite="test_candidate_floor.js")
                self.assertNotEqual(
                    code, 0,
                    f"{label} survived test_candidate_floor.js -- the check "
                    f"watching it is vacuous:\n{out}")

    def test_every_js_guard_in_the_repo_is_registered_with_the_runner(self):
        """An unregistered guard is an unexecuted one -- three times over now.

        `test_qd_archive.js` went unrun because there was no host;
        `test_candidate_floor.js` because the host had an allow-list nobody
        revisited; the two e2e guards because they live in a different scripts
        directory and the runner resolved tests against its own. Each was found
        by hand, one at a time. This is the inventory check that ends that: the
        set of guard files on disk and the set the runner can name are the same
        set, and adding a file without registering it fails here.
        """
        on_disk = {p.name
                   for d in (SCRIPTS_DIR, RJT.E2E_SCRIPTS)
                   for p in d.glob("test_*.js")}
        self.assertEqual(
            on_disk - set(RJT.JS_TESTS), set(),
            "these JS guards exist but no runner can execute them")
        self.assertEqual(
            set(RJT.JS_TESTS) - on_disk, set(),
            "the runner registers JS guards that are no longer on disk")

    def test_expert_skills_guard_passes(self):
        _require_engine()
        code, output = RJT.run_js_test("test_expert_skills_off_identical.js")
        self.assertEqual(code, 0, output)
        self.assertIn("PASS:", output)

    def test_analysis_skill_guard_passes(self):
        _require_engine()
        code, output = RJT.run_js_test("test_analysis_skill_off_identical.js")
        self.assertEqual(code, 0, output)
        self.assertIn("PASS:", output)

    def test_the_expert_skills_guard_catches_a_lost_identity(self):
        """OFF must inject nothing. Mutated in the lane it reads."""
        _require_engine()
        name = "test_expert_skills_off_identical.js"
        lane = RJT.LANE.read_text(encoding="utf-8")
        e2e = RJT.E2E.read_text(encoding="utf-8")
        here = RJT.SUITE_DIRS[name]
        body = _test_body(name, here)

        def run(lane_text):
            return RJT.run_body({str(RJT.E2E): e2e, str(RJT.LANE): lane_text},
                                body, here)

        baseline, output = run(lane)
        self.assertEqual(baseline, 0, f"unmutated lane must pass first:\n{output}")
        for label, old, new in SKILLS_MUTANTS:
            with self.subTest(mutant=label):
                self.assertIn(
                    old, lane,
                    f"mutant {label!r} no longer applies -- the lane was "
                    f"refactored and this probe is now watching nothing")
                code, out = run(lane.replace(old, new, 1))
                self.assertNotEqual(
                    code, 0,
                    f"{label} survived {name} -- the check watching it is "
                    f"vacuous:\n{out}")

    def test_mode_dispatch_js_passes(self):
        _require_engine()
        code, output = RJT.run_js_test("test_mode_dispatch.js")
        self.assertEqual(code, 0, output)
        self.assertIn("PASS:", output)

    def test_the_dispatch_suite_catches_a_broken_dispatcher(self):
        """The behavioural guard, mutated in the file it executes."""
        _require_engine()
        lane = RJT.LANE.read_text(encoding="utf-8")
        wf = RJT.WORKFLOW.read_text(encoding="utf-8")
        baseline, output = _run_against(lane, wf, suite="test_mode_dispatch.js")
        self.assertEqual(baseline, 0, f"unmutated sources must pass first:\n{output}")
        for label, old, new in DISPATCH_MUTANTS:
            with self.subTest(mutant=label):
                self.assertIn(
                    old, wf,
                    f"mutant {label!r} no longer applies -- kernel_workflow.js "
                    f"was refactored and this probe is now watching nothing")
                code, out = _run_against(lane, wf.replace(old, new, 1),
                                         suite="test_mode_dispatch.js")
                self.assertNotEqual(
                    code, 0,
                    f"{label} survived test_mode_dispatch.js -- the check "
                    f"watching it is vacuous:\n{out}")

    def test_a_suite_that_never_finishes_is_not_reported_as_a_pass(self):
        """The harness's own (128): silence from an unrun suite is not success.

        `test_mode_dispatch.js` is one async IIFE, and a synchronous `eval`
        returns at its first `await` with no assertion run and no exit code. The
        body below is that shape with a `await` that never settles, so the only
        thing distinguishing it from a passing suite is the sentinel.
        """
        _require_engine()
        code, out = RJT.run_body(
            {}, "(async () => { await new Promise(() => {}); "
                "console.log('unreachable'); process.exit(0); })();")
        self.assertEqual(code, RJT.INCOMPLETE, out)
        self.assertIn("never called process.exit", out)
        self.assertNotIn("unreachable", out)

    def test_the_js_suite_catches_a_broken_invariant_in_the_bakeoff(self):
        """Finding (65)'s consumer half, mutated in kernel_workflow.js."""
        _require_engine()
        lane = RJT.LANE.read_text(encoding="utf-8")
        wf = RJT.WORKFLOW.read_text(encoding="utf-8")
        baseline, output = _run_against(lane, wf)
        self.assertEqual(baseline, 0, f"unmutated sources must pass first:\n{output}")
        for label, old, new in WF_MUTANTS:
            with self.subTest(mutant=label):
                self.assertIn(
                    old, wf,
                    f"mutant {label!r} no longer applies -- kernel_workflow.js was "
                    f"refactored and this probe is now watching nothing")
                code, out = _run_against(lane, wf.replace(old, new, 1))
                self.assertNotEqual(
                    code, 0,
                    f"{label} survived the JS suite -- the check watching it "
                    f"is vacuous:\n{out}")

    def test_the_prelude_refuses_a_source_it_does_not_serve(self):
        """Finding (66). The shim's stated contract is that a test reaching for an
        unintended file fails loudly. `path.join` used to return the one allowed
        path for any arguments, so the read succeeded and returned the WRONG file's
        text -- every regex the caller then ran described a source it had not
        opened. This is the direction a test harness must never fail in, so the
        refusal itself now has a test."""
        _require_engine()
        probe = """
        var fs = require('fs'), path = require('path');
        var ROOT = path.resolve(__dirname, '..', '..');
        var lane = path.join(ROOT, 'kernel_workflow', 'kernel_lane.js');
        var wf = path.join(ROOT, 'kernel_workflow', 'kernel_workflow.js');
        var other = path.join(ROOT, 'kernel_workflow', 'roles', 'director.md');
        console.log('distinct:' + (lane !== wf));
        // Markers unique to each file: the read must return the file that was asked
        // for, not merely *a* file. Cross-checked both ways so a shim that serves one
        // source for both paths cannot pass.
        var laneText = fs.readFileSync(lane), wfText = fs.readFileSync(wf);
        console.log('lane_is_lane:' + (laneText.indexOf('qdAdmissionCheck') >= 0
                                       && laneText.indexOf("phase('Bakeoff')") < 0));
        console.log('wf_is_wf:' + (wfText.indexOf("phase('Bakeoff')") >= 0
                                   && wfText.indexOf('qdAdmissionCheck') < 0));
        try { fs.readFileSync(other); console.log('refused:false'); }
        catch (e) { console.log('refused:' + /refusing to read/.test(e.message)); }
        console.log('dirname:' + path.dirname(lane).indexOf('kernel_workflow'));
        """
        script = (RJT._prelude({
            str(RJT.LANE): RJT.LANE.read_text(encoding="utf-8"),
            str(RJT.WORKFLOW): RJT.WORKFLOW.read_text(encoding="utf-8"),
        }) + probe + RJT.EPILOGUE)
        out = json.loads(RJT._make_context().eval(script))["output"]
        self.assertIn("distinct:true", out, out)
        self.assertIn("lane_is_lane:true", out, out)
        self.assertIn("wf_is_wf:true", out, out)
        self.assertIn("refused:true", out, out)
        self.assertNotIn("dirname:-1", out, out)


class RobustParityTest(unittest.TestCase):
    """Finding (58), behavioural half. `test_qd_lane_parity.py` checks the two
    noise-floor TABLES match by parsing them; this executes the lane's actual
    `qdCaseRobust` in V8 and compares its output to `qd_robust_stats.robust_stats`
    on the same samples. Matching tables with divergent arithmetic would still be
    two different admission rules, which is the failure the tables were meant to
    prevent.

    It also checks both directions of the floor. A floor that refuses everything
    is not a fix -- it is the archive never filling again, which this project has
    already shipped once as finding (44).
    """

    CONTEXTS = {"decode_m2_square": 0.02456, "prefill_m256_down": 0.11816,
                "prefill_m512_up": 0.16476}

    def setUp(self):
        _require_engine()
        import re
        self.lane = RJT.LANE.read_text(encoding="utf-8")
        pieces = [
            r"const QD_NOISE_FLOOR_BY_MACHINE = new Map\([\s\S]*?\n\]\);\n"
            r"[\s\S]*?const QD_CURRENT_MACHINE[^\n]*\n"
            r"const QD_NOISE_FLOOR = [^\n]*\n",
            r"const QD_DEFAULT_NOISE_FLOOR = Math\.max\([\s\S]*?;\n",
            r"const qdNoiseFloor = [\s\S]*?;\n",
            r"const qdMedian = \(xs\) => \{[\s\S]*?"
            r"const qdCaseRobust = \(ver, contextId\) => \{[\s\S]*?\n\};\n",
        ]
        self.block = ""
        for pattern in pieces:
            match = re.search(pattern, self.lane)
            self.assertIsNotNone(
                match, f"kernel_lane.js no longer contains /{pattern}/; this test "
                       f"cannot extract the admission arithmetic it is meant to check")
            self.block += match.group(0) + "\n"

    def _ctx(self, context, baseline_ms):
        engine = RJT._make_context()
        engine.eval(
            f"var QD_BASELINE_MS = new Map([[{json.dumps(context)}, {baseline_ms!r}]]);\n"
            + self.block
            + "function batch(rows) { return JSON.stringify(rows.map(function (s) {\n"
              f"  return qdCaseRobust({{case_measurement_samples: "
              f"[{{name: {json.dumps(context)}, samples: s}}]}}, {json.dumps(context)});\n"
              "})); }")
        return engine

    def _draw(self, rng, centre_ms, spread, n=3):
        grid = 1e-5  # the harness reports ms to five decimals; ties are real
        return [round(max(grid, rng.gauss(centre_ms, centre_ms * spread)) / grid) * grid
                for _ in range(n)]

    def test_js_and_python_robust_stats_agree_sample_for_sample(self):
        import random
        import qd_robust_stats as QRS
        rng = random.Random(58)
        for context, baseline in self.CONTEXTS.items():
            with self.subTest(context=context):
                floor = QRS.noise_floor(context)
                draws = [self._draw(rng, baseline, floor / 2) for _ in range(400)]
                rows = json.loads(self._ctx(context, baseline).eval(
                    "batch(" + json.dumps(draws) + ")"))
                for js, samples in zip(rows, draws):
                    expected = QRS.robust_stats([baseline / s for s in samples], context)
                    self.assertIsNotNone(js, samples)
                    for field in ("median", "mad", "lower", "upper"):
                        self.assertAlmostEqual(
                            js[field], expected[field], places=12,
                            msg=f"{context} {field} diverges on {samples}")

    def test_identical_arms_almost_never_replace_the_elite(self):
        import random
        import qd_robust_stats as QRS
        rng = random.Random(581)
        trials = 2000
        for context, baseline in self.CONTEXTS.items():
            with self.subTest(context=context):
                floor = QRS.noise_floor(context)
                pairs = [(self._draw(rng, baseline, floor / 2),
                          self._draw(rng, baseline, floor / 2)) for _ in range(trials)]
                rows = json.loads(self._ctx(context, baseline).eval(
                    "batch(" + json.dumps([p[0] for p in pairs] + [p[1] for p in pairs]) + ")"))
                cand, inc = rows[:trials], rows[trials:]
                replaced = sum(1 for a, b in zip(cand, inc)
                               if a and b and a["lower"] > b["upper"])
                # Unfloored this was ~9% -- measured on this same code path.
                self.assertLessEqual(
                    replaced / trials, 0.005,
                    f"{context}: two arms of the SAME kernel replaced the archive "
                    f"elite {replaced}/{trials} times; the admission interval is "
                    f"narrower than the route's own run-to-run spread")

    def test_a_real_improvement_is_still_admitted(self):
        # The fail-closed direction is only acceptable if it still opens. A gain
        # comfortably outside the floor must pass, or (58) has re-created (44).
        import random
        import qd_robust_stats as QRS
        rng = random.Random(582)
        trials = 300
        for context, baseline in self.CONTEXTS.items():
            with self.subTest(context=context):
                floor = QRS.noise_floor(context)
                # Twice the floor's break-even, (1+f)/(1-f). A gain only just
                # past break-even is admitted ~88% of the time on the loudest
                # route, which is correct behaviour and not what this test is
                # for: the question here is whether a clearly-real win still
                # gets in, not where the power curve crosses.
                gain = 2 * ((1 + floor) / (1 - floor) - 1)
                pairs = [(self._draw(rng, baseline / (1 + gain), floor / 2),
                          self._draw(rng, baseline, floor / 2)) for _ in range(trials)]
                rows = json.loads(self._ctx(context, baseline).eval(
                    "batch(" + json.dumps([p[0] for p in pairs] + [p[1] for p in pairs]) + ")"))
                cand, inc = rows[:trials], rows[trials:]
                replaced = sum(1 for a, b in zip(cand, inc)
                               if a and b and a["lower"] > b["upper"])
                self.assertGreaterEqual(
                    replaced / trials, 0.95,
                    f"{context}: a genuine {gain:.0%} speedup was admitted only "
                    f"{replaced}/{trials} times; the floor is refusing real work")


class DescriptorRejectParityTest(unittest.TestCase):
    """Finding (61). `qd_descriptor_v2.descriptor_reject` says in its own
    docstring that it mirrors `qdDescriptorReject` "token for token", and
    `test_qd_lane_parity.py` checked that claim by grepping the JS for the
    tokens. That is the (58) shape exactly: two tables can list the same strings
    and still be reached under different conditions. This runs the SAME
    descriptor through both implementations and compares the returned token.

    Every case below is a rule, not a sample -- if a rule is added to one side
    only, the matrix here has no row for it, so `test_qd_lane_parity`'s
    token-inventory check stays as the guard against that and this stays as the
    guard against the two sides disagreeing on a rule they both have.
    """

    VALID = {"compute_primitive": "native_mfma", "wave_schedule": "independent",
             "k_pipeline": "lds_single", "decomposition": "tile_grid",
             "output_path": "direct_store", "rasterization": "grouped_m",
             "plan_binding": "static"}

    def _js(self):
        import re
        lane = RJT.LANE.read_text(encoding="utf-8")

        def grab(pattern):
            match = re.search(pattern, lane)
            self.assertIsNotNone(match, f"kernel_lane.js no longer contains /{pattern}/")
            return match.group(0)

        def arch_list(name):
            match = re.search(rf"const {name} = (\[[^\]]*\]);", lane)
            self.assertIsNotNone(match, name)
            return json.loads(match.group(1).replace("'", '"'))

        engine = RJT._make_context()
        engine.eval(
            f"var QD_SUPPORTED_ARCHES = {json.dumps(arch_list('QD_SUPPORTED_ARCHES'))};\n"
            f"var QD_MULTI_DIE_ARCHES = {json.dumps(arch_list('QD_MULTI_DIE_ARCHES'))};\n"
            "var QD_ARCH = 'gfx942', QD_DTYPE = 'bf16';\n"
            + grab(r"const QD_VOCAB = \{[\s\S]*?const qdDescriptorReject = "
                   r"\(d\) => \{[\s\S]*?\n\};\n")
            + "function rej(d, a, t) { QD_ARCH = a; QD_DTYPE = t;\n"
              "  return JSON.stringify(qdDescriptorReject(d)); }")
        return engine

    def test_both_implementations_refuse_the_same_descriptor_with_the_same_token(self):
        _require_engine()
        import qd_descriptor_v2 as QD
        engine = self._js()
        no_k = {k: v for k, v in self.VALID.items() if k != "k_pipeline"}
        cases = [
            ("legal", self.VALID, "gfx942", "bf16"),
            ("legal on gfx90a", self.VALID, "gfx90a", "bf16"),
            ("absent", None, "gfx942", "bf16"),
            ("missing axis", no_k, "gfx942", "bf16"),
            ("bad axis value", {**self.VALID, "k_pipeline": "lds_double"}, "gfx942", "bf16"),
            ("reduction without fixup",
             {**self.VALID, "decomposition": "split_k"}, "gfx942", "bf16"),
            ("fixup without reduction",
             {**self.VALID, "output_path": "atomic_fixup"}, "gfx942", "bf16"),
            ("pingpong on valu",
             {**self.VALID, "compute_primitive": "valu",
              "wave_schedule": "symmetric_pingpong"}, "gfx942", "bf16"),
            ("runtime tuned without reduction",
             {**self.VALID, "plan_binding": "runtime_tuned"}, "gfx942", "bf16"),
            ("xcd remap on multi-die",
             {**self.VALID, "rasterization": "xcd_remapped_grouped"}, "gfx942", "bf16"),
            ("xcd remap on single-die",
             {**self.VALID, "rasterization": "xcd_remapped_grouped"}, "gfx90a", "bf16"),
            ("unsupported arch", self.VALID, "gfx1100", "bf16"),
            ("unsupported dtype", self.VALID, "gfx942", "fp8"),
        ]
        tokens = {}
        for label, descriptor, arch, dtype in cases:
            with self.subTest(case=label):
                js = json.loads(engine.eval(
                    f"rej({json.dumps(descriptor)}, {json.dumps(arch)}, {json.dumps(dtype)})"))
                py = QD.descriptor_reject(descriptor, arch=arch, dtype=dtype)
                self.assertEqual(
                    js, py,
                    f"{label}: kernel_lane.js returns {js!r} and qd_descriptor_v2.py returns "
                    f"{py!r}; the two sides are meant to be one policy, and an agent reading "
                    f"one wording while the orchestrator logs another is two policies")
                tokens[label] = py
        # (60): two rules that refuse with the same token are one rule as far as
        # anything reading the log is concerned.
        refusals = [t for label, t in tokens.items() if t is not None
                    and not label.startswith("unsupported")]
        self.assertEqual(len(refusals), len(set(refusals)),
                         f"two distinct violations share a token: {sorted(refusals)}")


if __name__ == "__main__":
    unittest.main()
