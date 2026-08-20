#!/usr/bin/env python3
"""Run the JS regression suite under pytest, and prove it is not vacuous.

Two tests live here.

`test_lane_gates_js_passes` runs `test_lane_gates.js` on the embedded V8 host in
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

The mutants are single-site on purpose. Several of these rules are enforced at
more than one call site, and the first version of this probe found that breaking
only one site left the suite green: `.test()` proves at least one site has the
rule, never that every site does. That hole is now closed by counting sites, and
these mutants are what keep it closed.

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
    ("the correctness gate goes back to a bare leading-word match",
     "String(v == null ? '' : v).trim().toLowerCase().startsWith(w) && !saysContradicted(v);",
     "String(v == null ? '' : v).trim().toLowerCase().startsWith(w);"),
    ("a partial pass fraction stops contradicting the word PASS",
     "for (let m = frac.exec(s); m; m = frac.exec(s)) if (Number(m[1]) !== Number(m[2])) return true;",
     "for (let m = frac.exec(s); m; m = frac.exec(s)) if (false) return true;"),
    ("the integrated result is compared against candidates in whatever unit it arrives in",
     "const integMetricReason = integMetric ? primMetricReason(integMetric) : null;",
     "const integMetricReason = null;"),
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
    ("a moved oracle becomes a per-candidate skip instead of failing the run closed",
     "    throw new Error(\n      `oracle:digest_drift",
     "    return (\n      `oracle:digest_drift"),
    ("a report with no oracle digest is treated as if it had matched",
     "  if (!got) {\n    return 'oracle:digest_missing",
     "  if (false) {\n    return 'oracle:digest_missing"),
    ("a run that never pinned its oracle can still be published as verified",
     "  if (!oraclePinned) {",
     "  if (false) {"),
    ("the final arbitration stops being held to the denominator it arbitrates",
     "const oracleFinal = validation ? oracleDrift(validation) : null;",
     "const oracleFinal = null;"),
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
    ("an unreadable launch statement is accepted as lockstep",
     "  if (s.exit_code === 3) {",
     "  if (s.exit_code === 3) { return null; }\n  if (false) {"),
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
    ("the planner may overspend by the cost of one direction",
     "    if (plannedCost + d.cost > remaining) {",
     "    if (plannedCost > remaining) {"),
    # The threshold moved into `judgeCandidate(cand)` when SELECTION and acceptance were made to
    # share one implementation, so the mutant follows it. The invariant is untouched.
    ("a candidate wins the round without clearing MIN_IMPROVE",
     "    const legacyImproved = cand.geomean > cumulative * (1 + MIN_IMPROVE);",
     "    const legacyImproved = cand.geomean > cumulative;"),
    ("an incorrect candidate reaches the archive",
     "          says(r.ver.status, 'verified') && says(r.ver.correctness, 'pass'))) return false;",
     "          says(r.ver.status, 'verified'))) return false;"),
    ("a refused validation verdict is treated as trusted",
     "const trusted = verdict.trust !== 'unverified' && verdict.trust !== 'refused';",
     "const trusted = verdict.trust !== 'unverified';"),
    ("only one side of the run declares the oracle digest, so nothing cross-checks it",
     "oracle_digest: { type: 'string' },   // (67): the arbiter's own recomputation of the pinned oracle",
     ""),
    ("the hipify twin receipt is stripped from one of the verdict schemas",
     "hip_twin_sync: HIP_TWIN_SCHEMA,            // (87)",
     ""),
    ("arming the ISA layer is made conditional on TARGET_LANGUAGE again",
     "const ISA_ENABLED = ISA_MODE !== 'off';",
     "const ISA_ENABLED = ISA_MODE !== 'off' && "
     "String(TARGET_LANGUAGE).toLowerCase() === 'hip';"),
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
    ("a missing ISA receipt starts refusing candidates",
     "  if (!receipt || typeof receipt !== 'object') return null;",
     "  if (!receipt || typeof receipt !== 'object') return 'isa:receipt_absent';"),
    ("an indeterminate ISA verdict starts refusing candidates",
     "  if (receipt.mechanism_verdict === 'refuted') {",
     "  if (receipt.mechanism_verdict !== 'realized') {"),
    ("the greedy filter computes the ISA reason and discards it",
     "    if (metricReason || isaReason) {",
     "    if (metricReason) {"),
    ("the ladder stops escalating on a stagnating lane",
     "  if (!(noImproveCount >= 1) && !priorRefuted) {",
     "  if (true) {"),
    ("a refuted mechanism stops escalating at all",
     "  if (!(noImproveCount >= 1) && !priorRefuted) {",
     "  if (!(noImproveCount >= 1)) {"),
    ("the ladder escalates without a localised hot kernel",
     "  if (!hasDominantHotKernel(profile)) {",
     "  if (false) {"),
    ("a requested evidence stage is recorded as though it were reached",
     "  return STAGE_FAILED;",
     "  return requested;"),
    ("an inconclusive attribution stops counting as reaching the rung",
     "  if (irAttribution && irAttribution.status !== 'unavailable') return STAGE_L3;",
     "  if (irAttribution && irAttribution.status === 'attributed') return STAGE_L3;"),
    # The two that keep L3 from decaying back into disassembly reading. An
    # `attributed` return naming no pass is exactly what the previous L3 produced,
    # and it is the shape a mutation would restore by relaxing either guard.
    ("an attribution naming no pass is admitted anyway",
     "  if (!String(attribution.attributed_pass || '').trim()) {",
     "  if (false) {"),
    ("L4 is entered on a request that carries no question",
     "  && String(attribution.compiler_question || '').trim().length > 0",
     "  && true"),
    # Legacy STATE. A resumed lane reading a pre-v2 round as L3/L4 would hand the
    # next planner evidence nobody ever produced.
    ("a pre-v2 round is read as though it held IR evidence",
     "  if (round.evidence_depth && round.evidence_depth !== 'pattern') return STAGE_LEGACY;",
     "  if (round.evidence_depth && round.evidence_depth !== 'pattern') return round.evidence_depth;"),
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
     "const legacyImproved = cand.geomean > cumulative * (1 + MIN_IMPROVE);",
     "const legacyImproved = cand.geomean > cumulative * (1 + CANDIDATE_FLOOR);"),
    # `madeProgress` is now the union of this suite arm and a per-route arm, so the guard being
    # mutated lives on `suiteProgress`. The invariant is unchanged: without `bestSeen > 0` a
    # first-round winner just above the floor makes progress trivially true (bestSeen is 0) and
    # resets a stall counter the pre-knob loop would have advanced.
    ("the progress signal drops its bestSeen > 0 guard",
     "const suiteProgress = !!(winner && bestSeen > 0 && "
     "winner.geomean > bestSeen * (1 + PROGRESS_DELTA));",
     "const suiteProgress = !!(winner && "
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


def _test_body(suite: str = "test_lane_gates.js", where: Path | None = None) -> str:
    body = ((where or SCRIPTS_DIR) / suite).read_text(encoding="utf-8")
    # V8, unlike node, does not strip a shebang. Blank it rather than drop it so
    # any reported line number still matches the file on disk.
    return "//" + body[2:] if body.startswith("#!") else body


def _run_against(lane_text: str, wf_text: str | None = None,
                 suite: str = "test_lane_gates.js") -> tuple[int, str]:
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
    def test_lane_gates_js_passes(self):
        _require_engine()
        code, output = RJT.run_js_test("test_lane_gates.js")
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

        `test_lane_gates.js` went unrun because there was no host;
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
        console.log('lane_is_lane:' + (laneText.indexOf('isaEvidenceReject') >= 0
                                       && laneText.indexOf("phase('Bakeoff')") < 0));
        console.log('wf_is_wf:' + (wfText.indexOf("phase('Bakeoff')") >= 0
                                   && wfText.indexOf('isaEvidenceReject') < 0));
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


if __name__ == "__main__":
    unittest.main()
