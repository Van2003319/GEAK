#!/usr/bin/env python3
"""Finding (127), executed: run the real resume block on V8 against the real STATE.json.

`test_resume_frame.py` pins this fix lexically -- it greps the lane source. That catches the
defect coming back verbatim, and it is what was reachable at the time, but it cannot answer
the question that actually matters before a resumed wave goes out:

    given THIS STATE.json, does the lane now decide IMPROVED=true on a round that improved?

Wave 2 answered that wrong three times in a row and nobody noticed until two winners had been
lost. A lexical guard would have stayed green through all of it if the fix had been subtly
wrong rather than absent -- so the guard and the fix would agree with each other and both be
wrong. This module closes that gap by extracting the resume block and the `improved` expression
from `kernel_lane.js` as source text and EXECUTING them, with the lane's own two frame variables
and the real `exp/state_greedy_coldstart_20260817/STATE.json` as input.

Finding (56)'s lesson, applied: an unexecuted gate is an unexercised one. No system `node` on
this machine, so it runs on the same embedded V8 (`py_mini_racer`) that `run_js_tests.py` uses.

What this does NOT do: it does not run the lane. It hosts two extracted fragments in a sandbox
with mocked `setup`/`log`. A change that moves the frame logic OUT of those fragments would pass
here while breaking the lane -- which is why the extractors below fail loudly rather than
falling back to a default when they cannot find their subject.
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1]
LANE = WORKFLOW / "kernel_lane.js"
STATE = WORKFLOW.parent / "exp" / "state_greedy_coldstart_20260817" / "STATE.json"
SOURCE = LANE.read_text(encoding="utf-8")
GATE_SOURCE = (WORKFLOW / "scripts" / "route_gate.py").read_text(encoding="utf-8")


def _v8():
    try:
        from py_mini_racer import MiniRacer
    except ImportError as exc:  # pragma: no cover
        raise unittest.SkipTest(f"py_mini_racer unavailable: {exc}")
    return MiniRacer()


def resume_block() -> str:
    """The literal body of `if (setup.resumed && setup.prior_state) { ... }`, braces included."""
    start = SOURCE.find("if (setup.resumed && setup.prior_state)")
    if start < 0:
        raise AssertionError(
            "the resume block is gone from kernel_lane.js. This test's subject has moved; it must "
            "be re-pointed before its silence means anything")
    depth, i = 0, SOURCE.index("{", start)
    for j in range(i, len(SOURCE)):
        if SOURCE[j] == "{":
            depth += 1
        elif SOURCE[j] == "}":
            depth -= 1
            if depth == 0:
                return SOURCE[start:j + 1]
    raise AssertionError("unbalanced braces in the resume block")


def improved_expr() -> str:
    """The suite-geomean rule that decides whether a round counts. Extracted, not retyped --
    retyping it would test my transcription of the rule instead of the rule.

    This used to read `const improved = ...`. It is now `legacyImproved`: an audit-only
    cross-session summary retained so resume-frame arithmetic remains visible during migration.
    The commit decision itself is the versioned receipt from route_gate.py, pinned separately below;
    missing evidence or Python failure never falls back to this expression.

    The rule has since moved again, into `judgeCandidate(cand)`, so that SELECTING a winner and
    ACCEPTING one are decided by one implementation rather than two. It is therefore written in
    terms of `cand` rather than `winner`; the harness below binds both names to the same object, so
    the extracted expression evaluates whichever way the lane spells it and this file does not have
    to be edited again the next time the rule is hosted somewhere else.
    """
    m = re.search(r"^\s*const legacyImproved = (.+?);\s*$", SOURCE, re.M)
    if not m:
        raise AssertionError("`const legacyImproved = ...` is no longer a single-line definition "
                             "in kernel_lane.js; the extractor needs updating")
    return m.group(1)


PRELUDE = """
var LOGS = [];
function log(s) { LOGS.push(String(s)); }
var history = { insights: [], ledger: [], asm_ledger: [], rounds: [], bottleneck_now: 'unknown' };
var keypointCitations = [];
var deepAttempts = [];
var processLog = [];
var restoredCanonicalHashes = null;
var PHASE_ORDER = ['phase1_source', 'phase2_asm', 'deep_explore'];
var START_PHASE_FORCED = false;
var searchPhase = 'phase1_source';
var phaseStall = 0;
var cycle = 1;
var phaseLog = [{phase: searchPhase, entered_round: 1, cycle: 1, reason: 'start'}];
var round = 0;
var cumulative = 1.0;
var priorCumulativeVsSeed = 1.0;
var cumulativeVsSeed = function () { return priorCumulativeVsSeed * cumulative; };
"""


def run_resume(prior_state: dict, winner_geomean: float, min_improve: float,
               buggy: bool = False) -> dict:
    """Execute the real resume block, then the real `improved` expression, over `prior_state`.

    `buggy=True` swaps in the pre-fix assignment instead, so the tests can show the guard is
    load-bearing rather than merely satisfied -- a test that passes both before and after a fix
    is not testing the fix.
    """
    block = resume_block()
    if buggy:
        block = block.replace(
            "if (Number.isFinite(ps.cumulative) && ps.cumulative > 0) priorCumulativeVsSeed = ps.cumulative;",
            "if (Number.isFinite(ps.cumulative) && ps.cumulative > cumulative) cumulative = ps.cumulative;")
    js = (PRELUDE
          + f"var setup = {json.dumps({'resumed': True, 'prior_state': prior_state})};\n"
          + block + "\n"
          + f"var MIN_IMPROVE = {min_improve};\n"
          + f"var cand = {json.dumps({'geomean': winner_geomean})};\n"
          + "var winner = cand;\n"
          + f"var improved = {improved_expr()};\n"
          + "JSON.stringify({cumulative: cumulative, prior: priorCumulativeVsSeed, "
            "vs_seed: cumulativeVsSeed(), improved: improved, logs: LOGS});")
    return json.loads(_v8().eval(js))


class TheRealStateResumesIntoTheRightFrames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not STATE.exists():
            raise unittest.SkipTest(f"{STATE} absent; nothing to resume from")
        cls.ps = json.loads(STATE.read_text(encoding="utf-8"))

    def test_wave_local_frame_starts_at_one(self):
        out = run_resume(self.ps, 1.006, 0.005)
        self.assertEqual(out["cumulative"], 1.0, (
            "a resumed wave's CANONICAL *is* the prior best, so the wave-local frame must start at "
            f"1.0. It started at {out['cumulative']}, which is the wave-2 defect"))

    def test_vs_seed_total_is_carried_forward(self):
        out = run_resume(self.ps, 1.006, 0.005)
        self.assertAlmostEqual(out["prior"], float(self.ps["cumulative"]), places=6,
                               msg="the lane total must survive the resume for reporting")
        self.assertAlmostEqual(out["vs_seed"], float(self.ps["cumulative"]), places=6,
                               msg="with cumulative==1.0 the derived lane total equals the stored one")

    def test_a_realistic_round_now_registers_as_improved(self):
        """+0.6% is the size this lane actually delivers per round (1.3524 -> 1.37928 -> 1.39884
        were +2.0% and +1.4%). Under the wave-2 bug every one of them read false."""
        out = run_resume(self.ps, 1.006, 0.005)
        self.assertTrue(out["improved"], (
            "a +0.6% winner against min_improve=0.005 must count as improved; this is the exact "
            "decision that was false for all of wave 2"))

    def test_the_old_line_still_reproduces_the_failure(self):
        """If this ever passes, the fix has stopped being load-bearing and these tests are theatre."""
        out = run_resume(self.ps, 1.006, 0.005, buggy=True)
        self.assertAlmostEqual(out["cumulative"], float(self.ps["cumulative"]), places=6)
        self.assertFalse(out["improved"], (
            "the pre-fix assignment should still produce IMPROVED=false on a genuine improvement; "
            "it did not, so this test no longer demonstrates what it claims"))

    def test_noise_sized_gains_are_still_rejected(self):
        """The fix must not turn into 'everything improves'. min_improve=0.005 sits just under this
        lane's ~1% arm-to-arm suite noise floor, so a +0.2% arm must still be refused."""
        self.assertFalse(run_resume(self.ps, 1.002, 0.005)["improved"],
                         "+0.2% is inside the noise floor and must not advance the ledger")

    def test_the_old_default_would_have_gated_this_lane_out(self):
        """Why min_improve was moved to 0.005: at the 0.02 default, wave 2's real +1.4% round
        registers as no-improvement, and MAX_NO_IMPROVE=2 then ends the wave early."""
        self.assertFalse(run_resume(self.ps, 1.014, 0.02)["improved"])
        self.assertTrue(run_resume(self.ps, 1.014, 0.005)["improved"])

    def test_the_dropped_per_case_import_is_announced_at_runtime(self):
        """The real STATE.json stores `best_per_case_vs_seed`, not `best_per_case`, so the notice
        should stay silent here -- and the resume must still announce the frame it restarts in."""
        out = run_resume(self.ps, 1.006, 0.005)
        joined = " ".join(out["logs"])
        self.assertIn("RESUMED from STATE_DIR", joined)
        self.assertIn("restarts at 1.000x", joined)
        if self.ps.get("best_per_case"):
            self.assertIn("NOT importing", joined)

    def test_a_state_carrying_best_per_case_triggers_the_notice(self):
        """Older waves wrote the un-suffixed key; a resume from one of those must say out loud that
        it is dropping the rows rather than silently ignoring them."""
        ps = dict(self.ps)
        ps["best_per_case"] = [{"case_id": "decode_m2_square", "speedup": 1.019}]
        out = run_resume(ps, 1.006, 0.005)
        self.assertIn("NOT importing", " ".join(out["logs"]))
        self.assertEqual(out["cumulative"], 1.0)


class DegenerateStatesDoNotCorruptTheFrame(unittest.TestCase):
    def test_missing_cumulative_leaves_the_total_at_one(self):
        out = run_resume({"insights": [], "ledger": []}, 1.006, 0.005)
        self.assertEqual(out["prior"], 1.0)
        self.assertTrue(out["improved"], "a state without a lane total must not block the wave")

    def test_nonsense_cumulative_is_refused(self):
        for bad in (0, -3.0, "4.35", None, float("nan")):
            with self.subTest(value=bad):
                ps = {"cumulative": bad if bad == bad else None}  # NaN -> null through JSON
                self.assertEqual(run_resume(ps, 1.006, 0.005)["prior"], 1.0,
                                 f"{bad!r} must not become the lane total")


class PythonReceiptOverridesLegacySummary(unittest.TestCase):
    """Resume arithmetic is diagnostic; the paired Python receipt owns commitment."""

    def test_python_receipt_is_the_commit_decision(self):
        self.assertIn("const routeReceipt = await routeGateForCandidate(cand, {", SOURCE)
        self.assertIn("python3 ${JSON.stringify(PAIRED_VERIFY_SCRIPT)} produce", SOURCE)
        self.assertIn("improved: !!(routeReceipt && routeReceipt.accepted === true)", SOURCE)
        self.assertNotIn("improved: legacyImproved", SOURCE)

    def test_missing_evidence_or_bridge_failure_has_no_summary_fallback(self):
        self.assertIn("legacy suite geomean are diagnostics only and cannot bank this candidate",
                      SOURCE)
        self.assertIn("Python route gate failed closed", SOURCE)

    def test_catastrophic_fence_lives_only_in_python(self):
        self.assertIn("CATASTROPHIC_REGRESSION = 0.10", GATE_SOURCE)
        self.assertIn("estimate_log < -math.log1p(CATASTROPHIC_REGRESSION)", GATE_SOURCE)
        self.assertNotIn("const CATASTROPHIC_REGRESSION", SOURCE)

    def test_the_decision_that_selects_is_the_decision_that_judges(self):
        self.assertIn("const passIdx = candidates.findIndex(c => judgeCandidate(c).improved);",
                      SOURCE)
        self.assertIn("const verdict = winner ? judgeCandidate(winner) : null;", SOURCE)

    def test_legacy_summary_is_still_computed_for_auditable_disagreement(self):
        self.assertIn("const legacyImproved =", SOURCE)
        self.assertIn("improved !== legacyImproved", SOURCE)
        self.assertIn("statistical receipt OVERTURNS the isolated legacy suite comparison", SOURCE)

class ForcedPhaseResumeKeepsHistoryButNotTheOldStall(unittest.TestCase):
    def test_forced_phase_appends_an_auditable_override_and_resets_stall(self):
        prior = {
            "cumulative": 2.0,
            "search_phase": "phase1_source",
            "phase_stall": 3,
            "cycle": 4,
            "last_round": 9,
            "phase_log": [
                {"phase": "phase1_source", "entered_round": 7, "cycle": 4, "reason": "start"}
            ],
            "process_log": [],
        }
        prelude = PRELUDE.replace(
            "var START_PHASE_FORCED = false;", "var START_PHASE_FORCED = true;"
        ).replace(
            "var searchPhase = 'phase1_source';", "var searchPhase = 'phase2_asm';"
        )
        js = (
            prelude
            + f"var setup = {json.dumps({'resumed': True, 'prior_state': prior})};\n"
            + resume_block()
            + "\nJSON.stringify({phase: searchPhase, stall: phaseStall, cycle: cycle, round: round, "
              "phase_log: phaseLog, process_log: processLog});"
        )
        out = json.loads(_v8().eval(js))
        self.assertEqual(out["phase"], "phase2_asm")
        self.assertEqual(out["stall"], 0)
        self.assertEqual(out["cycle"], 4)
        self.assertEqual(out["round"], 9)
        self.assertEqual(out["phase_log"][-1]["entered_round"], 10)
        self.assertIn("operator forced start_phase=phase2_asm", out["phase_log"][-1]["reason"])
        self.assertEqual(out["process_log"][-2]["event"], "resume_phase_override")
        self.assertEqual(out["process_log"][-1]["event"], "state_resumed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
