#!/usr/bin/env python3
"""Finding (127), second half, executed: the ledger may only move when the tree moved.

The resume-frame bug is what made `improved` false. This is the half that turned that into
lost work: the canonical-commit agent's return value was discarded outright, so
`committed: false`, a refusal, and a dead agent all advanced `cumulative`/`bestPerCase`/
`finalWinner` exactly like a success. Rounds 6 and 7 both had to be repaired by hand and
round 6's parent was lost once, because the ledger described a tree that did not exist and
the next round forked engineers off it.

`test_resume_frame.py` pins the shape of the fix lexically. This module executes the real
`headMoved`/`commitOK` expressions -- extracted from `kernel_lane.js`, not retyped -- against
the specific agent results that actually occur, and asserts what moves and what does not.

The case that matters most is `committed: true` with two EQUAL SHAs. That is an agent making a
positive claim that nothing changed while reporting success, and it is exactly what
`git apply` produces when it prints "Skipped patch" and still exits 0 -- the trap this lane has
already been caught by once. A gate that trusts `committed` alone waves it straight through.

Runs on the same embedded V8 as `test_resume_frame_exec.py` (no system `node` here). Same
limitation, stated for the same reason: this hosts extracted fragments in a sandbox, it does not
run the lane. The extractors fail loudly rather than defaulting, so a refactor that moves this
logic elsewhere breaks the test instead of silently emptying it.
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1]
LANE = WORKFLOW / "kernel_lane.js"
SOURCE = LANE.read_text(encoding="utf-8")

BEFORE = "0000000aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
AFTER = "1111111bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _v8():
    try:
        from py_mini_racer import MiniRacer
    except ImportError as exc:  # pragma: no cover
        raise unittest.SkipTest(f"py_mini_racer unavailable: {exc}")
    return MiniRacer()


def _extract(pattern: str, what: str) -> str:
    m = re.search(pattern, SOURCE, re.S | re.M)
    if not m:
        raise AssertionError(
            f"could not extract {what} from kernel_lane.js. Either the guard is gone or it has been "
            "restructured; this test must be re-pointed before its silence means anything")
    return m.group(0)


def gate_source() -> str:
    """`const headMoved = ...;` through the end of `const commitOK = ...;`."""
    return _extract(r"const headMoved = .*?const commitOK = .*?;\n", "the commitOK gate")


PRELUDE = """
var LOGS = [];
function log(s) { LOGS.push(String(s)); }
var round = 8;
var winner = { geomean: 1.42, source: 'r8_d1', patch: '/tmp/r8_d1.diff',
               per_case: [{ case_id: 'prefill_m1024_down', speedup: 1.01 }] };
// The ledger as it stands before the commit attempt. If the gate lets anything through that
// should not pass, these three are what get corrupted.
var cumulative = 1.39884;
var bestPerCase = [{ case_id: 'prefill_m1024_down', speedup: 0.973 }];
var finalWinner = null;
"""

LEDGER_MOVE = """
if (commitOK) {
  cumulative = winner.geomean;
  bestPerCase = winner.per_case && winner.per_case.length ? winner.per_case : bestPerCase;
  finalWinner = winner;
}
var committedThisRound = commitOK;
"""


def run_gate(commit) -> dict:
    js = (PRELUDE
          + f"var commit = {json.dumps(commit)};\n"
          + gate_source()
          + LEDGER_MOVE
          + "JSON.stringify({headMoved: headMoved, commitOK: commitOK, cumulative: cumulative, "
            "finalWinner: finalWinner ? finalWinner.source : null, "
            "bestCase: bestPerCase[0].speedup, committedThisRound: committedThisRound});")
    return json.loads(_v8().eval(js))


class OnlyARealCommitAdvancesTheLedger(unittest.TestCase):
    def assertLedgerFrozen(self, out, why):
        self.assertFalse(out["commitOK"], why)
        self.assertAlmostEqual(out["cumulative"], 1.39884, places=6,
                               msg=f"cumulative moved anyway: {why}")
        self.assertIsNone(out["finalWinner"], f"finalWinner was set anyway: {why}")
        self.assertAlmostEqual(out["bestCase"], 0.973, places=6,
                               msg=f"bestPerCase was overwritten anyway: {why}")
        self.assertFalse(out["committedThisRound"])

    def test_a_genuine_commit_advances_everything(self):
        out = run_gate({"committed": True, "head_sha_before": BEFORE, "head_sha_after": AFTER})
        self.assertTrue(out["headMoved"])
        self.assertTrue(out["commitOK"])
        self.assertAlmostEqual(out["cumulative"], 1.42, places=6)
        self.assertEqual(out["finalWinner"], "r8_d1")
        self.assertAlmostEqual(out["bestCase"], 1.01, places=6)

    def test_committed_true_with_an_unmoved_head_is_refused(self):
        """The `git apply` "Skipped patch ... exit 0" trap, which has caught this lane before:
        the agent honestly believes it committed, and HEAD says otherwise. HEAD wins."""
        self.assertLedgerFrozen(
            run_gate({"committed": True, "head_sha_before": BEFORE, "head_sha_after": BEFORE}),
            "equal SHAs are a positive claim that nothing changed, contradicting committed:true")

    def test_whitespace_around_a_sha_does_not_fake_a_move(self):
        """`git rev-parse` output arrives with a trailing newline. Comparing unstripped would make
        every commit look successful -- the failure this gate exists to catch, reintroduced by a
        missing .trim()."""
        self.assertLedgerFrozen(
            run_gate({"committed": True, "head_sha_before": BEFORE + "\n",
                      "head_sha_after": "  " + BEFORE + "  "}),
            "the same SHA with different surrounding whitespace is still the same SHA")

    def test_committed_false_is_refused(self):
        self.assertLedgerFrozen(
            run_gate({"committed": False, "head_sha_before": BEFORE, "head_sha_after": BEFORE,
                      "note": "patch did not apply"}),
            "an explicit refusal must not advance the ledger")

    def test_a_dead_agent_is_refused(self):
        """`agentT` returns null when a subagent dies on a terminal error. Before the fix this was
        indistinguishable from success."""
        self.assertLedgerFrozen(run_gate(None), "a null agent result must not advance the ledger")

    def test_a_result_missing_committed_is_refused(self):
        self.assertLedgerFrozen(
            run_gate({"head_sha_before": BEFORE, "head_sha_after": AFTER}),
            "`committed` absent is not `committed: true`")

    def test_a_truthy_non_true_committed_is_refused(self):
        """`committed === true` is deliberately strict: a schema-violating "yes" must not pass."""
        for value in ("true", 1, "yes"):
            with self.subTest(value=value):
                self.assertLedgerFrozen(
                    run_gate({"committed": value, "head_sha_before": BEFORE, "head_sha_after": AFTER}),
                    f"committed={value!r} is not the boolean true")

    def test_absent_shas_are_tolerated_when_committed_is_true(self):
        """Deliberate: an older TechLead prompt does not report SHAs at all. Refusing those would
        make the gate reject every legitimate commit from a role file that has not been updated --
        a gate that blocks all progress gets disabled, which is worse than one that is lenient in
        exactly this one documented direction."""
        out = run_gate({"committed": True})
        self.assertFalse(out["headMoved"], "no SHAs means no demonstrated move")
        self.assertTrue(out["commitOK"], "but absent SHAs must not block a claimed commit")
        self.assertAlmostEqual(out["cumulative"], 1.42, places=6)

    def test_one_sha_present_is_treated_as_absent(self):
        """Half a receipt cannot demonstrate a move, and must not be read as demonstrating one."""
        out = run_gate({"committed": True, "head_sha_after": AFTER})
        self.assertFalse(out["headMoved"])
        self.assertTrue(out["commitOK"])

    def test_the_refusal_is_loud(self):
        """This failure was invisible for two rounds. Silence is the thing being fixed."""
        self.assertIn("REFUSED to advance the ledger", SOURCE)
        self.assertIn("still on disk", SOURCE,
                      "the log must say the patch survives, or a reader assumes the work is lost")


if __name__ == "__main__":
    unittest.main(verbosity=2)
