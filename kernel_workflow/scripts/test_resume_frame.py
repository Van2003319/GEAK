#!/usr/bin/env python3
"""Finding (127): the resumed wave must not mix the vs-seed and vs-incumbent frames.

What went wrong
---------------
`kernel_lane.js` carries two different "cumulative speedup" quantities:

* **wave-local** -- vs `BASELINE_PER_CASE`, which the benchmark engineer measures on
  `CANONICAL` at the top of the run. On a resumed wave `CANONICAL` *is* the previous
  waves' cumulative best, so this is 1.0 at wave start by construction.
* **vs-seed** -- the lane total, several waves deep, read back out of `STATE.json`.

The resume block used to load the second into the variable holding the first
(`if (ps.cumulative > cumulative) cumulative = ps.cumulative`). Three symptoms
followed from that one line, all of them silent:

1. `improved = winner.geomean > cumulative * (1 + MIN_IMPROVE)` compared a verifier's
   wave-local ~1.01 against a vs-seed 4.35 and was false for every round of a wave in
   which three consecutive rounds really did improve.
2. The canonical-commit block is `if (improved)`, so no winner was ever committed and
   the canonical tree silently stayed on the previous round's code -- twice, and once
   the round's parent was lost because of it.
3. The sibling line imported the previous wave's `best_per_case` as if it described the
   current tree; it was read at face value by three separate verifiers, implying an
   incumbent geomean of 1.019 when the incumbent measured ~1.39.

These are lexical checks against the lane source, in the style of
the lane-parity guards. They catch the defect coming back verbatim or by deletion of
the guard; they are not a substitute for executing the lane. Labelled as such on
purpose -- see that module's docstring for why the weaker check is still worth having.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1]
LANE = WORKFLOW / "kernel_lane.js"
TECH_LEAD = WORKFLOW / "roles" / "tech_lead.md"
SOURCE = LANE.read_text(encoding="utf-8")


def resume_block() -> str:
    """The body of `if (setup.resumed && setup.prior_state) { ... }`."""
    start = SOURCE.find("if (setup.resumed && setup.prior_state)")
    if start < 0:
        raise AssertionError(
            "the deep-mode resume block is gone from kernel_lane.js; this test's whole "
            "subject has moved and the parser needs updating before its silence means anything")
    depth, i = 0, SOURCE.index("{", start)
    for j in range(i, len(SOURCE)):
        if SOURCE[j] == "{":
            depth += 1
        elif SOURCE[j] == "}":
            depth -= 1
            if depth == 0:
                return SOURCE[i:j + 1]
    raise AssertionError("unbalanced braces in the resume block")


def code_lines(block: str) -> list[str]:
    """Executable lines only. The block deliberately quotes the old buggy line in its
    comment -- explaining a defect is not committing it, and a checker that cannot tell
    the two apart forbids writing the explanation down."""
    return [ln.strip() for ln in block.splitlines() if not ln.strip().startswith("//")]


class ResumeKeepsTheFramesApart(unittest.TestCase):
    def test_prior_cumulative_is_not_loaded_into_the_wave_local_variable(self):
        body = resume_block()
        offenders = [ln for ln in code_lines(body)
                     if re.search(r"\bcumulative\s*=\s*ps\.cumulative", ln)]
        self.assertEqual(offenders, [], (
            "the resume block assigns the vs-seed lane total into `cumulative`, which every "
            "downstream comparison reads as wave-local. This is finding (127) verbatim: it makes "
            "`improved` false for a whole wave and stops the winner from ever being committed. "
            f"offending line(s): {offenders}"))

    def test_prior_cumulative_is_kept_in_its_own_variable(self):
        self.assertIn("priorCumulativeVsSeed = ps.cumulative", resume_block(),
                      "the vs-seed total must still be carried forward -- just not in `cumulative`. "
                      "Dropping it entirely loses the lane history at the next STATE.json write.")

    def test_the_vs_seed_total_is_derived_not_stored(self):
        self.assertRegex(
            SOURCE, r"const\s+cumulativeVsSeed\s*=\s*\(\)\s*=>\s*priorCumulativeVsSeed\s*\*\s*cumulative",
            "the lane total must be derived from the two frames at read time; a third stored copy "
            "is a third thing to forget to update")

    def test_stale_per_case_rows_are_not_imported(self):
        body = resume_block()
        offenders = [ln for ln in code_lines(body)
                     if re.search(r"\bbestPerCase\s*=\s*ps\.best_per_case", ln)]
        self.assertEqual(offenders, [], (
            "the previous wave's per-case table is being imported as the current incumbent's "
            "timings. BASELINE_PER_CASE was measured on that same tree THIS run and supersedes it; "
            f"importing it is how a 1.019-vs-1.39 frame error reached three verifiers. {offenders}"))

    def test_the_dropped_import_is_announced(self):
        self.assertIn("NOT importing prior_state.best_per_case", resume_block(),
                      "silently dropping an input a prior wave took the trouble to persist is "
                      "indistinguishable from never having read it; say so in the log")


class CommitAdvancesOnlyWhenTheTreeMoved(unittest.TestCase):
    def test_the_commit_result_is_captured(self):
        self.assertRegex(SOURCE, r"const\s+commit\s*=\s*await\s+agentT\(", (
            "the canonical-commit agent's result is discarded again. `committed: false` and a dead "
            "agent then advance the ledger exactly like a success"))

    def test_head_shas_are_required_and_compared(self):
        self.assertIn("head_sha_before", SOURCE)
        self.assertIn("head_sha_after", SOURCE)
        self.assertRegex(SOURCE, r"head_sha_before\.trim\(\)\s*!==\s*commit\.head_sha_after\.trim\(\)", (
            "the two SHAs must actually be compared -- reporting them and not checking them is the "
            "same clean-looking receipt for a commit that never happened"))

    def test_the_ledger_advance_is_gated(self):
        match = re.search(r"if \(commitOK\) \{(.*?)\n    \} else \{", SOURCE, re.S)
        self.assertIsNotNone(match, "the `cumulative = winner.geomean` advance is no longer gated on "
                                    "the commit having succeeded")
        for field in ("cumulative = winner.geomean", "bestPerCase =", "finalWinner = winner"):
            self.assertIn(field, match.group(1),
                          f"`{field}` moved outside the commitOK guard; it must not advance when the "
                          "canonical tree did not")

    def test_a_failed_commit_is_logged_loudly(self):
        self.assertIn("REFUSED to advance the ledger", SOURCE,
                      "a winner that could not be committed must be named in the log; this failure "
                      "was invisible for two rounds")

    def test_reprofile_is_gated_on_the_commit_too(self):
        self.assertRegex(
            SOURCE, r"if \(commitOK\) \{\s*\n\s*profileSummary = await agentT\(", (
                "re-profiling an unchanged canonical and filing it as the new best's bottleneck shift "
                "manufactures a shift out of profiler drift (up to 17% between rocprofv3 invocations "
                "on this lane) and steers the next round off it"))


class BothFramesAreLabelledWhereAgentsReadThem(unittest.TestCase):
    def test_planner_and_report_get_the_vs_seed_number(self):
        self.assertEqual(SOURCE.count("CUMULATIVE_VS_SEED: cumulativeVsSeed()"), 3, (
            "all three agent sites that receive CUMULATIVE_SPEEDUP -- planner, update_memory and "
            "report -- must also receive the vs-seed total, or the one that lacks it will infer it"))

    def test_the_wave_local_frame_is_spelled_out(self):
        self.assertIn("CUMULATIVE_SPEEDUP_FRAME", SOURCE,
                      "an unlabelled speedup is read in whichever frame the reader expects; that is "
                      "the whole mechanism of this finding")

    def test_tech_lead_writes_state_from_the_vs_seed_number(self):
        text = TECH_LEAD.read_text(encoding="utf-8")
        self.assertIn("cumulative: <CUMULATIVE_VS_SEED>", text, (
            "STATE.json's `cumulative` is read back by the NEXT wave as the lane total. Writing the "
            "wave-local number there truncates a 4.35x lane to 1.01x"))
        self.assertIn("MUST come from `CUMULATIVE_VS_SEED`", text,
                      "the reason has to travel with the instruction, or the next edit reverts it")


if __name__ == "__main__":
    unittest.main(verbosity=2)
