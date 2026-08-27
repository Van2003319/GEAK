#!/usr/bin/env python3
"""A task's `sol_shapes.json` must still be the shapes that task actually runs.

Why this file exists. `SOL_STATUS` is a ratio between a bound computed from (m,n,k) and a time
measured on a route NAME. The two halves arrive from different places -- the shapes from a manifest,
the times from the harness -- so a manifest that has drifted from the suite does not fail, it
produces a headroom figure that is wrong without being implausible. `prefill_m1024_down`'s bound
against `prefill_m2048_square`'s time is a number the planner would act on.

The manifest is data rather than something the lane derives because the lane is task-agnostic: it
has no filesystem and no way to parse a task's Python. That is a real trade, and this file is what
pays for it -- the duplicate is allowed to exist only for as long as something proves it is a copy.

Every task that ships a `sol_shapes.json` is checked. A task WITHOUT one is not a failure: SOL is
opt-in, and the lane names its absence in the setup echo rather than guessing shapes.

    python3 kernel_workflow/scripts/test_sol_shapes_manifest.py
"""
from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "examples" / "tasks"


def _cases_of(task_runner: Path) -> list[dict] | None:
    """The task's own CASES table, or None when it does not declare one that way."""
    m = re.search(r"^CASES = (\[.*?^\])", task_runner.read_text(), re.S | re.M)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return None


class SolShapesMatchTheSuite(unittest.TestCase):
    def test_every_manifest_matches_its_tasks_case_table(self):
        manifests = sorted(TASKS.glob("*/sol_shapes.json"))
        if not manifests:
            self.skipTest("no task ships a sol_shapes.json yet")
        for manifest in manifests:
            with self.subTest(task=manifest.parent.name):
                doc = json.loads(manifest.read_text())
                runner = manifest.parent / "scripts" / "task_runner.py"
                self.assertTrue(
                    runner.exists(),
                    f"{manifest} has no scripts/task_runner.py beside it, so nothing can say the "
                    f"shapes are the ones that run")
                cases = _cases_of(runner)
                self.assertIsNotNone(
                    cases,
                    f"{runner} declares no literal CASES table, so this manifest cannot be checked "
                    f"against the suite -- either restore the table or drop the manifest, because "
                    f"an unverifiable copy of the shapes is worse than no SOL at all")
                want = [{"route": c["id"], "m": c["M"], "n": c["N"], "k": c["K"]} for c in cases]
                got = [{k: r[k] for k in ("route", "m", "n", "k")} for r in doc["routes"]]
                self.assertEqual(
                    want, got,
                    f"{manifest} has drifted from {runner}:CASES. Order matters as much as content "
                    f"-- a reader comparing the two by eye reads them positionally.")

    def test_the_declared_dtype_is_one_the_card_models(self):
        """A dtype the card has no peak for is a run-time refusal, and it is cheaper to find here.

        `sol_card.py` raises rather than substituting a default, which is the right behaviour and
        also means the failure surfaces mid-run, after the baseline profile has been paid for.
        """
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import sol_card  # noqa: E402

        for manifest in sorted(TASKS.glob("*/sol_shapes.json")):
            with self.subTest(task=manifest.parent.name):
                dtype = json.loads(manifest.read_text())["dtype"]
                modelled = set()
                for cards in (getattr(sol_card, "ARCH_CARDS", {}),
                              getattr(sol_card, "THEORETICAL_CARDS", {})):
                    for card in cards.values():
                        modelled |= set((card.get("peak_flops") or {}).keys())
                self.assertIn(
                    dtype, modelled,
                    f"{manifest} declares dtype {dtype!r}, which no card models. The lane would "
                    f"reach the SOL step and get a refusal after paying for a baseline profile.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
