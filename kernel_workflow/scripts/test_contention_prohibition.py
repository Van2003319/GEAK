#!/usr/bin/env python3
"""Finding (128): no role may buy itself a GPU by disabling the idleness check.

`gpu_lock.sh` refuses a lane whose card is busy or holds VRAM, because a foreign
tenant is invisible to `flock` but not to the timings. The check is load-bearing and
it is also *trivially* removable from the caller's side -- `GEAK_GPU_REQUIRE_IDLE=0`,
a raised `GEAK_GPU_MAX_BUSY_PCT`/`GEAK_GPU_MAX_VRAM_MB`, or an exported
`HIP_VISIBLE_DEVICES` that skips the wrapper entirely.

Every one of those looks like progress to a blocked agent that has been told to solve
its own obstacles: the command stops hanging and starts returning numbers. The numbers
are measured against another container's load. A blocked run is visible; a contaminated
one is not, which makes it the worse of the two failures.

`gpu_lock.sh` itself is pinned by `test_gpu_lock.py` (the gate defaults to on, and the
source may not contain `GEAK_GPU_REQUIRE_IDLE:-0`). This module pins the other half:
that the roles which actually run timings are TOLD not to reach for the override, and
told why. A prohibition nobody is given is not a control.
"""
from __future__ import annotations

import unittest
from pathlib import Path

ROLES = Path(__file__).resolve().parents[1] / "roles"

# Every role that runs a GPU timing of its own. A role added here without the paragraph
# fails; a role that stops measuring should be removed from this list deliberately.
MEASURING_ROLES = [
    "engineer.md",
    "deep_engineer.md",
    "author_engineer.md",
    "verify_engineer.md",
    "integrator.md",
    "benchmark_engineer.md",
]

# The overrides that would each silently buy a contended GPU.
FORBIDDEN_KNOBS = ["GEAK_GPU_REQUIRE_IDLE=0", "GEAK_GPU_MAX_BUSY_PCT", "GEAK_GPU_MAX_VRAM_MB"]


class EveryMeasuringRoleIsWarned(unittest.TestCase):
    def role(self, name: str) -> str:
        path = ROLES / name
        self.assertTrue(path.exists(), f"{name} is gone from roles/; update MEASURING_ROLES")
        return path.read_text(encoding="utf-8")

    def test_the_prohibition_is_present(self):
        for name in MEASURING_ROLES:
            with self.subTest(role=name):
                self.assertIn("finding (128)", self.role(name), (
                    f"{name} runs GPU timings but is not told that disabling the idleness gate "
                    "produces numbers measured against a co-tenant's load"))

    def test_each_override_is_named(self):
        """Naming them is the point: 'do not bypass the lock' does not stop someone who
        does not recognise `GEAK_GPU_REQUIRE_IDLE=0` as a bypass."""
        for name in MEASURING_ROLES:
            text = self.role(name)
            for knob in FORBIDDEN_KNOBS:
                with self.subTest(role=name, knob=knob):
                    self.assertIn(knob, text, f"{name} does not name {knob} as forbidden")

    def test_the_one_legal_knob_is_distinguished(self):
        """`GEAK_GPU_POOL_WAIT` only changes how long you WAIT -- it cannot contaminate a
        measurement. Lumping it in with the others would leave an agent with no legal move
        at all, which is how prohibitions get ignored wholesale."""
        for name in MEASURING_ROLES:
            with self.subTest(role=name):
                self.assertIn("GEAK_GPU_POOL_WAIT", self.role(name),
                              f"{name} forbids the bypasses without offering the one safe knob")

    def test_the_reason_travels_with_the_rule(self):
        for name in MEASURING_ROLES:
            with self.subTest(role=name):
                self.assertIn("a number produced under contention is not", self.role(name),
                              f"{name} states the rule without the consequence; a rule whose reason "
                              "is missing is the first one dropped in the next edit")

    def test_contention_is_a_reportable_result(self):
        """The escape hatch has to exist and has to be legitimate, or the only way out of a
        blocked run is the forbidden one."""
        for name in MEASURING_ROLES:
            with self.subTest(role=name):
                self.assertIn("report the contention AS your result", self.role(name),
                              f"{name} leaves an agent blocked with no sanctioned way to finish")


if __name__ == "__main__":
    unittest.main(verbosity=2)
