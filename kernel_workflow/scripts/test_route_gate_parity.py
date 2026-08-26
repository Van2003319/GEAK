#!/usr/bin/env python3
"""Execute BOTH route gates on the same inputs and require the same verdict.

Why this file exists. `route_gate.py:decide()` and `routeGate` in `kernel_lane.js` are twins, and
until now nothing compared them -- the Python side was pinned by `test_route_gate.py`, the JS side
by `test_lane_gates.js`, and each guarded only its own implementation. So the two drifted, and by
the time anyone looked they disagreed on four separate rules at once:

  * a route with no measured band: Python RAISED, JS fell back to MIN_ROUTE_WIN
  * the suite condition: JS required suiteRatio > 1, Python had no suite test at all
  * the catastrophic fence: JS only
  * target_routes: Python VETOED a win off the declared route, JS only noted it

Every one of those is a different ACCEPT/REFUSE answer on the same evidence. `kernel_lane.js` is the
gate that decides a real commit, so it is authoritative and the Python twin was brought to it; this
file is what stops the next drift.

It runs the JS through the same embedded V8 (`py_mini_racer`) that `run_js_tests.py` uses, extracting
`routeGate` from the lane source rather than restating it -- a reimplementation agreeing with a
reimplementation would prove nothing about the code that runs.
"""
from __future__ import annotations

import itertools
import json
import re
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import route_gate as RG  # noqa: E402

LANE = SCRIPTS.parent / "kernel_lane.js"

# Same import `run_js_tests.py:96` uses. The nested `from py_mini_racer import py_mini_racer` spelling
# is the OLD package layout and raises ImportError on the version installed here, which would have
# made this file skip silently -- a parity guard that skips is the same as no parity guard.
try:
    from py_mini_racer import MiniRacer  # type: ignore
    _V8_IMPORT_ERROR = None
except Exception as exc:                       # pragma: no cover - environment guard
    MiniRacer = None
    _V8_IMPORT_ERROR = exc


def grab(pattern: str, name: str, text: str) -> str:
    m = re.search(pattern, text, re.S)
    if not m:
        raise AssertionError(f"could not extract {name} from kernel_lane.js")
    return m.group(0)


def js_constant(name: str, text: str) -> float:
    m = re.search(rf"^const {name} = ([0-9.]+);", text, re.M)
    if not m:
        raise AssertionError(f"could not extract {name} from kernel_lane.js")
    return float(m.group(1))


class GateParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if MiniRacer is None:
            raise unittest.SkipTest(f"py_mini_racer unavailable: {_V8_IMPORT_ERROR}")
        src = LANE.read_text(encoding="utf-8")
        cls.min_route_win = js_constant("MIN_ROUTE_WIN", src)
        cls.catastrophic = js_constant("CATASTROPHIC_REGRESSION", src)
        body = grab(r"const routeGate = \(candPerCase, incPerCase, bands, opts\) => \{.*?\n\};\n",
                    "routeGate", src)
        rows = grab(r"const perCaseRows = \(.*?\n\};\n", "perCaseRows", src) \
            if re.search(r"const perCaseRows = \(", src) else ""
        cls.ctx = MiniRacer()
        cls.ctx.eval(f"""
            const MIN_ROUTE_WIN = {cls.min_route_win};
            const CATASTROPHIC_REGRESSION = {cls.catastrophic};
            {rows}
            {body}
            function decide(spec) {{
              const v = routeGate(spec.candidate_per_case, spec.incumbent_per_case, spec.bands,
                                  {{ targetRoutes: spec.target_routes }});
              return JSON.stringify(v);
            }}
        """)

    def js(self, **spec):
        return json.loads(self.ctx.call("decide", spec))

    def py(self, **spec):
        return RG.decide(candidate_per_case=spec["candidate_per_case"],
                         incumbent_per_case=spec["incumbent_per_case"],
                         bands=spec["bands"], target_routes=spec.get("target_routes"))

    def assertSameVerdict(self, spec, note=""):
        j = self.js(**spec)
        p = self.py(**spec)
        self.assertEqual(
            bool(j["accepted"]), bool(p.accepted),
            f"ACCEPT/REFUSE disagree{(' -- ' + note) if note else ''}\n"
            f"  spec: {json.dumps(spec, sort_keys=True)}\n"
            f"  js  : accepted={j['accepted']} {j.get('reason', '')}\n"
            f"  py  : accepted={p.accepted} {p.reason}")
        self.assertEqual(sorted(j["improved"]), sorted(p.improved),
                         f"improved lists disagree{(' -- ' + note) if note else ''}")
        self.assertEqual(sorted(j["catastrophic"]), sorted(p.catastrophic),
                         f"catastrophic lists disagree{(' -- ' + note) if note else ''}")
        if j.get("suiteRatio") is not None and p.suite_ratio is not None:
            self.assertAlmostEqual(j["suiteRatio"], p.suite_ratio, places=9,
                                   msg="suite ratio disagrees")
        return j, p

    # --- the four rules the two used to disagree on ------------------------------------------

    def test_the_constants_are_the_same_number_on_both_sides(self):
        self.assertEqual(self.min_route_win, RG.MIN_ROUTE_WIN)
        self.assertEqual(self.catastrophic, RG.CATASTROPHIC_REGRESSION)

    def test_a_route_with_no_measured_band_does_not_refuse_on_either_side(self):
        """Python used to RAISE here. The band producer is deleted, so raising refuses everything."""
        rows = lambda a, b: [{"name": "r0", "optimized_ms": a}, {"name": "r1", "optimized_ms": b}]
        self.assertSameVerdict(dict(candidate_per_case=rows(0.090, 0.200),
                                    incumbent_per_case=rows(0.100, 0.200),
                                    bands={}), "no bands at all")

    def test_the_suite_condition_is_applied_on_both_sides(self):
        """JS required suiteRatio > 1; Python had no suite test. A candidate that wins one route
        and loses more elsewhere is the case that separates them."""
        cand = [{"name": "r0", "optimized_ms": 0.090}, {"name": "r1", "optimized_ms": 0.260}]
        inc = [{"name": "r0", "optimized_ms": 0.100}, {"name": "r1", "optimized_ms": 0.200}]
        j, p = self.assertSameVerdict(dict(candidate_per_case=cand, incumbent_per_case=inc,
                                           bands={"r0": 0.01, "r1": 0.50}),
                                      "one route wins, the suite is worse")
        self.assertFalse(p.accepted, "a candidate that makes the suite worse must be refused")

    def test_the_catastrophic_fence_exists_on_both_sides(self):
        """A route a third slower must refuse even when the suite average improves."""
        cand = [{"name": "r0", "optimized_ms": 0.010}, {"name": "r1", "optimized_ms": 0.300}]
        inc = [{"name": "r0", "optimized_ms": 0.100}, {"name": "r1", "optimized_ms": 0.200}]
        j, p = self.assertSameVerdict(dict(candidate_per_case=cand, incumbent_per_case=inc,
                                           bands={"r0": 0.01, "r1": 0.60}),
                                      "suite improves, one route 50% slower")
        self.assertFalse(p.accepted)
        self.assertIn("r1", p.catastrophic)

    def test_target_routes_is_a_note_not_a_veto_on_both_sides(self):
        """Python used to REFUSE a win that landed off the declared route; the live gate notes it."""
        cand = [{"name": "r0", "optimized_ms": 0.090}, {"name": "r1", "optimized_ms": 0.200}]
        inc = [{"name": "r0", "optimized_ms": 0.100}, {"name": "r1", "optimized_ms": 0.200}]
        j, p = self.assertSameVerdict(dict(candidate_per_case=cand, incumbent_per_case=inc,
                                           bands={"r0": 0.01, "r1": 0.01},
                                           target_routes=["r1"]),
                                      "win on r0 while r1 was declared")
        self.assertTrue(p.accepted, "an off-target win is banked, and the mismatch is reported")

    def test_an_empty_target_list_narrows_nothing_on_either_side(self):
        cand = [{"name": "r0", "optimized_ms": 0.090}, {"name": "r1", "optimized_ms": 0.200}]
        inc = [{"name": "r0", "optimized_ms": 0.100}, {"name": "r1", "optimized_ms": 0.200}]
        self.assertSameVerdict(dict(candidate_per_case=cand, incumbent_per_case=inc,
                                    bands={"r0": 0.01, "r1": 0.01}, target_routes=[]),
                               "empty target_routes")

    # --- a sweep, because four hand-picked cases is how the last drift survived ---------------

    def test_a_sweep_of_paired_timings_agrees_everywhere(self):
        """Every combination of two routes over a grid that straddles MIN_ROUTE_WIN, the
        catastrophic fence and the suite boundary. Hand-picked cases are what the previous guards
        already had, on both sides, and the drift went through them untouched."""
        inc_ms = 0.100
        grid = [0.060, 0.089, 0.098, 0.0999, 0.100, 0.1001, 0.102, 0.109, 0.111, 0.140]
        band_sets = [{}, {"r0": 0.05}, {"r0": 0.01, "r1": 0.01}, {"r1": 0.08}]
        checked = 0
        for a, b in itertools.product(grid, repeat=2):
            for bands in band_sets:
                spec = dict(
                    candidate_per_case=[{"name": "r0", "optimized_ms": a},
                                        {"name": "r1", "optimized_ms": b}],
                    incumbent_per_case=[{"name": "r0", "optimized_ms": inc_ms},
                                        {"name": "r1", "optimized_ms": inc_ms}],
                    bands=bands)
                with self.subTest(a=a, b=b, bands=tuple(sorted(bands.items()))):
                    self.assertSameVerdict(spec)
                checked += 1
        self.assertGreaterEqual(checked, 400, "the sweep must actually be a sweep")

    def test_a_route_the_candidate_did_not_measure_refuses_on_both_sides(self):
        cand = [{"name": "r0", "optimized_ms": 0.090}]
        inc = [{"name": "r0", "optimized_ms": 0.100}, {"name": "r1", "optimized_ms": 0.200}]
        j = self.js(candidate_per_case=cand, incumbent_per_case=inc, bands={})
        p = self.py(candidate_per_case=cand, incumbent_per_case=inc, bands={})
        self.assertFalse(j["accepted"])
        self.assertFalse(p.accepted)
        self.assertIn("did not measure", p.reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
