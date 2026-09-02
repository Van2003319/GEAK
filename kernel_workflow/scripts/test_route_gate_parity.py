#!/usr/bin/env python3
"""Parity means one Python verdict and one strict JavaScript receipt consumer.

The workflow sandbox has no filesystem, ``process`` or ``require``. The live lane
therefore uses its supported narrow command-agent bridge to feed exact JSON to
``route_gate.py``. These tests execute the JavaScript receipt validator in the
same embedded V8 as the other workflow tests and prove that it accepts Python's
receipt verbatim while rejecting stale or malformed protocol output. No second
statistical implementation exists in JavaScript.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import route_gate as RG  # noqa: E402

LANE = SCRIPTS.parent / "kernel_lane.js"

try:
    from py_mini_racer import MiniRacer  # type: ignore
    _V8_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment guard
    MiniRacer = None
    _V8_IMPORT_ERROR = exc


def grab(pattern: str, name: str, text: str) -> str:
    match = re.search(pattern, text, re.S | re.M)
    if not match:
        raise AssertionError(f"could not extract {name} from kernel_lane.js")
    return match.group(0)


def spec(gain: float = 0.03) -> dict:
    ids = [f"pair-{i:02d}" for i in range(RG.MIN_PAIRS)]
    routes = []
    for name, route_gain in (("target", gain), ("guard", 0.0)):
        control = [1.0 + i * 0.001 for i in range(RG.MIN_PAIRS)]
        candidate = [value * (1.0 - route_gain) for value in control]
        routes.append({
            "route": name,
            "candidate": {"session_id": "session-parity", "pair_ids": list(ids),
                          "samples_ms": candidate},
            "control": {"session_id": "session-parity", "pair_ids": list(ids),
                        "samples_ms": control},
        })
    return {
        "schema": RG.INPUT_SCHEMA,
        "session_id": "session-parity",
        "expected_routes": ["target", "guard"],
        "expected_route_count": 2,
        "target_routes": ["target"],
        "paired_routes": routes,
    }


class ReceiptConsumerParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if MiniRacer is None:
            raise unittest.SkipTest(f"py_mini_racer unavailable: {_V8_IMPORT_ERROR}")
        src = LANE.read_text(encoding="utf-8")
        constants = "\n".join([
            grab(r"^const ROUTE_GATE_INPUT_SCHEMA = .*?;$", "input schema", src),
            grab(r"^const ROUTE_GATE_RECEIPT_SCHEMA = .*?;$", "receipt schema", src),
            grab(r"^const ROUTE_GATE_INVOCATION_ERROR_SCHEMA = .*?;$", "error schema", src),
            grab(r"^const ROUTE_GATE_ALGORITHM = .*?;$", "algorithm", src),
            grab(r"^const ROUTE_GATE_BOOTSTRAP_SEED = .*?;$", "seed", src),
            grab(r"^const ROUTE_GATE_BOOTSTRAP_RESAMPLES = .*?;$", "resamples", src),
            grab(r"^const ROUTE_GATE_MIN_PAIRS = .*?;$", "minimum pairs", src),
            f"const ROUTE_GATE_SCRIPT = {json.dumps(str(SCRIPTS / 'route_gate.py'))};",
        ])
        helpers = "\n".join([
            grab(r"^const sortedStrings = .*?;$", "sortedStrings", src),
            grab(r"^const sameStrings = .*?;$", "sameStrings", src),
            grab(r"^const finiteNumber = .*?;$", "finiteNumber", src),
            grab(r"^const routeGateInvocationFailure = .*?\n\}\);", "invocation failure", src),
            grab(r"^const validateRouteGateReceipt = .*?\n\};", "receipt validator", src),
            grab(r"^const routeGateCommand = .*?^\};", "gate command", src),
        ])
        cls.source = src
        cls.validator_source = helpers
        cls.ctx = MiniRacer()
        cls.ctx.eval(constants + "\n" + helpers + r"""
          function consume(spec, receipt) {
            try { return JSON.stringify(validateRouteGateReceipt(receipt, spec)); }
            catch (e) {
              return JSON.stringify(routeGateInvocationFailure(
                'Python route gate failed closed: ' + e.message));
            }
          }
          function commandFor(spec) { return routeGateCommand(spec); }
        """)

    def consume(self, gate_spec: dict, receipt: dict) -> dict:
        return json.loads(self.ctx.call("consume", gate_spec, receipt))

    def assert_invocation_error(self, gate_spec: dict, receipt: dict, token: str) -> None:
        out = self.consume(gate_spec, receipt)
        self.assertEqual(out["decision"], "invocation_error")
        self.assertFalse(out["accepted"])
        self.assertIn(token, out["reason"])

    def test_accept_receipt_crosses_boundary_without_recomputation(self):
        gate_spec = spec(0.03)
        python_receipt = RG.decide(gate_spec)
        self.assertEqual(self.consume(gate_spec, python_receipt), python_receipt)

    def test_statistical_refusal_crosses_boundary_without_recomputation(self):
        gate_spec = spec(0.01)
        python_receipt = RG.decide(gate_spec)
        self.assertEqual(python_receipt["decision"], "refuse")
        self.assertEqual(self.consume(gate_spec, python_receipt), python_receipt)

    def test_invalid_evidence_receipt_is_consumed_but_never_accepted(self):
        gate_spec = spec()
        gate_spec.pop("paired_routes")
        receipt = RG.decide_fail_closed(gate_spec)
        out = self.consume(gate_spec, receipt)
        self.assertEqual(out["decision"], "invalid_evidence")
        self.assertFalse(out["accepted"])

    def test_wrong_schema_or_algorithm_seed_fails_closed(self):
        gate_spec = spec()
        receipt = RG.decide(gate_spec)
        receipt["schema"] = "geak.route-gate-receipt/v1"
        self.assert_invocation_error(gate_spec, receipt, "schema")
        receipt = RG.decide(gate_spec)
        receipt["algorithm"]["seed"] += 1
        self.assert_invocation_error(gate_spec, receipt, "seed")

    def test_stale_session_targets_or_suite_count_fail_closed(self):
        gate_spec = spec()
        for field, value, token in (
                ("session_id", "old-session", "session"),
                ("target_routes", ["guard"], "targets"),
                ("expected_route_count", 1, "route_count")):
            with self.subTest(field=field):
                receipt = RG.decide(gate_spec)
                receipt[field] = value
                self.assert_invocation_error(gate_spec, receipt, token)

    def test_incomplete_routes_or_stale_pair_ids_fail_closed(self):
        gate_spec = spec()
        receipt = RG.decide(gate_spec)
        receipt["routes"] = receipt["routes"][:1]
        self.assert_invocation_error(gate_spec, receipt, "complete")
        receipt = RG.decide(gate_spec)
        receipt["pair_ids"] = receipt["pair_ids"][1:]
        self.assert_invocation_error(gate_spec, receipt, "pair_ids")

    def test_nonfinite_or_incomplete_endpoint_fails_closed(self):
        gate_spec = spec()
        receipt = RG.decide(gate_spec)
        del receipt["suite"]["ci95_lower_log_ratio"]
        self.assert_invocation_error(gate_spec, receipt, "suite")
        receipt = RG.decide(gate_spec)
        receipt["routes"][0]["bootstrap_stream_seed"] = "not-lossless"
        self.assert_invocation_error(gate_spec, receipt, "endpoint")

    def test_real_cli_receipt_is_the_one_javascript_consumes(self):
        gate_spec = spec(0.07)
        proc = subprocess.run([sys.executable, str(SCRIPTS / "route_gate.py"), "-", "--json"],
                              input=json.dumps(gate_spec), text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        cli_receipt = json.loads(proc.stdout)
        self.assertEqual(self.consume(gate_spec, cli_receipt), cli_receipt)

    def test_relay_schema_declares_every_receipt_field(self):
        schema = grab(r"^const ROUTE_GATE_OUTPUT_SCHEMA = .*?^};", "relay schema", self.source)
        for field in ("session_id", "expected_route_count", "input_sha256", "algorithm",
                      "effective_margins", "route_weights", "suite", "routes", "failures"):
            with self.subTest(field=field):
                self.assertRegex(schema, rf"\b{field}:")
        self.assertIn("type: ['object', 'null']", schema)
        self.assertIn("bootstrap_stream_seed", schema)

    def test_bridge_command_embeds_exact_json_under_a_quoted_heredoc(self):
        gate_spec = spec()
        command = self.ctx.call("commandFor", gate_spec)
        self.assertIn(f"python3 {json.dumps(str(SCRIPTS / 'route_gate.py'))} - --json", command)
        self.assertIn("<<'GEAK_ROUTE_GATE_INPUT'", command)
        payload = command.split(" - --json\n", 1)[1].split("\nGEAK_ROUTE_GATE_INPUT", 1)[0]
        self.assertEqual(json.loads(payload), gate_spec)
        self.assertIn('GEAK_ROUTE_GATE_STATUS" -ne 3', command)

    def test_generated_shell_bridge_executes_accept_and_refusal_receipts(self):
        for gain, decision in ((0.03, "accept"), (0.01, "refuse")):
            with self.subTest(decision=decision):
                command = self.ctx.call("commandFor", spec(gain))
                shell = command.split("```bash\n", 1)[1].rsplit("\n```", 1)[0]
                proc = subprocess.run(["bash", "-c", shell], text=True,
                                      capture_output=True, check=False)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(json.loads(proc.stdout)["decision"], decision)

    def test_javascript_uses_supported_bridge_not_a_statistical_twin(self):
        self.assertNotIn("const routeGate =", self.source)
        self.assertNotIn("Math.log", self.validator_source)
        self.assertRegex(self.source, r"const routeGateFromPython = async")
        self.assertIn("agentT(routeGateCommand(submitted)", self.source)
        self.assertIn("python3 ${JSON.stringify(ROUTE_GATE_SCRIPT)} - --json", self.source)
        boundary = grab(r"^const routeGateCommand = .*?^\};$", "Python bridge", self.source)
        for forbidden in ("require(", "child_process", "spawnSync", "suiteRatio", "delta_frac"):
            self.assertNotIn(forbidden, boundary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
