from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hl = load_module("measurement_harness_v2", HERE.parent / "harness_lib.py")
route_mod = load_module(
    "bf16_route_manifest_v2",
    REPO / "examples/tasks/dense_bf16_gemm_fused/route_manifest.py",
)


class FakeOutput:
    def __init__(self, ptr, ms):
        self.ptr = ptr
        self.ms = ms

    def data_ptr(self):
        return self.ptr


class ProtocolTest(unittest.TestCase):
    def protocol(self, name="semantic_weight_rotating", **overrides):
        values = {
            "seed": 73,
            "latency_cycles": 3,
            "throughput_cycles": 3,
            "throughput_inner": 4,
            "warmup_cycles": 0,
            "output_ring_size": 2,
            "rotation_target_mb": 2,
            "assumed_cache_mb": 1,
            "max_slots": 4,
        }
        values.update(overrides)
        return hl.measurement_protocol(name, **values)

    def slots(self):
        specs = hl.semantic_weight_slot_specs(101, 2, self.protocol())
        a = object()
        a_scale = object()
        return [{**spec, "args": (a, object(), a_scale, object()),
                 "weight_bytes": 1 << 20}
                for spec in specs]

    def run_receipt(self, metric="single_call_latency", protocol=None):
        protocol = protocol or self.protocol()
        slots = self.slots()
        calls = []
        pointer = iter(range(1000, 10000))
        arm_ms = {"candidate": 1.0, "control": 2.0, "backend": 3.0}

        def make_arm(name):
            def arm(slot):
                calls.append((name, slot["slot_id"],
                              *(id(value) for value in slot["args"])))
                return FakeOutput(next(pointer), arm_ms[name])
            return arm

        def timer(call, inner, retain):
            outputs = [call() for _ in range(inner)]
            for out in outputs:
                retain(out)
            return {"ms": outputs[-1].ms, "wall_ms": outputs[-1].ms + 0.1,
                    "host_ms": 0.01, "timer": "fake_event", "primed": True,
                    "output_ptrs": [out.data_ptr() for out in outputs]}

        with mock.patch.object(hl, "sync", lambda *args, **kwargs: None):
            receipt = hl.measure_paired_arms(
                {name: make_arm(name) for name in ("candidate", "control", "backend")},
                slots,
                protocol=protocol,
                metric=metric,
                sample_timer=timer,
                cache_prepare=lambda policy: None,
            )
        return receipt, calls, slots

    def test_supported_protocols_and_default(self):
        for name in ("legacy_write_flush", "read_evict", "warm",
                     "semantic_weight_rotating"):
            self.assertEqual(self.protocol(name)["name"], name)
        self.assertEqual(self.protocol("semantic-weight-rotating")["name"],
                         "semantic_weight_rotating")

    def test_score_metric_is_versioned_env_aware_and_conflict_closed(self):
        protocol = self.protocol()
        latency = hl.score_metric_config(protocol=protocol)
        self.assertEqual(latency["schema"], "geak.score-metric/v1")
        self.assertEqual(latency["id"], "single_call_latency")
        self.assertEqual(latency["inner"], 1)
        self.assertEqual(latency["hash"], hl.score_metric_hash(latency))
        with mock.patch.dict(os.environ, {"GEAK_SCORE_METRIC": "sustained_throughput"}):
            throughput = hl.score_metric_config(protocol=protocol)
            self.assertGreater(throughput["inner"], 1)
            with self.assertRaisesRegex(ValueError, "score metric conflict"):
                hl.score_metric_config("single_call_latency", protocol=protocol)
        with self.assertRaisesRegex(ValueError, "semantic_weight_rotating"):
            hl.score_metric_config("sustained_throughput", protocol=self.protocol("warm"))
        stale = dict(throughput, inner=1)
        with self.assertRaisesRegex(ValueError, "hash does not match"):
            hl.validate_score_metric(stale, protocol)

    def test_three_arms_share_the_exact_slot_in_every_block(self):
        receipt, calls, slots = self.run_receipt()
        self.assertTrue(receipt["invariants"]["all_arms_in_every_block"])
        self.assertTrue(receipt["invariants"]["same_slot_within_block"])
        self.assertEqual(len(calls), len(receipt["blocks"]) * 3)
        for offset, block in enumerate(receipt["blocks"]):
            group = calls[offset * 3:(offset + 1) * 3]
            self.assertEqual({row[1] for row in group}, {block["slot_id"]})
            self.assertEqual(len({row[2:] for row in group}), 1)
        self.assertEqual(len({id(slot["args"][0]) for slot in slots}), 1)
        self.assertEqual(len({id(slot["args"][2]) for slot in slots}), 1)
        self.assertEqual(len({id(slot["args"][1]) for slot in slots}), len(slots))
        self.assertEqual(len({id(slot["args"][3]) for slot in slots}), len(slots))
        for slot in slots:
            self.assertEqual(slot["semantic"]["a_seed"],
                             slot["semantic"]["a_scale_seed"])
            self.assertEqual(slot["semantic"]["b_seed"],
                             slot["semantic"]["b_scale_seed"])

    def test_a_b_and_scale_seed_pairs_cannot_drift_silently(self):
        specs = hl.semantic_weight_slot_specs(101, 4, self.protocol())
        self.assertEqual(len({row["semantic"]["a_seed"] for row in specs}), 1)
        self.assertEqual(len({row["semantic"]["b_seed"] for row in specs}), 4)
        for row in specs:
            semantic = row["semantic"]
            self.assertEqual(semantic["a_scale_seed"], semantic["a_seed"])
            self.assertEqual(semantic["b_scale_seed"], semantic["b_seed"])
        self.assertEqual(len({row["fingerprint"] for row in specs}), 4)

    def test_protocol_hash_is_stable_sensitive_and_verified(self):
        one = self.protocol()
        two = self.protocol()
        changed = self.protocol(seed=74)
        self.assertEqual(one["hash"], two["hash"])
        self.assertEqual(one["hash"], hl.protocol_hash(one))
        self.assertNotEqual(one["hash"], changed["hash"])
        self.assertNotEqual(one["hash"], self.protocol(timing_prime_mb=5)["hash"])
        self.assertNotEqual(one["hash"], self.protocol(synthetic_evict_mb=128)["hash"])
        one["seed"] += 1
        with self.assertRaisesRegex(ValueError, "hash does not match"):
            self.run_receipt(protocol=one)

    def test_raw_receipt_has_samples_slots_order_and_output_lifetime(self):
        receipt, _, _ = self.run_receipt()
        self.assertEqual(receipt["schema"], "geak.measurement-receipt/v2")
        self.assertEqual(receipt["protocol_hash"], receipt["protocol"]["hash"])
        self.assertEqual(len(receipt["samples"]), len(receipt["blocks"]) * 3)
        self.assertEqual(receipt["slot_order"], [b["slot_id"] for b in receipt["blocks"]])
        for block in receipt["blocks"]:
            self.assertEqual(list(block["samples"]), block["order"])
            self.assertEqual({sample["slot_fingerprint"]
                              for sample in block["samples"].values()},
                             {block["slot_fingerprint"]})
        self.assertTrue(all(row["addresses_rotated"]
                            for row in receipt["output_lifecycle"].values()))

    def test_latency_and_throughput_are_distinct_and_legacy_aliases_latency_only(self):
        latency, _, _ = self.run_receipt("single_call_latency")
        throughput, _, _ = self.run_receipt("sustained_throughput")
        self.assertEqual(latency["metric"]["inner"], 1)
        self.assertGreater(throughput["metric"]["inner"], 1)
        aliases = hl.legacy_timing_aliases(
            latency, candidate_arm="candidate", baseline_arm="control")
        self.assertEqual(aliases["execution_time_ms"], 1.0)
        self.assertEqual(aliases["speedup"], 2.0)
        self.assertEqual(aliases["legacy_metric_source"], "single_call_latency")
        self.assertTrue(aliases["baseline_timing"]["primed"])
        self.assertTrue(aliases["candidate_timing"]["primed"])
        self.assertEqual(aliases["baseline_timing"]["ms"], 2.0)
        self.assertEqual(aliases["candidate_timing"]["ms"], 1.0)
        old_receipt = hl.legacy_timing_receipt([
            {"test_case_id": "case", **aliases},
        ])
        self.assertTrue(old_receipt["all_primed"])
        self.assertFalse(old_receipt["timer_unprimed"])
        self.assertEqual(old_receipt["legacy_metric_source"],
                         "single_call_latency")
        self.assertIn("case", old_receipt["cases"])
        selected = hl.score_timing_aliases(
            throughput, candidate_arm="candidate", baseline_arm="control")
        self.assertEqual(selected["execution_time_ms"], 1.0)
        self.assertEqual(selected["speedup"], 2.0)
        self.assertEqual(selected["score_metric_id"], "sustained_throughput")
        self.assertGreater(selected["score_metric_inner"], 1)
        with self.assertRaisesRegex(ValueError, "single-call latency"):
            hl.legacy_timing_aliases(
                throughput, candidate_arm="candidate", baseline_arm="control")

    def test_legacy_write_flush_keeps_fixed_slot_and_cache_hook(self):
        protocol = self.protocol("legacy_write_flush", latency_cycles=2)
        seen = []
        pointer = iter(range(20000, 21000))

        def timer(call, inner, retain):
            out = call()
            retain(out)
            return {"ms": out.ms, "wall_ms": out.ms, "host_ms": 0.0,
                    "timer": "fake_event", "primed": True,
                    "output_ptrs": [out.data_ptr()]}

        arms = {name: (lambda slot, name=name: FakeOutput(next(pointer), 1.0))
                for name in ("candidate", "control", "backend")}
        with mock.patch.object(hl, "sync", lambda *args, **kwargs: None):
            receipt = hl.measure_paired_arms(
                arms, self.slots(), protocol=protocol, sample_timer=timer,
                cache_prepare=seen.append)
        self.assertEqual(receipt["slot_order"], ["weight-000", "weight-000"])
        self.assertEqual(seen, ["legacy_write_flush"] * 6)

    def test_external_gpu_lock_receipt_does_not_touch_lock_file(self):
        with mock.patch.dict(os.environ, {"GEAK_GPU_LOCK_HELD": "1",
                                          "GEAK_GPU_LOCK_FILE": "/does/not/exist"}, clear=False):
            with hl.gpu_measurement_lock() as receipt:
                self.assertEqual(receipt["mode"], "external")


class RouteFreezeTest(unittest.TestCase):
    def payload(self, backend="control"):
        metric = {
            "schema": "geak.score-metric/v1", "version": 1,
            "id": "single_call_latency", "unit": "ms/call", "inner": 1,
            "protocol_hash": "abc",
        }
        metric["hash"] = hl.score_metric_hash(metric)
        return {
            "schema": route_mod.SCHEMA,
            "task": "task",
            "frozen": True,
            "selection_metric": "single_call_latency",
            "score_metric": metric,
            "protocol_hash": "abc",
            "routes": {"case": {"M": 2, "N": 3, "K": 4, "backend": backend}},
        }

    def test_route_is_loaded_once_and_frozen_for_process_lifetime(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "routes.json"
            path.write_text(json.dumps(self.payload("control")))
            frozen = route_mod.FrozenRouteManifest(path, "task")
            path.write_text(json.dumps(self.payload("backend")))
            self.assertEqual(frozen.backend(2, 3, 4), "control")
            self.assertEqual(frozen.snapshot()["routes"]["case"]["backend"], "control")

    def test_route_validation_pins_protocol_and_shape_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "routes.json"
            path.write_text(json.dumps(self.payload()))
            frozen = route_mod.FrozenRouteManifest(path, "task")
            cases = [{"id": "case", "M": 2, "N": 3, "K": 4}]
            metric = self.payload()["score_metric"]
            self.assertEqual(frozen.validate_for("abc", metric, cases), (True, []))
            ok, errors = frozen.validate_for("different", metric, cases)
            self.assertFalse(ok)
            self.assertIn("protocol hash differs; re-run oracle-bakeoff", errors)

    def test_legacy_latency_manifest_is_readable_but_cannot_be_relabelled_throughput(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "routes.json"
            payload = self.payload()
            payload.pop("score_metric")
            path.write_text(json.dumps(payload))
            frozen = route_mod.FrozenRouteManifest(path, "task")
            cases = [{"id": "case", "M": 2, "N": 3, "K": 4}]
            self.assertEqual(frozen.validate_for("abc", cases), (True, []))
            throughput = dict(self.payload()["score_metric"],
                              id="sustained_throughput", inner=4)
            throughput["hash"] = hl.score_metric_hash(throughput)
            ok, errors = frozen.validate_for("abc", throughput, cases)
            self.assertFalse(ok)
            self.assertTrue(any("legacy oracle" in error for error in errors))

    def test_atomic_writer_supports_rebake_and_rejects_unfrozen_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "routes.json"
            route_mod.write_route_manifest(path, self.payload("backend"))
            frozen = route_mod.FrozenRouteManifest(path, "task")
            self.assertEqual(frozen.backend(2, 3, 4), "backend")
            rejected = self.payload("control")
            rejected["frozen"] = False
            with self.assertRaisesRegex(ValueError, "complete frozen"):
                route_mod.write_route_manifest(path, rejected)
            self.assertEqual(frozen.backend(2, 3, 4), "backend")

    def test_both_tasks_vendor_identical_manifest_logic(self):
        bf16 = REPO / "examples/tasks/dense_bf16_gemm_fused/route_manifest.py"
        fp8 = REPO / "examples/tasks/dense_fp8_gemm_rowwise/route_manifest.py"
        self.assertEqual(bf16.read_bytes(), fp8.read_bytes())


class OracleAdapterContractTest(unittest.TestCase):
    def test_bf16_adapter_never_mutates_process_global_blas_preference(self):
        task = REPO / "examples/tasks/dense_bf16_gemm_fused"
        sources = "\n".join((task / name).read_text() for name in (
            "oracle_wrapper.py", "oracle_loader.py", "src/hipblaslt_baseline.cpp",
            "src/rocblas_bindings.cpp", "scripts/task_runner.py"))
        self.assertNotIn("preferred_blas_library(", sources)
        self.assertIn("hipblasLtMatmul", sources)
        for field in ("algo_index", "solution", "kernel", "version", "workspace_bytes"):
            self.assertIn(f'out["{field}"]', sources)

    def test_bf16_rebake_discovers_fresh_algorithms_under_selected_metric(self):
        runner = (REPO / "examples/tasks/dense_bf16_gemm_fused/scripts/task_runner.py").read_text()
        bake = runner[runner.index("def run_oracle_bakeoff"):runner.index("def run_profile")]
        self.assertIn("oracle_backends(frozen_algorithms=False)", bake)
        self.assertIn('selection_metric": score_metric["id"]', bake)
        self.assertIn('"secondary": {', bake)

    def test_both_task_pins_are_the_common_v2_harness(self):
        common = (REPO / "e2e_workflow/scripts/harness_lib.py").read_bytes()
        for task in ("dense_bf16_gemm_fused", "dense_fp8_gemm_rowwise"):
            root = REPO / "examples/tasks" / task
            self.assertEqual((root / "harness_lib.py").read_bytes(), common)
            provenance = json.loads((root / "PROVENANCE.json").read_text())
            pin = provenance["frozen_harness"]
            self.assertEqual(pin["timer_contract"], "measurement-protocol-v2")
            self.assertEqual(pin["protocol_schema"], "geak.measurement-protocol/v2")
            self.assertEqual(pin["default_protocol"], "semantic_weight_rotating")
            self.assertEqual(hashlib.sha256((root / pin["path"]).read_bytes()).hexdigest(),
                             pin["sha256"])
            immutable = provenance.get("immutable_oracle")
            if immutable and "path" in immutable:
                self.assertEqual(
                    hashlib.sha256((root / immutable["path"]).read_bytes()).hexdigest(),
                    immutable["sha256"])
            for path, expected in (immutable or {}).get("source_sha256", {}).items():
                self.assertEqual(hashlib.sha256((root / path).read_bytes()).hexdigest(),
                                 expected)
            runner = (root / "scripts/task_runner.py").read_text()
            for field in ("baseline_ms", "candidate_ms", "execution_time_ms", "speedup",
                          "baseline_tflops", "candidate_tflops", "baseline_timing",
                          "candidate_timing", "timer_unprimed"):
                self.assertIn(f'"{field}"', runner + (HERE.parent / "harness_lib.py").read_text())

    def test_task_paired_verify_uses_incumbent_for_promotion_and_oracle_for_report_only(self):
        for task in ("dense_bf16_gemm_fused", "dense_fp8_gemm_rowwise"):
            runner = (REPO / "examples/tasks" / task / "scripts/task_runner.py").read_text()
            self.assertIn('"paired-verify"', runner)
            self.assertIn('"candidate": lambda slot', runner)
            self.assertIn('"incumbent": lambda slot', runner)
            self.assertIn('"promotion_control": "incumbent"', runner)
            self.assertIn('"oracle_role": "absolute_report_only"', runner)
            self.assertIn('"geak.paired-measurement-suite/v2"', runner)
            self.assertIn("common_slot_count = max", runner)
            self.assertIn("_make_measurement_slots(case, protocol, common_slot_count)", runner)
            self.assertIn('if trusted:', runner)
            self.assertNotIn("oracle routes unavailable for absolute report", runner)

    def test_checked_in_manifests_fail_closed_for_v2_until_rebaked(self):
        protocol = hl.measurement_protocol(
            "semantic_weight_rotating", seed=73, latency_cycles=3,
            throughput_cycles=3, throughput_inner=4, warmup_cycles=0,
            output_ring_size=2, rotation_target_mb=2, assumed_cache_mb=1,
            max_slots=4)
        for task in ("dense_bf16_gemm_fused", "dense_fp8_gemm_rowwise"):
            root = REPO / "examples/tasks" / task
            frozen = route_mod.FrozenRouteManifest(root / "oracle_routes.v2.json", task)
            payload = frozen.snapshot()
            tree = ast.parse((root / "scripts/task_runner.py").read_text())
            cases = next(
                ast.literal_eval(node.value)
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "CASES"
                        for target in node.targets)
            )
            self.assertEqual(set(payload["routes"]), {case["id"] for case in cases})
            score_metric = hl.score_metric_config(
                "sustained_throughput", protocol=protocol)
            ok, errors = frozen.validate_for(protocol["hash"], score_metric, cases)
            self.assertFalse(ok)
            self.assertTrue(any("selection metric differs" in error for error in errors))

if __name__ == "__main__":
    unittest.main(verbosity=2)
