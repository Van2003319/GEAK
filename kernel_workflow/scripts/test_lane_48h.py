#!/usr/bin/env python3
"""Contract tests for the fresh-state 48-hour launcher."""
from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).with_name("lane_48h.py")
SPEC = importlib.util.spec_from_file_location("lane_48h", SCRIPT)
assert SPEC and SPEC.loader
L = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = L
SPEC.loader.exec_module(L)
ROOT = Path(__file__).resolve().parents[2]


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(repo), *args], text=True,
                          capture_output=True, check=True)


def commit_all(repo: Path) -> str:
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    subprocess.run([
        "git", "-C", str(repo), "-c", "user.name=launch-test",
        "-c", "user.email=launch-test@example.invalid", "commit", "-qm", "fixture",
    ], check=True)
    return git(repo, "rev-parse", "HEAD").stdout.strip()


def deadline(now: int = 1_000_000) -> L.Deadline:
    hard = now + 48 * 3600
    return L.Deadline(hard, hard - 1800, hard + 3600, 48 * 3600)


def current_spec(lane: str = "bf16", state: Path | None = None) -> L.LaunchSpec:
    task = L.LANES[lane]["task"]
    state = state or ROOT / "exp" / f"test-never-created-{lane}"
    return L.build_spec(
        lane=lane,
        root=ROOT,
        state_dir=state,
        expected_commit="0" * 40,
        tuner_registry=ROOT / "asm_tuner" / "launch" / task / "registry.json",
        tuner_table=ROOT / "asm_tuner" / "launch" / task / "runtime_table.json",
        deadline=deadline(),
    )


def control_fixture(base: Path, *, now: int = 1_000_000) -> tuple[L.LaunchSpec, L.ControlStore]:
    state = base / "state"
    launch = state / ".launch"
    launch.mkdir(parents=True)
    spec = L.LaunchSpec(
        lane="bf16", task="dense_bf16_gemm_fused", root=base,
        state_dir=state, expected_commit="a" * 40,
        template_path=base / "template", allocation_path=base / "allocation",
        tuner_registry=base / "registry", tuner_table=base / "table",
        gpu_ids=(2, 3, 4), deadline=deadline(now), raw_lane={}, args={}, meta={},
        config_sha256="f" * 64,
    )
    L._atomic_json(launch / "wave_state.json", {"next_wave": 1, "restarts_used": 0})
    return spec, L.ControlStore(spec)


class TemplateContractTest(unittest.TestCase):
    def test_templates_are_root_parameterized_cold_starts(self):
        for lane in L.LANES:
            path = ROOT / L.LANES[lane]["template"]
            text = path.read_text()
            doc = json.loads(text)
            self.assertNotIn("GEAK_asmopt", text)
            self.assertEqual("@ROOT@", doc["kernel_path"].split("/examples/")[0])
            self.assertEqual("off", doc["warm_start"])
            self.assertEqual("off", doc["_require"]["warm_start"])
            self.assertEqual("sustained_throughput", doc["score_metric"])
            self.assertEqual("sustained_throughput", doc["_require"]["score_metric"])
            self.assertEqual(L.EXPECTED_SCORE_METRIC,
                             doc["_launch"]["expected_score_metric"])
            self.assertEqual("@STOP_NEW_ROUND_EPOCH@", doc["deadline_epoch"])
            self.assertIn("sol_shapes", doc)

    def test_default_gpu_plan_is_disjoint_three_plus_three(self):
        allocation = L.load_allocation(ROOT, ROOT / "kernel_workflow/launch/gpu_allocation.json")
        self.assertEqual((2, 3, 4), allocation["bf16"])
        self.assertEqual((5, 6, 7), allocation["fp8"])
        self.assertFalse(set(allocation["bf16"]) & set(allocation["fp8"]))

    def test_bf16_sol_shapes_has_the_exact_suite(self):
        routes = L.expected_routes(
            ROOT / "examples/tasks/dense_bf16_gemm_fused/sol_shapes.json")
        self.assertEqual(11, len(routes))
        self.assertEqual((2048, 4096, 4096), routes["prefill_m2048_square"])

    def test_rendered_paths_stay_in_one_root_and_deadline_is_30m_early(self):
        spec = current_spec()
        for key in ("kernel_path", "workflow_dir", "state_dir", "sol_shapes"):
            self.assertTrue(L.within(Path(spec.args[key]), ROOT, allow_missing=True))
        self.assertEqual(spec.deadline.hard_end_epoch - 1800, spec.args["deadline_epoch"])
        self.assertEqual("2,3,4", spec.args["gpu_ids"])

    def test_path_mismatch_is_refused(self):
        spec = current_spec()
        bad_args = dict(spec.args, kernel_path="/home/yxh/GEAK_asmopt/examples/tasks/dense_bf16_gemm_fused")
        bad = dataclasses.replace(spec, args=bad_args)
        with self.assertRaises(L.LaunchError) as caught:
            L.validate_lane_args(bad)
        self.assertIn("outside expected root", str(caught.exception))


class GitContractTest(unittest.TestCase):
    def test_dirty_worktree_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "tracked").write_text("clean\n")
            head = commit_all(repo)
            (repo / "tracked").write_text("dirty\n")
            with self.assertRaises(L.LaunchError) as caught:
                L.check_git(repo, head, base_commit=head)
            self.assertIn("dirty", str(caught.exception))


class ManifestContractTest(unittest.TestCase):
    def test_stale_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task = root / "examples/tasks/dense_bf16_gemm_fused"
            task.mkdir(parents=True)
            routes = [
                {"route": f"route_{i}", "m": i + 1, "n": 64, "k": 64}
                for i in range(11)
            ]
            (task / "sol_shapes.json").write_text(json.dumps({"dtype": "bf16", "routes": routes}))
            manifest_routes = {
                row["route"]: {"M": row["m"], "N": row["n"], "K": row["k"],
                               "backend": "direct_rocblas"}
                for row in routes
            }
            (task / "oracle_routes.v2.json").write_text(json.dumps({
                "schema": "geak.oracle-route-manifest/v2",
                "task": "dense_bf16_gemm_fused",
                "frozen": True,
                "selection_state": "baked",
                "selection_metric": L.EXPECTED_SCORE_METRIC_ID,
                "score_metric": L.EXPECTED_SCORE_METRIC,
                "protocol_hash": "stale",
                "protocol": {"schema": "geak.measurement-protocol/v2", "hash": "stale"},
                "routes": manifest_routes,
            }))
            head = commit_all(root)
            spec = L.LaunchSpec(
                lane="bf16", task="dense_bf16_gemm_fused", root=root,
                state_dir=root / "exp/new", expected_commit=head,
                template_path=root / "template", allocation_path=root / "allocation",
                tuner_registry=root / "registry", tuner_table=root / "table",
                gpu_ids=(2, 3, 4), deadline=deadline(), raw_lane={},
                args={"kernel_path": str(task), "sol_shapes": str(task / "sol_shapes.json")},
                meta={}, config_sha256="0" * 64,
            )
            with self.assertRaises(L.LaunchError) as caught:
                L.check_oracle_manifest(spec)
            self.assertEqual(L.EXIT_MANIFEST, caught.exception.code)
            self.assertIn("protocol hash", str(caught.exception))


class DeadlineContractTest(unittest.TestCase):
    def test_hard_end_cannot_silently_extend_the_48h_budget(self):
        now = 1_000_000
        with self.assertRaises(L.LaunchError):
            L.new_deadline(now, hours=48, hard_end_epoch=now + 49 * 3600,
                           slurm_end_epoch=now + 50 * 3600)


class TunerContractTest(unittest.TestCase):
    def test_registry_and_table_are_cross_bound_to_workflow_and_protocol(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            consumer = root / "kernel_workflow/kernel_lane.js"
            consumer.parent.mkdir(parents=True)
            consumer.write_text(
                "const registry = A.tuner_registry; const table = A.tuner_table;\n"
                "const bridge = 'asm_tuner runtime_table';\n")
            artifact = root / "artifacts/kernel.so"
            artifact.parent.mkdir()
            artifact.write_bytes(b"ELF-test-artifact")
            registry = root / "tuner/registry.json"
            table = root / "tuner/runtime_table.json"
            registry.parent.mkdir()
            routes = {f"route_{i}": (i + 1, 64, 64) for i in range(11)}
            table_doc = {
                "schema": L.RUNTIME_TABLE_SCHEMA,
                "task": "dense_bf16_gemm_fused",
                "protocol_hash": L.EXPECTED_PROTOCOL_HASH,
                "score_metric": L.EXPECTED_SCORE_METRIC,
                "revision": 1,
                "routes": {
                    name: {
                        "shape": {"name": name, "m": shape[0], "n": shape[1], "k": shape[2]},
                        "entries": [{"config_id": "cfg-ready",
                                     "metric_id": L.EXPECTED_SCORE_METRIC_ID,
                                     "metric_inner": L.EXPECTED_SCORE_METRIC["inner"]}],
                    }
                    for name, shape in routes.items()
                },
            }
            table.write_text(json.dumps(table_doc))
            registry_doc = {
                "schema": L.CONFIG_REGISTRY_SCHEMA,
                "schema_version": 1,
                "launch_contract": {
                    "task": "dense_bf16_gemm_fused",
                    "protocol_hash": L.EXPECTED_PROTOCOL_HASH,
                    "score_metric": L.EXPECTED_SCORE_METRIC,
                    "runtime_table": str(table),
                    "runtime_table_sha256": L.sha256_file(table),
                    "workflow_consumer": {
                        "path": str(consumer.relative_to(root)),
                        "sha256": L.sha256_file(consumer),
                    },
                },
                "configs": [{
                    "id": "cfg-ready", "precompiled": True, "status": "ready",
                    "artifact": str(artifact), "artifact_sha256": L.sha256_file(artifact),
                    "build_receipt": {
                        "task": "dense_bf16_gemm_fused",
                        "protocol_hash": L.EXPECTED_PROTOCOL_HASH,
                        "score_metric": L.EXPECTED_SCORE_METRIC,
                    },
                }],
            }
            registry.write_text(json.dumps(registry_doc))
            commit_all(root)
            spec = L.LaunchSpec(
                lane="bf16", task="dense_bf16_gemm_fused", root=root,
                state_dir=root / "exp/new", expected_commit="0" * 40,
                template_path=root / "template", allocation_path=root / "allocation",
                tuner_registry=registry, tuner_table=table, gpu_ids=(2, 3, 4),
                deadline=deadline(), raw_lane={},
                args={"tuner_registry": str(registry), "tuner_table": str(table)},
                meta={}, config_sha256="0" * 64,
            )
            L.check_tuner(spec, routes)
            registry_doc["launch_contract"]["score_metric"] = {
                **L.EXPECTED_SCORE_METRIC, "id": "single_call_latency"}
            registry.write_text(json.dumps(registry_doc))
            with self.assertRaises(L.LaunchError) as metric_caught:
                L.check_tuner(spec, routes)
            self.assertIn("score-metric", str(metric_caught.exception))
            registry_doc["launch_contract"]["score_metric"] = L.EXPECTED_SCORE_METRIC
            registry.write_text(json.dumps(registry_doc))
            table_doc["protocol_hash"] = "stale"
            table.write_text(json.dumps(table_doc))
            registry_doc["launch_contract"]["runtime_table_sha256"] = L.sha256_file(table)
            registry.write_text(json.dumps(registry_doc))
            with self.assertRaises(L.LaunchError) as caught:
                L.check_tuner(spec, routes)
            self.assertEqual(L.EXIT_TUNER, caught.exception.code)
            self.assertIn("stale protocol", str(caught.exception))


class LockContractTest(unittest.TestCase):
    def test_duplicate_supervisor_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lock = root / "kernel_workflow/launch/locks/bf16.lock"
            lock.parent.mkdir(parents=True)
            lock.touch()
            commit_all(root)
            with L.lane_lock(root, "bf16"):
                with self.assertRaises(L.LaunchError) as caught:
                    with L.lane_lock(root, "bf16"):
                        pass
            self.assertEqual(L.EXIT_DUPLICATE, caught.exception.code)


class DryRunContractTest(unittest.TestCase):
    def test_preflight_path_does_not_create_state_or_touch_lane_lock(self):
        spec = current_spec(state=ROOT / "exp/test-dry-run-must-not-exist")
        self.assertFalse(spec.state_dir.exists())
        lock = L.lock_path(ROOT, "bf16")
        before = lock.stat()
        fake_api = L.ApiConfig(Path("/credential"), "https://example.invalid/v1/messages",
                               "model", {}, "example.invalid", 443)

        @contextlib.contextmanager
        def fake_gpu(*args, **kwargs):
            yield [{"gpu": gpu, "busy_pct": 0, "vram_mb": 0} for gpu in spec.gpu_ids]

        routes = {f"r{i}": (i, 1, 1) for i in range(11)}
        with mock.patch.object(L, "check_identity_and_credentials", return_value=fake_api), \
             mock.patch.object(L, "check_git"), \
             mock.patch.object(L, "check_oracle_manifest", return_value=routes), \
             mock.patch.object(L, "check_tuner"), \
             mock.patch.object(L, "check_storage", return_value={"free_bytes": 1, "free_inodes": 1}), \
             mock.patch.object(L, "gpu_preflight_locks", fake_gpu), \
             mock.patch.object(L.shutil, "which", return_value="/usr/bin/claude"), \
             mock.patch.object(L.time, "time", return_value=1_000_000):
            with L.lane_lock(ROOT, "bf16"):
                report, _ = L.perform_preflight(
                    spec, resume=False, api_probe=lambda api: None,
                    tunnel_probe=lambda api: None)
        after = lock.stat()
        self.assertEqual("ok", report["status"])
        self.assertTrue(report["read_only"])
        self.assertFalse(spec.state_dir.exists())
        self.assertEqual((before.st_size, before.st_mtime_ns), (after.st_size, after.st_mtime_ns))


class SupervisorContractTest(unittest.TestCase):
    def test_expiry_launches_no_wave(self):
        with tempfile.TemporaryDirectory() as td:
            spec, control = control_fixture(Path(td))
            calls: list[int] = []
            rc = L.supervise_engine(
                spec, control, None,
                launch_wave=lambda wave: calls.append(wave) or L.WaveResult(0, 2000),
                now=lambda: spec.deadline.hard_end_epoch,
                sleep=lambda _: None,
            )
            self.assertEqual(0, rc)
            self.assertEqual([], calls)
            self.assertEqual(1, control.state()["next_wave"])

    def test_interrupted_supervisor_resume_consumes_persistent_restart(self):
        with tempfile.TemporaryDirectory() as td:
            spec, control = control_fixture(Path(td))
            state = control.state()
            state.update({"active_wave": 7, "active_pgid": 999999999})
            L._atomic_json(control.wave_state, state)
            sleeps: list[int] = []
            rc = control.recover_interrupted(now=lambda: 1_000_000,
                                             sleep=lambda value: sleeps.append(int(value)))
            self.assertEqual(0, rc)
            self.assertEqual([60], sleeps)
            recovered = control.state()
            self.assertEqual(1, recovered["restarts_used"])
            self.assertIsNone(recovered["active_wave"])
            self.assertIsNone(recovered["active_pgid"])

    def test_three_restarts_use_60_300_900_and_then_fail_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            now = [1_000_000.0]
            spec, control = control_fixture(Path(td), now=int(now[0]))
            calls: list[int] = []
            sleeps: list[int] = []

            def sleep(value: float) -> None:
                sleeps.append(int(value))
                now[0] += value

            def launch(wave: int) -> L.WaveResult:
                calls.append(wave)
                return L.WaveResult(9, 10, reason="transient")

            rc = L.supervise_engine(spec, control, None, launch_wave=launch,
                                    now=lambda: now[0], sleep=sleep)
            self.assertEqual(9, rc)
            self.assertEqual([1, 2, 3, 4], calls)
            self.assertEqual([60, 300, 900], sleeps)
            self.assertEqual(3, control.state()["restarts_used"])
            self.assertEqual(5, control.state()["next_wave"])

    def test_permanent_failure_is_never_restarted(self):
        with tempfile.TemporaryDirectory() as td:
            spec, control = control_fixture(Path(td))
            calls: list[int] = []
            rc = L.supervise_engine(
                spec, control, None,
                launch_wave=lambda wave: calls.append(wave) or
                    L.WaveResult(21, 5, permanent=True, reason="manifest"),
                now=lambda: 1_000_000,
                sleep=lambda _: self.fail("permanent failure slept"),
            )
            self.assertEqual(21, rc)
            self.assertEqual([1], calls)
            self.assertEqual(0, control.state()["restarts_used"])

    def test_permanent_error_is_not_hidden_by_zero_exit(self):
        with tempfile.TemporaryDirectory() as td:
            spec, control = control_fixture(Path(td))
            calls: list[int] = []
            rc = L.supervise_engine(
                spec, control, None,
                launch_wave=lambda wave: calls.append(wave) or
                    L.WaveResult(0, 7200, permanent=True, reason="manifest"),
                now=lambda: 1_000_000,
                sleep=lambda _: self.fail("permanent failure slept"),
            )
            self.assertEqual(L.EXIT_RUNTIME, rc)
            self.assertEqual([1], calls)

    def test_failure_near_drain_cutoff_cannot_turn_into_success(self):
        with tempfile.TemporaryDirectory() as td:
            now = [1_000_000.0]
            spec, control = control_fixture(Path(td), now=int(now[0]))
            now[0] = spec.deadline.stop_new_round_epoch - 30
            sleeps: list[int] = []
            rc = L.supervise_engine(
                spec, control, None,
                launch_wave=lambda wave: L.WaveResult(17, 10, reason="transient"),
                now=lambda: now[0],
                sleep=lambda value: sleeps.append(int(value)),
            )
            self.assertEqual(17, rc)
            self.assertEqual([], sleeps)


if __name__ == "__main__":
    unittest.main(verbosity=2)
