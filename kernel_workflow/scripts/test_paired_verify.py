#!/usr/bin/env python3
"""CPU tests for the deterministic task-v2 paired producer/adapter."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[1]
sys.path.insert(0, str(SCRIPTS))
import paired_verify as pv  # noqa: E402
import route_gate as rg  # noqa: E402
import workflow_state as ws  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "paired_verify_harness", REPO / "e2e_workflow/scripts/harness_lib.py")
assert spec and spec.loader
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)

LOCK = "/tmp/geak-paired-test-gpu.lock"


class FakeOutput:
    def __init__(self, ptr: int, ms: float):
        self.ptr = ptr
        self.ms = ms

    def data_ptr(self) -> int:
        return self.ptr


def protocol() -> dict:
    return harness.measurement_protocol(
        "semantic_weight_rotating", seed=7, latency_cycles=4,
        throughput_cycles=4, throughput_inner=2, warmup_cycles=1,
        output_ring_size=3, rotation_target_mb=2, assumed_cache_mb=1,
        max_slots=2, timing_prime_mb=1, synthetic_evict_mb=2)


def measurement(route: str, gain: float, metric_id: str) -> dict:
    proto = protocol()
    slots = [
        {"slot_id": f"{route}-slot-{index}", "fingerprint": f"{index + 1:064x}",
         "semantic": {"route": route, "index": index}, "weight_bytes": 1 << 20,
         "args": (object(),)}
        for index in range(2)
    ]
    pointer = iter(range(1000, 100000))
    arm_ms = {"candidate": 1.0 - gain, "incumbent": 1.0, "oracle:report": 0.5}

    def arm(name):
        return lambda _slot: FakeOutput(next(pointer), arm_ms[name])

    def timer(call, inner, retain):
        outputs = [call() for _ in range(inner)]
        for out in outputs:
            retain(out)
        return {"ms": outputs[-1].ms, "wall_ms": outputs[-1].ms, "host_ms": 0.01,
                "timer": "cuda_event", "primed": True,
                "output_ptrs": [out.data_ptr() for out in outputs]}

    with mock.patch.dict(os.environ, {"GEAK_GPU_LOCK_HELD": "1",
                                      "GEAK_GPU_LOCK_FILE": LOCK}, clear=False),             mock.patch.object(harness, "sync", lambda *args, **kwargs: None):
        return harness.measure_paired_arms(
            {name: arm(name) for name in ("candidate", "incumbent", "oracle:report")},
            slots, protocol=proto, metric=metric_id,
            sample_timer=timer, cache_prepare=lambda _policy: None)


def raw_suite(routes=("target", "guard"), metric_id="sustained_throughput") -> dict:
    proto = protocol()
    score_metric = harness.score_metric_config(metric_id, protocol=proto)
    secondary_metric = harness.secondary_score_metric(score_metric, protocol=proto)
    rows = []
    for route in routes:
        gain = 0.07 if route == "target" else 0.0
        score_receipt = measurement(route, gain, score_metric["id"])
        secondary_receipt = measurement(route, gain, secondary_metric["id"])
        candidate_ms = score_receipt["medians_ms"]["candidate"]
        incumbent_ms = score_receipt["medians_ms"]["incumbent"]
        rows.append({
            "route": route,
            "score_metric": score_metric,
            "execution_time_ms": candidate_ms,
            "incumbent_ms": incumbent_ms,
            "speedup": incumbent_ms / candidate_ms,
            "measurement_receipt": score_receipt,
            "secondary_metric": secondary_metric,
            "secondary_measurement_receipt": secondary_receipt,
            "absolute_report": {"oracle_arm": "oracle:report"},
        })
    return {
        "schema": pv.RAW_SUITE_SCHEMA,
        "expected_routes": sorted(routes),
        "protocol_hash": rows[0]["measurement_receipt"]["protocol_hash"],
        "score_metric": score_metric,
        "metric_id": score_metric["id"],
        "inner": score_metric["inner"],
        "secondary_metric": secondary_metric,
        "promotion_control": "incumbent",
        "task_entrypoint": "dense_test_gemm",
        "candidate_dispatch": {
            "execution_path": "gemm_wrapper.runtime_table",
            "table_digest": "a" * 64,
            "release_id": "candidate-release",
        },
        "incumbent_dispatch": {
            "execution_path": "gemm_wrapper.runtime_table",
            "table_digest": "b" * 64,
            "release_id": "incumbent-release",
        },
        "oracle_role": "absolute_report_only",
        "routes": rows,
    }


def roots(tmp_path: Path) -> tuple[Path, Path]:
    candidate = tmp_path / "candidate"
    incumbent = tmp_path / "incumbent"
    candidate.mkdir(parents=True)
    incumbent.mkdir(parents=True)
    (candidate / "kernel.py").write_text("candidate = 1\n", encoding="utf-8")
    (incumbent / "kernel.py").write_text("incumbent = 1\n", encoding="utf-8")
    return candidate, incumbent


def task_receipt(tmp_path: Path, routes=("target", "guard")) -> dict:
    candidate, incumbent = roots(tmp_path)
    candidate_hashes = ws.canonical_hashes(candidate)
    incumbent_hashes = ws.canonical_hashes(incumbent)
    task_root = tmp_path / "task"
    task_root.mkdir()
    (task_root / "adapter.py").write_text("# immutable adapter\n", encoding="utf-8")
    with mock.patch.dict(os.environ, {"GEAK_GPU_LOCK_HELD": "1",
                                      "GEAK_GPU_LOCK_FILE": LOCK}, clear=False):
        receipt = pv.seal_task_receipt(
            raw_suite(routes), candidate_root=candidate, incumbent_root=incumbent,
            candidate_hashes_before=candidate_hashes, candidate_hashes_after=candidate_hashes,
            incumbent_hashes_before=incumbent_hashes, incumbent_hashes_after=incumbent_hashes,
            expected_routes=routes, lock_path=LOCK, command=["python3", str(task_root / "adapter.py")])
    receipt["task_root"] = str(task_root.resolve())
    receipt["task_hashes"] = ws.canonical_hashes(task_root)
    receipt.pop("canonical_input_sha256")
    receipt["canonical_input_sha256"] = pv.canonical_sha256(receipt)
    return receipt


def test_task_receipt_to_adapter_to_python_gate(tmp_path: Path) -> None:
    task = task_receipt(tmp_path)
    gate_input = pv.adapt(
        task, session_id="lane-session-1", expected_routes=["target", "guard"],
        target_routes=["target"])
    assert gate_input["schema"] == rg.INPUT_SCHEMA
    assert gate_input["measurement"]["canonical_input_sha256"] == task["canonical_input_sha256"]
    assert gate_input["measurement"]["protocol_hash"] == task["protocol_hash"]
    assert gate_input["measurement"]["score_metric"] == task["score_metric"]
    assert gate_input["measurement"]["metric_id"] == "sustained_throughput"
    assert gate_input["measurement"]["inner"] > 1
    assert gate_input["measurement"]["gpu_lock"]["path"] == LOCK
    assert gate_input["measurement"]["priming"]["all_samples_primed"] is True
    assert gate_input["measurement"]["output_lifecycle"]["all_arms_rotated"] is True
    assert gate_input["measurement"]["task_entrypoint"] == "dense_test_gemm"
    assert gate_input["measurement"]["candidate_dispatch"]["execution_path"] == \
        "gemm_wrapper.runtime_table"
    assert gate_input["paired_routes"][0]["candidate"]["pair_ids"] == [
        f"lane-session-1:block-{index:04d}" for index in range(8)]
    assert "oracle" not in json.dumps(gate_input["paired_routes"])
    receipt = rg.decide(gate_input)
    assert receipt["accepted"] is True, receipt["reason"]
    assert receipt["measurement"] == gate_input["measurement"]
    assert receipt["input_sha256"] == rg._input_digest(gate_input)


def test_complete_suite_missing_one_route_is_rejected(tmp_path: Path) -> None:
    task = task_receipt(tmp_path)
    task["routes"] = task["routes"][:1]
    task_without_digest = dict(task)
    task_without_digest.pop("canonical_input_sha256")
    task["canonical_input_sha256"] = pv.canonical_sha256(task_without_digest)
    with pytest.raises(pv.PairedVerifyError, match="missing complete-suite routes"):
        pv.adapt(task, session_id="lane-session", expected_routes=["target", "guard"],
                 target_routes=["target"])


def test_stale_task_digest_and_protocol_hash_are_rejected(tmp_path: Path) -> None:
    task = task_receipt(tmp_path)
    task["expected_routes"].reverse()
    with pytest.raises(pv.PairedVerifyError, match="canonical input SHA"):
        pv.adapt(task, session_id="lane-session", expected_routes=["target", "guard"],
                 target_routes=["target"])
    task = task_receipt(tmp_path / "second")
    task["protocol"]["seed"] += 1
    unhashed = dict(task)
    unhashed.pop("canonical_input_sha256")
    task["canonical_input_sha256"] = pv.canonical_sha256(unhashed)
    with pytest.raises(pv.PairedVerifyError, match="protocol hash"):
        pv.adapt(task, session_id="lane-session", expected_routes=["target", "guard"],
                 target_routes=["target"])


def test_metric_id_and_inner_drift_are_rejected(tmp_path: Path) -> None:
    task = task_receipt(tmp_path)
    task["inner"] = 1
    unhashed = dict(task)
    unhashed.pop("canonical_input_sha256")
    task["canonical_input_sha256"] = pv.canonical_sha256(unhashed)
    with pytest.raises(pv.PairedVerifyError, match="metric_id/inner"):
        pv.adapt(task, session_id="lane-session", expected_routes=["target", "guard"],
                 target_routes=["target"])


def test_latency_legacy_selection_remains_explicitly_sealable(tmp_path: Path) -> None:
    candidate, incumbent = roots(tmp_path)
    hashes_c = ws.canonical_hashes(candidate)
    hashes_i = ws.canonical_hashes(incumbent)
    task_root = tmp_path / "task"
    task_root.mkdir()
    adapter = task_root / "adapter.py"
    adapter.write_text("# adapter\n")
    receipt = pv.seal_task_receipt(
        raw_suite(metric_id="single_call_latency"), candidate_root=candidate,
        incumbent_root=incumbent, candidate_hashes_before=hashes_c,
        candidate_hashes_after=hashes_c, incumbent_hashes_before=hashes_i,
        incumbent_hashes_after=hashes_i, expected_routes=["target", "guard"],
        lock_path=LOCK, command=[sys.executable, str(adapter)],
        expected_metric_id="single_call_latency")
    assert receipt["metric_id"] == "single_call_latency"
    assert receipt["inner"] == 1


def test_unprimed_or_stale_output_lifecycle_is_rejected(tmp_path: Path) -> None:
    task = task_receipt(tmp_path)
    route = task["routes"][0]
    receipt = route["measurement_receipt"]
    receipt["blocks"][0]["samples"]["candidate"]["primed"] = False
    receipt["samples"][0]["primed"] = False
    route["measurement_receipt_sha256"] = pv.canonical_sha256(receipt)
    unhashed = dict(task)
    unhashed.pop("canonical_input_sha256")
    task["canonical_input_sha256"] = pv.canonical_sha256(unhashed)
    with pytest.raises(pv.PairedVerifyError, match="priming"):
        pv.adapt(task, session_id="lane-session", expected_routes=["target", "guard"],
                 target_routes=["target"])


def test_produce_runs_one_command_under_external_lock_and_seals_roots(tmp_path: Path) -> None:
    candidate, incumbent = roots(tmp_path)
    raw_path = tmp_path / "seed_raw.json"
    raw_path.write_text(json.dumps(raw_suite()), encoding="utf-8")
    output = tmp_path / "paired_task.json"
    task_root = tmp_path / "task"
    task_root.mkdir()
    adapter = task_root / "adapter.py"
    adapter.write_text("import os,shutil\nshutil.copyfile(" + repr(str(raw_path))
                       + ", os.environ['GEAK_PAIRED_RAW_RECEIPT'])\n", encoding="utf-8")
    with mock.patch.dict(os.environ, {"GEAK_GPU_LOCK_HELD": "1",
                                      "GEAK_GPU_LOCK_FILE": LOCK}, clear=False):
        receipt = pv.produce(
            candidate_root=candidate, incumbent_root=incumbent, task_root=task_root,
            expected_routes=["target", "guard"], command=[sys.executable, str(adapter)],
            output=output, score_metric_id="sustained_throughput")
    assert output.is_file()
    assert receipt["incumbent_unchanged"] is True
    assert receipt["gpu_lock"] == {"mode": "external", "path": LOCK}
    assert pv.validate_task_receipt(receipt, ["target", "guard"])[0] == receipt


def test_produce_refuses_when_incumbent_is_modified(tmp_path: Path) -> None:
    candidate, incumbent = roots(tmp_path)
    raw_path = tmp_path / "seed_raw.json"
    raw_path.write_text(json.dumps(raw_suite()), encoding="utf-8")
    task_root = tmp_path / "task"
    task_root.mkdir()
    adapter = task_root / "adapter.py"
    adapter.write_text(
        "import os,shutil,pathlib\nshutil.copyfile(" + repr(str(raw_path))
        + ", os.environ['GEAK_PAIRED_RAW_RECEIPT'])\n"
        + "(pathlib.Path(os.environ['GEAK_PAIRED_INCUMBENT_ROOT'])/'kernel.py').write_text('changed=1')\n",
        encoding="utf-8")
    with mock.patch.dict(os.environ, {"GEAK_GPU_LOCK_HELD": "1",
                                      "GEAK_GPU_LOCK_FILE": LOCK}, clear=False), \
            pytest.raises(pv.PairedVerifyError, match="modified the incumbent"):
        pv.produce(candidate_root=candidate, incumbent_root=incumbent, task_root=task_root,
                   expected_routes=["target", "guard"], command=[sys.executable, str(adapter)],
                   output=tmp_path / "out.json", score_metric_id="sustained_throughput")
