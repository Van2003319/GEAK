import datetime as dt
import importlib.util
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

import asm_tuner.integration as integration
from asm_tuner.contracts import TaskContract, sha256_file
from asm_tuner.integration import IntegrationError, materialize_release, verify_materialization
from asm_tuner.runtime_table import RuntimeTableStore


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "asm_tuner/manifests/tasks/dense_bf16_gemm_fused.json"


def rebaked_contract(tmp_path):
    loaded = TaskContract.load(MANIFEST)
    task = tmp_path / "rebaked-task"
    shutil.copytree(loaded.task_root, task,
                    ignore=shutil.ignore_patterns("build", ".torch_ext", "__pycache__"))
    oracle_path = task / "oracle_routes.v2.json"
    oracle = json.loads(oracle_path.read_text())
    oracle["selection_metric"] = loaded.score_metric_id
    oracle["score_metric"] = dict(loaded.score_metric)
    oracle_path.write_text(json.dumps(oracle, indent=2, sort_keys=True) + "\n")
    return TaskContract.load(
        MANIFEST, task_root=task, require_oracle_compatible=True)


class FakeLibrary:
    def __getattr__(self, name):
        return object()


def fake_release(contract, tmp_path):
    configs = [contract.default_config, replace(contract.default_config, wgm=4)]
    for config in configs:
        contract.config_by_id(config.config_id)
    rows = []
    for index, config in enumerate(configs):
        source = tmp_path / ("shard-%d.hip" % index)
        source.write_text("// generated source %d\n" % index)
        artifact = tmp_path / ("shard-%d.so" % index)
        artifact.write_bytes(("compiled artifact %d" % index).encode())
        compile_manifest = tmp_path / ("manifest-%d.json" % index)
        compile_manifest.write_text("{}\n")
        rows.append({
            "config_id": config.config_id,
            "config": config.to_dict(include_derived=True),
            "kernel": contract.adapter.kernel_name(config),
            "symbol": contract.adapter.launch_name(config),
            "isa_hash": str(index + 1) * 64,
            "cache_key": "cache-%d" % index,
            "compile_manifest": str(compile_manifest),
            "compile_manifest_sha256": sha256_file(compile_manifest),
            "artifact": str(artifact),
            "artifact_sha256": sha256_file(artifact),
            "source": str(source),
            "source_sha256": sha256_file(source),
        })
    route_selection = {route.name: configs[0].config_id for route in contract.routes}
    route_rankings = {
        route.name: [configs[0].config_id, configs[1].config_id]
        for route in contract.routes
    }
    release = {
        "schema": integration.RELEASE_SCHEMA,
        "task": contract.identity(),
        "protocol": dict(contract.protocol),
        "score_metric": dict(contract.score_metric),
        "oracle": {
            "manifest_sha256": contract.oracle_sha256,
            "protocol_hash": contract.protocol["hash"],
            "compatible": True,
            "selection_metric": contract.score_metric_id,
            "score_metric": dict(contract.score_metric),
        },
        "fallback": {"kind": "canonical", "reason": "outside suite"},
        "search_space_manifest": contract.value["search_space_manifest"],
        "variant": "unit-top-k",
        "selection_reason": "unit test",
        "route_selection": route_selection,
        "route_rankings": route_rankings,
        "search": {"search_id": "a" * 64, "compiled_candidates": 2},
        "artifacts": rows,
    }
    release["release_id"] = integration._release_identity(release)
    release["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return release


def test_materialization_is_complete_atomic_ranked_and_independently_verifiable(tmp_path, monkeypatch):
    contract = rebaked_contract(tmp_path)
    release = fake_release(contract, tmp_path)
    workspace = tmp_path / "workspace"
    shutil.copytree(contract.task_root, workspace,
                    ignore=shutil.ignore_patterns("build", ".torch_ext", "__pycache__"))
    monkeypatch.setattr(integration.ctypes, "CDLL", lambda path: FakeLibrary())

    receipt = materialize_release(contract, release, workspace=workspace, registry_id="cfg-unit")
    assert receipt["ok"] is True
    assert receipt["route_count"] == 11
    assert len(receipt["config_ids"]) == 2
    table = RuntimeTableStore(workspace / "asm_tuner_runtime_table.json").load()
    assert table["suite"]["atomic"] is True
    assert table["suite"]["complete"] is True
    assert len(table["routes"]) == 11
    assert all([entry["rank"] for entry in row["entries"]] == [0, 1]
               for row in table["routes"].values())
    assert table["fallback"]["kind"] == "canonical"
    assert table["task"]["source_digest"] == contract.source_digest
    assert table["search"]["compiled_candidates"] == 2
    assert table["score_metric"] == dict(contract.score_metric)
    assert all(entry["metric_id"] == contract.score_metric_id
               and entry["metric_inner"] == contract.score_metric_inner
               for row in table["routes"].values() for entry in row["entries"])
    assert (workspace / "asm_tuner_selection.json").is_file()

    spec = importlib.util.spec_from_file_location(
        "materialized_geak_tuner_runtime", workspace / "geak_tuner_runtime.py",
    )
    assert spec and spec.loader
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    loaded = runtime._load_table()
    first_route = contract.routes[0]
    route_id, selected = runtime._select(
        loaded, (first_route.m, first_route.n, first_route.k),
    )
    assert route_id == first_route.name
    assert selected["rank"] == 0
    assert selected["config_id"] == release["route_selection"][first_route.name]

    verified = verify_materialization(contract, workspace=workspace,
                                        rebuild=False, independent=True)
    assert verified["ok"] is True
    assert verified["independent_workspace"] is True
    assert verified["routes_verified"] == 22
    assert verified["task_digest"] == contract.task_digest


def test_materialized_source_tamper_fails_closed(tmp_path, monkeypatch):
    contract = rebaked_contract(tmp_path)
    release = fake_release(contract, tmp_path)
    workspace = tmp_path / "workspace"
    shutil.copytree(contract.task_root, workspace,
                    ignore=shutil.ignore_patterns("build", ".torch_ext", "__pycache__"))
    monkeypatch.setattr(integration.ctypes, "CDLL", lambda path: FakeLibrary())
    materialize_release(contract, release, workspace=workspace, registry_id="cfg-unit")
    table = RuntimeTableStore(workspace / "asm_tuner_runtime_table.json").load()
    entry = table["routes"][table["suite"]["route_ids"][0]]["entries"][1]
    (workspace / entry["source"]).write_text("tampered\n")
    with pytest.raises(IntegrationError, match="source hash mismatch"):
        verify_materialization(contract, workspace=workspace, independent=False)
