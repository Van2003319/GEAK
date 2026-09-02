import copy
import datetime as dt
import importlib.util
import json
import os
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
        return lambda *args: 0


def fake_confirmation_gate(contract, route, row):
    candidate_id = row["config_id"]
    blocks = []
    for index in range(8):
        samples = {
            candidate_id: {
                "ms": 0.9, "slot_fingerprint": "slot", "arm": candidate_id,
            },
            integration.CANONICAL_CONTROL_ARM: {
                "ms": 1.0, "slot_fingerprint": "slot",
                "arm": integration.CANONICAL_CONTROL_ARM,
            },
        }
        blocks.append({
            "block": index, "cycle": index, "slot_id": "slot",
            "slot_fingerprint": "slot", "samples": samples,
        })
    gate = integration._candidate_vs_canonical_gate(
        {
            "blocks": blocks,
            "protocol_hash": contract.protocol["hash"],
            "metric_id": contract.score_metric_id,
            "metric": {"inner": contract.score_metric_inner},
            "medians_ms": {
                candidate_id: 0.9,
                integration.CANONICAL_CONTROL_ARM: 1.0,
            },
        },
        route=route, candidate_id=candidate_id,
        evidence_phase="independent_materialized_task_dispatch",
    )
    gate["confirmation"] = {
        "workspace": "isolated_materialized_copy",
        "task_entrypoint": contract.value["bindings"]["entrypoint"]["symbol"],
        "runtime_table_schema": "geak.runtime_table/v3",
        "table_digest": "d" * 64,
        "correctness": {"passed": True, "max_relative_error": 0.0},
        "dispatch_receipt": {
            "route": route,
            "config_id": candidate_id,
            "symbol": row["symbol"],
            "table_digest": "d" * 64,
            "artifact_sha256": row["artifact_sha256"],
            "source_sha256": row["source_sha256"],
            "compile_manifest_sha256": row["compile_manifest_sha256"],
        },
    }
    return gate


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
    winner_route = contract.routes[0].name
    route_selection = {
        route.name: configs[0].config_id if route.name == winner_route else None
        for route in contract.routes
    }
    route_rankings = {
        route.name: (
            [configs[0].config_id, configs[1].config_id]
            if route.name == winner_route else []
        )
        for route in contract.routes
    }
    route_outcomes = {
        route.name: (
            {
                "status": "winner",
                "selection": "candidate",
                "reason": "unit candidate-vs-canonical paired MDE gate passed",
                "config_ids": [configs[0].config_id, configs[1].config_id],
                "candidate_gates": [
                    fake_confirmation_gate(contract, winner_route, row) for row in rows
                ],
            }
            if route.name == winner_route else
            {
                "status": "unmeasured",
                "selection": "canonical",
                "reason": "route was outside this unit search target",
                "candidate_gates": [],
            }
        )
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
        "route_outcomes": route_outcomes,
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
    assert [entry["rank"] for entry in table["routes"][contract.routes[0].name]["entries"]] == [0, 1]
    assert all(
        row["entries"] == [] and row["selection"]["kind"] == "canonical"
        for route, row in table["routes"].items() if route != contract.routes[0].name
    )
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
    original_sha256 = runtime._sha256
    hash_calls = []
    runtime._sha256 = lambda path: (
        hash_calls.append(str(path)) or original_sha256(path)
    )
    loaded = runtime._load_table()
    first_load_hashes = len(hash_calls)
    assert first_load_hashes > 0
    assert runtime._load_table() is loaded
    assert len(hash_calls) == first_load_hashes
    first_route = contract.routes[0]
    route_id, selected = runtime._select(
        loaded, (first_route.m, first_route.n, first_route.k),
    )
    assert route_id == first_route.name
    assert selected["rank"] == 0
    assert selected["config_id"] == release["route_selection"][first_route.name]
    fallback_route = contract.routes[1]
    assert runtime._select(
        loaded, (fallback_route.m, fallback_route.n, fallback_route.k),
    ) is None
    os.utime(workspace / "asm_tuner_runtime_table.json", None)
    assert runtime._load_table()["table_digest"] == loaded["table_digest"]
    assert len(hash_calls) > first_load_hashes

    verified = verify_materialization(contract, workspace=workspace,
                                        rebuild=False, independent=True)
    assert verified["ok"] is True
    assert verified["independent_workspace"] is True
    assert verified["routes_verified"] == 2
    assert verified["route_selections"][fallback_route.name]["status"] == "unmeasured"
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
    entry = next(
        row["entries"][1] for row in table["routes"].values() if row["entries"]
    )
    (workspace / entry["source"]).write_text("tampered\n")
    with pytest.raises(IntegrationError, match="source hash mismatch"):
        verify_materialization(contract, workspace=workspace, independent=False)


@pytest.mark.parametrize("attack", ["lower", "mde", "delete_pair", "duplicate_pair"])
def test_release_recomputes_raw_confirmation_pairs(tmp_path, attack):
    contract = rebaked_contract(tmp_path)
    release = copy.deepcopy(fake_release(contract, tmp_path))
    route = contract.routes[0].name
    gate = release["route_outcomes"][route]["candidate_gates"][0]
    if attack == "lower":
        gate["ci95_lower_log_ratio"] += 0.1
    elif attack == "mde":
        gate["practical_mde"] = 0.001
        gate["threshold_log_ratio"] = -integration.math.log1p(-0.001)
    elif attack == "delete_pair":
        gate["pairs"].pop()
    else:
        gate["pairs"].append(copy.deepcopy(gate["pairs"][0]))
        gate["pair_ids"].append(gate["pair_ids"][0])
        gate["n_pairs"] += 1
    release["release_id"] = integration._release_identity(release)
    with pytest.raises(IntegrationError, match="paired gate"):
        integration.validate_release(release, contract)
