import datetime as dt
import shutil
from pathlib import Path

import pytest

import asm_tuner.integration as integration
from asm_tuner.contracts import TaskContract, sha256_file
from asm_tuner.integration import IntegrationError, materialize_release, verify_materialization
from asm_tuner.runtime_table import RuntimeTableStore


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "asm_tuner/manifests/tasks/dense_bf16_gemm_fused.json"


class FakeLibrary:
    def __getattr__(self, name):
        return object()


def fake_release(contract, tmp_path):
    config = contract.default_config
    source = tmp_path / "shard.hip"
    source.write_text("// generated source\n")
    artifact = tmp_path / "shard.so"
    artifact.write_bytes(b"compiled artifact")
    compile_manifest = tmp_path / "manifest.json"
    compile_manifest.write_text("{}\n")
    row = {
        "config_id": config.config_id,
        "config": config.to_dict(include_derived=True),
        "kernel": contract.adapter.kernel_name(config),
        "symbol": contract.adapter.launch_name(config),
        "isa_hash": "1" * 64,
        "cache_key": "cache",
        "compile_manifest": str(compile_manifest),
        "compile_manifest_sha256": sha256_file(compile_manifest),
        "artifact": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "source": str(source),
        "source_sha256": sha256_file(source),
    }
    release = {
        "schema": integration.RELEASE_SCHEMA,
        "task": contract.identity(),
        "protocol": dict(contract.protocol),
        "oracle": {
            "manifest_sha256": contract.oracle_sha256,
            "protocol_hash": contract.protocol["hash"],
            "compatible": True,
            "selection_metric": "single_call_latency",
        },
        "fallback": {"kind": "canonical", "reason": "outside suite"},
        "search_space_manifest": contract.value["search_space_manifest"],
        "variant": "seed",
        "selection_reason": "unit test",
        "route_selection": {route.name: config.config_id for route in contract.routes},
        "artifacts": [row],
    }
    release["release_id"] = integration._release_identity(release)
    release["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return release


def test_materialization_is_complete_atomic_and_independently_verifiable(tmp_path, monkeypatch):
    contract = TaskContract.load(MANIFEST, require_oracle_compatible=True)
    release = fake_release(contract, tmp_path)
    workspace = tmp_path / "workspace"
    shutil.copytree(contract.task_root, workspace,
                    ignore=shutil.ignore_patterns("build", ".torch_ext", "__pycache__"))
    monkeypatch.setattr(integration.ctypes, "CDLL", lambda path: FakeLibrary())

    receipt = materialize_release(contract, release, workspace=workspace, registry_id="cfg-unit")
    assert receipt["ok"] is True
    assert receipt["route_count"] == 11
    table = RuntimeTableStore(workspace / ".geak_tuner/runtime_table.json").load()
    assert table["suite"]["atomic"] is True
    assert table["suite"]["complete"] is True
    assert len(table["routes"]) == 11
    assert table["fallback"]["kind"] == "canonical"
    assert (workspace / "asm_tuner_selection.json").is_file()

    verified = verify_materialization(contract, workspace=workspace,
                                        rebuild=False, independent=True)
    assert verified["ok"] is True
    assert verified["independent_workspace"] is True
    assert verified["routes_verified"] == 11
    assert verified["task_digest"] == contract.task_digest


def test_materialized_source_tamper_fails_closed(tmp_path, monkeypatch):
    contract = TaskContract.load(MANIFEST, require_oracle_compatible=True)
    release = fake_release(contract, tmp_path)
    workspace = tmp_path / "workspace"
    shutil.copytree(contract.task_root, workspace,
                    ignore=shutil.ignore_patterns("build", ".torch_ext", "__pycache__"))
    monkeypatch.setattr(integration.ctypes, "CDLL", lambda path: FakeLibrary())
    materialize_release(contract, release, workspace=workspace, registry_id="cfg-unit")
    table = RuntimeTableStore(workspace / ".geak_tuner/runtime_table.json").load()
    entry = table["routes"][table["suite"]["route_ids"][0]]["entries"][0]
    (workspace / entry["source"]).write_text("tampered\n")
    with pytest.raises(IntegrationError, match="source hash mismatch"):
        verify_materialization(contract, workspace=workspace, independent=False)
