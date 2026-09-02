#!/usr/bin/env python3
"""GPU-free tests for the phase-independent config_search capability."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from dataclasses import replace

import pytest
from pathlib import Path

SCRIPT = Path(__file__).with_name("config_search.py")
SPEC = importlib.util.spec_from_file_location("config_search_under_test", SCRIPT)
assert SPEC and SPEC.loader
CS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CS
SPEC.loader.exec_module(CS)


def test_create_builds_hashes_registers_and_searches_a_precompiled_config(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    builder = workspace / "builder.py"
    builder.write_text(
        "from pathlib import Path\nimport sys\nPath(sys.argv[1]).write_bytes(b'compiled-config')\n",
        encoding="utf-8",
    )
    artifact = workspace / "compiled" / "cfg.bin"
    artifact.parent.mkdir()
    request = workspace / "request.json"
    request.write_text(json.dumps({
        "config": {"tile": 128, "waves": 2},
        "mechanism": "occupancy-two-waves",
        "builder_argv": [sys.executable, str(builder), str(artifact)],
        "artifact": str(artifact),
        "cwd": str(workspace),
    }), encoding="utf-8")
    registry = tmp_path / "state" / "config_search" / "registry.json"
    created = CS.create_config(registry, request_file=request, workspace=workspace, timeout=10)
    assert created["ok"] is True
    assert created["config"]["precompiled"] is True
    assert len(created["config"]["artifact_sha256"]) == 64
    found = CS.search_configs(registry, query={"tile": 128})
    assert found["phase_independent"] is True
    assert found["selection_contract"] == "precompiled-and-hash-valid-only"
    assert [row["id"] for row in found["matches"]] == [created["config"]["id"]]


def test_search_never_returns_uncompiled_or_hash_stale_entries(tmp_path: Path) -> None:
    artifact = tmp_path / "cfg.bin"
    artifact.write_bytes(b"compiled")
    registry = tmp_path / "registry.json"
    CS.register_config(registry, config={"tile": 64}, artifact=artifact)
    artifact.write_bytes(b"tampered")
    found = CS.search_configs(registry, query={})
    assert found["matches"] == []
    assert "artifact_hash_mismatch" in found["invalidated"][0]["reason"]


def test_search_records_hash_unavailable_without_selecting_entry(
    tmp_path: Path, monkeypatch,
) -> None:
    artifact = tmp_path / "cfg.bin"
    artifact.write_bytes(b"compiled")
    registry = tmp_path / "registry.json"
    CS.register_config(registry, config={"tile": 64}, artifact=artifact)
    original = CS._sha256

    def unavailable(path):
        if Path(path).resolve() == artifact.resolve():
            raise PermissionError("fixture unreadable")
        return original(path)

    monkeypatch.setattr(CS, "_sha256", unavailable)
    found = CS.search_configs(registry, query={})
    assert found["matches"] == []
    assert found["invalidated"][0]["reason"] == \
        "artifact_hash_unavailable(PermissionError)"


def test_query_matches_config_fields_not_prose(tmp_path: Path) -> None:
    artifact = tmp_path / "cfg.bin"
    artifact.write_bytes(b"compiled")
    registry = tmp_path / "registry.json"
    CS.register_config(registry, config={"tile": 64, "dtype": "bf16"}, artifact=artifact,
                       mechanism="mentions tile 128 in prose")
    assert CS.search_configs(registry, query={"tile": 128})["matches"] == []
    assert len(CS.search_configs(registry, query={"tile": 64})["matches"]) == 1


def test_create_refuses_artifact_outside_authorised_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    request = workspace / "request.json"
    request.write_text(json.dumps({
        "config": {"tile": 32},
        "builder_argv": [sys.executable, "-c", "pass"],
        "artifact": str(tmp_path / "escape.bin"),
        "cwd": str(workspace),
    }), encoding="utf-8")
    try:
        CS.create_config(tmp_path / "registry.json", request_file=request,
                         workspace=workspace, timeout=10)
    except CS.ConfigSearchError as exc:
        assert "artifact escapes" in str(exc)
    else:
        raise AssertionError("out-of-workspace artifact was accepted")


def test_interface_has_no_phase_lock() -> None:
    parser_help = CS._parser().format_help()
    assert "--phase" not in parser_help
    assert "phase-independent" in (CS.__doc__ or "").lower()


def test_exact_selection_revalidates_precompiled_artifact_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "cfg.bin"
    artifact.write_bytes(b"compiled")
    registry = tmp_path / "registry.json"
    registered = CS.register_config(registry, config={"tile": 256}, artifact=artifact)
    config_id = registered["config"]["id"]
    selected = CS.select_config(registry, config_id=config_id)
    assert selected["ok"] is True
    assert selected["phase_independent"] is True
    assert selected["config"]["config"] == {"tile": 256}
    artifact.write_bytes(b"stale")
    refused = CS.select_config(registry, config_id=config_id)
    assert refused["ok"] is False
    assert "artifact_hash_mismatch" in refused["reason"]
    assert CS.select_config(registry, config_id="missing")["reason"] == "config_not_found"


def test_relative_builder_cwd_is_resolved_inside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    build = workspace / "build"
    build.mkdir(parents=True)
    builder = build / "builder.py"
    builder.write_text(
        "from pathlib import Path\nPath('cfg.bin').write_bytes(b'compiled')\n",
        encoding="utf-8",
    )
    request = workspace / "request.json"
    request.write_text(json.dumps({
        "config": {"tile": 16},
        "builder_argv": [sys.executable, "builder.py"],
        "artifact": "cfg.bin",
        "cwd": "build",
    }), encoding="utf-8")
    created = CS.create_config(tmp_path / "registry.json", request_file=request,
                               workspace=workspace, timeout=10)
    assert created["ok"] is True


def test_reregister_repairs_stale_content_addressed_row_without_duplicate_id(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "cfg.bin"
    artifact.write_bytes(b"compiled")
    registry = tmp_path / "registry.json"
    first = CS.register_config(registry, config={"tile": 32}, artifact=artifact)
    config_id = first["config"]["id"]
    artifact.write_bytes(b"tampered")
    assert CS.select_config(registry, config_id=config_id)["ok"] is False
    artifact.write_bytes(b"compiled")
    repaired = CS.register_config(registry, config={"tile": 32}, artifact=artifact)
    assert repaired["created"] is True
    rows = json.loads(registry.read_text(encoding="utf-8"))["configs"]
    assert [row["id"] for row in rows] == [config_id]
    assert CS.select_config(registry, config_id=config_id)["ok"] is True


def test_prepare_orders_search_create_select_materialize_before_engineer(
    tmp_path: Path, monkeypatch,
) -> None:
    manifest = Path(__file__).resolve().parents[2] / "asm_tuner/manifests/tasks/dense_bf16_gemm_fused.json"
    loaded = CS.TaskContract.load(manifest)
    contract = replace(
        loaded, oracle_compatible=True,
        oracle={**loaded.oracle, "selection_metric": loaded.score_metric_id,
                "score_metric": dict(loaded.score_metric)},
    )
    monkeypatch.setattr(CS.TaskContract, "load", classmethod(
        lambda cls, *args, **kwargs: contract))
    registry = tmp_path / "registry.json"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    release_manifest = tmp_path / "release.json"
    release_manifest.write_text("{}\n")

    def fake_create(registry_value, **kwargs):
        config = {
            "kind": "asm_tuner_release", "task": contract.task,
            "adapter": contract.adapter_name, "task_digest": contract.task_digest,
            "protocol_hash": contract.protocol["hash"],
            "score_metric": dict(contract.score_metric),
            "score_metric_hash": contract.score_metric["hash"],
            "score_metric_id": contract.score_metric_id,
            "score_metric_inner": contract.score_metric_inner,
            "oracle_manifest_sha256": contract.oracle_sha256,
            "release_id": "release-unit", "variant": "seed", "priority": 100,
        }
        return CS.register_config(registry_value, config=config, artifact=release_manifest)

    def fake_materialize(registry_value, **kwargs):
        selected = CS.select_config(registry_value, config_id=kwargs["config_id"])
        return {
            "ok": True, "schema": CS.SCHEMA, "phase_independent": True,
            "config": selected["config"],
            "materialization": {"ok": True, "table_digest": "a" * 64},
        }

    monkeypatch.setattr(CS, "create_asm_config", fake_create)
    monkeypatch.setattr(CS, "search_asm_configs", fake_create)
    monkeypatch.setattr(CS, "materialize_asm_config", fake_materialize)
    monkeypatch.setenv("GEAK_GPU_LOCK_HELD", "1")
    monkeypatch.setenv("GEAK_GPU_FENCE_HELD", "1")
    result = CS.prepare_asm_config(
        registry, task_manifest=manifest, workspace=workspace, cache=tmp_path / "cache")
    assert result["ok"] is True
    assert result["prepared_before_engineer"] is True
    assert [row["operation"] for row in result["operations"]] == [
        "search", "create", "select", "materialize"]


def test_prepare_requires_both_gpu_lock_and_process_fence(tmp_path: Path, monkeypatch) -> None:
    manifest = Path(__file__).resolve().parents[2] / "asm_tuner/manifests/tasks/dense_bf16_gemm_fused.json"
    monkeypatch.delenv("GEAK_GPU_LOCK_HELD", raising=False)
    monkeypatch.delenv("GEAK_GPU_FENCE_HELD", raising=False)
    with pytest.raises(CS.ConfigSearchError, match="gpu_lock"):
        CS.prepare_asm_config(tmp_path / "registry.json", task_manifest=manifest,
                              workspace=tmp_path, cache=tmp_path / "cache")
    monkeypatch.setenv("GEAK_GPU_LOCK_HELD", "1")
    with pytest.raises(CS.ConfigSearchError, match="gpu_fence"):
        CS.prepare_asm_config(tmp_path / "registry.json", task_manifest=manifest,
                              workspace=tmp_path, cache=tmp_path / "cache")


def _tracked_publication_fixture(tmp_path: Path, monkeypatch):
    manifest = (
        Path(__file__).resolve().parents[2]
        / "asm_tuner/manifests/tasks/dense_bf16_gemm_fused.json"
    )
    contract = CS.TaskContract.load(manifest, require_oracle_compatible=True)
    workspace = tmp_path / "clean-clone-task"
    shutil.copytree(
        contract.task_root, workspace,
        ignore=shutil.ignore_patterns(
            ".geak_tuner", "build", ".torch_ext", "__pycache__", "*.pyc", "*.o",
        ),
    )
    assert not (workspace / ".geak_tuner").exists()
    monkeypatch.setattr(CS.TaskContract, "load", classmethod(
        lambda cls, *args, **kwargs: contract))
    monkeypatch.setenv("GEAK_GPU_LOCK_HELD", "1")
    monkeypatch.setenv("GEAK_GPU_FENCE_HELD", "1")
    return manifest, contract, workspace


@pytest.mark.parametrize("previously_invalidated", [False, True])
def test_prepare_hydrates_missing_implicit_tracked_release_in_clean_clone(
    tmp_path: Path, monkeypatch, previously_invalidated: bool,
) -> None:
    manifest, contract, workspace = _tracked_publication_fixture(tmp_path, monkeypatch)
    registry = contract.materialization_path("registry", task_root=workspace)
    table = contract.materialization_path("runtime_table", task_root=workspace)
    selection = contract.materialization_path("selection_file", task_root=workspace)
    old_selection = json.loads(selection.read_text())
    if previously_invalidated:
        registry_doc = json.loads(registry.read_text())
        incumbent = next(
            row for row in registry_doc["configs"]
            if row["id"] == old_selection["registry_id"]
        )
        incumbent.update({
            "status": "invalidated",
            "invalidated_reason": "artifact_missing",
            "invalidated_unix_ns": 1,
        })
        registry.write_text(json.dumps(registry_doc))
    release_manifest = tmp_path / "cache/new-release.json"
    release_manifest.parent.mkdir()
    release_manifest.write_text("{}\n")

    def fake_create(registry_value, **kwargs):
        config = {
            "kind": "asm_tuner_release",
            "task": contract.task,
            "adapter": contract.adapter_name,
            "task_digest": contract.task_digest,
            "protocol_hash": contract.protocol["hash"],
            "score_metric": dict(contract.score_metric),
            "score_metric_hash": contract.score_metric["hash"],
            "score_metric_id": contract.score_metric_id,
            "score_metric_inner": contract.score_metric_inner,
            "oracle_manifest_sha256": contract.oracle_sha256,
            "release_id": "hydrated-release",
            "variant": "unit",
            "config_ids": [],
            "priority": 0,
        }
        return CS.register_config(
            registry_value, config=config, artifact=release_manifest,
            mechanism="bounded hydration fixture",
        )

    def fake_materialize(registry_value, **kwargs):
        selected = CS.select_config(registry_value, config_id=kwargs["config_id"])
        generation = {
            "generation": "hydrated",
            "registry_id": kwargs["config_id"],
            "release_id": "hydrated-release",
        }
        CS._atomic_json(table, {**generation, "kind": "runtime_table"})
        CS._atomic_json(selection, {**generation, "kind": "selection"})
        CS._atomic_json(registry, {
            "schema": CS.SCHEMA,
            "schema_version": CS.SCHEMA_VERSION,
            "configs": [selected["config"]],
            "generation": "hydrated",
        })
        return {
            "ok": True,
            "schema": CS.SCHEMA,
            "phase_independent": True,
            "config": selected["config"],
            "materialization": {"ok": True, "table_digest": "a" * 64},
        }

    monkeypatch.setattr(CS, "search_asm_configs", fake_create)
    monkeypatch.setattr(CS, "create_asm_config", fake_create)
    monkeypatch.setattr(CS, "materialize_asm_config", fake_materialize)
    result = CS.prepare_asm_config(
        registry, task_manifest=manifest, workspace=workspace, cache=tmp_path / "cache",
    )
    assert result["ok"] is True
    operations = result["operations"]
    assert [row["operation"] for row in operations] == [
        "validate_tracked_selection", "search", "clear_implicit_requested",
        "search_after_invalidation", "create", "select", "materialize",
    ]
    invalidated = operations[1]["invalidated"]
    assert invalidated == ([] if previously_invalidated else [{
        "id": old_selection["registry_id"], "reason": "artifact_missing",
    }])
    assert operations[2]["reason"] == "artifact_missing"
    new_id = result["config"]["id"]
    assert new_id != old_selection["registry_id"]
    assert all(
        json.loads(path.read_text())["generation"] == "hydrated"
        for path in (registry, table, selection)
    )


def test_prepare_explicit_missing_id_remains_fail_closed(
    tmp_path: Path, monkeypatch,
) -> None:
    manifest, contract, workspace = _tracked_publication_fixture(tmp_path, monkeypatch)
    registry = contract.materialization_path("registry", task_root=workspace)
    selection = contract.materialization_path("selection_file", task_root=workspace)
    requested = json.loads(selection.read_text())["registry_id"]
    monkeypatch.setattr(
        CS, "search_asm_configs",
        lambda *args, **kwargs: pytest.fail("explicit missing id must not create"),
    )
    result = CS.prepare_asm_config(
        registry, task_manifest=manifest, workspace=workspace,
        cache=tmp_path / "cache", config_id=requested,
    )
    assert result["ok"] is False
    assert result["reason"] == "requested_config_not_viable"
    assert all(row["operation"] != "create" for row in result["operations"])
    requested_op = result["operations"][-1]
    assert requested_op["operation"] == "select_requested"
    assert requested_op["source"] == "explicit"
    assert requested_op["reason"] == "status_not_ready('invalidated')"
    assert result["operations"][0]["invalidated"][0]["reason"] == "artifact_missing"


def test_prepare_missing_artifact_with_stale_registry_contract_does_not_recover(
    tmp_path: Path, monkeypatch,
) -> None:
    manifest, contract, workspace = _tracked_publication_fixture(tmp_path, monkeypatch)
    registry = contract.materialization_path("registry", task_root=workspace)
    selection = contract.materialization_path("selection_file", task_root=workspace)
    requested = json.loads(selection.read_text())["registry_id"]
    registry_doc = json.loads(registry.read_text())
    row = next(row for row in registry_doc["configs"] if row["id"] == requested)
    row["config"]["task_digest"] = "0" * 64
    registry.write_text(json.dumps(registry_doc))
    monkeypatch.setattr(
        CS, "search_asm_configs",
        lambda *args, **kwargs: pytest.fail("stale registry row must not create"),
    )
    result = CS.prepare_asm_config(
        registry, task_manifest=manifest, workspace=workspace, cache=tmp_path / "cache",
    )
    assert result["ok"] is False
    assert result["reason"] == "requested_config_not_viable"
    assert all(row["operation"] != "create" for row in result["operations"])
    assert result["operations"][-1]["source"] == "tracked_selection"


def test_prepare_stale_tracked_selection_semantics_remain_fail_closed(
    tmp_path: Path, monkeypatch,
) -> None:
    manifest, contract, workspace = _tracked_publication_fixture(tmp_path, monkeypatch)
    registry = contract.materialization_path("registry", task_root=workspace)
    table = contract.materialization_path("runtime_table", task_root=workspace)
    selection = contract.materialization_path("selection_file", task_root=workspace)
    value = json.loads(selection.read_text())
    value["source_digest"] = "0" * 64
    selection.write_text(json.dumps(value))
    registry_before = registry.read_bytes()
    table_before = table.read_bytes()
    monkeypatch.setattr(
        CS, "search_configs",
        lambda *args, **kwargs: pytest.fail("stale selection must fail before registry search"),
    )
    result = CS.prepare_asm_config(
        registry, task_manifest=manifest, workspace=workspace, cache=tmp_path / "cache",
    )
    assert result["ok"] is False
    assert result["reason"] == "tracked_selection_contract_mismatch"
    assert result["operations"][0]["mismatches"]["source_digest"]["tracked"] == "0" * 64
    assert registry.read_bytes() == registry_before
    assert table.read_bytes() == table_before
