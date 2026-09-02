import hashlib
import json
import shutil
from pathlib import Path

import pytest

from asm_tuner.contracts import TaskContract
from asm_tuner.schema_survival import generated_survival_candidate


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = ROOT / "asm_tuner" / "manifests" / "tasks"
MANIFESTS = sorted(MANIFEST_DIR.glob("*.json"))


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: p.stem)
def test_contract_binds_cases_builders_correctness_protocol_and_oracle(manifest, tmp_path):
    contract = TaskContract.load(manifest, require_oracle_compatible=True)
    assert len(contract.routes) == 11
    assert [route.name for route in contract.routes] == [case["id"] for case in contract.cases]
    assert contract.protocol["schema"] == "geak.measurement-protocol/v2"
    assert contract.score_metric["schema"] == "geak.score-metric/v1"
    assert contract.score_metric_id == "sustained_throughput"
    assert contract.score_metric_inner > 1
    assert contract.oracle_compatible is True
    assert contract.oracle["frozen"] is True
    assert len(contract.oracle["routes"]) == 11
    assert set(contract.oracle["routes"]) == {case["id"] for case in contract.cases}
    assert contract.oracle["protocol_hash"] == contract.protocol["hash"]
    assert contract.oracle["score_metric"] == contract.score_metric
    assert contract.oracle["selection_metric"] == contract.score_metric_id

    for drift in ("metric", "protocol"):
        copied = tmp_path / drift
        shutil.copytree(
            contract.task_root, copied,
            ignore=shutil.ignore_patterns("build", ".torch_ext"),
        )
        oracle_path = copied / "oracle_routes.v2.json"
        oracle = json.loads(oracle_path.read_text())
        if drift == "metric":
            oracle["score_metric"]["inner"] += 1
        else:
            oracle["protocol_hash"] = "0" * 64
            oracle["score_metric"]["protocol_hash"] = "0" * 64
        metric_body = dict(oracle["score_metric"])
        metric_body.pop("hash")
        oracle["score_metric"]["hash"] = hashlib.sha256(json.dumps(
            metric_body, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("utf-8")).hexdigest()
        oracle_path.write_text(json.dumps(oracle))
        with pytest.raises(Exception, match="re-run oracle-bakeoff"):
            TaskContract.load(
                manifest, task_root=copied, require_oracle_compatible=True)

    assert len(contract.task_digest) == 64
    assert set(contract.binding_hashes) == {"cases", "input_builder", "correctness"}
    assert "scripts/task_runner.py" in contract.semantic_hashes
    assert "harness_lib.py" in contract.semantic_hashes
    assert "geak_tuner_runtime.py" in contract.semantic_hashes
    assert len(contract.source_digest) == 64
    assert contract.materialization_relative_path("registry") == Path("asm_tuner_registry.json")
    assert contract.materialization_relative_path("runtime_table") == Path(
        "asm_tuner_runtime_table.json")
    assert contract.materialization_relative_path("selection_file") == Path(
        "asm_tuner_selection.json")
    assert len(contract.identity()["search_space_sha256"]) == 64


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: p.stem)
def test_search_space_expands_dependent_domains_with_bounded_policy(manifest):
    contract = TaskContract.load(manifest)
    materialized = contract.search_materialization
    assert len(materialized.generated) > 50_000
    assert len(materialized.accepted) > 30_000
    assert len(materialized.rejected) > 0
    assert contract.adapter.axis_status()["pgr"]["state"] == "active"
    assert contract.search_space["domains"]["pgr"] == [1, 2]
    assert contract.search_space["domains"]["wave_count"] == [2, 4, 8]
    assert contract.search_space["fail_closed"]["inactive_neutral_values"] == {
        "plr": 0, "xcc": 1,
    }
    assert contract.search_space["search_policy"]["full"] == {
        "cu_count": 304,
        "eta": 2,
        "first_round_top_n": 12,
        "max_compiled_candidates": 160,
        "max_route_arm_evaluations": 24,
        "rounds": 3,
        "shard_size": 12,
        "top_k": 2,
    }
    if contract.adapter_name == "bf16":
        required = {64, 96, 112, 128, 160, 176, 192, 224, 256}
        assert required <= set(contract.search_space["domains"]["bm"])
        assert required <= set(contract.search_space["domains"]["bn"])
        assert {32, 64, 128} <= set(contract.search_space["domains"]["bk"])
        assert contract.search_space["bootstrap_search"]["cases"] == ["prefill_m2048_square"]
    else:
        assert 128 in contract.search_space["domains"]["bm"]
        assert 128 in contract.search_space["domains"]["bn"]
        assert 128 in contract.search_space["domains"]["bk"]
        assert contract.search_space["bootstrap_search"]["cases"] == ["prefill_m1024_down"]


@pytest.mark.parametrize(
    "manifest_name,fixture_name,geometry,miwt",
    [
        ("dense_bf16_gemm_fused.json", "bf16_solution205554",
         (128, 224, 64, 2, 2), (4, 7)),
        ("dense_fp8_gemm_rowwise.json", "fp8_solution262788",
         (128, 128, 128, 2, 2), (4, 4)),
    ],
)
def test_schema_survival_points_are_generated_not_default_winners(
    manifest_name, fixture_name, geometry, miwt,
):
    contract = TaskContract.load(MANIFEST_DIR / manifest_name)
    config = generated_survival_candidate(contract.search_configs, fixture_name)
    assert (config.bm, config.bn, config.bk,
            config.wave_grid.m, config.wave_grid.n) == geometry
    assert (config.miwt.m, config.miwt.n) == miwt
    assert config.pgr == 2 and config.plr == 0 and config.xcc == 1
    assert config.a_lds.buffers == config.b_lds.buffers == 1
    assert config.config_id != contract.default_config.config_id
    assert config.config_id not in contract.search_space["variants"]


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: p.stem)
def test_semantic_task_drift_changes_digest(manifest, tmp_path):
    original = TaskContract.load(manifest)
    copied = tmp_path / "task"
    shutil.copytree(original.task_root, copied, ignore=shutil.ignore_patterns("build", ".torch_ext"))
    runner = copied / "scripts" / "task_runner.py"
    runner.write_text(runner.read_text() + "\n# semantic drift\n")
    changed = TaskContract.load(manifest, task_root=copied)
    assert changed.task_digest != original.task_digest


def test_runtime_loader_is_identical_for_bf16_and_fp8():
    bf16 = ROOT / "examples/tasks/dense_bf16_gemm_fused/geak_tuner_runtime.py"
    fp8 = ROOT / "examples/tasks/dense_fp8_gemm_rowwise/geak_tuner_runtime.py"
    assert bf16.read_bytes() == fp8.read_bytes()
    text = bf16.read_text()
    assert "ranks are not contiguous" in text
    assert '_TABLE_RELATIVE_PATH = "asm_tuner_runtime_table.json"' in text
    for wrapper in (
        ROOT / "examples/tasks/dense_bf16_gemm_fused/gemm_wrapper.py",
        ROOT / "examples/tasks/dense_fp8_gemm_rowwise/gemm_wrapper.py",
    ):
        text = wrapper.read_text()
        assert "_tuner_dispatch" in text
        assert text.index("_tuner_dispatch") < text.index("from kernel_loader import gemm_ext")


def test_committed_gpu_evidence_proves_pgr2_correctness_and_distinct_isa():
    evidence = json.loads((ROOT / "asm_tuner/evidence/gfx942_pgr2_smoke_20260902.json").read_text())
    assert evidence["arch"] == "gfx942"
    manifests = {
        "bf16": MANIFEST_DIR / "dense_bf16_gemm_fused.json",
        "fp8_rowwise": MANIFEST_DIR / "dense_fp8_gemm_rowwise.json",
    }
    for adapter, row in evidence["adapters"].items():
        current = TaskContract.load(manifests[adapter])
        # This historical GPU evidence predates the score-metric contract and is not rewritten to
        # claim a run that did not happen. New formal search evidence will carry current.task_digest.
        assert len(row["task_digest"]) == 64
        assert current.score_metric_id == "sustained_throughput"
        assert row["source_version"] == "rocwmma-pgr2-v2"
        assert row["search_space"]["accepted"] > 30_000
        assert row["pgr1"]["correctness"]["passed"] is True
        assert row["pgr2"]["correctness"]["passed"] is True
        assert row["pgr1"]["isa_hash"] != row["pgr2"]["isa_hash"]
        assert row["pgr2"]["scratch_bytes_per_workitem"] == 0
        assert row["pgr2"]["lds_bytes"] == row["pgr1"]["lds_bytes"]
