import json
import shutil
from pathlib import Path

import pytest

from asm_tuner.adapters.base import InactiveAxisError
from asm_tuner.contracts import TaskContract


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = ROOT / "asm_tuner" / "manifests" / "tasks"
MANIFESTS = sorted(MANIFEST_DIR.glob("*.json"))


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: p.stem)
def test_contract_binds_cases_builders_correctness_protocol_and_oracle(manifest):
    contract = TaskContract.load(manifest, require_oracle_compatible=True)
    assert len(contract.routes) == 11
    assert [route.name for route in contract.routes] == [case["id"] for case in contract.cases]
    assert contract.protocol["schema"] == "geak.measurement-protocol/v2"
    assert contract.oracle_compatible is True
    assert len(contract.task_digest) == 64
    assert set(contract.binding_hashes) == {"cases", "input_builder", "correctness"}
    assert "scripts/task_runner.py" in contract.semantic_hashes
    assert "harness_lib.py" in contract.semantic_hashes
    assert "geak_tuner_runtime.py" in contract.semantic_hashes


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: p.stem)
def test_search_space_has_real_seed_and_nondefault_geometry_only(manifest):
    contract = TaskContract.load(manifest)
    seed = contract.config_variant("seed")
    geometry = contract.config_variant("geometry_m32_n64_w1x4")
    assert seed.config_id != geometry.config_id
    assert (seed.bm, seed.bn, seed.wave_grid.m, seed.wave_grid.n) == (64, 64, 2, 2)
    assert (geometry.bm, geometry.bn, geometry.wave_grid.m, geometry.wave_grid.n) == (32, 64, 1, 4)
    assert contract.search_space["fail_closed"]["inactive_neutral_values"] == {
        "pgr": 1, "plr": 0, "xcc": 1,
    }


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
    for wrapper in (
        ROOT / "examples/tasks/dense_bf16_gemm_fused/gemm_wrapper.py",
        ROOT / "examples/tasks/dense_fp8_gemm_rowwise/gemm_wrapper.py",
    ):
        text = wrapper.read_text()
        assert "_tuner_dispatch" in text
        assert text.index("_tuner_dispatch") < text.index("from kernel_loader import gemm_ext")


def test_committed_gpu_evidence_proves_symbols_changed():
    evidence = json.loads((ROOT / "asm_tuner/evidence/gfx942_canonical_smoke_20260902.json").read_text())
    assert evidence["dispatch_changed"] == {"bf16": True, "fp8": True}
    assert evidence["protocol"]["schema"] == "geak.measurement-protocol/v2"
    assert all(row["correctness"] == "pass" for row in evidence["smokes"].values())
    assert all(row["independent_rebuild"]["status"] == "pass"
               for row in evidence["smokes"].values())
    assert all(search["status"] == "pass"
               for search in evidence["protocol_v2_paired_search"].values())
