import copy
from functools import lru_cache
from pathlib import Path

import pytest

from asm_tuner.contracts import TaskContract
from asm_tuner.search_space import SearchSpaceError, plan_route_candidates, validate_search_manifest


ROOT = Path(__file__).resolve().parents[2]
BF16 = ROOT / "asm_tuner/manifests/tasks/dense_bf16_gemm_fused.json"
FP8 = ROOT / "asm_tuner/manifests/tasks/dense_fp8_gemm_rowwise.json"


@lru_cache(maxsize=16)
def _contract_and_plan(path, route_name, limit=8):
    contract = TaskContract.load(Path(path))
    route = next(route for route in contract.routes if route.name == route_name)
    plan = plan_route_candidates(
        contract.search_space, contract.search_configs, route, cu_count=304, limit=limit,
    )
    return contract, route, plan


def _geometry(config):
    return (
        config.bm, config.bn, config.bk,
        config.wave_grid.m, config.wave_grid.n, config.pgr,
    )


def test_route_candidate_prefix_is_bounded_and_stratifies_architecture_and_profiles():
    contract, route, plan = _contract_and_plan(str(BF16), "prefill_m2048_square")
    assert len(plan.candidates) == 8
    assert {config.pgr for config in plan.candidates} == {1, 2}
    assert len({config.nw for config in plan.candidates}) >= 2
    assert len({row["diversity"]["tile_family"] for row in plan.selection}) >= 3
    assert len({row["diversity"]["cta_tail_band"] for row in plan.selection}) >= 2
    sweeps = {config.provenance.get("sweep", "declared") for config in plan.candidates}
    assert {"pipeline_memory", "global_epilogue_io", "mapping_schedule"} <= sweeps
    assert any(
        config.a_lds.buffers != 1 or config.b_lds.buffers != 1 or
        config.a_lds.pad_elements or config.b_lds.pad_elements
        for config in plan.candidates
    )
    assert any(config.store.mode.value == "non_temporal" for config in plan.candidates)
    assert any(config.wgm != 1 or config.stagger != 0 for config in plan.candidates)
    assert plan.accepted_count > len(plan.candidates)
    assert all(row["score_components"] and row["selection_reason"] for row in plan.selection)


@pytest.mark.parametrize("manifest,route_name,target", [
    (BF16, "prefill_m2048_square", (128, 224, 64, 2, 2, 2)),
    (FP8, "prefill_m1024_down", (128, 128, 128, 2, 2, 2)),
])
def test_vendor_like_geometry_is_domain_generated_and_reaches_bootstrap_top8(
    manifest, route_name, target,
):
    contract, route, plan = _contract_and_plan(str(manifest), route_name)
    generated = [config for config in contract.search_configs if _geometry(config) == target]
    assert generated
    assert all(config.provenance.get("vendor_default") is False for config in generated)
    position = next(
        index for index, config in enumerate(plan.candidates, 1) if _geometry(config) == target
    )
    assert position <= 8
    row = plan.selection[position - 1]
    assert row["pareto_rank"] == 0
    assert row["score_components"]["compute_regime_weight"] > 0.9
    assert row["score_components"]["cta_tail_fraction"] <= 0.2
    assert row["score_components"]["lds_read_per_mfma"] <= 0.25
    assert row["score_components"]["weighted_score"] == row["weighted_score"]


@pytest.mark.parametrize("manifest", [BF16, FP8])
def test_decode_prefix_retains_small_tiles(manifest):
    contract, route, plan = _contract_and_plan(str(manifest), "decode_m2_square")
    assert any(config.bm <= 64 and config.bn <= 64 for config in plan.candidates)
    assert len({config.nw for config in plan.candidates}) >= 2
    assert {config.pgr for config in plan.candidates} == {1, 2}
    compute_weights = [
        row["score_components"]["compute_regime_weight"] for row in plan.selection
    ]
    assert max(compute_weights) < 0.1


def test_manifest_cannot_turn_an_inactive_axis_into_a_search_domain():
    contract = TaskContract.load(BF16)
    value = copy.deepcopy(contract.search_space)
    value["domains"]["plr"] = [0, 1]
    value["generator"]["profile_sweeps"].append({"name": "fake_plr", "axes": ["plr"]})
    with pytest.raises(SearchSpaceError, match="plr.*inactive"):
        validate_search_manifest(value, "bf16")


def test_manifest_budget_must_cover_declared_halving_rounds():
    contract = TaskContract.load(BF16)
    value = copy.deepcopy(contract.search_space)
    value["search_policy"]["bootstrap"]["max_route_arm_evaluations"] = 11
    with pytest.raises(SearchSpaceError, match="cannot cover bounded halving cost 12"):
        validate_search_manifest(value, "bf16")
