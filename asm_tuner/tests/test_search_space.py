import copy
from pathlib import Path

import pytest

from asm_tuner.contracts import TaskContract
from asm_tuner.search_space import SearchSpaceError, plan_route_candidates, validate_search_manifest


ROOT = Path(__file__).resolve().parents[2]
BF16 = ROOT / "asm_tuner/manifests/tasks/dense_bf16_gemm_fused.json"


def test_route_candidate_prefix_is_bounded_and_stratifies_pgr_and_sweeps():
    contract = TaskContract.load(BF16)
    route = next(route for route in contract.routes if route.name == "prefill_m2048_square")
    plan = plan_route_candidates(
        contract.search_space, contract.search_configs, route, cu_count=304, limit=8,
    )
    assert len(plan.candidates) == 8
    assert {config.pgr for config in plan.candidates} == {1, 2}
    sweeps = {config.provenance.get("sweep", "declared") for config in plan.candidates}
    assert {"pipeline_memory", "global_epilogue_io", "mapping_schedule"} <= sweeps
    assert "declared" in sweeps
    assert any(
        config.a_lds.buffers != 1 or config.b_lds.buffers != 1 or
        config.a_lds.pad_elements or config.b_lds.pad_elements
        for config in plan.candidates
    )
    assert any(config.store.mode.value == "non_temporal" for config in plan.candidates)
    assert any(config.wgm != 1 or config.stagger != 0 for config in plan.candidates)
    assert plan.accepted_count > len(plan.candidates)


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
