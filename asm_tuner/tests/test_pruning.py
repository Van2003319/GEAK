from dataclasses import replace

from asm_tuner.pruning import (
    rank_for_route,
    route_static_prune,
    shape_sort_components,
    static_prune,
)
from asm_tuner.schema import DType, KernelConfig, Route
from asm_tuner.schema_survival import default_seed


def test_grid_smaller_than_cu_is_ranked_not_pruned():
    config = default_seed(DType.BF16)
    route = Route("tiny_ragged", 1, 17, 32)
    result = route_static_prune([config], route)
    assert result.accepted == (config,)
    assert config.grid_for(route)[2] == 1
    assert rank_for_route([config], route, cu_count=304) == [config]
    scores = shape_sort_components(config, route, cu_count=304)
    assert scores["grid_deficit"] > 0
    assert scores["reuse_inverse"] > 0


def test_only_exact_static_legality_prunes_not_grid_or_reuse():
    legal = default_seed(DType.FP8_E4M3_FNUZ)
    raw = legal.to_dict()
    raw["lds_scratch_bytes"] = 65536
    illegal = KernelConfig.from_dict(raw)
    result = static_prune([legal, illegal])
    assert [x.config_id for x in result.accepted] == [legal.config_id]
    assert len(result.rejected) == 1
    assert any("exceeds" in error for error in result.rejected[0].errors)


def test_route_static_dependencies_reject_noop_pgr2_and_stagger():
    seed = default_seed(DType.BF16)
    one_stage = Route("one_stage", 64, 64, seed.bk)
    result = route_static_prune(
        [replace(seed, pgr=2), replace(seed, stagger=8)], one_stage,
    )
    assert not result.accepted
    errors = [error for row in result.rejected for error in row.errors]
    assert any("PGR2 requires" in error for error in errors)
    assert any("stagger requires" in error for error in errors)


def test_pgr2_register_estimate_rejects_explosive_geometry_before_compile():
    seed = default_seed(DType.BF16).to_dict()
    seed.update({"bm": 256, "bn": 256, "bk": 128,
                 "wave_grid": [1, 2], "pgr": 2,
                 "lds_scratch_bytes": 2048})
    config = KernelConfig.from_dict(seed)
    report = static_prune([config])
    assert report.rejected
    assert any("VGPR" in error for error in report.rejected[0].errors)
