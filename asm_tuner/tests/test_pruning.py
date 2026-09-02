from asm_tuner.pruning import rank_for_route, static_prune
from asm_tuner.schema import DType, KernelConfig, Route
from asm_tuner.schema_survival import default_seed


def test_grid_smaller_than_cu_is_ranked_not_pruned():
    config = default_seed(DType.BF16)
    route = Route("tiny_ragged", 1, 17, 32)
    result = static_prune([config])
    assert result.accepted == (config,)
    assert config.grid_for(route)[2] == 1
    assert rank_for_route([config], route, cu_count=304) == [config]


def test_only_static_legality_prunes():
    legal = default_seed(DType.FP8_E4M3_FNUZ)
    raw = legal.to_dict()
    raw["lds_scratch_bytes"] = 65536
    illegal = KernelConfig.from_dict(raw)
    result = static_prune([legal, illegal])
    assert [x.config_id for x in result.accepted] == [legal.config_id]
    assert len(result.rejected) == 1
    assert any("exceeds" in error for error in result.rejected[0].errors)
