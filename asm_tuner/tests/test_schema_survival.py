from dataclasses import replace
from pathlib import Path

import pytest

from asm_tuner.adapters import get_adapter
from asm_tuner.adapters.base import InactiveAxisError
from asm_tuner.contracts import TaskContract
from asm_tuner.pruning import route_static_prune, static_prune
from asm_tuner.schema import DType, Route, validate_config
from asm_tuner.schema_survival import (
    default_seed,
    generated_survival_candidate,
    get_fixture,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "asm_tuner/manifests/tasks"


def test_bf16_vendor_record_survives_schema_normalization_but_plr_fails_closed():
    config = get_fixture("bf16_solution205554")
    assert validate_config(config).valid
    assert (config.bm, config.bn, config.bk) == (128, 224, 64)
    assert config.provenance["kind"] == "vendor_schema_survival"
    with pytest.raises(InactiveAxisError, match="plr=1.*inactive"):
        get_adapter("bf16").render_source([config])


def test_fp8_vendor_record_survives_schema_normalization_but_inactive_axes_fail_closed():
    config = get_fixture("fp8_solution262788")
    assert validate_config(config).valid
    assert (config.bm, config.bn, config.bk) == (128, 128, 128)
    assert (config.wave_grid.m, config.wave_grid.n) == (2, 2)
    assert (config.miwt.m, config.miwt.n) == (4, 4)
    with pytest.raises(InactiveAxisError, match="plr=1.*inactive"):
        get_adapter("fp8_rowwise").render_source([config])


def test_generated_survival_counterparts_pass_static_and_route_pruning():
    rows = [
        ("dense_bf16_gemm_fused.json", "bf16_solution205554", Route("large", 2048, 4096, 4096)),
        ("dense_fp8_gemm_rowwise.json", "fp8_solution262788", Route("large", 1024, 4096, 11008)),
    ]
    for manifest, fixture, route in rows:
        contract = TaskContract.load(MANIFESTS / manifest)
        config = generated_survival_candidate(contract.search_configs, fixture)
        assert static_prune([config]).accepted == (config,)
        assert route_static_prune([config], route).accepted == (config,)
        contract.adapter.validate(config)


def test_schema_survival_records_are_not_default_seeds():
    assert default_seed(DType.BF16).provenance["vendor_default"] is False
    assert default_seed(DType.FP8_E4M3_FNUZ).provenance["vendor_default"] is False
    assert default_seed(DType.BF16).config_id != get_fixture("bf16_solution205554").config_id
    source = get_adapter("bf16").render_source([default_seed(DType.BF16)])
    assert get_adapter("bf16").kernel_name(default_seed(DType.BF16)) in source


def test_pgr2_is_active_register_prefetch_and_does_not_change_lds():
    seed = default_seed(DType.BF16)
    pgr2 = replace(seed, pgr=2)
    adapter = get_adapter("bf16")
    source1 = adapter.render_source([seed])
    source2 = adapter.render_source([pgr2])
    assert source1 != source2
    assert "PGR2 owns exactly one future global stage" in source2
    assert "a_prefetch" in source2 and "b_prefetch" in source2
    assert seed.lds == pgr2.lds
    assert pgr2.a_lds.buffers == pgr2.b_lds.buffers == 1
    assert adapter.axis_status()["pgr"]["state"] == "active"


def test_unimplemented_axes_still_fail_closed():
    seed = default_seed(DType.BF16)
    adapter = get_adapter("bf16")
    for config, axis in (
        (replace(seed, plr=1), "plr"),
        (replace(seed, xcc=2), "xcc"),
        (replace(seed, gsu=2), "GSU"),
        (replace(seed, stream_k=1), "StreamK"),
    ):
        with pytest.raises((InactiveAxisError, ValueError), match=axis):
            adapter.render_source([config])
