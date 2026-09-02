from dataclasses import replace

import pytest

from asm_tuner.adapters import get_adapter
from asm_tuner.adapters.base import InactiveAxisError
from asm_tuner.schema import DType, validate_config
from asm_tuner.vendor_fixtures import default_seed, get_fixture


def test_bf16_solution205554_survives_and_renders_ragged_edge():
    config = get_fixture("bf16_solution205554")
    source = get_adapter("bf16").render_source([config])
    assert validate_config(config).valid
    assert (config.bm, config.bn, config.bk) == (128, 224, 64)
    assert "const int gx = (n + 224 - 1) / 224;" in source
    assert "col + VWS <= n" in source
    assert get_adapter("bf16").kernel_name(config) in source


def test_fp8_solution262788_survives_and_renders_native_fnuz():
    config = get_fixture("fp8_solution262788")
    source = get_adapter("fp8_rowwise").render_source([config])
    assert validate_config(config).valid
    assert (config.bm, config.bn, config.bk) == (128, 128, 128)
    assert (config.wave_grid.m, config.wave_grid.n) == (2, 2)
    assert (config.miwt.m, config.miwt.n) == (4, 4)
    assert "rocwmma::float8_fnuz_t" in source
    assert "a_scale[row] * b_scale[col + e]" in source


def test_vendor_fixtures_are_not_default_seeds():
    assert default_seed(DType.BF16).provenance["vendor_default"] is False
    assert default_seed(DType.FP8_E4M3_FNUZ).provenance["vendor_default"] is False
    assert default_seed(DType.BF16).config_id != get_fixture("bf16_solution205554").config_id


def test_pgr_variation_fails_instead_of_fake_search():
    config = get_fixture("bf16_solution205554")
    with pytest.raises(InactiveAxisError, match="pgr.*inactive"):
        get_adapter("bf16").render_source([config, replace(config, pgr=1)])
    assert get_adapter("bf16").axis_status()["pgr"]["state"] == "inactive"
