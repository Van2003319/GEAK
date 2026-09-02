from dataclasses import replace

import pytest

from asm_tuner.adapters import get_adapter
from asm_tuner.adapters.base import InactiveAxisError
from asm_tuner.schema import DType, validate_config
from asm_tuner.schema_survival import default_seed, get_fixture


def test_bf16_vendor_record_survives_schema_normalization_but_is_not_executable():
    config = get_fixture("bf16_solution205554")
    assert validate_config(config).valid
    assert (config.bm, config.bn, config.bk) == (128, 224, 64)
    assert config.provenance["kind"] == "vendor_schema_survival"
    with pytest.raises(InactiveAxisError, match="pgr=2.*inactive"):
        get_adapter("bf16").render_source([config])


def test_fp8_vendor_record_survives_schema_normalization_but_is_not_a_kernel():
    config = get_fixture("fp8_solution262788")
    assert validate_config(config).valid
    assert (config.bm, config.bn, config.bk) == (128, 128, 128)
    assert (config.wave_grid.m, config.wave_grid.n) == (2, 2)
    assert (config.miwt.m, config.miwt.n) == (4, 4)
    with pytest.raises(InactiveAxisError, match="pgr=2.*inactive"):
        get_adapter("fp8_rowwise").render_source([config])


def test_schema_survival_records_are_not_default_seeds():
    assert default_seed(DType.BF16).provenance["vendor_default"] is False
    assert default_seed(DType.FP8_E4M3_FNUZ).provenance["vendor_default"] is False
    assert default_seed(DType.BF16).config_id != get_fixture("bf16_solution205554").config_id
    source = get_adapter("bf16").render_source([default_seed(DType.BF16)])
    assert get_adapter("bf16").kernel_name(default_seed(DType.BF16)) in source


def test_pgr_variation_fails_instead_of_fake_search():
    seed = default_seed(DType.BF16)
    with pytest.raises(InactiveAxisError, match="pgr=2.*inactive"):
        get_adapter("bf16").render_source([replace(seed, pgr=2)])
    with pytest.raises(InactiveAxisError, match="pgr.*inactive"):
        get_adapter("bf16").render_source([seed, replace(seed, pgr=2)])
    assert get_adapter("bf16").axis_status()["pgr"]["state"] == "inactive"
