from dataclasses import replace

from asm_tuner.schema import DType, KernelConfig, validate_config
from asm_tuner.schema_survival import default_seed, get_fixture


def test_bf16_fixture_derives_logical_wave_tile_and_miwt():
    config = get_fixture("bf16_solution205554")
    assert (config.bm, config.bn, config.bk, config.nw) == (128, 224, 64, 4)
    assert (config.wave_tile.m, config.wave_tile.n) == (64, 112)
    assert (config.miwt.m, config.miwt.n) == (4, 7)
    assert config.provenance["raw_macro_tile"] == [224, 128]
    assert config.provenance["raw_miwt"] == [7, 4]
    assert validate_config(config).valid


def test_lds_accounts_independent_buffers_pads_and_scratch_not_pgr():
    seed = default_seed(DType.BF16)
    value = seed.to_dict()
    value.update({
        "pgr": 7,
        "a_lds": {"buffers": 2, "pad": 8},
        "b_lds": {"buffers": 3, "pad": 16},
        "lds_scratch_bytes": 4096,
    })
    config = KernelConfig.from_dict(value)
    assert config.lds.a_bytes == 2 * 64 * (32 + 8) * 2
    assert config.lds.b_bytes == 3 * 64 * (32 + 16) * 2
    assert config.lds.total_bytes == config.lds.a_bytes + config.lds.b_bytes + 4096
    assert config.lds.pgr_counted_as_buffer is False
    assert replace(config, pgr=1).lds == config.lds
    assert validate_config(config).valid


def test_illegal_dependency_chain_is_precise():
    seed = default_seed(DType.BF16).to_dict()
    seed["bk"] = 24
    seed["wave_grid"] = [3, 2]
    report = validate_config(KernelConfig.from_dict(seed))
    assert not report.valid
    assert "BK must be divisible by MFMA.K" in report.errors
    assert any("WaveTile" in error or "divide" in error for error in report.errors)


def test_fp8_fnuz_arch_dependency_is_enforced():
    raw = default_seed(DType.FP8_E4M3_FNUZ).to_dict()
    raw["arch"] = "gfx90a"
    report = validate_config(KernelConfig.from_dict(raw))
    assert not report.valid
    assert any("only on gfx942" in error for error in report.errors)
