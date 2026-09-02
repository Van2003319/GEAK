import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from asm_tuner.adapters import get_adapter
from asm_tuner.compiler import ShardCompiler, shard_configs
from asm_tuner.metadata import MetadataError
from asm_tuner.schema import DType
from asm_tuner.schema_survival import default_seed


def _assembly(names, lds, scratch=0, vgpr=48):
    chunks = []
    for name in names:
        chunks.append(f"""
{name}: ; @{name}
\ts_mov_b32 s0, 0
\ts_endpgm
\t.amdhsa_kernel {name}
\t\t.amdhsa_group_segment_fixed_size {lds}
\t\t.amdhsa_private_segment_fixed_size {scratch}
\t\t.amdhsa_next_free_vgpr {vgpr}
\t\t.amdhsa_next_free_sgpr 32
\t\t.amdhsa_accum_offset 32
\t.end_amdhsa_kernel
.Lfunc_end0:
\t.set {name}.num_vgpr, 30
\t.set {name}.num_agpr, 16
    .name:           {name}
    .sgpr_count:     28
    .vgpr_count:     {vgpr}
""")
    return "\n".join(chunks)


def test_sharding_is_balanced_8_to_16_or_explicit_on_demand():
    seed = default_seed(DType.BF16)
    configs = [replace(seed, wgm=index + 1) for index in range(25)]
    assert [len(x) for x in shard_configs(configs, shard_size=12)] == [9, 8, 8]
    assert [len(x) for x in shard_configs(configs[:3], on_demand=True)] == [1, 1, 1]
    with pytest.raises(ValueError, match="at least 8"):
        shard_configs(configs[:3], shard_size=12)


def test_compile_cache_metadata_and_cross_shard_isa_dedupe(tmp_path, monkeypatch):
    seed = default_seed(DType.BF16)
    configs = [seed, replace(seed, wgm=2)]
    calls = []

    def fake_run(command, text=True, stdout=None, stderr=None):
        calls.append(list(command))
        if "--version" in command:
            return SimpleNamespace(returncode=0, stdout="fake hipcc 7.2")
        output = Path(command[command.index("-o") + 1])
        source = Path(next(value for value in command if str(value).endswith(".hip")))
        names = sorted(set(re.findall(r"void (geak_bf16_[0-9a-f]+)\(", source.read_text())))
        if "-S" in command:
            output.write_text(_assembly(names, seed.lds.total_bytes))
        else:
            output.write_bytes(b"artifact")
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr("asm_tuner.compiler.subprocess.run", fake_run)
    compiler = ShardCompiler(tmp_path / "cache", hipcc="/fake/hipcc")
    first = compiler.compile(get_adapter("bf16"), configs, on_demand=True, shared=True)
    assert [item.cache_hit for item in first] == [False, False]
    duplicate_rows = [kernel["duplicate_of"] for result in first for kernel in result.manifest["kernels"]]
    assert sum(value is not None for value in duplicate_rows) == 1
    row = first[0].manifest["kernels"][0]
    assert row["vgpr"] == 30
    assert row["agpr"] == 16
    assert row["unified_vgpr"] == 48
    assert row["sgpr"] == 28
    assert row["lds_bytes"] == seed.lds.total_bytes
    assert row["scratch_bytes_per_workitem"] == 0
    assert row["occupancy"]["workgroups_per_cu"] > 0
    assert first[0].manifest["metadata_gate"].startswith("no-spill")
    assert (tmp_path / "cache" / "isa_index.json").exists()

    calls_before = len(calls)
    second = compiler.compile(get_adapter("bf16"), configs, on_demand=True, shared=True)
    assert [item.cache_hit for item in second] == [True, True]
    assert len(calls) == calls_before


def test_metadata_gate_rejects_register_spill(tmp_path, monkeypatch):
    seed = default_seed(DType.BF16)

    def fake_run(command, text=True, stdout=None, stderr=None):
        if "--version" in command:
            return SimpleNamespace(returncode=0, stdout="fake hipcc 7.2")
        output = Path(command[command.index("-o") + 1])
        source = Path(next(value for value in command if str(value).endswith(".hip")))
        names = sorted(set(re.findall(r"void (geak_bf16_[0-9a-f]+)\(", source.read_text())))
        if "-S" in command:
            output.write_text(_assembly(names, seed.lds.total_bytes, scratch=16))
        else:
            output.write_bytes(b"artifact")
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr("asm_tuner.compiler.subprocess.run", fake_run)
    with pytest.raises(MetadataError, match="register spill"):
        ShardCompiler(tmp_path / "cache", hipcc="/fake/hipcc").compile_shard(
            get_adapter("bf16"), [seed], shared=True,
        )


def test_metadata_gate_rejects_identical_pgr1_pgr2_isa(tmp_path, monkeypatch):
    seed = default_seed(DType.BF16)
    pgr2 = replace(seed, pgr=2)

    def fake_run(command, text=True, stdout=None, stderr=None):
        if "--version" in command:
            return SimpleNamespace(returncode=0, stdout="fake hipcc 7.2")
        output = Path(command[command.index("-o") + 1])
        source = Path(next(value for value in command if str(value).endswith(".hip")))
        names = sorted(set(re.findall(r"void (geak_bf16_[0-9a-f]+)\(", source.read_text())))
        if "-S" in command:
            output.write_text(_assembly(names, seed.lds.total_bytes))
        else:
            output.write_bytes(b"artifact")
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr("asm_tuner.compiler.subprocess.run", fake_run)
    with pytest.raises(MetadataError, match="PGR.*identical"):
        ShardCompiler(tmp_path / "cache", hipcc="/fake/hipcc").compile_shard(
            get_adapter("bf16"), [seed, pgr2], shared=True,
        )
