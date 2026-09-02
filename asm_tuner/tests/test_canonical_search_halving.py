import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

import asm_tuner.integration as integration
from asm_tuner.contracts import TaskContract


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "asm_tuner/manifests/tasks/dense_bf16_gemm_fused.json"


class FakeResult:
    def __init__(self, directory, adapter, configs):
        self.directory = directory
        self.cache_key = "cache-unit"
        self.cache_hit = False
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "shard.hip").write_text("// source\n")
        (directory / "shard.so").write_bytes(b"shared")
        self.manifest = {
            "configs": [config.to_dict(include_derived=True) for config in configs],
            "kernels": [{
                "config_id": config.config_id,
                "kernel": adapter.kernel_name(config),
                "isa_hash": (config.config_id * 4)[:64],
            } for config in configs],
            "artifacts": {"source": "shard.hip", "shared": "shard.so"},
        }
        (directory / "manifest.json").write_text(json.dumps(self.manifest))

    def artifact(self, kind):
        return self.directory / self.manifest["artifacts"][kind]


class FakeCompiler:
    def __init__(self, cache_root, **kwargs):
        self.cache_root = Path(cache_root)

    def compile(self, adapter, configs, **kwargs):
        assert 8 <= len(configs) <= 16
        assert 8 <= kwargs["shard_size"] <= 16
        return [FakeResult(self.cache_root / "fake-shard", adapter, configs)]


class FakeCallable:
    def __init__(self, adapter, artifact):
        self.config_id = artifact["config_id"]

    def __call__(self, a, b):
        return (a.float() @ b.float().t()).to(torch.bfloat16)


class FakeRunner:
    @staticmethod
    def _make_measurement_slots(case, protocol):
        a = torch.ones((2, 2), dtype=torch.bfloat16)
        b = torch.ones((2, 2), dtype=torch.bfloat16)
        return [{"args": (a, b), "slot_id": "unit"}]


class FakeHarness:
    @staticmethod
    def correct(output, reference, tolerance):
        return True, 0.0

    @staticmethod
    def measure_paired_arms(arms, slots, *, protocol, metric):
        names = list(arms)
        for call in arms.values():
            call(slots[0])
        return {
            "protocol_hash": protocol["hash"],
            "medians_ms": {name: float(index + 1) for index, name in enumerate(reversed(names))},
            "invariants": {"paired": True, "semantic_slots": True},
            "gpu_lock": {"mode": "external"},
        }


def test_canonical_generated_search_halves_per_route_keeps_top_k_and_resumes(
    tmp_path, monkeypatch,
):
    contract = TaskContract.load(MANIFEST, require_oracle_compatible=True)
    monkeypatch.setenv("GEAK_GPU_LOCK_HELD", "1")
    monkeypatch.setenv("GEAK_GPU_FENCE_HELD", "1")
    monkeypatch.setattr(integration, "ShardCompiler", FakeCompiler)
    monkeypatch.setattr(integration, "_CompiledCallable", FakeCallable)
    monkeypatch.setattr(integration, "_task_import_path", lambda path: contextlib.nullcontext())

    def fake_import(path, name):
        sys.modules["harness_lib"] = FakeHarness
        return FakeRunner

    monkeypatch.setattr(integration, "_import_path", fake_import)
    kwargs = {
        "cache_root": tmp_path / "cache",
        "output_dir": tmp_path / "output",
        "case_ids": ["prefill_m2048_square"],
        "bootstrap": True,
    }
    first = integration.search_release(contract, **kwargs)
    measured = first["summary"]["measurements"]["prefill_m2048_square"]
    assert [len(row["candidate_ids"]) for row in measured["rounds"]] == [8, 4]
    assert len(first["release"]["route_rankings"]["prefill_m2048_square"]) == 2
    assert first["summary"]["cache"] == {
        "round_hits": 0, "round_misses": 2, "compile_hits": 0, "compile_misses": 1,
    }

    second = integration.search_release(contract, **kwargs)
    assert second["release"]["route_rankings"] == first["release"]["route_rankings"]
    assert second["summary"]["cache"]["round_hits"] == 2
    assert second["summary"]["cache"]["round_misses"] == 0
