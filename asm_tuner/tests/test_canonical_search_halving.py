import contextlib
import json
import sys
from dataclasses import replace
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
    candidate_ratios = {}
    default_candidate_ratio = 0.90

    @staticmethod
    def correct(output, reference, tolerance):
        return True, 0.0

    @staticmethod
    def measure_paired_arms(arms, slots, *, protocol, metric):
        names = list(arms)
        for call in arms.values():
            call(slots[0])
        assert metric == "sustained_throughput"
        medians = {
            name: (
                1.0 if name == integration.CANONICAL_CONTROL_ARM
                else float(FakeHarness.candidate_ratios.get(
                    name, FakeHarness.default_candidate_ratio,
                ))
            )
            for name in names
        }
        blocks = []
        samples = []
        for block in range(10):
            block_samples = {
                name: {
                    "block": block,
                    "cycle": block,
                    "arm": name,
                    "slot_id": "unit",
                    "slot_fingerprint": "same-slot",
                    "ms": value,
                }
                for name, value in medians.items()
            }
            blocks.append({
                "block": block,
                "cycle": block,
                "slot_id": "unit",
                "slot_fingerprint": "same-slot",
                "samples": block_samples,
            })
            samples.extend(block_samples.values())
        return {
            "protocol_hash": protocol["hash"],
            "metric_id": metric,
            "metric": dict(protocol["metrics"][metric]),
            "medians_ms": medians,
            "blocks": blocks,
            "samples": samples,
            "invariants": {"paired": True, "semantic_slots": True},
            "gpu_lock": {"mode": "external"},
        }


def fake_confirm(contract, *, artifact, route_name, output_dir):
    candidate_id = artifact["config_id"]
    receipt = FakeHarness.measure_paired_arms(
        {
            candidate_id: lambda slot: None,
            integration.CANONICAL_CONTROL_ARM: lambda slot: None,
        },
        [{"slot_id": "unit"}],
        protocol=dict(contract.protocol),
        metric=contract.score_metric_id,
    )
    gate = integration._candidate_vs_canonical_gate(
        receipt, route=route_name, candidate_id=candidate_id,
        evidence_phase="independent_materialized_task_dispatch",
    )
    gate["confirmation"] = {
        "workspace": "isolated_materialized_copy",
        "task_entrypoint": contract.value["bindings"]["entrypoint"]["symbol"],
        "runtime_table_schema": "geak.runtime_table/v3",
        "table_digest": "d" * 64,
        "correctness": {"passed": True, "max_relative_error": 0.0},
        "dispatch_receipt": {
            "route": route_name,
            "config_id": candidate_id,
            "symbol": artifact["symbol"],
            "table_digest": "d" * 64,
            "artifact_sha256": artifact["artifact_sha256"],
            "source_sha256": artifact["source_sha256"],
            "compile_manifest_sha256": artifact["compile_manifest_sha256"],
        },
    }
    return gate


def test_canonical_generated_search_halves_per_route_keeps_top_k_and_resumes(
    tmp_path, monkeypatch,
):
    loaded = TaskContract.load(MANIFEST)
    contract = replace(
        loaded, oracle_compatible=True,
        oracle={**loaded.oracle, "selection_metric": loaded.score_metric_id,
                "score_metric": dict(loaded.score_metric)},
    )
    monkeypatch.setenv("GEAK_GPU_LOCK_HELD", "1")
    monkeypatch.setenv("GEAK_GPU_FENCE_HELD", "1")
    monkeypatch.setattr(integration, "ShardCompiler", FakeCompiler)
    monkeypatch.setattr(integration, "_CompiledCallable", FakeCallable)
    monkeypatch.setattr(integration, "_confirm_materialized_candidate", fake_confirm)
    monkeypatch.setattr(integration, "_task_import_path", lambda path: contextlib.nullcontext())
    monkeypatch.setattr(
        integration, "_canonical_task_callable",
        lambda contract: (lambda slot: FakeCallable("", {"config_id": "canonical"})(
            *slot["args"]
        )),
    )
    FakeHarness.candidate_ratios = {}
    FakeHarness.default_candidate_ratio = 0.90

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
    assert measured["comparison"] == "candidate_vs_canonical_task_implementation"
    assert measured["outcome"]["status"] == "winner"
    assert measured["selected"][0]["selection_gate"]["pair_ids"] != \
        measured["selected"][0]["task_dispatch_confirmation_gate"]["pair_ids"]
    assert measured["selected"][0]["task_dispatch_confirmation_gate"]["evidence_phase"] == \
        "independent_materialized_task_dispatch"
    untouched = {
        route: ranking for route, ranking in first["release"]["route_rankings"].items()
        if route != "prefill_m2048_square"
    }
    assert len(untouched) == 10
    assert all(ranking == [] for ranking in untouched.values())
    assert all(
        outcome["status"] == "unmeasured" and outcome["selection"] == "canonical"
        for route, outcome in first["release"]["route_outcomes"].items()
        if route != "prefill_m2048_square"
    )
    selection = measured["candidate_plan"]["selection"]
    assert len(selection) == 8
    assert all(row["score_components"] and row["selection_reason"] for row in selection)
    round_receipt = json.loads(Path(measured["rounds"][0]["receipt"]).read_text())
    assert round_receipt["canonical_control"]["arm"] == integration.CANONICAL_CONTROL_ARM
    assert integration.CANONICAL_CONTROL_ARM in round_receipt["receipt"]["medians_ms"]
    assert set(round_receipt["candidate_selection"]) == set(measured["rounds"][0]["candidate_ids"])
    assert all(
        row["score_components"] and row["selection_reason"]
        for row in round_receipt["candidate_selection"].values()
    )
    assert first["summary"]["cache"] == {
        "round_hits": 0, "round_misses": 2, "compile_hits": 0, "compile_misses": 1,
    }

    second = integration.search_release(contract, **kwargs)
    assert second["release"]["route_rankings"] == first["release"]["route_rankings"]
    assert second["summary"]["cache"]["round_hits"] == 2
    assert second["summary"]["cache"]["round_misses"] == 0


def test_all_slower_candidates_record_no_winner_and_keep_every_route_canonical(
    tmp_path, monkeypatch,
):
    loaded = TaskContract.load(MANIFEST)
    contract = replace(
        loaded, oracle_compatible=True,
        oracle={**loaded.oracle, "selection_metric": loaded.score_metric_id,
                "score_metric": dict(loaded.score_metric)},
    )
    monkeypatch.setenv("GEAK_GPU_LOCK_HELD", "1")
    monkeypatch.setenv("GEAK_GPU_FENCE_HELD", "1")
    monkeypatch.setattr(integration, "ShardCompiler", FakeCompiler)
    monkeypatch.setattr(integration, "_CompiledCallable", FakeCallable)
    monkeypatch.setattr(integration, "_confirm_materialized_candidate", fake_confirm)
    monkeypatch.setattr(integration, "_task_import_path", lambda path: contextlib.nullcontext())
    monkeypatch.setattr(
        integration, "_canonical_task_callable",
        lambda contract: (lambda slot: FakeCallable("", {"config_id": "canonical"})(
            *slot["args"]
        )),
    )
    monkeypatch.setattr(
        integration, "_import_path",
        lambda path, name: (
            sys.modules.__setitem__("harness_lib", FakeHarness) or FakeRunner
        ),
    )
    FakeHarness.candidate_ratios = {}
    FakeHarness.default_candidate_ratio = 1.05

    result = integration.search_release(
        contract, cache_root=tmp_path / "cache", output_dir=tmp_path / "output",
        case_ids=["prefill_m2048_square"], bootstrap=True,
    )

    assert all(ranking == [] for ranking in result["release"]["route_rankings"].values())
    assert all(selection is None for selection in result["release"]["route_selection"].values())
    target = result["release"]["route_outcomes"]["prefill_m2048_square"]
    assert target["status"] == "no_winner"
    assert target["selection"] == "canonical"
    assert target["candidate_gates"]
    assert all(not gate["accepted"] for gate in target["candidate_gates"])


def test_one_five_percent_candidate_only_owns_the_measured_route(tmp_path, monkeypatch):
    loaded = TaskContract.load(MANIFEST)
    contract = replace(
        loaded, oracle_compatible=True,
        oracle={**loaded.oracle, "selection_metric": loaded.score_metric_id,
                "score_metric": dict(loaded.score_metric)},
    )
    monkeypatch.setenv("GEAK_GPU_LOCK_HELD", "1")
    monkeypatch.setenv("GEAK_GPU_FENCE_HELD", "1")
    monkeypatch.setattr(integration, "ShardCompiler", FakeCompiler)
    monkeypatch.setattr(integration, "_CompiledCallable", FakeCallable)
    monkeypatch.setattr(integration, "_confirm_materialized_candidate", fake_confirm)
    monkeypatch.setattr(integration, "_task_import_path", lambda path: contextlib.nullcontext())
    monkeypatch.setattr(
        integration, "_canonical_task_callable",
        lambda contract: (lambda slot: FakeCallable("", {"config_id": "canonical"})(
            *slot["args"]
        )),
    )
    monkeypatch.setattr(
        integration, "_import_path",
        lambda path, name: (
            sys.modules.__setitem__("harness_lib", FakeHarness) or FakeRunner
        ),
    )
    target_route = next(
        route for route in contract.routes if route.name == "prefill_m2048_square"
    )
    plan = integration.plan_route_candidates(
        contract.search_space, contract.search_configs, target_route,
        cu_count=304, limit=8, assume_static_valid=True,
    )
    winner = min(config.config_id for config in plan.candidates)
    FakeHarness.candidate_ratios = {winner: 0.95}
    FakeHarness.default_candidate_ratio = 1.10

    result = integration.search_release(
        contract, cache_root=tmp_path / "cache", output_dir=tmp_path / "output",
        case_ids=[target_route.name], bootstrap=True,
    )

    assert result["release"]["route_rankings"][target_route.name] == [winner]
    assert result["release"]["route_selection"][target_route.name] == winner
    assert result["release"]["route_outcomes"][target_route.name]["status"] == "winner"
    assert all(
        ranking == [] for route, ranking in result["release"]["route_rankings"].items()
        if route != target_route.name
    )
