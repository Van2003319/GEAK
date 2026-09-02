import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from asm_tuner.adapters import get_adapter
from asm_tuner.adapters.base import Measurement
from asm_tuner.runner import Candidate, SuccessiveHalvingRunner
from asm_tuner.runtime_table import (
    RuntimeSelector, RuntimeTableError, RuntimeTableStore, table_digest,
)
from asm_tuner.schema import DType, Route
from asm_tuner.schema_survival import default_seed

import pytest


def test_legacy_unpaired_halving_refuses_to_publish_runtime_table_v3(tmp_path):
    seed = default_seed(DType.BF16)
    configs = [replace(seed, wgm=value) for value in (1, 2, 3)]
    candidates = []
    for config in configs:
        directory = tmp_path / config.config_id
        directory.mkdir()
        source = directory / "shard.hip"
        source.write_text("// %s\n" % config.config_id)
        shared = directory / "shard.so"
        shared.write_bytes(("so-" + config.config_id).encode())
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps({"artifacts": {"source": source.name}}))
        candidates.append(Candidate(
            config=config,
            kernel={"kernel": "k_" + config.config_id, "isa_hash": "isa_" + config.config_id},
            cache_key="cache_" + config.config_id,
            manifest_path=manifest,
            shared_object=shared,
        ))
    route_a = Route("route_a", 64, 65, 32)
    route_b = Route("route_b", 65, 64, 32)
    order = {config.config_id: index for index, config in enumerate(configs)}

    def measure(candidate, route, samples, check):
        index = order[candidate.config_id]
        median = float(index + 1) if route.name == "route_a" else float(3 - index)
        return Measurement(
            samples_us=tuple(median for _ in range(samples)),
            median_us=median,
            correctness={"checked": check, "passed": True},
            timer={"dispatch_primed": True, "samples": samples},
        )

    protocol = {
        "schema": "geak.measurement-protocol/v2", "version": 2,
        "name": "unit", "hash": "b" * 64,
    }
    score_metric = {
        "schema": "geak.score-metric/v1", "version": 1,
        "id": "sustained_throughput", "unit": "ms/call", "inner": 4,
        "protocol_hash": protocol["hash"],
    }
    score_metric["hash"] = hashlib.sha256(json.dumps(
        score_metric, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    source_contract = {
        "semantic_hashes": {"task.py": "e" * 64},
        "binding_hashes": {"cases": "f" * 64},
    }
    source_digest = hashlib.sha256(json.dumps(
        source_contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    release = {
        "release_id": "release-test",
        "task": {
            "name": "unit-task", "adapter": "bf16", "digest": "a" * 64,
            "source_digest": source_digest,
            **source_contract,
        },
        "protocol": protocol,
        "score_metric": score_metric,
        "oracle": {"manifest_sha256": "c" * 64, "compatible": True},
        "fallback": {"kind": "canonical", "reason": "outside suite"},
    }
    table_path = tmp_path / "runtime_table.json"
    runner = SuccessiveHalvingRunner(
        adapter=get_adapter("bf16"),
        output_dir=tmp_path / "out",
        runtime_table=RuntimeTableStore(table_path),
        budgets=(1, 3), eta=2, top_k=2, warmups=1,
        measure=measure, release=release,
    )
    with pytest.raises(RuntimeError, match="same-slot canonical control"):
        runner.run(candidates, [route_a, route_b])
    assert not table_path.exists()
