import json
from dataclasses import replace
from pathlib import Path

from asm_tuner.adapters import get_adapter
from asm_tuner.adapters.base import Measurement
from asm_tuner.runner import Candidate, SuccessiveHalvingRunner
from asm_tuner.runtime_table import RuntimeSelector, RuntimeTableStore
from asm_tuner.schema import DType, Route
from asm_tuner.vendor_fixtures import default_seed


def test_per_route_halving_raw_receipts_versioned_table_and_selector(tmp_path):
    seed = default_seed(DType.BF16)
    configs = [replace(seed, wgm=value) for value in (1, 2, 3)]
    candidates = [
        Candidate(
            config=config,
            kernel={"kernel": "k_" + config.config_id, "isa_hash": "isa_" + config.config_id},
            cache_key="cache_" + config.config_id,
            manifest_path=tmp_path / (config.config_id + ".manifest.json"),
            shared_object=tmp_path / (config.config_id + ".so"),
        )
        for config in configs
    ]
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

    table_path = tmp_path / "runtime_table.json"
    runner = SuccessiveHalvingRunner(
        adapter=get_adapter("bf16"),
        output_dir=tmp_path / "out",
        runtime_table=RuntimeTableStore(table_path),
        budgets=(1, 3),
        eta=2,
        top_k=1,
        warmups=1,
        measure=measure,
    )
    summary = runner.run(candidates, [route_a, route_b])
    assert summary["routes"]["route_a"]["selected"][0]["config_id"] == configs[0].config_id
    assert summary["routes"]["route_b"]["selected"][0]["config_id"] == configs[2].config_id

    table = RuntimeTableStore(table_path).load()
    assert table["revision"] == 2
    assert (tmp_path / "runtime_table.json.v1.json").exists()
    assert (tmp_path / "runtime_table.json.v2.json").exists()
    assert RuntimeSelector(table).select("route_a")["config_id"] == configs[0].config_id
    assert RuntimeSelector.from_path(table_path).select("route_b")["config_id"] == configs[2].config_id

    receipts = list((tmp_path / "out" / "receipts").rglob("*.json"))
    assert len(receipts) == 10
    raw = json.loads(receipts[0].read_text())
    assert raw["schema"] == "geak.raw_measurement_receipt/v1"
    assert raw["status"] == "ok"
    assert raw["measurement"]["timer"]["dispatch_primed"] is True
