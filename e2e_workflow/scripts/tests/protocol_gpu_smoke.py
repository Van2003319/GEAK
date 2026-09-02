#!/usr/bin/env python3
"""Tiny locked GPU integration smoke for measurement-protocol/v2 (not a benchmark)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness_lib as harness  # noqa: E402


def main() -> None:
    protocol = harness.measurement_protocol(
        "semantic_weight_rotating",
        seed=7,
        latency_cycles=4,
        throughput_cycles=3,
        throughput_inner=2,
        warmup_cycles=1,
        output_ring_size=2,
        rotation_target_mb=2,
        assumed_cache_mb=1,
        max_slots=2,
    )
    specs = harness.semantic_weight_slot_specs(11, 2, protocol)
    a = torch.arange(512 * 512, device="cuda", dtype=torch.float32).reshape(512, 512)
    slots = []
    for index, spec in enumerate(specs):
        b = torch.full_like(a, float(index + 1))
        slots.append({**spec, "weight_bytes": b.numel() * b.element_size(),
                      "args": (a, b)})
    arms = {
        "candidate": lambda slot: torch.add(*slot["args"]),
        "incumbent": lambda slot: torch.add(*slot["args"]),
        "oracle:torch_add": lambda slot: torch.add(*slot["args"]),
    }
    receipt = harness.measure_paired_arms(
        arms, slots, protocol=protocol, metric="single_call_latency")
    assert all(receipt["invariants"].values()), receipt["invariants"]
    assert all(row["addresses_rotated"] for row in receipt["output_lifecycle"].values())
    assert receipt["gpu_lock"]["mode"] == "external", receipt["gpu_lock"]
    assert len(receipt["blocks"]) == 8
    assert len(receipt["samples"]) == 24
    expected = json.loads(os.environ.get("GEAK_PAIRED_EXPECTED_ROUTES", '["smoke_512"]'))
    if expected != ["smoke_512"]:
        raise AssertionError(f"smoke producer expected one smoke_512 route, got {expected}")
    raw = {
        "schema": "geak.paired-measurement-suite/v2",
        "expected_routes": expected,
        "protocol_hash": receipt["protocol_hash"],
        "promotion_control": "incumbent",
        "oracle_role": "absolute_report_only",
        "routes": [{"route": "smoke_512", "measurement_receipt": receipt,
                    "absolute_report": {"oracle_arm": "oracle:torch_add"}}],
    }
    output = os.environ.get("GEAK_PAIRED_RAW_RECEIPT")
    if output:
        harness.atomic_write_json(output, raw)
    print(json.dumps({
        "status": "ok",
        "protocol_hash": receipt["protocol_hash"],
        "blocks": len(receipt["blocks"]),
        "samples": len(receipt["samples"]),
        "gpu_lock": receipt["gpu_lock"],
        "invariants": receipt["invariants"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
