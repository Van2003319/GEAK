#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

from asm_tuner.compiler import ShardCompiler
from asm_tuner.contracts import TaskContract
from asm_tuner.schema import Route
from asm_tuner.schema_survival import generated_survival_candidate


ROOT = Path(__file__).resolve().parents[2]


def run(cache: Path) -> Dict[str, Any]:
    if os.environ.get("GEAK_GPU_LOCK_HELD") != "1" or os.environ.get("GEAK_GPU_FENCE_HELD") != "1":
        raise RuntimeError("GPU smoke must run under gpu_lock.sh + gpu_fence_run.sh")
    cases = [
        (
            "bf16",
            ROOT / "asm_tuner/manifests/tasks/dense_bf16_gemm_fused.json",
            "bf16_solution205554",
            Route("pgr2_smoke", 128, 224, 128),
        ),
        (
            "fp8_rowwise",
            ROOT / "asm_tuner/manifests/tasks/dense_fp8_gemm_rowwise.json",
            "fp8_solution262788",
            Route("pgr2_smoke", 128, 128, 256),
        ),
    ]
    evidence: Dict[str, Any] = {
        "schema": "geak.gfx942-pgr2-smoke/v1",
        "arch": "gfx942",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpu_boundary": {
            "lock": os.environ.get("GEAK_GPU_LOCK_FILE"),
            "fence": True,
            "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        },
        "adapters": {},
    }
    for adapter_name, manifest, fixture_name, route in cases:
        contract = TaskContract.load(manifest)
        pgr2 = generated_survival_candidate(contract.search_configs, fixture_name)
        pgr1 = replace(pgr2, pgr=1)
        contract.config_by_id(pgr1.config_id)
        compiler = ShardCompiler(cache / adapter_name)
        compiled = compiler.compile_shard(contract.adapter, [pgr1, pgr2], shared=True)
        kernels = {row["config_id"]: row for row in compiled.manifest["kernels"]}
        if kernels[pgr1.config_id]["isa_hash"] == kernels[pgr2.config_id]["isa_hash"]:
            raise RuntimeError("PGR1/PGR2 ISA hash collision for %s" % adapter_name)
        rows: Dict[str, Any] = {}
        for label, config in (("pgr1", pgr1), ("pgr2", pgr2)):
            measurement = contract.adapter.measure(
                compiled.artifact("shared"), config, route,
                samples=2, warmups=1, device=0, check=True,
            )
            metadata = kernels[config.config_id]
            rows[label] = {
                "config_id": config.config_id,
                "config": config.to_dict(include_derived=True),
                "isa_hash": metadata["isa_hash"],
                "instruction_count": metadata["instruction_count"],
                "vgpr": metadata["vgpr"],
                "agpr": metadata["agpr"],
                "unified_vgpr": metadata["unified_vgpr"],
                "sgpr": metadata["sgpr"],
                "lds_bytes": metadata["lds_bytes"],
                "scratch_bytes_per_workitem": metadata["scratch_bytes_per_workitem"],
                "occupancy": metadata["occupancy"],
                "correctness": dict(measurement.correctness),
                "median_us": measurement.median_us,
                "samples_us": list(measurement.samples_us),
            }
        evidence["adapters"][adapter_name] = {
            "fixture": fixture_name,
            "task_digest": contract.task_digest,
            "source_version": contract.adapter.source_version,
            "search_space": contract.search_materialization.to_dict(),
            "route": route.to_dict(),
            "k_stages": (route.k + pgr2.bk - 1) // pgr2.bk,
            "compile_cache_key": compiled.cache_key,
            "compiler": compiled.manifest["compiler"],
            "pgr1": rows["pgr1"],
            "pgr2": rows["pgr2"],
        }
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run(args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(args.output))
    print(json.dumps({
        "ok": True,
        "output": str(args.output),
        "adapters": {
            name: {
                "pgr1": row["pgr1"]["isa_hash"],
                "pgr2": row["pgr2"]["isa_hash"],
            } for name, row in value["adapters"].items()
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
