from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .adapters.base import Measurement, TaskAdapter
from .compiler import CompileResult
from .runtime_table import RuntimeTableStore
from .schema import KernelConfig, Route


RECEIPT_VERSION = "geak.raw_measurement_receipt/v1"
RUN_VERSION = "geak.successive_halving_run/v1"


@dataclass(frozen=True)
class Candidate:
    config: KernelConfig
    kernel: Mapping[str, Any]
    cache_key: str
    manifest_path: Path
    shared_object: Path

    @property
    def config_id(self) -> str:
        return self.config.config_id


MeasureFn = Callable[[Candidate, Route, int, bool], Measurement]


def candidates_from_compile_results(results: Sequence[CompileResult]) -> List[Candidate]:
    candidates: List[Candidate] = []
    seen_isa = set()
    for result in results:
        configs = {item.config_id: item for item in (KernelConfig.from_dict(row) for row in result.manifest["configs"])}
        shared_name = result.manifest["artifacts"].get("shared")
        if not shared_name:
            raise ValueError("runner requires compile artifacts built with shared=True")
        for kernel in result.manifest["kernels"]:
            # An ISA index duplicate may point at an older cache entry that is
            # not part of this run. Keep the first representative in the
            # current result set; only dedupe hashes we can actually execute.
            if kernel["isa_hash"] in seen_isa:
                continue
            seen_isa.add(kernel["isa_hash"])
            config = configs[kernel["config_id"]]
            candidates.append(Candidate(
                config=config,
                kernel=kernel,
                cache_key=result.cache_key,
                manifest_path=result.directory / "manifest.json",
                shared_object=result.directory / shared_name,
            ))
    return candidates


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp.%d" % os.getpid())
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(str(temp), str(path))


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "route"


class SuccessiveHalvingRunner:
    """Legacy unpaired runner retained only to fail closed.

    The executable search path is :func:`asm_tuner.integration.search_release`,
    which measures candidates and the canonical task implementation in the same
    semantic slot.  This API accepts one arm at a time and therefore cannot
    produce admissible candidate-vs-canonical evidence.
    """
    def __init__(
        self,
        adapter: TaskAdapter,
        output_dir: Path,
        runtime_table: RuntimeTableStore,
        budgets: Sequence[int] = (1, 3, 7),
        eta: int = 2,
        top_k: int = 1,
        warmups: int = 2,
        device: int = 0,
        measure: Optional[MeasureFn] = None,
        release: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not budgets or any(value < 1 for value in budgets):
            raise ValueError("budgets must contain positive sample counts")
        if list(budgets) != sorted(set(budgets)):
            raise ValueError("budgets must be strictly increasing")
        if eta < 2 or top_k < 1 or warmups < 1:
            raise ValueError("eta >= 2, top_k >= 1, and warmups >= 1 are required")
        self.adapter = adapter
        self.output_dir = Path(output_dir)
        self.runtime_table = runtime_table
        self.budgets = tuple(int(x) for x in budgets)
        self.eta = eta
        self.top_k = top_k
        self.warmups = warmups
        self.device = device
        self._measure = measure
        self.release = dict(release or {})
        if not self.release:
            raise ValueError("runner requires canonical task/protocol release metadata")

    def _default_measure(self, candidate: Candidate, route: Route, samples: int, check: bool) -> Measurement:
        return self.adapter.measure(
            candidate.shared_object,
            candidate.config,
            route,
            samples=samples,
            warmups=self.warmups,
            device=self.device,
            check=check,
        )

    def run(self, candidates: Sequence[Candidate], routes: Sequence[Route]) -> Mapping[str, Any]:
        raise RuntimeError(
            "legacy unpaired successive halving cannot publish runtime_table/v3; "
            "use integration.search_release for same-slot canonical control and paired MDE gating"
        )
        if not candidates:
            raise ValueError("no unique-ISA candidates to run")
        if not routes:
            raise ValueError("no routes to run")
        run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
        receipt_root = self.output_dir / "receipts" / run_id
        measure = self._measure or self._default_measure
        summary: Dict[str, Any] = {
            "schema": RUN_VERSION,
            "run_id": run_id,
            "adapter": self.adapter.name,
            "budgets": list(self.budgets),
            "eta": self.eta,
            "top_k": self.top_k,
            "device": self.device,
            "routes": {},
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
        }

        selected_by_route: Dict[str, Sequence[Mapping[str, Any]]] = {}
        for route in routes:
            survivors = list(candidates)
            last_scores: Dict[str, Tuple[float, Path]] = {}
            rounds = []
            for round_index, budget in enumerate(self.budgets):
                scored: List[Tuple[float, Candidate, Path]] = []
                failures = []
                for candidate in survivors:
                    receipt_path = receipt_root / _slug(route.name) / ("round_%02d" % round_index) / (candidate.config_id + ".json")
                    receipt: Dict[str, Any] = {
                        "schema": RECEIPT_VERSION,
                        "run_id": run_id,
                        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "route": route.to_dict(),
                        "round": round_index,
                        "sample_budget": budget,
                        "adapter": self.adapter.name,
                        "source_version": self.adapter.source_version,
                        "config_id": candidate.config_id,
                        "config": candidate.config.to_dict(include_derived=True),
                        "kernel_metadata": dict(candidate.kernel),
                        "cache_key": candidate.cache_key,
                        "compile_manifest": str(candidate.manifest_path),
                        "shared_object": str(candidate.shared_object),
                    }
                    try:
                        result = measure(candidate, route, budget, round_index == 0)
                        receipt["status"] = "ok"
                        receipt["measurement"] = result.to_dict()
                        scored.append((result.median_us, candidate, receipt_path))
                        last_scores[candidate.config_id] = (result.median_us, receipt_path)
                    except Exception as exc:
                        receipt["status"] = "error"
                        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
                        failures.append({"config_id": candidate.config_id, "error": receipt["error"]})
                    _atomic_json(receipt_path, receipt)

                scored.sort(key=lambda row: (row[0], row[1].config_id))
                if not scored:
                    raise RuntimeError("every candidate failed on route %s round %d" % (route.name, round_index))
                final_round = round_index == len(self.budgets) - 1
                keep = min(len(scored), self.top_k if final_round else max(self.top_k, int(math.ceil(len(scored) / float(self.eta)))))
                survivors = [row[1] for row in scored[:keep]]
                rounds.append({
                    "round": round_index,
                    "sample_budget": budget,
                    "measured": len(scored),
                    "kept": [candidate.config_id for candidate in survivors],
                    "failures": failures,
                })

            selected = []
            for candidate in sorted(survivors, key=lambda item: (last_scores[item.config_id][0], item.config_id))[:self.top_k]:
                median_us, receipt_path = last_scores[candidate.config_id]
                compile_manifest = json.loads(candidate.manifest_path.read_text(encoding="utf-8"))
                source_path = candidate.manifest_path.parent / compile_manifest["artifacts"]["source"]
                selected.append({
                    "config_id": candidate.config_id,
                    "median_us": median_us,
                    "isa_hash": candidate.kernel["isa_hash"],
                    "kernel": candidate.kernel["kernel"],
                    "symbol": self.adapter.launch_name(candidate.config),
                    "cache_key": candidate.cache_key,
                    "compile_manifest": str(candidate.manifest_path),
                    "compile_manifest_sha256": hashlib.sha256(candidate.manifest_path.read_bytes()).hexdigest(),
                    "artifact": str(candidate.shared_object),
                    "artifact_sha256": hashlib.sha256(candidate.shared_object.read_bytes()).hexdigest(),
                    "source": str(source_path),
                    "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    "receipt": str(receipt_path),
                    "config": candidate.config.to_dict(include_derived=True),
                })
            selected_by_route[route.name] = selected
            summary["routes"][route.name] = {"rounds": rounds, "selected": selected}

        table = self.runtime_table.publish_suite(
            routes, selected_by_route, run_id=run_id, receipts_root=receipt_root,
            release=self.release,
        )
        for route in routes:
            summary["routes"][route.name]["table_revision"] = table["revision"]
        summary["table_digest"] = table["table_digest"]
        summary["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        summary_path = self.output_dir / "runs" / (run_id + ".json")
        _atomic_json(summary_path, summary)
        summary["summary_path"] = str(summary_path)
        return summary
