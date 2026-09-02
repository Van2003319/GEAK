"""Immutable-in-process oracle route manifests.

A bake-off writes a new manifest atomically. A scoring process loads it once and never observes a
mid-run route change, even if another process re-bakes the file concurrently.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

SCHEMA = "geak.oracle-route-manifest/v2"


class FrozenRouteManifest:
    def __init__(self, path: str | os.PathLike[str], task: str):
        self.path = Path(path)
        self.task = task
        self._payload = json.loads(self.path.read_text(encoding="utf-8"))
        if self._payload.get("schema") != SCHEMA:
            raise ValueError(f"unsupported oracle route schema in {self.path}")
        if self._payload.get("task") != task:
            raise ValueError(f"oracle route manifest is for {self._payload.get('task')}, not {task}")
        routes = self._payload.get("routes")
        if not isinstance(routes, dict) or not routes:
            raise ValueError("oracle route manifest must contain routes")
        self._by_shape = {}
        for case_id, entry in routes.items():
            shape = tuple(int(entry[k]) for k in ("M", "N", "K"))
            if shape in self._by_shape:
                raise ValueError(f"duplicate oracle route shape {shape}")
            self._by_shape[shape] = (case_id, copy.deepcopy(entry))

    def snapshot(self) -> dict:
        return copy.deepcopy(self._payload)

    def entry(self, m: int, n: int, k: int) -> dict:
        try:
            return copy.deepcopy(self._by_shape[(int(m), int(n), int(k))][1])
        except KeyError as exc:
            raise KeyError(f"no frozen oracle route for shape {(m, n, k)}") from exc

    def backend(self, m: int, n: int, k: int) -> str:
        return str(self.entry(m, n, k)["backend"])

    def validate_for(self, protocol_hash: str, score_metric: dict | str,
                     cases: list[dict] | None = None) -> tuple[bool, list[str]]:
        # Two-argument callers are legacy latency lanes. They remain readable, but every new bake
        # writes the complete versioned object below.
        if cases is None:
            cases = score_metric  # type: ignore[assignment]
            score_metric = "single_call_latency"
        expected_id = (score_metric.get("id") if isinstance(score_metric, dict)
                       else str(score_metric))
        errors = []
        if self._payload.get("frozen") is not True:
            errors.append("manifest is not frozen")
        if expected_id not in ("single_call_latency", "sustained_throughput"):
            errors.append(f"unsupported requested score metric {expected_id!r}")
        if self._payload.get("selection_metric") != expected_id:
            errors.append(
                f"selection metric differs: manifest={self._payload.get('selection_metric')!r} "
                f"requested={expected_id!r}; re-run oracle-bakeoff")
        manifest_metric = self._payload.get("score_metric")
        if manifest_metric is None:
            if expected_id != "single_call_latency":
                errors.append("legacy oracle has no score_metric binding; re-run oracle-bakeoff")
        elif not isinstance(score_metric, dict) or manifest_metric != score_metric:
            errors.append("score_metric object differs; re-run oracle-bakeoff")
        if isinstance(score_metric, dict):
            metric_body = dict(score_metric)
            metric_digest = metric_body.pop("hash", None)
            computed = hashlib.sha256(json.dumps(
                metric_body, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
            if (score_metric.get("schema") != "geak.score-metric/v1"
                    or isinstance(score_metric.get("version"), bool)
                    or score_metric.get("version") != 1
                    or score_metric.get("unit") != "ms/call"
                    or score_metric.get("protocol_hash") != protocol_hash
                    or metric_digest != computed):
                errors.append("requested score_metric binding is malformed or stale")
            if ((expected_id == "single_call_latency"
                 and (isinstance(score_metric.get("inner"), bool)
                      or score_metric.get("inner") != 1))
                    or (expected_id == "sustained_throughput"
                        and (not isinstance(score_metric.get("inner"), int)
                             or score_metric["inner"] <= 1))):
                errors.append("requested score_metric inner is invalid")
        if self._payload.get("protocol_hash") != protocol_hash:
            errors.append("protocol hash differs; re-run oracle-bakeoff")
        routes = self._payload.get("routes", {})
        for case in cases:
            entry = routes.get(case["id"])
            if entry is None:
                errors.append(f"missing route {case['id']}")
                continue
            expected = tuple(int(case[k]) for k in ("M", "N", "K"))
            actual = tuple(int(entry[k]) for k in ("M", "N", "K"))
            if actual != expected:
                errors.append(f"shape drift for {case['id']}: {actual} != {expected}")
            if not entry.get("backend"):
                errors.append(f"missing backend for {case['id']}")
        extra = sorted(set(routes) - {case["id"] for case in cases})
        if extra:
            errors.append(f"extra routes: {extra}")
        return not errors, errors


def write_route_manifest(path: str | os.PathLike[str], payload: dict) -> None:
    metric = payload.get("score_metric")
    metric_body = dict(metric) if isinstance(metric, dict) else {}
    metric_digest = metric_body.pop("hash", None)
    computed = hashlib.sha256(json.dumps(
        metric_body, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
    if (payload.get("schema") != SCHEMA or payload.get("frozen") is not True
            or not isinstance(metric, dict) or metric.get("schema") != "geak.score-metric/v1"
            or isinstance(metric.get("version"), bool) or metric.get("version") != 1
            or metric_digest != computed
            or metric.get("unit") != "ms/call"
            or metric.get("id") not in ("single_call_latency", "sustained_throughput")
            or (metric.get("id") == "single_call_latency"
                and (isinstance(metric.get("inner"), bool) or metric.get("inner") != 1))
            or (metric.get("id") == "sustained_throughput"
                and (not isinstance(metric.get("inner"), int) or metric["inner"] <= 1))
            or payload.get("selection_metric") != metric.get("id")
            or payload.get("protocol_hash") != metric.get("protocol_hash")):
        raise ValueError("only a complete frozen score-metric-bound v2 route manifest may be written")
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
