"""Immutable-in-process oracle route manifests.

A bake-off writes a new manifest atomically. A scoring process loads it once and never observes a
mid-run route change, even if another process re-bakes the file concurrently.
"""
from __future__ import annotations

import copy
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

    def validate_for(self, protocol_hash: str, cases: list[dict]) -> tuple[bool, list[str]]:
        errors = []
        if self._payload.get("frozen") is not True:
            errors.append("manifest is not frozen")
        if self._payload.get("selection_metric") != "single_call_latency":
            errors.append("selection metric is not single_call_latency")
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
    if payload.get("schema") != SCHEMA or payload.get("frozen") is not True:
        raise ValueError("only a complete frozen v2 route manifest may be written")
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
