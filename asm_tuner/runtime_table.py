from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence

from .schema import Route


LEGACY_RUNTIME_TABLE_VERSIONS = {"geak.runtime_table/v1", "geak.runtime_table/v2"}
RUNTIME_TABLE_VERSION = "geak.runtime_table/v3"


class RuntimeTableError(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def table_digest(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("table_digest", None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _score_metric_digest(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("hash", None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _validate_score_metric(value: object, protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != "geak.score-metric/v1":
        raise RuntimeTableError("runtime table lacks score-metric/v1")
    metric_id = value.get("id")
    inner = value.get("inner")
    if (isinstance(value.get("version"), bool) or value.get("version") != 1
            or metric_id not in ("single_call_latency", "sustained_throughput")
            or value.get("unit") != "ms/call"
            or isinstance(inner, bool) or not isinstance(inner, int)
            or (metric_id == "single_call_latency" and inner != 1)
            or (metric_id == "sustained_throughput" and inner <= 1)
            or value.get("protocol_hash") != protocol.get("hash")
            or value.get("hash") != _score_metric_digest(value)):
        raise RuntimeTableError("runtime table score metric is malformed or protocol-mismatched")
    return value


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(".%s.tmp.%d.%s" % (path.name, os.getpid(), uuid.uuid4().hex))
    try:
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temp), str(path))
        try:
            directory = os.open(str(path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


@contextlib.contextmanager
def _lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _route_ids(routes: Sequence[Route]) -> list[str]:
    names = [route.name for route in routes]
    if not names or len(names) != len(set(names)):
        raise RuntimeTableError("suite routes must be non-empty and uniquely named")
    return names


def _validate_v2(value: Mapping[str, Any]) -> None:
    if value.get("schema") != RUNTIME_TABLE_VERSION:
        raise RuntimeTableError("unsupported runtime table schema %r" % value.get("schema"))
    expected = table_digest(value)
    if value.get("table_digest") != expected:
        raise RuntimeTableError("runtime table digest mismatch")
    suite = value.get("suite")
    routes = value.get("routes")
    if not isinstance(suite, dict) or suite.get("atomic") is not True or suite.get("complete") is not True:
        raise RuntimeTableError("runtime table is not a complete atomic suite release")
    if not isinstance(routes, dict) or not routes:
        raise RuntimeTableError("runtime table routes must be non-empty")
    declared = suite.get("route_ids")
    if not isinstance(declared, list) or declared != sorted(routes):
        raise RuntimeTableError("suite route_ids do not exactly name the published routes")
    task = value.get("task") or {}
    protocol = value.get("protocol") or {}
    score_metric = value.get("score_metric") or {}
    semantic_hashes = task.get("semantic_hashes") or {}
    binding_hashes = task.get("binding_hashes") or {}
    source_digest = hashlib.sha256(_canonical({
        "semantic_hashes": semantic_hashes,
        "binding_hashes": binding_hashes,
    })).hexdigest()
    if (not task.get("name") or len(str(task.get("digest") or "")) != 64
            or not isinstance(semantic_hashes, dict) or not semantic_hashes
            or not isinstance(binding_hashes, dict) or not binding_hashes
            or task.get("source_digest") != source_digest):
        raise RuntimeTableError("runtime table lacks canonical task/source digests")
    if protocol.get("schema") != "geak.measurement-protocol/v2" or len(str(protocol.get("hash") or "")) != 64:
        raise RuntimeTableError("runtime table lacks measurement protocol v2")
    _validate_score_metric(score_metric, protocol)
    fallback = value.get("fallback") or {}
    if fallback.get("kind") not in ("canonical", "fail_closed"):
        raise RuntimeTableError("runtime table must declare canonical or fail_closed fallback")
    expected_shapes = set()
    for route_id, row in routes.items():
        if row.get("route") != route_id:
            raise RuntimeTableError("route identity mismatch for %s" % route_id)
        shape = row.get("shape") or {}
        shape_key = tuple(int(shape.get(k, -1)) for k in ("m", "n", "k"))
        if min(shape_key) <= 0 or shape_key in expected_shapes:
            raise RuntimeTableError("invalid or duplicate shape for route %s" % route_id)
        expected_shapes.add(shape_key)
        entries = row.get("entries")
        selection = row.get("selection")
        route_fallback = row.get("fallback") or {}
        if not isinstance(entries, list) or not isinstance(selection, dict):
            raise RuntimeTableError("route %s lacks entries/selection contract" % route_id)
        if entries:
            if (selection.get("kind") != "candidate"
                    or selection.get("status") != "winner"
                    or selection.get("config_ids")
                    != [entry.get("config_id") for entry in entries]):
                raise RuntimeTableError("route %s candidate selection differs from entries" % route_id)
        elif (selection.get("kind") != "canonical"
              or selection.get("status") not in ("unmeasured", "no_winner")
              or route_fallback.get("kind") != "canonical"):
            raise RuntimeTableError("route %s has no explicit canonical fallback" % route_id)
        for rank, entry in enumerate(entries):
            if entry.get("rank") != rank:
                raise RuntimeTableError("route %s ranks are not contiguous" % route_id)
            for key in (
                "config_id", "config", "symbol", "isa_hash",
                "artifact", "artifact_sha256", "source", "source_sha256",
                "compile_manifest", "compile_manifest_sha256",
            ):
                if not entry.get(key):
                    raise RuntimeTableError("route %s entry lacks %s" % (route_id, key))
            if entry.get("task_digest") != task.get("digest"):
                raise RuntimeTableError("route %s task digest differs from suite" % route_id)
            if entry.get("protocol_hash") != protocol.get("hash"):
                raise RuntimeTableError("route %s protocol hash differs from suite" % route_id)
            if (entry.get("score_metric_hash") != score_metric.get("hash")
                    or entry.get("metric_id") != score_metric.get("id")
                    or entry.get("metric_inner") != score_metric.get("inner")):
                raise RuntimeTableError("route %s score metric differs from suite" % route_id)


class RuntimeTableStore:
    """One atomic publication per complete route suite.

    v1 updated one route at a time, exposing mixed revisions to readers.  v3 builds
    the whole object under one lock and makes the single os.replace the publication
    point.  Artifacts are immutable and installed before this method is called.
    """

    def __init__(self, path: Path, *, state_root: Path | None = None) -> None:
        self.path = Path(path)
        self.state_root = Path(state_root) if state_root is not None else None

    @staticmethod
    def empty() -> Dict[str, Any]:
        return {
            "schema": RUNTIME_TABLE_VERSION,
            "revision": 0,
            "updated_at": None,
            "table_digest": None,
            "task": {},
            "protocol": {},
            "score_metric": {},
            "oracle": {},
            "fallback": {"kind": "fail_closed", "reason": "no suite published"},
            "suite": {"atomic": True, "complete": False, "route_ids": []},
            "routes": {},
            "history": [],
        }

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return self.empty()
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeTableError("cannot read runtime table: %s" % exc) from exc
        if value.get("schema") in LEGACY_RUNTIME_TABLE_VERSIONS:
            raise RuntimeTableError("legacy per-route runtime tables are not executable; republish the full suite")
        _validate_v2(value)
        return value

    def publish_suite(
        self,
        routes: Sequence[Route],
        entries_by_route: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        run_id: str,
        receipts_root: Path,
        release: Mapping[str, Any],
    ) -> Dict[str, Any]:
        names = _route_ids(routes)
        if set(entries_by_route) != set(names):
            missing = sorted(set(names) - set(entries_by_route))
            extra = sorted(set(entries_by_route) - set(names))
            raise RuntimeTableError("suite selection mismatch: missing=%s extra=%s" % (missing, extra))
        task = dict(release.get("task") or {})
        protocol = dict(release.get("protocol") or {})
        score_metric = dict(release.get("score_metric") or {})
        oracle = dict(release.get("oracle") or {})
        fallback = dict(release.get("fallback") or {})
        search = dict(release.get("search") or {})
        route_outcomes = release.get("route_outcomes") or {}
        semantic_hashes = task.get("semantic_hashes") or {}
        binding_hashes = task.get("binding_hashes") or {}
        source_digest = hashlib.sha256(_canonical({
            "semantic_hashes": semantic_hashes,
            "binding_hashes": binding_hashes,
        })).hexdigest()
        if (not task.get("name") or len(str(task.get("digest") or "")) != 64
                or not isinstance(semantic_hashes, dict) or not semantic_hashes
                or not isinstance(binding_hashes, dict) or not binding_hashes
                or task.get("source_digest") != source_digest):
            raise RuntimeTableError("release requires task name and 64-byte task/source digests")
        if protocol.get("schema") != "geak.measurement-protocol/v2" or len(str(protocol.get("hash") or "")) != 64:
            raise RuntimeTableError("release requires measurement protocol v2 hash")
        _validate_score_metric(score_metric, protocol)
        if fallback.get("kind") not in ("canonical", "fail_closed"):
            raise RuntimeTableError("release requires an explicit fallback")
        if set(route_outcomes) != set(names):
            raise RuntimeTableError("release requires one selection outcome per route")

        lock_path = (
            self.state_root / "locks" / (self.path.name + ".lock")
            if self.state_root is not None
            else self.path.with_suffix(self.path.suffix + ".lock")
        )
        with _lock(lock_path):
            if self.path.exists():
                raw_previous = json.loads(self.path.read_text(encoding="utf-8"))
                if raw_previous.get("schema") in LEGACY_RUNTIME_TABLE_VERSIONS:
                    if (raw_previous.get("table_digest") != table_digest(raw_previous)
                            or (raw_previous.get("suite") or {}).get("atomic") is not True):
                        raise RuntimeTableError("legacy runtime table is not hash-valid for migration")
                    previous = self.empty()
                    previous["revision"] = int(raw_previous.get("revision", 0))
                    previous["history"] = list(raw_previous.get("history") or [])
                    previous["table_digest"] = raw_previous.get("table_digest")
                else:
                    previous = self.load()
            else:
                previous = self.empty()
            revision = int(previous.get("revision", 0)) + 1
            now = dt.datetime.now(dt.timezone.utc).isoformat()
            table_routes: Dict[str, Any] = {}
            for route in routes:
                selected = entries_by_route[route.name]
                ranked = []
                for rank, raw in enumerate(selected):
                    row = dict(raw)
                    row.update({
                        "rank": rank,
                        "task_digest": task["digest"],
                        "protocol_hash": protocol["hash"],
                        "score_metric_hash": score_metric["hash"],
                        "metric_id": score_metric["id"],
                        "metric_inner": score_metric["inner"],
                    })
                    ranked.append(row)
                outcome = dict(route_outcomes[route.name])
                selection = {
                    "kind": "candidate" if ranked else "canonical",
                    "status": outcome.get("status"),
                    "reason": outcome.get("reason"),
                    "config_ids": [entry["config_id"] for entry in ranked],
                    "candidate_gates": list(outcome.get("candidate_gates") or []),
                }
                table_routes[route.name] = {
                    "route": route.name,
                    "shape": {"m": route.m, "n": route.n, "k": route.k},
                    "weight": route.weight,
                    "entries": ranked,
                    "selection": selection,
                    "fallback": fallback,
                }
            history = list(previous.get("history") or [])
            history.append({
                "revision": revision,
                "run_id": run_id,
                "release_id": release.get("release_id"),
                "route_ids": sorted(names),
                "previous_table_digest": previous.get("table_digest"),
                "updated_at": now,
            })
            value: Dict[str, Any] = {
                "schema": RUNTIME_TABLE_VERSION,
                "revision": revision,
                "release_id": release.get("release_id"),
                "updated_at": now,
                "run_id": run_id,
                "receipts_root": str(receipts_root),
                "task": task,
                "protocol": protocol,
                "score_metric": score_metric,
                "oracle": oracle,
                "fallback": fallback,
                "search": search,
                "suite": {"atomic": True, "complete": True, "route_ids": sorted(names)},
                "routes": table_routes,
                "history": history,
            }
            value["table_digest"] = table_digest(value)
            _validate_v2(value)
            # The live file is replaced exactly once, after every route and hash is
            # present.  A reader sees the old complete suite or the new one, never a mix.
            _atomic_write(self.path, value)
            snapshot = (
                self.state_root / "snapshots" / (self.path.name + ".v%d.json" % revision)
                if self.state_root is not None
                else self.path.with_name(self.path.name + ".v%d.json" % revision)
            )
            _atomic_write(snapshot, value)
            return value

    def update_route(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise RuntimeTableError(
            "per-route publication was removed in runtime_table/v3; call publish_suite once"
        )


class RuntimeSelector:
    def __init__(self, table: Mapping[str, Any]) -> None:
        _validate_v2(table)
        self.table = table

    @classmethod
    def from_path(cls, path: Path) -> "RuntimeSelector":
        return cls(RuntimeTableStore(path).load())

    def select(self, route: str, rank: int = 0) -> Mapping[str, Any]:
        try:
            entries = self.table["routes"][route]["entries"]
        except KeyError as exc:
            raise RuntimeTableError("route %r is absent from runtime table" % route) from exc
        if rank < 0 or rank >= len(entries):
            raise RuntimeTableError("route %r has no rank %d" % (route, rank))
        return entries[rank]

    def select_shape(self, m: int, n: int, k: int, rank: int = 0) -> Mapping[str, Any]:
        wanted = (int(m), int(n), int(k))
        for route, row in self.table["routes"].items():
            shape = row["shape"]
            if (int(shape["m"]), int(shape["n"]), int(shape["k"])) == wanted:
                return self.select(route, rank)
        raise RuntimeTableError("shape %r is absent from runtime table" % (wanted,))
