#!/usr/bin/env python3
"""Phase-independent configuration search and materialisation capability.

Consumers may select only hash-valid, precompiled registry entries.  Creating a new configuration
also goes through this capability: it executes an argv array without a shell, validates that the
artifact stays inside the authorised workspace, hashes it, and atomically registers the receipt.
Canonical asm releases publish a tracked task-root registry/table/selection contract while keeping
content-addressed release bytes and receipts below the ignored ``.geak_tuner`` directory.  There is
intentionally no phase argument or phase lock.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

# The helper is executed by absolute path from task workspaces, where the repo
# root is not otherwise importable.  The bridge must load the exact asm_tuner
# implementation beside this workflow, never an ambient wheel.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asm_tuner.contracts import ContractError, TaskContract
from asm_tuner.integration import (
    IntegrationError,
    SELECTION_SCHEMA,
    create_release,
    load_release,
    materialize_release,
    search_release,
    smoke_materialization,
    verify_materialization,
)

SCHEMA = "geak.config-search/v2"
LEGACY_SCHEMA = "geak.config-search/v1"
SCHEMA_VERSION = 2
LAUNCH_CONTRACT_SCHEMA = "geak.asm-launch-contract/v1"
ARTIFACT_SET_SCHEMA = "geak.asm-artifact-set/v1"


class ConfigSearchError(RuntimeError):
    """A config cannot safely be searched, built, or registered."""


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigSearchError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigSearchError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ConfigSearchError(f"cannot hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _registry_path(value: os.PathLike[str] | str) -> Path:
    path = Path(value)
    return path / "registry.json" if path.suffix.lower() != ".json" else path


def _registry_state_root(registry: Path) -> Path:
    if registry.name == "asm_tuner_registry.json":
        return registry.parent / ".geak_tuner" / "config_search"
    return registry.parent


def _entry_path(registry: Path, value: object) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else registry.parent / path


def _stored_path(registry: Path, value: os.PathLike[str] | str) -> str:
    path = Path(value).resolve()
    try:
        return path.relative_to(registry.parent.resolve()).as_posix()
    except ValueError:
        return str(path)


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


@contextlib.contextmanager
def _locked(registry: Path) -> Iterator[None]:
    state_root = _registry_state_root(registry)
    state_root.mkdir(parents=True, exist_ok=True)
    with (state_root / ".config-search.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "configs": []}
    value = _read_object(path)
    # v1 rows were single-artifact generic configs.  Their bytes remain valid,
    # but they are not canonical asm releases and can never satisfy a v2 task
    # query.  Migrate the envelope explicitly instead of silently discarding them.
    if value.get("schema") == LEGACY_SCHEMA and value.get("schema_version") == 1:
        value = dict(value)
        value["schema"] = SCHEMA
        value["schema_version"] = SCHEMA_VERSION
        value["migrated_from"] = LEGACY_SCHEMA
    elif value.get("schema") != SCHEMA or value.get("schema_version") != SCHEMA_VERSION:
        raise ConfigSearchError("registry schema mismatch; refusing to reinterpret entries")
    if not isinstance(value.get("configs"), list):
        raise ConfigSearchError("registry configs must be an array")
    return value


def _within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _config_id(config: Mapping[str, Any], artifact_sha256: str) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return "cfg-" + hashlib.sha256(encoded + b"\0" + artifact_sha256.encode()).hexdigest()[:20]


def _entry_valid(entry: Mapping[str, Any], registry: Path) -> tuple[bool, str]:
    if entry.get("precompiled") is not True:
        return False, "not_precompiled"
    if entry.get("status") != "ready":
        return False, f"status_not_ready({entry.get('status')!r})"
    if not isinstance(entry.get("id"), str) or not entry["id"]:
        return False, "config_id_missing"
    if not isinstance(entry.get("config"), dict) or not entry["config"]:
        return False, "config_payload_missing"
    config = entry["config"]
    if config.get("kind") == "asm_tuner_release":
        metric = config.get("score_metric")
        metric_payload = dict(metric) if isinstance(metric, dict) else {}
        metric_digest = metric_payload.pop("hash", None)
        try:
            computed_metric_digest = hashlib.sha256(json.dumps(
                metric_payload, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
        except (TypeError, ValueError):
            return False, "score_metric_binding_missing_or_stale"
        if (not isinstance(metric, dict) or metric.get("schema") != "geak.score-metric/v1"
                or isinstance(metric.get("version"), bool) or metric.get("version") != 1
                or metric.get("id") not in ("single_call_latency", "sustained_throughput")
                or isinstance(metric.get("inner"), bool) or not isinstance(metric.get("inner"), int)
                or (metric.get("id") == "single_call_latency" and metric.get("inner") != 1)
                or (metric.get("id") == "sustained_throughput" and metric.get("inner") <= 1)
                or metric.get("unit") != "ms/call"
                or metric.get("protocol_hash") != config.get("protocol_hash")
                or metric_digest != computed_metric_digest
                or metric.get("hash") != config.get("score_metric_hash")
                or metric.get("id") != config.get("score_metric_id")
                or metric.get("inner") != config.get("score_metric_inner")):
            return False, "score_metric_binding_missing_or_stale"
    artifact = _entry_path(registry, entry.get("artifact"))
    if not _regular(artifact):
        return False, "artifact_missing"
    try:
        actual = _sha256(artifact)
    except OSError as exc:
        return False, "artifact_hash_unavailable(%s)" % type(exc).__name__
    if actual != entry.get("artifact_sha256"):
        return False, f"artifact_hash_mismatch(expected={entry.get('artifact_sha256')},actual={actual})"
    artifact_set = entry.get("artifact_set") or []
    if not isinstance(artifact_set, list):
        return False, "artifact_set_not_array"
    for row in artifact_set:
        path = _entry_path(registry, (row or {}).get("path"))
        expected = str((row or {}).get("sha256") or "")
        if not _regular(path):
            return False, "nested_artifact_missing(%s)" % path
        try:
            nested_actual = _sha256(path)
        except OSError as exc:
            return False, "nested_artifact_hash_unavailable(%s:%s)" % (
                path, type(exc).__name__,
            )
        if nested_actual != expected:
            return False, "nested_artifact_hash_mismatch(%s)" % path
    return True, ""


def register_config(
    registry_value: os.PathLike[str] | str,
    *,
    config: Mapping[str, Any],
    artifact: os.PathLike[str] | str,
    mechanism: str = "",
    build_receipt: Mapping[str, Any] | None = None,
    artifact_set: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    registry_path = _registry_path(registry_value)
    artifact_path = Path(artifact).resolve()
    if not _regular(artifact_path):
        raise ConfigSearchError(f"precompiled artifact missing or not regular: {artifact_path}")
    artifact_hash = _sha256(artifact_path)
    identifier = _config_id(config, artifact_hash)
    now = time.time_ns()
    entry = {
        "id": identifier,
        "config": dict(config),
        "mechanism": mechanism,
        "artifact": _stored_path(registry_path, artifact_path),
        "artifact_sha256": artifact_hash,
        "precompiled": True,
        "status": "ready",
        "created_unix_ns": now,
        "build_receipt": dict(build_receipt or {}),
        "artifact_set": [
            {
                **dict(row),
                "path": _stored_path(registry_path, str(row.get("path") or "")),
            }
            for row in (artifact_set or [])
        ],
    }
    with _locked(registry_path):
        registry = _load_registry(registry_path)
        replaced = False
        for index, old in enumerate(registry["configs"]):
            if old.get("id") != identifier:
                continue
            valid, _ = _entry_valid(old, registry_path)
            if valid:
                return {"ok": True, "schema": SCHEMA, "created": False, "config": old}
            # The content-addressed id is unchanged, so replace the stale row in place. Appending a
            # duplicate id would let exact selection hit the invalid row before the repaired one.
            registry["configs"][index] = entry
            replaced = True
            break
        if not replaced:
            registry["configs"].append(entry)
        registry["updated_unix_ns"] = now
        _atomic_json(registry_path, registry)
    return {"ok": True, "schema": SCHEMA, "created": True, "config": entry}


def _matches(candidate: Mapping[str, Any], query: Mapping[str, Any]) -> bool:
    config = candidate.get("config") or {}
    for key, expected in query.items():
        if config.get(key) != expected:
            return False
    return True


def search_configs(
    registry_value: os.PathLike[str] | str,
    *,
    query: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    registry_path = _registry_path(registry_value)
    query = dict(query or {})
    viable: list[dict[str, Any]] = []
    invalidated: list[dict[str, str]] = []
    with _locked(registry_path):
        registry = _load_registry(registry_path)
        dirty = False
        for entry in registry["configs"]:
            if entry.get("status") == "invalidated":
                continue
            valid, why = _entry_valid(entry, registry_path)
            if not valid:
                entry.update({"status": "invalidated", "invalidated_reason": why,
                              "invalidated_unix_ns": time.time_ns()})
                invalidated.append({"id": str(entry.get("id") or ""), "reason": why})
                dirty = True
                continue
            if _matches(entry, query):
                viable.append(dict(entry))
        if dirty:
            registry["updated_unix_ns"] = time.time_ns()
            _atomic_json(registry_path, registry)
    return {
        "ok": True,
        "schema": SCHEMA,
        "phase_independent": True,
        "selection_contract": "precompiled-and-hash-valid-only",
        "query": query,
        "matches": viable,
        "invalidated": invalidated,
    }


def select_config(
    registry_value: os.PathLike[str] | str,
    *,
    config_id: str,
) -> dict[str, Any]:
    """Select one exact precompiled config, invalidating stale bytes before returning it."""
    if not config_id.strip():
        raise ConfigSearchError("config_id must be non-empty")
    registry_path = _registry_path(registry_value)
    with _locked(registry_path):
        registry = _load_registry(registry_path)
        for entry in registry["configs"]:
            if entry.get("id") != config_id:
                continue
            valid, why = _entry_valid(entry, registry_path)
            if not valid:
                entry.update({"status": "invalidated", "invalidated_reason": why,
                              "invalidated_unix_ns": time.time_ns()})
                registry["updated_unix_ns"] = time.time_ns()
                _atomic_json(registry_path, registry)
                return {
                    "ok": False,
                    "schema": SCHEMA,
                    "phase_independent": True,
                    "selection_contract": "precompiled-and-hash-valid-only",
                    "reason": why,
                    "config_id": config_id,
                }
            return {
                "ok": True,
                "schema": SCHEMA,
                "phase_independent": True,
                "selection_contract": "precompiled-and-hash-valid-only",
                "config": dict(entry),
            }
    return {
        "ok": False,
        "schema": SCHEMA,
        "phase_independent": True,
        "selection_contract": "precompiled-and-hash-valid-only",
        "reason": "config_not_found",
        "config_id": config_id,
    }


def _registry_entry_by_id(
    registry_value: os.PathLike[str] | str, config_id: str,
) -> dict[str, Any] | None:
    registry_path = _registry_path(registry_value)
    with _locked(registry_path):
        registry = _load_registry(registry_path)
        for entry in registry["configs"]:
            if entry.get("id") == config_id:
                return dict(entry)
    return None


def create_config(
    registry_value: os.PathLike[str] | str,
    *,
    request_file: os.PathLike[str] | str,
    workspace: os.PathLike[str] | str,
    timeout: int,
) -> dict[str, Any]:
    request = _read_object(Path(request_file))
    config = request.get("config")
    argv = request.get("builder_argv")
    artifact = Path(str(request.get("artifact") or ""))
    root = Path(workspace).resolve()
    cwd = Path(str(request.get("cwd") or root))
    if not cwd.is_absolute():
        cwd = root / cwd
    if not isinstance(config, dict) or not config:
        raise ConfigSearchError("create request config must be a non-empty object")
    if not isinstance(argv, list) or not argv or not all(isinstance(v, str) and v for v in argv):
        raise ConfigSearchError("builder_argv must be a non-empty string array")
    if not _within(cwd, root):
        raise ConfigSearchError(f"builder cwd escapes authorised workspace: {cwd}")
    artifact_abs = artifact if artifact.is_absolute() else cwd / artifact
    if not _within(artifact_abs, root):
        raise ConfigSearchError(f"artifact escapes authorised workspace: {artifact_abs}")
    env = os.environ.copy()
    extra_env = request.get("env") or {}
    if not isinstance(extra_env, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                  for k, v in extra_env.items()):
        raise ConfigSearchError("env must be a string-to-string object")
    env.update(extra_env)
    started = time.time_ns()
    try:
        proc = subprocess.run(argv, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=max(1, timeout), check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConfigSearchError(f"builder failed to execute: {exc}") from exc
    receipt = {
        "builder_argv": argv,
        "cwd": str(cwd.resolve()),
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "started_unix_ns": started,
        "finished_unix_ns": time.time_ns(),
    }
    receipt_dir = _registry_state_root(_registry_path(registry_value)) / "receipts"
    receipt_name = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
    receipt_path = receipt_dir / f"create-{receipt_name}-{started}.json"
    _atomic_json(receipt_path, receipt)
    if proc.returncode != 0:
        return {"ok": False, "schema": SCHEMA, "created": False,
                "reason": "builder_failed", "build_receipt": str(receipt_path)}
    if not _regular(artifact_abs):
        return {"ok": False, "schema": SCHEMA, "created": False,
                "reason": "builder_produced_no_artifact", "build_receipt": str(receipt_path)}
    result = register_config(registry_value, config=config, artifact=artifact_abs,
                             mechanism=str(request.get("mechanism") or ""),
                             build_receipt={"path": str(receipt_path),
                                            "sha256": _sha256(receipt_path)})
    result["build_receipt"] = str(receipt_path)
    return result


def _require_lane_gpu_boundary() -> None:
    if os.environ.get("GEAK_GPU_LOCK_HELD") != "1":
        raise ConfigSearchError("asm_tuner lane preparation requires gpu_lock.sh")
    if os.environ.get("GEAK_GPU_FENCE_HELD") != "1":
        raise ConfigSearchError("asm_tuner lane preparation requires gpu_fence_run.sh")


def _release_artifact_set(release: Mapping[str, Any]) -> list[dict[str, str]]:
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for artifact in release.get("artifacts") or []:
        for key, hash_key in (("artifact", "artifact_sha256"),
                              ("source", "source_sha256"),
                              ("compile_manifest", "compile_manifest_sha256")):
            row = {"kind": key, "path": str(artifact[key]),
                   "sha256": str(artifact[hash_key])}
            unique[(row["kind"], row["path"], row["sha256"])] = row
    return [unique[key] for key in sorted(unique)]


def create_asm_config(
    registry_value: os.PathLike[str] | str,
    *,
    task_manifest: os.PathLike[str] | str,
    cache: os.PathLike[str] | str,
    variant: str = "",
    require_gpu_boundary: bool = True,
) -> dict[str, Any]:
    if require_gpu_boundary:
        _require_lane_gpu_boundary()
    contract = TaskContract.load(Path(task_manifest), require_oracle_compatible=True)
    chosen = variant or str(contract.search_space["default_variant"])
    built = create_release(contract, cache_root=Path(cache), variant=chosen)
    release = built["release"]
    registry_config = {
        "kind": "asm_tuner_release",
        "task": contract.task,
        "adapter": contract.adapter_name,
        "task_digest": contract.task_digest,
        "protocol_hash": contract.protocol["hash"],
        "score_metric": dict(contract.score_metric),
        "score_metric_hash": contract.score_metric["hash"],
        "score_metric_id": contract.score_metric_id,
        "score_metric_inner": contract.score_metric_inner,
        "oracle_manifest_sha256": contract.oracle_sha256,
        "release_id": release["release_id"],
        "variant": chosen,
        "config_ids": sorted({
            config_id for config_id in release["route_selection"].values()
            if config_id is not None
        }),
        "priority": 100 if chosen == contract.search_space["default_variant"] else 200,
    }
    result = register_config(
        registry_value,
        config=registry_config,
        artifact=built["manifest"],
        mechanism="canonical asm_tuner compile/materialize",
        build_receipt={
            "kind": "asm_tuner_release",
            "cache_hit": built["cache_hit"],
            "task_digest": contract.task_digest,
            "protocol_hash": contract.protocol["hash"],
            "score_metric": dict(contract.score_metric),
        },
        artifact_set=_release_artifact_set(release),
    )
    result["release_manifest"] = built["manifest"]
    result["release_id"] = release["release_id"]
    return result


def search_asm_configs(
    registry_value: os.PathLike[str] | str,
    *, task_manifest: os.PathLike[str] | str, cache: os.PathLike[str] | str,
    output: os.PathLike[str] | str, variants: Sequence[str] = (),
    cases: Sequence[str] = (), bootstrap: bool = False,
) -> dict[str, Any]:
    _require_lane_gpu_boundary()
    contract = TaskContract.load(Path(task_manifest), require_oracle_compatible=True)
    searched = search_release(
        contract, cache_root=Path(cache), output_dir=Path(output),
        variants=variants, case_ids=cases or None, bootstrap=bootstrap,
    )
    release = searched["release"]
    registry_config = {
        "kind": "asm_tuner_release", "task": contract.task,
        "adapter": contract.adapter_name, "task_digest": contract.task_digest,
        "protocol_hash": contract.protocol["hash"],
        "score_metric": dict(contract.score_metric),
        "score_metric_hash": contract.score_metric["hash"],
        "score_metric_id": contract.score_metric_id,
        "score_metric_inner": contract.score_metric_inner,
        "oracle_manifest_sha256": contract.oracle_sha256,
        "release_id": release["release_id"], "variant": release["variant"],
        "config_ids": sorted({
            config_id for ranking in release["route_rankings"].values()
            for config_id in ranking
        }),
        "priority": 0,
    }
    registered = register_config(
        registry_value, config=registry_config, artifact=searched["manifest"],
        mechanism="canonical protocol-v2 paired search",
        build_receipt=searched["summary"], artifact_set=_release_artifact_set(release),
    )
    registered["search"] = searched["summary"]
    return registered


def _materialized_artifact_set(
    workspace: Path, table: Mapping[str, Any],
) -> list[dict[str, str]]:
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for route_id in table.get("suite", {}).get("route_ids") or []:
        route = (table.get("routes") or {}).get(route_id) or {}
        for entry in route.get("entries") or []:
            for kind, path_key, hash_key in (
                ("artifact", "artifact", "artifact_sha256"),
                ("source", "source", "source_sha256"),
                ("compile_manifest", "compile_manifest", "compile_manifest_sha256"),
            ):
                raw = Path(str(entry.get(path_key) or ""))
                expected = str(entry.get(hash_key) or "")
                if raw.is_absolute() or ".." in raw.parts or len(expected) != 64:
                    raise ConfigSearchError(
                        "runtime table has non-portable %s path/hash for %s" % (kind, route_id)
                    )
                path = (workspace / raw).resolve()
                if (not _within(path, workspace) or not _regular(path)
                        or _sha256(path) != expected):
                    raise ConfigSearchError(
                        "runtime table %s is missing or hash-stale for %s" % (kind, route_id)
                    )
                row = {"kind": kind, "path": raw.as_posix(), "sha256": expected}
                unique[(kind, row["path"], expected)] = row
    return [unique[key] for key in sorted(unique)]


def _artifact_set_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    normalized = [
        {"kind": str(row["kind"]), "path": str(row["path"]), "sha256": str(row["sha256"])}
        for row in rows
    ]
    kinds: dict[str, int] = {}
    for row in normalized:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    return {
        "schema": ARTIFACT_SET_SCHEMA,
        "count": len(normalized),
        "kinds": dict(sorted(kinds.items())),
        "sha256": _canonical_digest(normalized),
    }


def _repo_relative(path: Path, root: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ConfigSearchError("%s is outside repository root: %s" % (label, path)) from exc


def _publish_launch_registry(
    source_entry: Mapping[str, Any],
    contract: TaskContract,
    *,
    workspace: Path,
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    workspace = workspace.resolve()
    local = TaskContract.load(
        contract.manifest_path, task_root=workspace, require_oracle_compatible=True,
    )
    registry_path = local.materialization_path("registry", task_root=workspace)
    table_path = local.materialization_path("runtime_table", task_root=workspace)
    selection_path = local.materialization_path("selection_file", task_root=workspace)
    table = _read_object(table_path)
    selection = _read_object(selection_path)
    artifact_rows = _materialized_artifact_set(workspace, table)
    artifact_summary = _artifact_set_summary(artifact_rows)
    table_config_ids = sorted({
        str(entry.get("config_id") or "")
        for route in (table.get("routes") or {}).values()
        for entry in (route.get("entries") or [])
    })
    config = dict(source_entry.get("config") or {})
    if ("" in table_config_ids
            or table_config_ids != sorted(set(config.get("config_ids") or []))):
        raise ConfigSearchError("selected registry release and runtime-table config sets differ")
    if (config.get("release_id") != table.get("release_id")
            or config.get("task_digest") != local.task_digest
            or config.get("oracle_manifest_sha256") != local.oracle_sha256):
        raise ConfigSearchError("selected registry release differs from materialized task/table")

    release_manifest = Path(str(materialization.get("release_manifest") or "")).resolve()
    if not _within(release_manifest, workspace) or not _regular(release_manifest):
        raise ConfigSearchError("materialized release manifest is outside the task workspace")
    release_relative = release_manifest.relative_to(workspace).as_posix()
    release_sha256 = _sha256(release_manifest)

    published_entry = dict(source_entry)
    published_entry.update({
        "artifact": release_relative,
        "artifact_sha256": release_sha256,
        "artifact_set": artifact_rows,
        "published_unix_ns": time.time_ns(),
    })
    build_receipt = {
        "schema": "geak.asm-materialized-release/v1",
        "task": local.task,
        "task_digest": local.task_digest,
        "source_digest": local.source_digest,
        "protocol_hash": local.protocol["hash"],
        "score_metric": dict(local.score_metric),
        "oracle_manifest_sha256": local.oracle_sha256,
        "release_id": table["release_id"],
        "table_digest": table["table_digest"],
        "artifact_set_sha256": artifact_summary["sha256"],
        "search_id": str((table.get("search") or {}).get("search_id") or ""),
    }
    published_entry["build_receipt"] = build_receipt
    valid, reason = _entry_valid(published_entry, registry_path)
    if not valid:
        raise ConfigSearchError("cannot publish materialized registry row: %s" % reason)

    consumer_path = contract.repo_root / "kernel_workflow" / "kernel_lane.js"
    manifest_path = contract.manifest_path
    loader_relative = local.materialization_relative_path("dispatch_module")
    loader_path = workspace / loader_relative
    if not _regular(consumer_path) or not _regular(loader_path):
        raise ConfigSearchError("workflow consumer or task runtime loader is missing")
    task_prefix = Path(str(contract.value["task_root"]))
    launch_contract = {
        "schema": LAUNCH_CONTRACT_SCHEMA,
        "task": local.task,
        "task_digest": local.task_digest,
        "source_digest": local.source_digest,
        "protocol": {
            "schema": local.protocol["schema"],
            "version": local.protocol["version"],
            "hash": local.protocol["hash"],
        },
        "protocol_hash": local.protocol["hash"],
        "score_metric": dict(local.score_metric),
        "score_metric_hash": local.score_metric["hash"],
        "score_metric_id": local.score_metric_id,
        "score_metric_inner": local.score_metric_inner,
        "oracle_manifest_sha256": local.oracle_sha256,
        "registry_id": str(published_entry["id"]),
        "release_id": str(table["release_id"]),
        "registry": str(local.materialization_relative_path("registry")),
        "runtime_table": str(local.materialization_relative_path("runtime_table")),
        "runtime_table_path_base": "task_root",
        "runtime_table_sha256": _sha256(table_path),
        "selection": {
            "path": str(local.materialization_relative_path("selection_file")),
            "path_base": "task_root",
            "sha256": _sha256(selection_path),
        },
        "task_manifest": {
            "path": _repo_relative(manifest_path, contract.repo_root, "task manifest"),
            "sha256": _sha256(manifest_path),
        },
        "task_runtime_loader": {
            "path": (task_prefix / loader_relative).as_posix(),
            "sha256": _sha256(loader_path),
            "runtime_table": str(local.materialization_relative_path("runtime_table")),
        },
        "workflow_consumer": {
            "path": _repo_relative(consumer_path, contract.repo_root, "workflow consumer"),
            "sha256": _sha256(consumer_path),
        },
        "artifact_set": artifact_summary,
    }
    registry = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "configs": [published_entry],
        "launch_contract": launch_contract,
        "updated_unix_ns": time.time_ns(),
    }
    with _locked(registry_path):
        _atomic_json(registry_path, registry)
    return {
        "registry": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "runtime_table": str(table_path),
        "runtime_table_sha256": launch_contract["runtime_table_sha256"],
        "selection_file": str(selection_path),
        "selection_sha256": launch_contract["selection"]["sha256"],
        "launch_contract": launch_contract,
        "config": published_entry,
    }


def _write_publication_receipt(
    workspace: Path,
    materialization: Mapping[str, Any],
    publication: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "schema": "geak.asm-tracked-publication/v1",
        "task": publication["launch_contract"]["task"],
        "task_digest": publication["launch_contract"]["task_digest"],
        "source_digest": publication["launch_contract"]["source_digest"],
        "release_id": publication["launch_contract"]["release_id"],
        "registry_id": publication["launch_contract"]["registry_id"],
        "protocol_hash": publication["launch_contract"]["protocol_hash"],
        "score_metric": publication["launch_contract"]["score_metric"],
        "oracle_manifest_sha256": publication["launch_contract"]["oracle_manifest_sha256"],
        "artifact_set": publication["launch_contract"]["artifact_set"],
        "published_files": [
            {"kind": "registry", "path": publication["registry"],
             "sha256": publication["registry_sha256"]},
            {"kind": "runtime_table", "path": publication["runtime_table"],
             "sha256": publication["runtime_table_sha256"]},
            {"kind": "selection", "path": publication["selection_file"],
             "sha256": publication["selection_sha256"]},
        ],
        "materialization_receipt": materialization.get("receipt"),
        "created_unix_ns": time.time_ns(),
    }
    receipt_id = _canonical_digest(core)
    receipt_path = workspace / ".geak_tuner" / "receipts" / (
        "tracked-publication-%s.json" % receipt_id
    )
    value = {**core, "receipt_id": receipt_id}
    _atomic_json(receipt_path, value)
    return {
        **value,
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
    }


def materialize_asm_config(
    registry_value: os.PathLike[str] | str,
    *,
    config_id: str,
    task_manifest: os.PathLike[str] | str,
    workspace: os.PathLike[str] | str,
) -> dict[str, Any]:
    source_registry = _registry_path(registry_value)
    selected = select_config(registry_value, config_id=config_id)
    if not selected.get("ok"):
        return selected
    entry = selected["config"]
    if (entry.get("config") or {}).get("kind") != "asm_tuner_release":
        raise ConfigSearchError("selected registry row is not an asm_tuner release")
    contract = TaskContract.load(Path(task_manifest), require_oracle_compatible=True)
    config = entry.get("config") or {}
    if (config.get("score_metric") != dict(contract.score_metric)
            or config.get("score_metric_hash") != contract.score_metric["hash"]
            or config.get("score_metric_id") != contract.score_metric_id
            or config.get("score_metric_inner") != contract.score_metric_inner):
        raise ConfigSearchError("selected registry row score metric differs from task contract")
    release = load_release(_entry_path(source_registry, entry["artifact"]), contract)
    workspace_path = Path(workspace).resolve()
    receipt = materialize_release(
        contract, release, workspace=workspace_path, registry_id=str(entry["id"]),
    )
    publication = _publish_launch_registry(
        entry, contract, workspace=workspace_path, materialization=receipt,
    )
    publication_receipt = _write_publication_receipt(
        workspace_path, receipt, publication,
    )
    receipt = {
        **receipt,
        "published_registry": publication["registry"],
        "published_registry_sha256": publication["registry_sha256"],
        "publication_receipt": publication_receipt["receipt"],
        "publication_receipt_sha256": publication_receipt["receipt_sha256"],
    }
    return {
        "ok": True,
        "schema": SCHEMA,
        "phase_independent": True,
        "selection_contract": "precompiled-hash-valid-canonical-suite-only",
        "config": publication["config"],
        "materialization": receipt,
        "publication": publication_receipt,
    }


def prepare_asm_config(
    registry_value: os.PathLike[str] | str,
    *,
    task_manifest: os.PathLike[str] | str,
    workspace: os.PathLike[str] | str,
    cache: os.PathLike[str] | str,
    config_id: str = "",
) -> dict[str, Any]:
    """Deterministic search -> create(if absent) -> select -> materialize barrier."""
    _require_lane_gpu_boundary()
    contract = TaskContract.load(Path(task_manifest), require_oracle_compatible=True)
    operations: list[dict[str, Any]] = []
    requested = config_id.strip()
    explicit_request = bool(requested)
    implicit_tracked_request = False
    selection_file = contract.materialization_path(
        "selection_file", task_root=Path(workspace),
    )
    if not requested and selection_file.is_file():
        incumbent = _read_object(selection_file)
        expected_selection = {
            "schema": SELECTION_SCHEMA,
            "task": contract.task,
            "task_digest": contract.task_digest,
            "source_digest": contract.source_digest,
            "protocol_hash": contract.protocol["hash"],
            "score_metric": dict(contract.score_metric),
            "score_metric_id": contract.score_metric_id,
            "score_metric_inner": contract.score_metric_inner,
            "oracle_manifest_sha256": contract.oracle_sha256,
        }
        mismatches = {
            key: {"tracked": incumbent.get(key), "expected": expected}
            for key, expected in expected_selection.items()
            if incumbent.get(key) != expected
        }
        requested = str(incumbent.get("registry_id") or "").strip()
        if mismatches or not requested:
            operations.append({
                "operation": "validate_tracked_selection",
                "ok": False,
                "registry_id": requested,
                "mismatches": mismatches,
            })
            return {
                "ok": False,
                "schema": SCHEMA,
                "phase_independent": True,
                "reason": "tracked_selection_contract_mismatch",
                "operations": operations,
            }
        implicit_tracked_request = True
        operations.append({
            "operation": "validate_tracked_selection",
            "ok": True,
            "registry_id": requested,
        })

    query = {
        "kind": "asm_tuner_release",
        "task": contract.task,
        "adapter": contract.adapter_name,
        "task_digest": contract.task_digest,
        "protocol_hash": contract.protocol["hash"],
        "score_metric": dict(contract.score_metric),
        "score_metric_hash": contract.score_metric["hash"],
        "score_metric_id": contract.score_metric_id,
        "score_metric_inner": contract.score_metric_inner,
        "oracle_manifest_sha256": contract.oracle_sha256,
    }
    found = search_configs(registry_value, query=query)
    operations.append({"operation": "search", "matches": len(found["matches"]),
                       "invalidated": found["invalidated"]})
    matches = sorted(found["matches"], key=lambda row: (
        int((row.get("config") or {}).get("priority", 1000000)), str(row.get("id") or "")))
    if requested:
        if not any(row.get("id") == requested for row in matches):
            invalidated_entry = _registry_entry_by_id(registry_value, requested)
            invalidation = next(
                (row for row in found["invalidated"] if row.get("id") == requested),
                None,
            )
            invalidation_reason = str(
                (invalidation or {}).get("reason")
                or (
                    invalidated_entry.get("invalidated_reason")
                    if invalidated_entry is not None
                    and invalidated_entry.get("status") == "invalidated"
                    else ""
                )
                or ""
            )
            hydration_unavailable = (
                invalidation_reason == "artifact_missing"
                or invalidation_reason.startswith("nested_artifact_missing(")
                or invalidation_reason.startswith("artifact_hash_unavailable(")
                or invalidation_reason.startswith("nested_artifact_hash_unavailable(")
            )
            invalidation_contract_match = (
                invalidated_entry is not None and _matches(invalidated_entry, query)
            )
            if (implicit_tracked_request and hydration_unavailable
                    and invalidation_contract_match):
                operations.append({
                    "operation": "clear_implicit_requested",
                    "id": requested,
                    "reason": invalidation_reason,
                    "recovery": "bounded_search_create_materialize",
                })
                requested = ""
                found = search_configs(registry_value, query=query)
                operations.append({
                    "operation": "search_after_invalidation",
                    "matches": len(found["matches"]),
                    "invalidated": found["invalidated"],
                })
                matches = sorted(found["matches"], key=lambda row: (
                    int((row.get("config") or {}).get("priority", 1000000)),
                    str(row.get("id") or ""),
                ))
            else:
                exact = select_config(registry_value, config_id=requested)
                contract_match = bool(exact.get("ok")) and _matches(exact["config"], query)
                operations.append({
                    "operation": "select_requested",
                    "id": requested,
                    "source": "explicit" if explicit_request else "tracked_selection",
                    "ok": bool(exact.get("ok")) and contract_match,
                    "reason": (
                        exact.get("reason") if not exact.get("ok")
                        else ("requested_contract_mismatch" if not contract_match else None)
                    ),
                })
                return {"ok": False, "schema": SCHEMA, "phase_independent": True,
                        "reason": "requested_config_not_viable", "operations": operations}
        if requested:
            selected_id = requested
        elif matches:
            selected_id = str(matches[0]["id"])
        else:
            selected_id = ""
    elif matches:
        selected_id = str(matches[0]["id"])
    else:
        selected_id = ""

    if not selected_id:
        bootstrap = contract.search_space.get("bootstrap_search") or {}
        variants = list(bootstrap.get("variants") or [])
        cases = list(bootstrap.get("cases") or [])
        if cases:
            created = search_asm_configs(
                registry_value, task_manifest=task_manifest, cache=cache,
                output=Path(cache).resolve().parent / "bootstrap_search",
                variants=variants, cases=cases, bootstrap=True,
            )
            create_mode = "protocol_v2_generated_halving_bootstrap"
        else:
            created = create_asm_config(
                registry_value, task_manifest=task_manifest, cache=cache,
                require_gpu_boundary=False,
            )
            create_mode = "declared_seed"
        operations.append({"operation": "create", "mode": create_mode,
                           "ok": bool(created.get("ok")),
                           "id": (created.get("config") or {}).get("id")})
        if not created.get("ok"):
            return {"ok": False, "schema": SCHEMA, "phase_independent": True,
                    "reason": "create_failed", "operations": operations}
        selected_id = str(created["config"]["id"])

    selected = select_config(registry_value, config_id=selected_id)
    operations.append({"operation": "select", "id": selected_id,
                       "ok": bool(selected.get("ok")), "reason": selected.get("reason")})
    if not selected.get("ok"):
        return {"ok": False, "schema": SCHEMA, "phase_independent": True,
                "reason": "selection_failed", "operations": operations}
    materialized = materialize_asm_config(
        registry_value, config_id=selected_id, task_manifest=task_manifest,
        workspace=workspace,
    )
    operations.append({"operation": "materialize", "id": selected_id,
                       "ok": bool(materialized.get("ok")),
                       "table_digest": (materialized.get("materialization") or {}).get("table_digest")})
    materialized["operations"] = operations
    materialized["prepared_before_engineer"] = True
    return materialized


def verify_asm_materialization(
    *, task_manifest: os.PathLike[str] | str, workspace: os.PathLike[str] | str,
    rebuild: bool, cache: os.PathLike[str] | str | None = None,
    independent: bool = True,
) -> dict[str, Any]:
    contract = TaskContract.load(Path(task_manifest), require_oracle_compatible=True)
    return verify_materialization(
        contract, workspace=Path(workspace), rebuild=rebuild,
        cache_root=Path(cache) if cache else None, independent=independent,
    )


def smoke_asm_materialization(
    *, task_manifest: os.PathLike[str] | str, workspace: os.PathLike[str] | str,
    case_id: str = "",
) -> dict[str, Any]:
    _require_lane_gpu_boundary()
    contract = TaskContract.load(Path(task_manifest), require_oracle_compatible=True)
    return smoke_materialization(contract, workspace=Path(workspace), case_id=case_id)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search")
    search.add_argument("--registry", required=True)
    search.add_argument("--query-file", default="")

    select = sub.add_parser("select")
    select.add_argument("--registry", required=True)
    select.add_argument("--id", required=True)

    register = sub.add_parser("register")
    register.add_argument("--registry", required=True)
    register.add_argument("--config-file", required=True)
    register.add_argument("--artifact", required=True)
    register.add_argument("--mechanism", default="")

    create = sub.add_parser("create")
    create.add_argument("--registry", required=True)
    create.add_argument("--request-file", required=True)
    create.add_argument("--workspace", required=True)
    create.add_argument("--timeout", type=int, default=1800)

    search_asm = sub.add_parser("search-asm")
    search_asm.add_argument("--registry", required=True)
    search_asm.add_argument("--task-manifest", required=True)
    search_asm.add_argument("--cache", required=True)
    search_asm.add_argument("--output", required=True)
    search_asm.add_argument("--variant", action="append", default=[])
    search_asm.add_argument("--case", action="append", default=[])
    search_asm.add_argument("--bootstrap", action="store_true")

    create_asm = sub.add_parser("create-asm")
    create_asm.add_argument("--registry", required=True)
    create_asm.add_argument("--task-manifest", required=True)
    create_asm.add_argument("--cache", required=True)
    create_asm.add_argument("--variant", default="")

    materialize = sub.add_parser("materialize")
    materialize.add_argument("--registry", required=True)
    materialize.add_argument("--id", required=True)
    materialize.add_argument("--task-manifest", required=True)
    materialize.add_argument("--workspace", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--registry", required=True)
    prepare.add_argument("--task-manifest", required=True)
    prepare.add_argument("--workspace", required=True)
    prepare.add_argument("--cache", required=True)
    prepare.add_argument("--id", default="")

    verify = sub.add_parser("verify-materialization")
    verify.add_argument("--task-manifest", required=True)
    verify.add_argument("--workspace", required=True)
    verify.add_argument("--cache", default="")
    verify.add_argument("--rebuild", action="store_true")
    verify.add_argument("--in-place", action="store_true")

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--task-manifest", required=True)
    smoke.add_argument("--workspace", required=True)
    smoke.add_argument("--case", required=True,
                       help="candidate-owned route whose winner dispatch must be confirmed")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "search":
            query = _read_object(Path(args.query_file)) if args.query_file else {}
            result = search_configs(args.registry, query=query)
        elif args.command == "select":
            result = select_config(args.registry, config_id=args.id)
        elif args.command == "register":
            config = _read_object(Path(args.config_file))
            result = register_config(args.registry, config=config, artifact=args.artifact,
                                     mechanism=args.mechanism)
        elif args.command == "create":
            result = create_config(args.registry, request_file=args.request_file,
                                   workspace=args.workspace, timeout=args.timeout)
        elif args.command == "search-asm":
            result = search_asm_configs(
                args.registry, task_manifest=args.task_manifest, cache=args.cache,
                output=args.output, variants=args.variant, cases=args.case,
                bootstrap=args.bootstrap,
            )
        elif args.command == "create-asm":
            result = create_asm_config(
                args.registry, task_manifest=args.task_manifest, cache=args.cache,
                variant=args.variant,
            )
        elif args.command == "materialize":
            result = materialize_asm_config(
                args.registry, config_id=args.id, task_manifest=args.task_manifest,
                workspace=args.workspace,
            )
        elif args.command == "prepare":
            result = prepare_asm_config(
                args.registry, task_manifest=args.task_manifest,
                workspace=args.workspace, cache=args.cache, config_id=args.id,
            )
        elif args.command == "verify-materialization":
            result = verify_asm_materialization(
                task_manifest=args.task_manifest, workspace=args.workspace,
                rebuild=args.rebuild, cache=args.cache or None,
                independent=not args.in_place,
            )
        else:
            result = smoke_asm_materialization(
                task_manifest=args.task_manifest, workspace=args.workspace,
                case_id=args.case,
            )
        sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
        return 0 if result.get("ok") else 2
    except (ConfigSearchError, ContractError, IntegrationError, OSError, ValueError) as exc:
        sys.stdout.write(json.dumps({"ok": False, "schema": SCHEMA, "error": str(exc)},
                                    sort_keys=True, separators=(",", ":")) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
