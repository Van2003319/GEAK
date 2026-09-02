#!/usr/bin/env python3
"""Phase-independent configuration search and materialisation capability.

Consumers may select only hash-valid, precompiled registry entries.  Creating a new configuration
also goes through this capability: it executes an argv array without a shell, validates that the
artifact stays inside the authorised workspace, hashes it, and atomically registers the receipt.
There is intentionally no phase argument or phase lock.
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


@contextlib.contextmanager
def _locked(registry: Path) -> Iterator[None]:
    registry.parent.mkdir(parents=True, exist_ok=True)
    with (registry.parent / ".config-search.lock").open("a+b") as lock:
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


def _entry_valid(entry: Mapping[str, Any]) -> tuple[bool, str]:
    if entry.get("precompiled") is not True:
        return False, "not_precompiled"
    if entry.get("status") != "ready":
        return False, f"status_not_ready({entry.get('status')!r})"
    if not isinstance(entry.get("id"), str) or not entry["id"]:
        return False, "config_id_missing"
    if not isinstance(entry.get("config"), dict) or not entry["config"]:
        return False, "config_payload_missing"
    artifact = Path(str(entry.get("artifact") or ""))
    if not _regular(artifact):
        return False, "artifact_missing"
    actual = _sha256(artifact)
    if actual != entry.get("artifact_sha256"):
        return False, f"artifact_hash_mismatch(expected={entry.get('artifact_sha256')},actual={actual})"
    artifact_set = entry.get("artifact_set") or []
    if not isinstance(artifact_set, list):
        return False, "artifact_set_not_array"
    for row in artifact_set:
        path = Path(str((row or {}).get("path") or ""))
        expected = str((row or {}).get("sha256") or "")
        if not _regular(path):
            return False, "nested_artifact_missing(%s)" % path
        nested_actual = _sha256(path)
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
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_hash,
        "precompiled": True,
        "status": "ready",
        "created_unix_ns": now,
        "build_receipt": dict(build_receipt or {}),
        "artifact_set": [dict(row) for row in (artifact_set or [])],
    }
    with _locked(registry_path):
        registry = _load_registry(registry_path)
        replaced = False
        for index, old in enumerate(registry["configs"]):
            if old.get("id") != identifier:
                continue
            valid, _ = _entry_valid(old)
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
            valid, why = _entry_valid(entry)
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
            valid, why = _entry_valid(entry)
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
    receipt_dir = _registry_path(registry_value).parent / "receipts"
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
    rows: list[dict[str, str]] = []
    for artifact in release.get("artifacts") or []:
        for key, hash_key in (("artifact", "artifact_sha256"),
                              ("source", "source_sha256"),
                              ("compile_manifest", "compile_manifest_sha256")):
            rows.append({"kind": key, "path": str(artifact[key]),
                         "sha256": str(artifact[hash_key])})
    return rows


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
        "oracle_manifest_sha256": contract.oracle_sha256,
        "release_id": release["release_id"],
        "variant": chosen,
        "config_ids": sorted(set(release["route_selection"].values())),
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
        },
        artifact_set=_release_artifact_set(release),
    )
    result["release_manifest"] = built["manifest"]
    result["release_id"] = release["release_id"]
    return result


def search_asm_configs(
    registry_value: os.PathLike[str] | str,
    *, task_manifest: os.PathLike[str] | str, cache: os.PathLike[str] | str,
    output: os.PathLike[str] | str, variants: Sequence[str], cases: Sequence[str],
) -> dict[str, Any]:
    _require_lane_gpu_boundary()
    contract = TaskContract.load(Path(task_manifest), require_oracle_compatible=True)
    searched = search_release(
        contract, cache_root=Path(cache), output_dir=Path(output),
        variants=variants, case_ids=cases or None,
    )
    release = searched["release"]
    registry_config = {
        "kind": "asm_tuner_release", "task": contract.task,
        "adapter": contract.adapter_name, "task_digest": contract.task_digest,
        "protocol_hash": contract.protocol["hash"],
        "oracle_manifest_sha256": contract.oracle_sha256,
        "release_id": release["release_id"], "variant": release["variant"],
        "config_ids": sorted(set(release["route_selection"].values())),
        "priority": 0,
    }
    registered = register_config(
        registry_value, config=registry_config, artifact=searched["manifest"],
        mechanism="canonical protocol-v2 paired search",
        build_receipt=searched["summary"], artifact_set=_release_artifact_set(release),
    )
    registered["search"] = searched["summary"]
    return registered


def materialize_asm_config(
    registry_value: os.PathLike[str] | str,
    *,
    config_id: str,
    task_manifest: os.PathLike[str] | str,
    workspace: os.PathLike[str] | str,
) -> dict[str, Any]:
    selected = select_config(registry_value, config_id=config_id)
    if not selected.get("ok"):
        return selected
    entry = selected["config"]
    if (entry.get("config") or {}).get("kind") != "asm_tuner_release":
        raise ConfigSearchError("selected registry row is not an asm_tuner release")
    contract = TaskContract.load(Path(task_manifest), require_oracle_compatible=True)
    release = load_release(Path(entry["artifact"]), contract)
    receipt = materialize_release(
        contract, release, workspace=Path(workspace), registry_id=str(entry["id"]),
    )
    return {
        "ok": True,
        "schema": SCHEMA,
        "phase_independent": True,
        "selection_contract": "precompiled-hash-valid-canonical-suite-only",
        "config": dict(entry),
        "materialization": receipt,
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
    selection_file = Path(workspace) / "asm_tuner_selection.json"
    if not requested and selection_file.is_file():
        incumbent = _read_object(selection_file)
        if (incumbent.get("task") == contract.task and
                incumbent.get("task_digest") == contract.task_digest):
            requested = str(incumbent.get("registry_id") or "").strip()

    query = {
        "kind": "asm_tuner_release",
        "task": contract.task,
        "adapter": contract.adapter_name,
        "task_digest": contract.task_digest,
        "protocol_hash": contract.protocol["hash"],
        "oracle_manifest_sha256": contract.oracle_sha256,
    }
    found = search_configs(registry_value, query=query)
    operations.append({"operation": "search", "matches": len(found["matches"]),
                       "invalidated": found["invalidated"]})
    matches = sorted(found["matches"], key=lambda row: (
        int((row.get("config") or {}).get("priority", 1000000)), str(row.get("id") or "")))
    if requested:
        if not any(row.get("id") == requested for row in matches):
            exact = select_config(registry_value, config_id=requested)
            operations.append({"operation": "select_requested", "id": requested,
                               "ok": bool(exact.get("ok")), "reason": exact.get("reason")})
            if not exact.get("ok"):
                return {"ok": False, "schema": SCHEMA, "phase_independent": True,
                        "reason": "requested_config_not_viable", "operations": operations}
        selected_id = requested
    elif matches:
        selected_id = str(matches[0]["id"])
    else:
        bootstrap = contract.search_space.get("bootstrap_search") or {}
        variants = list(bootstrap.get("variants") or [])
        cases = list(bootstrap.get("cases") or [])
        if variants:
            created = search_asm_configs(
                registry_value, task_manifest=task_manifest, cache=cache,
                output=_registry_path(registry_value).parent / "bootstrap_search",
                variants=variants, cases=cases,
            )
            create_mode = "protocol_v2_paired_bootstrap"
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
    search_asm.add_argument("--variant", action="append", required=True)
    search_asm.add_argument("--case", action="append", default=[])

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
    smoke.add_argument("--case", default="")
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
