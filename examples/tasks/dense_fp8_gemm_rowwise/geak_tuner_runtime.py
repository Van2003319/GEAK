"""Task-local loader for an atomically materialized asm_tuner runtime table.

This file deliberately has no dependency on the installed GEAK package.  Engineer
and verifier workspaces are tar copies of one task, so task dispatch must remain
self-contained and must execute the exact artifact named by the table.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


_TABLE_SCHEMA = "geak.runtime_table/v3"
_EXPECTED_SCORE_METRIC_ID = "sustained_throughput"
_EXPECTED_SCORE_METRIC_INNER = 16
_TASK_ROOT = Path(__file__).resolve().parent
_TABLE_RELATIVE_PATH = "asm_tuner_runtime_table.json"
_TABLE_PATH = _TASK_ROOT / _TABLE_RELATIVE_PATH
_LIBRARIES: Dict[str, Any] = {}
_FUNCTIONS: Dict[Tuple[str, str], Any] = {}
_ROUTES_BY_SHAPE: Dict[Tuple[int, int, int], Optional[Tuple[str, Mapping[str, Any]]]] = {}
_TABLE: Optional[Mapping[str, Any]] = None
_UNLOADED = object()
_TABLE_IDENTITY: Any = _UNLOADED


class TunerDispatchError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _table_digest(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("table_digest", None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _score_metric_digest(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("hash", None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(".%s.tmp.%d.%s" % (path.name, os.getpid(), uuid.uuid4().hex))
    try:
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temp), str(path))
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _table_identity(path: Path) -> Optional[Tuple[int, int, int, int]]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))


def _load_table() -> Optional[Mapping[str, Any]]:
    global _TABLE, _TABLE_IDENTITY, _FUNCTIONS, _ROUTES_BY_SHAPE
    identity = _table_identity(_TABLE_PATH)
    if identity == _TABLE_IDENTITY:
        if _TABLE is None and os.environ.get("GEAK_TUNER_REQUIRED") == "1":
            raise TunerDispatchError(
                "asm_tuner is required but no materialized runtime table exists")
        return _TABLE
    if identity is None:
        _TABLE = None
        _TABLE_IDENTITY = None
        _FUNCTIONS = {}
        _ROUTES_BY_SHAPE = {}
        if os.environ.get("GEAK_TUNER_REQUIRED") == "1":
            raise TunerDispatchError("asm_tuner is required but no materialized runtime table exists")
        return None
    for _attempt in range(3):
        before = _table_identity(_TABLE_PATH)
        try:
            value = json.loads(_TABLE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TunerDispatchError("cannot read materialized runtime table: %s" % exc) from exc
        after = _table_identity(_TABLE_PATH)
        if before is not None and before == after:
            identity = after
            break
    else:
        raise TunerDispatchError("runtime table changed repeatedly while loading")
    if value.get("schema") != _TABLE_SCHEMA or value.get("table_digest") != _table_digest(value):
        raise TunerDispatchError("runtime table schema/digest mismatch")
    suite = value.get("suite") or {}
    routes = value.get("routes") or {}
    if suite.get("atomic") is not True or suite.get("complete") is not True:
        raise TunerDispatchError("runtime table is not a complete atomic suite")
    if suite.get("route_ids") != sorted(routes):
        raise TunerDispatchError("runtime table route set is incomplete")
    task = value.get("task") or {}
    semantic = task.get("semantic_hashes") or {}
    binding = task.get("binding_hashes") or {}
    source_digest = hashlib.sha256(_canonical({
        "semantic_hashes": semantic,
        "binding_hashes": binding,
    })).hexdigest()
    if (not semantic or not binding or task.get("source_digest") != source_digest):
        raise TunerDispatchError("runtime table has no valid canonical source digest")
    for relative, expected in semantic.items():
        path = (_TASK_ROOT / relative).resolve()
        if not _within(path, _TASK_ROOT) or not path.is_file() or _sha256(path) != expected:
            raise TunerDispatchError("canonical semantic task hash mismatch: %s" % relative)
    protocol = value.get("protocol") or {}
    if protocol.get("schema") != "geak.measurement-protocol/v2" or len(str(protocol.get("hash") or "")) != 64:
        raise TunerDispatchError("runtime table is not bound to measurement protocol v2")
    score_metric = value.get("score_metric") or {}
    if (score_metric.get("schema") != "geak.score-metric/v1"
            or isinstance(score_metric.get("version"), bool) or score_metric.get("version") != 1
            or score_metric.get("id") != _EXPECTED_SCORE_METRIC_ID
            or score_metric.get("unit") != "ms/call"
            or score_metric.get("inner") != _EXPECTED_SCORE_METRIC_INNER
            or score_metric.get("protocol_hash") != protocol.get("hash")
            or score_metric.get("hash") != _score_metric_digest(score_metric)):
        raise TunerDispatchError("runtime table score metric differs from the task contract")
    ambient_metric = os.environ.get("GEAK_SCORE_METRIC")
    if ambient_metric is not None and ambient_metric != score_metric["id"]:
        raise TunerDispatchError("runtime table score metric differs from GEAK_SCORE_METRIC")
    functions: Dict[Tuple[str, str], Any] = {}
    routes_by_shape: Dict[
        Tuple[int, int, int], Optional[Tuple[str, Mapping[str, Any]]]
    ] = {}
    for route_id, row in routes.items():
        entries = row.get("entries") or []
        selection = row.get("selection") or {}
        route_fallback = row.get("fallback") or {}
        if entries:
            if (selection.get("kind") != "candidate"
                    or selection.get("status") != "winner"
                    or selection.get("config_ids")
                    != [entry.get("config_id") for entry in entries]):
                raise TunerDispatchError("route %s candidate selection differs from entries" % route_id)
        elif (selection.get("kind") != "canonical"
              or selection.get("status") not in ("unmeasured", "no_winner")
              or route_fallback.get("kind") != "canonical"):
            raise TunerDispatchError("route %s has no explicit canonical fallback" % route_id)
        route_shape = row.get("shape") or {}
        shape = tuple(int(route_shape.get(key, -1)) for key in ("m", "n", "k"))
        routes_by_shape[shape] = (route_id, entries[0]) if entries else None
        for rank, entry in enumerate(entries):
            if entry.get("rank") != rank:
                raise TunerDispatchError("route %s ranks are not contiguous" % route_id)
            if (entry.get("task_digest") != task.get("digest")
                    or entry.get("protocol_hash") != protocol.get("hash")
                    or entry.get("score_metric_hash") != score_metric.get("hash")
                    or entry.get("metric_id") != score_metric.get("id")
                    or entry.get("metric_inner") != score_metric.get("inner")):
                raise TunerDispatchError("route %s rank %d binding mismatch" % (route_id, rank))
            functions[(route_id, str(entry["config_id"]))] = _prepare_entry(entry)
    _TABLE = value
    _TABLE_IDENTITY = identity
    _FUNCTIONS = functions
    _ROUTES_BY_SHAPE = routes_by_shape
    return value


def _select(table: Mapping[str, Any], shape: Tuple[int, int, int]) -> Optional[Tuple[str, Mapping[str, Any]]]:
    if shape in _ROUTES_BY_SHAPE:
        return _ROUTES_BY_SHAPE[shape]
    fallback = table.get("fallback") or {}
    if fallback.get("kind") == "canonical":
        return None
    raise TunerDispatchError("shape %r is absent and fallback is fail_closed" % (shape,))


def _prepare_entry(entry: Mapping[str, Any]) -> Any:
    artifact = (_TASK_ROOT / str(entry["artifact"])).resolve()
    source = (_TASK_ROOT / str(entry["source"])).resolve()
    compile_manifest = (_TASK_ROOT / str(entry["compile_manifest"])).resolve()
    if (not _within(artifact, _TASK_ROOT) or not _within(source, _TASK_ROOT) or
            not _within(compile_manifest, _TASK_ROOT)):
        raise TunerDispatchError("runtime table artifact/source/manifest escapes task workspace")
    if (not artifact.is_file() or artifact.is_symlink() or
            _sha256(artifact) != entry.get("artifact_sha256")):
        raise TunerDispatchError("materialized artifact hash mismatch")
    if (not source.is_file() or source.is_symlink() or
            _sha256(source) != entry.get("source_sha256")):
        raise TunerDispatchError("materialized generated-source hash mismatch")
    if (not compile_manifest.is_file() or compile_manifest.is_symlink() or
            _sha256(compile_manifest) != entry.get("compile_manifest_sha256")):
        raise TunerDispatchError("materialized compile-manifest hash mismatch")
    key = "%s:%s" % (artifact, entry["artifact_sha256"])
    library = _LIBRARIES.get(key)
    if library is None:
        try:
            library = ctypes.CDLL(str(artifact))
        except OSError as exc:
            raise TunerDispatchError("cannot load selected artifact: %s" % exc) from exc
        _LIBRARIES[key] = library
    symbol = str(entry["symbol"])
    try:
        launch = getattr(library, symbol)
    except AttributeError as exc:
        raise TunerDispatchError("artifact lacks selected symbol %s" % symbol) from exc
    launch.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 6
    launch.restype = ctypes.c_int
    return launch


def _trace(table: Mapping[str, Any], route: str, entry: Mapping[str, Any]) -> None:
    path = os.environ.get("GEAK_TUNER_TRACE")
    if not path:
        return
    _atomic_json(Path(path), {
        "schema": "geak.asm-dispatch-receipt/v1",
        "route": route,
        "release_id": table.get("release_id"),
        "table_digest": table.get("table_digest"),
        "task_digest": table.get("task", {}).get("digest"),
        "protocol_hash": table.get("protocol", {}).get("hash"),
        "score_metric": table.get("score_metric"),
        "metric_id": entry.get("metric_id"),
        "metric_inner": entry.get("metric_inner"),
        "config_id": entry.get("config_id"),
        "symbol": entry.get("symbol"),
        "artifact_sha256": entry.get("artifact_sha256"),
        "source_sha256": entry.get("source_sha256"),
        "compile_manifest_sha256": entry.get("compile_manifest_sha256"),
    })


def dispatch(adapter: str, args: Tuple[Any, ...]) -> Optional[Any]:
    table = _load_table()
    if table is None:
        return None
    if table.get("task", {}).get("adapter") != adapter:
        raise TunerDispatchError("runtime table adapter mismatch")
    if adapter == "bf16":
        if len(args) != 2:
            raise TunerDispatchError("BF16 tuner dispatch expects A and B")
        a, b = args
        scales = (None, None)
    elif adapter == "fp8_rowwise":
        if len(args) != 4:
            raise TunerDispatchError("FP8 tuner dispatch expects A, B and two scales")
        a, b, a_scale, b_scale = args
        scales = (a_scale, b_scale)
    else:
        raise TunerDispatchError("unsupported canonical adapter %r" % adapter)
    shape = (int(a.shape[0]), int(b.shape[0]), int(a.shape[1]))
    selected = _select(table, shape)
    if selected is None:
        return None
    route, entry = selected
    launch = _FUNCTIONS[(route, str(entry["config_id"]))]

    import torch
    out = torch.empty((shape[0], shape[1]), device=a.device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream(a.device)
    ptr = lambda tensor: ctypes.c_void_p(0 if tensor is None else int(tensor.data_ptr()))
    status = int(launch(
        shape[0], shape[1], shape[2], ptr(a), ptr(b), ptr(scales[0]), ptr(scales[1]),
        ptr(out), ctypes.c_void_p(int(stream.cuda_stream)),
    ))
    if status != 0:
        raise TunerDispatchError("selected HIP launch failed with status %d" % status)
    _trace(table, route, entry)
    return out
