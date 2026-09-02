#!/usr/bin/env python3
"""Deterministic protocol-v2 paired producer and route-gate adapter.

The lane owns the session and planner target routes.  Task code emits raw
``geak.measurement-receipt/v2`` blocks; this module validates those blocks and
derives pair ids.  Oracle arms may remain in the task receipt for absolute
reporting, but only candidate/incumbent samples reach the promotion gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from workflow_state import canonical_hashes

RAW_SUITE_SCHEMA = "geak.paired-measurement-suite/v2"
TASK_RECEIPT_SCHEMA = "geak.paired-verify-task-receipt/v2"
MEASUREMENT_SCHEMA = "geak.measurement-receipt/v2"
PROTOCOL_SCHEMA = "geak.measurement-protocol/v2"
SCORE_METRIC_SCHEMA = "geak.score-metric/v1"
SCORE_METRIC_IDS = ("single_call_latency", "sustained_throughput")
GATE_INPUT_SCHEMA = "geak.route-gate-input/v2"
BINDING_SCHEMA = "geak.paired-verify-binding/v2"
MIN_PAIRS = 8
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_GATE_CONFIG = frozenset({
    "practical_mde", "non_regression_margin",
    "suite_non_inferiority_margin", "route_weights",
})


class PairedVerifyError(ValueError):
    """Raw task evidence cannot support paired promotion."""


def _json_number_normalize(value: object) -> object:
    # JSON has one number type. JavaScript therefore reserializes 1.0 as 1; normalize that
    # representation before hashing so the producer, V8 transport, and adapter agree.
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_json_number_normalize(item) for item in value]
    if isinstance(value, tuple):
        return [_json_number_normalize(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_number_normalize(item) for key, item in value.items()}
    return value


def canonical_json(value: object) -> str:
    return json.dumps(_json_number_normalize(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _load_json(path: os.PathLike[str] | str) -> object:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedVerifyError(f"cannot read JSON receipt {path}: {exc}") from exc


def _names(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise PairedVerifyError(f"{label} must be a non-empty array")
    out: list[str] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, str) or not raw.strip() or raw != raw.strip():
            raise PairedVerifyError(f"{label}[{index}] is blank or non-canonical")
        if raw in out:
            raise PairedVerifyError(f"{label} repeats {raw!r}")
        out.append(raw)
    return out


def _positive(value: object, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) <= 0):
        raise PairedVerifyError(f"{label} must be a finite positive number")
    return float(value)


def _validate_hashes(value: object, label: str) -> None:
    if not isinstance(value, Mapping):
        raise PairedVerifyError(f"{label} is missing")
    for kind in ("source", "asm", "binary"):
        row = value.get(kind)
        if (not isinstance(row, Mapping) or row.get("algorithm") != "sha256"
                or not isinstance(row.get("sha256"), str)
                or SHA256_RE.fullmatch(row["sha256"]) is None
                or isinstance(row.get("files"), bool) or not isinstance(row.get("files"), int)
                or row["files"] < 0 or not isinstance(row.get("present"), bool)
                or row["present"] != (row["files"] > 0)):
            raise PairedVerifyError(f"{label}.{kind} is not a canonical hash record")


def _protocol_hash(protocol: Mapping[str, object]) -> str:
    payload = dict(protocol)
    payload.pop("hash", None)
    return canonical_sha256(payload)


def _validate_protocol(receipt: Mapping[str, object], label: str) -> tuple[dict[str, Any], str]:
    protocol = receipt.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise PairedVerifyError(f"{label} has no measurement-protocol/v2 object")
    protocol = dict(protocol)
    digest = _protocol_hash(protocol)
    if protocol.get("hash") != digest or receipt.get("protocol_hash") != digest:
        raise PairedVerifyError(f"{label} protocol hash does not match canonical contents")
    return protocol, digest


def _score_metric_hash(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("hash", None)
    return canonical_sha256(payload)


def _validate_score_metric(
    value: object, protocol: Mapping[str, object], label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != SCORE_METRIC_SCHEMA:
        raise PairedVerifyError(f"{label} is not a {SCORE_METRIC_SCHEMA} object")
    metric = dict(value)
    metric_id = metric.get("id")
    inner = metric.get("inner")
    protocol_metrics = protocol.get("metrics")
    protocol_metric = (protocol_metrics.get(metric_id)
                       if isinstance(protocol_metrics, Mapping) else None)
    if (isinstance(metric.get("version"), bool) or metric.get("version") != 1
            or metric_id not in SCORE_METRIC_IDS
            or metric.get("hash") != _score_metric_hash(metric)
            or metric.get("protocol_hash") != protocol.get("hash")
            or not isinstance(protocol_metric, Mapping)
            or protocol_metric.get("id") != metric_id
            or protocol_metric.get("unit") != metric.get("unit")
            or protocol_metric.get("inner") != inner):
        raise PairedVerifyError(f"{label} is malformed or differs from its protocol metric")
    if (metric_id == "single_call_latency"
            and (isinstance(inner, bool) or inner != 1)):
        raise PairedVerifyError(f"{label} latency must use inner=1")
    if (metric_id == "sustained_throughput" and (
            isinstance(inner, bool) or not isinstance(inner, int) or inner <= 1
            or protocol.get("slot_policy") != "semantic_weight_rotation"
            or protocol.get("cache_policy") != "semantic_weight_rotating")):
        raise PairedVerifyError(
            f"{label} sustained throughput requires inner>1 semantic same-slot rotation")
    return metric


def _validate_lifecycle(receipt: Mapping[str, object], arms: Sequence[str], label: str) -> None:
    lifecycle = receipt.get("output_lifecycle")
    if not isinstance(lifecycle, Mapping) or set(lifecycle) != set(arms):
        raise PairedVerifyError(f"{label} output lifecycle does not name every arm")
    for arm in arms:
        row = lifecycle.get(arm)
        if not isinstance(row, Mapping):
            raise PairedVerifyError(f"{label} output lifecycle for {arm!r} is missing")
        ring, observed, unique = (row.get("ring_size"), row.get("observed_addresses"),
                                  row.get("unique_addresses"))
        if (isinstance(ring, bool) or not isinstance(ring, int) or ring < 2
                or isinstance(observed, bool) or not isinstance(observed, int) or observed < 2
                or isinstance(unique, bool) or not isinstance(unique, int) or unique < 2
                or unique > observed or row.get("addresses_rotated") is not True):
            raise PairedVerifyError(f"{label} output lifecycle for {arm!r} is not a live ring")


def validate_measurement_receipt(
    value: object, label: str, *, required_lock_path: str | None = None,
    expected_score_metric: Mapping[str, object] | None = None,
) -> tuple[dict[str, Any], str]:
    """Recompute slot/raw invariants instead of trusting their summary booleans."""
    if not isinstance(value, Mapping) or value.get("schema") != MEASUREMENT_SCHEMA:
        raise PairedVerifyError(f"{label} is not a {MEASUREMENT_SCHEMA} receipt")
    receipt = dict(value)
    protocol, protocol_digest = _validate_protocol(receipt, label)
    metric = receipt.get("metric")
    metric_id = receipt.get("metric_id")
    protocol_metrics = protocol.get("metrics")
    protocol_metric = (protocol_metrics.get(metric_id)
                       if isinstance(protocol_metrics, Mapping) else None)
    if (metric_id not in SCORE_METRIC_IDS or not isinstance(metric, Mapping)
            or not isinstance(protocol_metric, Mapping)
            or canonical_json(metric) != canonical_json(protocol_metric)):
        raise PairedVerifyError(f"{label} metric identity differs from its protocol")
    metric_inner = metric.get("inner")
    if (metric_id == "single_call_latency"
            and (isinstance(metric_inner, bool) or metric_inner != 1)):
        raise PairedVerifyError(f"{label} latency must use inner=1")
    if (metric_id == "sustained_throughput" and (
            isinstance(metric_inner, bool) or not isinstance(metric_inner, int)
            or metric_inner <= 1
            or protocol.get("slot_policy") != "semantic_weight_rotation"
            or protocol.get("cache_policy") != "semantic_weight_rotating")):
        raise PairedVerifyError(
            f"{label} sustained throughput requires inner>1 semantic same-slot rotation")
    if expected_score_metric is not None:
        expected = _validate_score_metric(expected_score_metric, protocol, f"{label} score_metric")
        if expected.get("id") != metric_id or expected.get("inner") != metric_inner:
            raise PairedVerifyError(f"{label} receipt metric/inner differs from score_metric")
    arms = receipt.get("arms")
    if (not isinstance(arms, list) or any(not isinstance(v, str) or not v for v in arms)
            or len(arms) != len(set(arms)) or "candidate" not in arms
            or "incumbent" not in arms):
        raise PairedVerifyError(f"{label} must contain unique candidate and incumbent arms")
    lock = receipt.get("gpu_lock")
    if (not isinstance(lock, Mapping) or lock.get("mode") != "external"
            or not isinstance(lock.get("path"), str) or not lock.get("path")):
        raise PairedVerifyError(f"{label} was not measured under one external GPU lock")
    if required_lock_path is not None and lock.get("path") != required_lock_path:
        raise PairedVerifyError(f"{label} GPU lock differs from the producer lock")
    invariants = receipt.get("invariants")
    metric_invariant = ("single_call_latency_inner_is_one" if metric_id == "single_call_latency"
                        else "sustained_throughput_inner_gt_one")
    required = (
        "all_arms_in_every_block", "same_slot_within_block",
        "arm_positions_balanced_per_slot", "semantic_weight_working_set_exceeds_cache",
        metric_invariant,
    )
    if (not isinstance(invariants, Mapping)
            or any(invariants.get(name) is not True for name in required)
            or any(flag is not True for flag in invariants.values())):
        raise PairedVerifyError(f"{label} reports a failed measurement invariant")
    slots = receipt.get("slots")
    if not isinstance(slots, list) or not slots:
        raise PairedVerifyError(f"{label} has no semantic slot inventory")
    slot_by_index: dict[int, Mapping[str, object]] = {}
    semantic_bytes = 0
    for index, slot in enumerate(slots):
        if (not isinstance(slot, Mapping) or slot.get("slot_index") != index
                or not isinstance(slot.get("slot_id"), (str, int))
                or not isinstance(slot.get("fingerprint"), str) or not slot.get("fingerprint")):
            raise PairedVerifyError(f"{label} slot inventory is malformed at index {index}")
        weight_bytes = slot.get("weight_bytes", 0)
        if isinstance(weight_bytes, bool) or not isinstance(weight_bytes, int) or weight_bytes < 0:
            raise PairedVerifyError(f"{label} slot {index} has invalid weight_bytes")
        semantic_bytes += weight_bytes
        slot_by_index[index] = slot
    if receipt.get("semantic_weight_working_set_bytes") != semantic_bytes:
        raise PairedVerifyError(f"{label} semantic working-set total is stale")
    if (protocol.get("slot_policy") == "semantic_weight_rotation"
            and semantic_bytes <= int(protocol.get("assumed_cache_bytes", 0))):
        raise PairedVerifyError(f"{label} semantic slots do not exceed the cache assumption")
    blocks = receipt.get("blocks")
    if not isinstance(blocks, list) or len(blocks) < MIN_PAIRS:
        raise PairedVerifyError(f"{label} needs at least {MIN_PAIRS} complete outer blocks")
    flat: list[object] = []
    for offset, block in enumerate(blocks):
        if not isinstance(block, Mapping) or block.get("block") != offset:
            raise PairedVerifyError(f"{label} block ids must be contiguous from zero")
        order, samples = block.get("order"), block.get("samples")
        if (not isinstance(order, list) or set(order) != set(arms) or len(order) != len(arms)
                or not isinstance(samples, Mapping) or set(samples) != set(arms)):
            raise PairedVerifyError(f"{label} block {offset} does not contain every arm once")
        fingerprint, slot_id = block.get("slot_fingerprint"), block.get("slot_id")
        slot_index = block.get("slot_index")
        slot = slot_by_index.get(slot_index) if isinstance(slot_index, int) else None
        if (slot is None or slot.get("slot_id") != slot_id
                or slot.get("fingerprint") != fingerprint):
            raise PairedVerifyError(f"{label} block {offset} is not bound to its declared slot")
        for position, arm in enumerate(order):
            sample = samples.get(arm)
            if (not isinstance(sample, Mapping) or sample.get("arm") != arm
                    or sample.get("block") != offset or sample.get("position") != position
                    or sample.get("cycle") != block.get("cycle")
                    or sample.get("slot_index") != slot_index or sample.get("slot_id") != slot_id
                    or sample.get("slot_fingerprint") != fingerprint
                    or sample.get("inner") != metric_inner or sample.get("timer") != "cuda_event"
                    or sample.get("primed") is not True):
                raise PairedVerifyError(
                    f"{label} block {offset} arm {arm!r} violates slot/timer/priming/inner invariants")
            _positive(sample.get("ms"), f"{label} block {offset} arm {arm!r} ms")
            flat.append(sample)
    if canonical_json(receipt.get("samples")) != canonical_json(flat):
        raise PairedVerifyError(f"{label} flat raw samples disagree with outer blocks")
    if receipt.get("slot_order") != [block.get("slot_id") for block in blocks]:
        raise PairedVerifyError(f"{label} slot order disagrees with outer blocks")
    _validate_lifecycle(receipt, arms, label)
    return receipt, protocol_digest


def validate_raw_suite(
    value: object, expected_routes: Sequence[str], *, required_lock_path: str | None = None,
    expected_metric_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not isinstance(value, Mapping) or value.get("schema") != RAW_SUITE_SCHEMA:
        raise PairedVerifyError(f"task output must be {RAW_SUITE_SCHEMA}")
    if value.get("promotion_control") != "incumbent":
        raise PairedVerifyError("task promotion_control must be the untouched incumbent")
    if value.get("oracle_role") != "absolute_report_only":
        raise PairedVerifyError("task oracle_role must be absolute_report_only")
    if not isinstance(value.get("task_entrypoint"), str) or not value["task_entrypoint"]:
        raise PairedVerifyError("task receipt does not name the formal entrypoint")
    for arm in ("candidate", "incumbent"):
        dispatch = value.get(f"{arm}_dispatch")
        if (not isinstance(dispatch, Mapping)
                or dispatch.get("execution_path") != "gemm_wrapper.runtime_table"
                or not isinstance(dispatch.get("table_digest"), str)
                or SHA256_RE.fullmatch(dispatch["table_digest"]) is None
                or not isinstance(dispatch.get("release_id"), str)
                or not dispatch["release_id"]):
            raise PairedVerifyError(
                f"task {arm} arm did not execute through gemm_wrapper/runtime table")
    expected = sorted(_names(list(expected_routes), "expected_routes"))
    declared = sorted(_names(value.get("expected_routes"), "task expected_routes"))
    if declared != expected:
        raise PairedVerifyError(
            f"task receipt is not the complete suite: expected={expected}, measured={declared}")
    raw_score_metric = value.get("score_metric")
    raw_secondary_metric = value.get("secondary_metric")
    if not isinstance(raw_score_metric, Mapping) or not isinstance(raw_secondary_metric, Mapping):
        raise PairedVerifyError("task receipt must bind score_metric and secondary_metric")
    if expected_metric_id is not None and raw_score_metric.get("id") != expected_metric_id:
        raise PairedVerifyError("task score metric differs from producer selection")
    if (value.get("metric_id") != raw_score_metric.get("id")
            or value.get("inner") != raw_score_metric.get("inner")):
        raise PairedVerifyError("task metric_id/inner differs from score_metric")
    if {raw_score_metric.get("id"), raw_secondary_metric.get("id")} != set(SCORE_METRIC_IDS):
        raise PairedVerifyError("score and secondary metrics must be the two distinct v2 metrics")
    rows = value.get("routes")
    if not isinstance(rows, list):
        raise PairedVerifyError("task receipt routes must be an array")
    parsed: dict[str, dict[str, Any]] = {}
    common_protocol = common_protocol_object = common_lock = None
    common_blocks: list[int] | None = None
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("route"), str):
            raise PairedVerifyError(f"task routes[{index}] has no route")
        route = row["route"]
        if route in parsed:
            raise PairedVerifyError(f"task receipt repeats route {route!r}")
        receipt, protocol_hash = validate_measurement_receipt(
            row.get("measurement_receipt"), f"route {route!r}",
            required_lock_path=required_lock_path,
            expected_score_metric=raw_score_metric)
        protocol = receipt["protocol"]
        score_metric = _validate_score_metric(raw_score_metric, protocol, "task score_metric")
        secondary_metric = _validate_score_metric(
            raw_secondary_metric, protocol, "task secondary_metric")
        secondary, secondary_protocol_hash = validate_measurement_receipt(
            row.get("secondary_measurement_receipt"), f"route {route!r} secondary",
            required_lock_path=required_lock_path,
            expected_score_metric=secondary_metric)
        if (secondary_protocol_hash != protocol_hash
                or canonical_json(secondary["protocol"]) != canonical_json(protocol)
                or secondary.get("arms") != receipt.get("arms")
                or canonical_json(secondary.get("slots")) != canonical_json(receipt.get("slots"))):
            raise PairedVerifyError(
                f"route {route!r} secondary report does not use the same protocol/arms/slots")
        if row.get("score_metric") != score_metric or row.get("secondary_metric") != secondary_metric:
            raise PairedVerifyError(f"route {route!r} metric binding differs from suite")
        candidate_ms = receipt["medians_ms"]["candidate"]
        incumbent_ms = receipt["medians_ms"]["incumbent"]
        if (row.get("execution_time_ms") != candidate_ms
                or row.get("incumbent_ms") != incumbent_ms
                or row.get("speedup") != incumbent_ms / candidate_ms):
            raise PairedVerifyError(
                f"route {route!r} execution_time/speedup is not sourced from score_metric")
        protocol_object = canonical_json(protocol)
        lock_object = canonical_json(receipt["gpu_lock"])
        block_ids = [block["block"] for block in receipt["blocks"]]
        if common_protocol is None:
            common_protocol, common_protocol_object = protocol_hash, protocol_object
            common_lock, common_blocks = lock_object, block_ids
        elif (protocol_hash != common_protocol or protocol_object != common_protocol_object
              or lock_object != common_lock or block_ids != common_blocks):
            raise PairedVerifyError(
                f"route {route!r} differs in protocol, GPU lock, or outer block inventory")
        parsed[route] = receipt
    if set(parsed) != set(expected):
        raise PairedVerifyError(
            "task receipt is not the complete suite: "
            f"missing={sorted(set(expected) - set(parsed))}, "
            f"extra={sorted(set(parsed) - set(expected))}")
    if value.get("protocol_hash") != common_protocol:
        raise PairedVerifyError("task protocol_hash disagrees with route receipts")
    return dict(value), parsed


def seal_task_receipt(
    raw_suite: object, *, candidate_root: Path, incumbent_root: Path,
    candidate_hashes_before: Mapping[str, object], candidate_hashes_after: Mapping[str, object],
    incumbent_hashes_before: Mapping[str, object], incumbent_hashes_after: Mapping[str, object],
    expected_routes: Sequence[str], lock_path: str, command: Sequence[str],
    expected_metric_id: str | None = None,
) -> dict[str, Any]:
    raw, routes = validate_raw_suite(
        raw_suite, expected_routes, required_lock_path=lock_path,
        expected_metric_id=expected_metric_id)
    if candidate_hashes_before != candidate_hashes_after:
        raise PairedVerifyError("task command changed candidate source/assembly/binary identity")
    if incumbent_hashes_before != incumbent_hashes_after:
        raise PairedVerifyError("task command modified the incumbent")
    first = routes[sorted(routes)[0]]
    raw_rows = {row["route"]: row for row in raw["routes"]}
    score_metric = _validate_score_metric(raw["score_metric"], first["protocol"], "task score_metric")
    secondary_metric = _validate_score_metric(
        raw["secondary_metric"], first["protocol"], "task secondary_metric")
    payload: dict[str, Any] = {
        "schema": TASK_RECEIPT_SCHEMA,
        "source_schema": RAW_SUITE_SCHEMA,
        "promotion_control": "incumbent",
        "oracle_role": "absolute_report_only",
        "task_entrypoint": raw["task_entrypoint"],
        "candidate_dispatch": dict(raw["candidate_dispatch"]),
        "incumbent_dispatch": dict(raw["incumbent_dispatch"]),
        "expected_routes": sorted(expected_routes),
        "protocol": first["protocol"],
        "protocol_hash": first["protocol_hash"],
        "score_metric": score_metric,
        "metric_id": score_metric["id"],
        "inner": score_metric["inner"],
        "secondary_metric": secondary_metric,
        "gpu_lock": first["gpu_lock"],
        "device_visibility": {key: os.environ.get(key) for key in (
            "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")},
        "candidate_root": str(candidate_root.resolve()),
        "incumbent_root": str(incumbent_root.resolve()),
        "candidate_hashes": dict(candidate_hashes_after),
        "incumbent_hashes": dict(incumbent_hashes_after),
        "incumbent_unchanged": True,
        "command_argv": list(command),
        "raw_suite_sha256": canonical_sha256(raw),
        "routes": [{
            "route": route,
            "score_metric": score_metric,
            "measurement_receipt_sha256": canonical_sha256(routes[route]),
            "measurement_receipt": routes[route],
            "secondary_metric": secondary_metric,
            "secondary_measurement_receipt_sha256": canonical_sha256(
                raw_rows[route]["secondary_measurement_receipt"]),
            "secondary_measurement_receipt": raw_rows[route]["secondary_measurement_receipt"],
            "execution_time_ms": raw_rows[route]["execution_time_ms"],
            "incumbent_ms": raw_rows[route]["incumbent_ms"],
            "speedup": raw_rows[route]["speedup"],
            "absolute_report": raw_rows[route].get("absolute_report"),
        } for route in sorted(routes)],
    }
    payload["canonical_input_sha256"] = canonical_sha256(payload)
    return payload


def _requested_score_metric(metric_id: str | None) -> str:
    explicit = str(metric_id).strip() if metric_id is not None else ""
    ambient = os.environ.get("GEAK_SCORE_METRIC")
    ambient = ambient.strip() if ambient is not None else ""
    if explicit and ambient and explicit != ambient:
        raise PairedVerifyError(
            f"score metric conflict: CLI {explicit!r} != GEAK_SCORE_METRIC {ambient!r}")
    selected = explicit or ambient or "single_call_latency"
    if selected not in SCORE_METRIC_IDS:
        raise PairedVerifyError(
            f"unsupported score metric {selected!r}; expected one of {list(SCORE_METRIC_IDS)}")
    return selected


def produce(
    *, candidate_root: Path, incumbent_root: Path, task_root: Path,
    expected_routes: Sequence[str], command: Sequence[str], output: Path,
    score_metric_id: str | None = None,
) -> dict[str, Any]:
    if os.environ.get("GEAK_GPU_LOCK_HELD") != "1":
        raise PairedVerifyError("produce must run inside kernel_workflow/scripts/gpu_lock.sh")
    lock_path = os.environ.get("GEAK_GPU_LOCK_FILE", "")
    if not lock_path:
        raise PairedVerifyError("GPU lock receipt has no GEAK_GPU_LOCK_FILE")
    selected_metric_id = _requested_score_metric(score_metric_id)
    candidate_root, incumbent_root, task_root = (
        candidate_root.resolve(), incumbent_root.resolve(), task_root.resolve())
    if not candidate_root.is_dir() or not incumbent_root.is_dir() or not task_root.is_dir():
        raise PairedVerifyError("candidate-root, incumbent-root, and task-root must be directories")
    if len({candidate_root, incumbent_root, task_root}) != 3:
        raise PairedVerifyError("candidate, incumbent, and immutable task roots must be distinct")
    expected = sorted(_names(list(expected_routes), "expected_routes"))
    command = [str(value) for value in command]
    if len(command) < 2 or not Path(command[0]).name.startswith("python"):
        raise PairedVerifyError("task command must invoke an immutable Python task adapter")
    interpreter_name = shutil.which(command[0]) if not Path(command[0]).is_absolute() else command[0]
    if not interpreter_name or not Path(interpreter_name).is_file():
        raise PairedVerifyError("task Python interpreter is not an executable file")
    interpreter = Path(interpreter_name).resolve()
    for mutable_root in (candidate_root, incumbent_root, task_root):
        try:
            interpreter.relative_to(mutable_root)
        except ValueError:
            continue
        raise PairedVerifyError("task Python interpreter must be outside measured/task roots")
    command[0] = str(interpreter)
    adapter = Path(command[1])
    if not adapter.is_absolute() or not adapter.is_file():
        raise PairedVerifyError("task adapter path must be an absolute existing file")
    try:
        adapter.resolve().relative_to(task_root)
    except ValueError as exc:
        raise PairedVerifyError("task adapter must live under immutable task-root") from exc
    candidate_before, incumbent_before = canonical_hashes(candidate_root), canonical_hashes(incumbent_root)
    task_before = canonical_hashes(task_root)
    raw_path = output.with_name(f".{output.name}.raw.{os.getpid()}.{uuid.uuid4().hex}.json")
    env = dict(os.environ)
    env.update({
        "GEAK_PAIRED_CANDIDATE_ROOT": str(candidate_root),
        "GEAK_PAIRED_INCUMBENT_ROOT": str(incumbent_root),
        "GEAK_PAIRED_EXPECTED_ROUTES": canonical_json(expected),
        "GEAK_PAIRED_RAW_RECEIPT": str(raw_path),
        "GEAK_SCORE_METRIC": selected_metric_id,
    })
    try:
        proc = subprocess.run(command, cwd=candidate_root, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            raise PairedVerifyError(
                f"task paired command exited {proc.returncode}: "
                f"{proc.stderr.strip() or proc.stdout.strip()}")
        raw = _load_json(raw_path)
        task_after = canonical_hashes(task_root)
        if task_before != task_after:
            raise PairedVerifyError("task command modified the immutable task adapter root")
        receipt = seal_task_receipt(
            raw, candidate_root=candidate_root, incumbent_root=incumbent_root,
            candidate_hashes_before=candidate_before,
            candidate_hashes_after=canonical_hashes(candidate_root),
            incumbent_hashes_before=incumbent_before,
            incumbent_hashes_after=canonical_hashes(incumbent_root),
            expected_routes=expected, lock_path=lock_path, command=command,
            expected_metric_id=selected_metric_id)
        receipt["task_root"] = str(task_root)
        receipt["task_hashes"] = task_after
        # Re-seal after adding immutable producer identity.
        receipt.pop("canonical_input_sha256", None)
        receipt["canonical_input_sha256"] = canonical_sha256(receipt)
        _atomic_json(output, receipt)
        return receipt
    finally:
        try:
            raw_path.unlink()
        except FileNotFoundError:
            pass


def validate_task_receipt(
    value: object, expected_routes: Sequence[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not isinstance(value, Mapping) or value.get("schema") != TASK_RECEIPT_SCHEMA:
        raise PairedVerifyError(f"paired task receipt must be {TASK_RECEIPT_SCHEMA}")
    task = dict(value)
    recorded = task.get("canonical_input_sha256")
    unhashed = dict(task)
    unhashed.pop("canonical_input_sha256", None)
    if not isinstance(recorded, str) or recorded != canonical_sha256(unhashed):
        raise PairedVerifyError("paired task canonical input SHA-256 does not match its contents")
    expected = sorted(_names(list(expected_routes), "expected_routes"))
    if sorted(_names(task.get("expected_routes"), "task expected_routes")) != expected:
        raise PairedVerifyError("paired task receipt does not cover the requested complete suite")
    if task.get("promotion_control") != "incumbent":
        raise PairedVerifyError("paired task promotion control is not the incumbent")
    if task.get("oracle_role") != "absolute_report_only":
        raise PairedVerifyError("paired task oracle is not report-only")
    if not isinstance(task.get("task_entrypoint"), str) or not task["task_entrypoint"]:
        raise PairedVerifyError("paired task formal entrypoint is missing")
    for arm in ("candidate", "incumbent"):
        dispatch = task.get(f"{arm}_dispatch")
        if (not isinstance(dispatch, Mapping)
                or dispatch.get("execution_path") != "gemm_wrapper.runtime_table"
                or not isinstance(dispatch.get("table_digest"), str)
                or SHA256_RE.fullmatch(dispatch["table_digest"]) is None):
            raise PairedVerifyError(f"paired task {arm} dispatch binding is invalid")
    if task.get("incumbent_unchanged") is not True:
        raise PairedVerifyError("paired task receipt does not prove an unmodified incumbent")
    for field in ("candidate_hashes", "incumbent_hashes", "task_hashes"):
        _validate_hashes(task.get(field), f"paired task {field}")
    task_root = task.get("task_root")
    if not isinstance(task_root, str) or not Path(task_root).is_absolute():
        raise PairedVerifyError("paired task immutable task_root is missing")
    protocol = task.get("protocol")
    if (not isinstance(protocol, Mapping) or _protocol_hash(protocol) != task.get("protocol_hash")
            or protocol.get("hash") != task.get("protocol_hash")):
        raise PairedVerifyError("paired task protocol hash is not canonical")
    score_metric = _validate_score_metric(task.get("score_metric"), protocol, "paired task score_metric")
    secondary_metric = _validate_score_metric(
        task.get("secondary_metric"), protocol, "paired task secondary_metric")
    if ({score_metric["id"], secondary_metric["id"]} != set(SCORE_METRIC_IDS)
            or task.get("metric_id") != score_metric["id"]
            or task.get("inner") != score_metric["inner"]):
        raise PairedVerifyError("paired task metric_id/inner binding is inconsistent")
    lock = task.get("gpu_lock")
    if (not isinstance(lock, Mapping) or lock.get("mode") != "external"
            or not isinstance(lock.get("path"), str) or not lock.get("path")):
        raise PairedVerifyError("paired task has no common external GPU lock")
    rows = task.get("routes")
    if not isinstance(rows, list):
        raise PairedVerifyError("paired task routes must be an array")
    parsed: dict[str, dict[str, Any]] = {}
    common_blocks: list[int] | None = None
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("route"), str):
            raise PairedVerifyError(f"paired task routes[{index}] is malformed")
        route = row["route"]
        if route in parsed:
            raise PairedVerifyError(f"paired task repeats route {route!r}")
        receipt, protocol_hash = validate_measurement_receipt(
            row.get("measurement_receipt"), f"route {route!r}",
            required_lock_path=lock["path"], expected_score_metric=score_metric)
        if protocol_hash != task.get("protocol_hash"):
            raise PairedVerifyError(f"route {route!r} uses a different protocol")
        if row.get("measurement_receipt_sha256") != canonical_sha256(receipt):
            raise PairedVerifyError(f"route {route!r} measurement receipt SHA-256 is stale")
        secondary, secondary_hash = validate_measurement_receipt(
            row.get("secondary_measurement_receipt"), f"route {route!r} secondary",
            required_lock_path=lock["path"], expected_score_metric=secondary_metric)
        if (secondary_hash != protocol_hash
                or row.get("secondary_measurement_receipt_sha256") != canonical_sha256(secondary)
                or secondary.get("arms") != receipt.get("arms")
                or canonical_json(secondary.get("slots")) != canonical_json(receipt.get("slots"))):
            raise PairedVerifyError(f"route {route!r} secondary receipt binding is stale")
        if row.get("score_metric") != score_metric or row.get("secondary_metric") != secondary_metric:
            raise PairedVerifyError(f"route {route!r} score metric binding differs")
        candidate_ms = receipt["medians_ms"]["candidate"]
        incumbent_ms = receipt["medians_ms"]["incumbent"]
        if (row.get("execution_time_ms") != candidate_ms
                or row.get("incumbent_ms") != incumbent_ms
                or row.get("speedup") != incumbent_ms / candidate_ms):
            raise PairedVerifyError(f"route {route!r} score fields mix metrics")
        block_ids = [block["block"] for block in receipt["blocks"]]
        if common_blocks is None:
            common_blocks = block_ids
        elif block_ids != common_blocks:
            raise PairedVerifyError(f"route {route!r} has a different outer block inventory")
        parsed[route] = receipt
    if set(parsed) != set(expected):
        raise PairedVerifyError(
            f"paired task is missing complete-suite routes: "
            f"missing={sorted(set(expected) - set(parsed))}, "
            f"extra={sorted(set(parsed) - set(expected))}")
    return task, parsed


def adapt(
    task_receipt: object, *, session_id: str, expected_routes: Sequence[str],
    target_routes: Sequence[str], gate_config: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    if not isinstance(session_id, str) or not session_id.strip() or session_id != session_id.strip():
        raise PairedVerifyError("lane session id is blank or non-canonical")
    expected = sorted(_names(list(expected_routes), "expected_routes"))
    targets = sorted(_names(list(target_routes), "target_routes"))
    unknown = sorted(set(targets) - set(expected))
    if unknown:
        raise PairedVerifyError(f"target_routes are outside the suite: {unknown}")
    task, routes = validate_task_receipt(task_receipt, expected)
    block_ids = [block["block"] for block in routes[expected[0]]["blocks"]]
    pair_ids = [f"{session_id}:block-{block:04d}" for block in block_ids]
    paired_routes, lifecycle, receipt_hashes = [], {}, {}
    for route in expected:
        receipt = routes[route]
        candidate = [float(block["samples"]["candidate"]["ms"]) for block in receipt["blocks"]]
        incumbent = [float(block["samples"]["incumbent"]["ms"]) for block in receipt["blocks"]]
        paired_routes.append({
            "route": route,
            "metric_id": task["metric_id"],
            "inner": task["inner"],
            "candidate": {"session_id": session_id, "pair_ids": list(pair_ids),
                          "samples_ms": candidate},
            "control": {"session_id": session_id, "pair_ids": list(pair_ids),
                        "samples_ms": incumbent},
        })
        lifecycle[route] = receipt["output_lifecycle"]
        receipt_hashes[route] = canonical_sha256(receipt)
    measurement = {
        "schema": BINDING_SCHEMA,
        "producer_schema": TASK_RECEIPT_SCHEMA,
        "promotion_control": "incumbent",
        "oracle_role": "absolute_report_only",
        "task_entrypoint": task["task_entrypoint"],
        "candidate_dispatch": task["candidate_dispatch"],
        "incumbent_dispatch": task["incumbent_dispatch"],
        "canonical_input_sha256": task["canonical_input_sha256"],
        "protocol": task["protocol"],
        "protocol_hash": task["protocol_hash"],
        "score_metric": task["score_metric"],
        "metric_id": task["metric_id"],
        "inner": task["inner"],
        "secondary_metric": task["secondary_metric"],
        "gpu_lock": task["gpu_lock"],
        "device_visibility": task.get("device_visibility"),
        "priming": {"required": True, "all_samples_primed": True,
                    "warmup_cycles": task["protocol"].get("warmup_cycles"),
                    "timer": "cuda_event"},
        "invariants": {
            "all_arms_in_every_block": True,
            "same_slot_within_block": True,
            "same_outer_blocks_across_suite": True,
            "selected_metric_inner_matches": True,
            ("single_call_latency_inner_is_one"
             if task["metric_id"] == "single_call_latency"
             else "sustained_throughput_inner_gt_one"): True,
            "semantic_slot_policy_bound": True,
        },
        "output_lifecycle": {"policy": "bounded-live-ring", "all_arms_rotated": True,
                             "per_route": lifecycle},
        "candidate_hashes": task.get("candidate_hashes"),
        "incumbent_hashes": task.get("incumbent_hashes"),
        "task_root": task.get("task_root"),
        "task_hashes": task.get("task_hashes"),
        "incumbent_unchanged": True,
        "route_measurement_sha256": receipt_hashes,
        "secondary_route_measurement_sha256": {
            row["route"]: row["secondary_measurement_receipt_sha256"]
            for row in task["routes"]
        },
    }
    spec: dict[str, Any] = {
        "schema": GATE_INPUT_SCHEMA,
        "session_id": session_id,
        "expected_routes": expected,
        "expected_route_count": len(expected),
        "target_routes": targets,
        "paired_routes": paired_routes,
        "measurement": measurement,
    }
    config = dict(gate_config or {})
    extra = sorted(set(config) - ALLOWED_GATE_CONFIG)
    if extra:
        raise PairedVerifyError(f"unsupported gate config key(s): {extra}")
    spec.update(config)
    return spec


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command_name", required=True)
    p = sub.add_parser("produce", help="run and seal one locked paired suite")
    p.add_argument("--candidate-root", required=True)
    p.add_argument("--incumbent-root", required=True)
    p.add_argument("--task-root", required=True)
    p.add_argument("--expected-route", action="append", default=[])
    p.add_argument("--score-metric", choices=SCORE_METRIC_IDS)
    p.add_argument("--output", required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = sub.add_parser("adapt", help="emit route-gate-input/v2 from a task receipt")
    a.add_argument("--receipt", required=True)
    a.add_argument("--session-id", required=True)
    a.add_argument("--expected-route", action="append", default=[])
    a.add_argument("--target-route", action="append", default=[])
    a.add_argument("--gate-config-json", default="{}")
    a.add_argument("--output", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command_name == "produce":
            command = list(args.command)
            if command and command[0] == "--":
                command.pop(0)
            result = produce(
                candidate_root=Path(args.candidate_root), incumbent_root=Path(args.incumbent_root),
                task_root=Path(args.task_root), expected_routes=args.expected_route,
                command=command, output=Path(args.output),
                score_metric_id=args.score_metric)
        else:
            config = json.loads(args.gate_config_json)
            if not isinstance(config, dict):
                raise PairedVerifyError("gate-config-json must decode to an object")
            result = adapt(
                _load_json(args.receipt), session_id=args.session_id,
                expected_routes=args.expected_route, target_routes=args.target_route,
                gate_config=config)
            if args.output:
                _atomic_json(Path(args.output), result)
        sys.stdout.write(canonical_json(result) + "\n")
        return 0
    except (PairedVerifyError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        sys.stderr.write(f"paired_verify: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
