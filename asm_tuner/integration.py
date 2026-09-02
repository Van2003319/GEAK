from __future__ import annotations

import contextlib
import ctypes
import datetime as dt
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .compiler import ShardCompiler
from .contracts import TaskContract, canonical_json, read_object, sha256_file
from .runtime_table import RuntimeTableStore
from .schema import KernelConfig


RELEASE_SCHEMA = "geak.asm-release/v1"
SELECTION_SCHEMA = "geak.asm-selection/v1"
MATERIALIZATION_SCHEMA = "geak.asm-materialization/v1"
VERIFICATION_SCHEMA = "geak.asm-materialization-verification/v1"
SMOKE_SCHEMA = "geak.asm-canonical-smoke/v1"


class IntegrationError(RuntimeError):
    """A release cannot cross the canonical task execution boundary safely."""


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.tmp.%d.%s" % (path.name, os.getpid(), uuid.uuid4().hex))
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp), str(path))
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _release_identity(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("release_id", None)
    value.pop("created_at", None)
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _artifact_row(contract: TaskContract, result: Any, config: KernelConfig) -> Dict[str, Any]:
    manifest_path = result.directory / "manifest.json"
    source = result.artifact("source").resolve()
    artifact = result.artifact("shared").resolve()
    kernel = next((row for row in result.manifest["kernels"]
                   if row["config_id"] == config.config_id), None)
    if kernel is None:
        raise IntegrationError("compile manifest lacks config %s" % config.config_id)
    return {
        "config_id": config.config_id,
        "config": config.to_dict(include_derived=True),
        "kernel": kernel["kernel"],
        "symbol": contract.adapter.launch_name(config),
        "isa_hash": kernel["isa_hash"],
        "cache_key": result.cache_key,
        "compile_manifest": str(manifest_path.resolve()),
        "compile_manifest_sha256": sha256_file(manifest_path),
        "artifact": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "source": str(source),
        "source_sha256": sha256_file(source),
    }


def create_release(
    contract: TaskContract,
    *,
    cache_root: Path,
    variant: Optional[str] = None,
    config: Optional[KernelConfig] = None,
    hipcc: Optional[str] = None,
    optimization: str = "-O3",
    selection_reason: str = "declared correctness seed; not performance tuned",
) -> Dict[str, Any]:
    """Compile one declared config and make it a complete canonical-suite release."""
    if not contract.oracle_compatible:
        raise IntegrationError(
            "canonical oracle is not frozen under measurement protocol v2; refusing a lane release"
        )
    if config is None:
        config = contract.config_variant(variant or str(contract.search_space["default_variant"]))
    contract.adapter.validate(config)
    # A config outside the committed search-space is not reproducible by a verifier.
    contract.config_by_id(config.config_id)
    compiler = ShardCompiler(Path(cache_root), hipcc=hipcc, optimization=optimization)
    result = compiler.compile_shard(contract.adapter, [config], shared=True)
    artifact = _artifact_row(contract, result, config)
    route_selection = {route.name: config.config_id for route in contract.routes}
    payload: Dict[str, Any] = {
        "schema": RELEASE_SCHEMA,
        "task": contract.identity(),
        "protocol": dict(contract.protocol),
        "oracle": {
            "manifest_sha256": contract.oracle_sha256,
            "protocol_hash": contract.oracle.get("protocol_hash"),
            "compatible": contract.oracle_compatible,
            "selection_metric": contract.oracle.get("selection_metric"),
        },
        "fallback": {
            "kind": "canonical",
            "reason": "shape absent from the exact canonical CASES table",
        },
        "search_space_manifest": str(contract.value["search_space_manifest"]),
        "variant": variant or str(contract.search_space["default_variant"]),
        "selection_reason": selection_reason,
        "route_selection": route_selection,
        "artifacts": [artifact],
    }
    payload["release_id"] = _release_identity(payload)
    payload["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    target = Path(cache_root) / "releases" / payload["release_id"] / "release.json"
    if target.exists():
        existing = read_object(target)
        validate_release(existing, contract)
        return {"release": existing, "manifest": str(target), "cache_hit": True}
    _atomic_json(target, payload)
    validate_release(payload, contract)
    return {"release": payload, "manifest": str(target), "cache_hit": False}


class _CompiledCallable:
    def __init__(self, adapter: str, artifact: Mapping[str, Any]) -> None:
        self.adapter = adapter
        self.artifact = artifact
        self.library = ctypes.CDLL(str(artifact["artifact"]))
        self.launch = getattr(self.library, str(artifact["symbol"]))
        self.launch.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 6
        self.launch.restype = ctypes.c_int

    def __call__(self, *args: Any) -> Any:
        import torch
        a, b = args[0], args[1]
        if self.adapter == "bf16":
            a_scale = b_scale = None
        else:
            a_scale, b_scale = args[2], args[3]
        m, n, k = int(a.shape[0]), int(b.shape[0]), int(a.shape[1])
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
        ptr = lambda value: ctypes.c_void_p(0 if value is None else int(value.data_ptr()))
        stream = torch.cuda.current_stream(a.device)
        status = int(self.launch(
            m, n, k, ptr(a), ptr(b), ptr(a_scale), ptr(b_scale), ptr(out),
            ctypes.c_void_p(int(stream.cuda_stream)),
        ))
        if status:
            raise IntegrationError("HIP launch failed for %s with %d" % (
                self.artifact["config_id"], status,
            ))
        return out


def search_release(
    contract: TaskContract,
    *,
    cache_root: Path,
    output_dir: Path,
    variants: Sequence[str],
    case_ids: Optional[Sequence[str]] = None,
    hipcc: Optional[str] = None,
    optimization: str = "-O3",
) -> Dict[str, Any]:
    """Canonical multi-arm search under the task's exact protocol-v2 builders.

    The arms share each semantic-weight slot and rotate order inside one external
    GPU lock/fence.  A restricted case list is a smoke search; unmeasured routes
    retain the declared seed, so the resulting runtime table is still a complete
    atomic suite and says which routes were actually measured.
    """
    if os.environ.get("GEAK_GPU_LOCK_HELD") != "1" or os.environ.get("GEAK_GPU_FENCE_HELD") != "1":
        raise IntegrationError("canonical search requires gpu_lock.sh + gpu_fence_run.sh")
    if not contract.oracle_compatible:
        raise IntegrationError("canonical search requires a protocol-v2 oracle manifest")
    variants = tuple(dict.fromkeys(str(name) for name in variants))
    if len(variants) < 2:
        raise IntegrationError("canonical search requires at least two declared variants")
    configs = [contract.config_variant(name) for name in variants]
    compiler = ShardCompiler(Path(cache_root), hipcc=hipcc, optimization=optimization)
    results = [compiler.compile_shard(contract.adapter, [config], shared=True) for config in configs]
    artifacts = [_artifact_row(contract, result, config) for result, config in zip(results, configs)]
    by_id = {row["config_id"]: row for row in artifacts}
    calls = {row["config_id"]: _CompiledCallable(contract.adapter_name, row) for row in artifacts}
    wanted = set(case_ids or [case["id"] for case in contract.cases])
    unknown = wanted - {case["id"] for case in contract.cases}
    if unknown:
        raise IntegrationError("unknown canonical search routes: %s" % sorted(unknown))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    route_selection = {route.name: contract.default_config.config_id for route in contract.routes}
    measurements: Dict[str, Any] = {}
    with _task_import_path(contract.task_root):
        runner = _import_path(
            contract.task_root / "scripts" / "task_runner.py",
            "geak_search_runner_%s" % uuid.uuid4().hex,
        )
        harness = sys.modules.get("harness_lib")
        if harness is None:
            raise IntegrationError("canonical task runner did not load its harness")
        for case in contract.cases:
            case_id = str(case["id"])
            if case_id not in wanted:
                continue
            slots = runner._make_measurement_slots(dict(case), dict(contract.protocol))
            first_args = tuple(slots[0]["args"])
            if contract.adapter_name == "bf16":
                import torch
                reference = (first_args[0].float() @ first_args[1].float().t()).to(torch.bfloat16)
            else:
                reference = runner._dequant_reference(*first_args)
            correctness = {}
            for config_id, call in calls.items():
                output = call(*first_args)
                ok, error = harness.correct(
                    output, reference, float(contract.value.get("tolerance", 0.02)),
                )
                correctness[config_id] = {"passed": bool(ok), "max_relative_error": error}
                if not ok:
                    raise IntegrationError("%s failed %s correctness" % (config_id, case_id))
            arms = {
                config_id: (lambda slot, fn=call: fn(*slot["args"]))
                for config_id, call in calls.items()
            }
            receipt = harness.measure_paired_arms(
                arms, slots, protocol=dict(contract.protocol), metric="single_call_latency",
            )
            if receipt.get("protocol_hash") != contract.protocol["hash"]:
                raise IntegrationError("measurement receipt protocol hash drift")
            invariants = receipt.get("invariants") or {}
            if not all(invariants.values()):
                raise IntegrationError("protocol-v2 receipt invariant failed for %s" % case_id)
            if (receipt.get("gpu_lock") or {}).get("mode") != "external":
                raise IntegrationError("measurement did not inherit the lane GPU lock")
            winner = min(calls, key=lambda config_id: (
                float(receipt["medians_ms"][config_id]), config_id,
            ))
            route_selection[case_id] = winner
            receipt_path = output_dir / "receipts" / (case_id + ".json")
            _atomic_json(receipt_path, receipt)
            measurements[case_id] = {
                "winner": winner,
                "medians_ms": receipt["medians_ms"],
                "correctness": correctness,
                "receipt": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
            }
    payload: Dict[str, Any] = {
        "schema": RELEASE_SCHEMA,
        "task": contract.identity(),
        "protocol": dict(contract.protocol),
        "oracle": {
            "manifest_sha256": contract.oracle_sha256,
            "protocol_hash": contract.oracle.get("protocol_hash"),
            "compatible": contract.oracle_compatible,
            "selection_metric": contract.oracle.get("selection_metric"),
        },
        "fallback": {"kind": "canonical", "reason": "shape absent from canonical CASES"},
        "search_space_manifest": str(contract.value["search_space_manifest"]),
        "variant": "paired-search:" + ",".join(variants),
        "selection_reason": "protocol-v2 paired search; measured_routes=%s" % ",".join(sorted(wanted)),
        "measured_routes": sorted(wanted),
        "route_selection": route_selection,
        "artifacts": artifacts,
    }
    payload["release_id"] = _release_identity(payload)
    payload["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    target = Path(cache_root) / "releases" / payload["release_id"] / "release.json"
    _atomic_json(target, payload)
    validate_release(payload, contract)
    summary = {
        "schema": "geak.asm-canonical-search/v1",
        "ok": True,
        "task": contract.task,
        "release_id": payload["release_id"],
        "release_manifest": str(target),
        "task_digest": contract.task_digest,
        "protocol_hash": contract.protocol["hash"],
        "oracle_manifest_sha256": contract.oracle_sha256,
        "measured_routes": sorted(wanted),
        "route_selection": route_selection,
        "measurements": measurements,
    }
    _atomic_json(output_dir / "search_summary.json", summary)
    return {"release": payload, "manifest": str(target), "summary": summary}


def validate_release(value: Mapping[str, Any], contract: TaskContract) -> None:
    if value.get("schema") != RELEASE_SCHEMA:
        raise IntegrationError("unsupported release schema %r" % value.get("schema"))
    if value.get("release_id") != _release_identity(value):
        raise IntegrationError("release identity hash mismatch")
    task = value.get("task") or {}
    if (task.get("task") != contract.task or task.get("adapter") != contract.adapter_name or
            task.get("task_digest") != contract.task_digest):
        raise IntegrationError("release belongs to a different canonical task digest")
    protocol = value.get("protocol") or {}
    if (protocol.get("schema") != "geak.measurement-protocol/v2" or
            protocol.get("hash") != contract.protocol.get("hash")):
        raise IntegrationError("release protocol differs from canonical protocol v2")
    oracle = value.get("oracle") or {}
    if (oracle.get("compatible") is not True or
            oracle.get("manifest_sha256") != contract.oracle_sha256 or
            oracle.get("protocol_hash") != contract.protocol.get("hash")):
        raise IntegrationError("release oracle manifest is stale or mismatched")
    routes = value.get("route_selection")
    if not isinstance(routes, dict) or set(routes) != {route.name for route in contract.routes}:
        raise IntegrationError("release does not select the complete canonical suite")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise IntegrationError("release has no compiled artifacts")
    by_id = {}
    for row in artifacts:
        config = KernelConfig.from_dict(row.get("config") or {})
        contract.adapter.validate(config)
        contract.config_by_id(config.config_id)
        if config.config_id != row.get("config_id"):
            raise IntegrationError("artifact config id mismatch")
        for key, hash_key in (("artifact", "artifact_sha256"), ("source", "source_sha256"),
                              ("compile_manifest", "compile_manifest_sha256")):
            path = Path(str(row.get(key) or ""))
            if not path.is_file() or path.is_symlink():
                raise IntegrationError("release %s is missing or not regular: %s" % (key, path))
            if sha256_file(path) != row.get(hash_key):
                raise IntegrationError("release %s hash mismatch: %s" % (key, path))
        if row.get("symbol") != contract.adapter.launch_name(config):
            raise IntegrationError("release launch symbol does not match config")
        by_id[config.config_id] = row
    if set(routes.values()) - set(by_id):
        raise IntegrationError("a route selects a config without an artifact")


def load_release(path: Path, contract: TaskContract) -> Dict[str, Any]:
    value = read_object(Path(path))
    validate_release(value, contract)
    return value


def _install_release_files(
    value: Mapping[str, Any], workspace: Path,
) -> Tuple[Path, Mapping[str, Mapping[str, str]]]:
    bundle_root = workspace / ".geak_tuner"
    releases = bundle_root / "releases"
    target = releases / str(value["release_id"])
    installed: Dict[str, Mapping[str, str]] = {}
    if not target.exists():
        staging_root = bundle_root / "staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=str(value["release_id"])[:16] + ".", dir=str(staging_root)))
        try:
            (stage / "artifacts").mkdir()
            (stage / "sources").mkdir()
            (stage / "manifests").mkdir()
            for row in value["artifacts"]:
                config_id = str(row["config_id"])
                artifact_name = "%s.so" % row["artifact_sha256"]
                source_name = "%s.hip" % row["source_sha256"]
                manifest_name = "%s.json" % row["compile_manifest_sha256"]
                shutil.copyfile(row["artifact"], stage / "artifacts" / artifact_name)
                shutil.copyfile(row["source"], stage / "sources" / source_name)
                shutil.copyfile(row["compile_manifest"], stage / "manifests" / manifest_name)
                installed[config_id] = {
                    "artifact": str(Path(".geak_tuner") / "releases" / value["release_id"] / "artifacts" / artifact_name),
                    "source": str(Path(".geak_tuner") / "releases" / value["release_id"] / "sources" / source_name),
                    "compile_manifest": str(Path(".geak_tuner") / "releases" / value["release_id"] / "manifests" / manifest_name),
                }
            _atomic_json(stage / "release.json", dict(value))
            releases.mkdir(parents=True, exist_ok=True)
            os.replace(str(stage), str(target))
        except Exception:
            shutil.rmtree(str(stage), ignore_errors=True)
            raise
    else:
        for row in value["artifacts"]:
            config_id = str(row["config_id"])
            installed[config_id] = {
                "artifact": str(Path(".geak_tuner") / "releases" / value["release_id"] / "artifacts" / (row["artifact_sha256"] + ".so")),
                "source": str(Path(".geak_tuner") / "releases" / value["release_id"] / "sources" / (row["source_sha256"] + ".hip")),
                "compile_manifest": str(Path(".geak_tuner") / "releases" / value["release_id"] / "manifests" / (row["compile_manifest_sha256"] + ".json")),
            }
    return target, installed


def materialize_release(
    contract: TaskContract,
    value: Mapping[str, Any],
    *,
    workspace: Path,
    registry_id: str = "",
) -> Dict[str, Any]:
    workspace = Path(workspace).resolve()
    local_contract = TaskContract.load(
        contract.manifest_path, task_root=workspace, require_oracle_compatible=True,
    )
    if local_contract.task_digest != contract.task_digest:
        raise IntegrationError("workspace semantic task digest differs before materialization")
    validate_release(value, local_contract)
    target, installed = _install_release_files(value, workspace)
    by_id = {str(row["config_id"]): row for row in value["artifacts"]}
    for config_id, local_paths in installed.items():
        source_row = by_id[config_id]
        for key, hash_key in (("artifact", "artifact_sha256"),
                              ("source", "source_sha256"),
                              ("compile_manifest", "compile_manifest_sha256")):
            path = (workspace / local_paths[key]).resolve()
            if (not _within(path, workspace) or not path.is_file() or path.is_symlink()
                    or sha256_file(path) != source_row[hash_key]):
                raise IntegrationError("installed %s hash mismatch for %s" % (key, config_id))
    selections: Dict[str, List[Mapping[str, Any]]] = {}
    for route in local_contract.routes:
        config_id = str(value["route_selection"][route.name])
        source = by_id[config_id]
        local = installed[config_id]
        selections[route.name] = [{
            "config_id": config_id,
            "config": source["config"],
            "kernel": source["kernel"],
            "symbol": source["symbol"],
            "isa_hash": source["isa_hash"],
            "artifact": local["artifact"],
            "artifact_sha256": source["artifact_sha256"],
            "source": local["source"],
            "source_sha256": source["source_sha256"],
            "compile_manifest": local["compile_manifest"],
            "compile_manifest_sha256": source["compile_manifest_sha256"],
            "selection_reason": value.get("selection_reason"),
        }]
    release_context = {
        "release_id": value["release_id"],
        "task": {
            "name": local_contract.task,
            "adapter": local_contract.adapter_name,
            "digest": local_contract.task_digest,
            "semantic_hashes": dict(local_contract.semantic_hashes),
            "binding_hashes": dict(local_contract.binding_hashes),
        },
        "protocol": {
            "schema": local_contract.protocol["schema"],
            "version": local_contract.protocol["version"],
            "name": local_contract.protocol["name"],
            "hash": local_contract.protocol["hash"],
        },
        "oracle": value["oracle"],
        "fallback": value["fallback"],
    }
    table_path = workspace / ".geak_tuner" / "runtime_table.json"
    table = RuntimeTableStore(table_path).publish_suite(
        local_contract.routes, selections,
        run_id="materialize-%s" % value["release_id"][:16],
        receipts_root=target,
        release=release_context,
    )
    selection = {
        "schema": SELECTION_SCHEMA,
        "task": local_contract.task,
        "task_digest": local_contract.task_digest,
        "release_id": value["release_id"],
        "registry_id": registry_id,
        "protocol_hash": local_contract.protocol["hash"],
        "oracle_manifest_sha256": local_contract.oracle_sha256,
        "table_digest": table["table_digest"],
        "config_ids": sorted(set(value["route_selection"].values())),
    }
    _atomic_json(workspace / "asm_tuner_selection.json", selection)
    receipt = {
        "schema": MATERIALIZATION_SCHEMA,
        "ok": True,
        "workspace": str(workspace),
        "release_id": value["release_id"],
        "registry_id": registry_id,
        "runtime_table": str(table_path),
        "table_digest": table["table_digest"],
        "task_digest": local_contract.task_digest,
        "protocol_hash": local_contract.protocol["hash"],
        "oracle_manifest_sha256": local_contract.oracle_sha256,
        "route_count": len(local_contract.routes),
        "config_ids": selection["config_ids"],
        "artifacts": [dict(row) for row in installed.values()],
    }
    receipt_path = workspace / ".geak_tuner" / "materialization_receipt.json"
    _atomic_json(receipt_path, receipt)
    receipt["receipt"] = str(receipt_path)
    return receipt


def _load_library_symbols(workspace: Path, table: Mapping[str, Any]) -> List[Mapping[str, str]]:
    loaded: Dict[str, Any] = {}
    rows: List[Mapping[str, str]] = []
    for route_id in table["suite"]["route_ids"]:
        entry = table["routes"][route_id]["entries"][0]
        artifact = (workspace / entry["artifact"]).resolve()
        source = (workspace / entry["source"]).resolve()
        if not _within(artifact, workspace) or not _within(source, workspace):
            raise IntegrationError("runtime table path escapes materialized workspace")
        if not artifact.is_file() or artifact.is_symlink() or sha256_file(artifact) != entry["artifact_sha256"]:
            raise IntegrationError("materialized artifact hash mismatch for %s" % route_id)
        if not source.is_file() or source.is_symlink() or sha256_file(source) != entry["source_sha256"]:
            raise IntegrationError("materialized source hash mismatch for %s" % route_id)
        key = str(artifact)
        library = loaded.get(key)
        if library is None:
            try:
                library = ctypes.CDLL(key)
            except OSError as exc:
                raise IntegrationError("cannot load materialized artifact %s: %s" % (artifact, exc)) from exc
            loaded[key] = library
        if not hasattr(library, str(entry["symbol"])):
            raise IntegrationError("artifact lacks dispatch symbol %s" % entry["symbol"])
        rows.append({
            "route": route_id,
            "config_id": entry["config_id"],
            "symbol": entry["symbol"],
            "artifact_sha256": entry["artifact_sha256"],
            "source_sha256": entry["source_sha256"],
        })
    return rows


def _verify_workspace(
    contract: TaskContract,
    workspace: Path,
    *,
    rebuild: bool,
    cache_root: Optional[Path],
) -> Dict[str, Any]:
    local = TaskContract.load(contract.manifest_path, task_root=workspace, require_oracle_compatible=True)
    table = RuntimeTableStore(workspace / ".geak_tuner" / "runtime_table.json").load()
    if table["task"]["name"] != local.task or table["task"]["digest"] != local.task_digest:
        raise IntegrationError("runtime table task digest differs from independent workspace")
    if table["task"].get("semantic_hashes") != dict(local.semantic_hashes):
        raise IntegrationError("runtime table semantic file hashes differ")
    if table["task"].get("binding_hashes") != dict(local.binding_hashes):
        raise IntegrationError("runtime table CASES/input/correctness binding hashes differ")
    if table["protocol"]["hash"] != local.protocol["hash"]:
        raise IntegrationError("runtime table protocol hash differs")
    oracle = table.get("oracle") or {}
    if oracle.get("manifest_sha256") != local.oracle_sha256 or oracle.get("compatible") is not True:
        raise IntegrationError("runtime table oracle manifest differs")
    if set(table["suite"]["route_ids"]) != {route.name for route in local.routes}:
        raise IntegrationError("runtime table is not the complete canonical suite")
    symbol_rows = _load_library_symbols(workspace, table)
    rebuilt: List[Mapping[str, Any]] = []
    if rebuild:
        root = Path(cache_root) if cache_root is not None else Path(tempfile.mkdtemp(prefix="geak_verify_cache_"))
        compiler = ShardCompiler(root)
        seen = set()
        for route_id in table["suite"]["route_ids"]:
            entry = table["routes"][route_id]["entries"][0]
            if entry["config_id"] in seen:
                continue
            seen.add(entry["config_id"])
            config = KernelConfig.from_dict(entry["config"])
            result = compiler.compile_shard(local.adapter, [config], shared=True)
            rebuilt_source = result.artifact("source")
            rebuilt_artifact = result.artifact("shared")
            kernel = next(row for row in result.manifest["kernels"]
                          if row["config_id"] == config.config_id)
            if sha256_file(rebuilt_source) != entry["source_sha256"]:
                raise IntegrationError("independent rebuild source hash mismatch")
            if kernel["isa_hash"] != entry["isa_hash"]:
                raise IntegrationError("independent rebuild ISA hash mismatch")
            library = ctypes.CDLL(str(rebuilt_artifact))
            if not hasattr(library, entry["symbol"]):
                raise IntegrationError("independent rebuild lacks dispatch symbol")
            rebuilt.append({
                "config_id": config.config_id,
                "source_sha256": sha256_file(rebuilt_source),
                "artifact_sha256": sha256_file(rebuilt_artifact),
                "isa_hash": kernel["isa_hash"],
                "symbol": entry["symbol"],
            })
    return {
        "schema": VERIFICATION_SCHEMA,
        "ok": True,
        "workspace": str(workspace),
        "independent_workspace": False,
        "task": local.task,
        "task_digest": local.task_digest,
        "protocol_hash": local.protocol["hash"],
        "oracle_manifest_sha256": local.oracle_sha256,
        "table_digest": table["table_digest"],
        "release_id": table["release_id"],
        "routes_verified": len(symbol_rows),
        "symbols": symbol_rows,
        "rebuilt": rebuilt,
    }


def verify_materialization(
    contract: TaskContract,
    *,
    workspace: Path,
    rebuild: bool = False,
    cache_root: Optional[Path] = None,
    independent: bool = True,
) -> Dict[str, Any]:
    workspace = Path(workspace).resolve()
    if not independent:
        return _verify_workspace(contract, workspace, rebuild=rebuild, cache_root=cache_root)
    with tempfile.TemporaryDirectory(prefix="geak_verify_workspace_") as temp:
        clone = Path(temp) / "task"
        shutil.copytree(
            workspace, clone, symlinks=True,
            ignore=shutil.ignore_patterns("build", ".torch_ext", "__pycache__", "*.pyc", "*.o"),
        )
        result = _verify_workspace(contract, clone, rebuild=rebuild, cache_root=cache_root)
        result["independent_workspace"] = True
        result["source_workspace"] = str(workspace)
        result["workspace"] = str(clone)
        return result


@contextlib.contextmanager
def _task_import_path(task_root: Path) -> Iterator[None]:
    old_path = list(sys.path)
    displaced = {name: sys.modules.pop(name, None) for name in (
        "harness_lib", "gemm_wrapper", "kernel_loader", "geak_tuner_runtime",
    )}
    sys.path.insert(0, str(task_root))
    try:
        yield
    finally:
        sys.path[:] = old_path
        for name in displaced:
            sys.modules.pop(name, None)
        for name, module in displaced.items():
            if module is not None:
                sys.modules[name] = module


def _import_path(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise IntegrationError("cannot import %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def smoke_materialization(
    contract: TaskContract,
    *,
    workspace: Path,
    case_id: str = "",
) -> Dict[str, Any]:
    if os.environ.get("GEAK_GPU_LOCK_HELD") != "1":
        raise IntegrationError("canonical GPU smoke requires gpu_lock.sh (GEAK_GPU_LOCK_HELD=1)")
    workspace = Path(workspace).resolve()
    verification = verify_materialization(contract, workspace=workspace, independent=False)
    case_id = case_id or str(contract.cases[0]["id"])
    case = contract.case(case_id)
    trace = workspace / ".geak_tuner" / ("dispatch-%s.json" % case_id)
    old_required = os.environ.get("GEAK_TUNER_REQUIRED")
    old_trace = os.environ.get("GEAK_TUNER_TRACE")
    os.environ["GEAK_TUNER_REQUIRED"] = "1"
    os.environ["GEAK_TUNER_TRACE"] = str(trace)
    try:
        with _task_import_path(workspace):
            runner = _import_path(workspace / "scripts" / "task_runner.py",
                                  "geak_smoke_runner_%s" % uuid.uuid4().hex)
            wrapper = _import_path(workspace / "gemm_wrapper.py", "gemm_wrapper")
            inputs = runner._make_inputs(dict(case))
            entrypoint = str(contract.value["bindings"]["entrypoint"]["symbol"])
            output = getattr(wrapper, entrypoint)(*inputs)
            if contract.adapter_name == "bf16":
                reference = (inputs[0].float() @ inputs[1].float().t()).to(output.dtype)
            else:
                reference_binding = contract.value["bindings"].get("reference") or {}
                reference = getattr(runner, str(reference_binding.get("symbol") or "_dequant_reference"))(*inputs)
            harness = sys.modules.get("harness_lib")
            ok, max_relative_error = harness.correct(output, reference, float(contract.value.get("tolerance", 0.02)))
            import torch
            torch.cuda.synchronize()
        if not ok:
            raise IntegrationError("canonical task correctness failed: max_relative_error=%s" % max_relative_error)
        dispatch = read_object(trace)
        if dispatch.get("route") != case_id or not dispatch.get("symbol") or not dispatch.get("config_id"):
            raise IntegrationError("task dispatch did not consume the versioned runtime table")
        return {
            "schema": SMOKE_SCHEMA,
            "ok": True,
            "task": contract.task,
            "case": case_id,
            "shape": {"m": case["M"], "n": case["N"], "k": case["K"]},
            "task_digest": contract.task_digest,
            "protocol_hash": contract.protocol["hash"],
            "table_digest": verification["table_digest"],
            "config_id": dispatch["config_id"],
            "symbol": dispatch["symbol"],
            "artifact_sha256": dispatch["artifact_sha256"],
            "source_sha256": dispatch["source_sha256"],
            "max_relative_error": max_relative_error,
            "gpu_lock": {"mode": "external", "path": os.environ.get("GEAK_GPU_LOCK_FILE")},
            "dispatch_receipt": str(trace),
        }
    finally:
        if old_required is None:
            os.environ.pop("GEAK_TUNER_REQUIRED", None)
        else:
            os.environ["GEAK_TUNER_REQUIRED"] = old_required
        if old_trace is None:
            os.environ.pop("GEAK_TUNER_TRACE", None)
        else:
            os.environ["GEAK_TUNER_TRACE"] = old_trace
