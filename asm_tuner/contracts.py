from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

from .adapters import get_adapter
from .schema import KernelConfig, Route


TASK_CONTRACT_SCHEMA = "geak.asm-task-contract/v1"
ROUTE_MANIFEST_SCHEMA = "geak.asm-route-manifest/v1"
SEARCH_SPACE_SCHEMA = "geak.asm-search-space/v1"
ORACLE_MANIFEST_SCHEMA = "geak.oracle-route-manifest/v2"


class ContractError(RuntimeError):
    """A canonical task no longer matches its checked-in tuner contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ContractError("cannot hash %s: %s" % (path, exc)) from exc
    return digest.hexdigest()


def read_object(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("cannot read JSON %s: %s" % (path, exc)) from exc
    if not isinstance(value, dict):
        raise ContractError("JSON root must be an object: %s" % path)
    return value


def _assignment(path: Path, name: str) -> Any:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise ContractError("cannot parse %s: %s" % (path, exc)) from exc
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            try:
                return ast.literal_eval(node.value)
            except (TypeError, ValueError) as exc:
                raise ContractError("%s:%s must be a literal" % (path, name)) from exc
    raise ContractError("%s does not define %s" % (path, name))


def _function_sha256(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            segment = ast.get_source_segment(text, node)
            if segment is None:
                lines = text.splitlines(keepends=True)
                segment = "".join(lines[node.lineno - 1:node.end_lineno])
            return sha256_bytes(segment.encode("utf-8"))
    raise ContractError("%s does not define function %s" % (path, name))


@contextlib.contextmanager
def _without_protocol_environment() -> Iterator[None]:
    names = (
        "GEAK_MEASUREMENT_PROTOCOL", "GEAK_MEASUREMENT_SEED", "GEAK_WARMUP_CYCLES",
        "GEAK_ROTATION_TARGET_MB", "GEAK_ASSUMED_CACHE_MB", "GEAK_MAX_WEIGHT_SLOTS",
        "GEAK_OUTPUT_RING_SIZE", "HARNESS_TIMING_PRIME_MB", "GEAK_SYNTHETIC_EVICT_MB",
        "HARNESS_CACHE_FLUSH_MB", "GEAK_LATENCY_CYCLES", "GEAK_THROUGHPUT_CYCLES",
        "GEAK_THROUGHPUT_INNER",
    )
    old = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _load_module(path: Path, prefix: str) -> Any:
    name = "%s_%s" % (prefix, sha256_bytes(str(path.resolve()).encode())[:12])
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ContractError("cannot load module from %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ContractError("cannot import %s: %s" % (path, exc)) from exc
    return module


@dataclass(frozen=True)
class TaskContract:
    manifest_path: Path
    repo_root: Path
    task_root: Path
    value: Mapping[str, Any]
    route_manifest: Mapping[str, Any]
    search_space: Mapping[str, Any]
    cases: Tuple[Mapping[str, Any], ...]
    task_digest: str
    semantic_hashes: Mapping[str, str]
    binding_hashes: Mapping[str, str]
    protocol: Mapping[str, Any]
    oracle: Mapping[str, Any]
    oracle_sha256: str
    oracle_compatible: bool

    @property
    def task(self) -> str:
        return str(self.value["task"])

    @property
    def adapter_name(self) -> str:
        return str(self.value["adapter"])

    @property
    def adapter(self) -> Any:
        return get_adapter(self.adapter_name)

    @property
    def routes(self) -> Tuple[Route, ...]:
        return tuple(Route(
            str(case["id"]), int(case["M"]), int(case["N"]), int(case["K"]),
            float(case.get("weight", 1.0)),
        ) for case in self.cases)

    @property
    def default_config(self) -> KernelConfig:
        variant = str(self.search_space.get("default_variant") or "seed")
        return self.config_variant(variant)

    def config_variant(self, name: str) -> KernelConfig:
        variants = self.search_space.get("variants")
        if not isinstance(variants, dict) or name not in variants:
            raise ContractError("search space has no variant %r" % name)
        config = KernelConfig.from_dict(variants[name])
        self.adapter.validate(config)
        return config

    def config_by_id(self, config_id: str) -> KernelConfig:
        for name in sorted((self.search_space.get("variants") or {})):
            config = self.config_variant(name)
            if config.config_id == config_id:
                return config
        raise ContractError("config %s is absent from the declared search space" % config_id)

    def case(self, case_id: str) -> Mapping[str, Any]:
        for case in self.cases:
            if case["id"] == case_id:
                return case
        raise ContractError("task %s has no case %s" % (self.task, case_id))

    def identity(self) -> Dict[str, Any]:
        return {
            "schema": TASK_CONTRACT_SCHEMA,
            "task": self.task,
            "adapter": self.adapter_name,
            "task_digest": self.task_digest,
            "semantic_hashes": dict(self.semantic_hashes),
            "binding_hashes": dict(self.binding_hashes),
            "case_ids": [case["id"] for case in self.cases],
            "protocol_hash": self.protocol["hash"],
            "oracle_manifest_sha256": self.oracle_sha256,
            "oracle_protocol_hash": self.oracle.get("protocol_hash"),
            "oracle_compatible": self.oracle_compatible,
        }

    @classmethod
    def load(
        cls,
        manifest_path: Path,
        *,
        task_root: Optional[Path] = None,
        require_oracle_compatible: bool = False,
    ) -> "TaskContract":
        manifest_path = Path(manifest_path).resolve()
        value = read_object(manifest_path)
        if value.get("schema") != TASK_CONTRACT_SCHEMA:
            raise ContractError("unsupported task contract schema %r" % value.get("schema"))
        # <repo>/asm_tuner/manifests/tasks/<name>.json
        try:
            repo_root = manifest_path.parents[3]
        except IndexError as exc:
            raise ContractError("task manifest is not under asm_tuner/manifests/tasks") from exc
        root = Path(task_root).resolve() if task_root is not None else (
            repo_root / str(value["task_root"])
        ).resolve()
        if not root.is_dir():
            raise ContractError("canonical task root is missing: %s" % root)

        route_path = (repo_root / str(value["route_manifest"])).resolve()
        route_manifest = read_object(route_path)
        if route_manifest.get("schema") != ROUTE_MANIFEST_SCHEMA:
            raise ContractError("unsupported route manifest schema")
        if route_manifest.get("task") != value.get("task"):
            raise ContractError("route manifest task mismatch")

        search_path = (repo_root / str(value["search_space_manifest"])).resolve()
        search_space = read_object(search_path)
        if search_space.get("schema") != SEARCH_SPACE_SCHEMA:
            raise ContractError("unsupported search-space schema")
        if (search_space.get("task") != value.get("task") or
                search_space.get("adapter") != value.get("adapter")):
            raise ContractError("search-space task/adapter mismatch")

        bindings = value.get("bindings") or {}
        case_binding = bindings.get("cases") or {}
        case_path = root / str(case_binding.get("path") or "")
        cases_value = _assignment(case_path, str(case_binding.get("symbol") or "CASES"))
        if not isinstance(cases_value, list) or not cases_value:
            raise ContractError("CASES must be a non-empty literal list")
        cases = tuple(dict(case) for case in cases_value)
        case_ids = [str(case.get("id") or "") for case in cases]
        if not all(case_ids) or len(case_ids) != len(set(case_ids)):
            raise ContractError("CASES ids must be non-empty and unique")
        declared_routes = route_manifest.get("routes")
        if not isinstance(declared_routes, list) or not declared_routes:
            raise ContractError("route manifest routes must be a non-empty array")
        normalized_cases = [
            {key: case[key] for key in ("id", "regime", "M", "N", "K", "seed")}
            for case in cases
        ]
        normalized_routes = [
            {key: route[key] for key in ("id", "regime", "M", "N", "K", "seed")}
            for route in declared_routes
        ]
        if normalized_routes != normalized_cases:
            raise ContractError("route manifest drifted from task_runner.py:CASES")

        semantic_files = value.get("semantic_files")
        if not isinstance(semantic_files, list) or not semantic_files:
            raise ContractError("semantic_files must be a non-empty array")
        semantic_hashes: Dict[str, str] = {}
        for relative in semantic_files:
            path = root / str(relative)
            if not path.is_file():
                raise ContractError("semantic task file missing: %s" % path)
            semantic_hashes[str(relative)] = sha256_file(path)

        binding_hashes = {
            "cases": sha256_bytes(canonical_json(normalized_cases).encode("utf-8")),
        }
        for key in ("input_builder", "correctness"):
            binding = bindings.get(key) or {}
            path = root / str(binding.get("path") or "")
            symbol = str(binding.get("symbol") or "")
            binding_hashes[key] = _function_sha256(path, symbol)

        protocol_spec = value.get("protocol") or {}
        harness_path = root / str(protocol_spec.get("path") or "harness_lib.py")
        harness = _load_module(harness_path, "geak_contract_harness")
        function = getattr(harness, str(protocol_spec.get("symbol") or "measurement_protocol"), None)
        if not callable(function):
            raise ContractError("protocol builder is not callable")
        arguments = dict(protocol_spec.get("arguments") or {})
        with _without_protocol_environment():
            protocol = function(str(protocol_spec.get("name") or "semantic_weight_rotating"), **arguments)
        if protocol.get("schema") != "geak.measurement-protocol/v2" or protocol.get("version") != 2:
            raise ContractError("canonical task did not produce measurement protocol v2")
        expected_protocol = protocol_spec.get("hash")
        if expected_protocol and protocol.get("hash") != expected_protocol:
            raise ContractError("protocol hash drift: %s != %s" % (
                protocol.get("hash"), expected_protocol,
            ))

        oracle_path = root / str(value["oracle_manifest"])
        oracle = read_object(oracle_path)
        if oracle.get("schema") != ORACLE_MANIFEST_SCHEMA or oracle.get("task") != value.get("task"):
            raise ContractError("oracle manifest schema/task mismatch")
        oracle_routes = oracle.get("routes") or {}
        if set(oracle_routes) != set(case_ids):
            raise ContractError("oracle manifest does not cover exactly the canonical CASES")
        for case in cases:
            row = oracle_routes[case["id"]]
            if tuple(int(row[k]) for k in ("M", "N", "K")) != tuple(
                    int(case[k]) for k in ("M", "N", "K")):
                raise ContractError("oracle shape drift for %s" % case["id"])
        oracle_sha256 = sha256_file(oracle_path)
        oracle_compatible = bool(
            oracle.get("frozen") is True and
            oracle.get("selection_metric") == "single_call_latency" and
            oracle.get("protocol_hash") == protocol.get("hash")
        )
        if require_oracle_compatible and not oracle_compatible:
            raise ContractError(
                "oracle manifest is not frozen under protocol %s" % protocol.get("hash")
            )

        identity_payload = {
            "schema": TASK_CONTRACT_SCHEMA,
            "task": value["task"],
            "adapter": value["adapter"],
            "cases": normalized_cases,
            "semantic_hashes": semantic_hashes,
            "binding_hashes": binding_hashes,
            "route_manifest_sha256": sha256_file(route_path),
            "protocol_hash": protocol["hash"],
            "oracle_manifest_sha256": oracle_sha256,
        }
        task_digest = sha256_bytes(canonical_json(identity_payload).encode("utf-8"))
        contract = cls(
            manifest_path, repo_root, root, value, route_manifest, search_space, cases,
            task_digest, semantic_hashes, binding_hashes, protocol, oracle,
            oracle_sha256, oracle_compatible,
        )
        # Validate every declared point now. This is where inactive PGR/PLR/XCC
        # values fail closed instead of surviving until a long GPU run.
        for variant in sorted((search_space.get("variants") or {})):
            contract.config_variant(variant)
        return contract
