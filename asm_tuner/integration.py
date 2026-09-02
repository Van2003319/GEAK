from __future__ import annotations

import contextlib
import ctypes
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .compiler import ShardCompiler
from .contracts import TaskContract, canonical_json, read_object, sha256_file
from .runtime_table import RuntimeTableStore
from .schema import KernelConfig
from .search_space import plan_route_candidates, policy_for


RELEASE_SCHEMA = "geak.asm-release/v3"
SELECTION_SCHEMA = "geak.asm-selection/v2"
MATERIALIZATION_SCHEMA = "geak.asm-materialization/v2"
VERIFICATION_SCHEMA = "geak.asm-materialization-verification/v1"
SMOKE_SCHEMA = "geak.asm-canonical-smoke/v1"
PAIR_GATE_SCHEMA = "geak.asm-candidate-vs-canonical-gate/v1"
PAIR_GATE_ALGORITHM = "paired-log-trimmed-bootstrap/v1"
CANONICAL_CONTROL_ARM = "__canonical_task_implementation__"
PAIR_GATE_CONFIDENCE = 0.95
PAIR_GATE_TRIM_FRACTION = 0.20
PAIR_GATE_RESAMPLES = 4096
PAIR_GATE_SEED = 20260902
PAIR_GATE_MIN_PAIRS = 8
PAIR_GATE_MDE = 0.02


class IntegrationError(RuntimeError):
    """A release cannot cross the canonical task execution boundary safely."""


class _SplitMix64:
    _MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = seed & self._MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self._MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & self._MASK
        return (value ^ (value >> 31)) & self._MASK

    def randbelow(self, size: int) -> int:
        return self.next_u64() % size


def _trimmed_mean(values: Sequence[float]) -> float:
    ordered = sorted(values)
    trim = int(math.floor(len(ordered) * PAIR_GATE_TRIM_FRACTION))
    kept = ordered[trim:len(ordered) - trim] if trim else ordered
    return sum(kept) / len(kept)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _paired_estimate(values: Sequence[float], route: str, candidate_id: str) -> Tuple[float, float]:
    point = _trimmed_mean(values)
    label = "%d:route:%s:candidate:%s" % (PAIR_GATE_SEED, route, candidate_id)
    seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    rng = _SplitMix64(seed)
    count = len(values)
    bootstrap = [
        _trimmed_mean([values[rng.randbelow(count)] for _ in range(count)])
        for _ in range(PAIR_GATE_RESAMPLES)
    ]
    return point, _quantile(bootstrap, 1.0 - PAIR_GATE_CONFIDENCE)


def _candidate_vs_canonical_gate(
    receipt: Mapping[str, Any],
    *,
    route: str,
    candidate_id: str,
    practical_mde: float = PAIR_GATE_MDE,
) -> Dict[str, Any]:
    if not (0.0 < practical_mde < 1.0):
        raise IntegrationError("candidate-vs-canonical practical MDE must be in (0, 1)")
    pairs: List[Dict[str, Any]] = []
    log_ratios: List[float] = []
    for block in receipt.get("blocks") or []:
        samples = block.get("samples") or {}
        candidate = samples.get(candidate_id) or {}
        control = samples.get(CANONICAL_CONTROL_ARM) or {}
        try:
            candidate_ms = float(candidate["ms"])
            control_ms = float(control["ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise IntegrationError(
                "paired receipt lacks candidate/canonical timing for %s" % candidate_id
            ) from exc
        if (not math.isfinite(candidate_ms) or not math.isfinite(control_ms)
                or candidate_ms <= 0.0 or control_ms <= 0.0
                or candidate.get("slot_fingerprint") != control.get("slot_fingerprint")):
            raise IntegrationError("candidate/canonical pair is not same-slot finite timing")
        pair_binding = {
            "route": route,
            "block": block.get("block"),
            "cycle": block.get("cycle"),
            "slot_id": block.get("slot_id"),
            "slot_fingerprint": block.get("slot_fingerprint"),
            "protocol_hash": receipt.get("protocol_hash"),
            "metric_id": receipt.get("metric_id"),
        }
        pair_id = hashlib.sha256(canonical_json(pair_binding).encode("utf-8")).hexdigest()
        pairs.append({
            "pair_id": pair_id,
            "block": block.get("block"),
            "slot_id": block.get("slot_id"),
            "slot_fingerprint": block.get("slot_fingerprint"),
            "candidate_ms": candidate_ms,
            "canonical_ms": control_ms,
        })
        log_ratios.append(math.log(control_ms / candidate_ms))
    if len(pairs) < PAIR_GATE_MIN_PAIRS:
        raise IntegrationError(
            "candidate-vs-canonical gate has %d pairs; need %d"
            % (len(pairs), PAIR_GATE_MIN_PAIRS)
        )
    point, lower = _paired_estimate(log_ratios, route, candidate_id)
    threshold = -math.log1p(-practical_mde)
    accepted = lower > threshold
    return {
        "schema": PAIR_GATE_SCHEMA,
        "comparison": "candidate_vs_canonical_task_implementation",
        "decision": "winner" if accepted else "no_winner",
        "accepted": accepted,
        "route": route,
        "candidate_id": candidate_id,
        "canonical_control_arm": CANONICAL_CONTROL_ARM,
        "protocol_hash": receipt.get("protocol_hash"),
        "metric_id": receipt.get("metric_id"),
        "metric_inner": (receipt.get("metric") or {}).get("inner"),
        "candidate_median_ms": float((receipt.get("medians_ms") or {})[candidate_id]),
        "canonical_median_ms": float(
            (receipt.get("medians_ms") or {})[CANONICAL_CONTROL_ARM]
        ),
        "n_pairs": len(pairs),
        "pair_ids": [pair["pair_id"] for pair in pairs],
        "pairs": pairs,
        "estimate_log_ratio": point,
        "estimate_speedup": math.exp(point),
        "ci95_lower_log_ratio": lower,
        "ci95_lower_speedup": math.exp(lower),
        "practical_mde": practical_mde,
        "threshold_log_ratio": threshold,
        "algorithm": {
            "name": PAIR_GATE_ALGORITHM,
            "effect": "log(canonical_ms/candidate_ms)",
            "estimator": "20% trimmed mean",
            "confidence": PAIR_GATE_CONFIDENCE,
            "interval": "one-sided paired percentile bootstrap lower bound",
            "resamples": PAIR_GATE_RESAMPLES,
            "seed": PAIR_GATE_SEED,
            "min_pairs": PAIR_GATE_MIN_PAIRS,
        },
        "reason": (
            "paired lower bound proves candidate improvement over canonical"
            if accepted else
            "paired lower bound does not prove MDE improvement over canonical"
        ),
    }


def _canonical_task_callable(contract: TaskContract) -> Callable[[Mapping[str, Any]], Any]:
    import kernel_loader

    entrypoint = str(contract.value["bindings"]["entrypoint"]["symbol"])
    operation = getattr(kernel_loader.gemm_ext, entrypoint)
    return lambda slot: operation(*slot["args"])


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
    """Create a complete canonical-fallback release without claiming a performance winner.

    Declared configs remain reproducible search seeds.  They cannot own a route until
    same-slot paired evidence proves an MDE improvement over the task implementation.
    """
    if not contract.oracle_compatible:
        raise IntegrationError(
            "canonical oracle is not frozen under measurement protocol v2; refusing a lane release"
        )
    if config is None:
        config = contract.config_variant(variant or str(contract.search_space["default_variant"]))
    contract.adapter.validate(config)
    contract.config_by_id(config.config_id)
    route_selection = {route.name: None for route in contract.routes}
    route_rankings = {route.name: [] for route in contract.routes}
    route_outcomes = {
        route.name: {
            "status": "unmeasured",
            "selection": "canonical",
            "reason": "declared seed has no candidate-vs-canonical paired evidence",
            "candidate_gates": [],
        }
        for route in contract.routes
    }
    payload: Dict[str, Any] = {
        "schema": RELEASE_SCHEMA,
        "task": contract.identity(),
        "protocol": dict(contract.protocol),
        "score_metric": dict(contract.score_metric),
        "oracle": {
            "manifest_sha256": contract.oracle_sha256,
            "protocol_hash": contract.oracle.get("protocol_hash"),
            "compatible": contract.oracle_compatible,
            "selection_metric": contract.oracle.get("selection_metric"),
            "score_metric": contract.oracle.get("score_metric"),
        },
        "fallback": {
            "kind": "canonical",
            "reason": "shape absent from the exact canonical CASES table",
        },
        "search_space_manifest": str(contract.value["search_space_manifest"]),
        "variant": variant or str(contract.search_space["default_variant"]),
        "selection_reason": selection_reason,
        "route_selection": route_selection,
        "route_rankings": route_rankings,
        "route_outcomes": route_outcomes,
        "artifacts": [],
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


def _search_round_cache(
    path: Path,
    *,
    search_id: str,
    route: str,
    round_index: int,
    candidate_ids: Sequence[str],
    fingerprints: Mapping[str, Mapping[str, str]],
    canonical_control: Mapping[str, str],
    protocol_hash: str,
    score_metric: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        value = read_object(path)
    except Exception:
        return None
    if (value.get("schema") != "geak.asm-canonical-search-round/v3" or
            value.get("search_id") != search_id or value.get("route") != route or
            int(value.get("round", -1)) != round_index or
            value.get("candidate_ids") != list(candidate_ids) or
            value.get("fingerprints") != dict(fingerprints) or
            value.get("canonical_control") != dict(canonical_control) or
            value.get("protocol_hash") != protocol_hash or
            value.get("score_metric") != dict(score_metric)):
        return None
    receipt = value.get("receipt") or {}
    medians = receipt.get("medians_ms") or {}
    if set(medians) != set(candidate_ids) | {CANONICAL_CONTROL_ARM}:
        return None
    try:
        if not all(math.isfinite(float(medians[name])) and float(medians[name]) > 0
                   for name in list(candidate_ids) + [CANONICAL_CONTROL_ARM]):
            return None
    except (TypeError, ValueError):
        return None
    if (receipt.get("protocol_hash") != protocol_hash
            or receipt.get("metric_id") != score_metric.get("id")
            or (receipt.get("metric") or {}).get("inner") != score_metric.get("inner")):
        return None
    invariants = receipt.get("invariants") or {}
    if not invariants or not all(bool(value) for value in invariants.values()):
        return None
    if (receipt.get("gpu_lock") or {}).get("mode") != "external":
        return None
    return value


def search_release(
    contract: TaskContract,
    *,
    cache_root: Path,
    output_dir: Path,
    variants: Sequence[str] = (),
    case_ids: Optional[Sequence[str]] = None,
    hipcc: Optional[str] = None,
    optimization: str = "-O3",
    bootstrap: bool = False,
) -> Dict[str, Any]:
    """Run bounded per-route successive halving under protocol v2.

    Candidate generation and route-static pruning happen before compilation.
    Every round is a protocol-v2 paired-arms receipt, and each route retains its
    own top-K. Compile artifacts and complete round receipts are content checked
    and reusable after interruption.
    """

    if os.environ.get("GEAK_GPU_LOCK_HELD") != "1" or os.environ.get("GEAK_GPU_FENCE_HELD") != "1":
        raise IntegrationError("canonical search requires gpu_lock.sh + gpu_fence_run.sh")
    if not contract.oracle_compatible:
        raise IntegrationError("canonical search requires a protocol-v2 oracle manifest")

    requested_cases = case_ids
    if bootstrap and not requested_cases:
        requested_cases = tuple(
            (contract.search_space.get("bootstrap_search") or {}).get("cases") or ()
        )
    wanted = tuple(dict.fromkeys(str(value) for value in (
        requested_cases or [case["id"] for case in contract.cases]
    )))
    known = {str(case["id"]) for case in contract.cases}
    unknown = set(wanted) - known
    if unknown:
        raise IntegrationError("unknown canonical search routes: %s" % sorted(unknown))
    policy = policy_for(contract.search_space, bootstrap=bootstrap)
    top_n = policy["first_round_top_n"]
    top_k = policy["top_k"]
    rounds_limit = policy["rounds"]
    eta = policy["eta"]
    cu_count = policy["cu_count"]

    requested_variants = tuple(dict.fromkeys(str(name) for name in variants))
    if requested_variants:
        pool = tuple(contract.config_variant(name) for name in requested_variants)
        if len(pool) < 3:
            raise IntegrationError("canonical search refuses fewer than three distinct candidates")
    else:
        pool = contract.search_configs

    plans: Dict[str, Any] = {}
    for route in contract.routes:
        if route.name not in wanted:
            continue
        plan = plan_route_candidates(
            contract.search_space, pool, route, cu_count=cu_count, limit=top_n,
            assume_static_valid=True,
        )
        if len(plan.candidates) < top_n:
            raise IntegrationError(
                "route %s produced only %d candidates below required top-N=%d"
                % (route.name, len(plan.candidates), top_n)
            )
        plans[route.name] = plan

    compile_ids = {contract.default_config.config_id}
    config_by_id = {config.config_id: config for config in pool}
    config_by_id[contract.default_config.config_id] = contract.default_config
    for plan in plans.values():
        compile_ids.update(config.config_id for config in plan.candidates)
    if len(compile_ids) > policy["max_compiled_candidates"]:
        raise IntegrationError(
            "bounded candidate union %d exceeds compile cap %d"
            % (len(compile_ids), policy["max_compiled_candidates"])
        )
    if len(compile_ids) < 8:
        raise IntegrationError("canonical search requires at least one 8-candidate compile shard")
    compile_configs = [config_by_id[config_id] for config_id in sorted(compile_ids)]

    compiler = ShardCompiler(Path(cache_root), hipcc=hipcc, optimization=optimization)
    results = compiler.compile(
        contract.adapter,
        compile_configs,
        shard_size=policy["shard_size"],
        on_demand=False,
        shared=True,
    )
    artifacts: List[Dict[str, Any]] = []
    for result in results:
        for row in result.manifest["configs"]:
            config = KernelConfig.from_dict(row)
            artifacts.append(_artifact_row(contract, result, config))
    by_id = {row["config_id"]: row for row in artifacts}
    if set(compile_ids) != set(by_id):
        raise IntegrationError("compile metadata gate did not admit the complete bounded candidate union")
    calls = {config_id: _CompiledCallable(contract.adapter_name, by_id[config_id])
             for config_id in compile_ids}

    search_identity = {
        "task_digest": contract.task_digest,
        "protocol_hash": contract.protocol["hash"],
        "score_metric": dict(contract.score_metric),
        "candidate_vs_canonical_gate": {
            "schema": PAIR_GATE_SCHEMA,
            "algorithm": PAIR_GATE_ALGORITHM,
            "practical_mde": PAIR_GATE_MDE,
            "confidence": PAIR_GATE_CONFIDENCE,
            "resamples": PAIR_GATE_RESAMPLES,
            "seed": PAIR_GATE_SEED,
        },
        "policy": policy,
        "bootstrap": bool(bootstrap),
        "routes": {name: plans[name].to_dict() for name in sorted(plans)},
        "artifacts": {config_id: {
            "artifact_sha256": by_id[config_id]["artifact_sha256"],
            "source_sha256": by_id[config_id]["source_sha256"],
            "compile_manifest_sha256": by_id[config_id]["compile_manifest_sha256"],
            "isa_hash": by_id[config_id]["isa_hash"],
        } for config_id in sorted(by_id)},
    }
    search_id = hashlib.sha256(canonical_json(search_identity).encode("utf-8")).hexdigest()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "search_state.json"
    state: Dict[str, Any] = {
        "schema": "geak.asm-canonical-search-state/v3",
        "search_id": search_id,
        "identity": search_identity,
        "status": "running",
        "completed_rounds": [],
    }
    if state_path.is_file():
        try:
            prior = read_object(state_path)
            if prior.get("search_id") == search_id:
                state = prior
                state["status"] = "running"
        except Exception:
            pass
    _atomic_json(state_path, state)

    route_selection = {route.name: None for route in contract.routes}
    route_rankings = {route.name: [] for route in contract.routes}
    route_outcomes: Dict[str, Any] = {
        route.name: {
            "status": "unmeasured",
            "selection": "canonical",
            "reason": "route was outside this search target",
            "candidate_gates": [],
        }
        for route in contract.routes
    }
    measurements: Dict[str, Any] = {}
    cache_hits = 0
    cache_misses = 0
    total_arm_evaluations = 0

    with _task_import_path(contract.task_root):
        runner = _import_path(
            contract.task_root / "scripts" / "task_runner.py",
            "geak_search_runner_%s" % uuid.uuid4().hex,
        )
        harness = sys.modules.get("harness_lib")
        if harness is None:
            raise IntegrationError("canonical task runner did not load its harness")
        canonical_call = _canonical_task_callable(contract)
        canonical_control = {
            "arm": CANONICAL_CONTROL_ARM,
            "kind": "canonical_task_implementation",
            "task_digest": contract.task_digest,
            "source_digest": contract.source_digest,
        }
        for case in contract.cases:
            case_id = str(case["id"])
            if case_id not in plans:
                continue
            route = next(route for route in contract.routes if route.name == case_id)
            slots = runner._make_measurement_slots(dict(case), dict(contract.protocol))
            survivors = list(plans[case_id].candidates)
            rounds: List[Dict[str, Any]] = []
            last_medians: Dict[str, float] = {}
            for round_index in range(rounds_limit):
                candidate_ids = [config.config_id for config in survivors]
                projected = total_arm_evaluations + len(candidate_ids)
                route_projected = sum(len(row["candidate_ids"]) for row in rounds) + len(candidate_ids)
                if route_projected > policy["max_route_arm_evaluations"]:
                    raise IntegrationError(
                        "route %s exceeds arm-evaluation budget %d"
                        % (case_id, policy["max_route_arm_evaluations"])
                    )
                fingerprints = {config_id: {
                    "artifact_sha256": by_id[config_id]["artifact_sha256"],
                    "source_sha256": by_id[config_id]["source_sha256"],
                    "compile_manifest_sha256": by_id[config_id]["compile_manifest_sha256"],
                    "isa_hash": by_id[config_id]["isa_hash"],
                } for config_id in candidate_ids}
                round_path = output_dir / "rounds" / case_id / ("round_%02d.json" % round_index)
                cached = _search_round_cache(
                    round_path,
                    search_id=search_id,
                    route=case_id,
                    round_index=round_index,
                    candidate_ids=candidate_ids,
                    fingerprints=fingerprints,
                    canonical_control=canonical_control,
                    protocol_hash=str(contract.protocol["hash"]),
                    score_metric=contract.score_metric,
                )
                round_cache_hit = cached is not None
                if cached is None:
                    correctness: Dict[str, Any] = {}
                    if round_index == 0:
                        first_args = tuple(slots[0]["args"])
                        if contract.adapter_name == "bf16":
                            import torch
                            reference = (first_args[0].float() @ first_args[1].float().t()).to(torch.bfloat16)
                        else:
                            reference = runner._dequant_reference(*first_args)
                        for config_id in candidate_ids:
                            output = calls[config_id](*first_args)
                            ok, error = harness.correct(
                                output, reference, float(contract.value.get("tolerance", 0.02)),
                            )
                            correctness[config_id] = {
                                "passed": bool(ok),
                                "max_relative_error": error,
                            }
                            if not ok:
                                raise IntegrationError(
                                    "%s failed %s correctness" % (config_id, case_id)
                                )
                        canonical_output = canonical_call(slots[0])
                        canonical_ok, canonical_error = harness.correct(
                            canonical_output, reference,
                            float(contract.value.get("tolerance", 0.02)),
                        )
                        correctness[CANONICAL_CONTROL_ARM] = {
                            "passed": bool(canonical_ok),
                            "max_relative_error": canonical_error,
                        }
                        if not canonical_ok:
                            raise IntegrationError(
                                "canonical task implementation failed %s correctness" % case_id
                            )
                    arms = {
                        config_id: (lambda slot, fn=calls[config_id]: fn(*slot["args"]))
                        for config_id in candidate_ids
                    }
                    arms[CANONICAL_CONTROL_ARM] = canonical_call
                    receipt = harness.measure_paired_arms(
                        arms, slots, protocol=dict(contract.protocol),
                        metric=contract.score_metric_id,
                    )
                    if (receipt.get("protocol_hash") != contract.protocol["hash"]
                            or receipt.get("metric_id") != contract.score_metric_id
                            or (receipt.get("metric") or {}).get("inner")
                            != contract.score_metric_inner):
                        raise IntegrationError("measurement receipt protocol/score metric drift")
                    invariants = receipt.get("invariants") or {}
                    if not invariants or not all(invariants.values()):
                        raise IntegrationError("protocol-v2 receipt invariant failed for %s" % case_id)
                    if (receipt.get("gpu_lock") or {}).get("mode") != "external":
                        raise IntegrationError("measurement did not inherit the lane GPU lock")
                    plan_selection = plans[case_id].selection_by_id()
                    cached = {
                        "schema": "geak.asm-canonical-search-round/v3",
                        "search_id": search_id,
                        "route": case_id,
                        "round": round_index,
                        "candidate_ids": candidate_ids,
                        "candidate_selection": {
                            config_id: plan_selection[config_id] for config_id in candidate_ids
                        },
                        "fingerprints": fingerprints,
                        "canonical_control": canonical_control,
                        "protocol_hash": contract.protocol["hash"],
                        "score_metric": dict(contract.score_metric),
                        "correctness": correctness,
                        "receipt": receipt,
                    }
                    _atomic_json(round_path, cached)
                    cache_misses += 1
                else:
                    cache_hits += 1
                medians = {name: float(value) for name, value in
                           cached["receipt"]["medians_ms"].items()}
                last_medians = medians
                final_round = round_index + 1 == rounds_limit
                keep = min(
                    len(survivors),
                    top_k if final_round else max(
                        top_k, int(math.ceil(len(survivors) / float(eta))),
                    ),
                )
                survivors = sorted(
                    survivors,
                    key=lambda config: (medians[config.config_id], config.config_id),
                )[:keep]
                total_arm_evaluations = projected
                round_row = {
                    "round": round_index,
                    "candidate_ids": candidate_ids,
                    "kept": [config.config_id for config in survivors],
                    "medians_ms": medians,
                    "receipt": str(round_path),
                    "receipt_sha256": sha256_file(round_path),
                    "cache_hit": round_cache_hit,
                    "protocol_samples_per_arm": len(slots) * int(
                        contract.protocol["metrics"][contract.score_metric_id]["cycles"]
                    ),
                }
                rounds.append(round_row)
                key = "%s:%d" % (case_id, round_index)
                completed = list(state.get("completed_rounds") or [])
                if key not in completed:
                    completed.append(key)
                state["completed_rounds"] = completed
                state["last_completed"] = key
                _atomic_json(state_path, state)

            selected = sorted(
                survivors,
                key=lambda config: (last_medians[config.config_id], config.config_id),
            )[:top_k]
            candidate_gates = [
                _candidate_vs_canonical_gate(
                    cached["receipt"], route=case_id, candidate_id=config.config_id,
                )
                for config in selected
            ]
            gate_by_id = {gate["candidate_id"]: gate for gate in candidate_gates}
            selected_ids = [
                config.config_id for config in selected
                if gate_by_id[config.config_id]["accepted"]
            ]
            route_selection[case_id] = selected_ids[0] if selected_ids else None
            route_rankings[case_id] = selected_ids
            if selected_ids:
                route_outcomes[case_id] = {
                    "status": "winner",
                    "selection": "candidate",
                    "reason": "candidate-vs-canonical paired MDE gate passed",
                    "config_ids": selected_ids,
                    "candidate_gates": candidate_gates,
                }
            else:
                route_outcomes[case_id] = {
                    "status": "no_winner",
                    "selection": "canonical",
                    "reason": "no candidate proved paired MDE improvement over canonical",
                    "candidate_gates": candidate_gates,
                }
            measurements[case_id] = {
                "candidate_plan": plans[case_id].to_dict(),
                "rounds": rounds,
                "comparison": "candidate_vs_canonical_task_implementation",
                "outcome": route_outcomes[case_id],
                "selected": [{
                    "rank": rank,
                    "config_id": config.config_id,
                    "median_ms": last_medians[config.config_id],
                    "isa_hash": by_id[config.config_id]["isa_hash"],
                    "candidate_vs_canonical_gate": gate_by_id[config.config_id],
                } for rank, config in enumerate(selected)],
            }

    selected_config_ids = sorted({
        config_id for ranking in route_rankings.values() for config_id in ranking
    })
    release_artifacts = [by_id[config_id] for config_id in selected_config_ids]
    payload: Dict[str, Any] = {
        "schema": RELEASE_SCHEMA,
        "task": contract.identity(),
        "protocol": dict(contract.protocol),
        "score_metric": dict(contract.score_metric),
        "oracle": {
            "manifest_sha256": contract.oracle_sha256,
            "protocol_hash": contract.oracle.get("protocol_hash"),
            "compatible": contract.oracle_compatible,
            "selection_metric": contract.oracle.get("selection_metric"),
            "score_metric": contract.oracle.get("score_metric"),
        },
        "fallback": {
            "kind": "canonical",
            "reason": "route is unmeasured or has no paired MDE winner",
        },
        "search_space_manifest": str(contract.value["search_space_manifest"]),
        "variant": ("declared:" + ",".join(requested_variants)
                    if requested_variants else "generated-successive-halving"),
        "selection_reason": "protocol-v2 per-route successive halving; measured_routes=%s"
                            % ",".join(sorted(wanted)),
        "measured_routes": sorted(wanted),
        "route_selection": route_selection,
        "route_rankings": route_rankings,
        "route_outcomes": route_outcomes,
        "search": {
            "search_id": search_id,
            "policy": policy,
            "compiled_candidates": len(compile_ids),
            "compile_shards": [len(result.manifest["configs"]) for result in results],
            "arm_evaluations": total_arm_evaluations,
        },
        "artifacts": release_artifacts,
    }
    payload["release_id"] = _release_identity(payload)
    payload["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    target = Path(cache_root) / "releases" / payload["release_id"] / "release.json"
    _atomic_json(target, payload)
    validate_release(payload, contract)
    summary = {
        "schema": "geak.asm-canonical-search/v3",
        "ok": True,
        "task": contract.task,
        "release_id": payload["release_id"],
        "release_manifest": str(target),
        "search_id": search_id,
        "task_digest": contract.task_digest,
        "protocol_hash": contract.protocol["hash"],
        "score_metric": dict(contract.score_metric),
        "oracle_manifest_sha256": contract.oracle_sha256,
        "measured_routes": sorted(wanted),
        "route_selection": route_selection,
        "route_rankings": route_rankings,
        "route_outcomes": route_outcomes,
        "search_space": contract.search_materialization.to_dict(),
        "budget": payload["search"],
        "cache": {"round_hits": cache_hits, "round_misses": cache_misses,
                  "compile_hits": sum(1 for result in results if result.cache_hit),
                  "compile_misses": sum(1 for result in results if not result.cache_hit)},
        "measurements": measurements,
    }
    state["status"] = "complete"
    state["release_id"] = payload["release_id"]
    _atomic_json(state_path, state)
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
    if value.get("score_metric") != dict(contract.score_metric):
        raise IntegrationError("release score metric differs from canonical task contract")
    oracle = value.get("oracle") or {}
    if (oracle.get("compatible") is not True or
            oracle.get("manifest_sha256") != contract.oracle_sha256 or
            oracle.get("protocol_hash") != contract.protocol.get("hash") or
            oracle.get("selection_metric") != contract.score_metric_id or
            oracle.get("score_metric") != dict(contract.score_metric)):
        raise IntegrationError("release oracle manifest is stale or metric-mismatched")
    routes = value.get("route_selection")
    route_names = {route.name for route in contract.routes}
    if not isinstance(routes, dict) or set(routes) != route_names:
        raise IntegrationError("release does not select the complete canonical suite")
    rankings = value.get("route_rankings")
    if not isinstance(rankings, dict) or set(rankings) != route_names:
        raise IntegrationError("release does not rank the complete canonical suite")
    outcomes = value.get("route_outcomes")
    if not isinstance(outcomes, dict) or set(outcomes) != route_names:
        raise IntegrationError("release does not record every route outcome")
    for route_name, config_ids in rankings.items():
        if not isinstance(config_ids, list) or len(config_ids) != len(set(config_ids)):
            raise IntegrationError("route %s rankings must be a unique list" % route_name)
        outcome = outcomes[route_name]
        if not isinstance(outcome, dict):
            raise IntegrationError("route %s outcome must be an object" % route_name)
        if config_ids:
            if (routes[route_name] != config_ids[0]
                    or outcome.get("status") != "winner"
                    or outcome.get("selection") != "candidate"
                    or outcome.get("config_ids") != config_ids):
                raise IntegrationError("route %s candidate winner/outcome differs" % route_name)
            gates = outcome.get("candidate_gates")
            if not isinstance(gates, list) or not gates:
                raise IntegrationError(
                    "route %s lacks candidate-vs-canonical winner evidence" % route_name
                )
            accepted = set()
            seen_gate_ids = set()
            for gate in gates:
                if not isinstance(gate, dict):
                    raise IntegrationError("route %s gate is not an object" % route_name)
                candidate_id = gate.get("candidate_id")
                try:
                    lower = float(gate["ci95_lower_log_ratio"])
                    threshold = float(gate["threshold_log_ratio"])
                    n_pairs = int(gate["n_pairs"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise IntegrationError(
                        "route %s gate lacks deterministic paired statistics" % route_name
                    ) from exc
                pair_ids = gate.get("pair_ids")
                accepted_gate = gate.get("accepted")
                if (gate.get("schema") != PAIR_GATE_SCHEMA
                        or gate.get("comparison") !=
                        "candidate_vs_canonical_task_implementation"
                        or gate.get("route") != route_name
                        or candidate_id in seen_gate_ids
                        or not isinstance(candidate_id, str)
                        or gate.get("protocol_hash") != contract.protocol["hash"]
                        or gate.get("metric_id") != contract.score_metric_id
                        or gate.get("metric_inner") != contract.score_metric_inner
                        or (gate.get("algorithm") or {}).get("name") != PAIR_GATE_ALGORITHM
                        or n_pairs < PAIR_GATE_MIN_PAIRS
                        or not isinstance(pair_ids, list) or len(pair_ids) != n_pairs
                        or any(not isinstance(pair_id, str) or len(pair_id) != 64
                               for pair_id in pair_ids)
                        or not math.isfinite(lower) or not math.isfinite(threshold)
                        or not isinstance(accepted_gate, bool)
                        or accepted_gate != (lower > threshold)):
                    raise IntegrationError(
                        "route %s has stale candidate-vs-canonical gate" % route_name
                    )
                seen_gate_ids.add(candidate_id)
                if accepted_gate:
                    accepted.add(candidate_id)
            if set(config_ids) != accepted:
                raise IntegrationError(
                    "route %s winner list differs from accepted paired gates" % route_name
                )
        elif (routes[route_name] is not None
              or outcome.get("status") not in ("unmeasured", "no_winner")
              or outcome.get("selection") != "canonical"
              or not isinstance(outcome.get("candidate_gates"), list)):
            raise IntegrationError("route %s lacks explicit canonical fallback outcome" % route_name)
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list):
        raise IntegrationError("release artifacts must be a list")
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
    ranked_ids = {config_id for config_ids in rankings.values() for config_id in config_ids}
    if ranked_ids != set(by_id):
        raise IntegrationError("release artifacts must exactly equal paired-gate winners")


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
    installed_release = read_object(target / "release.json")
    validate_release(installed_release, local_contract)
    if canonical_json(installed_release) != canonical_json(dict(value)):
        raise IntegrationError("installed release manifest differs from selected registry release")
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
        ranked: List[Mapping[str, Any]] = []
        for config_id in value["route_rankings"][route.name]:
            config_id = str(config_id)
            source = by_id[config_id]
            local = installed[config_id]
            ranked.append({
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
            })
        selections[route.name] = ranked
    release_context = {
        "release_id": value["release_id"],
        "task": {
            "name": local_contract.task,
            "adapter": local_contract.adapter_name,
            "digest": local_contract.task_digest,
            "source_digest": local_contract.source_digest,
            "semantic_hashes": dict(local_contract.semantic_hashes),
            "binding_hashes": dict(local_contract.binding_hashes),
        },
        "protocol": {
            "schema": local_contract.protocol["schema"],
            "version": local_contract.protocol["version"],
            "name": local_contract.protocol["name"],
            "hash": local_contract.protocol["hash"],
        },
        "score_metric": dict(local_contract.score_metric),
        "oracle": value["oracle"],
        "fallback": value["fallback"],
        "search": dict(value.get("search") or {}),
        "route_outcomes": {
            route: dict(outcome) for route, outcome in value["route_outcomes"].items()
        },
    }
    table_path = local_contract.materialization_path("runtime_table", task_root=workspace)
    table = RuntimeTableStore(
        table_path, state_root=workspace / ".geak_tuner" / "runtime_table_state",
    ).publish_suite(
        local_contract.routes, selections,
        run_id="materialize-%s" % value["release_id"][:16],
        receipts_root=target.relative_to(workspace),
        release=release_context,
    )
    table_sha256 = sha256_file(table_path)
    selection = {
        "schema": SELECTION_SCHEMA,
        "task": local_contract.task,
        "task_digest": local_contract.task_digest,
        "source_digest": local_contract.source_digest,
        "release_id": value["release_id"],
        "registry_id": registry_id,
        "protocol_hash": local_contract.protocol["hash"],
        "score_metric": dict(local_contract.score_metric),
        "score_metric_id": local_contract.score_metric_id,
        "score_metric_inner": local_contract.score_metric_inner,
        "oracle_manifest_sha256": local_contract.oracle_sha256,
        "runtime_table": str(local_contract.materialization_relative_path("runtime_table")),
        "runtime_table_sha256": table_sha256,
        "table_digest": table["table_digest"],
        "config_ids": sorted({
            config_id for ranking in value["route_rankings"].values() for config_id in ranking
        }),
        "route_outcomes": {
            route: dict(outcome) for route, outcome in value["route_outcomes"].items()
        },
    }
    selection_path = local_contract.materialization_path("selection_file", task_root=workspace)
    _atomic_json(selection_path, selection)
    selection_sha256 = sha256_file(selection_path)
    release_manifest_path = target / "release.json"
    receipt = {
        "schema": MATERIALIZATION_SCHEMA,
        "ok": True,
        "workspace": str(workspace),
        "release_id": value["release_id"],
        "registry_id": registry_id,
        "runtime_table": str(table_path),
        "runtime_table_sha256": table_sha256,
        "table_digest": table["table_digest"],
        "task_digest": local_contract.task_digest,
        "source_digest": local_contract.source_digest,
        "protocol_hash": local_contract.protocol["hash"],
        "score_metric": dict(local_contract.score_metric),
        "score_metric_id": local_contract.score_metric_id,
        "score_metric_inner": local_contract.score_metric_inner,
        "oracle_manifest_sha256": local_contract.oracle_sha256,
        "route_count": len(local_contract.routes),
        "config_ids": selection["config_ids"],
        "route_outcomes": selection["route_outcomes"],
        "selection_file": str(selection_path),
        "selection_sha256": selection_sha256,
        "release_manifest": str(release_manifest_path),
        "release_manifest_sha256": sha256_file(release_manifest_path),
        "artifacts": [dict(row) for row in installed.values()],
    }
    receipt_path = workspace / ".geak_tuner" / "materialization_receipt.json"
    _atomic_json(receipt_path, receipt)
    receipt["receipt"] = str(receipt_path)
    return receipt


def _load_library_symbols(workspace: Path, table: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    loaded: Dict[str, Any] = {}
    rows: List[Mapping[str, Any]] = []
    for route_id in table["suite"]["route_ids"]:
        for entry in table["routes"][route_id]["entries"]:
            artifact = (workspace / entry["artifact"]).resolve()
            source = (workspace / entry["source"]).resolve()
            compile_manifest = (workspace / entry["compile_manifest"]).resolve()
            if (not _within(artifact, workspace) or not _within(source, workspace) or
                    not _within(compile_manifest, workspace)):
                raise IntegrationError("runtime table path escapes materialized workspace")
            if not artifact.is_file() or artifact.is_symlink() or sha256_file(artifact) != entry["artifact_sha256"]:
                raise IntegrationError("materialized artifact hash mismatch for %s" % route_id)
            if not source.is_file() or source.is_symlink() or sha256_file(source) != entry["source_sha256"]:
                raise IntegrationError("materialized source hash mismatch for %s" % route_id)
            if (not compile_manifest.is_file() or compile_manifest.is_symlink() or
                    sha256_file(compile_manifest) != entry["compile_manifest_sha256"]):
                raise IntegrationError("materialized compile manifest hash mismatch for %s" % route_id)
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
                "rank": int(entry["rank"]),
                "config_id": entry["config_id"],
                "symbol": entry["symbol"],
                "artifact_sha256": entry["artifact_sha256"],
                "source_sha256": entry["source_sha256"],
                "compile_manifest_sha256": entry["compile_manifest_sha256"],
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
    table = RuntimeTableStore(
        local.materialization_path("runtime_table", task_root=workspace),
        state_root=workspace / ".geak_tuner" / "runtime_table_state",
    ).load()
    if table["task"]["name"] != local.task or table["task"]["digest"] != local.task_digest:
        raise IntegrationError("runtime table task digest differs from independent workspace")
    if table["task"].get("semantic_hashes") != dict(local.semantic_hashes):
        raise IntegrationError("runtime table semantic file hashes differ")
    if table["task"].get("binding_hashes") != dict(local.binding_hashes):
        raise IntegrationError("runtime table CASES/input/correctness binding hashes differ")
    if table["protocol"]["hash"] != local.protocol["hash"]:
        raise IntegrationError("runtime table protocol hash differs")
    if table.get("score_metric") != dict(local.score_metric):
        raise IntegrationError("runtime table score metric differs")
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
            for entry in table["routes"][route_id]["entries"]:
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
        "score_metric": dict(local.score_metric),
        "oracle_manifest_sha256": local.oracle_sha256,
        "table_digest": table["table_digest"],
        "release_id": table["release_id"],
        "routes_verified": len(symbol_rows),
        "route_selections": {
            route_id: dict(table["routes"][route_id]["selection"])
            for route_id in table["suite"]["route_ids"]
        },
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
    table = RuntimeTableStore(
        contract.materialization_path("runtime_table", task_root=workspace),
        state_root=workspace / ".geak_tuner" / "runtime_table_state",
    ).load()
    route_row = table["routes"][case_id]
    route_selection = dict(route_row["selection"])
    expect_candidate = bool(route_row["entries"])
    trace = workspace / ".geak_tuner" / ("dispatch-%s.json" % case_id)
    trace.unlink(missing_ok=True)
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
        if expect_candidate:
            dispatch = read_object(trace)
            if (dispatch.get("route") != case_id or not dispatch.get("symbol")
                    or not dispatch.get("config_id")):
                raise IntegrationError("task dispatch did not consume the versioned runtime table")
        else:
            if trace.exists():
                raise IntegrationError("canonical-fallback route unexpectedly dispatched a tuner artifact")
            dispatch = {
                "config_id": None,
                "symbol": None,
                "artifact_sha256": None,
                "source_sha256": None,
            }
        return {
            "schema": SMOKE_SCHEMA,
            "ok": True,
            "task": contract.task,
            "case": case_id,
            "shape": {"m": case["M"], "n": case["N"], "k": case["K"]},
            "task_digest": contract.task_digest,
            "protocol_hash": contract.protocol["hash"],
            "score_metric": dict(contract.score_metric),
            "table_digest": verification["table_digest"],
            "dispatch": "candidate" if expect_candidate else "canonical",
            "route_selection": route_selection,
            "config_id": dispatch["config_id"],
            "symbol": dispatch["symbol"],
            "artifact_sha256": dispatch["artifact_sha256"],
            "source_sha256": dispatch["source_sha256"],
            "max_relative_error": max_relative_error,
            "gpu_lock": {"mode": "external", "path": os.environ.get("GEAK_GPU_LOCK_FILE")},
            "dispatch_receipt": str(trace) if expect_candidate else None,
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
