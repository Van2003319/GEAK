from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .adapters import get_adapter
from .adapters.base import AxisState
from .pruning import PruneResult, RejectedConfig, route_static_prune, shape_sort_key, static_prune
from .schema import KernelConfig, Mfma, Route


SEARCH_SPACE_SCHEMA = "geak.asm-search-space/v2"
GENERATOR_KIND = "geak.dependent-domain-sweeps/v1"


class SearchSpaceError(ValueError):
    """A search-space manifest cannot be expanded safely or reproducibly."""


@dataclass(frozen=True)
class SearchSpaceMaterialization:
    generated: Tuple[KernelConfig, ...]
    accepted: Tuple[KernelConfig, ...]
    rejected: Tuple[RejectedConfig, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated": len(self.generated),
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
        }


@dataclass(frozen=True)
class RouteCandidatePlan:
    route: Route
    accepted_count: int
    rejected_count: int
    ranked_count: int
    candidates: Tuple[KernelConfig, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route.name,
            "accepted": self.accepted_count,
            "rejected": self.rejected_count,
            "ranked": self.ranked_count,
            "selected": [config.config_id for config in self.candidates],
        }


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, list) or not value:
        raise SearchSpaceError("domain %s must be a non-empty array" % name)
    return value


def _domain(manifest: Mapping[str, Any], name: str) -> Sequence[Any]:
    domains = manifest.get("domains") or {}
    if not isinstance(domains, Mapping) or name not in domains:
        raise SearchSpaceError("generator references missing domain %s" % name)
    return _sequence(domains[name], name)


def _factor_pairs(value: int) -> Tuple[Tuple[int, int], ...]:
    if value <= 0:
        raise SearchSpaceError("wave-count domain values must be positive")
    return tuple((m, value // m) for m in range(1, value + 1) if value % m == 0)


def _base_with_auto_fields(row: Mapping[str, Any], provenance: Mapping[str, Any]) -> KernelConfig:
    value = copy.deepcopy(dict(row))
    wave = value.get("wave_grid") or {}
    mfma = Mfma.from_obj(value.get("mfma"))
    if value.get("lds_scratch_bytes") in (None, "auto"):
        value["lds_scratch_bytes"] = int(wave["m"]) * int(wave["n"]) * mfma.m * mfma.n * 4
    value["provenance"] = dict(provenance)
    return KernelConfig.from_dict(value)


def _validate_policy(policy: Mapping[str, Any], name: str) -> None:
    top_n = int(policy.get("first_round_top_n", 0))
    top_k = int(policy.get("top_k", 0))
    rounds = int(policy.get("rounds", 0))
    eta = int(policy.get("eta", 0))
    shard_size = int(policy.get("shard_size", 0))
    max_compiled = int(policy.get("max_compiled_candidates", 0))
    max_evaluations = int(policy.get("max_route_arm_evaluations", 0))
    if top_n < 3 or top_k < 1 or top_k >= top_n:
        raise SearchSpaceError("%s policy requires first_round_top_n >= 3 and 1 <= top_k < top_n" % name)
    if rounds < 2 or eta < 2:
        raise SearchSpaceError("%s policy requires rounds >= 2 and eta >= 2" % name)
    if shard_size < 8 or shard_size > 16:
        raise SearchSpaceError("%s policy shard_size must be in [8,16]" % name)
    if max_compiled < top_n:
        raise SearchSpaceError("%s policy max_compiled_candidates is below first-round top-N" % name)
    theoretical = 0
    survivors = top_n
    for _ in range(rounds):
        theoretical += survivors
        survivors = max(top_k, int(math.ceil(survivors / float(eta))))
    if max_evaluations < theoretical:
        raise SearchSpaceError(
            "%s policy budget %d cannot cover bounded halving cost %d"
            % (name, max_evaluations, theoretical)
        )


def validate_search_manifest(manifest: Mapping[str, Any], adapter_name: str) -> None:
    if manifest.get("schema") != SEARCH_SPACE_SCHEMA:
        raise SearchSpaceError("unsupported search-space schema %r" % manifest.get("schema"))
    adapter = get_adapter(adapter_name)
    if manifest.get("adapter") != adapter_name:
        raise SearchSpaceError("search-space adapter mismatch")
    if manifest.get("axis_status") != adapter.axis_status():
        raise SearchSpaceError("search-space axis_status drifted from adapter capabilities")
    generator = manifest.get("generator") or {}
    if generator.get("kind") != GENERATOR_KIND:
        raise SearchSpaceError("unsupported dependent generator %r" % generator.get("kind"))
    domains = manifest.get("domains") or {}
    if not isinstance(domains, Mapping) or not domains:
        raise SearchSpaceError("search-space domains must be a non-empty object")
    for name, values in domains.items():
        _sequence(values, str(name))

    geometry = generator.get("geometry") or {}
    required_geometry = {"bm", "bn", "bk", "wave_count"}
    if set(geometry) != required_geometry:
        raise SearchSpaceError("generator geometry must define exactly %s" % sorted(required_geometry))
    referenced = set(str(value) for value in geometry.values())
    sweeps = generator.get("profile_sweeps") or []
    if not isinstance(sweeps, list) or not sweeps:
        raise SearchSpaceError("generator profile_sweeps must be non-empty")
    sweep_names = set()
    for sweep in sweeps:
        name = str((sweep or {}).get("name") or "")
        axes = (sweep or {}).get("axes") or []
        if not name or name in sweep_names or not isinstance(axes, list) or not axes:
            raise SearchSpaceError("each profile sweep needs a unique name and non-empty axes")
        sweep_names.add(name)
        for axis in axes:
            axis = str(axis)
            referenced.add(axis)
            capability = adapter.axis_capabilities.get(axis)
            if capability is None:
                raise SearchSpaceError("sweep %s references unknown axis %s" % (name, axis))
            if capability.state is not AxisState.ACTIVE:
                raise SearchSpaceError(
                    "sweep %s varies %s but that axis is %s"
                    % (name, axis, capability.state.value)
                )
    missing = set(str(name) for name in domains) - referenced
    if missing:
        raise SearchSpaceError("domains are declared but never generated: %s" % sorted(missing))
    for name in referenced:
        _domain(manifest, name)

    dependencies = generator.get("dependencies") or []
    supported_dependencies = {
        "wave_grid_factor_pairs",
        "integral_mfma_wave_tile",
        "auto_epilogue_scratch",
    }
    if set(dependencies) != supported_dependencies:
        raise SearchSpaceError(
            "generator dependencies must be exactly %s" % sorted(supported_dependencies)
        )
    route_dependencies = generator.get("route_dependencies") or []
    supported_route_dependencies = {"pgr2_requires_two_k_stages", "stagger_requires_two_k_stages"}
    if set(route_dependencies) != supported_route_dependencies:
        raise SearchSpaceError(
            "generator route_dependencies must be exactly %s"
            % sorted(supported_route_dependencies)
        )

    base = generator.get("base") or {}
    if not isinstance(base, Mapping):
        raise SearchSpaceError("generator base must be an object")
    neutral = dict(adapter.inactive_neutral_values)
    fail_closed = manifest.get("fail_closed") or {}
    if fail_closed.get("inactive_neutral_values") != neutral:
        raise SearchSpaceError("fail_closed inactive neutral values drifted from the adapter")
    if fail_closed.get("unsupported") != {"gsu": 1, "stream_k": 0}:
        raise SearchSpaceError("fail_closed unsupported-axis values must be GSU=1 and StreamK=0")
    for axis, expected in neutral.items():
        if base.get(axis) != expected:
            raise SearchSpaceError("inactive axis %s must stay at neutral value %r" % (axis, expected))
    if base.get("gsu") != 1 or base.get("stream_k") != 0:
        raise SearchSpaceError("GSU and StreamK must remain fail-closed neutral values")
    pgr_values = {int(value) for value in _domain(manifest, "pgr")}
    if pgr_values != {1, 2}:
        raise SearchSpaceError("production PGR domain must contain exactly PGR1 and PGR2")

    policies = manifest.get("search_policy") or {}
    if not isinstance(policies, Mapping):
        raise SearchSpaceError("search_policy must be an object")
    for name in ("full", "bootstrap"):
        policy = policies.get(name)
        if not isinstance(policy, Mapping):
            raise SearchSpaceError("search_policy.%s is required" % name)
        _validate_policy(policy, name)


def _expand_manifest(manifest: Mapping[str, Any]) -> Tuple[KernelConfig, ...]:
    generator = manifest["generator"]
    base = copy.deepcopy(dict(generator["base"]))
    geometry = generator["geometry"]
    sweeps = generator["profile_sweeps"]
    mfma = Mfma.from_obj(base["mfma"])
    configs: Dict[str, KernelConfig] = {}

    variants = manifest.get("variants") or {}
    if not isinstance(variants, Mapping) or not variants:
        raise SearchSpaceError("search-space variants must contain at least the declared seed")
    for name in sorted(variants):
        config = KernelConfig.from_dict(variants[name])
        configs[config.config_id] = config

    bm_domain = _domain(manifest, geometry["bm"])
    bn_domain = _domain(manifest, geometry["bn"])
    bk_domain = _domain(manifest, geometry["bk"])
    wave_domain = _domain(manifest, geometry["wave_count"])
    for bm, bn, bk, waves in itertools.product(bm_domain, bn_domain, bk_domain, wave_domain):
        bm, bn, bk, waves = int(bm), int(bn), int(bk), int(waves)
        for waves_m, waves_n in _factor_pairs(waves):
            if bm % (waves_m * mfma.m) or bn % (waves_n * mfma.n):
                continue
            geometry_row = copy.deepcopy(base)
            geometry_row.update({
                "bm": bm,
                "bn": bn,
                "bk": bk,
                "wave_grid": {"m": waves_m, "n": waves_n},
            })
            for sweep in sweeps:
                axes = [str(axis) for axis in sweep["axes"]]
                values = [_domain(manifest, axis) for axis in axes]
                for choice in itertools.product(*values):
                    row = copy.deepcopy(geometry_row)
                    for axis, value in zip(axes, choice):
                        row[axis] = copy.deepcopy(value)
                    provenance = {
                        "kind": "dependent_domain_generator",
                        "generator": GENERATOR_KIND,
                        "sweep": str(sweep["name"]),
                        "vendor_default": False,
                    }
                    config = _base_with_auto_fields(row, provenance)
                    configs.setdefault(config.config_id, config)
    return tuple(sorted(configs.values(), key=lambda config: config.config_id))


@lru_cache(maxsize=8)
def _generate_cached(document: str) -> Tuple[KernelConfig, ...]:
    return _expand_manifest(json.loads(document))


def generate_configs(manifest: Mapping[str, Any]) -> Tuple[KernelConfig, ...]:
    """Expand deterministic geometry products and dependent profile sweeps."""

    return _generate_cached(_canonical(manifest))


@lru_cache(maxsize=8)
def _materialize_cached(document: str, adapter_name: str) -> SearchSpaceMaterialization:
    manifest = json.loads(document)
    validate_search_manifest(manifest, adapter_name)
    generated = _generate_cached(document)
    result = static_prune(generated)
    adapter = get_adapter(adapter_name)
    accepted: List[KernelConfig] = []
    for config in result.accepted:
        try:
            adapter.validate(config)
        except Exception as exc:
            raise SearchSpaceError(
                "generated config %s crosses adapter fail-closed boundary: %s"
                % (config.config_id, exc)
            ) from exc
        accepted.append(config)
    if len(accepted) < 3:
        raise SearchSpaceError("dependent generator produced fewer than three executable candidates")
    return SearchSpaceMaterialization(generated, tuple(accepted), result.rejected)


def materialize_search_space(
    manifest: Mapping[str, Any], adapter_name: str,
) -> SearchSpaceMaterialization:
    return _materialize_cached(_canonical(manifest), adapter_name)


def policy_for(manifest: Mapping[str, Any], bootstrap: bool) -> Dict[str, int]:
    name = "bootstrap" if bootstrap else "full"
    raw = (manifest.get("search_policy") or {}).get(name) or {}
    _validate_policy(raw, name)
    return {str(key): int(value) for key, value in raw.items()}


def _stratum(config: KernelConfig) -> Tuple[int, str]:
    return (int(config.pgr), str(config.provenance.get("sweep") or "declared"))


def _profile(config: KernelConfig) -> Tuple[Any, ...]:
    sweep = str(config.provenance.get("sweep") or "declared")
    if sweep == "pipeline_memory":
        return (
            config.a_lds.buffers, config.a_lds.pad_elements,
            config.b_lds.buffers, config.b_lds.pad_elements,
        )
    if sweep == "global_epilogue_io":
        return (
            config.vector_widths.a, config.vector_widths.b,
            config.vector_widths.store, config.store.mode.value,
        )
    if sweep == "mapping_schedule":
        return (config.wgm, config.stagger)
    return ("declared",)


def _profile_round_robin(
    configs: Sequence[KernelConfig], *, salt: str, stratum: Tuple[int, str],
) -> List[KernelConfig]:
    profiles: Dict[Tuple[Any, ...], List[KernelConfig]] = {}
    for config in configs:
        profiles.setdefault(_profile(config), []).append(config)
    keys = sorted(profiles, key=lambda value: repr(value))
    if len(keys) > 1:
        digest = hashlib.sha256((salt + repr(stratum)).encode("utf-8")).digest()
        shift = int.from_bytes(digest[:4], "big") % len(keys)
        keys = keys[shift:] + keys[:shift]
    ordered: List[KernelConfig] = []
    depth = 0
    while True:
        progressed = False
        for key in keys:
            bucket = profiles[key]
            if depth < len(bucket):
                ordered.append(bucket[depth])
                progressed = True
        if not progressed:
            return ordered
        depth += 1


def _diverse_prefix(
    ranked: Sequence[KernelConfig], limit: int, *, salt: str,
) -> Tuple[KernelConfig, ...]:
    raw_buckets: Dict[Tuple[int, str], List[KernelConfig]] = {}
    for config in ranked:
        raw_buckets.setdefault(_stratum(config), []).append(config)
    buckets = {
        key: _profile_round_robin(configs, salt=salt, stratum=key)
        for key, configs in raw_buckets.items()
    }
    ordered_keys = sorted(buckets)
    chosen: List[KernelConfig] = []
    seen = set()
    index = 0
    while len(chosen) < limit:
        progressed = False
        for key in ordered_keys:
            bucket = buckets[key]
            if index >= len(bucket):
                continue
            config = bucket[index]
            progressed = True
            if config.config_id not in seen:
                seen.add(config.config_id)
                chosen.append(config)
                if len(chosen) == limit:
                    break
        if not progressed:
            break
        index += 1
    if len(chosen) < limit:
        for config in ranked:
            if config.config_id in seen:
                continue
            chosen.append(config)
            seen.add(config.config_id)
            if len(chosen) == limit:
                break
    return tuple(chosen)


def plan_route_candidates(
    manifest: Mapping[str, Any],
    configs: Iterable[KernelConfig],
    route: Route,
    *,
    cu_count: int,
    limit: int,
    assume_static_valid: bool = False,
) -> RouteCandidatePlan:
    dependencies = tuple((manifest.get("generator") or {}).get("route_dependencies") or ())
    result: PruneResult = route_static_prune(
        configs, route, dependencies=dependencies, assume_static_valid=assume_static_valid,
    )
    ranked = sorted(result.accepted, key=lambda config: shape_sort_key(config, route, cu_count))
    selected = _diverse_prefix(ranked, min(int(limit), len(ranked)), salt=route.name)
    return RouteCandidatePlan(route, len(result.accepted), len(result.rejected), len(ranked), selected)
