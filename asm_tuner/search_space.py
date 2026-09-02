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
from .pruning import (
    SHAPE_OBJECTIVES, PruneResult, RejectedConfig, route_static_prune,
    shape_sort_components, shape_sort_key, static_prune,
)
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
    selection: Tuple[Mapping[str, Any], ...] = ()

    def selection_by_id(self) -> Dict[str, Dict[str, Any]]:
        return {str(row["config_id"]): dict(row) for row in self.selection}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route.name,
            "accepted": self.accepted_count,
            "rejected": self.rejected_count,
            "ranked": self.ranked_count,
            "selected": [config.config_id for config in self.candidates],
            "selection": [dict(row) for row in self.selection],
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


def _geometry(config: KernelConfig) -> Tuple[int, ...]:
    return (
        config.bm, config.bn, config.bk,
        config.wave_grid.m, config.wave_grid.n, config.pgr,
    )


def _tile_family(config: KernelConfig) -> str:
    # Quarter-octave bins preserve meaningful neighboring tile families without
    # naming any vendor geometry or relying on configuration identities.
    bins = tuple(int(round(math.log2(value) * 4.0)) for value in (
        config.bm, config.bn, config.bk,
    ))
    return "m%d-n%d-k%d" % bins


def _cta_tail_band(config: KernelConfig, route: Route, cu_count: int) -> str:
    grid = config.grid_for(route)[2]
    cus = max(1, int(cu_count))
    nearest = max(1, int(round(grid / float(cus))))
    distance = abs(grid - nearest * cus) / float(nearest * cus)
    if distance <= 0.05:
        return "exact_cu_round"
    if distance <= 0.20:
        return "near_cu_round"
    return "partial_cu_round"


def _profile_sweep(config: KernelConfig) -> str:
    return str(config.provenance.get("sweep") or "declared")


def _profile_novelty(config: KernelConfig) -> int:
    sweep = _profile_sweep(config)
    if sweep == "pipeline_memory":
        return sum((
            int(config.a_lds.buffers != 1), int(config.b_lds.buffers != 1),
            int(bool(config.a_lds.pad_elements)), int(bool(config.b_lds.pad_elements)),
        ))
    if sweep == "global_epilogue_io":
        return sum((
            int(config.vector_widths.a != 1), int(config.vector_widths.b != 1),
            int(config.vector_widths.store != 1),
            int(config.store.mode.value == "non_temporal"),
        ))
    if sweep == "mapping_schedule":
        return int(config.wgm != 1) + int(config.stagger != 0)
    return 0


def _pareto_ranks(
    configs: Sequence[KernelConfig], components: Mapping[str, Mapping[str, float]],
) -> Dict[Tuple[int, ...], int]:
    """Assign exact non-dominated sorting ranks to a bounded diverse pool."""

    keys = [_geometry(config) for config in configs]
    vectors = {
        _geometry(config): tuple(float(components[config.config_id][name]) for name in SHAPE_OBJECTIVES)
        for config in configs
    }
    dominates: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {key: [] for key in keys}
    counts = {key: 0 for key in keys}
    for index, left in enumerate(keys):
        lv = vectors[left]
        for right in keys[index + 1:]:
            rv = vectors[right]
            left_dominates = all(a <= b for a, b in zip(lv, rv)) and any(
                a < b for a, b in zip(lv, rv)
            )
            right_dominates = all(b <= a for a, b in zip(lv, rv)) and any(
                b < a for a, b in zip(lv, rv)
            )
            if left_dominates:
                dominates[left].append(right)
                counts[right] += 1
            elif right_dominates:
                dominates[right].append(left)
                counts[left] += 1
    frontier = [key for key in keys if counts[key] == 0]
    ranks: Dict[Tuple[int, ...], int] = {}
    rank = 0
    while frontier:
        next_frontier: List[Tuple[int, ...]] = []
        for key in frontier:
            ranks[key] = rank
            for other in dominates[key]:
                counts[other] -= 1
                if counts[other] == 0:
                    next_frontier.append(other)
        frontier = next_frontier
        rank += 1
    return ranks


def _stratified_pareto_prefix(
    ranked: Sequence[KernelConfig], route: Route, cu_count: int, limit: int, *, salt: str,
) -> Tuple[Tuple[KernelConfig, ...], Tuple[Mapping[str, Any], ...]]:
    """Select a weighted/Pareto prefix with objective and architecture diversity."""

    components = {
        config.config_id: shape_sort_components(config, route, cu_count)
        for config in ranked
    }
    members: Dict[Tuple[int, ...], List[KernelConfig]] = {}
    for config in ranked:
        members.setdefault(_geometry(config), []).append(config)
    representatives = [
        min(group, key=lambda config: (
            components[config.config_id]["weighted_score"], config.config_id,
        ))
        for group in members.values()
    ]
    representatives.sort(key=lambda config: (
        components[config.config_id]["weighted_score"], config.config_id,
    ))

    # Pareto sorting is performed on a bounded union of weighted, objective,
    # tile-family, wave-count and CTA-tail champions. This avoids quadratic work
    # over tens of thousands of profile-equivalent generated configurations.
    pool: Dict[Tuple[int, ...], KernelConfig] = {}

    def admit(values: Sequence[KernelConfig], count: int) -> None:
        for config in values[:count]:
            pool.setdefault(_geometry(config), config)

    breadth = max(8, min(32, int(limit) * 2))
    admit(representatives, max(64, int(limit) * 8))
    pgr_values = sorted({config.pgr for config in representatives})
    for pgr in pgr_values:
        group = [config for config in representatives if config.pgr == pgr]
        admit(group, breadth)
        for objective in SHAPE_OBJECTIVES:
            admit(sorted(group, key=lambda config: (
                components[config.config_id][objective],
                components[config.config_id]["weighted_score"],
                config.config_id,
            )), breadth)
    for wave_count in sorted({config.nw for config in representatives}):
        admit([config for config in representatives if config.nw == wave_count], 4)
    for classifier in (
        lambda config: _tile_family(config),
        lambda config: _cta_tail_band(config, route, cu_count),
    ):
        groups: Dict[str, List[KernelConfig]] = {}
        for config in representatives:
            groups.setdefault(classifier(config), []).append(config)
        for group in groups.values():
            admit(group, 2)

    pareto_rank = _pareto_ranks(tuple(pool.values()), components)
    chosen_geometries: List[Tuple[int, ...]] = []
    reasons: Dict[Tuple[int, ...], str] = {}

    def choose(config: KernelConfig, reason: str) -> None:
        key = _geometry(config)
        if key not in reasons and len(chosen_geometries) < limit:
            chosen_geometries.append(key)
            reasons[key] = reason

    # Two weighted representatives per PGR retain both overlap/resource models.
    # Objective extrema then prevent a good edge, tail, or reuse trade-off from
    # being hidden by the aggregate score.
    by_pgr = {
        pgr: [config for config in representatives if config.pgr == pgr]
        for pgr in pgr_values
    }
    for depth in range(2):
        for pgr in pgr_values:
            if depth < len(by_pgr[pgr]):
                choose(by_pgr[pgr][depth], "weighted_pgr_stratum")

    edge_ranked = sorted(representatives, key=lambda config: (
        components[config.config_id]["edge_waste_normalized"],
        pareto_rank.get(_geometry(config), 999),
        components[config.config_id]["weighted_score"],
        config.config_id,
    ))
    added = 0
    for config in edge_ranked:
        before = len(chosen_geometries)
        choose(config, "pareto_edge_frontier")
        if len(chosen_geometries) > before:
            added += 1
        if added == 2 or len(chosen_geometries) == limit:
            break

    # Ensure the bounded prefix contains more than one wave-count hypothesis.
    seen_waves = {key[3] * key[4] for key in chosen_geometries}
    for wave_count in sorted({config.nw for config in representatives}):
        if wave_count in seen_waves:
            continue
        group = [config for config in representatives if config.nw == wave_count]
        if group:
            choose(group[0], "wave_count_diversity")
            seen_waves.add(wave_count)
        if len(chosen_geometries) == limit:
            break

    for objective in (
        "cta_tail_fraction", "lds_read_per_mfma", "reuse_penalty", "resource_pressure",
    ):
        ordered = sorted(representatives, key=lambda config: (
            components[config.config_id][objective],
            pareto_rank.get(_geometry(config), 999),
            components[config.config_id]["weighted_score"],
            config.config_id,
        ))
        for config in ordered[:breadth]:
            choose(config, "pareto_objective:%s" % objective)
            if len(chosen_geometries) == limit:
                break
        if len(chosen_geometries) == limit:
            break

    # Greedy category novelty covers tile families and target CTA-tail behavior.
    seen_families = set()
    seen_tails = set()
    for key in chosen_geometries:
        config = pool.get(key) or representatives[0]
        seen_families.add(_tile_family(config))
        seen_tails.add(_cta_tail_band(config, route, cu_count))
    while len(chosen_geometries) < limit:
        remaining = [config for config in representatives if _geometry(config) not in reasons]
        if not remaining:
            break
        config = min(remaining, key=lambda candidate: (
            -int(_tile_family(candidate) not in seen_families)
            -int(_cta_tail_band(candidate, route, cu_count) not in seen_tails),
            pareto_rank.get(_geometry(candidate), 999),
            components[candidate.config_id]["weighted_score"],
            candidate.config_id,
        ))
        choose(config, "tile_tail_diversity")
        seen_families.add(_tile_family(config))
        seen_tails.add(_cta_tail_band(config, route, cu_count))

    for config in representatives:
        choose(config, "weighted_fill")
        if len(chosen_geometries) == limit:
            break

    # Rotate profile sweep representatives independently of geometry ranking so
    # the shortlist also exercises every dependent profile family.
    sweeps = sorted({_profile_sweep(config) for config in ranked})
    if sweeps:
        digest = hashlib.sha256(salt.encode("utf-8")).digest()
        shift = int.from_bytes(digest[:4], "big") % len(sweeps)
        sweeps = sweeps[shift:] + sweeps[:shift]

    selected: List[KernelConfig] = []
    selection: List[Mapping[str, Any]] = []
    for index, key in enumerate(chosen_geometries):
        group = members[key]
        desired_sweep = sweeps[index % len(sweeps)] if sweeps else _profile_sweep(group[0])
        eligible = [config for config in group if _profile_sweep(config) == desired_sweep] or group
        config = min(eligible, key=lambda candidate: (
            -_profile_novelty(candidate),
            components[candidate.config_id]["weighted_score"], candidate.config_id,
        ))
        scores = components[config.config_id]
        layer = pareto_rank.get(key, 999)
        reason = reasons[key]
        reason += ";pareto_frontier" if layer == 0 else ";pareto_layer=%d" % layer
        row = {
            "rank": index + 1,
            "config_id": config.config_id,
            "selection_reason": reason,
            "weighted_score": scores["weighted_score"],
            "pareto_rank": layer,
            "pareto_scope": "geometry_best_profile",
            "score_components": {name: scores[name] for name in sorted(scores)},
            "diversity": {
                "wave_count": config.nw,
                "pgr": config.pgr,
                "tile_family": _tile_family(config),
                "cta_tail_band": _cta_tail_band(config, route, cu_count),
                "profile_sweep": _profile_sweep(config),
            },
        }
        selected.append(config)
        selection.append(row)
    return tuple(selected), tuple(selection)


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
    selected, selection = _stratified_pareto_prefix(
        ranked, route, cu_count, min(int(limit), len(ranked)), salt=route.name,
    )
    return RouteCandidatePlan(
        route, len(result.accepted), len(result.rejected), len(ranked), selected, selection,
    )
