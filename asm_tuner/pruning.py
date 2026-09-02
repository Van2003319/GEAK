from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .schema import KernelConfig, Route, validate_config


@dataclass(frozen=True)
class RejectedConfig:
    config_id: str
    errors: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"config_id": self.config_id, "errors": list(self.errors)}


@dataclass(frozen=True)
class PruneResult:
    accepted: Tuple[KernelConfig, ...]
    rejected: Tuple[RejectedConfig, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": [c.to_dict(include_derived=True) for c in self.accepted],
            "rejected": [r.to_dict() for r in self.rejected],
        }


def static_prune(configs: Iterable[KernelConfig]) -> PruneResult:
    accepted: List[KernelConfig] = []
    rejected: List[RejectedConfig] = []
    seen = set()
    for config in configs:
        if config.config_id in seen:
            continue
        seen.add(config.config_id)
        report = validate_config(config)
        if report.valid:
            accepted.append(config)
        else:
            rejected.append(RejectedConfig(config.config_id, report.errors))
    return PruneResult(tuple(accepted), tuple(rejected))


def route_static_prune(
    configs: Iterable[KernelConfig],
    route: Route,
    *,
    dependencies: Sequence[str] = (
        "pgr2_requires_two_k_stages",
        "stagger_requires_two_k_stages",
    ),
    assume_static_valid: bool = False,
) -> PruneResult:
    """Apply compile-time legality plus route-dependent no-op rejection.

    Grid size, CU coverage, edge waste, and data reuse deliberately stay out of
    this hard gate. They are performance hypotheses and belong in soft ranking.
    """

    supported = {"pgr2_requires_two_k_stages", "stagger_requires_two_k_stages"}
    unknown = set(dependencies) - supported
    if unknown:
        raise ValueError("unknown route-static dependencies: %s" % sorted(unknown))
    if assume_static_valid:
        base = PruneResult(tuple(configs), tuple())
    else:
        base = static_prune(configs)
    accepted: List[KernelConfig] = []
    rejected = list(base.rejected)
    for config in base.accepted:
        stages = (route.k + config.bk - 1) // config.bk
        errors: List[str] = []
        if ("pgr2_requires_two_k_stages" in dependencies and
                config.pgr == 2 and stages < 2):
            errors.append("PGR2 requires at least two K stages on route %s" % route.name)
        if ("stagger_requires_two_k_stages" in dependencies and
                config.stagger > 0 and stages < 2):
            errors.append("stagger requires at least two K stages on route %s" % route.name)
        if errors:
            rejected.append(RejectedConfig(config.config_id, tuple(errors)))
        else:
            accepted.append(config)
    return PruneResult(tuple(accepted), tuple(rejected))


# These are normalized minimization objectives. Keeping the names public makes
# the planner's Pareto decisions and the emitted search receipts reproducible.
SHAPE_OBJECTIVES = (
    "edge_waste_normalized",
    "wave_target_distance",
    "cta_tail_fraction",
    "reuse_penalty",
    "lds_read_per_mfma",
    "resource_pressure",
    "prefetch_regime_penalty",
    "aspect_distance_normalized",
)


def _unit_log_waste(value: float) -> float:
    # A logarithm preserves useful resolution for decode (where all BM values
    # over-cover M) without letting edge waste erase every other objective.
    return min(1.0, math.log2(1.0 + max(0.0, value)) / 12.0)


def shape_sort_components(config: KernelConfig, route: Route, cu_count: int) -> Dict[str, float]:
    """Return explainable route-aware objectives and their weighted aggregate.

    The route arithmetic intensity continuously blends 4- and 8-waves/CU
    hypotheses: compute-heavy routes favor four resident waves while decode and
    bandwidth-heavy routes favor eight. No individual objective is a hard gate.
    """

    gx, gy, grid = config.grid_for(route)
    covered_m = gy * config.bm
    covered_n = gx * config.bn
    useful = max(1, route.m * route.n)
    edge_waste = max(0.0, (covered_m * covered_n - useful) / float(useful))
    edge_normalized = _unit_log_waste(edge_waste)

    cus = max(1, int(cu_count))
    launched_waves = grid * config.nw
    waves_per_cu = launched_waves / float(cus)

    # GEMM-level roofline proxy. Output accumulation is conservatively treated
    # as two-byte traffic for both supported adapters.
    input_bytes = config.dtype_bytes * route.k * (route.m + route.n)
    output_bytes = 2 * route.m * route.n
    route_ai = (2.0 * route.m * route.n * route.k) / float(max(1, input_bytes + output_bytes))
    compute_weight = route_ai / (route_ai + 64.0)
    bandwidth_weight = 1.0 - compute_weight
    effective_target_waves_per_cu = 4.0 * compute_weight + 8.0 * bandwidth_weight

    def wave_distance(target: float) -> float:
        ratio = max(waves_per_cu, 1.0 / cus) / target
        return min(1.0, abs(math.log2(ratio)) / 3.0)

    distance_4 = wave_distance(4.0)
    distance_8 = wave_distance(8.0)
    wave_target_distance = compute_weight * distance_4 + bandwidth_weight * distance_8

    nearest_cu_round = max(1, int(round(grid / float(cus))))
    cta_tail_fraction = min(
        1.0,
        abs(grid - nearest_cu_round * cus) / float(nearest_cu_round * cus),
    )

    # Tile arithmetic intensity captures K reuse; LDS reads per MFMA captures
    # the orthogonal wave-tile reuse that favors balanced MFMA grids.
    tile_traffic = (
        config.dtype_bytes * config.bk * (config.bm + config.bn)
        + 2 * config.bm * config.bn
    )
    tile_ai = (2.0 * config.bm * config.bn * config.bk) / float(max(1, tile_traffic))
    reuse_penalty = 1.0 / (1.0 + tile_ai / 32.0)
    lds_reads_raw = 1.0 / config.miwt.m + 1.0 / config.miwt.n
    lds_read_per_mfma = min(1.0, lds_reads_raw / 2.0)

    aspect_distance = abs(math.log(
        (config.bm / float(config.bn)) /
        max(route.m / float(route.n), 1e-12)
    ))
    aspect_normalized = min(1.0, aspect_distance / 4.0)
    register_pressure = config.register_estimate.unified_vgpr / float(
        config.arch_spec.vgprs_per_simd
    )
    lds_pressure = config.lds.total_bytes / float(config.arch_spec.lds_bytes_per_cu)
    resource_pressure = min(1.0, 0.55 * register_pressure + 0.45 * lds_pressure)

    # PGR2 has an overlap benefit on compute-heavy routes and an explicit
    # pressure cost on bandwidth-heavy routes. Resource pressure remains a
    # separate objective, so PGR1 and PGR2 can both remain Pareto-relevant.
    prefetch_penalty = (
        compute_weight * float(max(0, 2 - config.pgr))
        + bandwidth_weight * float(max(0, config.pgr - 1))
    )

    weights = {
        "edge_waste_normalized": 0.16,
        "wave_target_distance": 0.20,
        "cta_tail_fraction": 0.10 + 0.08 * compute_weight,
        "reuse_penalty": 0.08 + 0.10 * compute_weight,
        "lds_read_per_mfma": 0.08 + 0.12 * compute_weight,
        "resource_pressure": 0.16 - 0.06 * compute_weight,
        "prefetch_regime_penalty": 0.12,
        "aspect_distance_normalized": 0.005,
    }
    objectives = {
        "edge_waste_normalized": edge_normalized,
        "wave_target_distance": wave_target_distance,
        "cta_tail_fraction": cta_tail_fraction,
        "reuse_penalty": reuse_penalty,
        "lds_read_per_mfma": lds_read_per_mfma,
        "resource_pressure": resource_pressure,
        "prefetch_regime_penalty": min(1.0, prefetch_penalty),
        "aspect_distance_normalized": aspect_normalized,
    }
    weight_total = sum(weights.values())
    weighted_score = sum(weights[name] * objectives[name] for name in SHAPE_OBJECTIVES) / weight_total

    return {
        # Raw/informational values are retained for receipts and diagnostics.
        "edge_waste": edge_waste,
        "grid_deficit": max(0.0, effective_target_waves_per_cu - waves_per_cu) / effective_target_waves_per_cu,
        "coverage_distance": wave_target_distance,
        "reuse_inverse": reuse_penalty,
        "k_reuse_inverse": 1.0 / float(min(max(1, route.k), config.bk)),
        "aspect_distance": aspect_distance,
        "register_pressure": register_pressure,
        "lds_pressure": lds_pressure,
        "route_arithmetic_intensity": route_ai,
        "compute_regime_weight": compute_weight,
        "effective_target_waves_per_cu": effective_target_waves_per_cu,
        "waves_per_cu": waves_per_cu,
        "distance_to_4_waves_per_cu": distance_4,
        "distance_to_8_waves_per_cu": distance_8,
        "tile_arithmetic_intensity": tile_ai,
        "lds_reads_per_mfma_raw": lds_reads_raw,
        **objectives,
        "weighted_score": weighted_score,
    }


def shape_objective_vector(config: KernelConfig, route: Route, cu_count: int) -> Tuple[float, ...]:
    scores = shape_sort_components(config, route, cu_count)
    return tuple(scores[name] for name in SHAPE_OBJECTIVES)


def shape_sort_key(config: KernelConfig, route: Route, cu_count: int) -> Tuple[Any, ...]:
    scores = shape_sort_components(config, route, cu_count)
    # Weighted aggregation is deliberate: this key no longer gives edge waste
    # veto power through a lexicographic first position.
    return (scores["weighted_score"], config.config_id)


def rank_for_route(configs: Sequence[KernelConfig], route: Route, cu_count: int) -> List[KernelConfig]:
    legal = route_static_prune(configs, route).accepted
    return sorted(legal, key=lambda config: shape_sort_key(config, route, cu_count))
