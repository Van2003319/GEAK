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


def shape_sort_components(config: KernelConfig, route: Route, cu_count: int) -> Dict[str, float]:
    """Return route-aware soft scores; lower is better for every component."""

    gx, gy, grid = config.grid_for(route)
    covered_m = gy * config.bm
    covered_n = gx * config.bn
    useful = max(1, route.m * route.n)
    edge_waste = (covered_m * covered_n - useful) / float(useful)
    cus = max(1, int(cu_count))
    launched_waves = grid * config.nw
    target_waves = cus * 8
    grid_deficit = max(0.0, target_waves - launched_waves) / float(target_waves)
    waves_per_cu = launched_waves / float(cus)
    coverage_distance = abs(math.log2(max(waves_per_cu, 1.0 / cus) / 8.0))
    # Larger M/N tiles reuse each K value across more FMAs. Keep this a score,
    # never a legality rule: decode routes often prefer the opposite trade-off.
    reuse_inverse = (config.bm + config.bn) / float(config.bm * config.bn)
    k_reuse_inverse = 1.0 / float(min(max(1, route.k), config.bk))
    aspect_distance = abs(math.log(
        (config.bm / float(config.bn)) /
        max(route.m / float(route.n), 1e-12)
    ))
    register_pressure = config.register_estimate.unified_vgpr / float(
        config.arch_spec.vgprs_per_simd
    )
    lds_pressure = config.lds.total_bytes / float(config.arch_spec.lds_bytes_per_cu)
    return {
        "edge_waste": edge_waste,
        "grid_deficit": grid_deficit,
        "coverage_distance": coverage_distance,
        "reuse_inverse": reuse_inverse,
        "k_reuse_inverse": k_reuse_inverse,
        "aspect_distance": aspect_distance,
        "register_pressure": register_pressure,
        "lds_pressure": lds_pressure,
    }


def shape_sort_key(config: KernelConfig, route: Route, cu_count: int) -> Tuple[Any, ...]:
    scores = shape_sort_components(config, route, cu_count)
    return (
        scores["edge_waste"],
        scores["grid_deficit"],
        scores["reuse_inverse"],
        scores["coverage_distance"],
        scores["k_reuse_inverse"],
        scores["aspect_distance"],
        scores["register_pressure"],
        scores["lds_pressure"],
        config.config_id,
    )


def rank_for_route(configs: Sequence[KernelConfig], route: Route, cu_count: int) -> List[KernelConfig]:
    legal = route_static_prune(configs, route).accepted
    return sorted(legal, key=lambda config: shape_sort_key(config, route, cu_count))
