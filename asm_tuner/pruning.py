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


def shape_sort_key(config: KernelConfig, route: Route, cu_count: int) -> Tuple[float, float, float, int, str]:
    """Rank legal shapes without turning grid/CU heuristics into hard gates."""

    gx, gy, grid = config.grid_for(route)
    covered_m = gy * config.bm
    covered_n = gx * config.bn
    useful = max(1, route.m * route.n)
    edge_waste = (covered_m * covered_n - useful) / float(useful)
    cus = max(1, cu_count)
    waves_per_cu = grid * config.nw / float(cus)
    coverage_distance = abs(math.log2(max(waves_per_cu, 1.0 / cus)))
    aspect_distance = abs(math.log((config.bm / float(config.bn)) / max(route.m / float(route.n), 1e-12)))
    return (edge_waste, coverage_distance, aspect_distance, config.lds.total_bytes, config.config_id)


def rank_for_route(configs: Sequence[KernelConfig], route: Route, cu_count: int) -> List[KernelConfig]:
    legal = static_prune(configs).accepted
    return sorted(legal, key=lambda config: shape_sort_key(config, route, cu_count))
