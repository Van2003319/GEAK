from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

from .schema import ArchSpec, KernelConfig


class MetadataError(ValueError):
    pass


@dataclass(frozen=True)
class Occupancy:
    workgroups_per_cu: int
    waves_per_cu: int
    limiting_resources: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workgroups_per_cu": self.workgroups_per_cu,
            "waves_per_cu": self.waves_per_cu,
            "limiting_resources": list(self.limiting_resources),
        }


@dataclass(frozen=True)
class KernelMetadata:
    kernel: str
    config_id: str
    vgpr: int
    agpr: int
    unified_vgpr: int
    sgpr: int
    next_free_vgpr: int
    next_free_sgpr: int
    lds_bytes: int
    scratch_bytes_per_workitem: int
    accum_offset: int
    isa_hash: str
    instruction_count: int
    occupancy: Occupancy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernel": self.kernel,
            "config_id": self.config_id,
            "vgpr": self.vgpr,
            "agpr": self.agpr,
            "unified_vgpr": self.unified_vgpr,
            "sgpr": self.sgpr,
            "next_free_vgpr": self.next_free_vgpr,
            "next_free_sgpr": self.next_free_sgpr,
            "lds_bytes": self.lds_bytes,
            "scratch_bytes_per_workitem": self.scratch_bytes_per_workitem,
            "accum_offset": self.accum_offset,
            "isa_hash": self.isa_hash,
            "instruction_count": self.instruction_count,
            "occupancy": self.occupancy.to_dict(),
        }


def _ceil_multiple(value: int, granule: int) -> int:
    return ((value + granule - 1) // granule) * granule


def estimate_occupancy(config: KernelConfig, next_free_vgpr: int, sgpr: int, lds_bytes: int) -> Occupancy:
    spec: ArchSpec = config.arch_spec
    waves_per_workgroup = config.nw
    allocated_vgpr = max(8, _ceil_multiple(next_free_vgpr, 8))
    allocated_sgpr = max(16, _ceil_multiple(sgpr, 16))
    waves_from_vgpr = min(8, spec.vgprs_per_simd // allocated_vgpr) * spec.simds_per_cu
    waves_from_sgpr = min(8, spec.sgprs_per_simd // allocated_sgpr) * spec.simds_per_cu
    register_waves = min(spec.max_waves_per_cu, waves_from_vgpr, waves_from_sgpr)
    workgroups_by_waves = register_waves // waves_per_workgroup
    if lds_bytes:
        allocated_lds = _ceil_multiple(lds_bytes, 256)
        workgroups_by_lds = spec.lds_bytes_per_cu // allocated_lds
    else:
        workgroups_by_lds = spec.max_workgroups_per_cu
    limits = {
        "waves": workgroups_by_waves,
        "lds": workgroups_by_lds,
        "workgroup_slots": spec.max_workgroups_per_cu,
    }
    workgroups = min(limits.values())
    limiting = tuple(sorted(name for name, value in limits.items() if value == workgroups))
    return Occupancy(workgroups, workgroups * waves_per_workgroup, limiting)


def _integer(block: str, field: str) -> int:
    match = re.search(r"(?m)^\s*\.amdhsa_%s\s+([^\s;]+)" % re.escape(field), block)
    if not match:
        return 0
    token = match.group(1)
    try:
        return int(token, 0)
    except ValueError as exc:
        raise MetadataError("non-integer .amdhsa_%s value %r" % (field, token)) from exc


def _instruction_stream(assembly: str, kernel: str) -> List[str]:
    match = re.search(r"(?m)^%s:\s*(?:;.*)?$" % re.escape(kernel), assembly)
    if not match:
        raise MetadataError("assembly lacks body label for %s" % kernel)
    end_candidates = []
    for marker in ("\n.Lfunc_end", "\n\t.amdhsa_kernel", "\n.amdhsa_kernel"):
        pos = assembly.find(marker, match.end())
        if pos >= 0:
            end_candidates.append(pos)
    end = min(end_candidates) if end_candidates else len(assembly)
    normalized: List[str] = []
    for raw in assembly[match.end():end].splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or line.startswith((".", "#")) or line.endswith(":"):
            continue
        line = re.sub(r"\.LBB\d+_\d+", ".LBB", line)
        line = re.sub(r"\.Ltmp\d+", ".Ltmp", line)
        normalized.append(" ".join(line.split()))
    if not normalized:
        raise MetadataError("assembly body for %s has no instructions" % kernel)
    return normalized


def _set_integer(assembly: str, kernel: str, suffix: str, default: int) -> int:
    match = re.search(
        r"(?m)^\s*\.set\s+%s\.%s,\s*(\d+)\s*$" % (re.escape(kernel), re.escape(suffix)),
        assembly,
    )
    return int(match.group(1)) if match else default


def _yaml_integer(assembly: str, kernel: str, field: str, default: int) -> int:
    match = re.search(r"(?m)^\s*\.name:\s+%s\s*$" % re.escape(kernel), assembly)
    if not match:
        return default
    window = assembly[match.end():match.end() + 1600]
    value = re.search(r"(?m)^\s*\.%s:\s+(\d+)\s*$" % re.escape(field), window)
    return int(value.group(1)) if value else default


def parse_assembly(assembly: str, expected: Mapping[str, KernelConfig]) -> Dict[str, KernelMetadata]:
    blocks = {
        match.group(1): match.group(2)
        for match in re.finditer(
            r"(?ms)^\s*\.amdhsa_kernel\s+(\S+)\s*$\n(.*?)^\s*\.end_amdhsa_kernel\s*$",
            assembly,
        )
    }
    parsed: Dict[str, KernelMetadata] = {}
    missing = sorted(set(expected) - set(blocks))
    if missing:
        raise MetadataError("missing amdhsa metadata for: %s" % ", ".join(missing))
    for kernel, config in expected.items():
        block = blocks[kernel]
        next_vgpr = _integer(block, "next_free_vgpr")
        next_sgpr = _integer(block, "next_free_sgpr")
        accum_offset = _integer(block, "accum_offset")
        lds = _integer(block, "group_segment_fixed_size")
        scratch = _integer(block, "private_segment_fixed_size")
        fallback_vgpr = accum_offset if 0 < accum_offset < next_vgpr else next_vgpr
        fallback_agpr = max(0, next_vgpr - accum_offset) if accum_offset else 0
        vgpr = _set_integer(assembly, kernel, "num_vgpr", fallback_vgpr)
        agpr = _set_integer(assembly, kernel, "num_agpr", fallback_agpr)
        unified_vgpr = _yaml_integer(assembly, kernel, "vgpr_count", max(vgpr + agpr, next_vgpr))
        sgpr = _yaml_integer(assembly, kernel, "sgpr_count", next_sgpr)
        instructions = _instruction_stream(assembly, kernel)
        digest = hashlib.sha256("\n".join(instructions).encode("utf-8")).hexdigest()
        parsed[kernel] = KernelMetadata(
            kernel=kernel,
            config_id=config.config_id,
            vgpr=vgpr,
            agpr=agpr,
            unified_vgpr=unified_vgpr,
            sgpr=sgpr,
            next_free_vgpr=next_vgpr,
            next_free_sgpr=next_sgpr,
            lds_bytes=lds,
            scratch_bytes_per_workitem=scratch,
            accum_offset=accum_offset,
            isa_hash=digest,
            instruction_count=len(instructions),
            occupancy=estimate_occupancy(config, unified_vgpr, sgpr, lds),
        )
    return parsed
