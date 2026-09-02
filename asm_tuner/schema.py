from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SCHEMA_VERSION = "geak.asm_config/v1"


class ConfigError(ValueError):
    """Raised when a config cannot be represented or is statically illegal."""


class Arch(str, Enum):
    GFX90A = "gfx90a"
    GFX942 = "gfx942"
    GFX950 = "gfx950"


class DType(str, Enum):
    BF16 = "bf16"
    FP8_E4M3_FNUZ = "fp8_e4m3_fnuz"


class Layout(str, Enum):
    """Logical task layouts, independent of Tensile's raw free-index order."""

    A_MK_B_NK = "a_mk_b_nk"
    A_MK_B_KN = "a_mk_b_kn"


class StoreMode(str, Enum):
    VECTOR = "vector"
    NON_TEMPORAL = "non_temporal"


@dataclass(frozen=True)
class ArchSpec:
    wave_size: int
    max_threads_per_workgroup: int
    max_waves_per_workgroup: int
    lds_bytes_per_cu: int
    max_waves_per_cu: int
    max_workgroups_per_cu: int
    simds_per_cu: int
    vgprs_per_simd: int
    sgprs_per_simd: int


ARCH_SPECS: Mapping[Arch, ArchSpec] = {
    Arch.GFX90A: ArchSpec(64, 1024, 16, 65536, 32, 16, 4, 512, 800),
    Arch.GFX942: ArchSpec(64, 1024, 16, 65536, 32, 16, 4, 512, 800),
    Arch.GFX950: ArchSpec(64, 1024, 16, 65536, 32, 16, 4, 512, 800),
}


DTYPE_BYTES: Mapping[DType, int] = {
    DType.BF16: 2,
    DType.FP8_E4M3_FNUZ: 1,
}


@dataclass(frozen=True)
class Mfma:
    m: int
    n: int
    k: int
    blocks: int = 1

    @classmethod
    def from_obj(cls, value: Any) -> "Mfma":
        if isinstance(value, Mfma):
            return value
        if isinstance(value, str):
            parts = value.lower().replace("mfma", "").strip("_x").split("x")
            if len(parts) not in (3, 4):
                raise ConfigError("mfma string must be MxNxK or MxNxKxB")
            nums = [int(x) for x in parts]
            return cls(*nums) if len(nums) == 4 else cls(*nums, 1)
        if isinstance(value, Mapping):
            return cls(int(value["m"]), int(value["n"]), int(value["k"]), int(value.get("blocks", 1)))
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            nums = [int(x) for x in value]
            if len(nums) == 3:
                nums.append(1)
            if len(nums) != 4:
                raise ConfigError("mfma sequence must have 3 or 4 entries")
            return cls(*nums)
        raise ConfigError("mfma must be a mapping, sequence, or MxNxK string")

    def to_dict(self) -> Dict[str, int]:
        return {"m": self.m, "n": self.n, "k": self.k, "blocks": self.blocks}


@dataclass(frozen=True)
class Tile2D:
    m: int
    n: int

    def to_dict(self) -> Dict[str, int]:
        return {"m": self.m, "n": self.n}


@dataclass(frozen=True)
class WaveGrid(Tile2D):
    @classmethod
    def from_obj(cls, value: Any) -> "WaveGrid":
        if isinstance(value, WaveGrid):
            return value
        if isinstance(value, Mapping):
            return cls(int(value["m"]), int(value["n"]))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) == 2:
            return cls(int(value[0]), int(value[1]))
        raise ConfigError("wave_grid must be {m,n} or [m,n]")


@dataclass(frozen=True)
class LdsOperand:
    buffers: int = 1
    pad_elements: int = 0

    @classmethod
    def from_obj(cls, value: Any) -> "LdsOperand":
        if isinstance(value, LdsOperand):
            return value
        if value is None:
            return cls()
        if isinstance(value, Mapping):
            return cls(int(value.get("buffers", 1)), int(value.get("pad_elements", value.get("pad", 0))))
        raise ConfigError("LDS operand must be a mapping")

    def to_dict(self) -> Dict[str, int]:
        return {"buffers": self.buffers, "pad_elements": self.pad_elements}


@dataclass(frozen=True)
class VectorWidths:
    a: int
    b: int
    store: int

    @classmethod
    def from_obj(cls, value: Any, dtype: DType) -> "VectorWidths":
        if isinstance(value, VectorWidths):
            return value
        default_load = 8 if dtype is DType.BF16 else 16
        if value is None:
            return cls(default_load, default_load, 4)
        if isinstance(value, int):
            return cls(value, value, value)
        if isinstance(value, Mapping):
            return cls(int(value.get("a", default_load)), int(value.get("b", default_load)), int(value.get("store", 4)))
        raise ConfigError("vector_widths must be an integer or mapping")

    def to_dict(self) -> Dict[str, int]:
        return {"a": self.a, "b": self.b, "store": self.store}


@dataclass(frozen=True)
class Store:
    mode: StoreMode = StoreMode.VECTOR

    @classmethod
    def from_obj(cls, value: Any) -> "Store":
        if isinstance(value, Store):
            return value
        if value is None:
            return cls()
        if isinstance(value, str):
            return cls(StoreMode(value))
        if isinstance(value, Mapping):
            return cls(StoreMode(value.get("mode", StoreMode.VECTOR.value)))
        raise ConfigError("store must be a string or mapping")

    def to_dict(self) -> Dict[str, str]:
        return {"mode": self.mode.value}


@dataclass(frozen=True)
class LdsEstimate:
    a_bytes: int
    b_bytes: int
    scratch_bytes: int
    total_bytes: int
    pgr_counted_as_buffer: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "a_bytes": self.a_bytes,
            "b_bytes": self.b_bytes,
            "scratch_bytes": self.scratch_bytes,
            "total_bytes": self.total_bytes,
            "pgr_counted_as_buffer": self.pgr_counted_as_buffer,
        }


@dataclass(frozen=True)
class RegisterEstimate:
    regular_vgpr: int
    accumulator_vgpr: int
    unified_vgpr: int
    pgr_register_vgpr: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "regular_vgpr": self.regular_vgpr,
            "accumulator_vgpr": self.accumulator_vgpr,
            "unified_vgpr": self.unified_vgpr,
            "pgr_register_vgpr": self.pgr_register_vgpr,
        }


@dataclass(frozen=True)
class Route:
    name: str
    m: int
    n: int
    k: int
    weight: float = 1.0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Route":
        return cls(str(value["name"]), int(value["m"]), int(value["n"]), int(value["k"]), float(value.get("weight", 1.0)))

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "m": self.m, "n": self.n, "k": self.k, "weight": self.weight}


@dataclass(frozen=True)
class KernelConfig:
    arch: Arch
    dtype: DType
    layout: Layout
    mfma: Mfma
    bm: int
    bn: int
    bk: int
    wave_grid: WaveGrid
    pgr: int
    plr: int
    a_lds: LdsOperand
    b_lds: LdsOperand
    vector_widths: VectorWidths
    gsu: int = 1
    stream_k: int = 0
    wgm: int = 1
    xcc: int = 1
    stagger: int = 0
    store: Store = field(default_factory=Store)
    lds_scratch_bytes: int = 0
    provenance: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False, repr=False)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KernelConfig":
        schema = value.get("schema")
        if schema is not None and schema != SCHEMA_VERSION:
            raise ConfigError("unsupported config schema %r" % schema)
        dtype = DType(value.get("dtype", value.get("DType")))
        a_lds = value.get("a_lds")
        b_lds = value.get("b_lds")
        if a_lds is None:
            a_lds = {"buffers": value.get("a_buffers", 1), "pad": value.get("a_pad", 0)}
        if b_lds is None:
            b_lds = {"buffers": value.get("b_buffers", 1), "pad": value.get("b_pad", 0)}
        return cls(
            arch=Arch(value.get("arch", value.get("Arch"))),
            dtype=dtype,
            layout=Layout(value.get("layout", Layout.A_MK_B_NK.value)),
            mfma=Mfma.from_obj(value.get("mfma", value.get("MFMA"))),
            bm=int(value.get("bm", value.get("BM"))),
            bn=int(value.get("bn", value.get("BN"))),
            bk=int(value.get("bk", value.get("BK"))),
            wave_grid=WaveGrid.from_obj(value.get("wave_grid", value.get("WaveGrid"))),
            pgr=int(value.get("pgr", value.get("PGR", 1))),
            plr=int(value.get("plr", value.get("PLR", 0))),
            a_lds=LdsOperand.from_obj(a_lds),
            b_lds=LdsOperand.from_obj(b_lds),
            vector_widths=VectorWidths.from_obj(value.get("vector_widths", value.get("vector_width")), dtype),
            gsu=int(value.get("gsu", value.get("GSU", 1))),
            stream_k=int(value.get("stream_k", value.get("StreamK", 0))),
            wgm=int(value.get("wgm", value.get("WGM", 1))),
            xcc=int(value.get("xcc", value.get("XCC", 1))),
            stagger=int(value.get("stagger", value.get("Stagger", 0))),
            store=Store.from_obj(value.get("store")),
            lds_scratch_bytes=int(value.get("lds_scratch_bytes", value.get("scratch_bytes", 0))),
            provenance=dict(value.get("provenance", {})),
        )

    @property
    def arch_spec(self) -> ArchSpec:
        return ARCH_SPECS[self.arch]

    @property
    def dtype_bytes(self) -> int:
        return DTYPE_BYTES[self.dtype]

    @property
    def nw(self) -> int:
        return self.wave_grid.m * self.wave_grid.n

    @property
    def wave_tile(self) -> Tile2D:
        if self.wave_grid.m <= 0 or self.wave_grid.n <= 0:
            raise ConfigError("wave-grid dimensions must be positive")
        if self.bm % self.wave_grid.m or self.bn % self.wave_grid.n:
            raise ConfigError("macro tile must divide exactly across the wave grid")
        return Tile2D(self.bm // self.wave_grid.m, self.bn // self.wave_grid.n)

    @property
    def miwt(self) -> Tile2D:
        wt = self.wave_tile
        if self.mfma.m <= 0 or self.mfma.n <= 0 or wt.m % self.mfma.m or wt.n % self.mfma.n:
            raise ConfigError("WaveTile must be an integer grid of MFMA tiles")
        return Tile2D(wt.m // self.mfma.m, wt.n // self.mfma.n)

    @property
    def required_epilogue_scratch_bytes(self) -> int:
        return self.nw * self.mfma.m * self.mfma.n * 4

    @property
    def lds(self) -> LdsEstimate:
        # PGR is a scheduling distance. It is intentionally absent here: it is
        # not an LDS-buffer count and must never multiply either operand.
        a_bytes = self.a_lds.buffers * self.bm * (self.bk + self.a_lds.pad_elements) * self.dtype_bytes
        b_bytes = self.b_lds.buffers * self.bn * (self.bk + self.b_lds.pad_elements) * self.dtype_bytes
        total = a_bytes + b_bytes + self.lds_scratch_bytes
        return LdsEstimate(a_bytes, b_bytes, self.lds_scratch_bytes, total, False)

    @property
    def register_estimate(self) -> RegisterEstimate:
        """Source-level register-pressure estimate used before compilation.

        PGR2 holds every per-thread global-load pack for the next K stage in
        ordinary VGPRs while the current LDS stage is consumed. Accumulators
        remain a separate AGPR allocation on CDNA. Compiler metadata is still
        the authoritative post-compile gate.
        """

        nt = max(1, self.nw * self.arch_spec.wave_size)
        a_width = max(1, self.vector_widths.a)
        b_width = max(1, self.vector_widths.b)
        a_chunks = self.bm * ((max(1, self.bk) + a_width - 1) // a_width)
        b_chunks = self.bn * ((max(1, self.bk) + b_width - 1) // b_width)
        a_packs = (a_chunks + nt - 1) // nt
        b_packs = (b_chunks + nt - 1) // nt
        a_pack_vgpr = (a_width * self.dtype_bytes + 3) // 4
        b_pack_vgpr = (b_width * self.dtype_bytes + 3) // 4
        if self.pgr == 2:
            pgr_vgpr = a_packs * a_pack_vgpr + b_packs * b_pack_vgpr
        else:
            pgr_vgpr = max(a_pack_vgpr, b_pack_vgpr)
        # Addressing, fragment descriptors, scales, and epilogue temporaries.
        regular = 32 + pgr_vgpr
        miwt = self.miwt
        accumulator = miwt.m * miwt.n * 4
        return RegisterEstimate(regular, accumulator, regular + accumulator, pgr_vgpr)

    def grid_for(self, route: Route) -> Tuple[int, int, int]:
        gx = (route.n + self.bn - 1) // self.bn
        gy = (route.m + self.bm - 1) // self.bm
        return gx, gy, gx * gy * max(self.gsu, 1)

    def to_dict(self, include_derived: bool = False, include_provenance: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "arch": self.arch.value,
            "dtype": self.dtype.value,
            "layout": self.layout.value,
            "mfma": self.mfma.to_dict(),
            "bm": self.bm,
            "bn": self.bn,
            "bk": self.bk,
            "wave_grid": self.wave_grid.to_dict(),
            "pgr": self.pgr,
            "plr": self.plr,
            "a_lds": self.a_lds.to_dict(),
            "b_lds": self.b_lds.to_dict(),
            "vector_widths": self.vector_widths.to_dict(),
            "gsu": self.gsu,
            "stream_k": self.stream_k,
            "wgm": self.wgm,
            "xcc": self.xcc,
            "stagger": self.stagger,
            "store": self.store.to_dict(),
            "lds_scratch_bytes": self.lds_scratch_bytes,
        }
        if include_derived:
            try:
                out["derived"] = {
                    "wave_tile": self.wave_tile.to_dict(),
                    "miwt": self.miwt.to_dict(),
                    "nw": self.nw,
                    "lds": self.lds.to_dict(),
                    "register_estimate": self.register_estimate.to_dict(),
                }
            except ConfigError as exc:
                out["derived_error"] = str(exc)
        if include_provenance and self.provenance:
            out["provenance"] = dict(self.provenance)
        return out

    @property
    def config_id(self) -> str:
        canonical = self.to_dict(include_derived=False, include_provenance=False)
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:20]


@dataclass(frozen=True)
class LegalityReport:
    config_id: str
    valid: bool
    errors: Tuple[str, ...]
    lds: LdsEstimate

    def require_valid(self) -> None:
        if not self.valid:
            raise ConfigError("; ".join(self.errors))

    def to_dict(self) -> Dict[str, Any]:
        return {"config_id": self.config_id, "valid": self.valid, "errors": list(self.errors), "lds": self.lds.to_dict()}


def _power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def validate_config(config: KernelConfig) -> LegalityReport:
    errors: List[str] = []
    spec = config.arch_spec
    for name, value in (("BM", config.bm), ("BN", config.bn), ("BK", config.bk)):
        if value <= 0:
            errors.append("%s must be positive" % name)
    for name, value in (("MFMA.M", config.mfma.m), ("MFMA.N", config.mfma.n), ("MFMA.K", config.mfma.k), ("MFMA.blocks", config.mfma.blocks)):
        if value <= 0:
            errors.append("%s must be positive" % name)

    if config.layout is not Layout.A_MK_B_NK:
        errors.append("MVP LDS model supports only logical A[M,K], B[N,K] layout")
    expected_k = 16 if config.dtype is DType.BF16 else 32
    if config.dtype is DType.FP8_E4M3_FNUZ and config.arch is not Arch.GFX942:
        errors.append("fp8_e4m3_fnuz rocWMMA execution is supported only on gfx942")
    if (config.mfma.m, config.mfma.n, config.mfma.k, config.mfma.blocks) != (16, 16, expected_k, 1):
        errors.append("%s requires MFMA 16x16x%dx1 in the MVP adapters" % (config.dtype.value, expected_k))
    if config.bk > 0 and config.mfma.k > 0 and config.bk % config.mfma.k:
        errors.append("BK must be divisible by MFMA.K")

    if config.wave_grid.m <= 0 or config.wave_grid.n <= 0:
        errors.append("wave_grid dimensions must be positive")
    if config.nw > spec.max_waves_per_workgroup:
        errors.append("wave_grid exceeds %d waves/workgroup" % spec.max_waves_per_workgroup)
    if config.nw * spec.wave_size > spec.max_threads_per_workgroup:
        errors.append("workgroup exceeds %d threads" % spec.max_threads_per_workgroup)
    try:
        config.miwt
    except ConfigError as exc:
        errors.append(str(exc))

    if config.pgr not in (1, 2):
        errors.append("PGR must be 1 (synchronous) or 2 (one-stage global-register prefetch)")
    if config.plr < 0:
        errors.append("PLR must be >= 0")
    for name, operand in (("A", config.a_lds), ("B", config.b_lds)):
        if operand.buffers < 1:
            errors.append("%s LDS buffers must be >= 1" % name)
        if operand.pad_elements < 0:
            errors.append("%s LDS pad must be >= 0" % name)

    for name, width in (("A", config.vector_widths.a), ("B", config.vector_widths.b)):
        if not _power_of_two(width) or width * config.dtype_bytes > 16:
            errors.append("%s vector width must be power-of-two and at most 16 bytes" % name)
        elif config.bk > 0 and config.bk % width:
            errors.append("BK must be divisible by %s vector width" % name)
        operand = config.a_lds if name == "A" else config.b_lds
        if width > 0 and (config.bk + operand.pad_elements) % width:
            errors.append("%s LDS row stride must be divisible by its vector width" % name)
    if not _power_of_two(config.vector_widths.store) or config.vector_widths.store > config.mfma.n:
        errors.append("store vector width must be a power-of-two no wider than MFMA.N")
    elif config.mfma.n % config.vector_widths.store:
        errors.append("MFMA.N must be divisible by store vector width")

    # A, B, and scratch are independent regions in one aligned allocation.
    # Requiring component sizes to be aligned means the arithmetic sum is exact.
    for name, size in (("A", config.lds.a_bytes), ("B", config.lds.b_bytes), ("scratch", config.lds.scratch_bytes)):
        if size % 16:
            errors.append("%s LDS footprint must be 16-byte aligned" % name)
    if config.lds.total_bytes > spec.lds_bytes_per_cu:
        errors.append("LDS footprint %d exceeds %d bytes" % (config.lds.total_bytes, spec.lds_bytes_per_cu))
    if config.lds_scratch_bytes < 0:
        errors.append("LDS scratch must be non-negative")
    elif config.lds_scratch_bytes < config.required_epilogue_scratch_bytes:
        errors.append(
            "LDS scratch must reserve at least %d bytes for the per-wave epilogue"
            % config.required_epilogue_scratch_bytes
        )

    try:
        registers = config.register_estimate
    except ConfigError:
        registers = None
    if registers is not None:
        if registers.regular_vgpr > 256:
            errors.append(
                "estimated regular VGPR demand %d exceeds the architectural 256-VGPR wave limit"
                % registers.regular_vgpr
            )
        if registers.accumulator_vgpr > 256:
            errors.append(
                "estimated accumulator VGPR demand %d exceeds the architectural 256-AGPR wave limit"
                % registers.accumulator_vgpr
            )
        if registers.unified_vgpr > config.arch_spec.vgprs_per_simd:
            errors.append(
                "estimated unified VGPR demand %d exceeds %d registers/SIMD"
                % (registers.unified_vgpr, config.arch_spec.vgprs_per_simd)
            )

    if config.gsu != 1:
        errors.append("MVP executable supports only GSU=1")
    if config.stream_k != 0:
        errors.append("MVP executable supports only StreamK=0")
    if config.wgm < 1 or config.xcc < 1:
        errors.append("WGM and XCC must be >= 1")
    if config.stagger < 0:
        errors.append("stagger must be >= 0")

    return LegalityReport(config.config_id, not errors, tuple(errors), config.lds)


def load_configs(values: Iterable[Mapping[str, Any]]) -> List[KernelConfig]:
    return [KernelConfig.from_dict(value) for value in values]
