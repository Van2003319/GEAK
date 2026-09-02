from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from ..schema import ConfigError, DType, KernelConfig, Layout, Route, validate_config


class AdapterError(RuntimeError):
    pass


class InactiveAxisError(AdapterError):
    pass


class AxisState(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class AxisCapability:
    state: AxisState
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return {"state": self.state.value, "reason": self.reason}


@dataclass(frozen=True)
class Measurement:
    samples_us: Tuple[float, ...]
    median_us: float
    correctness: Mapping[str, Any]
    timer: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "samples_us": list(self.samples_us),
            "median_us": self.median_us,
            "correctness": dict(self.correctness),
            "timer": dict(self.timer),
        }


def _axis_value(config: KernelConfig, axis: str) -> Any:
    if axis == "wave_grid":
        return config.wave_grid
    if axis == "a_lds":
        return config.a_lds
    if axis == "b_lds":
        return config.b_lds
    if axis == "vector_widths":
        return config.vector_widths
    if axis == "store":
        return config.store
    return getattr(config, axis)


class TaskAdapter(ABC):
    name: str
    dtype: DType
    source_version = "rocwmma-mvp-v1"

    axis_capabilities: Mapping[str, AxisCapability] = {
        "bm": AxisCapability(AxisState.ACTIVE, "compile-time macro-tile M"),
        "bn": AxisCapability(AxisState.ACTIVE, "compile-time macro-tile N"),
        "bk": AxisCapability(AxisState.ACTIVE, "compile-time K stage"),
        "wave_grid": AxisCapability(AxisState.ACTIVE, "compile-time 2-D wave ownership"),
        "a_lds": AxisCapability(AxisState.ACTIVE, "independent A slots and row pad"),
        "b_lds": AxisCapability(AxisState.ACTIVE, "independent B slots and row pad"),
        "vector_widths": AxisCapability(AxisState.ACTIVE, "typed global loads and output stores"),
        "wgm": AxisCapability(AxisState.ACTIVE, "bijective 1-D workgroup remapping"),
        "stagger": AxisCapability(AxisState.ACTIVE, "rotates the semantic K-stage order"),
        "store": AxisCapability(AxisState.ACTIVE, "vector or non-temporal epilogue"),
        "pgr": AxisCapability(
            AxisState.INACTIVE,
            "global-read prefetch distance is recorded but this MVP body is synchronous",
        ),
        "plr": AxisCapability(
            AxisState.INACTIVE,
            "local-read prefetch distance is recorded but this MVP body is synchronous",
        ),
        "xcc": AxisCapability(
            AxisState.INACTIVE,
            "XCC-affine remapping is not implemented; do not search it",
        ),
        "gsu": AxisCapability(
            AxisState.UNSUPPORTED,
            "workspace reduction is not implemented; only GSU=1 is executable",
        ),
        "stream_k": AxisCapability(
            AxisState.UNSUPPORTED,
            "StreamK counters/workspace are not implemented; only StreamK=0 is executable",
        ),
    }

    def validate(self, config: KernelConfig) -> None:
        report = validate_config(config)
        report.require_valid()
        if config.dtype is not self.dtype:
            raise ConfigError("%s adapter requires %s" % (self.name, self.dtype.value))
        if config.layout is not Layout.A_MK_B_NK:
            raise ConfigError("%s adapter currently requires A[M,K], B[N,K]" % self.name)
        if config.lds_scratch_bytes < config.required_epilogue_scratch_bytes:
            raise ConfigError(
                "LDS scratch %d is smaller than the %d-byte per-wave epilogue requirement"
                % (config.lds_scratch_bytes, config.required_epilogue_scratch_bytes)
            )
        if config.gsu != 1:
            raise ConfigError("adapter executable boundary is GSU=1")
        if config.stream_k != 0:
            raise ConfigError("adapter executable boundary is StreamK=0")

    def validate_axis_variation(self, configs: Sequence[KernelConfig]) -> None:
        for config in configs:
            self.validate(config)
        for axis, capability in self.axis_capabilities.items():
            values = {_axis_value(config, axis) for config in configs}
            if len(values) <= 1 or capability.state is AxisState.ACTIVE:
                continue
            raise InactiveAxisError(
                "%s varies %s, but that axis is %s: %s"
                % (self.name, axis, capability.state.value, capability.reason)
            )

    def axis_status(self) -> Dict[str, Dict[str, str]]:
        return {name: capability.to_dict() for name, capability in sorted(self.axis_capabilities.items())}

    def kernel_name(self, config: KernelConfig) -> str:
        return "geak_%s_%s" % (self.name, config.config_id)

    def launch_name(self, config: KernelConfig) -> str:
        return "geak_launch_%s_%s" % (self.name, config.config_id)

    @abstractmethod
    def render_source(self, configs: Sequence[KernelConfig]) -> str:
        raise NotImplementedError

    def measure(
        self,
        shared_object: Path,
        config: KernelConfig,
        route: Route,
        samples: int,
        warmups: int,
        device: int,
        check: bool,
    ) -> Measurement:
        from .torch_runtime import measure_torch

        self.validate(config)
        return measure_torch(self, shared_object, config, route, samples, warmups, device, check)
