"""Typed, phase-independent AMD GEMM tuning primitives."""

from .schema import KernelConfig, Route, validate_config
from .schema_survival import get_fixture

__all__ = ["KernelConfig", "Route", "get_fixture", "validate_config"]
__version__ = "0.1.0"
