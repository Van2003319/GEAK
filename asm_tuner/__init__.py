"""Typed, phase-independent AMD GEMM tuning primitives."""

from .schema import KernelConfig, Route, validate_config
from .schema_survival import get_fixture
from .search_space import generate_configs, materialize_search_space

__all__ = ["KernelConfig", "Route", "generate_configs", "get_fixture", "materialize_search_space", "validate_config"]
__version__ = "0.2.0"
