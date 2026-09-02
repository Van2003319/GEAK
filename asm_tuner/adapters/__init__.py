from __future__ import annotations

from typing import Dict

from .base import TaskAdapter
from .bf16 import BF16Adapter
from .fp8 import FP8RowwiseAdapter


_ADAPTERS: Dict[str, TaskAdapter] = {
    "bf16": BF16Adapter(),
    "fp8": FP8RowwiseAdapter(),
    "fp8_rowwise": FP8RowwiseAdapter(),
}


def get_adapter(name: str) -> TaskAdapter:
    try:
        return _ADAPTERS[name]
    except KeyError:
        raise KeyError("unknown adapter %r (expected bf16 or fp8_rowwise)" % name)


__all__ = ["BF16Adapter", "FP8RowwiseAdapter", "TaskAdapter", "get_adapter"]
