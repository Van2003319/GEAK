from __future__ import annotations

from typing import Sequence

from ..schema import DType, KernelConfig
from .base import TaskAdapter
from .source import render_rocwmma_source


class BF16Adapter(TaskAdapter):
    name = "bf16"
    dtype = DType.BF16

    def render_source(self, configs: Sequence[KernelConfig]) -> str:
        self.validate_axis_variation(configs)
        return render_rocwmma_source(
            self.name,
            configs,
            input_type="rocwmma::bfloat16_t",
            mfma_k=16,
            scaled=False,
        )
