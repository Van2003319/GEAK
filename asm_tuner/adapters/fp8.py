from __future__ import annotations

from typing import Sequence

from ..schema import DType, KernelConfig
from .base import TaskAdapter
from .source import render_rocwmma_source


class FP8RowwiseAdapter(TaskAdapter):
    name = "fp8_rowwise"
    dtype = DType.FP8_E4M3_FNUZ

    def render_source(self, configs: Sequence[KernelConfig]) -> str:
        self.validate_axis_variation(configs)
        return render_rocwmma_source(
            self.name,
            configs,
            input_type="rocwmma::float8_fnuz_t",
            mfma_k=32,
            scaled=True,
        )
