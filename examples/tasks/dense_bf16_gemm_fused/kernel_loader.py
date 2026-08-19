import os

from torch.utils.cpp_extension import ROCM_HOME, load


_TASK_DIR = os.path.dirname(os.path.abspath(__file__))
_ROCM_HOME = ROCM_HOME or os.environ.get("ROCM_HOME", "/opt/rocm")


# This artifact contains only candidate-owned code and has no external math
# library linkage. The immutable comparison implementation is built separately.
gemm_ext = load(
    name="geak_dense_bf16_gemm_fused_candidate",
    sources=[
        os.path.join(_TASK_DIR, "src", "gemm_bindings.cpp"),
        os.path.join(_TASK_DIR, "src", "dense_bf16_gemm.hip"),
        os.path.join(_TASK_DIR, "src", "custom_gemm.hip"),
    ],
    extra_include_paths=[os.path.join(_ROCM_HOME, "include")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)
