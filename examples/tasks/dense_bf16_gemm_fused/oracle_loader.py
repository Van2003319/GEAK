import os

from torch.utils.cpp_extension import ROCM_HOME, load


_TASK_DIR = os.path.dirname(os.path.abspath(__file__))
_ROCM_HOME = ROCM_HOME or os.environ.get("ROCM_HOME", "/opt/rocm")


# Frozen denominator/oracle exception. Candidate policy scans must exempt these
# sources explicitly and must never exempt the candidate extension artifact.
oracle_ext = load(
    name="geak_dense_bf16_gemm_fused_oracle",
    sources=[
        os.path.join(_TASK_DIR, "src", "rocblas_bindings.cpp"),
        os.path.join(_TASK_DIR, "src", "rocblas_baseline.cpp"),
    ],
    extra_include_paths=[os.path.join(_ROCM_HOME, "include")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=[f"-L{os.path.join(_ROCM_HOME, 'lib')}", "-lrocblas"],
    verbose=True,
)
