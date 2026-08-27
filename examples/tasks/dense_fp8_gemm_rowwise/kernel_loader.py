import os

import torch
from torch.utils.cpp_extension import ROCM_HOME, load


_TASK_DIR = os.path.dirname(os.path.abspath(__file__))
_ROCM_HOME = ROCM_HOME or os.environ.get("ROCM_HOME", "/opt/rocm")


def _pin_offload_arch() -> str:
    """Build for THIS GPU only, and say so in the environment torch reads.

    This image exports PYTORCH_ROCM_ARCH as the full nine-target fan-out
    (gfx90a;gfx942;gfx950;gfx11xx;gfx12xx), and rocWMMA's fp8 16x16x32 fragment does not compile
    for all of them -- gfx950 fails an internal AccessT/StorageT size assertion, so the build
    dies on a target this box does not have. A multi-target value is therefore an image default
    rather than a request, and it is overridden. A value naming exactly ONE target is a
    deliberate choice and is honoured.

    Pinning also cuts every candidate rebuild during the run from nine device compiles to one,
    which over two days of iteration is the larger effect.
    """
    requested = os.environ.get("PYTORCH_ROCM_ARCH", "")
    targets = [t for t in requested.replace(",", ";").split(";") if t.strip()]
    if len(targets) == 1:
        return targets[0]
    detected = "gfx942"
    try:
        if torch.cuda.is_available():
            # gcnArchName looks like "gfx942:sramecc+:xnack-"; the feature suffix is not an
            # --offload-arch value.
            detected = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        pass
    os.environ["PYTORCH_ROCM_ARCH"] = detected
    return detected


_ARCH = _pin_offload_arch()


# This artifact contains only candidate-owned code and has no external math library linkage.
# rocWMMA is header-only and stays allowed: the fragment abstraction compiles down to
# v_mfma_f32_16x16x32_fp8_fp8 with nothing to link against. The immutable comparison
# implementation is not built here at all -- it is library Python, see oracle_wrapper.py.
gemm_ext = load(
    name="geak_dense_fp8_gemm_rowwise_candidate",
    sources=[
        os.path.join(_TASK_DIR, "src", "fp8_gemm_bindings.cpp"),
        os.path.join(_TASK_DIR, "src", "dense_fp8_gemm.hip"),
        os.path.join(_TASK_DIR, "src", "custom_fp8_gemm.hip"),
    ],
    extra_include_paths=[os.path.join(_ROCM_HOME, "include")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)
