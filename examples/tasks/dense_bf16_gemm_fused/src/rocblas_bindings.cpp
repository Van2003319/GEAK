#include <ATen/hip/HIPContext.h>
#include <c10/core/DeviceGuard.h>
#include <rocblas/rocblas.h>
#include <torch/extension.h>

#include <limits>
#include <string>
#include <vector>

#include "hipblaslt_baseline.h"
#include "rocblas_baseline.h"

namespace {

void check_inputs(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "A and B must be GPU tensors");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16,
                "A and B must be BF16 tensors");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "A and B must be rank-2 tensors");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "A and B must be contiguous");
    TORCH_CHECK(a.device() == b.device(), "A and B must be on the same device");
    TORCH_CHECK(a.size(1) == b.size(1), "A and B must have matching K dimensions");
    TORCH_CHECK(a.size(0) > 0 && b.size(0) > 0 && a.size(1) > 0,
                "M, N, and K must be positive");
    TORCH_CHECK(a.size(0) <= std::numeric_limits<int>::max() &&
                    b.size(0) <= std::numeric_limits<int>::max() &&
                    a.size(1) <= std::numeric_limits<int>::max(),
                "M, N, and K must fit the BLAS 32-bit interfaces");
}

torch::Tensor rocblas_gemm(const torch::Tensor& a, const torch::Tensor& b) {
    check_inputs(a, b);
    c10::DeviceGuard guard(a.device());
    auto c = torch::empty({a.size(0), b.size(0)}, a.options());
    const auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    rocblas_bf16_gemm_launcher(
        static_cast<int>(a.size(0)), static_cast<int>(b.size(0)),
        static_cast<int>(a.size(1)), a.data_ptr<c10::BFloat16>(),
        b.data_ptr<c10::BFloat16>(), c.data_ptr<c10::BFloat16>(),
        a.get_device(), stream);
    return c;
}

torch::Tensor hipblaslt_gemm(const torch::Tensor& a, const torch::Tensor& b,
                             int algo_index) {
    check_inputs(a, b);
    c10::DeviceGuard guard(a.device());
    // Deliberately identical output shape/dtype/layout/allocation API to rocblas_gemm and candidate.
    auto c = torch::empty({a.size(0), b.size(0)}, a.options());
    const auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    hipblaslt_bf16_gemm_launcher(a, b, c, stream, algo_index);
    return c;
}

pybind11::dict hipblaslt_info(const torch::Tensor& a, const torch::Tensor& b,
                              int algo_index) {
    check_inputs(a, b);
    c10::DeviceGuard guard(a.device());
    const auto info = hipblaslt_bf16_gemm_plan_info(a, b, algo_index);
    pybind11::dict out;
    out["backend"] = "direct_hipblaslt";
    out["algo_index"] = info.algo_index;
    out["solution"] = info.solution;
    out["kernel"] = info.kernel;
    out["version"] = info.version;
    out["git_revision"] = info.git_revision;
    out["arch"] = info.arch;
    out["workspace_bytes"] = info.workspace_bytes;
    out["workspace_limit_bytes"] = 0;
    out["layout"] = "row_major_A_MK_x_row_major_B_NK_transpose_to_row_major_C_MN";
    out["output_dtype"] = "bfloat16";
    out["allocation"] = "fresh_torch_empty_per_call; plan/workspace cached outside timed calls";
    return out;
}

pybind11::dict rocblas_info() {
    std::string version = "unknown";
    std::size_t size = 0;
    if (rocblas_get_version_string_size(&size) == rocblas_status_success && size > 0) {
        std::vector<char> buffer(size);
        if (rocblas_get_version_string(buffer.data(), buffer.size()) == rocblas_status_success) {
            version = buffer.data();
        }
    }
    pybind11::dict out;
    out["backend"] = "direct_rocblas";
    out["algo_index"] = 0;
    out["solution"] = "rocblas_gemm_algo_standard";
    out["kernel"] = "runtime_selected_not_exposed_by_rocblas_gemm_ex";
    out["version"] = version;
    out["workspace_bytes"] = 0;
    out["layout"] = "row_major_A_MK_x_row_major_B_NK_transpose_to_row_major_C_MN";
    out["output_dtype"] = "bfloat16";
    out["allocation"] = "fresh_torch_empty_per_call";
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rocblas_bf16_gemm", &rocblas_gemm);
    m.def("hipblaslt_bf16_gemm", &hipblaslt_gemm,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("algo_index") = -1);
    m.def("hipblaslt_bf16_gemm_info", &hipblaslt_info,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("algo_index") = -1);
    m.def("rocblas_bf16_gemm_info", &rocblas_info);
}
