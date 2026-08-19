#include <ATen/hip/HIPContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>

#include <limits>

#include "dense_bf16_gemm.h"

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
                "M, N, and K must fit the rocBLAS 32-bit interface");
}

torch::Tensor run_gemm(const torch::Tensor& a, const torch::Tensor& b, bool baseline) {
    check_inputs(a, b);
    c10::DeviceGuard guard(a.device());
    auto c = torch::empty({a.size(0), b.size(0)}, a.options());
    const auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    const int m = static_cast<int>(a.size(0));
    const int n = static_cast<int>(b.size(0));
    const int k = static_cast<int>(a.size(1));

    if (baseline) {
        rocblas_bf16_gemm_launcher(
            m, n, k, a.data_ptr<c10::BFloat16>(), b.data_ptr<c10::BFloat16>(),
            c.data_ptr<c10::BFloat16>(), a.get_device(), stream);
    } else {
        dense_bf16_gemm_launcher(
            m, n, k, a.data_ptr<c10::BFloat16>(), b.data_ptr<c10::BFloat16>(),
            c.data_ptr<c10::BFloat16>(), a.get_device(), stream);
    }
    return c;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dense_bf16_gemm", [](const torch::Tensor& a, const torch::Tensor& b) {
        return run_gemm(a, b, false);
    });
    m.def("rocblas_bf16_gemm", [](const torch::Tensor& a, const torch::Tensor& b) {
        return run_gemm(a, b, true);
    });
}
