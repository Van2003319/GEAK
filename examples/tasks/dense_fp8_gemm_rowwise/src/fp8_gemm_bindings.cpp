#include <ATen/hip/HIPContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>

#include <limits>

#include "dense_fp8_gemm.h"

namespace {

void check_inputs(const torch::Tensor& a, const torch::Tensor& b,
                  const torch::Tensor& a_scale, const torch::Tensor& b_scale) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "A and B must be GPU tensors");
    TORCH_CHECK(a.scalar_type() == at::kFloat8_e4m3fnuz &&
                    b.scalar_type() == at::kFloat8_e4m3fnuz,
                "A and B must be float8_e4m3fnuz tensors");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "A and B must be rank-2 tensors");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "A and B must be contiguous");
    TORCH_CHECK(a.device() == b.device(), "A and B must be on the same device");
    TORCH_CHECK(a.size(1) == b.size(1), "A and B must have matching K dimensions");
    TORCH_CHECK(a.size(0) > 0 && b.size(0) > 0 && a.size(1) > 0,
                "M, N, and K must be positive");
    TORCH_CHECK(a.size(0) <= std::numeric_limits<int>::max() &&
                    b.size(0) <= std::numeric_limits<int>::max() &&
                    a.size(1) <= std::numeric_limits<int>::max(),
                "M, N, and K must fit the candidate 32-bit interface");

    TORCH_CHECK(a_scale.is_cuda() && b_scale.is_cuda(), "scales must be GPU tensors");
    TORCH_CHECK(a_scale.scalar_type() == at::kFloat && b_scale.scalar_type() == at::kFloat,
                "scales must be float32 tensors");
    TORCH_CHECK(a_scale.is_contiguous() && b_scale.is_contiguous(),
                "scales must be contiguous");
    TORCH_CHECK(a_scale.device() == a.device() && b_scale.device() == a.device(),
                "scales must be on the same device as A and B");
    TORCH_CHECK(a_scale.numel() == a.size(0),
                "a_scale must hold exactly M elements (per-token)");
    TORCH_CHECK(b_scale.numel() == b.size(0),
                "b_scale must hold exactly N elements (per-channel)");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dense_fp8_gemm", [](const torch::Tensor& a, const torch::Tensor& b,
                               const torch::Tensor& a_scale, const torch::Tensor& b_scale) {
        check_inputs(a, b, a_scale, b_scale);
        c10::DeviceGuard guard(a.device());
        auto c = torch::empty({a.size(0), b.size(0)},
                              a.options().dtype(at::kBFloat16));
        const auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
        dense_fp8_gemm_launcher(
            static_cast<int>(a.size(0)), static_cast<int>(b.size(0)),
            static_cast<int>(a.size(1)),
            a.data_ptr<c10::Float8_e4m3fnuz>(),
            b.data_ptr<c10::Float8_e4m3fnuz>(),
            a_scale.data_ptr<float>(),
            b_scale.data_ptr<float>(),
            c.data_ptr<c10::BFloat16>(),
            a.get_device(), stream);
        return c;
    });
}
