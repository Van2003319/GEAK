#pragma once

#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cstddef>
#include <string>

struct HipblasltPlanInfo {
    int algo_index = -1;
    std::size_t workspace_bytes = 0;
    std::string solution;
    std::string kernel;
    int version = 0;
    std::string git_revision;
    std::string arch;
};

void hipblaslt_bf16_gemm_launcher(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& c,
    hipStream_t stream,
    int requested_algo_index = -1);

HipblasltPlanInfo hipblaslt_bf16_gemm_plan_info(
    const torch::Tensor& a,
    const torch::Tensor& b,
    int requested_algo_index = -1);
