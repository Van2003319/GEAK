#include "hipblaslt_baseline.h"

#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kWorkspaceLimit = 0;
constexpr int kHeuristicCount = 32;

void check_lt(hipblasStatus_t status, const char* operation) {
    TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS, operation,
                " failed with hipBLASLt status ", static_cast<int>(status));
}

struct Plan {
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a_layout = nullptr;
    hipblasLtMatrixLayout_t b_layout = nullptr;
    hipblasLtMatrixLayout_t c_layout = nullptr;
    hipblasLtMatrixLayout_t d_layout = nullptr;
    hipblasLtMatmulAlgo_t algo{};
    torch::Tensor workspace;
    HipblasltPlanInfo info;

    Plan(hipblasLtHandle_t handle, int m, int n, int k, int requested_algo,
         const torch::TensorOptions& options) {
        check_lt(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                 "hipblasLtMatmulDescCreate");
        // Column-major equivalent of row-major C[M,N] = A[M,K] * B[N,K]^T:
        // C^T[N,M] = B[N,K] * A[M,K]^T. Physical row-major B is column-major
        // [K,N], and physical row-major A is column-major [K,M].
        const hipblasOperation_t trans_a = HIPBLAS_OP_T;
        check_lt(hipblasLtMatmulDescSetAttribute(
                     op, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)),
                 "hipblasLtMatmulDescSetAttribute(TRANSA)");

        check_lt(hipblasLtMatrixLayoutCreate(&a_layout, HIP_R_16BF, k, n, k),
                 "hipblasLtMatrixLayoutCreate(B-as-A)");
        check_lt(hipblasLtMatrixLayoutCreate(&b_layout, HIP_R_16BF, k, m, k),
                 "hipblasLtMatrixLayoutCreate(A-as-B)");
        check_lt(hipblasLtMatrixLayoutCreate(&c_layout, HIP_R_16BF, n, m, n),
                 "hipblasLtMatrixLayoutCreate(C-transposed)");
        check_lt(hipblasLtMatrixLayoutCreate(&d_layout, HIP_R_16BF, n, m, n),
                 "hipblasLtMatrixLayoutCreate(D-transposed)");

        const float alpha = 1.0f;
        const float beta = 0.0f;
        std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
        if (requested_algo >= 0) {
            std::vector<int> indexes{requested_algo};
            check_lt(hipblaslt_ext::getAlgosFromIndex(handle, indexes, candidates),
                     "hipblaslt_ext::getAlgosFromIndex");
        } else {
            hipblasLtMatmulPreference_t preference = nullptr;
            check_lt(hipblasLtMatmulPreferenceCreate(&preference),
                     "hipblasLtMatmulPreferenceCreate");
            check_lt(hipblasLtMatmulPreferenceSetAttribute(
                         preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                         &kWorkspaceLimit, sizeof(kWorkspaceLimit)),
                     "hipblasLtMatmulPreferenceSetAttribute(MAX_WORKSPACE)");
            std::array<hipblasLtMatmulHeuristicResult_t, kHeuristicCount> heuristic{};
            int returned = 0;
            const auto status = hipblasLtMatmulAlgoGetHeuristic(
                handle, op, a_layout, b_layout, c_layout, d_layout, preference,
                kHeuristicCount, heuristic.data(), &returned);
            hipblasLtMatmulPreferenceDestroy(preference);
            check_lt(status, "hipblasLtMatmulAlgoGetHeuristic");
            candidates.assign(heuristic.begin(), heuristic.begin() + returned);
        }

        bool selected = false;
        for (auto candidate : candidates) {
            if (candidate.state != HIPBLAS_STATUS_SUCCESS) {
                continue;
            }
            std::size_t workspace_bytes = 0;
            auto candidate_algo = candidate.algo;
            const auto status = hipblaslt_ext::matmulIsAlgoSupported(
                handle, op, &alpha, a_layout, b_layout, &beta, c_layout, d_layout,
                candidate_algo, workspace_bytes);
            if (status != HIPBLAS_STATUS_SUCCESS || workspace_bytes > kWorkspaceLimit) {
                continue;
            }
            algo = candidate_algo;
            info.algo_index = hipblaslt_ext::getIndexFromAlgo(algo);
            if (requested_algo >= 0 && info.algo_index != requested_algo) {
                continue;
            }
            info.workspace_bytes = workspace_bytes;
            info.solution = hipblaslt_ext::getSolutionNameFromAlgo(handle, algo);
            info.kernel = hipblaslt_ext::getKernelNameFromAlgo(handle, algo);
            selected = true;
            break;
        }
        TORCH_CHECK(selected, "no supported hipBLASLt BF16 algorithm for shape ",
                    m, "x", n, "x", k, " and requested algo ", requested_algo);

        if (info.workspace_bytes != 0) {
            workspace = torch::empty(
                {static_cast<std::int64_t>(info.workspace_bytes)},
                options.dtype(torch::kUInt8));
        }
        check_lt(hipblasLtGetVersion(handle, &info.version), "hipblasLtGetVersion");
        char revision[128] = {};
        if (hipblasLtGetGitRevision(handle, revision) == HIPBLAS_STATUS_SUCCESS) {
            info.git_revision = revision;
        }
        char* arch = nullptr;
        if (hipblasLtGetArchName(&arch) == HIPBLAS_STATUS_SUCCESS && arch != nullptr) {
            info.arch = arch;
        }
    }

    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;

    ~Plan() {
        if (d_layout) hipblasLtMatrixLayoutDestroy(d_layout);
        if (c_layout) hipblasLtMatrixLayoutDestroy(c_layout);
        if (b_layout) hipblasLtMatrixLayoutDestroy(b_layout);
        if (a_layout) hipblasLtMatrixLayoutDestroy(a_layout);
        if (op) hipblasLtMatmulDescDestroy(op);
    }
};

using PlanKey = std::tuple<int, int, int, int>;

struct State {
    hipblasLtHandle_t handle = nullptr;
    int device = -1;
    std::map<PlanKey, std::unique_ptr<Plan>> plans;

    ~State() { reset(); }

    void reset() {
        plans.clear();
        if (handle != nullptr) {
            hipblasLtDestroy(handle);
            handle = nullptr;
        }
        device = -1;
    }

    void ensure_device(int requested_device) {
        if (handle != nullptr && device == requested_device) {
            return;
        }
        reset();
        check_lt(hipblasLtCreate(&handle), "hipblasLtCreate");
        device = requested_device;
    }

    Plan& plan(const torch::Tensor& a, const torch::Tensor& b, int requested_algo) {
        ensure_device(a.get_device());
        const auto key = PlanKey{static_cast<int>(a.size(0)), static_cast<int>(b.size(0)),
                                 static_cast<int>(a.size(1)), requested_algo};
        auto found = plans.find(key);
        if (found == plans.end()) {
            auto value = std::make_unique<Plan>(
                handle, std::get<0>(key), std::get<1>(key), std::get<2>(key),
                requested_algo, a.options());
            found = plans.emplace(key, std::move(value)).first;
        }
        return *found->second;
    }
};

thread_local State state;

}  // namespace

void hipblaslt_bf16_gemm_launcher(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& c,
    hipStream_t stream,
    int requested_algo_index) {
    auto& plan = state.plan(a, b, requested_algo_index);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    void* workspace = plan.workspace.defined() ? plan.workspace.data_ptr() : nullptr;
    check_lt(hipblasLtMatmul(
                 state.handle, plan.op, &alpha, b.data_ptr(), plan.a_layout,
                 a.data_ptr(), plan.b_layout, &beta, c.data_ptr(), plan.c_layout,
                 c.data_ptr(), plan.d_layout, &plan.algo, workspace,
                 plan.info.workspace_bytes, stream),
             "hipblasLtMatmul");
}

HipblasltPlanInfo hipblaslt_bf16_gemm_plan_info(
    const torch::Tensor& a,
    const torch::Tensor& b,
    int requested_algo_index) {
    return state.plan(a, b, requested_algo_index).info;
}
