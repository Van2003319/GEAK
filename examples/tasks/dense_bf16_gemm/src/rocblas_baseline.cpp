#include "dense_bf16_gemm.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <torch/extension.h>

#include <memory>

namespace {

void check_rocblas(rocblas_status status, const char* operation) {
    TORCH_CHECK(status == rocblas_status_success, operation, " failed: ",
                rocblas_status_to_string(status));
}

struct HandleState {
    rocblas_handle handle = nullptr;
    int device = -1;

    ~HandleState() {
        if (handle != nullptr) {
            rocblas_destroy_handle(handle);
        }
    }

    rocblas_handle get(int requested_device, hipStream_t stream) {
        if (handle == nullptr || device != requested_device) {
            if (handle != nullptr) {
                check_rocblas(rocblas_destroy_handle(handle), "rocblas_destroy_handle");
                handle = nullptr;
            }
            check_rocblas(rocblas_create_handle(&handle), "rocblas_create_handle");
            check_rocblas(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host),
                          "rocblas_set_pointer_mode");
            device = requested_device;
        }
        check_rocblas(rocblas_set_stream(handle, stream), "rocblas_set_stream");
        return handle;
    }
};

thread_local HandleState handle_state;

}  // namespace

void rocblas_bf16_gemm_launcher(
    int m,
    int n,
    int k,
    const c10::BFloat16* a,
    const c10::BFloat16* b,
    c10::BFloat16* c,
    int device,
    hipStream_t stream) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto handle = handle_state.get(device, stream);

    // Row-major C[M,N] = A[M,K] * B[N,K]^T is column-major
    // C^T[N,M] = B[N,K] * A[M,K]^T. Row-major B is viewed as a
    // column-major KxN matrix, so transposing it yields NxK.
    check_rocblas(
        rocblas_gemm_ex(
            handle,
            rocblas_operation_transpose,
            rocblas_operation_none,
            n,
            m,
            k,
            &alpha,
            b,
            rocblas_datatype_bf16_r,
            k,
            a,
            rocblas_datatype_bf16_r,
            k,
            &beta,
            c,
            rocblas_datatype_bf16_r,
            n,
            c,
            rocblas_datatype_bf16_r,
            n,
            rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard,
            0,
            rocblas_gemm_flags_none),
        "rocblas_gemm_ex");
}
