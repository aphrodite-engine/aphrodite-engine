#include <cuda_fp16.h>
#include "hgemm.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_devctx.cuh"

/*

Row-major matmul using cuBLAS, a @ b -> c
- if c is float16, operation is float16 @ float16 -> float16 (float16 accumulate)
- if c is float32, operation is float16 @ float16 -> float32 (float32 accumulate)
*/

using bfloat16 = __nv_bfloat16;

void hgemm_gr
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    bool output_fp32 = c.dtype() == at::kFloat;
    bool output_fp16 = c.dtype() == at::kHalf;

    TORCH_CHECK(output_fp32 || output_fp16, "c must be float32 or float16");

    // Check shapes of a,b,c are compatible
    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(b, kHalf);
    TORCH_CHECK_DIM(b, 2);
    TORCH_CHECK_SHAPES(a, -1, b, 0, 1);
    TORCH_CHECK_SHAPES(b, 1, c, -1, 1);

    const half* a_ptr = (const half*) a.data_ptr();
    const half* b_ptr = (const half*) b.data_ptr();

    int size_k = a.size(-1);
    int size_m = a.numel() / size_k;
    int size_n = b.size(-1);

    // Set cuBLAS modes and workspace
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    int device;
    cudaGetDevice(&device);
    void* ws = DevCtx::instance().get_ws(device);
    cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);

    if (output_fp16)
    {
        half alpha_ = __float2half(1.0f);
        half beta_ = __float2half(0.0f);

        half* c_ptr = (half*) c.data_ptr();
        auto r = cublasHgemm
        (
            cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            size_n, size_m, size_k,
            &alpha_, b_ptr, size_n,
                     a_ptr, size_k,
            &beta_,  c_ptr, size_n
        );
        cublas_check(r);
        cuda_check(cudaPeekAtLastError());
    }
    if (output_fp32)
    {
        float alpha_ = 1.0f;
        float beta_ = 0.0f;

        float* c_ptr = (float*) c.data_ptr();
        auto r = cublasGemmEx
        (
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            size_n, size_m, size_k,
            &alpha_, b_ptr, CUDA_R_16F, size_n,
                     a_ptr, CUDA_R_16F, size_k,
            &beta_,  c_ptr, CUDA_R_32F, size_n,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
        cublas_check(r);
        cuda_check(cudaPeekAtLastError());
    }

    if (graph) graph->need_cublas = true;
}

void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c
)
{
    hgemm_gr(a, b, c, nullptr);
}