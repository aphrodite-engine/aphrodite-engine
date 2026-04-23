#include <cuda_fp16.h>
#include "exl3_gemv.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "exl3_gemv_kernel.cuh"
#include "exl3_kernel_map.cuh"
#include "exl3_devctx.cuh"
#include "hadamard.cuh"
//#include <set>

#include <ATen/Tensor.h>

#define K_SPLIT 1

/*
EXL3 matmul, A @ B -> C

- A: row-major A tensor, shape (m, k), dtype float16, contiguous
- B: EXL3-quantized B tensor, shape (k//16, n//16, 16*bits), dtype uint16
- C: empty row-major C tensor, shape (m, n), dtype float16 or float32, contiguous. Does not need to be zero-initialized
- suh: optional, packed input scales/flips, shape (k//16), dtype float16
- A_had: required if suh given, may be reference to A, temporary storage for input transform, size and dtype as A
- svh: optional, packed output scales/flips, shape (n//16), dtype float16

limitations:
- k % 16 == 0
- n % 128 == 0
*/

//std::set<void*> kernel_attr_set[MAX_DEVICES] = {};

void exl3_gemv
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    bool mcg,
    bool mul1
)
{
//    had_r_128(A, A_had.value(), suh, c10::nullopt, 1.0);

    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(B, 3);
    TORCH_CHECK_SHAPES(A, 1, B, 0, 16);
    TORCH_CHECK_SHAPES(C, -1, B, 1, 16);
    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kShort);
    bool c_fp32 = C.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);

    int size_m = A.size(0);
    int size_k = A.size(1);
    int size_n = B.size(1) * 16;

    TORCH_CHECK(size_m <= 8, "size_m must be <= 8");

//    dim3 blocks(size_k / TILESIZE_K, size_n / TILESIZE_N);


//    dim3 blocks(1, size_n / TILESIZE_N, K_SPLIT);
    dim3 blocks(1, size_n / TILESIZE_N, 1);
    dim3 threads(32, 1, TILESIZE_K / 16);

//    int cb = 0;
//    if (mcg) cb = 1;
//    if (mul1) cb = 2;

    const half* A_ptr = (const half*) A.data_ptr();
//    const half* A_ptr = (const half*) OPTPTR(A_had);
    const uint16_t* B_ptr = (const uint16_t*) B.data_ptr();
    void* C_ptr = (void*) C.data_ptr();

//    DBGI3(blocks.x, blocks.y, blocks.z);
//    DBGI3(threads.x, threads.y, threads.z);

    if (!c_fp32)
    {
//        DBGI3(size_m, size_k, size_n);
        exl3_gemv_kernel<4, false, 0, K_SPLIT><<<blocks, threads, 0, stream>>>
        (
            A_ptr,
            B_ptr,
            C_ptr,
            size_m,
            size_k,
            size_n
    //        (void*)& suh_ptr,
    //        (void*)& A_had_ptr,
    //        (void*)& svh_ptr,
        );
    }
    else
    {
//        DBGI3(size_m, size_k, size_n);
        exl3_gemv_kernel<4, true, 0, K_SPLIT><<<blocks, threads, 0, stream>>>
        (
            A_ptr,
            B_ptr,
            C_ptr,
            size_m,
            size_k,
            size_n
    //        (void*)& suh_ptr,
    //        (void*)& A_had_ptr,
    //        (void*)& svh_ptr,
        );
    }

    cuda_check(cudaPeekAtLastError());
}