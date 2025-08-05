#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Utility unions for efficient type conversion
union half2_uint32 {
    uint32_t as_uint32;
    half2 as_half2;
    __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
    __device__ half2_uint32(half2 val) : as_half2(val) {}
    __device__ half2_uint32() : as_uint32(0) {}
};

union half_uint16 {
    uint16_t as_uint16;
    half as_half;
    __device__ half_uint16(uint16_t val) : as_uint16(val) {}
    __device__ half_uint16(half val) : as_half(val) {}
    __device__ half_uint16() : as_uint16(0) {}
};

// "3INST" procedural codebook decoder (based on ExLlamaV3)
template <int cb>
__device__ inline half decode_3inst(uint32_t x, uint32_t mult)
{
    if constexpr (cb == 0)
    {
        x *= 89226354u;
        x += 64248484u;
        // Simplified version without inline assembly for compatibility
        half2_uint32 xu(x);
        // Extract high and low halves and add them
        uint32_t low_bits = x & 0xFFFF;
        uint32_t high_bits = (x >> 16) & 0xFFFF;
        // Convert to half precision values
        half low_half = __ushort_as_half((uint16_t)((low_bits & 0x8FFF) | 0x3B60));
        half high_half = __ushort_as_half((uint16_t)((high_bits & 0x8FFF) | 0x3B60));
        return __hadd(low_half, high_half);
    }
    if constexpr (cb == 1)
    {
        x *= mult;
        // Same simplified conversion for MCG mode
        uint32_t low_bits = x & 0xFFFF;
        uint32_t high_bits = (x >> 16) & 0xFFFF;
        half low_half = __ushort_as_half((uint16_t)((low_bits & 0x8FFF) | 0x3B60));
        half high_half = __ushort_as_half((uint16_t)((high_bits & 0x8FFF) | 0x3B60));
        return __hadd(low_half, high_half);
    }
    if constexpr (cb == 2)
    {
        x *= mult;
        // MUL1 mode with scaling
        const half k_inv_h = __ushort_as_half(0x1eee);  //  0.00677 = 1/147.7
        const half k_bias_h = __ushort_as_half(0xc931);  // -10.39
        half_uint16 h((uint16_t)(x + 0x6400u));
        return __hfma(h.as_half, k_inv_h, k_bias_h);
    }
}

// Determine which codebook mode to use based on multipliers
__device__ inline int get_cb_mode(uint32_t mcg_mult, uint32_t mul1_mult)
{
    if (mul1_mult != 0) return 2;  // MUL1 mode
    if (mcg_mult != 0) return 1;   // MCG mode
    return 0;                      // Default mode
}

// Get the appropriate multiplier for the mode
__device__ inline uint32_t get_mult(int cb, uint32_t mcg_mult, uint32_t mul1_mult)
{
    if (cb == 1) return mcg_mult;
    if (cb == 2) return mul1_mult;
    return 89226354u;  // Default multiplier
}

// Simplified EXL3 GEMM kernel that reconstructs weights on-the-fly
__global__ void exl3_gemm_kernel(
    const half* __restrict__ A,          // [batch_size, in_features]
    const int16_t* __restrict__ B,       // [tiles_k, tiles_n, K*16] trellis
    half* __restrict__ C,                // [batch_size, out_features]
    int batch_size,
    int in_features,
    int out_features,
    int tiles_k,
    int tiles_n,
    int K,
    uint32_t mcg_mult,
    uint32_t mul1_mult
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Determine codebook mode
    int cb = get_cb_mode(mcg_mult, mul1_mult);
    uint32_t mult = get_mult(cb, mcg_mult, mul1_mult);
    
    // Each block handles a 16x16 output tile
    int tile_row = by;
    int tile_col = bx;
    
    if (tile_row >= batch_size || tile_col >= (out_features + 15) / 16) return;
    
    // Each thread computes one output element
    int out_row = tile_row;
    int out_col = tile_col * 16 + tx;
    
    if (out_row >= batch_size || out_col >= out_features) return;
    
    float acc = 0.0f;
    
    // Iterate over input feature tiles
    for (int k_tile = 0; k_tile < tiles_k; k_tile++) {
        // Process 16 elements from the input
        for (int k_local = 0; k_local < 16; k_local++) {
            int k_global = k_tile * 16 + k_local;
            
            if (k_global >= in_features) break;
            
            // Get input value
            half a_val = A[out_row * in_features + k_global];
            
            // Reconstruct weight from trellis
            int trellis_tile_k = k_tile;
            int trellis_tile_n = tile_col;
            
            if (trellis_tile_n < tiles_n) {
                // Calculate trellis index
                int elem_idx = (k_local * 16 + tx) % (K * 16);
                
                // Get quantized value
                int16_t quant_val = B[trellis_tile_k * tiles_n * K * 16 + 
                                     trellis_tile_n * K * 16 + 
                                     elem_idx];
                
                // Decode using 3INST procedural codebook
                half b_val;
                uint32_t quant_u32 = static_cast<uint32_t>(static_cast<uint16_t>(quant_val));
                
                if (cb == 0) {
                    b_val = decode_3inst<0>(quant_u32, mult);
                } else if (cb == 1) {
                    b_val = decode_3inst<1>(quant_u32, mult);
                } else {
                    b_val = decode_3inst<2>(quant_u32, mult);
                }
                
                // Accumulate
                acc += __half2float(__hmul(a_val, b_val));
            }
        }
    }
    
    // Store result
    C[out_row * out_features + out_col] = __float2half(acc);
}

// Host functions matching ExLlamaV3 interface
torch::Tensor exl3_gemm(
    torch::Tensor input,     // [batch_size, in_features]
    torch::Tensor trellis,   // [tiles_k, tiles_n, K*16]
    torch::Tensor suh,       // [in_features] - input signs (optional)
    torch::Tensor svh,       // [out_features] - output signs (optional)
    int64_t mcg_mult,
    int64_t mul1_mult
) {
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    
    TORCH_CHECK(input.dtype() == torch::kFloat16, "Input must be float16");
    TORCH_CHECK(trellis.dtype() == torch::kInt16, "Trellis must be int16");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(trellis.dim() == 3, "Trellis must be 3D");
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int tiles_k = trellis.size(0);
    int tiles_n = trellis.size(1);
    int K_times_16 = trellis.size(2);
    int K = K_times_16 / 16;
    int out_features = tiles_n * 16;
    
    // Create output tensor
    torch::Tensor output = torch::zeros({batch_size, out_features}, 
                                       torch::TensorOptions()
                                       .dtype(torch::kFloat16)
                                       .device(input.device()));
    
    // Launch kernel
    dim3 threads(16, 1);
    dim3 blocks((out_features + 15) / 16, batch_size);
    
    exl3_gemm_kernel<<<blocks, threads>>>(
        input.data_ptr<half>(),
        trellis.data_ptr<int16_t>(),
        output.data_ptr<half>(),
        batch_size,
        in_features,
        out_features,
        tiles_k,
        tiles_n,
        K,
        static_cast<uint32_t>(mcg_mult),
        static_cast<uint32_t>(mul1_mult)
    );
    
    // Apply sign factors if provided
    if (suh.numel() > 0) {
        // Apply input signs by broadcasting multiplication
        // This is a simplified version - real implementation would use Hadamard transforms
        // For now, just apply the signs directly
        TORCH_CHECK(suh.size(0) == in_features, "suh size mismatch");
        // Note: This is incomplete - would need proper Hadamard transform
    }
    
    if (svh.numel() > 0) {
        // Apply output signs
        TORCH_CHECK(svh.size(0) == out_features, "svh size mismatch");
        output = output * svh.unsqueeze(0);
    }
    
    return output;
}

torch::Tensor exl3_reconstruct(
    torch::Tensor trellis,
    int64_t in_features,
    int64_t out_features,
    int64_t mcg_mult,
    int64_t mul1_mult
) {
    const at::cuda::OptionalCUDAGuard device_guard(trellis.device());
    
    TORCH_CHECK(trellis.dtype() == torch::kInt16, "Trellis must be int16");
    TORCH_CHECK(trellis.dim() == 3, "Trellis must be 3D");
    
    int tiles_k = trellis.size(0);
    int tiles_n = trellis.size(1);
    int K_times_16 = trellis.size(2);
    int K = K_times_16 / 16;
    
    // Create weight tensor
    torch::Tensor weight = torch::zeros({in_features, out_features}, 
                                       torch::TensorOptions()
                                       .dtype(torch::kFloat16)
                                       .device(trellis.device()));
    
    // Use the GEMM kernel with identity input to reconstruct weights
    torch::Tensor identity = torch::eye(in_features, 
                                       torch::TensorOptions()
                                       .dtype(torch::kFloat16)
                                       .device(trellis.device()));
    
    // Empty sign factors
    torch::Tensor empty_signs = torch::empty({0}, 
                                            torch::TensorOptions()
                                            .dtype(torch::kFloat16)
                                            .device(trellis.device()));
    
    weight = exl3_gemm(identity, trellis, empty_signs, empty_signs, mcg_mult, mul1_mult);
    
    return weight;
} 