#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 700
  #include <mma.h>
#endif

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/DeviceGuard.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

template <typename U, typename V>
constexpr __host__ __device__ auto divDown(U a, V b) -> decltype(a + b) {
  static_assert(std::is_integral<U>::value && std::is_integral<V>::value, "");
  return (a / b);
}

template <typename U, typename V>
constexpr __host__ __device__ auto divUp(U a, V b) -> decltype(a + b) {
  static_assert(std::is_integral<U>::value && std::is_integral<V>::value, "");
  // Overflow safe variant of (a + b - 1) / b
  const uint64_t blocks = a / b + (a % b != 0);
  return blocks;
}

template <typename U, typename V>
constexpr __host__ __device__ auto roundDown(U a, V b) -> decltype(a + b) {
  static_assert(std::is_integral<U>::value && std::is_integral<V>::value, "");
  return divDown(a, b) * b;
}

template <typename U, typename V>
constexpr __host__ __device__ auto roundUp(U a, V b) -> decltype(a + b) {
  static_assert(std::is_integral<U>::value && std::is_integral<V>::value, "");
  return divUp(a, b) * b;
}

[[maybe_unused]] constexpr int32_t kWarpSize = 32;
[[maybe_unused]] constexpr int32_t KTilesPerWarp = 8;
[[maybe_unused]] constexpr int32_t kMTileSize = 16;
[[maybe_unused]] constexpr int32_t kNTileSize = 8;
[[maybe_unused]] constexpr int32_t kKTileSize = 16;

struct __align__(16) f16x2x4_u32 { uint32_t vals[4]; };
struct __align__(16) f16x2x2_u32 { uint32_t vals[2]; };

struct ALayout_RM {
  template <int KTilesToLoad>
  static __device__ void load(const half* A, int32_t m, int32_t k,
                              int32_t mTiles, int32_t mTile, int32_t kTiles,
                              int32_t kTileStart, int32_t laneId,
                              f16x2x4_u32 out[KTilesToLoad]) {
    const auto mLane = mTile * kMTileSize + (laneId / 4);
    const auto kLane = kTileStart * kKTileSize + (laneId % 4) * 4;

    // access
    // [mTile * kMTileSize + (laneId / 4)]
    // [kTileStart * kKTileSize + (laneId % 4) * 2]
    auto aPtr = A + mLane * k + kLane;

    auto aPtrPlus8Rows = aPtr + 8 * k;

    bool m0InBounds = mLane < m;
    bool m1InBounds = (mLane + 8) < m;

#pragma unroll
    for (int i = 0; i < KTilesToLoad; ++i) {
      out[i].vals[0] =
          m0InBounds ? *reinterpret_cast<const uint32_t*>(aPtr + i * kKTileSize)
                     : uint32_t(0);
      out[i].vals[1] = m1InBounds ? *reinterpret_cast<const uint32_t*>(
                                        aPtrPlus8Rows + i * kKTileSize)
                                  : uint32_t(0);

      out[i].vals[2] =
          m0InBounds
              ? *reinterpret_cast<const uint32_t*>(aPtr + i * kKTileSize + 2)
              : uint32_t(0);
      out[i].vals[3] = m1InBounds ? *reinterpret_cast<const uint32_t*>(
                                        aPtrPlus8Rows + i * kKTileSize + 2)
                                  : uint32_t(0);
    }
  }

  static __device__ void store(half* C, int32_t m, int32_t n, int32_t mOutTiles,
                               int32_t mTile, int32_t nOutTiles, int32_t nTile,
                               int32_t laneId, const float4& out) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800

    // sum.x / sum.y are written at
    // [laneId / 4], [(laneId % 4) * 2, (laneId % 4) * 2 + 1]
    // sum.z / sum.w are written at
    // [8 + (laneId / 4)], [(laneId % 4) * 2, (laneId % 4) * 2 + 1]
    // i.e., same columns, different row.
    const int outRow = mTile * kMTileSize + (laneId / 4);
    const int outCol = nTile * kNTileSize + (laneId % 4) * 2;

    // Pointer where sum.x / sum.y is written
    auto cPtr = C + outRow * n + outCol;

    auto v01 = __float22half2_rn(float2{out.x, out.y});
    auto v23 = __float22half2_rn(float2{out.z, out.w});

    if (outRow < m) {
      *reinterpret_cast<half2*>(cPtr) = v01;
    }

    // sum.z, sum.w at +8 rows from cPtr
    if (outRow + 8 < m) {
      *reinterpret_cast<half2*>(cPtr + 8 * n) = v23;
    }
#endif
  }
};

struct BLayout_D4 {
  static constexpr bool use_codebook = true;

  template <int KTilesPerIteration>
  static __device__ void load(const void* __restrict__ B,
                              const uint64_t* __restrict__ CB, int32_t n,
                              int32_t k, int32_t nTiles, int32_t nTile,
                              int32_t kTiles, int32_t kTileStart,
                              int32_t laneId,
                              f16x2x2_u32 b[KTilesPerIteration]) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800
    auto Bptr = reinterpret_cast<const uint8_t*>(B);
  #pragma unroll
    for (int i = 0; i < KTilesPerIteration; ++i) {
      const int row = nTile * kNTileSize + laneId / 4;
      const int col = (kTileStart + i) * kKTileSize / 4 + laneId % 4;
      *(reinterpret_cast<uint64_t*>(b[i].vals)) = CB[Bptr[row * k / 4 + col]];
    }
#endif
  }
};

struct BLayout_HI {
  static constexpr bool use_codebook = false;

  template <int KTilesPerIteration>
  static __device__ void load(const void* __restrict__ B,
                              const uint64_t* __restrict__ CB, int32_t n,
                              int32_t k, int32_t nTiles, int32_t nTile,
                              int32_t kTiles, int32_t kTileStart,
                              int32_t laneId,
                              f16x2x2_u32 b[KTilesPerIteration]) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800
    auto Bptr = reinterpret_cast<const uint32_t*>(B);
  #pragma unroll
    for (int i = 0; i < KTilesPerIteration; ++i) {
      const int row = nTile * kNTileSize + laneId / 4;
      const int col = (kTileStart + i) * kKTileSize / 8 + (laneId % 4) / 2;
      // simply use code - 7.5 instead of reading codebook
      uint32_t code = Bptr[row * k / 8 + col];

      const uint32_t c0 = 0x64086408;
      const half y16_ = __float2half_rn(1.0f / 16.0f);
      const half2 y16 = __halves2half2(y16_, y16_);
      const half z16_ = __float2half_rn(-1024.0f / 16.0f - 8.0f);
      const half2 z16 = __halves2half2(z16_, z16_);

      uint32_t qa = code >> ((laneId & 1) * 8);
      uint32_t q0 = (((qa & 0x000f000f) << 4) | c0);
      uint32_t q1 = ((qa & 0x00f000f0) | c0);
      *(half2*)(b[i].vals) = __hfma2(*((half2*)(&q0)), y16, z16);
      *(half2*)(b[i].vals + 1) = __hfma2(*((half2*)(&q1)), y16, z16);
    }
#endif
  }
};

struct BLayout_E8 {
  static constexpr bool use_codebook = true;

  __device__ static inline uint64_t decode8weights(
      uint16_t weight_compressed, const int64_t* __restrict__ codebook_abs) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800

    uint8_t bits_sign = weight_compressed & 0xff;
    uint8_t parity = __popc(bits_sign) & 1;
    uint8_t sign_vec = bits_sign ^ parity;
    uint8_t bits_abs = (weight_compressed >> 8);
    int64_t packed = codebook_abs[bits_abs];

    uint64_t decoded_sign = sign_vec * 0x8040201008040201ll;
    decoded_sign &= 0x8080808080808080;
    decoded_sign >>= 7;
    decoded_sign *= 255 - 3;
    packed ^= decoded_sign;
    packed |= 0x0101010101010101;
    packed -= parity * 0x0202020202020202;

    return packed;
#else
    return 0;
#endif
  }

  __device__ static inline uint32_t decode8weights(
      uint16_t weight_compressed, const int64_t* __restrict__ codebook_abs,
      int idx) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800
    uint8_t bits_sign =
        weight_compressed & 0xff;  //__brev(weight_compressed) >> 24;
    const uint32_t magic_nums[2] = {0x08040201ll, 0x80402010ll};
    uint8_t parity = __popc(bits_sign) & 1;
    uint8_t sign_vec = bits_sign ^ parity;  // (parity << 7);
    uint16_t bits_abs = (weight_compressed >> 8);
    uint32_t packed = ((uint32_t*)codebook_abs)[(bits_abs << 1) + idx];
    uint32_t magic_num = magic_nums[idx];
    uint32_t decoded_sign = sign_vec * magic_num;
    decoded_sign &= 0x80808080;
    decoded_sign >>= 7;
    decoded_sign *= 255 - 3;
    packed ^= decoded_sign;
    packed |= 0x01010101;
    packed -= parity * 0x02020202;
    return packed;
#else
    return 0;
#endif
  };

  template <int KTilesPerIteration>
  static __device__ void load(const void* __restrict__ B,
                              const uint64_t* __restrict__ CB, int32_t n,
                              int32_t k, int32_t nTiles, int32_t nTile,
                              int32_t kTiles, int32_t kTileStart,
                              int32_t laneId,
                              f16x2x2_u32 b[KTilesPerIteration]) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800
    auto Bptr = (const uint16_t*)B;
  #pragma unroll
    for (int i = 0; i < KTilesPerIteration; ++i) {
      const int row = nTile * kNTileSize + laneId / 4;
      const int col = (kTileStart + i) * kKTileSize / 8 + laneId % 4 / 2;
      uint32_t decoded = decode8weights(Bptr[row * k / 8 + col],
                                        (const int64_t*)CB, laneId & 1);
      half2 unpacked[2];
      uint32_t lower_half = decoded & 0x00ff00ff;
      lower_half = (lower_half ^ 0x5c805c80);
      memcpy(unpacked, &lower_half, sizeof(uint32_t));
      uint32_t upper_half = (decoded & 0xff00ff00) >> 8;
      upper_half = (upper_half ^ 0x5c805c80);
      memcpy(unpacked + 1, &upper_half, sizeof(uint32_t));

      const half adjust_ = __float2half_rn(-288.0f);
      const half2 adjust = __halves2half2(adjust_, adjust_);
      unpacked[0] = __hadd2(unpacked[0], adjust);
      unpacked[1] = __hadd2(unpacked[1], adjust);
      *(reinterpret_cast<uint64_t*>(b[i].vals)) =
          *(reinterpret_cast<uint64_t*>(unpacked));
      //*((half*)(b[i].vals)) = unpacked[0];
      //*((half*)(b[i].vals) + 1) = unpacked[0].y;
      //*((half*)(b[i].vals) + 2) = unpacked[1].x;
      //*((half*)(b[i].vals) + 3) = unpacked[1].y;
    }
#endif
  }
};

template <typename ALayout, typename BLayout, typename CLayout, int Warps,
          int KTilesPerIteration>
__global__ __launch_bounds__(256) void tinygemm_m16n8k16_chunk_kernel(
    // Data for the A matrix, loaded as per ALayout
    const half* __restrict__ A, const void* __restrict__ B,
    const uint64_t* __restrict__ CB,

    // Output data for the C matrix, stored as per CLayout
    half* __restrict__ C,

    // The size of the matrix multiplication
    int32_t m, int32_t n, int32_t k,

    // The size of the matrix multiplication, in multiples of our TC tile size
    int32_t mTiles, int32_t nTiles, int32_t kTiles) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800
  __shared__ uint64_t CB_[256];
  if (BLayout::use_codebook) {
    CB_[threadIdx.x + threadIdx.y * 32] = CB[threadIdx.x + threadIdx.y * 32];
    __syncthreads();
  }

  auto warpId = threadIdx.y;
  auto laneId = threadIdx.x;

  int32_t mTile = blockIdx.z;
  int32_t nTile = blockIdx.y;

  float4 c{0.0f, 0.0f, 0.0f, 0.0f};

  // First, handle whole multiples of KTilesPerIteration
  auto kTilesLimit = roundDown(kTiles, KTilesPerIteration);

  // Each warp handles a set of KTilesPerIteration under the above limit
  for (int32_t kTileBase = warpId * KTilesPerIteration; kTileBase < kTilesLimit;
       kTileBase += Warps * KTilesPerIteration) {
    //
    // Load data from A
    //
    f16x2x4_u32 a[KTilesPerIteration];
    ALayout::template load<KTilesPerIteration>(A, m, k, mTiles, mTile, kTiles,
                                               kTileBase, laneId, a);

    //
    // Load data from B and de-quantize as needed
    //
    f16x2x2_u32 b[KTilesPerIteration];
    BLayout::template load<KTilesPerIteration>(B, CB_, n, k, nTiles, nTile,
                                               kTiles, kTileBase, laneId, b);

  // Now, perform the matrix multiplication
  //
  #pragma unroll
    for (int i = 0; i < KTilesPerIteration / 2; ++i) {
      float4 cTmp[2];

  #pragma unroll
      for (int k = 0; k < 2; ++k) {
        cTmp[k] = float4{0.0f, 0.0f, 0.0f, 0.0f};
      }

  #pragma unroll
      for (int k = 0; k < 2; ++k) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=f"(cTmp[k].x), "=f"(cTmp[k].y), "=f"(cTmp[k].z), "=f"(cTmp[k].w)
            : "r"(a[i * 2 + k].vals[0]), "r"(a[i * 2 + k].vals[1]),
              "r"(a[i * 2 + k].vals[2]), "r"(a[i * 2 + k].vals[3]),
              "r"(b[i * 2 + k].vals[0]), "r"(b[i * 2 + k].vals[1]),
              "f"(cTmp[k].x), "f"(cTmp[k].y), "f"(cTmp[k].z), "f"(cTmp[k].w));
      }
  #pragma unroll
      for (int k = 0; k < 2; ++k) {
        c.x += cTmp[k].x;
        c.y += cTmp[k].y;
        c.z += cTmp[k].z;
        c.w += cTmp[k].w;
      }
    }

  }  // for all tiles under kTilesLimit

  auto kTileBaseRemaining = kTilesLimit + warpId;

  // If we have any remainder k-tiles, some warps will handle them, processing
  // kInnerKTiles k-tiles at a time
  if (kTileBaseRemaining < kTiles) {
    f16x2x4_u32 a;
    ALayout::template load<1>(A, m, k, mTiles, mTile, kTiles,
                              kTileBaseRemaining, laneId, &a);

    f16x2x2_u32 b;
    BLayout::template load<1>(B, CB, n, k, nTiles, nTile, kTiles,
                              kTileBaseRemaining, laneId, &b);

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(c.x), "=f"(c.y), "=f"(c.z), "=f"(c.w)
        : "r"(a.vals[0]), "r"(a.vals[1]), "r"(a.vals[2]), "r"(a.vals[3]),
          "r"(b.vals[0]), "r"(b.vals[1]), "f"(c.x), "f"(c.y), "f"(c.z),
          "f"(c.w));
  }
  // Reduce independent k-tiles (same m/n) across warps
  __shared__ float4 smem_sum[Warps][kWarpSize];

  smem_sum[warpId][laneId] = c;

  __syncthreads();

  if (warpId == 0) {
    float4 sum_f32{0.0f, 0.0f, 0.0f, 0.0f};

    // Reduce across the block in the first warp
    for (int i = 0; i < Warps; ++i) {
      float4 v = smem_sum[i][laneId];
      sum_f32.x += v.x;
      sum_f32.y += v.y;
      sum_f32.z += v.z;
      sum_f32.w += v.w;
    }

    // Write the reduced result (in the first warp) into the output
    CLayout::store(C, m, n, mTiles, mTile,
                   // n for C output becomes k for A input, so for m16n8k16,
                   // we need to halve the tiles
                   nTiles / 2, nTile, laneId, sum_f32);
  }
#endif
}

at::Tensor d4_mm_origorder(const at::Tensor& A, const at::Tensor& B,
                           const at::Tensor& CB) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800
  c10::cuda::CUDAGuard g(A.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  constexpr int Warps = 8;

  // row major layout
  auto m = A.size(0);
  auto mTiles = divUp(m, kMTileSize);

  // tensor core layout
  auto n = B.size(0);
  auto nTiles = divUp(n, kNTileSize);

  // row major layout
  auto k = A.size(1);
  auto kTiles = divUp(k, kKTileSize);

  // Output is a standard row-major matrix
  auto C_final = at::empty(
      {m, n}, at::TensorOptions().dtype(A.dtype()).device(A.device()));

  auto grid = dim3(1, nTiles, mTiles);
  auto block = dim3(kWarpSize, Warps);
  auto kernel =
      tinygemm_m16n8k16_chunk_kernel<ALayout_RM, BLayout_D4, ALayout_RM, 8, 8>;

  kernel<<<grid, block, 0, stream>>>(
      (const half*)A.data_ptr(), (const void*)B.data_ptr(),
      (const uint64_t*)CB.data_ptr(), (half*)C_final.data_ptr(), m, n, k,
      mTiles, nTiles, kTiles);

  return C_final;
#endif
  return {};  // Squash missing return statement warning
}

at::Tensor e8p_mm_origorder(const at::Tensor& A, const at::Tensor& B,
                            const at::Tensor& CB) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800
  c10::cuda::CUDAGuard g(A.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  constexpr int Warps = 8;

  // row major layout
  auto m = A.size(0);
  auto mTiles = divUp(m, kMTileSize);

  // tensor core layout
  auto n = B.size(0);
  auto nTiles = divUp(n, kNTileSize);

  // row major layout
  auto k = A.size(1);
  auto kTiles = divUp(k, kKTileSize);

  // Output is a standard row-major matrix
  auto C_final = at::empty(
      {m, n}, at::TensorOptions().dtype(A.dtype()).device(A.device()));

  auto grid = dim3(1, nTiles, mTiles);
  auto block = dim3(kWarpSize, Warps);
  auto kernel =
      tinygemm_m16n8k16_chunk_kernel<ALayout_RM, BLayout_E8, ALayout_RM, 8, 8>;
  kernel<<<grid, block, 0, stream>>>(
      (const half*)A.data_ptr(), (const void*)B.data_ptr(),
      (const uint64_t*)CB.data_ptr(), (half*)C_final.data_ptr(), m, n, k,
      mTiles, nTiles, kTiles);

  return C_final;
#endif
  return {};  // Squash missing return statement warning
}

at::Tensor hi_mm_origorder(const at::Tensor& A, const at::Tensor& B) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800
  c10::cuda::CUDAGuard g(A.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  constexpr int Warps = 8;

  // row major layout
  auto m = A.size(0);
  auto mTiles = divUp(m, kMTileSize);

  // tensor core layout
  auto n = B.size(0);
  auto nTiles = divUp(n, kNTileSize);

  // row major layout
  auto k = A.size(1);
  auto kTiles = divUp(k, kKTileSize);

  // Output is a standard row-major matrix
  auto C_final = at::empty(
      {m, n}, at::TensorOptions().dtype(A.dtype()).device(A.device()));

  auto grid = dim3(1, nTiles, mTiles);
  auto block = dim3(kWarpSize, Warps);
  auto kernel =
      tinygemm_m16n8k16_chunk_kernel<ALayout_RM, BLayout_HI, ALayout_RM, 8, 8>;
  kernel<<<grid, block, 0, stream>>>(
      (const half*)A.data_ptr(), (const void*)B.data_ptr(), nullptr,
      (half*)C_final.data_ptr(), m, n, k, mTiles, nTiles, kTiles);

  return C_final;
#endif
  return {};  // Squash missing return statement warning
}

#define DECOMPRESS_D4_BLOCK_SIZE 256

__global__ void cuda_decompress_d4_origorder_kernel(
    const uint8_t* __restrict__ YIs,   // m x (n/4)
    const c10::Half* __restrict__ CB,  // 256 x 4
    c10::Half* __restrict__ Y          // m x n
) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800
  const long i = threadIdx.x + DECOMPRESS_D4_BLOCK_SIZE * blockIdx.x;

  for (long r = 0; r < 4; r++) {
    uint8_t yidx = ((uint8_t*)YIs)[i * 4 + r];
    ((uint64_t*)Y)[i * 4 + r] = ((uint64_t*)CB)[yidx & 255];
  }
#endif
}

void decompress_d4_origorder(torch::Tensor YIs,  // m x (n/4)
                             torch::Tensor CB,   // 256 x 4
                             torch::Tensor Y     // m x n
) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800
  size_t m = Y.sizes()[0];
  size_t n = Y.sizes()[1];

  assert(YIs.is_contiguous());
  assert(CB.is_contiguous());
  assert(Y.is_contiguous());

  assert(YIs.sizes()[0] == m);
  assert(YIs.sizes()[1] * 4 == n);
  assert(CB.sizes()[0] == 256);

  const dim3 threads(DECOMPRESS_D4_BLOCK_SIZE);
  const dim3 blocks(m * n / (16 * DECOMPRESS_D4_BLOCK_SIZE));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cuda_decompress_d4_origorder_kernel<<<blocks, threads, 0, stream>>>(
      YIs.data_ptr<uint8_t>(), CB.data_ptr<c10::Half>(),
      Y.data_ptr<c10::Half>());
#endif
}

#define DECOMPRESS_E8P_BLOCK_SIZE 256

__global__ void cuda_decompress_e8p_origorder_kernel(
    const int16_t* __restrict__ YIs,  // m x (n/8)
    const int64_t* __restrict__ CB,   // 256 x 8
    c10::Half* __restrict__ Y         // m x n
) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800
  const long i = threadIdx.x + DECOMPRESS_E8P_BLOCK_SIZE * blockIdx.x;
  uint16_t yidx = ((uint16_t*)YIs)[i];
  uint64_t decoded = BLayout_E8::decode8weights(yidx, CB);

  half2 unpacked[2][2];
  uint64_t lower_half = decoded & 0x00ff00ff00ff00ff;
  lower_half = (lower_half ^ 0x5c805c805c805c80);
  memcpy(unpacked[0], &lower_half, sizeof(uint64_t));
  uint64_t upper_half = (decoded & 0xff00ff00ff00ff00) >> 8;
  upper_half = (upper_half ^ 0x5c805c805c805c80);
  memcpy(unpacked[1], &upper_half, sizeof(uint64_t));

  const half adjust_ = __float2half_rn(-288.0f);
  const half2 adjust = __halves2half2(adjust_, adjust_);

  ((__half2*)Y)[i * 4] = __hadd2(unpacked[0][0], adjust);      // 01
  ((__half2*)Y)[i * 4 + 2] = __hadd2(unpacked[0][1], adjust);  // 45
  ((__half2*)Y)[i * 4 + 1] = __hadd2(unpacked[1][0], adjust);  // 23
  ((__half2*)Y)[i * 4 + 3] = __hadd2(unpacked[1][1], adjust);  // 67
#endif
}

void decompress_e8p_origorder(torch::Tensor YIs,  // m x (n/8)
                              torch::Tensor CB,   // 256 x 8
                              torch::Tensor& Y    // m x n
) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800
  size_t m = Y.sizes()[0];
  size_t n = Y.sizes()[1];

  assert(YIs.is_contiguous());
  assert(CB.is_contiguous());
  assert(Y.is_contiguous());

  assert(YIs.sizes()[0] == m);
  assert(YIs.sizes()[1] * 8 == n);
  assert(CB.sizes()[0] == 256);

  const dim3 threads(DECOMPRESS_E8P_BLOCK_SIZE);
  const dim3 blocks(m * n / (8 * DECOMPRESS_E8P_BLOCK_SIZE));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cuda_decompress_e8p_origorder_kernel<<<blocks, threads, 0, stream>>>(
      YIs.data_ptr<int16_t>(), CB.data_ptr<int64_t>(), Y.data_ptr<c10::Half>());
#endif
}

#define DECOMPRESS_HI_BLOCK_SIZE 256

__global__ void cuda_decompress_hi_origorder_kernel(
    const uint32_t* __restrict__ YIs,  // m x (n/8)
    c10::Half* __restrict__ Y          // m x n
) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800
  const long i = threadIdx.x + DECOMPRESS_HI_BLOCK_SIZE * blockIdx.x;
  uint32_t qa = YIs[i];

  const uint32_t c0 = 0x64086408;
  const half y16_ = __float2half_rn(1.0f / 16.0f);
  const half2 y16 = __halves2half2(y16_, y16_);
  const half z16_ = __float2half_rn(-1024.0f / 16.0f - 8.0f);
  const half2 z16 = __halves2half2(z16_, z16_);

  uint32_t q0 = (((qa & 0x000f000f) << 4) | c0);
  uint32_t q1 = ((qa & 0x00f000f0) | c0);
  qa >>= 8;
  uint32_t q2 = (((qa & 0x000f000f) << 4) | c0);
  uint32_t q3 = ((qa & 0x00f000f0) | c0);
  ((__half2*)Y)[i * 4] = __hfma2(*((half2*)(&q0)), y16, z16);
  ((__half2*)Y)[i * 4 + 1] = __hfma2(*((half2*)(&q1)), y16, z16);
  ((__half2*)Y)[i * 4 + 2] = __hfma2(*((half2*)(&q2)), y16, z16);
  ((__half2*)Y)[i * 4 + 3] = __hfma2(*((half2*)(&q3)), y16, z16);
#endif
}

void decompress_hi_origorder(torch::Tensor YIs,  // m x (n/8)
                             torch::Tensor Y     // m x n
) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800
  size_t m = Y.sizes()[0];
  size_t n = Y.sizes()[1];

  assert(YIs.is_contiguous());
  assert(Y.is_contiguous());

  assert(YIs.sizes()[0] == m);
  assert(YIs.sizes()[1] * 8 == n);

  const dim3 threads(DECOMPRESS_HI_BLOCK_SIZE);
  const dim3 blocks(m * n / (8 * DECOMPRESS_HI_BLOCK_SIZE));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cuda_decompress_hi_origorder_kernel<<<blocks, threads, 0, stream>>>(
      (uint32_t*)YIs.data_ptr<int32_t>(), Y.data_ptr<c10::Half>());
#endif
}
