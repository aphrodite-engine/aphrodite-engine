#include <cuda.h>
#include <stdio.h>
#include <ATen/ATen.h>
#include <cute/tensor.hpp>
#include "qgemm_kernel.hpp"


template <
  typename T,
  typename TQ,
  typename T2,
  typename NumBits,
  typename GroupSize
>
void
_qgemm_raw(int M,
           int N,
           int K,
           int P,
           const T * const __restrict__ A,
           const TQ* const __restrict__ Q,
                 T *       __restrict__ D,
           const T * const __restrict__ S,
           const T * const __restrict__ QM,
           const T2* const __restrict__ QM2,
               void*       __restrict__ workspace,
           const int                    template_id,
           const int                    num_sms,
           const cudaStream_t           stream)
{

    using namespace cute;
    static constexpr config::QuantMapModeEnum      kVectorized    = config::QuantMapModeEnum     ::Vectorized;
    static constexpr config::QuantMapModeEnum      kVectorized_32 = config::QuantMapModeEnum     ::Vectorized_32;
    static constexpr config::QuantMapModeEnum      kVectorized_16 = config::QuantMapModeEnum     ::Vectorized_16;
    static constexpr config::QuantMapModeEnum      kVectorized_8  = config::QuantMapModeEnum     ::Vectorized_8;
    static constexpr config::AccumulationModeEnum  kLow           = config::AccumulationModeEnum ::Low;
    static constexpr config::AccumulationModeEnum  kHigh          = config::AccumulationModeEnum ::High;
    static constexpr config::AccumulationModeEnum  kMixed         = config::AccumulationModeEnum ::Mixed;
    static constexpr config::DecompositionModeEnum kStreamK       = config::DecompositionModeEnum::StreamK;

#define RUN_QGEMM(T,                           \
                  TQ,                          \
                  T2,                          \
                  SMS_MULTIPLE,                \
                  THREADS,                     \
                  TILE_M,                      \
                  TILE_K,                      \
                  TILE_P,                      \
                  STAGES,                      \
                  NUM_BITS,                    \
                  GROUP_SIZE,                  \
                  QUANT_MAP_MODE,              \
                  ACCUMULATION_MODE,           \
                  DECOMPOSITION_MODE,          \
                  G2S_TILED_COPY_SIZE_S,       \
                  MMA_PRM_K)                   \
    do {                                       \
        qgemm_host<                            \
            T,                                 \
            TQ,                                \
            T2,                                \
            cute::Int<THREADS>,                \
            cute::Int<TILE_M>,                 \
            cute::Int<TILE_K>,                 \
            cute::Int<TILE_P>,                 \
            cute::Int<STAGES>,                 \
            cute::Int<NUM_BITS>,               \
            cute::Int<GROUP_SIZE>,             \
            QUANT_MAP_MODE,                    \
            ACCUMULATION_MODE,                 \
            DECOMPOSITION_MODE,                \
            cute::Int<G2S_TILED_COPY_SIZE_S>,  \
            cute::Int<MMA_PRM_K>               \
        > (                                    \
            M,                                 \
            N,                                 \
            K,                                 \
            P,                                 \
            A,                                 \
            Q,                                 \
            D,                                 \
            S,                                 \
            QM,                                \
            QM2,                               \
            workspace,                         \
            num_sms * SMS_MULTIPLE,            \
            stream);                           \
    } while (false)

    // Generated Code Below
    if constexpr (cute::is_same_v<NumBits, cute::Int<2>>)
    {
        switch (template_id)
        {
        case 0:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 1:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 2:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 3:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 4:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 5:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 6:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 7:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 8:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 9:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 10:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 11:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 12:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 13:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 14:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 15:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 16:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 17:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 18:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 19:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 20:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 21:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 22:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 23:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 24:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 25:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 26:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 27:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 28:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 29:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 30:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 31:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 32:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 33:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 34:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 35:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        default:
            AT_ERROR("Unsupported template_id value");
        }
    }
    else if constexpr (cute::is_same_v<NumBits, cute::Int<3>>)
    {
        switch (template_id)
        {
        case 0:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 1:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 2:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 3:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 4:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 5:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 6:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 7:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 8:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 9:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 10:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 11:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 12:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 13:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 14:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 15:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 16:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 17:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 18:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 19:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 20:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 21:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 22:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 23:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 24:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 25:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 26:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 27:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 28:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 29:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 30:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 31:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 32:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 33:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 34:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 35:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        default:
            AT_ERROR("Unsupported template_id value");
        }
    }
    else if constexpr (cute::is_same_v<NumBits, cute::Int<4>>)
    {
        switch (template_id)
        {
        case 0:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 1:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 2, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 2:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 2, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 3:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 2, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 4:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 5:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 3, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 6:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 3, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 7:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 3, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 8:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 9:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 4, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 10:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 4, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 11:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 4, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 12:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 13:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 5, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 14:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 5, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 15:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 64, 5, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 16:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 17:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 18:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 19:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 20:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 21:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 22:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 23:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 24:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 25:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 26:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 27:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 28:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 29:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 30:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 31:
            RUN_QGEMM(T, TQ, T2, 1, 256, 32, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 32:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 33:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 34:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 35:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 36:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 37:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 38:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 39:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 40:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 41:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 42:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 43:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 44:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 45:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 46:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 47:
            RUN_QGEMM(T, TQ, T2, 1, 128, 16, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 48:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 49:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 2, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 50:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 2, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 51:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 2, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 52:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 53:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 3, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 54:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 3, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 55:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 3, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 56:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 57:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 4, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 58:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 4, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 59:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 4, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 60:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 61:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 5, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 62:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 5, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 63:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 64, 5, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 64:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 65:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 66:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 67:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 68:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 69:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 70:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 71:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 72:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 73:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 74:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 75:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 76:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 77:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 78:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 79:
            RUN_QGEMM(T, TQ, T2, 2, 256, 32, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 80:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 81:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 82:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 83:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 84:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 85:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 86:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 87:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 88:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 89:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 90:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 91:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 92:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 93:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 94:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 95:
            RUN_QGEMM(T, TQ, T2, 2, 128, 16, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 96:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 97:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 2, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 98:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 2, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 99:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 2, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 100:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 101:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 3, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 102:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 3, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 103:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 3, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 104:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 105:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 4, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 106:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 4, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 107:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 4, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 108:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 109:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 5, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 110:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 5, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 111:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 64, 5, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 112:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 113:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 114:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 115:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 116:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 117:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 118:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 119:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 120:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 121:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 122:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 123:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 124:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 125:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 126:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 127:
            RUN_QGEMM(T, TQ, T2, 4, 256, 32, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 128:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 129:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 130:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 131:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 2, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 132:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 133:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 134:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 135:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 3, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 136:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 137:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 138:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 139:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 4, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        case 140:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized   , kMixed, kStreamK, 2, 1);
            break;
        case 141:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized_32, kMixed, kStreamK, 2, 1);
            break;
        case 142:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized_16, kMixed, kStreamK, 2, 1);
            break;
        case 143:
            RUN_QGEMM(T, TQ, T2, 4, 128, 16, 64, 32, 5, NumBits::value, GroupSize::value, kVectorized_8 , kMixed, kStreamK, 2, 1);
            break;
        default:
            AT_ERROR("Unsupported template_id value");
        }
    }
    else
    {
        AT_ERROR("Unsupported NumBits value");
    }
}


#define INSTANTIATE_TEMPLATE(T,                    \
                             TQ,                   \
                             T2,                   \
                             NUM_BITS,             \
                             GROUP_SIZE)           \
    template                                       \
    void                                           \
    _qgemm_raw<                                    \
        T,                                         \
        TQ,                                        \
        T2,                                        \
        cute::Int<NUM_BITS>,                       \
        cute::Int<GROUP_SIZE>                      \
    > (                                            \
        int M,                                     \
        int N,                                     \
        int K,                                     \
        int P,                                     \
        const T * const __restrict__ A,            \
        const TQ* const __restrict__ Q,            \
              T *       __restrict__ D,            \
        const T * const __restrict__ S,            \
        const T * const __restrict__ QM,           \
        const T2* const __restrict__ QM2,          \
            void*       __restrict__ workspace,    \
        const int                    template_id,  \
        const int                    num_sms,      \
        const cudaStream_t           stream)


// INSTANTIATE_TEMPLATE(cute::half_t    , cute::uint16_t, __half2       , 2, 32);
INSTANTIATE_TEMPLATE(cute::half_t    , cute::uint16_t, __half2       , 2, 64);
INSTANTIATE_TEMPLATE(cute::half_t    , cute::uint16_t, __half2       , 2, 128);
INSTANTIATE_TEMPLATE(cute::half_t    , cute::uint16_t, __half2       , 2, 256);
// INSTANTIATE_TEMPLATE(cute::half_t    , cute::uint16_t, __half2       , 3, 32);
INSTANTIATE_TEMPLATE(cute::half_t    , cute::uint16_t, __half2       , 3, 64);
INSTANTIATE_TEMPLATE(cute::half_t    , cute::uint16_t, __half2       , 3, 128);
INSTANTIATE_TEMPLATE(cute::half_t    , cute::uint16_t, __half2       , 3, 256);
// INSTANTIATE_TEMPLATE(cute::half_t    , cute::uint16_t, __half2       , 4, 32);
INSTANTIATE_TEMPLATE(cute::half_t    , cute::uint16_t, __half2       , 4, 64);
INSTANTIATE_TEMPLATE(cute::half_t    , cute::uint16_t, __half2       , 4, 128);
INSTANTIATE_TEMPLATE(cute::half_t    , cute::uint16_t, __half2       , 4, 256);

// INSTANTIATE_TEMPLATE(cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 2, 32);
INSTANTIATE_TEMPLATE(cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 2, 64);
INSTANTIATE_TEMPLATE(cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 2, 128);
INSTANTIATE_TEMPLATE(cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 2, 256);
// INSTANTIATE_TEMPLATE(cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 3, 32);
INSTANTIATE_TEMPLATE(cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 3, 64);
INSTANTIATE_TEMPLATE(cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 3, 128);
INSTANTIATE_TEMPLATE(cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 3, 256);
// INSTANTIATE_TEMPLATE(cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 4, 32);
INSTANTIATE_TEMPLATE(cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 4, 64);
INSTANTIATE_TEMPLATE(cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 4, 128);
INSTANTIATE_TEMPLATE(cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 4, 256);