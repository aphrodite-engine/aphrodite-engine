#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#include "config.hpp"
#include "marlin_utils.hpp"


namespace packbits_utils {


template <class SourceEngine   , class SourceLayout   ,
          class SourceEngine2  , class SourceLayout2  ,
          class TargetEngine   , class TargetLayout   ,
          class ScaleEngine    , class ScaleLayout    ,
          class QuantMapEngine , class QuantMapLayout ,
          class QuantMapEngine2, class QuantMapLayout2,
          class QuantMapEngine3, class QuantMapLayout3,
          class NumBits,
          config::QuantMapModeEnum QuantMapMode>
struct DequantizationTraits
{

  CUTE_DEVICE static
  void
  apply(
      cute::Tensor<SourceEngine   , SourceLayout   > const& source,
      cute::Tensor<SourceEngine2  , SourceLayout2  > const& source2,
      cute::Tensor<TargetEngine   , TargetLayout   >      & target,
      cute::Tensor<ScaleEngine    , ScaleLayout    > const& scale,
      cute::Tensor<QuantMapEngine , QuantMapLayout > const& qmap,
      cute::Tensor<QuantMapEngine2, QuantMapLayout2> const& qmap2,
      cute::Tensor<QuantMapEngine3, QuantMapLayout3> const& qmap3)
  {

    using TQ  = cute::uint16_t;
    using TQ2 = cute::uint32_t;
    using T   = typename TargetEngine::value_type;
    using TI  = cute::conditional_t<cute::is_same_v<T, cute::half_t>, __half , __nv_bfloat16 >;
    using T2  = cute::conditional_t<cute::is_same_v<T, cute::half_t>, __half2, __nv_bfloat162>;
    CUTE_STATIC_ASSERT(cute::is_same_v<T , cute::half_t                        > == true ||
                       cute::is_same_v<T , cute::bfloat16_t                    > == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<TQ, typename SourceEngine   ::value_type> == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<TQ, typename SourceEngine2  ::value_type> == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<T , typename TargetEngine   ::value_type> == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<T , typename ScaleEngine    ::value_type> == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<T , typename QuantMapEngine ::value_type> == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<T2, typename QuantMapEngine2::value_type> == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<T , typename QuantMapEngine3::value_type> == true);

    static constexpr int            kNumBits    = NumBits::value;
    static constexpr int            kNumPacked2 = NumBits::value == 4 ? 8          : 16;
    static constexpr cute::uint16_t kMask       = NumBits::value == 4 ? 0x000f     : 0x0003;
    static constexpr cute::uint32_t kMask2      = NumBits::value == 4 ? 0x000000ff : 0x0000000f;
    static constexpr cute::uint32_t kMaskSync   = 0xffffffff;

    // vectorize the source and target
    auto source_vec  = cute::recast<TQ2>(source);
    auto source2_vec = cute::recast<TQ2>(source2);  // unused
    auto target_vec  = cute::recast<T2 >(target);
    auto scale_vec   = cute::recast<T2 >(scale);
    auto qmap_view   = cute::recast<TI >(qmap);
    auto qmap2_view  = cute::recast<T2 >(qmap2);
    auto qmap3_view  = cute::recast<TI >(qmap3);

    CUTE_STATIC_ASSERT_V(NumBits{} == cute::_4{} || NumBits{} == cute::_2{});
    CUTE_STATIC_ASSERT_V(cute::size<0>(source ) == cute::size<0>(source_vec ) * cute::_2{});
    CUTE_STATIC_ASSERT_V(cute::size<0>(source2) == cute::size<0>(source2_vec) * cute::_2{});
    CUTE_STATIC_ASSERT_V(cute::size<0>(target ) == cute::size<0>(target_vec ) * cute::_2{});
    CUTE_STATIC_ASSERT_V(cute::size<0>(scale  ) == cute::size<0>(scale_vec  ) * cute::_2{});
    CUTE_STATIC_ASSERT_V(cute::size<1>(source ) == cute::size<1>(source_vec ));
    CUTE_STATIC_ASSERT_V(cute::size<1>(source2) == cute::size<1>(source2_vec));
    CUTE_STATIC_ASSERT_V(cute::size<1>(target ) == cute::size<1>(target_vec ));
    CUTE_STATIC_ASSERT_V(cute::size<1>(scale  ) == cute::size<1>(scale_vec  ));
    CUTE_STATIC_ASSERT_V(cute::size   (qmap   ) == cute::size   (qmap_view  ));
    CUTE_STATIC_ASSERT_V(cute::size   (qmap2  ) == cute::size   (qmap2_view ));
    CUTE_STATIC_ASSERT_V(cute::size   (qmap3  ) == cute::size   (qmap3_view ));

    CUTE_UNROLL
    for (int i = 0; i < cute::size<0>(source_vec); ++i)
    {

      CUTE_UNROLL
      for (int p = 0; p < cute::size<1>(source_vec); ++p)
      {
        auto src_crd = cute::make_coord(i, p);

        CUTE_UNROLL
        for (int k2 = 0; k2 < kNumPacked2; k2 += 2)
        {
          auto k = k2 / 2;
          auto tgt_crd = cute::make_coord(i, k * cute::size<1>(source_vec) + p);
          auto src_raw = source_vec(src_crd) >> (k2 * kNumBits);
          T2   src_val;

          if constexpr ((QuantMapMode == config::QuantMapModeEnum::Vectorized   ) ||
                        (QuantMapMode == config::QuantMapModeEnum::Vectorized_32) ||
                        (QuantMapMode == config::QuantMapModeEnum::Vectorized_16) ||
                        (QuantMapMode == config::QuantMapModeEnum::Vectorized_8))
          {
            // vectorized table lookup
            src_val = qmap2_view[src_raw & kMask2];

          }
          else
          {

            TI src_val_0;
            TI src_val_1;

            if constexpr (QuantMapMode == config::QuantMapModeEnum::WarpShuffle)
            {
              // in-register table lookup
              src_val_0 = __shfl_sync(kMaskSync, qmap3_view(0), (src_raw >> kNumBits) & kMask);
              src_val_1 = __shfl_sync(kMaskSync, qmap3_view(0), (src_raw            ) & kMask);
            }
            else
            {
              // normal table lookup
              src_val_0 = qmap_view[(src_raw >> kNumBits) & kMask];
              src_val_1 = qmap_view[(src_raw            ) & kMask];
            }

            if constexpr (cute::is_same_v<T, cute::half_t>)
            {
              src_val = __halves2half2    (src_val_0, src_val_1);
            }
            else
            {
              src_val = __halves2bfloat162(src_val_0, src_val_1);
            }

          }

          // vectorized scaling
          target_vec(tgt_crd) = __hmul2(src_val, scale_vec(tgt_crd));
        }
      }
    }
  }
};


template <class SourceEngine   , class SourceLayout   ,
          class SourceEngine2  , class SourceLayout2  ,
          class TargetEngine   , class TargetLayout   ,
          class ScaleEngine    , class ScaleLayout    ,
          class QuantMapEngine , class QuantMapLayout ,
          class QuantMapEngine2, class QuantMapLayout2,
          class QuantMapEngine3, class QuantMapLayout3>
struct DequantizationTraits<SourceEngine   , SourceLayout   ,
                            SourceEngine2  , SourceLayout2  ,
                            TargetEngine   , TargetLayout   ,
                            ScaleEngine    , ScaleLayout    ,
                            QuantMapEngine , QuantMapLayout ,
                            QuantMapEngine2, QuantMapLayout2,
                            QuantMapEngine3, QuantMapLayout3,
                            cute::Int<4>,
                            config::QuantMapModeEnum::Marlin>
{

  CUTE_DEVICE static
  void
  apply(
      cute::Tensor<SourceEngine   , SourceLayout   > const& source,
      cute::Tensor<SourceEngine2  , SourceLayout2  > const& source2,
      cute::Tensor<TargetEngine   , TargetLayout   >      & target,
      cute::Tensor<ScaleEngine    , ScaleLayout    > const& scale,
      cute::Tensor<QuantMapEngine , QuantMapLayout > const& qmap,
      cute::Tensor<QuantMapEngine2, QuantMapLayout2> const& qmap2,
      cute::Tensor<QuantMapEngine3, QuantMapLayout3> const& qmap3)
  {

    using TQ  = cute::uint16_t;
    using TQ2 = cute::uint32_t;
    using T   = typename TargetEngine::value_type;
    using TI  = cute::conditional_t<cute::is_same_v<T, cute::half_t>, __half , __nv_bfloat16 >;
    using T2  = cute::conditional_t<cute::is_same_v<T, cute::half_t>, __half2, __nv_bfloat162>;
    CUTE_STATIC_ASSERT(cute::is_same_v<T , cute::half_t                        > == true ||
                       cute::is_same_v<T , cute::bfloat16_t                    > == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<TQ, typename SourceEngine   ::value_type> == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<TQ, typename SourceEngine2  ::value_type> == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<T , typename TargetEngine   ::value_type> == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<T , typename ScaleEngine    ::value_type> == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<T , typename QuantMapEngine ::value_type> == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<T2, typename QuantMapEngine2::value_type> == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<T , typename QuantMapEngine3::value_type> == true);

    static constexpr int kNumBits    = 4;
    static constexpr int kNumPacked2 = 8;

    // vectorize the source and target
    auto source_vec  = cute::recast<TQ2>(source);
    auto source2_vec = cute::recast<TQ2>(source2);  // unused
    auto target_vec  = cute::recast<T2 >(target);
    auto scale_vec   = cute::recast<T2 >(scale);
    auto qmap_view   = cute::recast<TI >(qmap);
    auto qmap2_view  = cute::recast<T2 >(qmap2);
    auto qmap3_view  = cute::recast<TI >(qmap3);

    CUTE_STATIC_ASSERT_V(cute::size<0>(source ) == cute::size<0>(source_vec ) * cute::_2{});
    CUTE_STATIC_ASSERT_V(cute::size<0>(source2) == cute::size<0>(source2_vec) * cute::_2{});
    CUTE_STATIC_ASSERT_V(cute::size<0>(target ) == cute::size<0>(target_vec ) * cute::_2{});
    CUTE_STATIC_ASSERT_V(cute::size<0>(scale  ) == cute::size<0>(scale_vec  ) * cute::_2{});
    CUTE_STATIC_ASSERT_V(cute::size<1>(source ) == cute::size<1>(source_vec ));
    CUTE_STATIC_ASSERT_V(cute::size<1>(source2) == cute::size<1>(source2_vec));
    CUTE_STATIC_ASSERT_V(cute::size<1>(target ) == cute::size<1>(target_vec ));
    CUTE_STATIC_ASSERT_V(cute::size<1>(scale  ) == cute::size<1>(scale_vec  ));
    CUTE_STATIC_ASSERT_V(cute::size   (qmap   ) == cute::size   (qmap_view  ));
    CUTE_STATIC_ASSERT_V(cute::size   (qmap2  ) == cute::size   (qmap2_view ));
    CUTE_STATIC_ASSERT_V(cute::size   (qmap3  ) == cute::size   (qmap3_view ));

    CUTE_UNROLL
    for (int i = 0; i < cute::size<0>(source_vec); ++i)
    {

      CUTE_UNROLL
      for (int p = 0; p < cute::size<1>(source_vec); ++p)
      {
        auto src_crd = cute::make_coord(i, p);

        CUTE_UNROLL
        for (int k4 = 0; k4 < kNumPacked2; k4 += 4)
        {
          auto k = k4 / 4;
          auto src_raw = source_vec(src_crd) >> (k * 8);
          auto src_val = marlin_utils::dequant(src_raw);

          auto tgt0_crd        = cute::make_coord(i, (k * 2 + 0) * cute::size<1>(source_vec) + p);
          auto tgt1_crd        = cute::make_coord(i, (k * 2 + 1) * cute::size<1>(source_vec) + p);
          target_vec(tgt0_crd) = __hmul2(src_val[0], scale_vec(tgt0_crd));
          target_vec(tgt1_crd) = __hmul2(src_val[1], scale_vec(tgt1_crd));
        }
      }
    }
  }
};


template <class SourceEngine   , class SourceLayout   ,
          class SourceEngine2  , class SourceLayout2  ,
          class TargetEngine   , class TargetLayout   ,
          class ScaleEngine    , class ScaleLayout    ,
          class QuantMapEngine , class QuantMapLayout ,
          class QuantMapEngine2, class QuantMapLayout2,
          class QuantMapEngine3, class QuantMapLayout3,
          config::QuantMapModeEnum QuantMapMode>
struct DequantizationTraits<SourceEngine   , SourceLayout   ,
                            SourceEngine2  , SourceLayout2  ,
                            TargetEngine   , TargetLayout   ,
                            ScaleEngine    , ScaleLayout    ,
                            QuantMapEngine , QuantMapLayout ,
                            QuantMapEngine2, QuantMapLayout2,
                            QuantMapEngine3, QuantMapLayout3,
                            cute::Int<3>,
                            QuantMapMode>
{

  CUTE_DEVICE static
  void
  apply(
      cute::Tensor<SourceEngine   , SourceLayout   > const& source,
      cute::Tensor<SourceEngine2  , SourceLayout2  > const& source2,
      cute::Tensor<TargetEngine   , TargetLayout   >      & target,
      cute::Tensor<ScaleEngine    , ScaleLayout    > const& scale,
      cute::Tensor<QuantMapEngine , QuantMapLayout > const& qmap,
      cute::Tensor<QuantMapEngine2, QuantMapLayout2> const& qmap2,
      cute::Tensor<QuantMapEngine3, QuantMapLayout3> const& qmap3)
  {

    using TQ  = cute::uint16_t;
    using TQ2 = cute::uint32_t;
    using T   = typename TargetEngine::value_type;
    using TI  = cute::conditional_t<cute::is_same_v<T, cute::half_t>, __half , __nv_bfloat16 >;
    using T2  = cute::conditional_t<cute::is_same_v<T, cute::half_t>, __half2, __nv_bfloat162>;
    CUTE_STATIC_ASSERT(cute::is_same_v<T , cute::half_t                        > == true ||
                       cute::is_same_v<T , cute::bfloat16_t                    > == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<TQ, typename SourceEngine   ::value_type> == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<TQ, typename SourceEngine2  ::value_type> == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<T , typename TargetEngine   ::value_type> == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<T , typename ScaleEngine    ::value_type> == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<T , typename QuantMapEngine ::value_type> == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<T2, typename QuantMapEngine2::value_type> == true);
    CUTE_STATIC_ASSERT(cute::is_same_v<T , typename QuantMapEngine3::value_type> == true);

    CUTE_STATIC_ASSERT  (QuantMapMode           == config::QuantMapModeEnum::Vectorized);
    CUTE_STATIC_ASSERT_V(cute::size<1>(source2) == cute::size<1>(source) * cute::_2{});
    CUTE_STATIC_ASSERT_V(cute::size<0>(source2) == cute::size<0>(source));

    static constexpr int            kNumBits    = 3;
    static constexpr int            kNumPacked2 = 10;
    static constexpr cute::uint32_t kMask2      = 0x0000003f;
    static constexpr cute::uint32_t kMaskF2     = 0x00000003;

    // vectorize the source and target
    auto source0_vec = cute::recast<TQ2>(source);
    auto source1_vec = cute::recast<TQ2>(source2);
    auto source2_vec = cute::recast<TQ2>(source2);  // the same as source2
    auto target_vec  = cute::recast<T2 >(target);
    auto scale_vec   = cute::recast<T2 >(scale);
    auto qmap_view   = cute::recast<TI >(qmap);
    auto qmap2_view  = cute::recast<T2 >(qmap2);
    auto qmap3_view  = cute::recast<TI >(qmap3);

    CUTE_STATIC_ASSERT_V(cute::size<0>(source ) == cute::size<0>(source0_vec) * cute::_2{});
    CUTE_STATIC_ASSERT_V(cute::size<0>(source2) == cute::size<0>(source1_vec) * cute::_2{});
    CUTE_STATIC_ASSERT_V(cute::size<0>(source2) == cute::size<0>(source2_vec) * cute::_2{});
    CUTE_STATIC_ASSERT_V(cute::size<0>(target ) == cute::size<0>(target_vec ) * cute::_2{});
    CUTE_STATIC_ASSERT_V(cute::size<0>(scale  ) == cute::size<0>(scale_vec  ) * cute::_2{});
    CUTE_STATIC_ASSERT_V(cute::size<1>(source ) == cute::size<1>(source0_vec));
    CUTE_STATIC_ASSERT_V(cute::size<1>(source2) == cute::size<1>(source1_vec));
    CUTE_STATIC_ASSERT_V(cute::size<1>(source2) == cute::size<1>(source2_vec));
    CUTE_STATIC_ASSERT_V(cute::size<1>(target ) == cute::size<1>(target_vec ));
    CUTE_STATIC_ASSERT_V(cute::size<1>(scale  ) == cute::size<1>(scale_vec  ));
    CUTE_STATIC_ASSERT_V(cute::size   (qmap   ) == cute::size   (qmap_view  ));
    CUTE_STATIC_ASSERT_V(cute::size   (qmap2  ) == cute::size   (qmap2_view ));
    CUTE_STATIC_ASSERT_V(cute::size   (qmap3  ) == cute::size   (qmap3_view ));

    CUTE_UNROLL
    for (int i = 0; i < cute::size<0>(source0_vec); ++i)
    {

      CUTE_UNROLL
      for (int p = 0; p < cute::size<1>(source0_vec); ++p)
      {
        auto src0_crd = cute::make_coord(i, p);
        auto src1_crd = cute::make_coord(i, p * 2);
        auto src2_crd = cute::make_coord(i, p * 2 + 1);

        CUTE_UNROLL
        for (int k2 = 0; k2 < kNumPacked2; k2 += 2)
        {
          auto k = k2 / 2;
          // using `source0_vec` for the stride, since others sahre the same value
          auto tgt0_crd        = cute::make_coord(i, (k * 3 + 0) * cute::size<1>(source0_vec) + p);
          auto tgt1_crd        = cute::make_coord(i, (k * 3 + 1) * cute::size<1>(source0_vec) + p);
          auto tgt2_crd        = cute::make_coord(i, (k * 3 + 2) * cute::size<1>(source0_vec) + p);

          auto src0_raw        = source0_vec(src0_crd) >> (k2 * kNumBits);
          auto src0_val        = qmap2_view [src0_raw  &   kMask2];
          target_vec(tgt0_crd) = __hmul2    (src0_val  ,   scale_vec(tgt0_crd));

          auto src1_raw        = source1_vec(src1_crd) >> (k2 * kNumBits);
          auto src1_val        = qmap2_view [src1_raw  &   kMask2];
          target_vec(tgt1_crd) = __hmul2    (src1_val  ,   scale_vec(tgt1_crd));

          auto src2_raw        = source2_vec(src2_crd) >> (k2 * kNumBits);
          auto src2_val        = qmap2_view [src2_raw  &   kMask2];
          target_vec(tgt2_crd) = __hmul2    (src2_val  ,   scale_vec(tgt2_crd));
        }

        // handle the last element
        auto tgt3_crd        = cute::make_coord(i, ((kNumPacked2 / 2) * 3) * cute::size<1>(source0_vec) + p);
        auto src3_raw        = ((((source0_vec(src0_crd) >> (kNumPacked2 * kNumBits)) & kMaskF2)     ) |
                                (((source1_vec(src1_crd) >> (kNumPacked2 * kNumBits)) & kMaskF2) << 2) |
                                (((source2_vec(src2_crd) >> (kNumPacked2 * kNumBits)) & kMaskF2) << 4));
        auto src3_val        = qmap2_view[src3_raw & kMask2];
        target_vec(tgt3_crd) = __hmul2(src3_val, scale_vec(tgt3_crd));
      }
    }
  }
};


template <config::QuantMapModeEnum QuantMapMode,
          class SourceEngine   , class SourceLayout   ,
          class SourceEngine2  , class SourceLayout2  ,
          class TargetEngine   , class TargetLayout   ,
          class ScaleEngine    , class ScaleLayout    ,
          class QuantMapEngine , class QuantMapLayout ,
          class QuantMapEngine2, class QuantMapLayout2,
          class QuantMapEngine3, class QuantMapLayout3,
          class NumBits>
CUTE_DEVICE
void
dequantize(
    cute::Tensor<SourceEngine   , SourceLayout   > const& source,
    cute::Tensor<SourceEngine2  , SourceLayout2  > const& source2,
    cute::Tensor<TargetEngine   , TargetLayout   >     && target,
    cute::Tensor<ScaleEngine    , ScaleLayout    > const& scale,
    cute::Tensor<QuantMapEngine , QuantMapLayout > const& qmap,
    cute::Tensor<QuantMapEngine2, QuantMapLayout2> const& qmap2,
    cute::Tensor<QuantMapEngine3, QuantMapLayout3> const& qmap3,
    NumBits)
{

  CUTE_STATIC_ASSERT_V(cute::rank   (source ) == cute::_2{});  // ((dim0, dim1), Mma_P)
  CUTE_STATIC_ASSERT_V(cute::rank   (source2) == cute::_2{});  // ((dim0, dim1), Mma_P * 2)
  CUTE_STATIC_ASSERT_V(cute::rank   (target ) == cute::_2{});  // ((dim0, dim1), Mma)
  CUTE_STATIC_ASSERT_V(cute::rank   (scale  ) == cute::_2{});  // ((dim0, dim1), Mma)
  CUTE_STATIC_ASSERT_V(cute::rank   (qmap   ) == cute::_1{});  // (2 ** (NumBits),)
  CUTE_STATIC_ASSERT_V(cute::rank   (qmap2  ) == cute::_1{});  // (2 ** (NumBits * 2),)
  CUTE_STATIC_ASSERT_V(cute::rank   (qmap3  ) == cute::_1{});  // (1,)
  CUTE_STATIC_ASSERT_V(cute::size<0>(target)  == cute::size<0>(source));
  CUTE_STATIC_ASSERT_V(cute::size<0>(target)  == cute::size<0>(source2));
  CUTE_STATIC_ASSERT_V(cute::size<0>(target)  == cute::size<0>(scale));
  CUTE_STATIC_ASSERT_V(cute::size<1>(target)  == cute::size<1>(scale));
  CUTE_STATIC_ASSERT_V(cute::size   (qmap3)   == cute::_1{});
  CUTE_STATIC_ASSERT  (cute::is_same_v<typename SourceEngine   ::value_type, cute::uint16_t> == true);
  CUTE_STATIC_ASSERT  (cute::is_same_v<typename SourceEngine2  ::value_type, cute::uint16_t> == true);
  CUTE_STATIC_ASSERT  (cute::is_same_v<typename TargetEngine   ::value_type, cute::half_t  > == true || cute::is_same_v<typename TargetEngine   ::value_type, cute::bfloat16_t> == true);
  CUTE_STATIC_ASSERT  (cute::is_same_v<typename ScaleEngine    ::value_type, cute::half_t  > == true || cute::is_same_v<typename ScaleEngine    ::value_type, cute::bfloat16_t> == true);
  CUTE_STATIC_ASSERT  (cute::is_same_v<typename QuantMapEngine ::value_type, cute::half_t  > == true || cute::is_same_v<typename QuantMapEngine ::value_type, cute::bfloat16_t> == true);
  CUTE_STATIC_ASSERT  (cute::is_same_v<typename QuantMapEngine2::value_type, __half2       > == true || cute::is_same_v<typename QuantMapEngine2::value_type, __nv_bfloat162  > == true);
  CUTE_STATIC_ASSERT  (cute::is_same_v<typename QuantMapEngine3::value_type, cute::half_t  > == true || cute::is_same_v<typename QuantMapEngine3::value_type, cute::bfloat16_t> == true);

  DequantizationTraits<
    SourceEngine   , SourceLayout   ,
    SourceEngine2  , SourceLayout2  ,
    TargetEngine   , TargetLayout   ,
    ScaleEngine    , ScaleLayout    ,
    QuantMapEngine , QuantMapLayout ,
    QuantMapEngine2, QuantMapLayout2,
    QuantMapEngine3, QuantMapLayout3,
    NumBits,
    QuantMapMode>::apply(
      source,
      source2,
      target,
      scale,
      qmap,
      qmap2,
      qmap3);
}

} // namespace packbits_utils