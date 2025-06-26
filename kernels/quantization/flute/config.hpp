#pragma once

#include <cuda.h>
#include <stdio.h>
#include <cute/tensor.hpp>


namespace config {

using namespace cute;

// --- Type Traits ---
template <typename TypeSize, typename CopySize>
struct G2SCopyOpTraits;


template <>
struct G2SCopyOpTraits<_16, _8> {
    using type = cute::uint128_t;
    using op   = SM80_CP_ASYNC_CACHEGLOBAL<type>;
};


template <>
struct G2SCopyOpTraits<_16, _4> {
    using type = cute::uint64_t;
    using op   = SM80_CP_ASYNC_CACHEALWAYS<type>;
};


template <>
struct G2SCopyOpTraits<_16, _2> {
    using type = cute::uint32_t;
    using op   = SM80_CP_ASYNC_CACHEALWAYS<type>;
};


template <>
struct G2SCopyOpTraits<_32, _2> {
    using type = cute::uint64_t;
    using op   = SM80_CP_ASYNC_CACHEALWAYS<type>;
};


template <>
struct G2SCopyOpTraits<_32, _1> {
    using type = cute::uint32_t;
    using op   = SM80_CP_ASYNC_CACHEALWAYS<type>;
};


template <typename T, typename CopySize>
struct G2SCopyTraits {
    using TypeSize = Int<cute::sizeof_bits_v<T>>;
    using Op       = typename G2SCopyOpTraits<TypeSize, CopySize>::op;
    using Traits   = Copy_Traits<Op>;
    using Atom     = Copy_Atom<Traits, T>;
};


// Strategies for decomposing the problem
enum class DecompositionModeEnum {
    // Split-K and Slice-K decomposition
    SplitK,

    // Stream-K decomposition
    StreamK
};


// Strategies for computing reductions between CTAs computing portions of a given output tile
enum class ReductionModeEnum {
    // Participating CTAs perform reduction in a turnstile fashion in order of the K extent
    // covered by each CTA. This requires a lock to be held exclusively be the CTA that is
    // currently accumulating.
    Deterministic,

    // Participating CTAs perform reduction atomically to the same workspace (mostly) without locking.
    // Locks are used only to wait for the first CTA to write its partial values (to initialize the
    // workspace), and for all but the final CTA to have accumulated (so that the final CTA can load
    // the accumulated value and accumulate it into registers on top of which the epilogue will
    // be performed).
    Nondeterministic
};


enum class QuantMapModeEnum {
    // look-up one entry at a time
    Basic,

    // look-up two entries at a time
    Vectorized,

    // vectorized and duplicated 32 times
    Vectorized_32,

    // vectorized and duplicated 16 times
    Vectorized_16,

    // vectorized and duplicated 8 times
    Vectorized_8,

    // look-up one entry at time using warp shuffle (deprecated)
    WarpShuffle,

    // Marlin-style integer dequantization
    Marlin
};


enum class AccumulationModeEnum {
    // accumulate in FP16/BF16
    Low,

    // accumulate in FP32
    High,

    // accumulate in FP32, but reduce in FP16/BF16
    Mixed
};


template <typename NumBits, QuantMapModeEnum QuantMapMode>
struct VectorizedQuantMapTraits {
    using Duplicates   = _1;
    using IsVectorized = false_type;
    using IsDuplicated = false_type;
};


template <typename NumBits>
struct VectorizedQuantMapTraits<NumBits, QuantMapModeEnum::Vectorized> {
    using Duplicates   = _1;
    using IsVectorized = true_type;
    using IsDuplicated = false_type;
};


template <typename NumBits>
struct VectorizedQuantMapTraits<NumBits, QuantMapModeEnum::Vectorized_32> {
    CUTE_STATIC_ASSERT_V(NumBits{} == _4{});
    using Duplicates   = _32;
    using IsVectorized = true_type;
    using IsDuplicated = true_type;
};


template <typename NumBits>
struct VectorizedQuantMapTraits<NumBits, QuantMapModeEnum::Vectorized_16> {
    CUTE_STATIC_ASSERT_V(NumBits{} == _4{});
    using Duplicates   = _16;
    using IsVectorized = true_type;
    using IsDuplicated = true_type;
};


template <typename NumBits>
struct VectorizedQuantMapTraits<NumBits, QuantMapModeEnum::Vectorized_8> {
    CUTE_STATIC_ASSERT_V(NumBits{} == _4{});
    using Duplicates   = _8;
    using IsVectorized = true_type;
    using IsDuplicated = true_type;
};


template <
    // types
    typename T_,
    typename TQ_,
    // threads and tile sizes
    typename Threads_,
    typename TileM_,
    typename TileK_,
    typename TileP_,
    typename Stages_,
    // quantization
    typename NumBits_,
    typename GroupSize_,
    // enums
    QuantMapModeEnum QuantMapMode_,
    AccumulationModeEnum AccumulationMode_,
    DecompositionModeEnum DecompositionMode_,
    // misc
    typename G2STiledCopySizeS_,
    typename MmaPrmK_
>
struct GemmConfig {

    // enums
    static constexpr QuantMapModeEnum      QuantMapMode      = QuantMapMode_;
    static constexpr AccumulationModeEnum  AccumulationMode  = AccumulationMode_;
    static constexpr DecompositionModeEnum DecompositionMode = DecompositionMode_;

    // type configuration
    using T  = T_;
    using TQ = TQ_;  // type for quantization
    using TC = conditional_t<AccumulationMode == AccumulationModeEnum::Low , T, float>;  // type for accumulation
    using TR = conditional_t<AccumulationMode != AccumulationModeEnum::High, T, float>;  // type for reduction
    using T2 = conditional_t<is_same_v<T, half_t>, __half2, __nv_bfloat162>;

    CUTE_STATIC_ASSERT(sizeof(T ) == 2);
    CUTE_STATIC_ASSERT(sizeof(TQ) == 2);
    CUTE_STATIC_ASSERT(is_same_v<T, half_t> == true || is_same_v<T, bfloat16_t> == true);

    // threads configuration
    using Warps   = _32;  // using compile-time constant instead of runtime `warpSize`
    using Threads = Threads_;
    CUTE_STATIC_ASSERT_V(Threads{} % _128{} == _0{});

    // quantization configuration
    using NumBits   = NumBits_;
    using NumPacked = decltype(_8{} * Int<sizeof(TQ)>{} / NumBits{});
    using GroupSize = GroupSize_;

    // 2-bit: 1 x 16-bit -> 8  x 2-bit
    // 4-bit: 1 x 16-bit -> 4  x 4-bit
    // 8-bit: 3 x 16-bit -> 16 x 3-bit
    using UnpackSourceSize = conditional_t<is_same_v<NumBits, _3>, _3, _3>;  // workaround: use size = 3 when NumBits != 3 as well
    using UnpackTargetSize = conditional_t<is_same_v<NumBits, _3>, _16, NumPacked>;

    // tile configuration
    using TileM  = TileM_;
    using TileK  = TileK_;
    using TileP  = TileP_;
    using Stages = Stages_;

    using G2STiledCopySizeA  = decltype(cute::min(TileM{} * TileK{} / Threads{}, _8{}));
    using G2STiledCopySizeQ  = _8;
    using G2STiledCopySizeQ2 = _8;
    using G2STiledCopySizeS  = G2STiledCopySizeS_;
    using EpilogueCopySizeC  = _8;
    CUTE_STATIC_ASSERT_V(G2STiledCopySizeS{} == _2{} || G2STiledCopySizeS{} == _8{});

    // derived tile configuration
    using TileP2 = decltype(TileP{} * _2{});
    using TileN  = decltype(TileP{} * UnpackTargetSize{});
    // we want `TileN x TileG = Threads x G2STiledCopySizeS` and `TileG >= G2STiledCopySizeS`
    using TileG  = decltype(Threads{} * G2STiledCopySizeS{} / cute::min(TileN{}, Threads{}));
    // note that for scales, each `TileG` can service multiple `TileK`, hence we only need
    // `Stages / (GroupSize * TileG / TileK)` stages of scales, with a minimum of 2 stages.
    using TileKsPerTileG = decltype(GroupSize{} * TileG{} / TileK{});
    // in the prefetching stage, we load `Stages{} - 1` stages of A and Q. We thus need to add
    // an additional stage to the `scales` similarly.
    using StagesGRaw     = decltype(ceil_div(Stages{} - _1{}, TileKsPerTileG{}) + _1{});
    using StagesG        = decltype(ceil_div(StagesGRaw{}, _2{}) * _2{});  // round up to the nearest multiple of 2

    CUTE_STATIC_ASSERT_V(TileM{} == _16{} || TileM{} == _32{} || TileM{} == _64{});
    CUTE_STATIC_ASSERT_V((TileP {} * UnpackTargetSize{}) == TileN{});  // TileP * NumPacked must be equal to TileN
    CUTE_STATIC_ASSERT_V((TileM {} * TileK{}) % (Threads{} * G2STiledCopySizeA {}) == _0{});  // TileA  is too small for G2S copy
    CUTE_STATIC_ASSERT_V((TileP {} * TileK{}) % (Threads{} * G2STiledCopySizeQ {}) == _0{});  // TileQ  is too small for G2S copy
    CUTE_STATIC_ASSERT_V((TileP2{} * TileK{}) % (Threads{} * G2STiledCopySizeQ2{}) == _0{});  // TileQ2 is too small for G2S copy
    CUTE_STATIC_ASSERT_V((TileN {} * TileG{}) % (Threads{} * G2STiledCopySizeS {}) == _0{});  // TileS  is too small for G2S copy
    // CUTE_STATIC_ASSERT_V(Stages{} >= StagesG{});  // do we really need this?
    // CUTE_STATIC_ASSERT_V(Stages{} % TileKsPerTileG{} == _0{});
    CUTE_STATIC_ASSERT_V((GroupSize{} * TileG{}) % TileK{} == _0{});
    CUTE_STATIC_ASSERT_V((GroupSize{} * TileG{}) == (TileK{} * TileKsPerTileG{}));
    CUTE_STATIC_ASSERT_V(((GroupSize{} >= TileK{}) && (GroupSize{} % TileK{} == _0{})) ||  // GroupSize must be a multiple of TileK
                         ((GroupSize{} <  TileK{}) && (TileK{} % GroupSize{} == _0{})));   // TileK must be a multiple of GroupSize

    //
    // --- SMEM Layouts ---
    //

    // https://github.com/NVIDIA/cutlass/blob/main/test/unit/gemm/device/default_gemm_configuration.hpp#L84
    using SmemLayoutAtomK = decltype(composition (Swizzle<3, 3, 3>{}, make_layout(make_shape(_8{}, _64{}), make_stride(_64{}, _1{}))));
    using SmemLayoutA     = decltype(tile_to_shape(SmemLayoutAtomK{}, make_shape(TileM{}, TileK{}, Stages{})));
    using SmemLayoutB     = decltype(tile_to_shape(SmemLayoutAtomK{}, make_shape(TileN{}, TileK{}, Stages{})));
    using SmemLayoutQ     = decltype(tile_to_shape(SmemLayoutAtomK{}, make_shape(TileP{}, TileK{}, Stages{})));
    using SmemLayoutQ2    = decltype(tile_to_shape(SmemLayoutAtomK{}, make_shape(TileP2{},TileK{}, Stages{})));
    using SmemLayoutAtomG = decltype(composition (Swizzle<3, 3, 3>{}, make_layout(make_shape(_8{}, TileG{}), make_stride(TileG{}, _1{}))));
    using SmemLayoutS     = decltype(tile_to_shape(SmemLayoutAtomG{}, make_shape(TileN{}, TileG{}, StagesG{})));

    // `SmemLayoutSView` is `SmemLayoutS` broadcasted to `TileK` dimension
    // 1. `GroupSize > TileK`: (TileN, TileK, (GroupSize / TileK, TileG, StagesG))
    //                         (TileG, 0    , (0                , 1    , TileN * TileG))
    // 2. `TileK > GroupSize`: (TileN, (GroupSize, TileK / GroupSize), (TileG / (TileK / GroupSize), StagesG))
    //                         (TileG, (0        , 1                ), (TileK / GroupSize          , TileN * TileG))
    using TileKsPerGroup             = decltype(ceil_div   (GroupSize{}, TileK{}));           // used when `GroupSize > TileK`
    using GroupsPerTileK             = decltype(ceil_div   (TileK    {}, GroupSize{}));       // used when `TileK > GroupSize`
    using TileGView                  = decltype(ceil_div   (TileG    {}, GroupsPerTileK{}));  // ceil(TileG / ceil(TileK / GroupSize))
    using SmemLayoutSViewShapeCase1  = decltype(make_shape (TileN    {}, TileK    {}, TileKsPerGroup{}, TileG         {}, StagesG{}));
    using SmemLayoutSViewShapeCase2  = decltype(make_shape (TileN    {}, GroupSize{}, GroupsPerTileK{}, TileGView     {}, StagesG{}));
    using SmemLayoutSViewStrideCase1 = decltype(make_stride(TileG    {},        _0{},             _0{},             _1{}, TileN{} * TileG{}));
    using SmemLayoutSViewStrideCase2 = decltype(make_stride(TileG    {},        _0{},             _1{}, GroupsPerTileK{}, TileN{} * TileG{}));
    using SmemLayoutSViewCaseRaw1    = decltype(composition(Swizzle<3, 3, 3>{}, make_layout(SmemLayoutSViewShapeCase1{}, SmemLayoutSViewStrideCase1{})));
    using SmemLayoutSViewCaseRaw2    = decltype(composition(Swizzle<3, 3, 3>{}, make_layout(SmemLayoutSViewShapeCase2{}, SmemLayoutSViewStrideCase2{})));
    using SmemLayoutSViewCase1       = decltype(            group<2, 5>(SmemLayoutSViewCaseRaw1{}));
    using SmemLayoutSViewCase2       = decltype(group<2, 4>(group<1, 3>(SmemLayoutSViewCaseRaw2{})));
    using SmemLayoutSView            = conditional_t<(TileK{} <= GroupSize{}), SmemLayoutSViewCase1, SmemLayoutSViewCase2>;
    using StagesGView                = decltype(size<2>    (SmemLayoutSView{}));

    // `SmemLayoutAtomK` requirements
    CUTE_STATIC_ASSERT_V(TileM{}  >= size<0>(SmemLayoutAtomK{}));
    CUTE_STATIC_ASSERT_V(TileN{}  >= size<0>(SmemLayoutAtomK{}));
    CUTE_STATIC_ASSERT_V(TileP{}  >= size<0>(SmemLayoutAtomK{}));
    CUTE_STATIC_ASSERT_V(TileP2{} >= size<0>(SmemLayoutAtomK{}));
    CUTE_STATIC_ASSERT_V(TileK{}  >= size<1>(SmemLayoutAtomK{}));

    //
    // --- Quantization Maps ---
    //

    using QuantMapVecTraits  = VectorizedQuantMapTraits<NumBits, QuantMapMode>;
    using QuantMapDuplicates = typename QuantMapVecTraits::Duplicates;

    // 2^NumBits
    using QuantMapSize       = decltype(_1{} << (NumBits{}));
    using QuantMapSize2      = decltype(_1{} << (NumBits{} * _2{}));
    using SmemLayoutQM       = decltype(make_layout(make_shape(QuantMapSize{})));
    using SmemLayoutQM2      = decltype(make_layout(make_shape(QuantMapSize2{})));
    using SmemLayoutQM3      = decltype(make_layout(make_shape(QuantMapSize2{}, QuantMapDuplicates{}), LayoutRight{}));  // TODO: ablate layout left
    using SmemLayoutQMView   = decltype(make_layout(make_shape(_1{}, QuantMapSize{})));
    using SmemLayoutQM2View  = decltype(make_layout(make_shape(QuantMapSize2{}, QuantMapDuplicates{}), make_stride(_1{}, _0{})));

    //
    // --- MMA ---
    //
    // https://github.com/NVIDIA/cutlass/discussions/1142
    // https://zhuanlan.zhihu.com/p/663092747
    // https://github.com/NVIDIA/cutlass/issues/1028#issuecomment-1668088899
    // It is recommended to tile TiledMma to cover the 4 sub-partitions of the SM (128 thread large tiled TiledMma, or some multiple of it).

    using MmaOpFP16 = conditional_t<AccumulationMode == AccumulationModeEnum::Low, SM80_16x8x16_F16F16F16F16_TN  , SM80_16x8x16_F32F16F16F32_TN  >;
    using MmaOpBF16 = conditional_t<AccumulationMode == AccumulationModeEnum::Low, SM80_16x8x16_F32BF16BF16F32_TN, SM80_16x8x16_F32BF16BF16F32_TN>;
    using MmaOp     = conditional_t<is_same_v<T, half_t> == true, MmaOpFP16, MmaOpBF16>;
    using MmaTraits = MMA_Traits<MmaOp>;
    using MmaAtom   = MMA_Atom<MmaTraits>;
    CUTE_STATIC_ASSERT(is_same_v<T, half_t> == true || AccumulationMode != AccumulationModeEnum::Low);  // BF16 does not low-precision accumulation

    // We can increase the size of the computation via
    // 1. adding more threads (`kMmaThr*`)
    using LargeM  = decltype(TileM  {} > _32{});
    using NumMmas = decltype(Threads{} / _32{});  // Each `MmaOp` takes 32 threads.
    using MmaThrM = conditional_t<LargeM{}, decltype(TileM{} / _32{}), decltype(TileM{} / _16{})>;
    using MmaThrN = decltype(NumMmas{} / MmaThrM{});
    using MmaThrK = _1;

    // 2. adding more works to threads (`kMmaPrm*`)
    using MmaPrmM = decltype(TileM{} / (get<0>(typename MmaTraits::Shape_MNK{}) * MmaThrM{}));
    using MmaPrmN = decltype(TileP{} / (get<1>(typename MmaTraits::Shape_MNK{}) * MmaThrN{}));
    using MmaPrmK = MmaPrmK_;

    using MmaPermutations   = decltype(make_tile(
        get<0>(typename MmaTraits::Shape_MNK{}) * MmaThrM{} * MmaPrmM{},
        get<1>(typename MmaTraits::Shape_MNK{}) * MmaThrN{} * MmaPrmN{} * UnpackTargetSize{},
        get<2>(typename MmaTraits::Shape_MNK{}) * MmaThrK{} * MmaPrmK{}));
    using MmaPermutationsQ  = decltype(make_tile(
        get<0>(typename MmaTraits::Shape_MNK{}) * MmaThrM{} * MmaPrmM{},
        get<1>(typename MmaTraits::Shape_MNK{}) * MmaThrN{} * MmaPrmN{},
        get<2>(typename MmaTraits::Shape_MNK{}) * MmaThrK{} * MmaPrmK{}));
    using MmaPermutationsQ2 = decltype(make_tile(
        get<0>(typename MmaTraits::Shape_MNK{}) * MmaThrM{} * MmaPrmM{},
        get<1>(typename MmaTraits::Shape_MNK{}) * MmaThrN{} * MmaPrmN{} * (UnpackSourceSize{} - _1{}),  // `-1` because we already have one in `MmaPermutationsQ`
        get<2>(typename MmaTraits::Shape_MNK{}) * MmaThrK{} * MmaPrmK{}));
    using MmaThrLayout     = decltype(make_layout(make_shape(MmaThrM{}, MmaThrN{}, MmaThrK{})));
    using TiledMma         = decltype(make_tiled_mma(MmaAtom{}, MmaThrLayout{}, MmaPermutations{}));
    using TiledMmaQ        = decltype(make_tiled_mma(MmaAtom{}, MmaThrLayout{}, MmaPermutationsQ{}));
    using TiledMmaQ2       = decltype(make_tiled_mma(MmaAtom{}, MmaThrLayout{}, MmaPermutationsQ2{}));

    CUTE_STATIC_ASSERT_V(MmaThrK{} == _1{});  // We don't know how to handle MmaThrK > 1, yet
    CUTE_STATIC_ASSERT_V(MmaPrmK{} == _1{} || MmaPrmK{} == _2{});
    CUTE_STATIC_ASSERT_V(Threads{} == size(TiledMma{}));
    CUTE_STATIC_ASSERT_V(Threads{} == size(TiledMmaQ{}));
    CUTE_STATIC_ASSERT_V(Threads{} == size(TiledMmaQ2{}));
    // we cannot compute with more data than we will load
    CUTE_STATIC_ASSERT_V(tile_size<0>(TiledMma{})   == TileM{});
    CUTE_STATIC_ASSERT_V(tile_size<1>(TiledMma{})   == TileN{});
    CUTE_STATIC_ASSERT_V(tile_size<2>(TiledMma{})   <= TileK{});
    CUTE_STATIC_ASSERT_V(tile_size<0>(TiledMmaQ{})  == TileM{});
    CUTE_STATIC_ASSERT_V(tile_size<1>(TiledMmaQ{})  == TileP{});
    CUTE_STATIC_ASSERT_V(tile_size<2>(TiledMmaQ{})  <= TileK{});
    CUTE_STATIC_ASSERT_V(tile_size<0>(TiledMmaQ2{}) == TileM{});
    CUTE_STATIC_ASSERT_V(tile_size<1>(TiledMmaQ2{}) == TileP2{});
    CUTE_STATIC_ASSERT_V(tile_size<2>(TiledMmaQ2{}) <= TileK{});
    CUTE_STATIC_ASSERT(is_same_v<typename MmaTraits::ValTypeD, TC> == true);  // Mma and TC must have the same accumulation type
    CUTE_STATIC_ASSERT_V(Threads{} == (MmaThrM{} * MmaThrN{} * MmaThrK{} * _32{}));  // TiledMma should cover all threads

    // just to make sure we allocate enough scratch space for StreamK
    CUTE_STATIC_ASSERT_V((MmaPrmM{})                      <=  _2{});
    CUTE_STATIC_ASSERT_V((MmaPrmN{} * UnpackTargetSize{}) <= _32{});

    //
    // --- Global to Shared Memory Copy ---
    //

    using G2STiledCopySizeQM  = decltype(ceil_div(QuantMapSize {}, Threads{}));
    using G2STiledCopySizeQM2 = decltype(ceil_div(QuantMapSize2{}, Threads{}));

    using G2SA_M   = TileM;
    using G2SQ_P   = TileP;
    using G2SQ2_P2 = TileP2;
    using G2SS_N   = decltype(cute::min(TileN{}, Threads{}));  // `G2SS_N` cannot exceed `Threads`, otherwise `G2SS_G` will be zero.
    using G2SA_K   = decltype(Threads{} / G2SA_M{});
    using G2SQ_K   = decltype(Threads{} / G2SQ_P{});
    using G2SQ2_K  = decltype(Threads{} / G2SQ2_P2{});
    using G2SS_G   = decltype(Threads{} / G2SS_N{});

    using G2SCopyAtomA   = typename G2SCopyTraits<T , G2STiledCopySizeA  >::Atom;
    using G2SCopyAtomQ   = typename G2SCopyTraits<TQ, G2STiledCopySizeQ  >::Atom;
    using G2SCopyAtomQ2  = typename G2SCopyTraits<TQ, G2STiledCopySizeQ2 >::Atom;
    using G2SCopyAtomS   = typename G2SCopyTraits<T , G2STiledCopySizeS  >::Atom;
    using G2SCopyAtomQM  = Copy_Atom<DefaultCopy, T>;  // the copy size is too small for `cp.async`
    using G2SCopyAtomQM2 = typename G2SCopyTraits<T2, G2STiledCopySizeQM2>::Atom;

    // https://zhuanlan.zhihu.com/p/664671157 (Section 2.2.2)
    // Similarly, we choose many atoms needed to cover the 4 sub-partitions of the SM (some multiple of 128 thread).
    // https://github.com/NVIDIA/cutlass/blob/v3.4.0/test/unit/gemm/device/default_gemm_configuration.hpp#L90
    // This operation uses Threads threads in a G2SA_M x G2SA_K shape. Each thread loads 1x8 elements
    // of 16-bits to cover, in total, a [G2SA_M, G2SA_K x 8] shape.

    using G2STiledCopyA   = decltype(make_tiled_copy(
        G2SCopyAtomA{},
        make_layout(make_shape(G2SA_M{}, G2SA_K{}), make_stride(G2SA_K{}, _1{})),
        make_layout(make_shape(_1{}, G2STiledCopySizeA{}))));

    using G2STiledCopyQ   = decltype(make_tiled_copy(
        G2SCopyAtomQ{},
        make_layout(make_shape(G2SQ_P{}, G2SQ_K{}), make_stride(G2SQ_K{}, _1{})),
        make_layout(make_shape(_1{}, G2STiledCopySizeQ{}))));

    using G2STiledCopyQ2  = decltype(make_tiled_copy(
        G2SCopyAtomQ2{},
        make_layout(make_shape(G2SQ2_P2{}, G2SQ2_K{}), make_stride(G2SQ2_K{}, _1{})),
        make_layout(make_shape(_1{}, G2STiledCopySizeQ2{}))));

    using G2STiledCopyS   = decltype(make_tiled_copy(
        G2SCopyAtomS{},
        make_layout(make_shape(G2SS_N{}, G2SS_G{}), make_stride(G2SS_G{}, _1{})),
        make_layout(make_shape(_1{}, G2STiledCopySizeS{}))));

    // each thread load one element, with predication.
    using G2STiledCopyQM  = decltype(make_tiled_copy(
        G2SCopyAtomQM{},
        make_layout(make_shape(Threads{})),
        make_layout(make_shape(G2STiledCopySizeQM{}))));

    using G2STiledCopyQM2 = decltype(make_tiled_copy(
        G2SCopyAtomQM2{},
        make_layout(make_shape(Threads{})),
        make_layout(make_shape(G2STiledCopySizeQM2{}))));

    using G2STiledCopyShapeA   = decltype(shape(typename G2STiledCopyA  ::Tiler_MN{}));
    using G2STiledCopyShapeQ   = decltype(shape(typename G2STiledCopyQ  ::Tiler_MN{}));
    using G2STiledCopyShapeQ2  = decltype(shape(typename G2STiledCopyQ2 ::Tiler_MN{}));
    using G2STiledCopyShapeS   = decltype(shape(typename G2STiledCopyS  ::Tiler_MN{}));
    using G2STiledCopyShapeQM  = decltype(shape(typename G2STiledCopyQM ::Tiler_MN{}));
    using G2STiledCopyShapeQM2 = decltype(shape(typename G2STiledCopyQM2::Tiler_MN{}));

    // CUTE_STATIC_ASSERT_V(G2SQ_P{} == _16{});  // Not supported yet
    CUTE_STATIC_ASSERT_V(Threads{} == size(G2STiledCopyA{}));
    CUTE_STATIC_ASSERT_V(Threads{} == size(G2STiledCopyQ{}));
    CUTE_STATIC_ASSERT_V(Threads{} == size(G2STiledCopyQ2{}));
    CUTE_STATIC_ASSERT_V(Threads{} == size(G2STiledCopyS{}));
    CUTE_STATIC_ASSERT_V(Threads{} == size(G2STiledCopyQM{}));
    CUTE_STATIC_ASSERT_V(Threads{} == size(G2STiledCopyQM2{}));
    CUTE_STATIC_ASSERT_V(TileM {}         % size<0>(G2STiledCopyShapeA  {}) == _0{});
    CUTE_STATIC_ASSERT_V(TileK {}         % size<1>(G2STiledCopyShapeA  {}) == _0{});
    CUTE_STATIC_ASSERT_V(TileP {}         % size<0>(G2STiledCopyShapeQ  {}) == _0{});
    CUTE_STATIC_ASSERT_V(TileK {}         % size<1>(G2STiledCopyShapeQ  {}) == _0{});
    CUTE_STATIC_ASSERT_V(TileP2{}         % size<0>(G2STiledCopyShapeQ2 {}) == _0{});
    CUTE_STATIC_ASSERT_V(TileK {}         % size<1>(G2STiledCopyShapeQ2 {}) == _0{});
    CUTE_STATIC_ASSERT_V(TileN {}         % size<0>(G2STiledCopyShapeS  {}) == _0{});
    CUTE_STATIC_ASSERT_V(TileG {}         % size<1>(G2STiledCopyShapeS  {}) == _0{});
    CUTE_STATIC_ASSERT_V(QuantMapSize {} <= size   (G2STiledCopyShapeQM {}));
    CUTE_STATIC_ASSERT_V(QuantMapSize2{} <= size   (G2STiledCopyShapeQM2{}));
    CUTE_STATIC_ASSERT_V(QuantMapSize {} <= Warps{});
    CUTE_STATIC_ASSERT_V(QuantMapSize {} <= Threads{});

    //
    // --- Shared Memory to Shared Memory Copy ---
    //

    using S2STiledCopySizeQM3 = _4;  // 128b = 4 x 32b
    using S2SQM3_1            = decltype(ceil_div(QuantMapDuplicates{}, S2STiledCopySizeQM3{}));
    using S2SQM3_0            = decltype(ceil_div(Threads           {}, S2SQM3_1           {}));
    using S2SCopyAtomQM3      = Copy_Atom<DefaultCopy, T2>;
    using S2STiledCopyQM3     = decltype(make_tiled_copy(
        S2SCopyAtomQM3{},
        make_layout(make_shape(S2SQM3_0{}, S2SQM3_1{}), make_stride(S2SQM3_1{}, _1{})),
        make_layout(make_shape(_1{}, S2STiledCopySizeQM3{}))));

    using S2STiledCopyShapeQM3 = decltype(shape(typename S2STiledCopyQM3::Tiler_MN{}));
    CUTE_STATIC_ASSERT  (sizeof(T2) == 4);
    CUTE_STATIC_ASSERT_V((Threads           {} == size   (S2STiledCopyQM3{})));
    CUTE_STATIC_ASSERT_V((QuantMapSize2     {}  % size<0>(S2STiledCopyShapeQM3{}) == _0{}) || (typename QuantMapVecTraits::IsDuplicated{} == false_type{}));
    CUTE_STATIC_ASSERT_V((QuantMapDuplicates{}  % size<1>(S2STiledCopyShapeQM3{}) == _0{}) || (typename QuantMapVecTraits::IsDuplicated{} == false_type{}));

    //
    // --- Shared Memory to Registers Copy ---
    //
    using S2RCopyOpA       = SM75_U32x4_LDSM_N;
    using S2RCopyOpQ       = conditional_t<is_same_v<MmaPrmK, _1>, SM75_U32x2_LDSM_N, SM75_U32x4_LDSM_N>;
    using S2RCopyOpQ2      = SM75_U32x4_LDSM_N;

    using S2RCopyTraitsA   = Copy_Traits<S2RCopyOpA>;
    using S2RCopyTraitsQ   = Copy_Traits<S2RCopyOpQ>;
    using S2RCopyTraitsQ2  = Copy_Traits<S2RCopyOpQ2>;

    using S2RCopyAtomA     = Copy_Atom<S2RCopyTraitsA , T>;
    using S2RCopyAtomQ     = Copy_Atom<S2RCopyTraitsQ , TQ>;
    using S2RCopyAtomQ2    = Copy_Atom<S2RCopyTraitsQ2, TQ>;
    using S2RCopyAtomSView = Copy_Atom<DefaultCopy    , T>;
    using S2RCopyAtomQM    = Copy_Atom<DefaultCopy    , T>;

    //
    // --- Epilogue (Register to Global via Shared Memory) ---
    //
    using TiledMmaM = Int<tile_size<0>(TiledMma{})>;
    using TiledMmaN = Int<tile_size<1>(TiledMma{})>;
    using SmemLayoutAtomC = decltype(composition(Swizzle<2, 3, 3>{}, make_layout(make_shape(TiledMmaM{}, TiledMmaN{}), make_stride(TiledMmaN{}, _1{}))));
    using SmemLayoutC     = decltype(tile_to_shape(SmemLayoutAtomC{}, make_shape(TileM{}, TileN{})));
    // for some reason, the compiler will complain if we use `cosize_t` directly
    using SmemLayoutCSize = cosize_t<decltype(make_layout(make_shape(TileM{}, TileN{})))>;

    // https://github.com/NVIDIA/cutlass/blob/v3.4.0/examples/50_hopper_gemm_with_epilogue_swizzle/50_hopper_gemm_with_epilogue_swizzle.cu
    // note that all of these copy atoms are of type `T` instead of `TC` because we will convert
    // the accumulated values to `T` before the epilogue. Performing epilogue type conversions before
    // these copy operations can both save memory bandwidth and size of epilogue shared memory buffer.
    using R2SCopyAtomC = Copy_Atom<DefaultCopy, T>;
    using S2RCopyAtomC = Copy_Atom<UniversalCopy<uint128_t>, T>;
    using R2GCopyAtomC = Copy_Atom<UniversalCopy<uint128_t>, T>;

    using S2RC_M        = TileM;
    using S2RC_N        = decltype(Threads{} / S2RC_M{});
    using S2RTiledCopyC = decltype(make_tiled_copy(
        S2RCopyAtomC{},
        make_layout(make_shape(S2RC_M{}, S2RC_N{}), make_stride(S2RC_N{}, _1{})),
        make_layout(make_shape(_1{}, EpilogueCopySizeC{}))));

    using S2RTiledCopyCShape = decltype(shape(typename S2RTiledCopyC::Tiler_MN{}));
    CUTE_STATIC_ASSERT_V(TileM{} % size<0>(S2RTiledCopyCShape{}) == _0{});
    CUTE_STATIC_ASSERT_V(TileN{} % size<1>(S2RTiledCopyCShape{}) == _0{});

    //
    // --- Shared Memory Size ---
    //

    struct SharedStorage
    {

        array_aligned<T , cosize_v<SmemLayoutA >> smem_A;
        array_aligned<TQ, cosize_v<SmemLayoutQ >> smem_Q;
        array_aligned<T , cosize_v<SmemLayoutS >> smem_S;
        array_aligned<T , cosize_v<SmemLayoutQM>> smem_QM;

        // optional
        static constexpr int kSmemLayoutQ2Size  = conditional_t<is_same_v<NumBits, _3>                                        , cosize_t<SmemLayoutQ2 >, _0>::value;
        static constexpr int kSmemLayoutQM2Size = conditional_t<is_same_v<typename QuantMapVecTraits::IsVectorized, true_type>, cosize_t<SmemLayoutQM2>, _0>::value;
        static constexpr int kSmemLayoutQM3Size = conditional_t<is_same_v<typename QuantMapVecTraits::IsDuplicated, true_type>, cosize_t<SmemLayoutQM3>, _0>::value;
        static constexpr int kSmemLayoutCSize   = conditional_t<DecompositionMode == DecompositionModeEnum::StreamK           , SmemLayoutCSize        , _0>::value;
        array_aligned<TQ, kSmemLayoutQ2Size > smem_Q2;
        array_aligned<T2, kSmemLayoutQM2Size> smem_QM2;
        array_aligned<T2, kSmemLayoutQM3Size> smem_QM3;
        array_aligned<T , kSmemLayoutCSize  > smem_C;
    };

    static constexpr int kSmemSize = int(sizeof(SharedStorage));

};

}  // namespace config