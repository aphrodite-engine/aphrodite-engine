#pragma once

#include "cute/tensor.hpp"
#include "cutlass/numeric_conversion.h"


namespace conversion_utils {
using namespace cute;


// https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp#L1260
template <
    class EngineSource,
    class EngineTarget,
    class TensorLayout,
    int ConversionVectorWidth = cosize_v<TensorLayout>
>
CUTLASS_DEVICE void
convert_tensor(
    Tensor<EngineSource, TensorLayout> const& source,
    Tensor<EngineTarget, TensorLayout>      & target,
    cute::Int<ConversionVectorWidth> width = {})
{

    /// This is an element-wise conversion where we expect both tensors to have the same layout.
    /// As a result, we can cast as a cutlass array to use the fast numeric converters without 
    /// worrying about indexing into the layout.
    constexpr int N = cosize_v<TensorLayout>; 

    /// The inputs must be backed by registers & be statically sized.
    static_assert(is_rmem<EngineSource>::value, "Input tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineTarget>::value, "Output tensor for A conversion must come from registers");
    static_assert(is_static_v<TensorLayout>, "Tensor layout for the conversion must be static");
    static_assert(cosize_v<TensorLayout> == size(TensorLayout{}), "Cosize and size of the layout must be equal.");
    static_assert(N % ConversionVectorWidth == 0, "Conversion vector width must divide cosize of the tensor layout.");

    using SrcType = typename EngineSource::value_type;
    using DstType = typename EngineTarget::value_type;

    using SrcArray = cutlass::Array<SrcType, ConversionVectorWidth>;
    using DstArray = cutlass::Array<DstType, ConversionVectorWidth>;

    constexpr cutlass::FloatRoundStyle RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
    using Converter = cutlass::NumericArrayConverter<DstType, SrcType, ConversionVectorWidth, RoundStyle>;

    constexpr int NumIterations = N / ConversionVectorWidth;

    for (int ii = 0; ii < NumIterations; ++ii)
    {
        SrcArray const* src_array_ptr = reinterpret_cast<SrcArray const*>(raw_pointer_cast(source.data())) + ii;
        DstArray* dst_array_ptr = reinterpret_cast<DstArray*>(raw_pointer_cast(target.data())) + ii;
        *dst_array_ptr = Converter::convert(*src_array_ptr);
    }
}


} // namespace conversion_utils