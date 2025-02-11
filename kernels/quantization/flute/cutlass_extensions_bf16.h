#pragma once

#include "cutlass/block_striped.h"


namespace cutlass {


/// Utility for performing block-striped access (load, store, reduce) of trivially-copyable,
/// statically-sized array types to global memory.
/// (Specialization for bfloat16_t.  Uses nv_bfloat162 vectorized-reduction.)
template <
  int BlockThreads,
  typename ArrayT>
struct BlockStripedReduce<BlockThreads, ArrayT, bfloat16_t> :
  BlockStriped<
    BlockThreads,
    ArrayT,
    nv_bfloat162>
{
  static_assert(BlockStripedReduce::kStripes % 2 == 0, "Array of half must be even number in length");

  /// Reduce
  CUTLASS_DEVICE
  static void reduce(ArrayT *ptr, const ArrayT &data, int thread_idx)
  {
    // This operation is natively supported by devices of compute
    // capability 9.x and higher, older devices use emulation path
    cutlass::atomic_add<nv_bfloat162> reduce;
    nv_bfloat162 *access_output = reinterpret_cast<nv_bfloat162*>(ptr);
    const nv_bfloat162 *access_data = reinterpret_cast<const nv_bfloat162*>(&data);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < BlockStripedReduce::kStripes; ++i)
    {
      reduce(access_output + (BlockThreads * i) + thread_idx, access_data[i]);
    }
  }
};


} // namespace cutlass