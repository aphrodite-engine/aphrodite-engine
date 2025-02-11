#pragma once

#include <cuda.h>
#include <stdio.h>
#include <cute/tensor.hpp>
#include "cutlass/array.h"
#include "cutlass/barrier.h"
#include "cutlass/fast_math.h"
#include "cutlass/block_striped.h"

// custom extensions to make `BlockStripedReduce` work with `nv_bfloat162`
#include "cutlass_extensions_bf16.h"

// debugging helpers
#define TS_DEBUG 0  // 1
#define TS_DEBUG_THR 0  // 255
#define TS_DEBUG_BLK 0  // 131071
#define PPRINT_HEADER() do { print("[thread: %d, block: %d]\t", TS_DEBUG_THR, TS_DEBUG_BLK); } while(0)
#define BACKWARDS 1


// 2/4
namespace config {

using namespace cute;


// Pad the given allocation size up to the nearest cache line
CUTE_HOST_DEVICE static
size_t
cacheline_align_up(size_t size)
{
    static const int CACHELINE_SIZE = 128;
    return (size + CACHELINE_SIZE - 1) / CACHELINE_SIZE * CACHELINE_SIZE;
}


// Get the workspace size needed for intermediate partial sums
CUTE_HOST_DEVICE
size_t
get_workspace_size_partials(int sk_blocks, int threads, int accum_size)
{
    return cacheline_align_up(sk_blocks * threads * accum_size);
}


// Get the workspace size needed for barrier
CUTE_HOST_DEVICE
size_t
get_workspace_size_barriers(int sk_blocks)
{
    // For atomic reduction, each SK-block needs a synchronization flag.  For parallel reduction,
    // each reduction block needs its own synchronization flag.
    return cacheline_align_up(sizeof(typename cutlass::Barrier::T) * sk_blocks);
}


template <typename Config_>
struct FixupHelper
{

    using Config  = Config_;
    using Threads = typename Config::Threads;
    static constexpr ReductionModeEnum ReductionMode = ReductionModeEnum::Nondeterministic;

    // Share accumulators with peers
    template <typename Tensor>
    CUTE_DEVICE static
    void
    initialize_or_accumulate(
        Tensor& accum,
        int     thread_index,
        int     block_index,
        int     block_index_first,
        void*   workspace_partials,
        void*   workspace_barriers)
    {

        using AccumulatorArrayT   = cutlass::Array<typename Tensor::value_type, size(Tensor{})>;
        using BlockStripedReduceT = cutlass::BlockStripedReduce<Threads{}, AccumulatorArrayT>;

        auto accum_array     = reinterpret_cast<AccumulatorArrayT*>(&accum);
        auto workspace_accum = reinterpret_cast<AccumulatorArrayT*>(workspace_partials);
        auto workspace_index = block_index_first * Threads{};

#if TS_DEBUG

        using BarrierT = typename cutlass::Barrier::T;
        BarrierT* flag_ptr = reinterpret_cast<BarrierT*>(workspace_barriers) + block_index_first;

        if(thread(TS_DEBUG_THR, TS_DEBUG_BLK))
        {
            print("\n--------------- Fixup ---------------\n");
            PPRINT_HEADER(); print("type:              %s\n", (block_index == block_index_first) ? "initialization" : "accumulation"); 
            PPRINT_HEADER(); print("thread_index:      %d\n", thread_index);
            PPRINT_HEADER(); print("block_index:       %d\n", block_index);
            PPRINT_HEADER(); print("block_index_first: %d\n", block_index_first);
            PPRINT_HEADER(); print("workspace_index:   %d\n", workspace_index);
            PPRINT_HEADER(); print("blocks_to_wait:    %d\n", BACKWARDS ? block_index_first - block_index : block_index - block_index_first);
            PPRINT_HEADER(); print("flag_index:        %d\n", block_index_first);
            PPRINT_HEADER(); print("flag_value (old):  %d\n", *flag_ptr);
        }
#endif

        if (block_index == block_index_first)
        {
            // First peer initializes the workspace partials
            BlockStripedReduceT::store(workspace_accum + workspace_index, *accum_array, thread_index);
        }
        else
        {
            // Subsequent peers atomically accumulate into the workspace partials
            if constexpr (ReductionMode == ReductionModeEnum::Nondeterministic)
            {
                // Non-deterministic reduction order: wait for the first peer to have initialized the partials before we add to them
                cutlass::Barrier::wait_lt(workspace_barriers, thread_index, block_index_first, 1);
            }
            else
            {
                // Turnstile reduction order: wait until the previous peer has written
#if BACKWARDS
                // in the BACKWARDS case, we define the `block_index_first` as the last
                // logical block, hence all non-first blocks will have lower logical index
                auto blocks_to_wait = block_index_first - block_index;
#else
                auto blocks_to_wait = block_index - block_index_first;
#endif
                cutlass::Barrier::wait_eq(workspace_barriers, thread_index, block_index_first, blocks_to_wait);
            }

            // Perform reduction in workspace
            BlockStripedReduceT::reduce(workspace_accum + workspace_index, *accum_array, thread_index);
        }

        // Signal our arrival
        cutlass::Barrier::arrive_inc(workspace_barriers, thread_index, block_index_first);

#if TS_DEBUG

        if(thread(TS_DEBUG_THR, TS_DEBUG_BLK))
        {
            PPRINT_HEADER(); print("flag_value (new):  %d\n", *flag_ptr);
        }
#endif

    }

    // Acquire accumulators from peers
    template <typename Tensor>
    CUTE_DEVICE static
    void
    acquire(
        Tensor& accum,
        int     thread_index,
        int     block_index,
        int     block_index_first,
        void*   workspace_partials,
        void*   workspace_barriers)
    {

        using AccumulatorArrayT   = cutlass::Array<typename Tensor::value_type, size(Tensor{})>;
        using BlockStripedReduceT = cutlass::BlockStripedReduce<Threads{}, AccumulatorArrayT>;

        auto accum_array     = reinterpret_cast<AccumulatorArrayT*>(&accum);
        auto workspace_accum = reinterpret_cast<AccumulatorArrayT*>(workspace_partials);
        auto workspace_index = block_index_first * Threads{};
#if BACKWARDS
        // in the BACKWARDS case, we define the `block_index_first` as the last
        // logical block, hence all non-first blocks will have lower logical index
        auto blocks_to_wait  = block_index_first - block_index;
#else
        auto blocks_to_wait  = block_index - block_index_first;
#endif


#if TS_DEBUG

        using BarrierT = typename cutlass::Barrier::T;
        BarrierT* flag_ptr = reinterpret_cast<BarrierT*>(workspace_barriers) + block_index_first;

        if(thread(TS_DEBUG_THR, TS_DEBUG_BLK))
        {
            print("\n--------------- Fixup ---------------\n");
            PPRINT_HEADER(); print("type:              %s\n", "acquire"); 
            PPRINT_HEADER(); print("thread_index:      %d\n", thread_index);
            PPRINT_HEADER(); print("block_index:       %d\n", block_index);
            PPRINT_HEADER(); print("block_index_first: %d\n", block_index_first);
            PPRINT_HEADER(); print("workspace_index:   %d\n", workspace_index);
            PPRINT_HEADER(); print("blocks_to_wait:    %d\n", blocks_to_wait);
            PPRINT_HEADER(); print("flag_index:        %d\n", block_index_first);
            PPRINT_HEADER(); print("flag_value (old):  %d\n", *flag_ptr);
        }
#endif

        // Wait for arrival
        cutlass::Barrier::wait_eq_reset(workspace_barriers, thread_index, block_index_first, blocks_to_wait);

        // Load and add peer-partials accumulator tile to local accumulator tile
        BlockStripedReduceT::load_add(*accum_array, workspace_accum + workspace_index, thread_index);

#if TS_DEBUG

        if(thread(TS_DEBUG_THR, TS_DEBUG_BLK))
        {
            PPRINT_HEADER(); print("flag_value (new):  %d\n", *flag_ptr);
        }
#endif

    }

};


template <typename Config_>
class TileScheduler
{

private:

    using Config         = Config_;
    using FixupHelperT   = FixupHelper<Config>;
    using Threads        = typename Config::Threads;
    using TileM          = typename Config::TileM;
    using TileN          = typename Config::TileN;
    using TileK          = typename Config::TileK;
    using TileP          = typename Config::TileP;
    using TileP2         = typename Config::TileP2;
    using TileG          = typename Config::TileG;
    using NumBits        = typename Config::NumBits;
    using GroupSize      = typename Config::GroupSize;
    using TileKsPerTileG = typename Config::TileKsPerTileG;
    static constexpr int kSmemSize = Config::kSmemSize;
    static constexpr DecompositionModeEnum DecompositionMode = Config::DecompositionMode;
    // CUTE_STATIC_ASSERT(((DecompositionMode == DecompositionModeEnum::SplitK ) && decltype(Blocks{} == _0{})::value) ||
    //                    ((DecompositionMode == DecompositionModeEnum::StreamK) && decltype(Slices{} == _0{})::value));

    //
    // Member state
    //

    int m_slices;
    int m_blocks;

    int m_M;
    int m_N;
    int m_K;
    int m_P;
    int m_P2;
    int m_G;
    int m_tiles;
    int m_tiles_M;
    int m_tiles_N;
    int m_tiles_K;
    int m_tiles_P;
    int m_tile_index;
    int m_tile_index_read;
    int m_tiles_per_block;
    int m_tiles_this_block;
    int m_tiles_typical_streamk;
    int m_tiles_special_streamk;
    int m_blocks_typical_streamk;
    int m_blocks_special_streamk;
    int m_smem_pipe_read_G_offset;

    CUTE_DEVICE
    auto
    get_block_index_streamk() const
    {
#if BACKWARDS
        return m_blocks - blockIdx.x - 1;
#else
        return blockIdx.x;
#endif
    }

    CUTE_DEVICE
    auto
    get_global_tile_index_streamk(int tile_index) const
    {
        // when the CTA is at `block_index`, it means it is at `block_index + 1` block,
        // and there are `block_index` blocks before it. We then need to figure out
        // how many of the previous `block_index` blocks are typical blocks and how many
        // are special blocks.
        auto block_index    = get_block_index_streamk();
        auto blocks_typical = cute::min(block_index, m_blocks_typical_streamk);
        auto blocks_special = cute::max(block_index, m_blocks_typical_streamk) - m_blocks_typical_streamk;
        return tile_index +
               blocks_typical * (m_tiles_per_block) +
               blocks_special * (m_tiles_per_block + 1);
    }

    CUTE_DEVICE
    auto
    get_tiles_this_block() const
    {
        if constexpr (DecompositionMode == DecompositionModeEnum::SplitK)
        {
            // we assume that partitioning is even in Split-K
            return m_tiles_per_block;
        }
        else
        {
            // we assume that the special blocks are at the end
            return (get_block_index_streamk() < m_blocks_typical_streamk)
                    ? m_tiles_per_block
                    : m_tiles_per_block + 1;
        }
    }

    CUTE_DEVICE
    auto
    get_tile_coord(int tile_index) const
    {
        if constexpr (DecompositionMode == DecompositionModeEnum::SplitK)
        {
            auto tile_M_index = blockIdx.y;
            auto tile_N_index = blockIdx.x;  // == tile_P_index == tile_P2_index (3-bit case)
            auto slice_index  = blockIdx.z;
            auto tile_K_index = slice_index * m_tiles_per_block + tile_index;
            return make_coord(tile_M_index, tile_N_index, tile_K_index);
        }
        else
        {
            auto tiles_shape       = make_shape(m_tiles_M, m_tiles_N, m_tiles_K);
            auto tiles_layout      = make_layout(tiles_shape, LayoutRight{});
            auto global_tile_index = get_global_tile_index_streamk(tile_index);
            return tiles_layout.get_hier_coord(global_tile_index);
        }
    }

    // Compute tile residues for predication
    // https://github.com/NVIDIA/cutlass/blob/v3.4.0/include/cutlass/gemm/kernel/sm70_gemm.hpp#L227

    CUTE_DEVICE
    auto
    get_residue_M(int tile_index) const
    {
        auto tile_coord   = get_tile_coord(tile_index);
        auto tile_M_index = get<0>(tile_coord);
        return m_M - TileM{} * tile_M_index;
    }

    CUTE_DEVICE
    auto
    get_residue_N(int tile_index) const
    {
        auto tile_coord   = get_tile_coord(tile_index);
        auto tile_N_index = get<1>(tile_coord);
        return m_N - TileN{} * tile_N_index;
    }

    CUTE_DEVICE
    auto
    get_residue_P(int tile_index) const
    {
        auto tile_coord   = get_tile_coord(tile_index);
        auto tile_P_index = get<1>(tile_coord);
        return m_P - TileP{} * tile_P_index;
    }

    CUTE_DEVICE
    auto
    get_residue_P2(int tile_index) const
    {
        auto tile_coord    = get_tile_coord(tile_index);
        auto tile_P2_index = get<1>(tile_coord);
        return m_P2 - TileP2{} * tile_P2_index;
    }

    template <
        typename PrdEngine, typename PrdLayout,
        typename CrdEngine, typename CrdLayout
    >
    CUTE_DEVICE
    auto
    set_predicates(
        cute::Tensor<PrdEngine, PrdLayout>      & pred,
        cute::Tensor<CrdEngine, CrdLayout> const& coord,
        int residue) const
    {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<0>(pred); ++i) {
            // tile coord < residue
            pred(i, 0) = get<0>(coord(0, i, 0)) < residue;
        }
    }

    CUTE_DEVICE
    bool
    finished_output_tile(int tile_index) const
    {
        auto tile_coord   = get_tile_coord(tile_index);
        auto tile_K_index = get<2>(tile_coord);
        return (tile_K_index == m_tiles_K - 1);
    }

    CUTE_DEVICE
    bool
    is_last_tile(int tile_index) const
    {
        return (tile_index == m_tiles_this_block - 1);
    }

    CUTE_DEVICE
    bool
    started_output_tile(int tile_index) const
    {
        // We assume that CTA processes tiles in order. If we are at `tile_K_index`
        // and have processed `tile_index > tile_K_index`, then we have started the
        // output tile. Similarly, if we are at `tile_K_index` and have processed
        // `tile_index < tile_K_index`, then we have not started the output tile.
        auto tile_coord   = get_tile_coord(tile_index);
        auto tile_K_index = get<2>(tile_coord);
        return tile_K_index <= tile_index;
    }

public:

    // Default Constructor
    TileScheduler() = default;

    // Constructor
    TileScheduler(
        int const M_,
        int const N_,
        int const K_,
        int const P_,
        int const slices_,
        int const blocks_)
    :
        m_slices(slices_),
        m_blocks(blocks_),
        m_M(M_),
        m_N(N_),
        m_K(K_),
        m_P(P_),
        m_P2(0),
        m_tile_index(0),
        m_tile_index_read(0),
        m_tiles_typical_streamk(0),
        m_tiles_special_streamk(0),
        m_blocks_typical_streamk(0),
        m_blocks_special_streamk(0)
    {

        // in the 3-bit case, we split Q into two sub-matrices
        // Q : [N / (sizeof(TQ) / 1), K], the first 1 bits
        // Q2: [N / (sizeof(TQ) / 2), K], the remaining 2 bits
        if constexpr (is_same_v<NumBits, _3>) {
            m_P  = ceil_div(m_N, 16);  // N / (sizeof(TQ) / 1) == N / (P / 3 * 1)
            m_P2 = ceil_div(m_N, 8);   // N / (sizeof(TQ) / 2) == N / (P / 3 * 2)

            // in an older version, we set
            // `tiles_P = P_ / SuperTileP = (N x 3 / 16) / (TileP + 2 x TileP)`
            // here, we instead set
            // `tiles_P = P / TileP = N / 16 / TileP`
        }

        // number of groups
        m_G       = ceil_div(m_K, GroupSize{});
        // Note that `tiles_N == tiles_P == tiles_P2 (3-bit case)`
        m_tiles_M = ceil_div(m_M, TileM{});
        m_tiles_N = ceil_div(m_N, TileN{});
        m_tiles_K = ceil_div(m_K, TileK{});
        m_tiles_P = ceil_div(m_P, TileP{});
        m_tiles   = m_tiles_M * m_tiles_N * m_tiles_K;

        if constexpr (DecompositionMode == DecompositionModeEnum::SplitK)
        {
            // we assume that K is divisible by TileK * Slices
            m_tiles_per_block = m_tiles_K / m_slices;
        }
        else
        {
            // the last `m_tiles_remaining` logical blocks will have one extra tile,
            // while the rest will have `m_tiles_per_block` tiles
            m_tiles_per_block        = m_tiles / m_blocks;
            m_blocks_special_streamk = m_tiles - m_tiles_per_block * m_blocks;
            m_blocks_typical_streamk = m_blocks - m_blocks_special_streamk;
            m_tiles_typical_streamk  = m_blocks_typical_streamk * (m_tiles_per_block);
            m_tiles_special_streamk  = m_blocks_special_streamk * (m_tiles_per_block + 1);
        }

#if TS_DEBUG
        dim3 grid_dim = grid();
        print("\n--------------- TileScheduler ---------------\n");
        print("M : %5d \t TileM : %5d \t tiles_M: %5d \n", m_M , TileM ::value, m_tiles_M);
        print("N : %5d \t TileN : %5d \t tiles_N: %5d \n", m_N , TileN ::value, m_tiles_N);
        print("K : %5d \t TileK : %5d \t tiles_K: %5d \n", m_K , TileK ::value, m_tiles_K);
        print("P : %5d \t TileP : %5d \t tiles_P: %5d \n", m_P , TileP ::value, m_tiles_P);
        print("P2: %5d \t TileP2: %5d                 \n", m_P2, TileP2::value);
        print("G:  %5d \t TileG : %5d                 \n", m_G , TileG ::value);
        print("tiles: %5d (%5d per block) \n", m_tiles, m_tiles_per_block);
        print("tiles_typical_streamk:  %5d \n", m_tiles_typical_streamk);
        print("tiles_special_streamk:  %5d \n", m_tiles_special_streamk);
        print("blocks_typical_streamk: %5d \n", m_blocks_typical_streamk);
        print("blocks_special_streamk: %5d \n", m_blocks_special_streamk);
        print("grid_dim: (%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z);
#endif

    }

    template <
        typename PrdEngineA , typename PrdLayoutA ,
        typename PrdEngineQ , typename PrdLayoutQ ,
        typename PrdEngineQ2, typename PrdLayoutQ2,
        typename PrdEngineS , typename PrdLayoutS ,
        typename CrdEngineA , typename CrdLayoutA ,
        typename CrdEngineQ , typename CrdLayoutQ ,
        typename CrdEngineQ2, typename CrdLayoutQ2,
        typename CrdEngineS , typename CrdLayoutS
    >
    CUTE_DEVICE
    void
    initialize(
        cute::Tensor<PrdEngineA , PrdLayoutA >      & pred_A,
        cute::Tensor<PrdEngineQ , PrdLayoutQ >      & pred_Q,
        cute::Tensor<PrdEngineQ2, PrdLayoutQ2>      & pred_Q2,
        cute::Tensor<PrdEngineS , PrdLayoutS >      & pred_S,
        cute::Tensor<CrdEngineA , CrdLayoutA > const& coord_A,
        cute::Tensor<CrdEngineQ , CrdLayoutQ > const& coord_Q,
        cute::Tensor<CrdEngineQ2, CrdLayoutQ2> const& coord_Q2,
        cute::Tensor<CrdEngineS , CrdLayoutS > const& coord_S)
    {
        m_tile_index       = 0;
        m_tile_index_read  = 0;
        m_tiles_this_block = get_tiles_this_block();

#if TS_DEBUG

        if(thread(TS_DEBUG_THR, TS_DEBUG_BLK))
        {
            print("\n--------------- Initialization ---------------\n");
            PPRINT_HEADER(); print("tile_index:              %d (ended = %d)\n", m_tile_index     , !tile_is_in_bound());
            PPRINT_HEADER(); print("tile_index_read:         %d (ended = %d)\n", m_tile_index_read, !tile_read_is_in_bound());
            PPRINT_HEADER(); print("tiles_this_block:        %d\n", m_tiles_this_block);
        }
#endif

        if constexpr (DecompositionMode == DecompositionModeEnum::StreamK)
        {
            // the starting K tile index might not be aligned wth the G tile
            auto tile_coord_init      = get_tile_coord(0);
            auto tile_K_index_init    = get<2>(tile_coord_init);
            m_smem_pipe_read_G_offset = tile_K_index_init % TileKsPerTileG{};

#if TS_DEBUG

            if(thread(TS_DEBUG_THR, TS_DEBUG_BLK))
            {
                PPRINT_HEADER(); print("tile_coord_init: "); print(tile_coord_init); print("\n");
                PPRINT_HEADER(); print("smem_pipe_read_G_offset: %d\n", m_smem_pipe_read_G_offset);
            }
#endif

        }

        set_predicates(pred_A , coord_A , get_residue_M (m_tile_index_read));
        set_predicates(pred_Q , coord_Q , get_residue_P (m_tile_index_read));
        set_predicates(pred_Q2, coord_Q2, get_residue_P2(m_tile_index_read));
        set_predicates(pred_S , coord_S , get_residue_N (m_tile_index_read));

#if TS_DEBUG

        if(thread(TS_DEBUG_THR, TS_DEBUG_BLK))
        {
            PPRINT_HEADER(); print("A : "); print(get_tile_coord_A ()); print("  \t residue_M : %5d \n", get_residue_M (m_tile_index_read));
            PPRINT_HEADER(); print("Q : "); print(get_tile_coord_Q ()); print("  \t residue_P : %5d \n", get_residue_P (m_tile_index_read));
            PPRINT_HEADER(); print("Q2: "); print(get_tile_coord_Q2()); print("  \t residue_P2: %5d \n", get_residue_P2(m_tile_index_read));
            PPRINT_HEADER(); print("S : "); print(get_tile_coord_S ()); print("  \t residue_N : %5d \n", get_residue_N (m_tile_index_read));
        }
#endif
    }

    template <
        typename PrdEngineA , typename PrdLayoutA ,
        typename PrdEngineQ , typename PrdLayoutQ ,
        typename PrdEngineQ2, typename PrdLayoutQ2,
        typename PrdEngineS , typename PrdLayoutS ,
        typename CrdEngineA , typename CrdLayoutA ,
        typename CrdEngineQ , typename CrdLayoutQ ,
        typename CrdEngineQ2, typename CrdLayoutQ2,
        typename CrdEngineS , typename CrdLayoutS
    >
    CUTE_DEVICE
    void
    step_read(
        cute::Tensor<PrdEngineA , PrdLayoutA >      & pred_A,
        cute::Tensor<PrdEngineQ , PrdLayoutQ >      & pred_Q,
        cute::Tensor<PrdEngineQ2, PrdLayoutQ2>      & pred_Q2,
        cute::Tensor<PrdEngineS , PrdLayoutS >      & pred_S,
        cute::Tensor<CrdEngineA , CrdLayoutA > const& coord_A,
        cute::Tensor<CrdEngineQ , CrdLayoutQ > const& coord_Q,
        cute::Tensor<CrdEngineQ2, CrdLayoutQ2> const& coord_Q2,
        cute::Tensor<CrdEngineS , CrdLayoutS > const& coord_S)
    {

        ++m_tile_index_read;

        // we don't need to reset predicates in Split-K
        if constexpr (DecompositionMode == DecompositionModeEnum::StreamK)
        {
            auto tile_coord_old = get_tile_coord(m_tile_index_read - 1);
            auto tile_coord_new = get_tile_coord(m_tile_index_read);
            // the K index changes, we need to update the predicates
            if (get<2>(tile_coord_old) != get<2>(tile_coord_new))
            {
                set_predicates(pred_A , coord_A , get_residue_M (m_tile_index_read));
                set_predicates(pred_Q , coord_Q , get_residue_P (m_tile_index_read));
                set_predicates(pred_Q2, coord_Q2, get_residue_P2(m_tile_index_read));
                set_predicates(pred_S , coord_S , get_residue_N (m_tile_index_read));
            }
        }

#if TS_DEBUG

        if(thread(TS_DEBUG_THR, TS_DEBUG_BLK))
        {
            print("\n--------------- Step Read ---------------\n");
            auto tile_coord = get_tile_coord(m_tile_index_read);
            PPRINT_HEADER(); print("tile_index_read: %d (ended = %d)\n", m_tile_index_read, !tile_read_is_in_bound());
            PPRINT_HEADER(); print("tile_coord: "); print(tile_coord); print("\n");
            PPRINT_HEADER(); print("A : "); print(get_tile_coord_A ()); print("  \t residue_M : %5d \n", get_residue_M (m_tile_index_read));
            PPRINT_HEADER(); print("Q : "); print(get_tile_coord_Q ()); print("  \t residue_P : %5d \n", get_residue_P (m_tile_index_read));
            PPRINT_HEADER(); print("Q2: "); print(get_tile_coord_Q2()); print("  \t residue_P2: %5d \n", get_residue_P2(m_tile_index_read));
            PPRINT_HEADER(); print("S : "); print(get_tile_coord_S ()); print("  \t residue_N : %5d \n", get_residue_N (m_tile_index_read));
        }
#endif

    }

    CUTE_DEVICE
    void
    step()
    {
        ++m_tile_index;

#if TS_DEBUG

        if(thread(TS_DEBUG_THR, TS_DEBUG_BLK))
        {
            print("\n--------------- Step ---------------\n");
            auto tile_coord = get_tile_coord(m_tile_index);
            PPRINT_HEADER(); print("tile_index:      %d (ended = %d)\n", m_tile_index     , !tile_is_in_bound());
            PPRINT_HEADER(); print("tile_coord: "); print(tile_coord); print("\n");
        }
#endif

    }

#if TS_DEBUG

    template <typename Tensor>
    CUTE_DEVICE
    void
    maybe_print_workspace_size(Tensor const& accum) const
    {

        if(thread(TS_DEBUG_THR, TS_DEBUG_BLK))
        {
            using AccumulatorArrayT = cutlass::Array<typename Tensor::value_type, size(Tensor{})>;
            dim3 grid_size = grid();
            auto blocks = grid_size.x * grid_size.y * grid_size.z;
            auto accum_size = sizeof(AccumulatorArrayT);
            auto workspace_size_barriers = get_workspace_size_barriers(blocks);
            auto workspace_size_partials = get_workspace_size_partials(blocks, Threads{}, accum_size);
            auto workspace_size = workspace_size_barriers + workspace_size_partials;

            print("\n--------------- Workspace Size ---------------\n");
            PPRINT_HEADER(); print("accum:                  "); print(accum.layout()); print("\n");
            PPRINT_HEADER(); print("blocks:                  %d\n", blocks);
            PPRINT_HEADER(); print("accum_size:              %d\n", accum_size);
            PPRINT_HEADER(); print("workspace_size_barriers: %d\n", workspace_size_barriers);
            PPRINT_HEADER(); print("workspace_size_partials: %d\n", workspace_size_partials);
            PPRINT_HEADER(); print("workspace_size:          %d\n", workspace_size);
        }
    }

    template <typename Coord>
    CUTE_DEVICE
    bool
    is_output_tile_coord(int tile_index, Coord const& coord) const
    {
        auto tile_coord = get_tile_coord(tile_index);
        return (get<0>(tile_coord) == get<0>(coord) &&
                get<1>(tile_coord) == get<1>(coord));
    }

#endif

    CUTE_DEVICE
    auto
    M() const
    {
        return m_M;
    }

    CUTE_DEVICE
    auto
    N() const
    {
        return m_N;
    }

    CUTE_DEVICE
    auto
    K() const
    {
        return m_K;
    }

    CUTE_DEVICE
    auto
    P() const
    {
        return m_P;
    }

    CUTE_DEVICE
    auto
    P2() const
    {
        return m_P2;
    }

    CUTE_DEVICE
    auto
    G() const
    {
        return m_G;
    }

    CUTE_DEVICE
    auto
    tile_index() const
    {
        return m_tile_index;
    }

    CUTE_DEVICE
    auto
    tile_index_read() const
    {
        return m_tile_index_read;
    }

    CUTE_DEVICE
    auto
    smem_pipe_read_G_offset() const
    {
        return m_smem_pipe_read_G_offset;
    }

    CUTE_HOST_DEVICE
    dim3
    grid() const
    {
        if constexpr (DecompositionMode == DecompositionModeEnum::SplitK)
        {
            // tiles_N == tiles_P == tiles_P2 (3-bit case)
            return dim3(m_tiles_N, m_tiles_M, m_slices);
        }
        else
        {
            return dim3(m_blocks);
        }
    }

    CUTE_HOST
    dim3
    block() const
    {
        return dim3(Threads{});
    }

    CUTE_HOST
    int
    smem_size() const
    {
        return kSmemSize;
    }

    CUTE_DEVICE
    auto
    workspace_size_barriers() const
    {
        int blocks;
        if constexpr (DecompositionMode == DecompositionModeEnum::SplitK)
        {
            blocks = m_tiles_N * m_tiles_M * m_slices;
        }
        else
        {
            blocks = m_blocks;
        }

        return get_workspace_size_barriers(blocks);
    }

    CUTE_DEVICE
    auto
    get_tile_coord_A() const
    {
        auto tile_coord   = get_tile_coord(m_tile_index_read);
        auto tile_M_index = get<0>(tile_coord);
        auto tile_K_index = get<2>(tile_coord);
        return make_coord(_, _, _, tile_M_index, tile_K_index);
    }

    CUTE_DEVICE
    auto
    get_tile_coord_Q() const
    {
        auto tile_coord   = get_tile_coord(m_tile_index_read);
        auto tile_P_index = get<1>(tile_coord);
        auto tile_K_index = get<2>(tile_coord);
        return make_coord(_, _, _, tile_P_index, tile_K_index);
    }

    CUTE_DEVICE
    auto
    get_tile_coord_Q2() const
    {
        return get_tile_coord_Q();
    }

    CUTE_DEVICE
    auto
    get_tile_coord_S() const
    {
        auto tile_coord   = get_tile_coord(m_tile_index_read);
        auto tile_N_index = get<1>(tile_coord);
        auto tile_K_index = get<2>(tile_coord);
        auto tile_G_index = tile_K_index / TileKsPerTileG{};
        return make_coord(_, _, _, tile_N_index, tile_G_index);
    }

    CUTE_DEVICE
    bool
    tile_read_is_in_bound() const
    {
        return (m_tile_index_read < m_tiles_this_block);
    }

    CUTE_DEVICE
    bool
    tile_is_in_bound() const
    {
        return (m_tile_index < m_tiles_this_block);
    }

    CUTE_DEVICE
    bool
    start_of_group() const
    {
        // the starting K tile index might not be aligned wth the G tile
        if (m_tile_index_read == 0)
        {
            return true;
        }
        auto tile_coord   = get_tile_coord(m_tile_index_read);
        auto tile_K_index = get<2>(tile_coord);
        return (tile_K_index % TileKsPerTileG{} == 0);
    }

    CUTE_DEVICE
    bool
    needs_fixup() const
    {
        // We needs fixup if either
        // 1. we are done with the last tile of the output tile
        // 2. we are done with the last tile of all of CTA's tiles
        return (finished_output_tile(m_tile_index) || is_last_tile(m_tile_index));
    }

    CUTE_DEVICE
    bool
    needs_epilogue() const
    {
        if constexpr (DecompositionMode == DecompositionModeEnum::StreamK && BACKWARDS == 1)
        {
            // the CTA that started the first tile of the output tile will do the epilogue
            return (needs_fixup() && started_output_tile(m_tile_index));
        }
        else
        {
            // the CTA that finished the last tile of the output tile will do the epilogue
            return finished_output_tile(m_tile_index);
        }
    }

    CUTE_DEVICE
    bool
    needs_to_clear_accum() const
    {
        // we need to clear the accumulator if both
        // 1. we are done with the output tile (perhaps partially), i.e., we did the fixup/epilogue
        // 2. we are not done with all of CTA's tiles
        return (needs_fixup() && (!is_last_tile(m_tile_index)));
    }

    template <typename Tensor>
    CUTE_DEVICE
    void
    maybe_fixup(
        Tensor& accum,
        int     thread_index,
        void*   workspace_partials,
        void*   workspace_barriers) const
    {

        int  block_index;
        int  block_index_first;
        auto block_started_output_tile  = started_output_tile(m_tile_index);
        auto block_finished_output_tile = finished_output_tile(m_tile_index);

        if constexpr (DecompositionMode == DecompositionModeEnum::SplitK)
        {
            if (m_slices == 1)
            {
                // Slice-K does not require fixup
                return;
            }

            // Split-K
            block_index       = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
            block_index_first = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z;

#if TS_DEBUG

            if(thread(TS_DEBUG_THR, TS_DEBUG_BLK))
            {
                print("\n--------------- Maybe Fixup (Split-K) ---------------\n");
                PPRINT_HEADER(); print("tile_index:                 %d\n", m_tile_index);
                PPRINT_HEADER(); print("tile_index_read:            %d\n", m_tile_index_read);
                PPRINT_HEADER(); print("block_index:                %d\n", block_index);
                PPRINT_HEADER(); print("block_index_first:          %d\n", block_index_first);
                PPRINT_HEADER(); print("block_started_output_tile:  %d\n", block_started_output_tile);
                PPRINT_HEADER(); print("block_finished_output_tile: %d\n", block_finished_output_tile);
            }
#endif

        }
        else
        {
            // Stream-K
            block_index               = get_block_index_streamk();
            auto tiles_shape          = make_shape(m_tiles_M, m_tiles_N, m_tiles_K);
            auto tiles_layout         = make_layout(tiles_shape, LayoutRight{});
            auto global_tile_index    = get_global_tile_index_streamk(m_tile_index);
            auto tile_coord           = tiles_layout.get_hier_coord(global_tile_index);
#if BACKWARDS
            // assuming all blocks are launched and run in lock-step, then the first block
            // to finish its portion of the output tile is the one that finishes the last
            // logical K-tile of the output tile. Similarly, the last block to finish its
            // portion of the output tile is the one that finishes the first logical K-tile
            auto tile_coord_first     = make_coord(get<0>(tile_coord), get<1>(tile_coord), m_tiles_K - 1);
#else
            // technically, even with BACKWARDS mode off, we still should define `tile_coord_first`
            // as above. However, when not all blocks are launched, this might cause deadlocks due
            // to circular dependencies. Hence, this is a workaround to avoid deadlocks.
            auto tile_coord_first     = make_coord(get<0>(tile_coord), get<1>(tile_coord), 0);
#endif
            auto tiles_to_first       = tiles_layout(tile_coord_first) + 1;
            auto blocks_typical_first = ceil_div(cute::min(tiles_to_first, m_tiles_typical_streamk)                          , m_tiles_per_block);
            auto blocks_special_first = ceil_div(cute::max(tiles_to_first, m_tiles_typical_streamk) - m_tiles_typical_streamk, m_tiles_per_block + 1);
            // almost `floor_div`, except when `tiles_to_first` is a multiple of `m_tiles_per_block`
            block_index_first         = blocks_typical_first + blocks_special_first - 1;

#if BACKWARDS
            // in the BACKWARDS case, the "finishing" block is the one that started the output tile,
            // who is likely the last block to finish its portion of the output tile. Similarly, the
            // non "finishing" blocks are the ones that did not start the output tile, and includes
            // the first block to finish its portion of the output tile.
            cutlass::swap(block_started_output_tile, block_finished_output_tile);
#endif


#if TS_DEBUG

            if(thread(TS_DEBUG_THR, TS_DEBUG_BLK))
            {
                print("\n--------------- Maybe Fixup (Stream-K) ---------------\n");
                PPRINT_HEADER(); print("tile_index:                 %d\n", m_tile_index);
                PPRINT_HEADER(); print("tile_index_read:            %d\n", m_tile_index_read);
                PPRINT_HEADER(); print("block_index:                %d\n", block_index);
                PPRINT_HEADER(); print("block_index_first:          %d\n", block_index_first);
                PPRINT_HEADER(); print("block_started_output_tile:  %d\n", block_started_output_tile);
                PPRINT_HEADER(); print("block_finished_output_tile: %d\n", block_finished_output_tile);
                PPRINT_HEADER(); print("tile_coord:                "); print(tile_coord); print("\n");
                PPRINT_HEADER(); print("tile_coord_first:          "); print(tile_coord_first); print("\n");
                PPRINT_HEADER(); print("tiles_to_first:             %d\n", tiles_to_first);
                PPRINT_HEADER(); print("blocks_typical_first:       %d\n", blocks_typical_first);
                PPRINT_HEADER(); print("blocks_special_first:       %d\n", blocks_special_first);
            }
#endif
        }


        if (!block_finished_output_tile)
        {
            // Non "finishing" SK blocks must share their partial accumulator sums through global scratch workspace
            FixupHelperT::initialize_or_accumulate(
                accum,
                thread_index,
                block_index,
                block_index_first,
                workspace_partials,
                workspace_barriers);
        }
        else
        {
            // DP blocks and "finishing" SK blocks must perform epilogue operations and write the output tile
            if (!block_started_output_tile)
            {
                // A "finishing" SK block must first aggregate its accumulator partial sums with those shared by peer threadblocks
                FixupHelperT::acquire(
                    accum,
                    thread_index,
                    block_index,
                    block_index_first,
                    workspace_partials,
                    workspace_barriers);
            }
        }
    }

    CUTE_DEVICE
    void
    prepare_epilogue(
        int & tile_M_index,
        int & tile_N_index,
        int & residue_M,
        int & residue_N) const
    {
        auto tile_coord = get_tile_coord(m_tile_index);
        tile_M_index    = get<0>(tile_coord);
        tile_N_index    = get<1>(tile_coord);
        residue_M       = get_residue_M(m_tile_index);
        residue_N       = get_residue_N(m_tile_index);

#if TS_DEBUG

        if(thread(TS_DEBUG_THR, TS_DEBUG_BLK))
        {
            print("\n--------------- Epilogue ---------------\n");
            PPRINT_HEADER(); print("tile_coord: "); print(tile_coord); print("\n");
            PPRINT_HEADER(); print("tile_index:      %d\n", m_tile_index);
            PPRINT_HEADER(); print("tile_index_read: %d\n", m_tile_index_read);
            PPRINT_HEADER(); print("residue_M:       %d\n", residue_M);
            PPRINT_HEADER(); print("residue_N:       %d\n", residue_N);
        }
#endif


    }

};

}  // namespace config