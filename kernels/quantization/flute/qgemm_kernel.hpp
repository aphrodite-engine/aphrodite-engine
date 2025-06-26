#pragma once

#include <cuda.h>
#include <stdio.h>
#include <cute/tensor.hpp>
#include "cutlass/workspace.h"

#include "config.hpp"
#include "packbits_utils.hpp"
#include "conversion_utils.hpp"
#include "tile_scheduler_utils.hpp"
#define DEBUG 0


// Essentially implements the following
// https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/sm80_mma_multistage.hpp
template <
  class Config,
  class TileScheduler,
  class T,
  class TQ,
  class T2
>
__global__ /* __launch_bounds__(128, 1) */
void
qgemm_device(const T * const __restrict__ A,
             const TQ* const __restrict__ Q,
                   T *       __restrict__ D,
             const T * const __restrict__ S,
             const T * const __restrict__ QM,
             const T2* const __restrict__ QM2,
                 void*       __restrict__ workspace,
             TileScheduler scheduler)
{
  using namespace cute;
  using X = Underscore;

  static constexpr config::QuantMapModeEnum      QuantMapMode      = Config::QuantMapMode;
  static constexpr config::AccumulationModeEnum  AccumulationMode  = Config::AccumulationMode;
  static constexpr config::DecompositionModeEnum DecompositionMode = Config::DecompositionMode;
  using SharedStorage = typename Config::SharedStorage;
  using TC = typename Config::TC;  // accumulator type
  using TR = typename Config::TR;  // reduction type
  CUTE_STATIC_ASSERT(is_same_v<T , typename Config::T > == true);
  CUTE_STATIC_ASSERT(is_same_v<TQ, typename Config::TQ> == true);
  CUTE_STATIC_ASSERT(is_same_v<T2, typename Config::T2> == true);

  using Warps              = typename Config::Warps;
  using Threads            = typename Config::Threads;
  using TileM              = typename Config::TileM;
  using TileN              = typename Config::TileN;
  using TileK              = typename Config::TileK;
  using TileP              = typename Config::TileP;
  using TileP2             = typename Config::TileP2;
  using TileG              = typename Config::TileG;
  using Stages             = typename Config::Stages;
  using StagesG            = typename Config::StagesG;
  using StagesGView        = typename Config::StagesGView;
  using NumBits            = typename Config::NumBits;
  using GroupSize          = typename Config::GroupSize;
  using QuantMapSize       = typename Config::QuantMapSize;
  using QuantMapSize2      = typename Config::QuantMapSize2;
  using QuantMapVecTraits  = typename Config::QuantMapVecTraits;
  using QuantMapDuplicates = typename Config::QuantMapDuplicates;
  using TileKsPerTileG     = typename Config::TileKsPerTileG;

  // shared memory layout
  using SmemLayoutA       = typename Config::SmemLayoutA;
  using SmemLayoutB       = typename Config::SmemLayoutB;
  using SmemLayoutC       = typename Config::SmemLayoutC;
  using SmemLayoutQ       = typename Config::SmemLayoutQ;
  using SmemLayoutQ2      = typename Config::SmemLayoutQ2;
  using SmemLayoutS       = typename Config::SmemLayoutS;
  using SmemLayoutSView   = typename Config::SmemLayoutSView;
  using SmemLayoutQM      = typename Config::SmemLayoutQM;
  using SmemLayoutQM2     = typename Config::SmemLayoutQM2;
  using SmemLayoutQM3     = typename Config::SmemLayoutQM3;
  using SmemLayoutQMView  = typename Config::SmemLayoutQMView;
  using SmemLayoutQM2View = typename Config::SmemLayoutQM2View;

  // global to shared memory copy
  using G2STiledCopyA   = typename Config::G2STiledCopyA;
  using G2STiledCopyQ   = typename Config::G2STiledCopyQ;
  using G2STiledCopyQ2  = typename Config::G2STiledCopyQ2;
  using G2STiledCopyS   = typename Config::G2STiledCopyS;
  using G2STiledCopyQM  = typename Config::G2STiledCopyQM;
  using G2STiledCopyQM2 = typename Config::G2STiledCopyQM2;

  // shared to shared memory copy
  using S2STiledCopyQM3 = typename Config::S2STiledCopyQM3;

  // shared to register copy
  using S2RCopyAtomA     = typename Config::S2RCopyAtomA;
  using S2RCopyAtomQ     = typename Config::S2RCopyAtomQ;
  using S2RCopyAtomQ2    = typename Config::S2RCopyAtomQ2;
  using S2RCopyAtomSView = typename Config::S2RCopyAtomSView;
  using S2RCopyAtomQM    = typename Config::S2RCopyAtomQM;

  // mma
  using TiledMma         = typename Config::TiledMma;
  using TiledMmaQ        = typename Config::TiledMmaQ;
  using TiledMmaQ2       = typename Config::TiledMmaQ2;
  using MmaPrmM          = typename Config::MmaPrmM;
  using MmaPrmN          = typename Config::MmaPrmN;
  using UnpackTargetSize = typename Config::UnpackTargetSize;

  // register to shared copy
  using R2SCopyAtomC = typename Config::R2SCopyAtomC;

  // shared to register copy
  using S2RTiledCopyC = typename Config::S2RTiledCopyC;

  // register to global copy
  using R2GCopyAtomC = typename Config::R2GCopyAtomC;

  // thread and lane index
  int thr_index  = threadIdx.x;
  int lane_index = threadIdx.x % Warps{};

#if DEBUG
  if(thread0()){
    print("\n\n");
    print("TileM  = "); print(TileM{}) ; print("\n");
    print("TileN  = "); print(TileN{}) ; print("\n");
    print("TileK  = "); print(TileK{}) ; print("\n");
    print("TileP  = "); print(TileP{}) ; print("\n");
    print("TileP2 = "); print(TileP2{}); print("\n");
    print("TileG  = "); print(TileG{}) ; print("\n");
    print("Stages = "); print(Stages{}); print("\n");
  }
#endif

  // use Tensor notation to represent device pointer + dimension
  Tensor mD   = make_tensor(make_gmem_ptr(D)  , make_shape(scheduler.M(), scheduler.N()), make_stride(scheduler.N(), _1{}));  // (M, N)
  Tensor mA   = make_tensor(make_gmem_ptr(A)  , make_shape(scheduler.M(), scheduler.K()), make_stride(scheduler.K(), _1{}));  // (M, K)
  Tensor mQ   = make_tensor(make_gmem_ptr(Q)  , make_shape(scheduler.P(), scheduler.K()), make_stride(scheduler.K(), _1{}));  // (P, K)
  Tensor mS   = make_tensor(make_gmem_ptr(S)  , make_shape(scheduler.N(), scheduler.G()), make_stride(scheduler.G(), _1{}));  // (N, G)
  Tensor mQM  = make_tensor(make_gmem_ptr(QM) , make_shape(QuantMapSize{}));
  Tensor mQM2 = make_tensor(make_gmem_ptr(QM2), make_shape(QuantMapSize2{}));
  // this implicitly assumes that Q is K-major.
  Tensor mQ2  = make_tensor(make_gmem_ptr(Q + scheduler.P() * scheduler.K()), make_shape(scheduler.P2(), scheduler.K()), make_stride(scheduler.K(), _1{}));  // (P2, K)

  // slice the tensor to small one which is used for current thread block.
  Tensor gD  = local_tile(mD,  make_tile(TileM{},  TileN{}), make_coord(_, _));  // (TileM , TileN, Num_Tiles_M , Num_Tiles_N)
  Tensor gA  = local_tile(mA,  make_tile(TileM{},  TileK{}), make_coord(_, _));  // (TileM , TileK, Num_Tiles_M , Num_Tiles_K)
  Tensor gQ  = local_tile(mQ,  make_tile(TileP{},  TileK{}), make_coord(_, _));  // (TileP , TileK, Num_Tiles_P , Num_Tiles_K)
  Tensor gQ2 = local_tile(mQ2, make_tile(TileP2{}, TileK{}), make_coord(_, _));  // (TileP2, TileK, Num_Tiles_P2, Num_Tiles_K)
  Tensor gS  = local_tile(mS,  make_tile(TileN{},  TileG{}), make_coord(_, _));  // (TileN , TileG, Num_Tiles_N , Num_Tiles_G)

  // shared memory
  extern __shared__ char smem_buf[];
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
  auto smem_C_data              = shared_storage.smem_C.data();
  if constexpr (DecompositionMode == config::DecompositionModeEnum::SplitK)
  {
    // In Stream-K mode, CTA might continue processing tiles after an epilogue, so we
    // cannot reuse the buffer for C. Instead, we allocate a separate buffer for C.
    // However, for Split-K mode, since CTA has processed all tiles after an epilogue,
    // we will reuse the same buffer for C.
    smem_C_data = reinterpret_cast<T*>(smem_buf);
  }

  auto sA    = make_tensor(make_smem_ptr(shared_storage.smem_A  .data()), SmemLayoutA{});   // (TileM,  TileK, Stages)
  auto sB    = make_tensor(make_smem_ptr(shared_storage.smem_A  .data()), SmemLayoutB{});   // (TileN,  TileK, Stages), using `smemA` as this is mostly as a placeholder to infer the shapes
  auto sQ    = make_tensor(make_smem_ptr(shared_storage.smem_Q  .data()), SmemLayoutQ{});   // (TileP,  TileK, Stages)
  auto sQ2   = make_tensor(make_smem_ptr(shared_storage.smem_Q2 .data()), SmemLayoutQ2{});  // (TileP2, TileK, Stages)
  auto sS    = make_tensor(make_smem_ptr(shared_storage.smem_S  .data()), SmemLayoutS{});   // (TileN,  TileG, StagesG)
  auto sQM   = make_tensor(make_smem_ptr(shared_storage.smem_QM .data()), SmemLayoutQM{});
  auto sQM2  = make_tensor(make_smem_ptr(shared_storage.smem_QM2.data()), SmemLayoutQM2{});
  auto sQM3  = make_tensor(make_smem_ptr(shared_storage.smem_QM3.data()), SmemLayoutQM3{});
  auto sC    = make_tensor(make_smem_ptr(smem_C_data                   ), SmemLayoutC{});
  // this is a view of `sS` broadcasted with `TileK`
  auto sSv   = make_tensor(make_smem_ptr(shared_storage.smem_S  .data()), SmemLayoutSView{});  // (TileN, TileK, (...), StagesG)
  // this is a view of `sQM` with an extra leading dimension of `1`
  auto sQMv  = make_tensor(make_smem_ptr(shared_storage.smem_QM .data()), SmemLayoutQMView{});
  // this is a view of `sQM2` with broadcasted to an extra dimension of `QuantMapDuplicates`
  auto sQM2v = make_tensor(make_smem_ptr(shared_storage.smem_QM2.data()), SmemLayoutQM2View{});

  CUTE_STATIC_ASSERT_V(size<0>(gA)  == size<0>(sA));   // TileM
  CUTE_STATIC_ASSERT_V(size<1>(gA)  == size<1>(sA));   // TileK
  CUTE_STATIC_ASSERT_V(size<0>(gQ)  == size<0>(sQ));   // TileP
  CUTE_STATIC_ASSERT_V(size<1>(gQ)  == size<1>(sQ));   // TileK
  CUTE_STATIC_ASSERT_V(size<0>(gQ2) == size<0>(sQ2));  // TileP2
  CUTE_STATIC_ASSERT_V(size<1>(gQ2) == size<1>(sQ2));  // TileK
  CUTE_STATIC_ASSERT_V(size<0>(gS)  == size<0>(sS));   // TileN
  CUTE_STATIC_ASSERT_V(size<1>(gS)  == size<1>(sS));   // TileG
  CUTE_STATIC_ASSERT_V(size<0>(gS)  == size<0>(sSv));  // TileN
  CUTE_STATIC_ASSERT_V(size<1>(sA)  == size<1>(sQ));   // TileK
  CUTE_STATIC_ASSERT_V(size<1>(sA)  == size<1>(sSv));  // TileK
  CUTE_STATIC_ASSERT_V(size<0>(sB)  == size<0>(sSv));  // TileN
  CUTE_STATIC_ASSERT_V(size<1>(sB)  == size<1>(sSv));  // TileK
  CUTE_STATIC_ASSERT_V(Stages{}     == size<2>(sA));   // Stages
  CUTE_STATIC_ASSERT_V(Stages{}     == size<2>(sQ));   // Stages
  CUTE_STATIC_ASSERT_V(Stages{}     == size<2>(sQ2));  // Stages
  CUTE_STATIC_ASSERT_V(StagesG{}    == size<2>(sS));   // StagesG

  //
  // MMA Atom partitioning
  //

  // dispatch TileA/TileB/TileC mma tensor into thread fragment via partition method
  TiledMma   tiled_mma;
  TiledMmaQ  tiled_mma_Q;   // primarily used to get the right shape for loading `Q`
  TiledMmaQ2 tiled_mma_Q2;  // primarily used to get the right shape for loading `Q2`
  auto thr_mma         = tiled_mma   .get_slice(thr_index);
  auto thr_mma_Q       = tiled_mma_Q .get_slice(thr_index);
  auto thr_mma_Q2      = tiled_mma_Q2.get_slice(thr_index);
  auto accum           = thr_mma     .partition_fragment_C(gD (_, _, _0{}, _0{}));  // (Mma, Mma_M,  Mma_N)
  auto tCrA            = thr_mma     .partition_fragment_A(sA (_, _, _0{}      ));  // (Mma, Mma_M,  Mma_K)
  auto tCrB            = thr_mma     .partition_fragment_B(sB (_, _, _0{}      ));  // (Mma, Mma_N,  Mma_K)
  auto tCrSv           = thr_mma     .partition_fragment_B(sSv(_, _, _0{}      ));  // (Mma, Mma_N,  Mma_K)
  auto tCrQ_tmp        = thr_mma_Q   .partition_fragment_B(sQ (_, _, _0{}      ));  // (Mma, Mma_P,  Mma_K)
  auto tCrQ2_tmp       = thr_mma_Q2  .partition_fragment_B(sQ2(_, _, _0{}      ));  // (Mma, Mma_P2, Mma_K)
  auto accum_epilogue  = make_fragment_like<T >(accum);                             // output types is the same as input types
  auto accum_reduction = make_fragment_like<TR>(accum);                             // reduction types could be different from compute types
  auto tCrQ            = make_fragment_like<TQ>(tCrQ_tmp);                          // (Mma, Mma_P,  Mma_K), tCrQ_tmp  is of type `T`
  auto tCrQ2           = make_fragment_like<TQ>(tCrQ2_tmp);                         // (Mma, Mma_P2, Mma_K), tCrQ2_tmp is of type `T`

  CUTE_STATIC_ASSERT_V(size<1>(tCrA)  == size<1>(accum));  // Mma_M
  CUTE_STATIC_ASSERT_V(size<1>(tCrB)  == size<2>(accum));  // Mma_N
  CUTE_STATIC_ASSERT_V(size<1>(tCrSv) == size<2>(accum));  // Mma_N
  CUTE_STATIC_ASSERT_V(size<2>(tCrA)  == size<2>(tCrB));   // Mma_K
  CUTE_STATIC_ASSERT_V(size<2>(tCrSv) == size<2>(tCrB));   // Mma_K
  CUTE_STATIC_ASSERT_V(size<2>(tCrA)  == size<2>(tCrQ));   // Mma_K
  CUTE_STATIC_ASSERT_V(size<2>(tCrA)  == size<2>(tCrQ2));  // Mma_K
  CUTE_STATIC_ASSERT_V(size<2>(tCrA)  == size<2>(tCrSv));  // Mma_K
  CUTE_STATIC_ASSERT_V(size<0>(accum) == size<0>(accum_epilogue));
  CUTE_STATIC_ASSERT_V(size<1>(accum) == size<1>(accum_epilogue));
  CUTE_STATIC_ASSERT_V(size<2>(accum) == size<2>(accum_epilogue));
  CUTE_STATIC_ASSERT_V(size<0>(accum) == size<0>(accum_reduction));
  CUTE_STATIC_ASSERT_V(size<1>(accum) == size<1>(accum_reduction));
  CUTE_STATIC_ASSERT_V(size<2>(accum) == size<2>(accum_reduction));
  CUTE_STATIC_ASSERT_V(size<1>(accum) == (MmaPrmM{}));
  CUTE_STATIC_ASSERT_V(size<2>(accum) == (MmaPrmN{} * UnpackTargetSize{}));

  //
  // Copy Atom
  //

  // global to shared memory copy, partition the copying of A and B tiles across the threads
  G2STiledCopyA  g2s_tiled_copy_A;
  G2STiledCopyQ  g2s_tiled_copy_Q;
  G2STiledCopyQ2 g2s_tiled_copy_Q2;
  G2STiledCopyS  g2s_tiled_copy_S;
  auto g2s_thr_copy_A  = g2s_tiled_copy_A .get_slice(thr_index);
  auto g2s_thr_copy_Q  = g2s_tiled_copy_Q .get_slice(thr_index);
  auto g2s_thr_copy_Q2 = g2s_tiled_copy_Q2.get_slice(thr_index);
  auto g2s_thr_copy_S  = g2s_tiled_copy_S .get_slice(thr_index);
  auto tAgA            = g2s_thr_copy_A .partition_S(gA);  // (G2S_CPY, G2S_CPY_M , G2S_CPY_K, Num_Tiles_M , Num_Tiles_K)
  auto tBgQ            = g2s_thr_copy_Q .partition_S(gQ);  // (G2S_CPY, G2S_CPY_P , G2S_CPY_K, Num_Tiles_P , Num_Tiles_K)
  auto tBgQ2           = g2s_thr_copy_Q2.partition_S(gQ2); // (G2S_CPY, G2S_CPY_P2, G2S_CPY_K, Num_Tiles_P2, Num_Tiles_K)
  auto tSgS            = g2s_thr_copy_S .partition_S(gS);  // (G2S_CPY, G2S_CPY_N , G2S_CPY_G, Num_Tiles_N , Num_Tiles_G)
  auto tAsA            = g2s_thr_copy_A .partition_D(sA);  // (G2S_CPY, G2S_CPY_M , G2S_CPY_K, Stages)
  auto tBsQ            = g2s_thr_copy_Q .partition_D(sQ);  // (G2S_CPY, G2S_CPY_P , G2S_CPY_K, Stages)
  auto tBsQ2           = g2s_thr_copy_Q2.partition_D(sQ2); // (G2S_CPY, G2S_CPY_P2, G2S_CPY_K, Stages)
  auto tSsS            = g2s_thr_copy_S .partition_D(sS);  // (G2S_CPY, G2S_CPY_N , G2S_CPY_G, Stages)

  CUTE_STATIC_ASSERT_V((size(tAsA) * Threads{}) == size(sA));  // sA is too small for Threads each with tAsA
  CUTE_STATIC_ASSERT_V((size(tBsQ) * Threads{}) == size(sQ));  // sQ is too small for Threads each with tBsQ
  CUTE_STATIC_ASSERT_V((size(tBsQ2)* Threads{}) == size(sQ2)); // sQ2 is too small for Threads each with tBsQ2
  CUTE_STATIC_ASSERT_V((size(tSsS) * Threads{}) == size(sS));  // sS is too small for Threads each with tSsS

  // shared to register copy
  auto s2r_tiled_copy_A  = make_tiled_copy_A(S2RCopyAtomA{}    , tiled_mma);
  auto s2r_tiled_copy_Q  = make_tiled_copy_B(S2RCopyAtomQ{}    , tiled_mma_Q);
  auto s2r_tiled_copy_Q2 = make_tiled_copy_B(S2RCopyAtomQ2{}   , tiled_mma_Q2);
  auto s2r_tiled_copy_Sv = make_tiled_copy_B(S2RCopyAtomSView{}, tiled_mma);
  auto s2r_thr_copy_A    = s2r_tiled_copy_A .get_slice(thr_index);
  auto s2r_thr_copy_Q    = s2r_tiled_copy_Q .get_slice(thr_index);
  auto s2r_thr_copy_Q2   = s2r_tiled_copy_Q2.get_slice(thr_index);
  auto s2r_thr_copy_Sv   = s2r_tiled_copy_Sv.get_slice(thr_index);
  auto tCsA              = s2r_thr_copy_A   .partition_S(sA);   // ? (S2R_CPY, S2R_CPY_M,  S2R_CPY_K, Stages)
  auto tCsQ              = s2r_thr_copy_Q   .partition_S(sQ);   // ? (S2R_CPY, S2R_CPY_P,  S2R_CPY_K, Stages)
  auto tCsQ2             = s2r_thr_copy_Q2  .partition_S(sQ2);  // ? (S2R_CPY, S2R_CPY_P2, S2R_CPY_K, Stages)
  auto tCsSv             = s2r_thr_copy_Sv  .partition_S(sSv);  // ? (S2R_CPY, S2R_CPY_N,  S2R_CPY_G, StagesG)
  auto tCrA_view         = s2r_thr_copy_A   .retile_D(tCrA);    // ? (S2R_CPY, S2R_CPY_M,  S2R_CPY_K)
  auto tCrQ_view         = s2r_thr_copy_Q   .retile_D(tCrQ);    // ? (S2R_CPY, S2R_CPY_P,  S2R_CPY_K)
  auto tCrQ2_view        = s2r_thr_copy_Q2  .retile_D(tCrQ2);   // ? (S2R_CPY, S2R_CPY_P2, S2R_CPY_K)
  auto tCrSv_view        = s2r_thr_copy_Sv  .retile_D(tCrSv);   // ? (S2R_CPY, S2R_CPY_N,  S2R_CPY_G)

  CUTE_STATIC_ASSERT_V(size<1>(tCsA)  == size<1>(tCrA_view));   // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tCsA)  == size<2>(tCrA_view));   // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tCsQ)  == size<1>(tCrQ_view));   // CPY_P
  CUTE_STATIC_ASSERT_V(size<2>(tCsQ)  == size<2>(tCrQ_view));   // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tCsQ2) == size<1>(tCrQ2_view));  // CPY_P2
  CUTE_STATIC_ASSERT_V(size<2>(tCsQ2) == size<2>(tCrQ2_view));  // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tCsSv) == size<1>(tCrSv_view));  // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tCsSv) == size<2>(tCrSv_view));  // CPY_G

  // quant map, note that every thread block loads the entire QM
  G2STiledCopyQM  g2s_tiled_copy_QM;
  G2STiledCopyQM2 g2s_tiled_copy_QM2;
  auto g2s_thr_copy_QM  = g2s_tiled_copy_QM .get_slice(thr_index);
  auto g2s_thr_copy_QM2 = g2s_tiled_copy_QM2.get_slice(thr_index);
  auto tQMgQM           = g2s_thr_copy_QM   .partition_S(mQM);    // ((1, 1), (1,))
  auto tQM2gQM2         = g2s_thr_copy_QM2  .partition_S(mQM2);   // ((1, ?), (?,))
  auto tQMsQM           = g2s_thr_copy_QM   .partition_D(sQM);    // ((1, 1), (1,))
  auto tQM2sQM2         = g2s_thr_copy_QM2  .partition_D(sQM2);   // ((1, ?), (?,))
  auto tQMrQM           = make_tensor<T>(Shape<_1>{});

  S2STiledCopyQM3 s2s_tiled_copy_QM3;
  auto s2s_thr_copy_QM3 = s2s_tiled_copy_QM3.get_slice(thr_index);
  auto tQM3sQM2         = s2s_thr_copy_QM3  .partition_S(sQM2v);  // ((1, ?), (?,))
  auto tQM3sQM3         = s2s_thr_copy_QM3  .partition_D(sQM3);   // ((1, ?), (?,))

  CUTE_STATIC_ASSERT_V(size(tQMgQM)  == _1{});  // QM is too large for Threads
  CUTE_STATIC_ASSERT_V(size(tQMsQM)  == _1{});  // QM is too large for Threads
  // CUTE_STATIC_ASSERT_V(size(tQM2gQM2) == _1{});  // QM is too large for Threads
  // CUTE_STATIC_ASSERT_V(size(tQM2sQM2) == _1{});  // QM is too large for Threads
  CUTE_STATIC_ASSERT_V(size(tQMrQM)  == _1{});  // QM is too large for Threads

  //
  // PREDICATES
  //

  // Allocate predicate tensors for m and n
  auto tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1, _0>{});
  auto tBpQ = make_tensor<bool>(make_shape(size<1>(tBsQ), size<2>(tBsQ)), Stride<_1, _0>{});
  auto tBpQ2= make_tensor<bool>(make_shape(size<1>(tBsQ2),size<2>(tBsQ2)),Stride<_1, _0>{});
  auto tSpS = make_tensor<bool>(make_shape(size<1>(tSsS), size<2>(tSsS)), Stride<_1, _0>{});

  // Construct identity layout for sA and sB
  auto cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));    // (TileM, TileK) -> (tile_m, tile_k)
  auto cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (TileP, TileK) -> (tile_p, tile_k)
  auto cQ2= make_identity_tensor(make_shape(size<0>(sQ2),size<1>(sQ2)));   // (TileP2,TileK) -> (tile_p2,tile_k)
  auto cS = make_identity_tensor(make_shape(size<0>(sS), size<1>(sS)));    // (TileN, TileG) -> (tile_n, tile_g)

  // Repeat the partitioning with identity layouts
  auto tAcA  = g2s_thr_copy_A .partition_S(cA);                            // (ACPY,ACPY_M,ACPY_K) -> (tile_m,tile_k)
  auto tBcQ  = g2s_thr_copy_Q .partition_S(cQ);                            // (BCPY,BCPY_N,BCPY_K) -> (tile_n,tile_k)
  auto tBcQ2 = g2s_thr_copy_Q2.partition_S(cQ2);                           // (BCPY,BCPY_N,BCPY_K) -> (tile_n,tile_k)
  auto tScS  = g2s_thr_copy_S .partition_S(cS);                            // (SCPY,SCPY_N,SCPY_G) -> (tile_n,tile_g)

  CUTE_STATIC_ASSERT_V(TileM {} == size<0>(gA ));
  CUTE_STATIC_ASSERT_V(TileN {} == size<1>(gD ));
  CUTE_STATIC_ASSERT_V(TileP {} == size<0>(gQ ));
  CUTE_STATIC_ASSERT_V(TileP2{} == size<0>(gQ2));

  //
  // Epilogue
  // https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/epilogue/collective/sm70_epilogue_vectorized.hpp
  //

  // Partition sC to match the accumulator partitioning
  // 1. note that we will write `accum_epilogue` instead of `accum`
  // 2. note that `tiled_mma` has accumulator of type `TC`, but we want to copy in type `T`,
  //    but it seems like `make_tiled_copy_C` uses just the layout, not the type info
  auto r2s_tiled_copy_C = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto r2s_thr_copy_C   = r2s_tiled_copy_C.get_slice(thr_index);
  auto tCaC             = r2s_thr_copy_C.retile_S(accum_epilogue);  // (R2S_CPY=(Atom, AtomNum),  Mma_M,  Mma_N)
  auto tCsC             = r2s_thr_copy_C.partition_D(sC);           // (R2S_CPY=(Atom, AtomNum), PIPE_M, PIPE_N)

  // Tile gD and gC by the shape of SmemLayout first
  auto gD_tile  = make_shape(size<0>(sC), size<1>(sC));
  auto gDt = flat_divide(gD, gD_tile);  // (SMEM_M, SMEM_N, TileM, TileN, Num_Tiles_M, Num_Tiles_N)

  // Partition sC, and gD for the output
  S2RTiledCopyC s2r_tiled_copy_C;
  auto s2r_thr_copy_C = s2r_tiled_copy_C.get_slice(thr_index);
  auto tDsC           = s2r_thr_copy_C.partition_S(sC);   // (S2R_COPY=(Atom, AtomNum), ATOM_M, ATOM_N)
  auto tDgD           = s2r_thr_copy_C.partition_D(gDt);  // (S2R_COPY=(Atom, AtomNum), ATOM_M, ATOM_N, TileM, TileN, Num_Tiles_M, Num_Tiles_N)

  // Allocate intermediate registers on the dst tensors
  // note that `tDrC` is of type `T` instead of `TC` because we will write `accum_epilogue` instead of `accum`
  auto tDrC = make_tensor<T>(take<0, 3>(shape(tDgD)));  // ((Atom, AtomNum), ATOM_M, ATOM_N)
  // auto tDrD = make_tensor<T>(shape(tDrC));              // ((Atom, AtomNum), ATOM_M, ATOM_N)

  // Repeat the D-partitioning for coordinates and predication
  auto cD   = make_identity_tensor(make_shape(size<0>(gD), size<1>(gD)));  // (TileM, TileN) -> (tile_m, tile_n)
  auto cDt  = flat_divide(cD, gD_tile);                                    // (TileM, TileN, Num_Tiles_M=1, Num_Tiles_N=1)
  auto tDcD = s2r_thr_copy_C.partition_D(cDt);                             // (S2R_COPY=(Atom, AtomNum), ATOM_M, ATOM_N, Num_Tiles_M=1, Num_Tiles_N=1)

  CUTE_STATIC_ASSERT(size<1>(tCaC) % size<3>(tDgD) == 0);  // TileM divides Mma_M
  CUTE_STATIC_ASSERT(size<2>(tCaC) % size<4>(tDgD) == 0);  // TileN divides Mma_N
  // CUTE_STATIC_ASSERT(typename S2RTiledCopyC::TiledNumThr{} == size<0>(typename TiledMma::AtomLayoutC_TV{}));


#if DEBUG

#define PPRINT_LAYOUT(name, tensor) \
  do { \
    print("\n\n"); \
    print(name); \
    print("\n"); \
    print(tensor.layout()); \
  } while(0)

#define PPRINT_DATA(name, data) \
  do { \
    print("\n\n"); \
    print(name); \
    print("\n"); \
    print(data); \
  } while(0)

  if(thread0())
  {
      PPRINT_LAYOUT("mD" , mD);
      PPRINT_LAYOUT("mA" , mA);
      PPRINT_LAYOUT("mQ" , mQ);
      PPRINT_LAYOUT("mQ2", mQ2);
      PPRINT_LAYOUT("mS" , mS);

      PPRINT_LAYOUT("gD" , gD);
      PPRINT_LAYOUT("gA" , gA);
      PPRINT_LAYOUT("gQ" , gQ);
      PPRINT_LAYOUT("gQ2", gQ2);
      PPRINT_LAYOUT("gS" , gS);

      PPRINT_LAYOUT("sA" , sA);
      PPRINT_LAYOUT("sB" , sB);
      PPRINT_LAYOUT("sQ" , sQ);
      PPRINT_LAYOUT("sQ2", sQ2);
      PPRINT_LAYOUT("sS" , sS);
      PPRINT_LAYOUT("sSv", sSv);
      PPRINT_LAYOUT("sC" , sC);
      print("\n\n");

      print("--------------------- Global to Shared ---------------------");
      PPRINT_LAYOUT("tAgA" , tAgA);
      PPRINT_LAYOUT("tBgQ" , tBgQ);
      PPRINT_LAYOUT("tBgQ2", tBgQ2);
      PPRINT_LAYOUT("tSgS" , tSgS);
      PPRINT_LAYOUT("tAsA" , tAsA);
      PPRINT_LAYOUT("tBsQ" , tBsQ);
      PPRINT_LAYOUT("tBsQ2", tBsQ2);
      PPRINT_LAYOUT("tSsS" , tSsS);
      print("\n\n");

      print("--------------------- Shared to Registers ---------------------");
      PPRINT_LAYOUT("tCsA" , tCsA);
      PPRINT_LAYOUT("tCsQ" , tCsQ);
      PPRINT_LAYOUT("tCsQ2", tCsQ2);
      PPRINT_LAYOUT("tCsSv", tCsSv);
      PPRINT_LAYOUT("tCrA_view" , tCrA_view);
      PPRINT_LAYOUT("tCrQ_view" , tCrQ_view);
      PPRINT_LAYOUT("tCrQ2_view", tCrQ2_view);
      PPRINT_LAYOUT("tCrSv_view", tCrSv_view);
      print("\n\n");

      print("--------------------- QM ---------------------");
      PPRINT_LAYOUT("mQM"  , mQM);
      PPRINT_LAYOUT("mQM2" , mQM2);
      PPRINT_LAYOUT("sQM"  , sQM);
      PPRINT_LAYOUT("sQM2" , sQM2);
      PPRINT_LAYOUT("sQM3" , sQM3);
      PPRINT_LAYOUT("sQMv" , sQMv);
      PPRINT_LAYOUT("sQM2v", sQM2v);
      PPRINT_LAYOUT("tQMgQM"  , tQMgQM);
      PPRINT_LAYOUT("tQM2gQM2", tQM2gQM2);
      PPRINT_LAYOUT("tQM3sQM2", tQM3sQM2);
      PPRINT_LAYOUT("tQMsQM"  , tQMsQM);
      PPRINT_LAYOUT("tQM2sQM2", tQM2sQM2);
      PPRINT_LAYOUT("tQM3sQM3", tQM3sQM3);
      PPRINT_LAYOUT("tQMrQM"  , tQMrQM);
      print("\n\n");

      print("--------------------- TiledMma ---------------------");
      print("\n\ntile_shape(tiled_mma)\n");
      print(tile_shape(tiled_mma));
      print("\n\ntile_shape(tiled_mma_Q)\n");
      print(tile_shape(tiled_mma_Q));
      print("\n\ntile_shape(tiled_mma_Q2)\n");
      print(tile_shape(tiled_mma_Q2));
      PPRINT_LAYOUT("accum"          , accum);
      PPRINT_LAYOUT("accum_epilogue" , accum_epilogue);
      PPRINT_LAYOUT("tCrA"           , tCrA);
      PPRINT_LAYOUT("tCrB"           , tCrB);
      PPRINT_LAYOUT("tCrQ"           , tCrQ);
      PPRINT_LAYOUT("tCrQ2"          , tCrQ2);
      PPRINT_LAYOUT("tCrSv"          , tCrSv);
      print("\n\n");

      print("--------------------- Predicates ---------------------");
      PPRINT_LAYOUT("tApA", tApA);
      PPRINT_LAYOUT("tBpQ", tBpQ);
      PPRINT_LAYOUT("tBpQ2",tBpQ2);
      PPRINT_LAYOUT("tSpS", tSpS);
      PPRINT_LAYOUT("cA"  , cA);
      PPRINT_LAYOUT("cQ"  , cQ);
      PPRINT_LAYOUT("cQ2" , cQ2);
      PPRINT_LAYOUT("cS"  , cS);
      PPRINT_LAYOUT("tAcA", tAcA);
      PPRINT_LAYOUT("tBcQ", tBcQ);
      PPRINT_LAYOUT("tBcQ2",tBcQ2);
      PPRINT_LAYOUT("tScS", tScS);
      print("\n\n");

      print("--------------------- Registers to Shared ---------------------");
      PPRINT_LAYOUT("tCaC", tCaC);
      PPRINT_LAYOUT("tCsC", tCsC);
      print("\n\n");

      print("--------------------- Shared to Registers ---------------------");
      PPRINT_LAYOUT("tDrC", tDrC);
      // PPRINT_LAYOUT("tDrD", tDrD);
      print("\n\n");

      print("--------------------- Registers to Global ---------------------");
      PPRINT_LAYOUT("gDt" , gDt);
      PPRINT_LAYOUT("tDsC", tDsC);
      PPRINT_LAYOUT("tDgD", tDgD);
      print("\n\n");

      print("--------------------- Epilogue Predicates ---------------------");
      PPRINT_LAYOUT("cD"  , cD);
      PPRINT_LAYOUT("cDt" , cDt);
      PPRINT_LAYOUT("tDcD", tDcD);
      print("\n\n");
  }
#endif

  //
  // PIPELINED MAIN LOOP
  //

  // initialize the tile scheduler
  scheduler.initialize(tApA, tBpQ, tBpQ2, tSpS, tAcA, tBcQ, tBcQ2, tScS);

  int smem_pipe_read     = 0;
  int smem_pipe_read_G   = scheduler.smem_pipe_read_G_offset() % StagesGView{};  // the starting K tile index might not be aligned wth the G tile
  int smem_pipe_read_raw = 0;
  int smem_pipe_write    = 0;
  int smem_pipe_write_G  = 0;

  // size of the register pipeline
  auto num_mma_K = size<2>(tCrA);

  // partition the workspace
  void* workspace_barriers = workspace;
  void* workspace_partials = static_cast<char*>(workspace) + scheduler.workspace_size_barriers();

  // Clear the smem tiles to account for predicated off loads
  clear(tAsA);
  clear(tBsQ);
  // fill zero for accumulator
  clear(accum);

  //
  // ------- Prefetching -------
  //

  // start async gmem -> shm loads for all pipes but the last

  // the copy size is too small for `cp.async` so this is sync
  if(thr_index < QuantMapSize{}) {
    cute::copy(g2s_tiled_copy_QM, tQMgQM, tQMsQM);
  }

  // prefetch quant map, relying on the next `cp_async_fence`.
  if constexpr (is_same_v<typename QuantMapVecTraits::IsVectorized, true_type>)
  {
    if(thr_index < QuantMapSize2{})
    {
      cute::copy(g2s_tiled_copy_QM2, tQM2gQM2, tQM2sQM2);
    }
  }

  CUTLASS_PRAGMA_UNROLL
  for (int stage_index = 0; stage_index < Stages{} - 1; ++stage_index)
  {
    auto tile_coord_A = scheduler.get_tile_coord_A();
    auto tile_coord_Q = scheduler.get_tile_coord_Q();
    cute::copy_if(g2s_tiled_copy_A, tApA, tAgA(tile_coord_A), tAsA(_, _, _, stage_index));
    cute::copy_if(g2s_tiled_copy_Q, tBpQ, tBgQ(tile_coord_Q), tBsQ(_, _, _, stage_index));

    if (scheduler.start_of_group())
    {
      auto tile_coord_S = scheduler.get_tile_coord_S();
      cute::copy_if(g2s_tiled_copy_S, tSpS, tSgS(tile_coord_S), tSsS(_, _, _, smem_pipe_write_G));
      smem_pipe_write_G = (smem_pipe_write_G + 1) % StagesG{};
    }

    if constexpr (is_same_v<NumBits, _3>)
    {
      auto tile_coord_Q2 = scheduler.get_tile_coord_Q2();
      cute::copy_if(g2s_tiled_copy_Q2, tBpQ2, tBgQ2(tile_coord_Q2), tBsQ2(_, _, _, stage_index));
    }

    cp_async_fence();

    // increment index
    ++smem_pipe_write;
    scheduler.step_read(tApA, tBpQ, tBpQ2, tSpS, tAcA, tBcQ, tBcQ2, tScS);

  }
  // smem_pipe_read  == Stages % Stages (== 0)
  // smem_pipe_write == Stages - 1

  // prefetch register pipeline, wait one submitted gmem->smem done
  cp_async_wait<Stages{} - 2>();
  __syncthreads();

  // prefer the first rmem from the first k-tile
  // smem -> reg
  cute::copy(S2RCopyAtomQM{}  , sQMv (_, lane_index % QuantMapSize{}), tQMrQM);
  cute::copy(s2r_tiled_copy_A , tCsA (_, _, _0{}, smem_pipe_read    ), tCrA_view (_, _, _0{}));
  cute::copy(s2r_tiled_copy_Q , tCsQ (_, _, _0{}, smem_pipe_read    ), tCrQ_view (_, _, _0{}));
  cute::copy(s2r_tiled_copy_Sv, tCsSv(_, _, _0{}, smem_pipe_read_G  ), tCrSv_view(_, _, _0{}));

  if constexpr (is_same_v<NumBits, _3>)
  {
    cute::copy(s2r_tiled_copy_Q2, tCsQ2(_, _, _0{}, smem_pipe_read), tCrQ2_view(_, _, _0{}));
  }

  // duplicate the quant map for each lane
  if constexpr (is_same_v<typename QuantMapVecTraits::IsDuplicated, true_type>)
  {
    cute::copy(s2s_tiled_copy_QM3, tQM3sQM2, tQM3sQM3);
  }

  //
  // ------- Main Loop -------
  //

  // loop over K: i. load tile, ii. mma
  CUTLASS_PRAGMA_NO_UNROLL
  for ( ; scheduler.tile_is_in_bound(); scheduler.step())
  {

    // Note, the for_each() function is required here to ensure `mma_index` is of type Int<N>.
    for_each(make_int_sequence<num_mma_K>{}, [&] (auto mma_index)
    {

      if (mma_index == num_mma_K - 1) {
        // increment tile index
        ++smem_pipe_read_raw;
        smem_pipe_read   = (smem_pipe_read_raw) % Stages{};
        smem_pipe_read_G = (smem_pipe_read_raw + scheduler.smem_pipe_read_G_offset()) % StagesGView{};

        // wait one submitted gmem->smem done
        cp_async_wait<Stages{} - 2>();
        __syncthreads();
      }

      // shm -> reg s[tile_index][mma_index + 1] -> r[mma_index + 1]
      auto mma_index_next = (mma_index + _1{}) % num_mma_K;
      cute::copy(s2r_tiled_copy_A , tCsA (_, _, mma_index_next, smem_pipe_read  ), tCrA_view (_, _, mma_index_next));
      cute::copy(s2r_tiled_copy_Q , tCsQ (_, _, mma_index_next, smem_pipe_read  ), tCrQ_view (_, _, mma_index_next));
      cute::copy(s2r_tiled_copy_Sv, tCsSv(_, _, mma_index_next, smem_pipe_read_G), tCrSv_view(_, _, mma_index_next));

      if constexpr (is_same_v<NumBits, _3>) {
        cute::copy(s2r_tiled_copy_Q2, tCsQ2(_, _, mma_index_next, smem_pipe_read), tCrQ2_view(_, _, mma_index_next));
      }

      // copy gmem to smem before computing gemm on each k-pipe
      if (mma_index == 0)
      {
        if (scheduler.tile_read_is_in_bound())
        {
          auto tile_coord_A = scheduler.get_tile_coord_A();
          auto tile_coord_Q = scheduler.get_tile_coord_Q();
          cute::copy_if(g2s_tiled_copy_A, tApA, tAgA(tile_coord_A), tAsA(_, _, _, smem_pipe_write));
          cute::copy_if(g2s_tiled_copy_Q, tBpQ, tBgQ(tile_coord_Q), tBsQ(_, _, _, smem_pipe_write));

          if (scheduler.start_of_group())
          {
            auto tile_coord_S = scheduler.get_tile_coord_S();
            cute::copy_if(g2s_tiled_copy_S, tSpS, tSgS(tile_coord_S), tSsS(_, _, _, smem_pipe_write_G));
            smem_pipe_write_G = (smem_pipe_write_G + 1) % StagesG{};
          }

          if constexpr (is_same_v<NumBits, _3>)
          {
            auto tile_coord_Q2 = scheduler.get_tile_coord_Q2();
            cute::copy_if(g2s_tiled_copy_Q2, tBpQ2, tBgQ2(tile_coord_Q2), tBsQ2(_, _, _, smem_pipe_write));
          }

          smem_pipe_write = (smem_pipe_write + 1) % Stages{};
          scheduler.step_read(tApA, tBpQ, tBpQ2, tSpS, tAcA, tBcQ, tBcQ2, tScS);

        }

        cp_async_fence();
      }

      // dequantize
      if constexpr (is_same_v<typename QuantMapVecTraits::IsDuplicated, false_type>)
      {
        packbits_utils::dequantize<QuantMapMode>(
          tCrQ (_, _, mma_index),
          tCrQ2(_, _, mma_index),
          tCrB (_, _, mma_index),
          tCrSv(_, _, mma_index),
          sQM,
          sQM2,
          tQMrQM,
          NumBits{});
      }
      else
      {
        packbits_utils::dequantize<QuantMapMode>(
          tCrQ (_, _, mma_index),
          tCrQ2(_, _, mma_index),
          tCrB (_, _, mma_index),
          tCrSv(_, _, mma_index),
          sQM,
          sQM3 (_, lane_index % QuantMapDuplicates{}),
          tQMrQM,
          NumBits{});
      }


      // mma
      cute::gemm(
        tiled_mma,
        accum,
        tCrA(_, _, mma_index),
        tCrB(_, _, mma_index),
        accum);

    });  // for mma_index


    //
    // ------- Epilogue -------
    //

    if (scheduler.needs_fixup())
    {
      if constexpr (AccumulationMode != config::AccumulationModeEnum::Mixed)
      {
        scheduler.maybe_fixup(accum, thr_index, workspace_partials, workspace_barriers);
      }
      else
      {
        conversion_utils::convert_tensor(accum, accum_reduction);
        scheduler.maybe_fixup(accum_reduction, thr_index, workspace_partials, workspace_barriers);
      }

    }

    if (scheduler.needs_epilogue())
    {
      int epilogue_tile_M_index;
      int epilogue_tile_N_index;
      int epilogue_residue_M;
      int epilogue_residue_N;
      scheduler.prepare_epilogue(epilogue_tile_M_index, epilogue_tile_N_index, epilogue_residue_M, epilogue_residue_N);

      // output the possibly lower-precision type
      // note that `tCaC` is a view of `accum_epilogue`
      if constexpr (AccumulationMode != config::AccumulationModeEnum::Mixed)
      {
        conversion_utils::convert_tensor(accum, accum_epilogue);
      }
      else
      {
        // this implicitly assumes that `needs_epilogue` always follows `needs_fixup`
        // otherwise, we need to convert `accum` to `accum_reduction` here
        conversion_utils::convert_tensor(accum_reduction, accum_epilogue);
      }

      // For each tiling needed for SmemLayout to cover shape(gD)
      CUTLASS_PRAGMA_UNROLL
      for (int step_m = 0; step_m < size<2>(cDt); ++step_m)  // Num_Tiles_M
      {
        CUTLASS_PRAGMA_UNROLL
        for (int step_n = 0; step_n < size<3>(cDt); ++step_n)  // Num_Tiles_N
        {
          // Step 1. Copy to SMEM
          CUTLASS_PRAGMA_UNROLL
          for (int pipe_m = 0; pipe_m < size<1>(tCsC); ++pipe_m)  // PIPE_M
          {
            CUTLASS_PRAGMA_UNROLL
            for (int pipe_n = 0; pipe_n < size<2>(tCsC); ++pipe_n)  // PIPE_N
            {
              int mma_m = step_m * size<1>(tCsC) + pipe_m;
              int mma_n = step_n * size<2>(tCsC) + pipe_n;
              copy(r2s_tiled_copy_C, tCaC(_, mma_m, mma_n), tCsC(_, pipe_m, pipe_n));
            }
          }

          // Step 2. Wait for SMEM writes to complete
          __syncthreads();

          // Step 3. Copy from SMEM into a fragment
          copy(s2r_tiled_copy_C, tDsC, tDrC);

          // Step 4. Wait for SMEM reads to complete
          __syncthreads();

          auto tDgDmn = tDgD(_, _, _, step_m, step_n, epilogue_tile_M_index, epilogue_tile_N_index);
          auto tDcDmn = tDcD(_, _, _, step_m, step_n);

          // Step 5. Elementwise operation with conversion
          // CUTLASS_PRAGMA_UNROLL
          // for (int i = 0; i < size(tDrC); ++i)
          // {
          //   tDrD(i) = epilogue_op(tDrC(i));
          // }

          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < size<1>(tDgDmn); ++m)
          {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < size<2>(tDgDmn); ++n)
            {
              // Predication
              if (get<0>(tDcDmn(0, m, n)) < epilogue_residue_M &&
                  get<1>(tDcDmn(0, m, n)) < epilogue_residue_N)
              {
                // Step 6. Copy to GMEM
                // copy(R2GCopyAtomC{}, tDrD(_, m, n), tDgDmn(_, m, n));
                copy(R2GCopyAtomC{}, tDrC(_, m, n), tDgDmn(_, m, n));
              }
            }
          }
        }  // for step_n
      }  // for step_m
    }

    if (scheduler.needs_to_clear_accum())
    {
      // Clear the accumulator for the next output tile
      clear(accum);
    }


  }  // tile_index
}


template <
  typename T,
  typename TQ,
  typename T2,
  typename Threads,
  typename TileM,
  typename TileK,
  typename TileP,
  typename Stages,
  typename NumBits,
  typename GroupSize,
  config::QuantMapModeEnum QuantMapMode,
  config::AccumulationModeEnum AccumulationMode,
  config::DecompositionModeEnum DecompositionMode,
  typename G2STiledCopySizeS,
  typename MmaPrmK
>
void
qgemm_host(int M,
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
           const int                    blocks,
           const cudaStream_t           stream)
{
    using namespace cute;

    CUTE_STATIC_ASSERT_V(Threads{} % _128{} == _0{});
    CUTE_STATIC_ASSERT_V(NumBits{} == _4{} || NumBits{} == _3{} || NumBits{} == _2{});

    using Config = config::GemmConfig<T,
                                      TQ,
                                      Threads,
                                      TileM,
                                      TileK,
                                      TileP,
                                      Stages,
                                      NumBits,
                                      GroupSize,
                                      QuantMapMode,
                                      AccumulationMode,
                                      DecompositionMode,
                                      G2STiledCopySizeS,
                                      MmaPrmK>;
    using TileScheduler = config::TileScheduler<Config>;
    auto qgemm_device_func = qgemm_device<Config, TileScheduler, T, TQ, T2>;

    // assume `slices = 0` since it is deprecated
    TileScheduler scheduler(M, N, K, P, 0, blocks);
    dim3 grid     = scheduler.grid();
    dim3 block    = scheduler.block();
    int smem_size = scheduler.smem_size();

#if DEBUG

#define CUDA_CHECK(call)                                    \
  do {                                                      \
    cudaError_t status = call;                              \
    if(status != cudaSuccess) {                             \
      printf("FAIL: call='%s'. Reason:%s\n", #call,         \
             cudaGetErrorString(status));                   \
    }                                                       \
  } while (0)

    int devId;
    int numProcs;
    CUDA_CHECK(cudaGetDevice(&devId));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &numProcs,
        cudaDevAttrMultiProcessorCount,
        devId));

    print("TiledMma\n");
    print(typename Config::TiledMma{});
    print("\n");
    print("TiledMmaQ\n");
    print(typename Config::TiledMmaQ{});
    print("\n");
    print("G2STiledCopyA\n");
    print(typename Config::G2STiledCopyA{});
    print("\n");
    print("G2STiledCopyQ\n");
    print(typename Config::G2STiledCopyQ{});
    print("\n");
    print("G2STiledCopyS\n");
    print(typename Config::G2STiledCopyS{});
    print("\n");
    print("G2STiledCopyQM\n");
    print(typename Config::G2STiledCopyQM{});
    print("\n");
    print("S2RTiledCopyC\n");
    print(typename Config::S2RTiledCopyC{});
    print("\n");
    print("Grid  = \t (%d, %d, %d)\n",  grid.x,  grid.y,  grid.z);
    print("Block = \t (%d, %d, %d)\n", block.x, block.y, block.z);
    print("numProcs = \t %d\n", numProcs);
#endif

    cudaFuncSetAttribute(
        qgemm_device_func,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);

    qgemm_device_func
        <<<grid, block, smem_size, stream>>>
        (A, Q, D, S, QM, QM2,
         workspace,
         scheduler);
}