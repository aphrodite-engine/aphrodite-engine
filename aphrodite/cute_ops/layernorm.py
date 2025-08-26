# Adapted from
# https://github.com/Dao-AILab/quack/blob/48c702e0defa69c24529dc3ebbbf4a586e2853b1/quack/layernorm.py


from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

import aphrodite.cute_ops.utils as utils
from aphrodite.cute_ops.reduction import ReductionBase, torch2cute_dtype_map


class LayerNorm(ReductionBase):
    def __init__(self, dtype: cutlass.Numeric, N: int):
        super().__init__(dtype, N, stage=2)  # 2 stages for mean and var
        self.reload_from = None if N <= 16384 else "smem"
        self.delay_w_load = False

    def _calculate_threads_per_row(self):
        N = self.N
        return (
            8
            if N <= 64
            else (
                16
                if N <= 128
                else (
                    32
                    if N <= 3072
                    else (
                        64
                        if N <= 6144
                        else (128 if N <= 16384 else 256)
                    )
                )
            )
        )

    def _set_cluster_n(self):
        N = self.N
        # cluster_n = 4 is faster and cluster_n = 2 for N=64k for some reason
        # Similarly cluster_n = 8 is faster for N=128k
        if cutlass.const_expr(self.dtype.width == 16):
            cluster_n = (
                1
                if N <= 16 * 1024
                else (
                    2
                    if N <= 32 * 1024
                    else (
                        4
                        if N <= 64 * 1024
                        else (8 if N <= 128 * 1024 else 16)
                    )
                )
            )
        else:  # fp32
            cluster_n = (
                1
                if N <= 32 * 1024
                else (
                    2
                    if N <= 64 * 1024
                    else (
                        4
                        if N <= 128 * 1024
                        else (8 if N <= 256 * 1024 else 16)
                    )
                )
            )
        self.cluster_n = cluster_n

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mO: cute.Tensor,
        mRstd: Optional[cute.Tensor],
        mMean: Optional[cute.Tensor],
        stream: cuda.CUstream,
        eps: cutlass.Float32 = 1e-6,
    ):
        assert mX.element_type == self.dtype
        assert mO.element_type == self.dtype
        self._set_cluster_n()
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE
        mW_expanded_layout = cute.prepend(
            mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)),
        )
        mW = cute.make_tensor(mW.iterator, mW_expanded_layout)
        if cutlass.const_expr(mRstd is not None):
            mRstd_expanded_layout = cute.append(
                mRstd.layout, cute.make_layout((self.N,), stride=(0,))
            )
            mRstd = cute.make_tensor(mRstd.iterator, mRstd_expanded_layout)
        if cutlass.const_expr(mMean is not None):
            mMean_expanded_layout = cute.append(
                mMean.layout, cute.make_layout((self.N,), stride=(0,))
            )
            mMean = cute.make_tensor(mMean.iterator, mMean_expanded_layout)
        self.kernel(
            mX,
            mW,
            mO,
            mRstd,
            mMean,
            eps,
            tv_layout,
            tiler_mn,
            self.reload_from,
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1]
            if cutlass.const_expr(self.cluster_n > 1)
            else None,
            smem=self._smem_size_in_bytes(tiler_mn, num_warps),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mO: cute.Tensor,
        mRstd: Optional[cute.Tensor],
        mMean: Optional[cute.Tensor],
        eps: cute.Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
        reload_from: cutlass.Constexpr = None,
        delay_w_load: cutlass.Constexpr = False,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout,
        )

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        # slice for CTAs
        # We use domain_offset_i64 to deal with tensors larger than 2^31
        # elements
        mX, mO = [utils.domain_offset_i64((bidx * tiler_mn[0], 0), mT)
                  for mT in (mX, mO)]
        gX, gO = [cute.local_tile(mT, tiler_mn, (0, cluster_y))
                  for mT in (mX, mO)]
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))
        gW = cute.local_tile(mW, tiler_mn, (0, cluster_y))
        gRstd = (
            cute.local_tile(mRstd, tiler_mn, (bidx, cluster_y))
            if cutlass.const_expr(mRstd is not None)
            else None
        )
        gMean = (
            cute.local_tile(mMean, tiler_mn, (bidx, cluster_y))
            if cutlass.const_expr(mMean is not None)
            else None
        )

        # declare the atoms which will be used later for memory copy
        copy_atom_load_X = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=128,
        )
        copy_atom_load_X_async = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mX.element_type,
            num_bits_per_copy=128,
        )
        copy_atom_load_W = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mW.element_type,
            num_bits_per_copy=128,
        )
        copy_atom_store_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mO.element_type,
            num_bits_per_copy=128,
        )

        # TODO(alpin): see if make_tiled_copy_tv can be used here instead
        thr_copy_X = (
            cute.make_tiled_copy(copy_atom_load_X_async, tv_layout, tiler_mn)
            .get_slice(tidx)
        )
        thr_copy_W = (
            cute.make_tiled_copy(copy_atom_load_W, tv_layout, tiler_mn)
            .get_slice(tidx)
        )
        thr_copy_O = (
            cute.make_tiled_copy(copy_atom_store_O, tv_layout, tiler_mn)
            .get_slice(tidx)
        )

        tWgW = thr_copy_W.partition_S(gW)
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_O.partition_D(gO)
        tXrRstd = (
            thr_copy_O.partition_D(gRstd)
            if cutlass.const_expr(mRstd is not None)
            else None
        )
        tXrMean = (
            thr_copy_O.partition_D(gMean)
            if cutlass.const_expr(mMean is not None) else None
        )
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        # allocate fragments for gmem->rmem
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)
        tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        tXpX = utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
        row = tXcX[0][0]
        if row < shape[0]:
            cute.copy(copy_atom_load_X_async, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()

        tWpW = utils.predicate_k(thr_copy_W.partition_S(cX), limit=shape[1])
        if cutlass.const_expr(not delay_w_load):
            cute.copy(copy_atom_load_W, tWgW, tWrW, pred=tWpW)

        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)
        threads_per_row = tv_layout.shape[0][0]
        sum_x = utils.row_reduce(
            x,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr + 0 if cutlass.const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
            hook_fn=(
                cute.arch.cluster_wait
                if cutlass.const_expr(self.cluster_n > 1)
                else None
            ),
        )
        mean = sum_x / shape[1]
        if cutlass.const_expr(reload_from == "smem"):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(cute.Float32)
        elif cutlass.const_expr(reload_from == "gmem"):
            cute.copy(copy_atom_load_X, tXgX, tXrX, pred=tXpX)
            x = tXrX.load().to(cute.Float32)

        sum_sq_x_sub_mean = utils.row_reduce(
            (x - mean) * (x - mean),
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 1],
            mbar_ptr + 1 if cutlass.const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
        )
        rstd = utils.rsqrt(sum_sq_x_sub_mean / shape[1] + eps)
        if cutlass.const_expr(mRstd is not None):
            # Only the thread corresponding to column 0 writes out the rstd to
            # gmem
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (
                    self.cluster_n == 1
                    or cute.arch.block_idx_in_cluster() == 0
                )
            ):
                tXrRstd[0] = rstd
        if cutlass.const_expr(mMean is not None):
            # Only the thread corresponding to column 0 writes out the mean to
            # gmem
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (
                    self.cluster_n == 1
                    or cute.arch.block_idx_in_cluster() == 0
                )
            ):
                tXrMean[0] = mean
        if cutlass.const_expr(delay_w_load):
            cute.copy(copy_atom_load_W, tWgW, tWrW, pred=tWpW)
        if cutlass.const_expr(reload_from == "smem"):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(cute.Float32)
        elif cutlass.const_expr(reload_from == "gmem"):
            cute.copy(copy_atom_load_X, tXgX, tXrX, pred=tXpX)
            x = tXrX.load().to(cute.Float32)
        x_hat = (x - mean) * rstd
        w = tXrW.load().to(cute.Float32)
        y = x_hat * w
        tXrO.store(y.to(tXrO.element_type))
        tOpO = utils.predicate_k(thr_copy_O.partition_S(cX), limit=shape[1])
        if row < shape[0]:
            cute.copy(copy_atom_store_O, tXrO, tXgO, pred=tOpO)


def layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    return_rstd: bool = False,
    return_mean: bool = False,
) -> torch.Tensor:
    """LayerNorm forward pass.

    Args:
        x: Input tensor of shape (M, N)
        weight: Weight tensor of shape (N,)
        eps: Small value for numerical stability
        return_rstd: Whether to return the reciprocal standard deviation
        return_mean: Whether to return the mean

    Returns:
        Normalized output tensor of same shape as x
        If return_rstd is True, also returns rstd tensor of shape (M,)
        If return_mean is True, also returns mean tensor of shape (M,)
    """
    assert x.dim() == 2, "Input must be 2D"
    assert weight.dim() == 1, "Weight must be 1D"
    assert x.shape[-1] == weight.shape[0], (
        "Last dimension of input must match weight dimension"
    )
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        "Unsupported dtype"
    )
    assert weight.dtype == torch.float32, "Weight must be float32"
    M, N = x.shape
    device = x.device
    out = torch.empty_like(x)
    rstd = (
        torch.empty(M, device=device, dtype=torch.float32)
        if return_rstd
        else None
    )
    mean = (
        torch.empty(M, device=device, dtype=torch.float32)
        if return_mean else None
    )
    dtype = torch2cute_dtype_map[x.dtype]
    convert_from_dlpack = lambda x: (
        from_dlpack(x.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
    )
    x_tensor, out_tensor = [
        convert_from_dlpack(t)
        for t in (x, out)
    ]
    weight_tensor = utils.convert_from_dlpack(
        weight.detach(),
        leading_dim=0,
        divisibility=128 // cutlass.Float32.width,
    )
    rstd_tensor = (
        from_dlpack(rstd.detach(), assumed_align=4)
        .mark_layout_dynamic(leading_dim=0)
        if rstd is not None
        else None
    )
    mean_tensor = (
        from_dlpack(mean.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=0,
        )
        if mean is not None
        else None
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_key = (dtype, N, rstd is not None, mean is not None)
    if compile_key not in layernorm.compile_cache:
        rmsnorm_op = LayerNorm(dtype, N)
        layernorm.compile_cache[compile_key] = cute.compile(
            rmsnorm_op,
            x_tensor,
            weight_tensor,
            out_tensor,
            rstd_tensor,
            mean_tensor,
            current_stream,
        )
    layernorm.compile_cache[compile_key](
        x_tensor,
        weight_tensor,
        out_tensor,
        rstd_tensor,
        mean_tensor,
        current_stream,
        eps,
    )
    return (
        (out, rstd, mean)
        if return_mean and return_rstd
        else (
            (out, rstd)
            if return_rstd and not return_mean
            else ((out, mean) if return_mean and not return_rstd else (out))
        )
    )


layernorm.compile_cache = {}
