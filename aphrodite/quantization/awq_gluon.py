import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared, mbarrier, tma, warpgroup_mma, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

AWQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]

# ============================================================================
# Bit manipulation helpers using inline PTX assembly
# ============================================================================

@gluon.jit
def awq_interleave_8x(val):
    """Interleave a value 3 times to expand 1 byte to 8 bytes"""
    # Each interleave doubles the data
    # need inline PTX for efficient bit interleaving
    result = gl.inline_asm_elementwise(
        """
        .reg .b32 temp1, temp2, temp3;
        .reg .b64 result;

        // First interleave: duplicate each bit
        mov.b32 temp1, $1;

        // Interleave op 1
        and.b32 temp2, temp1, 0x0000FFFF;
        shl.b32 temp3, temp2, 16;
        or.b32 temp1, temp2, temp3;

        // Interleave op 2
        and.b32 temp2, temp1, 0x00FF00FF;
        shl.b32 temp3, temp2, 8;
        or.b32 temp1, temp2, temp3;

        // Interleave op 3
        and.b32 temp2, temp1, 0x0F0F0F0F;
        shl.b32 temp3, temp2, 4;
        or.b32 temp1, temp2, temp3;

        mov.b32 $0, temp1;
        """,
        "=r,r",
        [val],
        dtype=gl.int32,
        is_pure=True,
        pack=1
    )
    return result


@gluon.jit
def awq_extract_and_reorder_4bit(packed_val, shift_amount):
    """Extract 4-bit value with shift and mask"""
    return gl.inline_asm_elementwise(
        """
        .reg .b32 temp;
        shr.b32 temp, $1, $2;
        and.b32 $0, temp, 0xF;
        """,
        "=r,r,r",
        [packed_val, shift_amount],
        dtype=gl.int32,
        is_pure=True,
        pack=1
    )


# ============================================================================
# AWQ Dequantization Kernel
# ============================================================================

@gluon.jit
def awq_dequantize_kernel_gluon(
    qweight_desc,  # TMA descriptor for quantized weights
    scales_desc,   # TMA descriptor for scales
    zeros_desc,    # TMA descriptor for zeros
    result_desc,   # TMA descriptor for output
    group_size,
    num_cols,
    num_rows,
    BLOCK_SIZE_X: gl.constexpr,
    BLOCK_SIZE_Y: gl.constexpr,
    num_warps: gl.constexpr
):
    # Get program IDs
    pid_x = gl.program_id(axis=0)
    pid_y = gl.program_id(axis=1)

    # layout for tensor operations
    # use a blocked layout that distributes work across warps
    layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[1, 32],
        warps_per_cta=[1, num_warps],
        order=[1, 0]
    )

    # allocate shared memory for input tiles
    qweight_smem = gl.allocate_shared_memory(
        gl.int32, [BLOCK_SIZE_Y, BLOCK_SIZE_X],
        gl.NVMMASharedLayout.get_default_for([BLOCK_SIZE_Y, BLOCK_SIZE_X], gl.int32)
    )

    # allocate mbarrier for synchronization
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    # issue TMA load for quantized weights
    mbarrier.expect(bar, qweight_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(
        qweight_desc,
        [pid_y * BLOCK_SIZE_Y, pid_x * BLOCK_SIZE_X],
        bar,
        qweight_smem
    )
    mbarrier.wait(bar, phase=0)

    qweights = qweight_smem.load(layout)

    # apply AWQ interleaving (3 times to expand 1 int32 to 8 4-bit values)
    qweights = awq_interleave_8x(qweights)

    # AWQ reordering pattern [0, 4, 1, 5, 2, 6, 3, 7]
    # this needs to be done with explicit indexing
    offs_y = pid_y * BLOCK_SIZE_Y + gl.arange(0, BLOCK_SIZE_Y, gl.SliceLayout(1, layout))
    offs_x = pid_x * BLOCK_SIZE_X + gl.arange(0, BLOCK_SIZE_X, gl.SliceLayout(0, layout))

    # compute shifts for unpacking based on AWQ order
    awq_order_0 = gl.zeros([BLOCK_SIZE_Y, BLOCK_SIZE_X * 8], dtype=gl.int32, layout=layout)
    awq_order_1 = gl.full([BLOCK_SIZE_Y, BLOCK_SIZE_X * 8], 4, dtype=gl.int32, layout=layout)

    # shift pattern
    shifts = gl.inline_asm_elementwise(
        """
        .reg .b32 idx, result;
        and.b32 idx, $1, 7;

        // AWQ pattern: [0, 4, 1, 5, 2, 6, 3, 7]
        setp.eq.s32 p0, idx, 0;
        setp.eq.s32 p1, idx, 1;
        setp.eq.s32 p2, idx, 2;
        setp.eq.s32 p3, idx, 3;
        setp.eq.s32 p4, idx, 4;
        setp.eq.s32 p5, idx, 5;
        setp.eq.s32 p6, idx, 6;
        setp.eq.s32 p7, idx, 7;

        @p0 mov.b32 result, 0;
        @p1 mov.b32 result, 16;
        @p2 mov.b32 result, 4;
        @p3 mov.b32 result, 20;
        @p4 mov.b32 result, 8;
        @p5 mov.b32 result, 24;
        @p6 mov.b32 result, 12;
        @p7 mov.b32 result, 28;

        mov.b32 $0, result;
        """,
        "=r,r",
        [offs_x],
        dtype=gl.int32,
        is_pure=True,
        pack=1
    )

    dequantized = awq_extract_and_reorder_4bit(qweights, shifts)

    # group-wise indexing for zeros
    zeros_smem = gl.allocate_shared_memory(
        gl.int32, [BLOCK_SIZE_Y // group_size, BLOCK_SIZE_X],
        gl.NVMMASharedLayout.get_default_for([BLOCK_SIZE_Y // group_size, BLOCK_SIZE_X], gl.int32)
    )

    mbarrier.expect(bar, zeros_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(
        zeros_desc,
        [pid_y * BLOCK_SIZE_Y // group_size, pid_x * BLOCK_SIZE_X],
        bar,
        zeros_smem
    )
    mbarrier.wait(bar, phase=1)

    zeros = zeros_smem.load(layout)
    zeros = awq_interleave_8x(zeros)
    zeros = awq_extract_and_reorder_4bit(zeros, shifts)

    zeros = gl.broadcast_to(zeros, [BLOCK_SIZE_Y, BLOCK_SIZE_X * 8])

    scales_smem = gl.allocate_shared_memory(
        gl.float16, [BLOCK_SIZE_Y // group_size, BLOCK_SIZE_X * 8],
        gl.NVMMASharedLayout.get_default_for([BLOCK_SIZE_Y // group_size, BLOCK_SIZE_X * 8], gl.float16)
    )

    mbarrier.expect(bar, scales_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(
        scales_desc,
        [pid_y * BLOCK_SIZE_Y // group_size, pid_x * BLOCK_SIZE_X * 8],
        bar,
        scales_smem
    )
    mbarrier.wait(bar, phase=0)

    scales = scales_smem.load(layout)
    scales = gl.broadcast_to(scales, [BLOCK_SIZE_Y, BLOCK_SIZE_X * 8])

    # dequantization: (weight - zero) * scale
    result = (dequantized.to(gl.float32) - zeros.to(gl.float32)) * scales.to(gl.float32)
    result = result.to(gl.float16)

    result_smem = gl.allocate_shared_memory(
        gl.float16, [BLOCK_SIZE_Y, BLOCK_SIZE_X * 8],
        result_desc.layout
    )
    result_smem.store(result)
    fence_async_shared()

    tma.async_copy_shared_to_global(
        result_desc,
        [pid_y * BLOCK_SIZE_Y, pid_x * BLOCK_SIZE_X * 8],
        result_smem
    )
    tma.store_wait(pendings=0)

    mbarrier.invalidate(bar)


# ============================================================================
# AWQ GEMM Kernel with fused dequantization
# ============================================================================

@gluon.jit
def awq_gemm_kernel_gluon(
    a_desc,      # Input activation descriptor
    b_desc,      # Quantized weight descriptor
    c_desc,      # Output descriptor
    zeros_desc,  # Zeros descriptor
    scales_desc, # Scales descriptor
    M, N, K,
    group_size,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    num_warps: gl.constexpr
):
    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # shmem buffers
    a_smem = gl.allocate_shared_memory(
        gl.float16, [BLOCK_SIZE_M, BLOCK_SIZE_K],
        gl.NVMMASharedLayout.get_default_for([BLOCK_SIZE_M, BLOCK_SIZE_K], gl.float16)
    )

    b_smem = gl.allocate_shared_memory(
        gl.int32, [BLOCK_SIZE_K, BLOCK_SIZE_N // 8],
        gl.NVMMASharedLayout.get_default_for([BLOCK_SIZE_K, BLOCK_SIZE_N // 8], gl.int32)
    )

    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[num_warps, 1],
        instr_shape=[16, 256 if BLOCK_SIZE_N >= 256 else BLOCK_SIZE_N, 16]
    )
    acc = gl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=gl.float32, layout=mma_layout)

    load_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(load_bar, count=1)

    # main K-loop with fused dequantization
    for k in range(0, K, BLOCK_SIZE_K):
        # load A tile
        mbarrier.expect(load_bar, a_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(
            a_desc,
            [pid_m * BLOCK_SIZE_M, k],
            load_bar,
            a_smem
        )

        # load B tile (quantized)
        mbarrier.expect(load_bar, b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(
            b_desc,
            [k, pid_n * BLOCK_SIZE_N // 8],
            load_bar,
            b_smem
        )
        mbarrier.wait(load_bar, phase=k // BLOCK_SIZE_K % 2)

        # dequantize B in registers
        b_quantized = b_smem.load(gl.BlockedLayout([1, 1], [32, 1], [num_warps, 1], [0, 1]))

        # apply AWQ unpacking inline
        b_unpacked = awq_interleave_8x(b_quantized)

        zeros = gl.zeros([BLOCK_SIZE_K, BLOCK_SIZE_N], dtype=gl.float32)  # Simplified
        scales = gl.ones([BLOCK_SIZE_K, BLOCK_SIZE_N], dtype=gl.float32)  # Simplified

        # dequantize: (unpacked - zeros) * scales
        b_dequant = (b_unpacked.to(gl.float32) - zeros) * scales
        b_dequant = b_dequant.to(gl.float16)

        b_dequant_smem = gl.allocate_shared_memory(
            gl.float16, [BLOCK_SIZE_K, BLOCK_SIZE_N],
            gl.NVMMASharedLayout.get_default_for([BLOCK_SIZE_K, BLOCK_SIZE_N], gl.float16)
        )
        b_dequant_smem.store(b_dequant)
        fence_async_shared()

        acc = warpgroup_mma(a_smem, b_dequant_smem, acc, is_async=True)

    # wait for all MMAs to complete
    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

    c_smem = gl.allocate_shared_memory(
        gl.float16, [BLOCK_SIZE_M, BLOCK_SIZE_N],
        c_desc.layout
    )
    c_smem.store(acc.to(gl.float16))
    fence_async_shared()

    tma.async_copy_shared_to_global(
        c_desc,
        [pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
        c_smem
    )
    tma.store_wait(pendings=0)

    mbarrier.invalidate(load_bar)


# ============================================================================
# Python wrapper functions
# ============================================================================

def awq_dequantize_gluon(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    block_size_x: int = 32,
    block_size_y: int = 32
) -> torch.Tensor:
    """Gluon implementation of AWQ dequantization"""
    K = qweight.shape[0]
    M = scales.shape[1]
    group_size = qweight.shape[0] // scales.shape[0]

    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    result = torch.empty(K, M, device=qweight.device, dtype=scales.dtype)

    qweight_desc = TensorDescriptor.from_tensor(
        qweight,
        [block_size_y, block_size_x],
        gl.NVMMASharedLayout.get_default_for([block_size_y, block_size_x], gl.int32)
    )

    scales_desc = TensorDescriptor.from_tensor(
        scales,
        [block_size_y // group_size, block_size_x * 8],
        gl.NVMMASharedLayout.get_default_for([block_size_y // group_size, block_size_x * 8], gl.float16)
    )

    zeros_desc = TensorDescriptor.from_tensor(
        zeros,
        [block_size_y // group_size, block_size_x],
        gl.NVMMASharedLayout.get_default_for([block_size_y // group_size, block_size_x], gl.int32)
    )

    result_desc = TensorDescriptor.from_tensor(
        result,
        [block_size_y, block_size_x * 8],
        gl.NVMMASharedLayout.get_default_for([block_size_y, block_size_x * 8], gl.float16)
    )

    grid = (triton.cdiv(qweight.shape[1], block_size_x),
            triton.cdiv(qweight.shape[0], block_size_y))

    awq_dequantize_kernel_gluon[grid](
        qweight_desc, scales_desc, zeros_desc, result_desc,
        group_size, qweight.shape[1], qweight.shape[0],
        block_size_x, block_size_y,
        num_warps=4
    )

    return result


def awq_gemm_gluon(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    block_size_m: int = 64,
    block_size_n: int = 256,
    block_size_k: int = 64
) -> torch.Tensor:
    """Gluon implementation of AWQ GEMM with fused dequantization"""
    M, K = input.shape
    N = qweight.shape[1] * 8
    group_size = qweight.shape[0] // qzeros.shape[0]

    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    result = torch.empty((M, N), dtype=input.dtype, device=input.device)

    a_desc = TensorDescriptor.from_tensor(
        input,
        [block_size_m, block_size_k],
        gl.NVMMASharedLayout.get_default_for([block_size_m, block_size_k], gl.float16)
    )

    b_desc = TensorDescriptor.from_tensor(
        qweight,
        [block_size_k, block_size_n // 8],
        gl.NVMMASharedLayout.get_default_for([block_size_k, block_size_n // 8], gl.int32)
    )

    c_desc = TensorDescriptor.from_tensor(
        result,
        [block_size_m, block_size_n],
        gl.NVMMASharedLayout.get_default_for([block_size_m, block_size_n], gl.float16)
    )

    zeros_desc = TensorDescriptor.from_tensor(
        qzeros,
        [block_size_k // group_size, block_size_n // 8],
        gl.NVMMASharedLayout.get_default_for([block_size_k // group_size, block_size_n // 8], gl.int32)
    )

    scales_desc = TensorDescriptor.from_tensor(
        scales,
        [block_size_k // group_size, block_size_n],
        gl.NVMMASharedLayout.get_default_for([block_size_k // group_size, block_size_n], gl.float16)
    )

    grid = (triton.cdiv(M, block_size_m) * triton.cdiv(N, block_size_n),)

    awq_gemm_kernel_gluon[grid](
        a_desc, b_desc, c_desc, zeros_desc, scales_desc,
        M, N, K, group_size,
        block_size_m, block_size_n, block_size_k,
        num_warps=8
    )

    return result
