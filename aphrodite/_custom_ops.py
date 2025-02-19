import contextlib
import functools
from typing import List, Optional, Tuple, Type, Union

import torch
from loguru import logger

import aphrodite.common.envs as envs
from aphrodite._core_ext import ScalarType
from aphrodite.common.utils import is_hip, print_warning_once
from aphrodite.platforms import current_platform

if not current_platform.is_tpu():
    try:
        import aphrodite._C
    except ImportError as e:
        if current_platform.is_cuda() or current_platform.is_rocm():
            print_warning_once(f"Failed to import from aphrodite._C with {e}")

if current_platform.is_rocm():
    import aphrodite._rocm_C  # noqa: F401

with contextlib.suppress(ImportError):
    # ruff: noqa: F401
    import aphrodite._moe_C

with contextlib.suppress(ImportError):
    # ruff: noqa: F401
    import aphrodite._xqa_C


def hint_on_error(fn):

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as e:
            msg = (
                f"Error in calling custom op {fn.__name__}: {e}\n"
                "Not implemented or built, mostly likely because the current current device "
                "does not support this kernel (less likely TORCH_CUDA_ARCH_LIST was set "
                "incorrectly while building)")
            logger.error(msg)
            raise NotImplementedError(msg) from e
        except AttributeError as e:
            msg = (
                f"Error in calling custom op {fn.__name__}: {e}\n"
                f"Possibly you have built or installed an obsolete version of aphrodite.\n"
                f"Please try a clean build and install of aphrodite,"
                f"or remove old built files such as aphrodite/*.so and build/ ."
            )
            logger.error(msg)
            raise e

    return wrapper


# activation ops
def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    torch.ops._C.silu_and_mul(out, x)


def gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    torch.ops._C.gelu_and_mul(out, x)


def gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    torch.ops._C.gelu_tanh_and_mul(out, x)


def gelu_fast(out: torch.Tensor, x: torch.Tensor) -> None:
    torch.ops._C.gelu_fast(out, x)


def gelu_new(out: torch.Tensor, x: torch.Tensor) -> None:
    torch.ops._C.gelu_new(out, x)


def gelu_quick(out: torch.Tensor, x: torch.Tensor) -> None:
    torch.ops._C.gelu_quick(out, x)


# page attention ops
def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    torch.ops._C.paged_attention_v1(
        out, query, key_cache, value_cache, num_kv_heads, scale, block_tables,
        seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype,
        k_scale, v_scale, tp_rank, blocksparse_local_blocks,
        blocksparse_vert_stride, blocksparse_block_size,
        blocksparse_head_sliding_step)


def paged_attention_v2(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    torch.ops._C.paged_attention_v2(
        out, exp_sum, max_logits, tmp_out, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len,
        alibi_slopes, kv_cache_dtype, k_scale, v_scale, tp_rank,
        blocksparse_local_blocks, blocksparse_vert_stride,
        blocksparse_block_size, blocksparse_head_sliding_step)


def paged_attention_rocm(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> None:
    torch.ops._rocm_C.paged_attention(out, exp_sum, max_logits, tmp_out, query,
                                      key_cache, value_cache, num_kv_heads,
                                      scale, block_tables, seq_lens,
                                      block_size, max_seq_len, alibi_slopes,
                                      kv_cache_dtype, k_scale, v_scale)


def xqa_paged_attention(
    out: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_embedding_dim: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> None:
    torch.ops._xqa_C.xqa_paged_attention(
        out, query, kv_cache, num_heads, num_kv_heads, rotary_embedding_dim, scale,
        block_tables, seq_lens, block_size, max_seq_len, kv_cache_dtype,
        k_scale, v_scale)


# pos encoding ops
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    torch.ops._C.rotary_embedding(positions, query, key, head_size,
                                  cos_sin_cache, is_neox)


def batched_rotary_embedding(positions: torch.Tensor, query: torch.Tensor,
                             key: torch.Tensor, head_size: int,
                             cos_sin_cache: torch.Tensor, is_neox: bool,
                             rot_dim: int,
                             cos_sin_cache_offsets: torch.Tensor) -> None:
    torch.ops._C.batched_rotary_embedding(positions, query, key, head_size,
                                          cos_sin_cache, is_neox, rot_dim,
                                          cos_sin_cache_offsets)


# layer norm ops
def rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
             epsilon: float) -> None:
    torch.ops._C.rms_norm(out, input, weight, epsilon)


def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, epsilon: float) -> None:
    torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)


def advance_step_flashattn(num_seqs: int, num_queries: int, block_size: int,
                           input_tokens: torch.Tensor,
                           sampled_token_ids: torch.Tensor,
                           input_positions: torch.Tensor,
                           seq_lens: torch.Tensor, slot_mapping: torch.Tensor,
                           block_tables: torch.Tensor) -> None:
    """Advance a step on GPU for existing inputs for a multi-step runner"""
    return torch.ops._C.advance_step_flashattn(num_seqs, num_queries,
                                               block_size, input_tokens,
                                               sampled_token_ids,
                                               input_positions, seq_lens,
                                               slot_mapping, block_tables)

def advance_step_flashinfer(num_seqs: int, num_queries: int, block_size: int,
                            input_tokens: torch.Tensor,
                            sampled_token_ids: torch.Tensor,
                            input_positions: torch.Tensor,
                            seq_lens: torch.Tensor, slot_mapping: torch.Tensor,
                            block_tables: torch.Tensor,
                            paged_kv_indices: torch.Tensor,
                            paged_kv_indptr: torch.Tensor,
                            paged_kv_last_page_len: torch.Tensor,
                            block_table_bound: torch.Tensor) -> None:
    return torch.ops._C.advance_step_flashinfer(
        num_seqs, num_queries, block_size, input_tokens, sampled_token_ids,
        input_positions, seq_lens, slot_mapping, block_tables,
        paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len,
        block_table_bound)


# quantization ops
# awq
def awq_dequantize(qweight: torch.Tensor, scales: torch.Tensor,
                   zeros: torch.Tensor, split_k_iters: int, thx: int,
                   thy: int) -> torch.Tensor:
    if envs.APHRODITE_USE_TRITON_AWQ:
        from aphrodite.quantization.awq_triton import awq_dequantize_triton
        return awq_dequantize_triton(qweight, scales, zeros)
    return torch.ops._C.awq_dequantize(qweight, scales, zeros, split_k_iters,
                                       thx, thy)


def awq_gemm(input: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor,
             scales: torch.Tensor, split_k_iters: int) -> torch.Tensor:
    if envs.APHRODITE_USE_TRITON_AWQ:
        from aphrodite.quantization.awq_triton import awq_gemm_triton
        return awq_gemm_triton(input, qweight, qzeros, scales, split_k_iters)
    return torch.ops._C.awq_gemm(input, qweight, qzeros, scales, split_k_iters)


# gptq
def gptq_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
              b_gptq_qzeros: torch.Tensor, b_gptq_scales: torch.Tensor,
              b_g_idx: torch.Tensor, use_exllama: bool,
              bit: int) -> torch.Tensor:
    return torch.ops._C.gptq_gemm(a, b_q_weight, b_gptq_qzeros, b_gptq_scales,
                                  b_g_idx, use_exllama, bit)


# TODO: has to be a better way to do this
try:
    torch.ops._C.gptq_gemm  # noqa B018
    @torch.library.register_fake("_C::gptq_gemm")
    def _gptq_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                        b_gptq_qzeros: torch.Tensor,
                        b_gptq_scales: torch.Tensor, b_g_idx: torch.Tensor,
                        use_exllama: bool, bit: int) -> torch.Tensor:
        return torch.empty((a.size(0), b_q_weight.size(1)),
                           dtype=a.dtype,
                           device=a.device)
except Exception:
    pass


def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor,
                 bit: int) -> None:
    torch.ops._C.gptq_shuffle(q_weight, q_perm, bit)


# squeezellm
def squeezellm_gemm(vec: torch.Tensor, mat: torch.Tensor, mul: torch.Tensor,
                    lookup_table: torch.Tensor) -> None:
    torch.ops._C.squeezellm_gemm(vec, mat, mul, lookup_table)


# marlin
def marlin_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                b_scales: torch.Tensor, workspace: torch.Tensor, size_m: int,
                size_n: int, size_k: int) -> torch.Tensor:
    return torch.ops._C.marlin_gemm(a, b_q_weight, b_scales, workspace, size_m,
                                    size_n, size_k)


# marlin_24
def gptq_marlin_24_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                        b_meta: torch.Tensor, b_scales: torch.Tensor,
                        workspace: torch.Tensor, b_q_type: ScalarType,
                        size_m: int, size_n: int, size_k: int) -> torch.Tensor:
    return torch.ops._C.gptq_marlin_24_gemm(a, b_q_weight, b_meta, b_scales,
                                            workspace, b_q_type, size_m,
                                            size_n, size_k)


# TODO: has to be a better way to do this
try:
    torch.ops._C.gptq_marlin_24_gemm  # noqa B018
    @torch.library.register_fake("_C::gptq_marlin_24_gemm")
    def _gptq_marlin_24_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                                  b_meta: torch.Tensor, b_scales: torch.Tensor,
                                  workspace: torch.Tensor,
                                  b_q_type: ScalarType, size_m: int,
                                  size_n: int, size_k: int) -> torch.Tensor:
        return torch.empty((size_m, size_n), device=a.device, dtype=a.dtype)

    @torch.library.register_fake("_C::gptq_marlin_gemm")
    def _gptq_marlin_gemm_fake(a: torch.Tensor,
                               b_q_weight: torch.Tensor,
                               b_scales: torch.Tensor,
                               b_zeros: torch.Tensor,
                               g_idx: torch.Tensor,
                               perm: torch.Tensor,
                               workspace: torch.Tensor,
                               b_q_type: ScalarType,
                               size_m: int,
                               size_n: int,
                               size_k: int,
                               is_k_full: bool,
                               has_zp: bool = False,
                               use_fp32_reduce: bool = False) -> torch.Tensor:
        return torch.empty((size_m, size_n), device=a.device, dtype=a.dtype)

    @torch.library.register_fake("_C::ggml_dequantize")
    def _ggml_dequantize_fake(W: torch.Tensor, quant_type: int, m: int,
                              n: int) -> torch.Tensor:
        return torch.empty((m, n), dtype=torch.float16, device=W.device)

    @torch.library.register_fake("_C::ggml_mul_mat_vec_a8")
    def _ggml_mul_mat_vec_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: int,
    ) -> torch.Tensor:
        return torch.empty((1, row), dtype=torch.float16, device=W.device)

    @torch.library.register_fake("_C::ggml_mul_mat_a8")
    def _ggml_mul_mat_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: int,
    ) -> torch.Tensor:
        batch = X.size(0)
        return torch.empty((batch, row), dtype=torch.float16, device=W.device)

    @torch.library.register_fake("_C::marlin_qqq_gemm")
    def _marlin_qqq_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                              s_tok: torch.Tensor, s_ch: torch.Tensor,
                              s_group: torch.Tensor, workspace: torch.Tensor,
                              size_m: int, size_n: int,
                              size_k: int) -> torch.Tensor:
        return torch.empty((size_m, size_n),
                           dtype=torch.float16,
                           device=a.device)

    @torch.library.register_fake("_C::marlin_gemm")
    def _marlin_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                          b_scales: torch.Tensor, workspace: torch.Tensor,
                          size_m: int, size_n: int,
                          size_k: int) -> torch.Tensor:
        return torch.empty((size_m, size_n),
                           dtype=torch.float16,
                           device=a.device)

    @torch.library.register_fake("_C::awq_dequantize")
    def _awq_dequantize_fake(qweight: torch.Tensor, scales: torch.Tensor,
                             zeros: torch.Tensor, split_k_iters: int, thx: int,
                             thy: int) -> torch.Tensor:
        in_c = qweight.size(0)
        qout_c = qweight.size(1)
        out_c = qout_c * 8
        return torch.empty((in_c, out_c),
                           dtype=scales.dtype,
                           device=scales.device)

    @torch.library.register_fake("_C::awq_gemm")
    def _awq_gemm_fake(input: torch.Tensor, qweight: torch.Tensor,
                       qzeros: torch.Tensor, scales: torch.Tensor,
                       split_k_iters: int) -> torch.Tensor:
        num_in_feats = input.size(0)
        return torch.empty((split_k_iters, num_in_feats, qweight.size(1) * 8),
                           dtype=input.dtype,
                           device=input.device).sum(0)

    @torch.library.register_fake("_C::aqlm_gemm")
    def _aqlm_gemm_fake(input: torch.Tensor, codes: torch.Tensor,
                        codebooks: torch.Tensor, scales: torch.Tensor,
                        codebook_partition_sizes: List[int],
                        bias: Optional[torch.Tensor]) -> torch.Tensor:
        out_features = codes.size(0) * codebooks.size(2)
        flat_input = input.reshape((-1, input.size(-1)))
        flat_output = torch.empty((flat_input.size(0), out_features),
                                  dtype=input.dtype,
                                  device=input.device)
        output_sizes = list(input.shape)
        output_sizes.pop()
        output_sizes.append(-1)
        return flat_output.reshape(tuple(output_sizes))

    @torch.library.register_fake("_C::aqlm_dequant")
    def _aqlm_dequant_fake(
            codes: torch.Tensor, codebooks: torch.Tensor,
            codebook_partition_sizes: List[int]) -> torch.Tensor:
        in_features = codes.size(1) * 8
        out_features = codes.size(0)
        return torch.empty((out_features, in_features),
                           dtype=codebooks.dtype,
                           device=codebooks.device)

    @torch.library.register_fake("_C::vptq_gemm")
    def _vptq_gemm_fake(input: torch.Tensor, indices: torch.Tensor,
                        codebooks: torch.Tensor, weight_scale: torch.Tensor,
                        weight_bias: torch.Tensor, g_i_o: List[int],
                        res: torch.Tensor, res_codebooks: torch.Tensor,
                        oi: torch.Tensor, oc: torch.Tensor,
                        invperm: torch.Tensor,
                        bias: torch.Tensor) -> torch.Tensor:
        out_features = g_i_o[2]
        flat_input = input.reshape((-1, input.size(-1)))
        flat_output = torch.empty((flat_input.size(0), out_features),
                                  dtype=input.dtype,
                                  device=input.device)

        output_sizes = list(input.shape)
        output_sizes.pop()
        output_sizes.append(-1)
        return flat_output.reshape(tuple(output_sizes))

    @torch.library.register_fake("_C::vptq_dequant")
    def _vptq_dequant_fake(indices: torch.Tensor, codebooks: torch.Tensor,
                           weight_scale: torch.Tensor,
                           weight_bias: torch.Tensor, g_i_o: List[int],
                           res: torch.Tensor, res_codebooks: torch.Tensor,
                           oi: torch.Tensor, oc: torch.Tensor,
                           invperm: torch.Tensor) -> torch.Tensor:
        in_features = g_i_o[1]
        out_features = g_i_o[2]
        return torch.empty((out_features, in_features),
                           dtype=codebooks.dtype,
                           device=codebooks.device)

    @torch.library.register_fake("_C::fp8_marlin_gemm")
    def _fp8_marlin_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                              b_scales: torch.Tensor, workspace: torch.Tensor,
                              num_bits: int, size_m: int, size_n: int,
                              size_k: int) -> torch.Tensor:
        return torch.empty((size_m, size_n), dtype=a.dtype, device=a.device)

    @torch.library.register_fake("_C::machete_gemm")
    def machete_gemm_fake(
        a: torch.Tensor,
        b_q: torch.Tensor,  # Should be the tensor returned by machete_prepack_B
        b_type: ScalarType,
        b_scales: Optional[torch.Tensor] = None,
        b_zeros: Optional[torch.Tensor] = None,
        b_group_size: Optional[int] = None,
        c: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        schedule: Optional[str] = None,
    ) -> torch.Tensor:
        m = a.size(0)
        n = b_q.size(1)
        return torch.empty((m, n), device=a.device, dtype=a.dtype)

    @torch.library.register_fake("_C::machete_prepack_B")
    def machete_prepack_B_fake(b_q_weight: torch.Tensor,
                               b_type: ScalarType) -> torch.Tensor:
        return torch.empty_like(b_q_weight)

    @torch.library.register_fake("_C::causal_conv1d_fwd")
    def causal_conv1d_fwd_fake(x: torch.Tensor, weight: torch.Tensor,
                               bias_: Optional[torch.Tensor],
                               conv_states: Optional[torch.Tensor],
                               cu_seq_len: Optional[torch.Tensor],
                               cache_indices: Optional[torch.Tensor],
                               has_initial_state: Optional[torch.Tensor],
                               silu_activation: bool) -> torch.Tensor:
        return torch.empty_like(x)

    @torch.library.register_fake("_C::causal_conv1d_update")
    def causal_conv1d_update_fake(
            x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor,
            bias_: Optional[torch.Tensor], silu_activation: bool,
            cache_seqlens: Optional[torch.Tensor],
            conv_state_indices: Optional[torch.Tensor]) -> torch.Tensor:
        return torch.empty_like(x)

    @torch.library.register_fake("_C::selective_scan_fwd")
    def selective_scan_fwd_fake(u: torch.Tensor, delta: torch.Tensor,
                                A: torch.Tensor, B: torch.Tensor,
                                C: torch.Tensor, D_: Optional[torch.Tensor],
                                z_: Optional[torch.Tensor],
                                delta_bias_: Optional[torch.Tensor],
                                delta_softplus: bool,
                                cu_seq_len: Optional[torch.Tensor],
                                cache_indices: Optional[torch.Tensor],
                                has_initial_state: Optional[torch.Tensor],
                                ssm_states: Optional[torch.Tensor]) -> None:
        return None


except Exception:
    pass


# fp8 marlin
def fp8_marlin_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                    b_scales: torch.Tensor, workspace: torch.Tensor,
                    num_bits: int, size_m: int, size_n: int,
                    size_k: int) -> torch.Tensor:
    return torch.ops._C.fp8_marlin_gemm(a, b_q_weight, b_scales, workspace,
                                        num_bits, size_m, size_n, size_k)


# cutlass
def cutlass_scaled_mm_supports_fp8(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_scaled_mm_supports_fp8(cuda_device_capability)


def cutlass_scaled_mm(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: Type[torch.dtype],
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)
    assert bias is None or bias.shape[0] == b.shape[
        1] and bias.dtype == out_dtype

    m = a.shape[0]
    n = b.shape[1]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)

    return out


def cutlass_scaled_mm_azp(a: torch.Tensor,
                          b: torch.Tensor,
                          scale_a: torch.Tensor,
                          scale_b: torch.Tensor,
                          out_dtype: torch.dtype,
                          azp_adj: torch.Tensor,
                          azp: Optional[torch.Tensor] = None,
                          bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)
    assert bias is None or bias.numel(
    ) == b.shape[1] and bias.dtype == out_dtype

    m = a.shape[0]
    n = b.shape[1]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    torch.ops._C.cutlass_scaled_mm_azp(out, a, b, scale_a, scale_b, azp_adj,
                                       azp, bias)
    return out


# aqlm
def aqlm_gemm(input: torch.Tensor, codes: torch.Tensor,
              codebooks: torch.Tensor, scales: torch.Tensor,
              codebook_partition_sizes: List[int],
              bias: Optional[torch.Tensor]) -> torch.Tensor:
    return torch.ops._C.aqlm_gemm(input, codes, codebooks, scales,
                                  codebook_partition_sizes, bias)


def aqlm_dequant(codes: torch.Tensor, codebooks: torch.Tensor,
                 codebook_partition_sizes: List[int]) -> torch.Tensor:
    return torch.ops._C.aqlm_dequant(codes, codebooks,
                                     codebook_partition_sizes)


# vptq
def vptq_gemm(input: torch.Tensor, indices: torch.Tensor,
              codebooks: torch.Tensor, weight_scale: torch.Tensor,
              weight_bias: torch.Tensor, g_i_o: List[int], res: torch.Tensor,
              res_codebooks: torch.Tensor, oi: torch.Tensor, oc: torch.Tensor,
              invperm: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.ops._C.vptq_gemm(input, indices, codebooks, weight_scale,
                                  weight_bias, g_i_o, res, res_codebooks, oi,
                                  oc, invperm, bias)


def vptq_dequant(
    indices: torch.Tensor,
    codebooks: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    g_i_o: List[int],
    res: torch.Tensor,
    res_codebooks: torch.Tensor,
    oi: torch.Tensor,
    oc: torch.Tensor,
    invperm: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._C.vptq_dequant(indices, codebooks, weight_scale,
                                     weight_bias, g_i_o, res, res_codebooks,
                                     oi, oc, invperm)


# gptq_marlin
def gptq_marlin_repack(b_q_weight: torch.Tensor, perm: torch.Tensor,
                       size_k: int, size_n: int,
                       num_bits: int) -> torch.Tensor:
    return torch.ops._C.gptq_marlin_repack(b_q_weight, perm, size_k, size_n,
                                           num_bits)


def awq_marlin_repack(b_q_weight: torch.Tensor, size_k: int, size_n: int,
                      num_bits: int) -> torch.Tensor:
    return torch.ops._C.awq_marlin_repack(b_q_weight, size_k, size_n, num_bits)


def gptq_marlin_moe_repack(b_q_weight: torch.Tensor, perm: torch.Tensor,
                           size_k: int, size_n: int,
                           num_bits: int) -> torch.Tensor:
    num_experts = b_q_weight.shape[0]
    assert size_k % 16 == 0
    output = torch.empty((num_experts, size_k // 16, size_n * (num_bits // 2)),
                         device=b_q_weight.device,
                         dtype=b_q_weight.dtype)
    for e in range(num_experts):
        output[e] = torch.ops._C.gptq_marlin_repack(b_q_weight[e], perm[e],
                                                    size_k, size_n, num_bits)
    return output


def gptq_marlin_gemm(a: torch.Tensor,
                     b_q_weight: torch.Tensor,
                     b_scales: torch.Tensor,
                     b_zeros: torch.Tensor,
                     g_idx: torch.Tensor,
                     perm: torch.Tensor,
                     workspace: torch.Tensor,
                     b_q_type: ScalarType,
                     size_m: int,
                     size_n: int,
                     size_k: int,
                     is_k_full: bool,
                     has_zp: bool = False,
                     use_fp32_reduce: bool = False) -> torch.Tensor:
    return torch.ops._C.gptq_marlin_gemm(a, b_q_weight, b_scales, b_zeros,
                                         g_idx, perm, workspace, b_q_type,
                                         size_m, size_n, size_k, is_k_full,
                                         has_zp, use_fp32_reduce)


# machete
def machete_supported_schedules(b_type: ScalarType) -> List[str]:
    return torch.ops._C.machete_supported_schedules(b_type)


def machete_gemm(
    a: torch.Tensor,
    b_q: torch.Tensor,  # Should be the tensor returned by machete_prepack_B
    b_type: ScalarType,
    b_scales: Optional[torch.Tensor] = None,
    b_zeros: Optional[torch.Tensor] = None,
    b_group_size: Optional[int] = None,
    c: Optional[torch.Tensor] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    schedule: Optional[str] = None,
) -> torch.Tensor:
    return torch.ops._C.machete_gemm(a, b_q, b_type, b_scales, b_zeros,
                                     b_group_size, c, alpha, beta, schedule)


def machete_prepack_B(b_q_weight: torch.Tensor,
                      b_type: ScalarType) -> torch.Tensor:
    return torch.ops._C.machete_prepack_B(b_q_weight, b_type)

def permute_cols(a: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return torch.ops._C.permute_cols(a, perm)


# fp8
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    num_token_padding: Optional[int] = None,
    scale_ub: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensors for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization
        num_token_padding: If specified, pad the first dimension
            of the output to at least this value.
         use_per_token_if_dynamic: Whether to do per_tensor or per_token
            in the dynamic quantization case.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    # This code assumes batch_dim and num_tokens are flattened
    assert (input.ndim == 2)
    shape = input.shape
     # For rocm, the output fp8 dtype is torch.float_e3m3fnuz
    out_dtype: torch.dtype = torch.float8_e4m3fnuz if \
        is_hip() else torch.float8_e4m3fn
    if num_token_padding:
        shape = (max(num_token_padding, input.shape[0]), shape[1])
    output = torch.empty(shape, device=input.device, dtype=out_dtype)

    if scale is None:
        if use_per_token_if_dynamic:
            scale = torch.empty((shape[0], 1),
                                device=input.device,
                                dtype=torch.float32)
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(
                output, input, scale, scale_ub)
        else:
            scale = torch.zeros(1, device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        # num_token_padding not implemented for this case
        assert (scale.numel() == 1 or num_token_padding is None)
        torch.ops._C.static_scaled_fp8_quant(output, input, scale)

    return output, scale


# int8
def scaled_int8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    azp: Optional[torch.Tensor] = None,
    symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize the input tensor to int8 and return the quantized tensor and scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
      Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] : Output int8 tensor, scales, and optionally azp.
    """
    output = torch.empty_like(input, dtype=torch.int8)
    if scale is not None:
        # static-per-tensor quantization.
        assert symmetric == (
            azp is
            None), "azp must only be provided for asymmetric quantization."
        torch.ops._C.static_scaled_int8_quant(output, input, scale, azp)
        return output, scale, None

    # dynamic-per-token quantization.
    input_scales = torch.empty((input.numel() // input.shape[-1], 1),
                               device=input.device,
                               dtype=torch.float32)
    input_azp = None if symmetric else torch.empty_like(input_scales,
                                                        dtype=torch.int32)
    torch.ops._C.dynamic_scaled_int8_quant(output, input, input_scales,
                                           input_azp)
    return output, input_scales, input_azp


# quip#
def quip_gemv(
    A: torch.Tensor,
    B: torch.Tensor,
    CB: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._C.quip_gemv(A, B, CB)


def quip_decompress(
    YIs: torch.Tensor,
    CB: torch.Tensor,
    Y: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._C.quip_decompress(YIs, CB, Y)


# qqq ops
def marlin_qqq_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                    s_tok: torch.Tensor, s_ch: torch.Tensor,
                    s_group: torch.Tensor, workspace: torch.Tensor,
                    size_m: int, size_n: int, size_k: int) -> torch.Tensor:
    return torch.ops._C.marlin_qqq_gemm(a, b_q_weight, s_tok, s_ch, s_group,
                                        workspace, size_m, size_n, size_k)


# gguf
def ggml_dequantize(W: torch.Tensor, quant_type: int, m: int,
                    n: int) -> torch.Tensor:
    return torch.ops._C.ggml_dequantize(W, quant_type, m, n)


def ggml_mul_mat_vec_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    return torch.ops._C.ggml_mul_mat_vec_a8(W, X, quant_type, row)


def ggml_mul_mat_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    return torch.ops._C.ggml_mul_mat_a8(W, X, quant_type, row)


# fp6
def fp_eXmY_linear_forward_cuda(
    EXPONENT: int,
    MANTISSA: int,
    _in_feats: torch.Tensor,
    _weights: torch.Tensor,
    _scales: torch.Tensor,
    splitK: int = 1,
) -> torch.Tensor:
    return torch.ops._C.fp_eXmY_linear_forward_cuda(EXPONENT, MANTISSA,
                                                    _in_feats, _weights,
                                                    _scales, splitK)


# mamba
def causal_conv1d_fwd(x: torch.Tensor, weight: torch.Tensor,
                      bias_: Optional[torch.Tensor],
                      conv_states: Optional[torch.Tensor],
                      query_start_loc: Optional[torch.Tensor],
                      cache_indices: Optional[torch.Tensor],
                      has_initial_state: Optional[torch.Tensor],
                      silu_activation: bool) -> torch.Tensor:
    return torch.ops._C.causal_conv1d_fwd(x, weight, bias_, conv_states,
                                          query_start_loc, cache_indices,
                                          has_initial_state, silu_activation)


def causal_conv1d_update(
        x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor,
        bias_: Optional[torch.Tensor], silu_activation: bool,
        cache_seqlens: Optional[torch.Tensor],
        conv_state_indices: Optional[torch.Tensor]) -> torch.Tensor:
    return torch.ops._C.causal_conv1d_update(x, conv_state, weight, bias_,
                                             silu_activation, cache_seqlens,
                                             conv_state_indices)


def selective_scan_fwd(
        u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
        C: torch.Tensor, D_: Optional[torch.Tensor],
        z_: Optional[torch.Tensor], delta_bias_: Optional[torch.Tensor],
        delta_softplus: bool, query_start_loc: Optional[torch.Tensor],
        cache_indices: Optional[torch.Tensor],
        has_initial_state: Optional[torch.Tensor], ssm_states: torch.Tensor):
    torch.ops._C.selective_scan_fwd(u, delta, A, B, C, D_, z_, delta_bias_,
                                    delta_softplus, query_start_loc,
                                    cache_indices, has_initial_state,
                                    ssm_states)


# moe
def moe_align_block_size(topk_ids: torch.Tensor, num_experts: int,
                         block_size: int, sorted_token_ids: torch.Tensor,
                         experts_ids: torch.Tensor,
                         num_tokens_post_pad: torch.Tensor) -> None:
    torch.ops._C.moe_align_block_size(topk_ids, num_experts, block_size,
                                      sorted_token_ids, experts_ids,
                                      num_tokens_post_pad)


def topk_softmax(topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                 token_expert_indicies: torch.Tensor,
                 gating_output: float) -> None:
    torch.ops._moe_C.topk_softmax(topk_weights, topk_ids,
                                  token_expert_indicies, gating_output)


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> None:
    torch.ops._C_cache_ops.reshape_and_cache(key, value, key_cache,
                                             value_cache, slot_mapping,
                                             kv_cache_dtype, k_scale, v_scale)


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> None:
    torch.ops._C_cache_ops.reshape_and_cache_flash(key, value, key_cache,
                                                   value_cache, slot_mapping,
                                                   kv_cache_dtype, k_scale,
                                                   v_scale)


def copy_blocks(key_caches: List[torch.Tensor],
                value_caches: List[torch.Tensor],
                block_mapping: torch.Tensor) -> None:
    torch.ops._C_cache_ops.copy_blocks(key_caches, value_caches, block_mapping)


def swap_blocks(src: torch.Tensor, dst: torch.Tensor,
                block_mapping: torch.Tensor) -> None:
    torch.ops._C_cache_ops.swap_blocks(src, dst, block_mapping)


def convert_fp8(output: torch.Tensor,
                input: torch.Tensor,
                scale: float = 1.0,
                kv_dtype: str = "fp8") -> None:
    torch.ops._C_cache_ops.convert_fp8(output, input, scale, kv_dtype)


def get_device_attribute(attribute: int, device: int) -> int:
    return torch.ops._C_cuda_utils.get_device_attribute(attribute, device)


def get_max_shared_memory_per_block_device_attribute(device: int) -> int:
    # ruff: noqa: E501
    return torch.ops._C_cuda_utils.get_max_shared_memory_per_block_device_attribute(
        device)


# custom ar
def init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor,
                   handles: List[str], offsets: List[int], rank: int,
                   full_nvlink: bool) -> int:
    return torch.ops._C_custom_ar.init_custom_ar(meta, rank_data, handles,
                                                 offsets, rank, full_nvlink)


def all_reduce_reg(fa: int, inp: torch.Tensor, out: torch.Tensor) -> None:
    torch.ops._C_custom_ar.all_reduce_reg(fa, inp, out)


def all_reduce_unreg(fa: int, inp: torch.Tensor, reg_buffer: torch.Tensor,
                     out: torch.Tensor) -> None:
    torch.ops._C_custom_ar.all_reduce_unreg(fa, inp, reg_buffer, out)


def dispose(fa: int) -> None:
    torch.ops._C_custom_ar.dispose(fa)


def meta_size() -> int:
    return torch.ops._C_custom_ar.meta_size()


def register_buffer(fa: int, t: torch.Tensor, handles: List[str],
                    offsets: List[int]) -> None:
    return torch.ops._C_custom_ar.register_buffer(fa, t, handles, offsets)


def get_graph_buffer_ipc_meta(fa: int) -> Tuple[List[str], List[int]]:
    return torch.ops._C_custom_ar.get_graph_buffer_ipc_meta(fa)


def register_graph_buffers(fa: int, handles: List[str],
                           offsets: List[List[int]]) -> None:
    torch.ops._C_custom_ar.register_graph_buffers(fa, handles, offsets)


# Sampling Kernels
def sampling_from_probs(probs: torch.Tensor,
                        uniform_samplers: torch.Tensor,
                        deterministic: bool = True,
                        check_nan: bool = False) -> torch.Tensor:
    if check_nan and torch.any(torch.isnan(probs)):
        raise ValueError("NaN detected in probs")
    return torch.ops._C.sampling_from_probs(probs, uniform_samplers,
                                            deterministic)

def _to_tensor_scalar_tuple(x):
    if isinstance(x, torch.Tensor):
        return (x, 0)
    else:
        return (None, x)
def top_p_sampling_from_probs(
        probs: torch.Tensor,
        uniform_samples: torch.Tensor,
        top_p: Union[torch.Tensor, float],
        deterministic: bool = True,
        check_nan: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    if check_nan and torch.any(torch.isnan(probs)):
        raise ValueError("NaN detected in probs")
    return torch.ops._C.top_p_sampling_from_probs(
        probs, uniform_samples, *_to_tensor_scalar_tuple(top_p), deterministic)

def top_k_sampling_from_probs(
        probs: torch.Tensor,
        uniform_samples: torch.Tensor,
        top_k: Union[torch.Tensor, int],
        deterministic: bool = True,
        check_nan: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    if check_nan and torch.any(torch.isnan(probs)):
        raise ValueError("NaN detected in probs")
    return torch.ops._C.top_k_sampling_from_probs(
        probs, uniform_samples, *_to_tensor_scalar_tuple(top_k), deterministic)

def min_p_sampling_from_probs(
        probs: torch.Tensor,
        uniform_samples: torch.Tensor,
        min_p: Union[torch.Tensor, float],
        deterministic: bool = True,
        check_nan: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    if check_nan and torch.any(torch.isnan(probs)):
        raise ValueError("NaN detected in probs")
    return torch.ops._C.min_p_sampling_from_probs(
        probs, uniform_samples, *_to_tensor_scalar_tuple(min_p), deterministic)

def top_k_mask_logits(
    logits: torch.Tensor,
    top_k: Union[torch.Tensor, int],
) -> torch.Tensor:
    return torch.ops._C.top_k_mask_logits(logits,
                                          *_to_tensor_scalar_tuple(top_k))

def top_p_renorm_prob(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    return torch.ops._C.top_p_renorm_prob(probs,
                                          *_to_tensor_scalar_tuple(top_p))

def top_k_renorm_prob(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
) -> torch.Tensor:
    return torch.ops._C.top_k_renorm_prob(probs,
                                          *_to_tensor_scalar_tuple(top_k))

def top_k_top_p_sampling_from_logits(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    check_nan: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if filter_apply_order == "top_k_first":
        masked_logits = top_k_mask_logits(probs, top_k)
        probs = torch.softmax(masked_logits, dim=-1)
        return top_p_sampling_from_probs(probs, uniform_samples, top_p,
                                         deterministic, check_nan)
    elif filter_apply_order == "joint":
        probs = torch.softmax(probs, dim=-1)
        if check_nan and torch.any(torch.isnan(probs)):
            raise ValueError("NaN detected in probs")
        return torch.ops._C.top_k_top_p_sampling_from_logits(
            probs, uniform_samples, *_to_tensor_scalar_tuple(top_k),
            *_to_tensor_scalar_tuple(top_p), deterministic)
    else:
        raise ValueError(f"Invalid filter_apply_order: {filter_apply_order}")

def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    check_nan: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if filter_apply_order == "top_k_first":
        renorm_probs = top_k_renorm_prob(probs, top_k)
        return top_p_sampling_from_probs(renorm_probs, uniform_samples, top_p,
                                         deterministic, check_nan)
    elif filter_apply_order == "joint":
        if check_nan and torch.any(torch.isnan(probs)):
            raise ValueError("NaN detected in probs")
        return torch.ops._C.top_k_top_p_sampling_from_probs(
            probs, uniform_samples, *_to_tensor_scalar_tuple(top_k),
            *_to_tensor_scalar_tuple(top_p), deterministic)
    else:
        raise ValueError(f"Invalid filter_apply_order: {filter_apply_order}")


# Flash Attention kernels
def fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alibi_slopes: torch.Tensor,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        softcap: float,
        return_softmax: bool,
        out: torch.Tensor,
        gen: Optional[torch.Generator] = None,
):
    return torch.ops._C.fwd(
        q,
        k,
        v,
        out,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        gen,
    )

def varlen_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        seqused_k: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
        alibi_slopes: Optional[torch.Tensor],
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float,
        softmax_scale: float,
        zero_tensors: bool,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        softcap: float,
        return_softmax: bool,
        gen: Optional[torch.Generator] = None,
):
    return torch.ops._C.varlen_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        gen,
    )


def fwd_kvcache(
        q: torch.Tensor,
        kcache: torch.Tensor,
        vcache: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        seqlens_k: Optional[torch.Tensor],
        rotary_cos: Optional[torch.Tensor],
        rotary_sin: Optional[torch.Tensor],
        cache_batch_idx: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
        alibi_slopes: Optional[torch.Tensor],
        out: Optional[torch.Tensor],
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        softcap: float,
        rotary_interleaved: bool,
        num_splits: int,
):
    return torch.ops._C.fwd_kvcache(
        q,
        kcache,
        vcache,
        k,
        v,
        seqlens_k,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        block_table,
        alibi_slopes,
        out,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        rotary_interleaved,
        num_splits,
    )


# TODO: remove this later
names_and_values = globals()
names_and_values_to_update = {}
# prepare variables to avoid dict size change during iteration
k, v, arg = None, None, None
fn_type = type(lambda x: x)
for k, v in names_and_values.items():
    # find functions that are defined in this file and have torch.Tensor
    # in their annotations. `arg == "torch.Tensor"` is used to handle
    # the case when users use `import __annotations__` to turn type
    # hints into strings.
    if isinstance(v, fn_type) \
        and v.__code__.co_filename == __file__ \
        and any(arg is torch.Tensor or arg == "torch.Tensor"
                for arg in v.__annotations__.values()):
        names_and_values_to_update[k] = hint_on_error(v)

names_and_values.update(names_and_values_to_update)
del names_and_values_to_update, names_and_values, v, k, fn_type
