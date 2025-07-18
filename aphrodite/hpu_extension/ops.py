###############################################################################
# Copyright (C) 2024-2025 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
import math
from typing import Optional, Tuple

import habana_frameworks.torch as htorch
import habana_frameworks.torch.core as htcore
import torch
import torch.nn.functional as F

from aphrodite.hpu_extension.flags import enabled_flags


def grouped_max(block_max, batch_size, block_groups):
    group_max = torch.full([batch_size + 1, *block_max.shape[1:]],
                           -math.inf,
                           dtype=block_max.dtype,
                           device=block_max.device)
    group_max = group_max.index_reduce_(0, block_groups, block_max, 'amax')
    group_max = group_max.index_select(0, block_groups)
    return group_max


def b2b_impl(tensor, block_mapping, matmul_op):
    shape = tuple(tensor.shape)
    return matmul_op(block_mapping, tensor.view(shape[0],
                                                -1)).view(-1, *shape[1:])


def batch2block(tensor, block_mapping, matmul_op=torch.matmul):
    return b2b_impl(tensor, block_mapping, matmul_op)


def block2batch(tensor, block_mapping, matmul_op=torch.matmul):
    return b2b_impl(tensor, block_mapping.t(), matmul_op)


def pipelined_pa(attn, value, block_groups, block_mapping, batch_size,
                 matmul_av_op, batch2block_matmul_op, block2batch_matmul_op):
    # When fp32_softmax is enabled attn is left in fp32 after Q@K
    # We can return to native dtype after we renormalize and calculate the adjustments

    # Normalize the attention scores and cast attn to native dtype
    block_max = attn.amax(dim=-1, keepdim=True)
    adjustment_target_shape = block_max.shape
    attn = attn.sub(block_max)
    attn = attn.exp()
    attn = attn.to(value.dtype)
    block_sums = attn.sum(dim=-1, keepdim=True)
    attn = matmul_av_op(attn, value)
    block_max = block_max.squeeze()
    block_sums = block_sums.squeeze()

    # Calculate maximum of blocks that belong to the same sequences
    # and cast adjustments to native dtype
    group_max = grouped_max(block_max, batch_size, block_groups)
    block_adjustment = (block_max - group_max).exp()
    block_adjustment = block_adjustment.to(value.dtype)
    sum_adjusted = block_sums.mul(block_adjustment)

    # Sum block's sums that belongs to the same sequences
    group_sum_adjusted = block2batch(sum_adjusted, block_mapping,
                                     block2batch_matmul_op)
    group_sum_adjusted = batch2block(group_sum_adjusted, block_mapping,
                                     batch2block_matmul_op)
    sum_adjusted = sum_adjusted.view(*adjustment_target_shape)
    group_sum_adjusted = group_sum_adjusted.view(*adjustment_target_shape)
    block_adjustment = block_adjustment.view(*adjustment_target_shape)

    # For stability in case some of the sums have been zeroed out during block aggretation
    group_sum_adjusted = torch.maximum(group_sum_adjusted, sum_adjusted)

    # Post processing for the attention scores
    rescale = block_adjustment.div(group_sum_adjusted)
    attn = attn.mul(rescale)
    return attn


def flat_pa(query, key_cache, value_cache, block_list, block_mapping,
            block_bias, block_groups, scale, matmul_qk_op, matmul_av_op,
            batch2block_matmul_op, block2batch_matmul_op, keys_fetch_func,
            values_fetch_func, **ignored_args):
    batch_size, _, hidden_size = query.shape
    _, _, kv_heads, head_size = key_cache.shape
    q_heads = hidden_size // head_size

    query_shape = (-1, q_heads, 1, head_size)
    query = batch2block(scale * query, block_mapping,
                        batch2block_matmul_op).view(query_shape)
    key = keys_fetch_func(key_cache, block_list).transpose(1, 2)
    value = values_fetch_func(value_cache, block_list).transpose(1, 2)
    block_bias = block_bias.view(key.size(0), 1, 1, -1)
    if kv_heads != q_heads:
        block_bias = block_bias.unsqueeze(1)
        query = query.unflatten(1, (kv_heads, -1))
        key = key.unflatten(1, (kv_heads, 1))
        value = value.unflatten(1, (kv_heads, 1))
        key = key.transpose(3, 4)
    else:
        key = key.transpose(2, 3)

    attn = matmul_qk_op(query, key)
    if 'fp32_softmax' in enabled_flags():
        attn = attn.float()
        htcore.mark_step()
    attn = attn + block_bias
    attn = pipelined_pa(attn,
                        value,
                        block_groups,
                        block_mapping,
                        batch_size=batch_size,
                        matmul_av_op=matmul_av_op,
                        batch2block_matmul_op=batch2block_matmul_op,
                        block2batch_matmul_op=block2batch_matmul_op)
    attn = block2batch(attn, block_mapping, block2batch_matmul_op)
    attn = attn.squeeze(-2)
    if kv_heads != q_heads:
        attn = attn.flatten(1, 2)
    return attn


def _flex_prompt_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    **ignored_args,
) -> torch.Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    def _causal(
        score: torch.Tensor,
        batch: torch.Tensor,
        head: torch.Tensor,
        token_q: torch.Tensor,
        token_kv: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(token_q >= token_kv, score, float("-inf"))

    from torch.nn.attention.flex_attention import flex_attention

    attn_weights = flex_attention(
        query,
        key,
        value,
        score_mod=_causal,
        enable_gqa=True,
        return_lse=False,
        block_mask=None,
        scale=scale,
    )

    attn_weights = attn_weights.transpose(1, 2)
    return attn_weights


def _naive_prompt_attention(query: torch.Tensor,
                            key: torch.Tensor,
                            value: torch.Tensor,
                            scale: float,
                            attn_bias: Optional[torch.Tensor] = None,
                            matmul_qk_op=torch.matmul,
                            softmax_op=torch.softmax,
                            matmul_av_op=torch.matmul,
                            **ignored_args) -> torch.Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    query_heads = query.size(1)
    kv_heads = key.size(1)
    if query_heads != kv_heads:
        query = query.unflatten(1, (kv_heads, -1))
        key = key.unflatten(1, (kv_heads, 1))
        value = value.unflatten(1, (kv_heads, 1))
        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(2)
    attn_weights = matmul_qk_op(query * scale, key.transpose(-1, -2))
    if 'fp32_softmax' in enabled_flags():
        softmax_op = torch.softmax
        attn_weights = attn_weights.float()
        htcore.mark_step()
    if attn_bias is not None:
        attn_weights = attn_weights.add(attn_bias)
    attn_weights = softmax_op(attn_weights, dim=-1)
    attn_weights = attn_weights.to(query.dtype)
    attn_weights = matmul_av_op(attn_weights, value)
    if query_heads != kv_heads:
        attn_weights = attn_weights.flatten(1, 2)
    attn_weights = attn_weights.transpose(1, 2)
    return attn_weights


def _fsdpa_prompt_attention(query: torch.Tensor,
                            key: torch.Tensor,
                            value: torch.Tensor,
                            scale: float,
                            fsdpa_op,
                            is_causal: bool,
                            attn_bias: Optional[torch.Tensor] = None,
                            valid_seq_lengths: Optional[torch.Tensor] = None,
                            **ignored_args) -> torch.Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    if 'fp32_softmax' in enabled_flags():
        softmax_mode = 'fp32'
    else:
        softmax_mode = 'fast'
    recompute_mode = True
    assert attn_bias is not None or valid_seq_lengths is not None, \
        'Either attn_bias or valid_seq_lengths must be != None'
    if is_causal and attn_bias is not None:
        # TODO: causal + attn_bias is not yet supported
        is_causal = False
        valid_seq_lengths = None
    attn_weights = fsdpa_op(query, key, value, attn_bias, 0.0, is_causal,
                            scale, softmax_mode, recompute_mode,
                            valid_seq_lengths, 'right')
    attn_weights = attn_weights.transpose(1, 2)
    return attn_weights


def prompt_attention(
    impl: str,
    **args,
) -> torch.Tensor:
    _get_context(args)
    impl_mapping = {
        'naive': _naive_prompt_attention,
        'fsdpa': _fsdpa_prompt_attention,
        'flex': _flex_prompt_attention,
    }
    assert impl in impl_mapping, f'Unsupported implementation: {impl}'
    return impl_mapping[impl](**args)


def _get_all(data, *keys):
    return [data.get(k, None) for k in keys]


def _include_past(tensor_str, fn_str, cache_str, args):
    all_tensors = _get_all(args, tensor_str, fn_str, cache_str, 'block_list')
    if all(t is not None for t in all_tensors):
        current, fn, cache, block_list = all_tensors
        past = fn(cache, block_list)
        past = past.reshape(current.size(0), -1, past.shape[2], past.shape[3])
        current = torch.concat((past, current), dim=1)
        args[tensor_str] = current


def _get_context(args):
    _include_past('key', 'keys_fetch_func', 'key_cache', args)
    _include_past('value', 'values_fetch_func', 'value_cache', args)


class LoraMask:
    lora_mask = None

    @staticmethod
    def setLoraMask(mask):
        LoraMask.lora_mask = mask

    @staticmethod
    def getLoraMask():
        return LoraMask.lora_mask


def dispatch_bgmv_linear(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_t_all: torch.Tensor,
    wb_t_all: torch.Tensor,
    layer_idx: int,
    scale: float,
):
    """
    `wa_t_all` and `wb_t_all` contains all LoRA A and LoRA B weight matrices
    stacked at dimension 0 into single tensors, assuming same rank. `wa` is the
    reshaped and transposed version of `wa_t_all` of shape
    (h_in, max_loras * lora_rank) and `wb` is the transposed and reshaped
    version of `wb_t_all` of shape (max_loras * lora_rank, h_out).

    Matmul input `x` with `wa`. Multiply `x` with a mask to zero-out inputs of
    inactive LoRA indices. Matmul masked output with `wb` and scale it to get
    the final output.
    """

    assert layer_idx == 0, f'layer_idx should be 0, but got {layer_idx}'
    mask = LoraMask.getLoraMask()

    wa = wa_t_all[:, 0, :, :]
    wb = wb_t_all[:, 0, :, :].transpose(1, 2)
    wa = wa.reshape(wa.shape[0] * wa.shape[1], wa.shape[2]).transpose(0, 1)
    wb = wb.reshape(wb.shape[0] * wb.shape[1], wb.shape[2])

    out = x @ wa
    assert (out.shape == mask.shape)
    out = out * mask
    out = out @ wb
    y += out * scale


def dispatch_bgmv_embedding(
    y: torch.Tensor,
    x: torch.Tensor,
    wb_t_all: torch.Tensor,
    layer_idx: int,
):
    """
    `wb_t_all` contains all LoRA-B weight matrices stacked at dimension 0 into
    a single tensor, assuming same rank. `wb` is the transposed and reshaped
    version of `wb_t_all` of shape (num_loras * lora_rank, embedding_dim).

    Output of LoRA-A embedding (tensor x) is repeated max_loras times to match
    the shape of `wb`. Multiply `x` with a mask to zero-out inputs of inactive
    LoRA indices. Matmul masked output with `wb` and scale it to get the final
    output.
    """

    assert layer_idx == 0, f'layer_idx should be 0, but got {layer_idx}'
    max_loras = wb_t_all.size(0)

    x = x.repeat(1, max_loras)
    x = x * LoraMask.getLoraMask()
    wb = wb_t_all[:, 0, :, :].transpose(1, 2)
    wb = wb.reshape(wb.shape[0] * wb.shape[1], wb.shape[2])
    out = x @ wb
    y += out


class MoeMatmul(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def set_weight(self, w):
        self.weight = w

    def forward(self, state, expert_id, w):
        raise NotImplementedError()


class AphroditeMixtureOfExpertsOp(torch.nn.Module):

    def __init__(self, num_total_experts):
        super().__init__()
        self.w13_list = torch.nn.ModuleList(
            [MoeMatmul() for _ in range(num_total_experts)])
        self.w2_list = torch.nn.ModuleList(
            [MoeMatmul() for _ in range(num_total_experts)])
        self.num_experts = num_total_experts

    def forward(self,
                hidden_states,
                expert_routing_table,
                router_weights,
                permuted_weights=True,
                activation="silu"):
        # pre-processing for custom op inputs
        experts_range = range(self.num_experts)
        w1_list = [self.w13_list[i].weight.squeeze() for i in experts_range]
        w2_list = [self.w2_list[i].weight.squeeze() for i in experts_range]
        return torch.ops.hpu.mixture_of_experts(
            hidden_states=hidden_states,
            expert_routing_table=expert_routing_table,
            router_weights=router_weights,
            w12=w1_list,
            w3=w2_list,
            permuted_weights=permuted_weights,
            activation=activation,
            experts_min=0,
            experts_max=self.num_experts - 1)


class DynamicFusedMOE(torch.nn.Module):

    def __init__(self, num_total_experts):
        super().__init__()
        self.MoeOp = AphroditeMixtureOfExpertsOp(num_total_experts)

    def forward(self, hidden_states, score, topk):
        htorch.core.mark_step()
        routing_weights = F.softmax(score, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights,
                                                       topk,
                                                       dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = self.MoeOp(
            hidden_states=hidden_states,
            expert_routing_table=selected_experts,
            router_weights=routing_weights,
            permuted_weights=True,
            activation="silu",
        )

        return final_hidden_states.view(-1, hidden_states.shape[1])


# fp8
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    batch_dim_padding: Optional[int] = None,
    scale_ub: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.
    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensor for downstream kernels that
    will benefit from padding.
    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization
        scale_ub: Optional upper bound for scaling factor in dynamic
            per token case
        batch_dim_padding: If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic: Whether to do per_tensor or per_token
            in the dynamic quantization case.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    if batch_dim_padding:
        shape = (max(batch_dim_padding, input.shape[0]), *input.shape[1:])
        output = torch.empty(shape,
                             device=input.device,
                             dtype=torch.float8_e4m3fn)
    else:
        output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    if scale is None:
        raise NotImplementedError(
            "dynamic scaled_fp8_quant not implemented for HPU")
        # TODO: calculate scale to match gaudi2 240 range instead of 448
        if use_per_token_if_dynamic:
            scale = torch.empty((input.numel() // input.shape[-1], 1),
                                device=input.device,
                                dtype=torch.float32)
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(
                output, input, scale, scale_ub)
        else:
            scale = torch.zeros(1, device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        output = torch.ops.hpu.cast_to_fp8_v2(input,
                                              1 / scale,
                                              False,
                                              False,
                                              dtype=torch.float8_e4m3fn)[0]

    return output, scale
