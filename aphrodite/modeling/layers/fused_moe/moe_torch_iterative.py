import torch
import torch.nn.functional as F

import aphrodite.common.envs as envs
from aphrodite.modeling.layers.fused_moe.fused_moe import (
    activation_without_mul, fused_topk)

def _moe_regular(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    override_config=None,
    use_grouped_topk: bool = False,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor = None,
) -> torch.Tensor:
    """
    This function is a pure python version of the fused_moe kernel.
    It is used for debugging and testing only.
    """
    #
    assert not inplace, "inplace=True is not supported in the pure python version"
    num_tokens, hidden_dim = hidden_states.shape
    num_experts = w1.shape[0]
    intermediate_size = w1.shape[2]
    # Get top-k experts for each token
    topk_weights, topk_ids, _ = fused_topk(hidden_states, gating_output, topk,
                                           renormalize, use_grouped_topk,
                                           num_expert_group, topk_group,
                                           scoring_func,
                                           routed_scaling_factor,
                                           e_score_correction_bias)
    final_hidden_states = torch.zeros_like(hidden_states)
    for i in range(num_tokens):
        for j in range(topk):
            expert_id = topk_ids[i, j].item()
            w1_expert = w1[expert_id, :, :]
            w2_expert = w2[expert_id, :, :]
            # gate_proj and up_proj
            gate_up = F.linear(hidden_states[i], w1_expert)
            # activation function
            gate, up = gate_up.chunk(2, dim=-1)
            gate = F.silu(gate)
            # element wise multiplication
            intermediate = gate * up
            # down_proj
            down = F.linear(intermediate, w2_expert)
            # multiply by routing weight
            down = down * topk_weights[i, j]
            # add to final hidden states
            final_hidden_states[i] += down
    return final_hidden_states


def _moe_lora(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    global_num_experts: int,
    expert_map: torch.Tensor = None,
    renormalize: bool = False,
) -> torch.Tensor:
    """
    Args:
        hidden_states: [*, hidden_size]
        w1: [num_experts, intermediate_size * 2, hidden_size]
        w2: [num_experts, hidden_size, intermediate_size]
        gating_output: [*, num_experts]
        expert_map: [num_experts]
    """
    orig_shape = hidden_states.shape
    hidden_size = hidden_states.shape[-1]
    num_tokens = hidden_states.shape[:-1].numel()
    num_experts = w1.shape[0]
    intermediate_size = w2.shape[-1]
    dtype = hidden_states.dtype

    hidden_states = hidden_states.view(num_tokens, hidden_size)
    gating_output = gating_output.view(num_tokens, global_num_experts)
    topk_weights = gating_output.softmax(dim=-1, dtype=torch.float)
    topk_weights, selected_experts = topk_weights.topk(topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    if expert_map is not None:
        selected_experts = expert_map[selected_experts]

    final_hidden_states = None
    for expert_idx in range(num_experts):
        expert_w1 = w1[expert_idx]
        expert_w2 = w2[expert_idx]
        expert_mask = (selected_experts == expert_idx)
        expert_weights = (topk_weights * expert_mask).sum(dim=-1, keepdim=True)
        x = F.linear(hidden_states, expert_w1)
        gate = F.silu(x[:, :intermediate_size])
        x = x[:, intermediate_size:] * gate
        x = F.linear(x, expert_w2)
        current_hidden_states = x * expert_weights
        if final_hidden_states is None:
            final_hidden_states = current_hidden_states
        else:
            final_hidden_states = final_hidden_states + current_hidden_states

    return final_hidden_states.view(orig_shape)  # type: ignore


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    override_config=None,
    use_grouped_topk: bool = False,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor = None,
    global_num_experts: int = None,
    expert_map: torch.Tensor = None,
) -> torch.Tensor:
    if envs.APHRODITE_ENABLE_LORA_ON_MOE:
        return _moe_lora(hidden_states, w1, w2, gating_output, topk,
                         global_num_experts, expert_map, renormalize)
    return _moe_regular(hidden_states, w1, w2, gating_output, topk,
                        renormalize, inplace, override_config,
                        use_grouped_topk, num_expert_group, topk_group,
                        scoring_func, routed_scaling_factor,
                        e_score_correction_bias)