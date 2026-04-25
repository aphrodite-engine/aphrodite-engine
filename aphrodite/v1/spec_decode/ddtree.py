# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DDTreeRuntimeTree:
    node_token_ids: torch.Tensor
    node_depths: torch.Tensor
    parents: list[int]
    child_maps: list[dict[int, int]]
    attn_bias: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.node_token_ids.numel())

    @property
    def query_len(self) -> int:
        return self.num_nodes + 1


def build_ddtree_tree(
    draft_logits: torch.Tensor,
    budget: int,
) -> DDTreeRuntimeTree:
    if draft_logits.dim() == 1:
        draft_logits = draft_logits.unsqueeze(0)

    if budget <= 0 or draft_logits.shape[0] == 0:
        attn_bias = torch.zeros((1, 1), dtype=torch.float32, device=draft_logits.device)
        return DDTreeRuntimeTree(
            node_token_ids=torch.empty(0, dtype=torch.int32, device=draft_logits.device),
            node_depths=torch.empty(0, dtype=torch.int32, device=draft_logits.device),
            parents=[-1],
            child_maps=[{}],
            attn_bias=attn_bias,
        )

    topk = min(budget, draft_logits.shape[-1])
    depth_limit = int(draft_logits.shape[0])

    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs = (top_logits - log_z).to(dtype=torch.float32, device="cpu").numpy()
    top_token_ids_np = top_token_ids.to(dtype=torch.int64, device="cpu").numpy()

    first_logw = float(top_log_probs[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [(-first_logw, (0,), 0, 1, 0, first_logw)]

    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    parents_np = np.empty(budget + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [dict()]
    node_count = 0

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids_np[depth - 1, rank])
        current_index = node_count + 1
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        parents_np[current_index] = parent_index
        child_maps.append({})
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = logw - float(top_log_probs[depth - 1, rank]) + float(top_log_probs[depth - 1, rank + 1])
            heapq.heappush(
                heap,
                (-sibling_logw, sibling_ranks, parent_index, depth, rank + 1, sibling_logw),
            )

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs[depth, 0])
            heapq.heappush(heap, (-child_logw, child_ranks, current_index, depth + 1, 0, child_logw))

    node_token_ids = torch.from_numpy(node_token_ids_np[:node_count]).to(device=draft_logits.device, dtype=torch.int32)
    node_depths = torch.from_numpy(node_depths_np[:node_count]).to(device=draft_logits.device, dtype=torch.int32)
    parents = parents_np[: node_count + 1].tolist()
    attn_bias = build_tree_attn_bias(parents, device=draft_logits.device)
    return DDTreeRuntimeTree(
        node_token_ids=node_token_ids,
        node_depths=node_depths,
        parents=parents,
        child_maps=child_maps,
        attn_bias=attn_bias,
    )


def build_tree_attn_bias(
    parents: list[int],
    device: torch.device,
) -> torch.Tensor:
    tree_len = len(parents)
    bias = torch.full((tree_len, tree_len), -torch.inf, dtype=torch.float32, device=device)
    bias.fill_diagonal_(0)
    bias[:, 0] = 0
    for node_idx in range(1, tree_len):
        parent_idx = parents[node_idx]
        while parent_idx > 0:
            bias[node_idx, parent_idx] = 0
            parent_idx = parents[parent_idx]
    return bias


def follow_verified_tree(
    tree: DDTreeRuntimeTree,
    posterior_token_ids: torch.Tensor,
) -> tuple[list[int], int]:
    posterior = posterior_token_ids.view(-1).tolist()
    accepted_indices = [0]
    current_index = 0
    next_token = int(posterior[current_index])

    while next_token in tree.child_maps[current_index]:
        current_index = tree.child_maps[current_index][next_token]
        accepted_indices.append(current_index)
        next_token = int(posterior[current_index])

    return accepted_indices, next_token
