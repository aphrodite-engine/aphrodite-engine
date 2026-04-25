# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from aphrodite.v1.spec_decode.ddtree import build_ddtree_tree, follow_verified_tree


def test_build_ddtree_tree_returns_expected_budget_and_root_access():
    logits = torch.tensor(
        [
            [9.0, 2.0, 1.0],
            [8.0, 7.0, 1.0],
            [6.0, 5.0, 4.0],
        ]
    )

    tree = build_ddtree_tree(logits, budget=3)

    assert tree.num_nodes == 3
    assert tree.query_len == 4
    assert tree.parents[0] == -1
    assert torch.all(tree.attn_bias[:, 0] == 0)
    assert torch.all(torch.diag(tree.attn_bias) == 0)


def test_follow_verified_tree_accepts_matching_prefix_then_returns_bonus():
    logits = torch.tensor(
        [
            [9.0, 2.0, 1.0],
            [8.0, 7.0, 1.0],
            [6.0, 5.0, 4.0],
        ]
    )
    tree = build_ddtree_tree(logits, budget=3)

    first_child = int(tree.node_token_ids[0])
    accepted_indices, bonus_token = follow_verified_tree(
        tree,
        torch.tensor([first_child, 1, 2, 0]),
    )

    assert accepted_indices == [0, 1]
    assert bonus_token == 1
