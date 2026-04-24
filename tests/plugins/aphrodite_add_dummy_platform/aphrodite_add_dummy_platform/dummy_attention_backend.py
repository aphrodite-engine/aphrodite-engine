# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
from aphrodite.attention.backends.placeholder_attn import PlaceholderAttentionBackend


class DummyAttentionBackend(PlaceholderAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "Dummy_Backend"
