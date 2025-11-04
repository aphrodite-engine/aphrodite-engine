from dataclasses import dataclass

import torch
from torch import nn

from aphrodite.adapter_commons.layers import AdapterMapping
from aphrodite.modeling.layers.vocab_parallel_embedding import VocabParallelEmbedding


@dataclass
class PromptAdapterMapping(AdapterMapping):
    pass


class VocabParallelEmbeddingWithPromptAdapter(nn.Module):
    def __init__(self, base_layer: VocabParallelEmbedding) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.emb_layer = self.base_layer
        if "LoRA" in base_layer.__class__.__name__:
            self.emb_layer = self.base_layer.base_layer

    def set_mapping(
        self,
        prompt_indices: torch.Tensor,
        prompt_embedding_indices: torch.Tensor,
    ):
        self.indices_gpu = prompt_indices.to(device=self.emb_layer.weight.device)
        self.embedding_indices_gpu = prompt_embedding_indices.to(device=self.emb_layer.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.base_layer(x)
        if self.embedding_indices_gpu.ndim > 1:
            valid_mask = self.indices_gpu != -1
            gathered_embeddings = self.embeddings_tensors[
                self.embedding_indices_gpu[:, 0], self.embedding_indices_gpu[:, 1]
            ]

            # Update hidden states
            hidden_states[valid_mask] = gathered_embeddings
        return hidden_states
