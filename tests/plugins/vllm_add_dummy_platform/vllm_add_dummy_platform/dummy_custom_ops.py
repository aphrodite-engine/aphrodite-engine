import torch

from aphrodite.modeling.layers.rotary_embedding import RotaryEmbedding


# Register CustomRotaryEmbedding to CustomOP.
@RotaryEmbedding.register_oot
class DummyRotaryEmbedding(RotaryEmbedding):
    """Original rotary positional embedding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.addition_config = True

    def forward_oot(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward_oot(*args, **kwargs)
