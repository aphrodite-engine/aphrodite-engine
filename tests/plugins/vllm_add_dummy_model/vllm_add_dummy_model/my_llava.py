import torch

from aphrodite.modeling.models.llava import (LlavaDummyInputsBuilder,
                                             LlavaForConditionalGeneration,
                                             LlavaMultiModalProcessor,
                                             LlavaProcessingInfo)
from aphrodite.multimodal import MULTIMODAL_REGISTRY


@MULTIMODAL_REGISTRY.register_processor(
    LlavaMultiModalProcessor,
    info=LlavaProcessingInfo,
    dummy_inputs=LlavaDummyInputsBuilder,
)
class MyLlava(LlavaForConditionalGeneration):
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        # this dummy model always predicts the first token
        logits = super().compute_logits(hidden_states)
        if logits is not None:
            logits.zero_()
            logits[:, 0] += 1.0
        return logits
