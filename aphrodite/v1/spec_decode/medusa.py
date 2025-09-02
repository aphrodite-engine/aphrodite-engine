import torch
import torch.nn as nn

from aphrodite.config import AphroditeConfig
from aphrodite.forward_context import set_forward_context
from aphrodite.modeling.model_loader import get_model
from aphrodite.v1.sample.metadata import SamplingMetadata


class MedusaProposer:
    """
    Medusa proposer class for generating token sequences
    """

    def __init__(
        self,
        aphrodite_config: AphroditeConfig,
        device: torch.device,
    ):
        # Save config parameters
        self.aphrodite_config = aphrodite_config
        self.device = device
        self.max_num_tokens = (
            aphrodite_config.scheduler_config.max_num_batched_tokens)
        self.hidden_size = aphrodite_config.speculative_config.\
            draft_model_config.get_hidden_size(
        )
        self.dtype = aphrodite_config.model_config.dtype

    def propose(
        self,
        target_hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> list[list[int]]:
        # Generate blocks and compute logits
        blocks = self.model(target_hidden_states)
        logits = self.model.compute_logits(blocks, None)

        # Get draft tokens and transpose the result
        draft_tokens = [logit.argmax(dim=-1).tolist() for logit in logits]
        return [list(row) for row in zip(*draft_tokens)]

    def load_model(self, target_model: nn.Module) -> None:
        from aphrodite.compilation.backends import set_model_tag
        with set_model_tag("medusa_head"):
            self.model = get_model(aphrodite_config=self.aphrodite_config,
                                   model_config=self.aphrodite_config.
                                   speculative_config.draft_model_config)

    @torch.inference_mode()
    def dummy_run(self, num_tokens: int) -> None:
        hidden_states = torch.zeros((self.max_num_tokens, self.hidden_size),
                                    dtype=self.dtype,
                                    device=self.device)
        with set_forward_context(None, self.aphrodite_config,
                                 num_tokens=num_tokens):
            self.model(hidden_states)
