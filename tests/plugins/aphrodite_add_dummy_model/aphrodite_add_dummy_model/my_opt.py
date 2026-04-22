import torch

from aphrodite.modeling.models.opt import OPTForCausalLM


class MyOPTForCausalLM(OPTForCausalLM):
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        # this dummy model always predicts the first token
        logits = super().compute_logits(hidden_states)
        if logits is not None:
            logits.zero_()
            logits[:, 0] += 1.0
        return logits
