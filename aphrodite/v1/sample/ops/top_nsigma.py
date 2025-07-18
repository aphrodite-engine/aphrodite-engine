import torch

from aphrodite.v1.sample.metadata import SamplingMetadata


def top_nsigma(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Apply top-nsigma truncation to the logits.

    Reference: https://arxiv.org/abs/2411.07641

    Args:
        logits: Logits of shape (num_tokens, vocab_size)
        nsigma: Number of standard deviations to use as threshold
    Returns:
        Modified logits with values below threshold set to -inf
    """
    nsigma = sampling_metadata.top_nsigma
    if nsigma is None:
        return logits

    std = logits.std(dim=-1, keepdim=True)
    threshold = (logits.max(dim=-1, keepdim=True).values -
                 nsigma.unsqueeze(dim=1) * std)
    logits[logits < threshold] = float("-inf")

    return logits
