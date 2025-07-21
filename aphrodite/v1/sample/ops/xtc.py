import torch
from aphrodite.v1.sample.metadata import SamplingMetadata


def xtc(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Apply Exclude Top Choices (XTC) sampling to the logits.
    Reference: https://github.com/oobabooga/text-generation-webui/pull/6335

    Args:
        logits: (num_tokens, vocab_size) The input logits.
        xtc_thresholds: (num_tokens,) The threshold for each token.
        xtc_probabilities: (num_tokens,) The probability of applying XTC
            for each token.

    Returns:
        torch.Tensor: The modified logits.
    """
    xtc_threshold = sampling_metadata.xtc_threshold
    xtc_probability = sampling_metadata.xtc_probability
    if xtc_threshold is None or xtc_probability is None:
        return logits

    # Ensure xtc_probability is a tensor with the right shape
    if not isinstance(xtc_probability, torch.Tensor):
        xtc_probability = torch.tensor(xtc_probability, device=logits.device,
                                       dtype=torch.float32)

    # Ensure the tensor has the right shape for broadcasting
    if xtc_probability.dim() == 0:
        xtc_probability = xtc_probability.unsqueeze(0)

    # Create random tensor with the same shape as xtc_probability
    random_tensor = torch.rand_like(xtc_probability)
    apply_xtc = random_tensor < xtc_probability

    # Use .any() with explicit dimension handling
    if not apply_xtc.any().item():
        return logits

    probs = torch.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # Find indices where the next probability is above the threshold
    # Skips the top choice, which later on becomes skipping the last choice.
    above_threshold = sorted_probs[..., 1:] >= xtc_threshold.unsqueeze(-1)

    # Apply XTC only to rows where it should be applied
    for i in range(logits.shape[0]):
        if apply_xtc[i].item():
            # Count logits above the threshold (skipping the first)
            indices_to_remove = above_threshold[i].count_nonzero(dim=-1).item()
            if indices_to_remove > 0:
                # Implies the top logit and at least one other is >= threshold.
                # Mask out above_thresh logits except the last/lowest one.
                logits[i].scatter_(
                    0, sorted_indices[i, :indices_to_remove], -float('inf'))

    return logits
