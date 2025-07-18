import torch
from aphrodite.v1.sample.metadata import SamplingMetadata


def skew(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """
    Apply skew sampling to the logits.
    
    Reference: https://github.com/turboderp/exllamav2/commit/1de4cdd70b09208e7b4f17ee322c190e16f60efd
    """
    skews = sampling_metadata.skew
    if skews is None:
        return logits
        
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)
    
    # Apply skew transformation
    cum_probs = torch.cumsum(probs, dim=-1)
    cum_probs = torch.pow(cum_probs, torch.exp(skews).unsqueeze(dim=1))
    probs = torch.diff(cum_probs, dim=-1,
                       prepend=torch.zeros_like(cum_probs[..., :1]))
    
    # Convert back to logits
    logits = torch.log(probs)
    
    return logits
