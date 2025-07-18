import torch

_SAMPLING_EPS = 1e-5


def apply_all_temperatures(
    logits: torch.Tensor,
    sampling_metadata,
) -> torch.Tensor:
    temp = sampling_metadata.temperature
    dynatemp_mins = sampling_metadata.dynatemp_min or torch.zeros_like(temp)
    dynatemp_maxs = sampling_metadata.dynatemp_max or torch.zeros_like(temp)
    dynatemp_exps = sampling_metadata.dynatemp_exp or torch.zeros_like(temp)

    dynatemp_mask = (dynatemp_mins != 0) | (dynatemp_maxs != 0)
    dynatemp_mins = dynatemp_mins[dynatemp_mask]
    dynatemp_maxs = dynatemp_maxs[dynatemp_mask]
    dynatemp_exps = dynatemp_exps[dynatemp_mask]

    dynatemp_logits = logits[dynatemp_mask]
    dynatemp_shifted_logits = torch.log_softmax(dynatemp_logits, dim=-1)
    dynatemp_probs = dynatemp_shifted_logits.exp()
    dynatemp_entropies = -(dynatemp_probs *
                           dynatemp_shifted_logits).nansum(dim=-1)
    dynatemp_max_entropies = torch.log_((dynatemp_logits
                                         > float("-inf")).sum(dim=-1).float())
    normalized_entropies = dynatemp_entropies.div_(dynatemp_max_entropies)
    dyn_temp = (dynatemp_mins + (dynatemp_maxs - dynatemp_mins) *
                normalized_entropies.pow_(dynatemp_exps))
    temp[dynatemp_mask] = dyn_temp

    temp[temp.isnan()] = _SAMPLING_EPS
    temp[temp <= _SAMPLING_EPS] = _SAMPLING_EPS
    # Use in-place division to avoid creating a new tensor.
    return logits.div_(temp.unsqueeze(dim=1))
