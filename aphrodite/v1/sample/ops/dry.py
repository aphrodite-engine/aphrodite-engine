import torch


def apply_all_dry(
    logits: torch.Tensor,
    input_token_ids: torch.Tensor,
    output_token_ids: torch.Tensor,
    multipliers: torch.Tensor,
    bases: torch.Tensor,
    allowed_lengths: torch.Tensor,
    sequence_breakers_ids: torch.Tensor,
    ranges: torch.Tensor,
    max_ngram: torch.Tensor,
    max_occurrences: torch.Tensor,
    early_exit_match_len: torch.Tensor,
) -> torch.Tensor:
    """
    Apply Don't Repeat Yourself (DRY) sampling to the logits.

    Reference: https://github.com/oobabooga/text-generation-webui/pull/5677 and
    https://github.com/AlpinDale/vllm/pull/1
    """
    VOCAB_SIZE = logits.size(-1)

    applies_to = multipliers.nonzero(as_tuple=True)[0]
    for irow in applies_to.tolist():
        prompt_len = len(input_token_ids[irow]) - (input_token_ids[irow]
                                                   == VOCAB_SIZE).sum().item()
        output_len = len(output_token_ids[irow]) - (
            output_token_ids[irow] == VOCAB_SIZE).sum().item()

        token_seq = torch.cat((input_token_ids[irow][:prompt_len],
                               output_token_ids[irow][:output_len]),
                              dim=0)

        range_limit = ranges[irow].item()
        if range_limit > 0:
            token_seq = token_seq[-range_limit:]

        if token_seq.size(0) < 2:
            continue

        last_token = token_seq[-1].item()
        if last_token in sequence_breakers_ids[irow]:
            continue

        break_mask = torch.zeros(len(token_seq),
                                 dtype=torch.bool,
                                 device=logits.device)
        for break_tok in sequence_breakers_ids[irow]:
            break_mask.logical_or_(token_seq == break_tok)

        curr_max_ngram = 0
        max_ngram_val = max_ngram[irow].item()
        for curr_max_ngram in range(
                min(len(break_mask),
                    int(max_ngram_val) + 1)):  # noqa: E501
            if break_mask[-curr_max_ngram - 1]:
                break

        min_ngram = allowed_lengths[irow].item()
        if curr_max_ngram <= min_ngram:
            continue

        ngram_lens = torch.zeros(VOCAB_SIZE,
                                 dtype=torch.int32,
                                 device=logits.device)

        endpoint_indexes_all = torch.nonzero(token_seq == last_token,
                                             as_tuple=True)[0].tolist()
        if len(endpoint_indexes_all) < 2:
            continue
        endpoint_indexes = endpoint_indexes_all[:-1]

        max_occurrences_val = max_occurrences[irow].item()
        if len(endpoint_indexes) > max_occurrences_val:
            endpoint_indexes = endpoint_indexes[-max_occurrences_val:]

        early_exit_match_len_val = early_exit_match_len[irow].item()
        for idx in reversed(endpoint_indexes):
            if idx == len(token_seq) - 1:
                continue

            match_len = 0
            for unwind in range(1, min(idx, curr_max_ngram) + 1):
                if break_mask[idx - unwind]:
                    break
                if token_seq[idx - unwind] != token_seq[-unwind - 1]:
                    break
                match_len = unwind

            if match_len > 0:
                next_tok = token_seq[idx + 1]
                new_len = match_len + 1

                ngram_lens[next_tok] = max(ngram_lens[next_tok].item(),
                                           new_len)

                if new_len >= early_exit_match_len_val:
                    break

        penalty_mask = ngram_lens > 0
        if penalty_mask.any():
            scales = bases[irow]**(ngram_lens[penalty_mask] - min_ngram)
            logits[irow][penalty_mask] -= multipliers[irow] * scales

    return logits
