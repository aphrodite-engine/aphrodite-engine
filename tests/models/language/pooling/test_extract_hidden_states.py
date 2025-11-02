import pytest
import torch

from aphrodite import TokensPrompt


@pytest.mark.parametrize(
    "model",
    ["Qwen/Qwen3-0.6B"],
)
@torch.inference_mode
def test_embed_models(hf_runner, aphrodite_runner, model: str):
    n_prompt_tokens = [55, 56, 57]
    token_prompts = [[1024 + i for i in range(n)] for n in n_prompt_tokens]

    with aphrodite_runner(
        model,
        max_model_len=128,
        enforce_eager=True,
        runner="pooling",
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
    ) as aphrodite_model:
        pooling_outputs = aphrodite_model.llm.encode(
            [TokensPrompt(prompt_token_ids=t) for t in token_prompts],
            pooling_task="token_embed",
        )

        for n, output in zip(n_prompt_tokens, pooling_outputs):
            assert len(output.prompt_token_ids) == n
            assert output.num_cached_tokens == 0
