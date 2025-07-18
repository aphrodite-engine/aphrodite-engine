import random
from typing import Optional

import pytest

from aphrodite import LLM, SamplingParams

MODEL = "facebook/opt-125m"
DTYPE = "half"


def _aphrodite_model(apc: bool, aphrodite_runner, monkeypatch):
    """Set up AphroditeRunner instance."""
    monkeypatch.setenv("APHRODITE_USE_V1", "1")
    return aphrodite_runner(
        MODEL,
        dtype=DTYPE,
        max_model_len=128,
        enforce_eager=True,
        enable_prefix_caching=apc,
        gpu_memory_utilization=0.5,
    )


@pytest.fixture(
    # Function scope decouples tests & allows
    # env var adjustment via monkeypatch
    scope="function",
    # Prefix caching
    params=[False, True])
def aphrodite_model(aphrodite_runner, request, monkeypatch):
    """AphroditeRunner test fixture parameterized by APC True/False."""
    with _aphrodite_model(request.param, aphrodite_runner,
                          monkeypatch) as aphrodite_model:
        yield aphrodite_model


@pytest.fixture(scope="function")
def aphrodite_model_apc(aphrodite_runner, monkeypatch):
    """AphroditeRunner test fixture with APC."""
    with _aphrodite_model(True, aphrodite_runner,
                          monkeypatch) as aphrodite_model:
        yield aphrodite_model


def _get_test_sampling_params(
    prompt_list: list[str],
    seed: Optional[int] = 42,
) -> tuple[list[SamplingParams], list[int]]:
    """Generate random sampling params for a batch."""

    def get_mostly_n_gt1() -> int:
        r"""Mostly n \in [2,20], ~1/3 n=1"""
        x = random.randint(0, 28)
        if x < 10:
            return 1
        else:
            return x - 8

    n_list = [get_mostly_n_gt1() for _ in range(len(prompt_list))]
    # High temperature to maximize the chance of unique completions
    return [
        SamplingParams(temperature=0.95, top_p=0.95, n=n, seed=seed)
        for n in n_list
    ], n_list


def test_parallel_sampling(aphrodite_model, example_prompts) -> None:
    """Test passes if parallel sampling `n>1` yields `n` unique completions.
    
    Args:
      aphrodite_model: AphroditeRunner instance under test.
      example_prompt: test fixture providing prompts for testing.
    """
    sampling_params_list, n_list = _get_test_sampling_params(example_prompts)
    model: LLM = aphrodite_model.model
    outputs = model.generate(example_prompts, sampling_params_list)

    # Validate each request response
    for out, n in zip(outputs, n_list):
        completion_counts: dict[str, int] = {}
        # Assert correct number of completions
        assert len(out.outputs) == n, (
            f"{len(out.outputs)} completions; {n} expected.")
        for idx in range(n):
            comp = out.outputs[idx]
            # Assert correct completion indices
            assert comp.index == idx, (f"Index {comp.index}; expected {idx}.")
            text = comp.text
            completion_counts[text] = completion_counts.get(text, 0) + 1
        # Assert unique completions
        if len(completion_counts) != n:
            repeats = {
                txt: num
                for (txt, num) in completion_counts.items() if num > 1
            }
            raise AssertionError(
                f"{len(completion_counts)} unique completions; expected"
                f" {n}. Repeats: {repeats}")
