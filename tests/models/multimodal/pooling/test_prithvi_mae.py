import pytest
import torch

from aphrodite.utils import set_default_torch_num_threads

from ....conftest import AphroditeRunner


def generate_test_mm_data():
    mm_data = {
        "pixel_values": torch.full((6, 512, 512), 1.0, dtype=torch.float16),
        "location_coords": torch.full((1, 2), 1.0, dtype=torch.float16),
    }
    return mm_data


def _run_test(
    aphrodite_runner: type[AphroditeRunner],
    model: str,
) -> None:

    prompt = [
        {
            # This model deals with no text input
            "prompt_token_ids": [1],
            "multi_modal_data": generate_test_mm_data(),
        } for _ in range(10)
    ]

    with (
            set_default_torch_num_threads(1),
            aphrodite_runner(
                model,
                runner="pooling",
                dtype=torch.float16,
                enforce_eager=True,
                skip_tokenizer_init=True,
                # Limit the maximum number of sequences to avoid the
                # test going OOM during the warmup run
                max_num_seqs=32,
            ) as aphrodite_model,
    ):
        aphrodite_model.encode(prompt)


MODELS = ["christian-pinto/Prithvi-EO-2.0-300M-TL-APHRODITE"]


@pytest.mark.core_model
@pytest.mark.parametrize("model", MODELS)
def test_models_image(
    hf_runner,
    aphrodite_runner,
    image_assets,
    model: str,
) -> None:
    _run_test(
        aphrodite_runner,
        model,
    )
