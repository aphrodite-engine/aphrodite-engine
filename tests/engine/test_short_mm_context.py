import pytest

from ..conftest import IMAGE_ASSETS

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "USER: <image>\nWhat's the content of the image?\nASSISTANT:",
    "cherry_blossom":
    "USER: <image>\nWhat is the season?\nASSISTANT:",
})

models = ["llava-hf/llava-1.5-7b-hf"]


@pytest.mark.parametrize("model", models)
def test_context_length_too_short(aphrodite_runner, image_assets, model):
    images = [asset.pil_image for asset in image_assets]

    with pytest.raises(ValueError,
                       match="longer than the maximum model length"):
        aphrodite_model = aphrodite_runner(
            model,
            max_model_len=128,  # LLaVA has a feature size of 576
            enforce_eager=True,
        )

        with aphrodite_model:
            aphrodite_model.generate_greedy([HF_IMAGE_PROMPTS[0]],
                                       max_tokens=1,
                                       images=[images[0]])
