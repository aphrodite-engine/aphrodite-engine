from typing import List, Optional, Type

import pytest
from transformers import AutoModelForVision2Seq, BatchEncoding

from aphrodite.common.utils import STR_DTYPE_TO_TORCH_DTYPE
from aphrodite.multimodal.utils import rescale_image_size

from ....conftest import IMAGE_ASSETS, AphroditeRunner, HfRunner, _ImageAssets
from ...utils import check_outputs_equal

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "USER: <image>\nWhat's the content of the image?\nASSISTANT:",
    "cherry_blossom":
    "USER: <image>\nWhat is the season?\nASSISTANT:",
})

models = ["facebook/chameleon-7b"]


def run_test(
    hf_runner: Type[HfRunner],
    aphrodite_runner: Type[AphroditeRunner],
    image_assets: _ImageAssets,
    model: str,
    *,
    size_factors: List[float],
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    """Inference result should be the same between hf and aphrodite.

    All the image fixtures for the test are from IMAGE_ASSETS.
    For huggingface runner, we provide the PIL images as input.
    For aphrodite runner, we provide MultiModalDataDict objects
    and corresponding vision language config as input.
    Note, the text input is also adjusted to abide by aphrodite contract.
    The text output is sanitized to be able to compare with hf.
    """
    torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]
    images = [asset.pil_image for asset in image_assets]

    inputs_per_image = [(
        [prompt for _ in size_factors],
        [rescale_image_size(image, factor) for factor in size_factors],
    ) for image, prompt in zip(images, HF_IMAGE_PROMPTS)]

    with aphrodite_runner(model,
                     max_model_len=4096,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as aphrodite_model:

        aphrodite_outputs_per_image = [
            aphrodite_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images)
            for prompts, images in inputs_per_image
        ]

    def process(hf_inputs: BatchEncoding):
        hf_inputs["pixel_values"] = hf_inputs["pixel_values"] \
            .to(torch_dtype)  # type: ignore
        return hf_inputs

    with hf_runner(model,
                   dtype=dtype,
                   postprocess_inputs=process,
                   auto_cls=AutoModelForVision2Seq) as hf_model:
        hf_outputs_per_image = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=images)
            for prompts, images in inputs_per_image
        ]

    for hf_outputs, aphrodite_outputs in zip(hf_outputs_per_image,
                                        aphrodite_outputs_per_image):
        # HF Logprobs include image tokens, unlike Aphrodite, so we don't
        # directly compare them
        check_outputs_equal(
            outputs_0_lst=[outputs[:2] for outputs in hf_outputs],
            outputs_1_lst=[outputs[:2] for outputs in aphrodite_outputs],
            name_0="hf",
            name_1="aphrodite",
        )


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "size_factors",
    [
        # No image
        [],
        # Single-scale
        [1.0],
        # Single-scale, batched
        [1.0, 1.0, 1.0],
        # Multi-scale
        [0.25, 0.5, 1.0],
    ],
)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [8])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(hf_runner, aphrodite_runner, image_assets, model, size_factors,
                dtype, max_tokens, num_logprobs) -> None:
    run_test(
        hf_runner,
        aphrodite_runner,
        image_assets,
        model,
        size_factors=size_factors,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )
