from typing import Optional

import pytest

from aphrodite import SamplingParams
from aphrodite.assets.audio import AudioAsset

from ....conftest import AphroditeRunner
from ....utils import create_new_process_for_each_test, multi_gpu_test

PROMPTS = [
    {
        "prompt":
        "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        "multi_modal_data": {
            "audio": AudioAsset("mary_had_lamb").audio_and_sample_rate,
        },
    },
    {  # Test explicit encoder/decoder prompt
        "encoder_prompt": {
            "prompt": "",
            "multi_modal_data": {
                "audio": AudioAsset("winning_call").audio_and_sample_rate,
            },
        },
        "decoder_prompt":
        "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
    }
]

EXPECTED = {
    "openai/whisper-tiny": [
        " He has birth words I spoke in the original corner of that. And a"
        " little piece of black coat poetry. Mary had a little sandwich,"
        " sweet, with white and snow. And everyone had it very went the last"
        " would sure to go.",
        " >> And the old one, fit John the way to Edgar Martinez. >> One more"
        " to line down the field line for our base camp. Here comes joy. Here"
        " is June and the third base. They're going to wave him in. The throw"
        " to the plate will be late. The Mariners are going to play for the"
        " American League Championship. I don't believe it. It just continues"
        " by all five."
    ],
    "openai/whisper-small": [
        " The first words I spoke in the original pornograph. A little piece"
        " of practical poetry. Mary had a little lamb, its fleece was quite a"
        " slow, and everywhere that Mary went the lamb was sure to go.",
        " And the old one pitch on the way to Edgar Martinez one month. Here"
        " comes joy. Here is Junior to third base. They're gonna wave him"
        " in. The throw to the plate will be late. The Mariners are going to"
        " play for the American League Championship. I don't believe it. It"
        " just continues. My, oh my."
    ],
    "openai/whisper-medium": [
        " The first words I spoke in the original phonograph, a little piece"
        " of practical poetry. Mary had a little lamb, its fleece was quite as"
        " slow, and everywhere that Mary went the lamb was sure to go.",
        " And the 0-1 pitch on the way to Edgar Martinez swung on the line"
        " down the left field line for Obeyshev. Here comes Joy. Here is"
        " Jorgen at third base. They're going to wave him in. The throw to the"
        " plate will be late. The Mariners are going to play for the American"
        " League Championship. I don't believe it. It just continues. My, oh"
        " my."
    ],
    "openai/whisper-large-v3": [
        " The first words I spoke in the original phonograph, a little piece"
        " of practical poetry. Mary had a little lamb, its feet were quite as"
        " slow, and everywhere that Mary went, the lamb was sure to go.",
        " And the 0-1 pitch on the way to Edgar Martinez. Swung on the line."
        " Now the left field line for a base hit. Here comes Joy. Here is"
        " Junior to third base. They're going to wave him in. The throw to the"
        " plate will be late. The Mariners are going to play for the American"
        " League Championship. I don't believe it. It just continues. My, oh,"
        " my."
    ],
    "openai/whisper-large-v3-turbo": [
        " The first words I spoke in the original phonograph, a little piece"
        " of practical poetry. Mary had a little lamb, its streets were quite"
        " as slow, and everywhere that Mary went the lamb was sure to go.",
        " And the 0-1 pitch on the way to Edgar Martinez. Swung on the line"
        " down the left field line for a base hit. Here comes Joy. Here is"
        " Junior to third base. They're going to wave him in. The throw to the"
        " plate will be late. The Mariners are going to play for the American"
        " League Championship. I don't believe it. It just continues. My, oh,"
        " my."
    ]
}


def run_test(
    aphrodite_runner: type[AphroditeRunner],
    model: str,
    *,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
) -> None:
    prompt_list = PROMPTS * 10
    expected_list = EXPECTED[model] * 10

    with aphrodite_runner(
            model,
            max_model_len=448,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
    ) as aphrodite_model:
        llm = aphrodite_model.model

        sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=200,
        )

        outputs = llm.generate(prompt_list, sampling_params)

    for output, expected in zip(outputs, expected_list):
        print(output.outputs[0].text)
        assert output.outputs[0].text == expected


@pytest.mark.core_model
@pytest.mark.parametrize(
    "model", ["openai/whisper-small", "openai/whisper-large-v3-turbo"])
@create_new_process_for_each_test()
def test_models(aphrodite_runner, model) -> None:
    run_test(
        aphrodite_runner,
        model,
        tensor_parallel_size=1,
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.core_model
@pytest.mark.parametrize("model", ["openai/whisper-large-v3-turbo"])
@pytest.mark.parametrize("distributed_executor_backend", ["ray", "mp"])
@create_new_process_for_each_test()
def test_models_distributed(
    aphrodite_runner,
    model,
    distributed_executor_backend,
) -> None:
    run_test(
        aphrodite_runner,
        model,
        tensor_parallel_size=2,
        distributed_executor_backend=distributed_executor_backend,
    )
