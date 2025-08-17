import pytest
import torch
import torch.nn.functional as F

from tests.models.utils import softmax
from aphrodite.config import PoolerConfig


@pytest.mark.parametrize(
    "model",
    [
        "jason9693/Qwen2.5-1.5B-apeach",
        "papluca/xlm-roberta-base-language-detection"
    ],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_classify_models_using_activation(
    hf_runner,
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:

    with aphrodite_runner(model,
                     max_model_len=512,
                     dtype=dtype,
                     override_pooler_config=PoolerConfig(
                         activation=False)) as aphrodite_model:
        wo_activation_out = aphrodite_model.classify(example_prompts)

    with aphrodite_runner(model,
                     max_model_len=512,
                     dtype=dtype,
                     override_pooler_config=PoolerConfig(
                         activation=True)) as aphrodite_model:
        w_activation_out = aphrodite_model.classify(example_prompts)

    for wo_activation, w_activation in zip(wo_activation_out,
                                           w_activation_out):
        wo_activation = torch.tensor(wo_activation)
        w_activation = torch.tensor(w_activation)

        assert not torch.allclose(
            wo_activation, w_activation,
            atol=1e-2), "override_pooler_config is not working"
        assert torch.allclose(softmax(wo_activation), w_activation,
                              1e-3 if dtype == "float" else 1e-2)


@pytest.mark.parametrize(
    "model",
    [
        "intfloat/multilingual-e5-small",
    ],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_embed_models_using_normalize(
    hf_runner,
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:

    with aphrodite_runner(model,
                     max_model_len=512,
                     dtype=dtype,
                     override_pooler_config=PoolerConfig(
                         normalize=False)) as aphrodite_model:
        wo_normalize = torch.tensor(aphrodite_model.embed(example_prompts))

    with aphrodite_runner(
            model,
            max_model_len=512,
            dtype=dtype,
            override_pooler_config=PoolerConfig(normalize=True)) as aphrodite_model:
        w_normalize = torch.tensor(aphrodite_model.embed(example_prompts))

    assert not torch.allclose(
        wo_normalize, w_normalize,
        atol=1e-2), "override_pooler_config normalize is not working"
    assert torch.allclose(
        F.normalize(wo_normalize, p=2, dim=-1), w_normalize,
        atol=1e-2), "w_normal should be close to normal(wo_normal)."


@pytest.mark.parametrize(
    "model",
    [
        "internlm/internlm2-1_8b-reward",
    ],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_reward_models_using_softmax(
    hf_runner,
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:

    with aphrodite_runner(
            model,
            max_model_len=1024,
            dtype=dtype,
            override_pooler_config=PoolerConfig(softmax=False)) as aphrodite_model:
        wo_softmax = aphrodite_model.encode(example_prompts)

    with aphrodite_runner(
            model,
            max_model_len=1024,
            dtype=dtype,
            override_pooler_config=PoolerConfig(softmax=True)) as aphrodite_model:
        w_softmax = aphrodite_model.encode(example_prompts)

    for wo, w in zip(wo_softmax, w_softmax):
        wo = torch.tensor(wo)
        w = torch.tensor(w)

        assert not torch.allclose(
            wo, w, atol=1e-2), "override_pooler_config softmax is not working"
        assert torch.allclose(
            softmax(wo), w,
            atol=1e-2), "w_softmax should be close to softmax(wo_softmax)."
