import os

import pytest

from aphrodite.modeling.layers.pooler import CLSPool, PoolingType
from aphrodite.modeling.models.bert import BertEmbeddingModel
from aphrodite.modeling.models.roberta import RobertaEmbeddingModel
from aphrodite.platforms import current_platform

MAX_MODEL_LEN = 128
MODEL_NAME = os.environ.get("MODEL_NAME", "BAAI/bge-base-en-v1.5")
REVISION = os.environ.get("REVISION", "main")

MODEL_NAME_ROBERTA = os.environ.get("MODEL_NAME",
                                    "intfloat/multilingual-e5-small")
REVISION_ROBERTA = os.environ.get("REVISION", "main")


@pytest.mark.skipif(current_platform.is_rocm(),
                    reason="Xformers backend is not supported on ROCm.")
def test_model_loading_with_params(aphrodite_runner):
    """
    Test parameter weight loading with tp>1.
    """
    with aphrodite_runner(model_name=MODEL_NAME,
                     revision=REVISION,
                     dtype="float16",
                     max_model_len=MAX_MODEL_LEN) as aphrodite_model:
        output = aphrodite_model.encode("Write a short story about a robot that"
                                   " dreams for the first time.\n")

        model_config = aphrodite_model.model.llm_engine.model_config
        model_tokenizer = aphrodite_model.model.llm_engine.tokenizer

        # asserts on the bert model config file
        assert model_config.encoder_config["max_seq_length"] == 512
        assert model_config.encoder_config["do_lower_case"]

        # asserts on the pooling config files
        assert model_config.pooler_config.pooling_type == PoolingType.CLS.name
        assert model_config.pooler_config.pooling_norm

        # asserts on the tokenizer loaded
        assert model_tokenizer.tokenizer_id == "BAAI/bge-base-en-v1.5"
        assert model_tokenizer.tokenizer_config["do_lower_case"]
        assert model_tokenizer.tokenizer.model_max_length == 512

        def check_model(model):
            assert isinstance(model, BertEmbeddingModel)
            assert model._pooler.pooling_type == PoolingType.CLS
            assert model._pooler.normalize

        aphrodite_model.apply_model(check_model)

        # assert output
        assert output


@pytest.mark.skipif(current_platform.is_rocm(),
                    reason="Xformers backend is not supported on ROCm.")
def test_roberta_model_loading_with_params(aphrodite_runner):
    """
    Test parameter weight loading with tp>1.
    """
    with aphrodite_runner(model_name=MODEL_NAME_ROBERTA,
                     revision=REVISION_ROBERTA,
                     dtype="float16",
                     max_model_len=MAX_MODEL_LEN) as aphrodite_model:
        output = aphrodite_model.encode("Write a short story about a robot that"
                                   " dreams for the first time.\n")

        model_config = aphrodite_model.model.llm_engine.model_config
        model_tokenizer = aphrodite_model.model.llm_engine.tokenizer

        # asserts on the bert model config file
        assert model_config.encoder_config["max_seq_length"] == 512
        assert not model_config.encoder_config["do_lower_case"]

        # asserts on the pooling config files
        assert model_config.pooler_config.pooling_type == PoolingType.MEAN.name
        assert model_config.pooler_config.pooling_norm

        # asserts on the tokenizer loaded
        assert model_tokenizer.tokenizer_id == "intfloat/multilingual-e5-small"
        assert not model_tokenizer.tokenizer_config["do_lower_case"]

        def check_model(model):
            assert isinstance(model, RobertaEmbeddingModel)
            assert model._pooler.pooling_type == PoolingType.MEAN
            assert model._pooler.normalize

        aphrodite_model.apply_model(check_model)

        # assert output
        assert output


@pytest.mark.skipif(current_platform.is_rocm(),
                    reason="Xformers backend is not supported on ROCm.")
def test_facebook_roberta_model_loading_with_params(aphrodite_runner):
    """
    Test loading roberta-base model with no lm_head.
    """
    model_name = "FacebookAI/roberta-base"
    with aphrodite_runner(model_name=model_name,
                     dtype="float16",
                     max_model_len=MAX_MODEL_LEN) as aphrodite_model:
        output = aphrodite_model.encode("Write a short story about a robot that"
                                   " dreams for the first time.\n")

        model_tokenizer = aphrodite_model.model.llm_engine.tokenizer
        assert model_tokenizer.tokenizer_id == model_name

        def check_model(model):
            assert isinstance(model, RobertaEmbeddingModel)
            assert not hasattr(model, "lm_head")
            assert isinstance(model._pooler, CLSPool)

        aphrodite_model.apply_model(check_model)

        assert output
