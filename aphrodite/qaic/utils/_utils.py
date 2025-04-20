# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from typing import List, Optional, Tuple, Union

import requests
from huggingface_hub import login, snapshot_download
from loguru import logger
from requests.exceptions import HTTPError
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from aphrodite.qaic.utils.constants import QEFF_MODELS_DIR


def login_and_download_hf_lm(model_name, *args, **kwargs):
    logger.info(f"loading HuggingFace model for {model_name}")
    hf_token = kwargs.pop("hf_token", None)
    cache_dir = kwargs.pop("cache_dir", None)
    if hf_token is not None:
        login(hf_token)
    model_path = hf_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        ignore_patterns=[
            "*.txt",
            "*.onnx",
            "*.ot",
            "*.md",
            "*.tflite",
            "*.pdf",
            "*.msgpack",
            "*.h5",
        ],
    )
    return model_path


def hf_download(
    repo_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
):
    # Setup cache_dir
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    max_retries = 5
    retry_count = 0
    while retry_count < max_retries:
        try:
            model_path = snapshot_download(
                repo_id,
                cache_dir=cache_dir,
                revision="main",
                resume_download=True,
                token=hf_token,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
            break
        except requests.ReadTimeout as e:
            logger.info(f"Read timeout: {e}")
            retry_count += 1

        except HTTPError as e:
            retry_count = max_retries
            if e.response.status_code == 401:
                logger.info(
                    "You need to pass a valid `--hf_token=...` to download "
                    "private checkpoints."
                )
            else:
                raise e

    return model_path


def qpc_exists(model_name: str, qpc_base_dir_name: str) -> Tuple[bool, str]:
    """
    Checks if qpc dir exists.
    Returns
    1. Boolean variable indicating if qpc files exist
    2. Path of the qpc dir if found.
    ---------
    :param model_name: str. HF Model card name.
    :param dir_path: str. Path of qpc directory.
    :return: Union[Tuple[bool, str]]: qpc_exists and path to qpc directory
    """
    model_card_dir = os.path.join(QEFF_MODELS_DIR, str(model_name))
    os.makedirs(model_card_dir, exist_ok=True)

    qpc_dir_path = os.path.join(model_card_dir, qpc_base_dir_name, "qpcs")

    # Compute the boolean indicating if the QPC exists
    qpc_exists_bool = os.path.isdir(qpc_dir_path) and os.path.isfile(
        os.path.join(qpc_dir_path, "programqpc.bin")
    )

    return qpc_exists_bool, qpc_dir_path


def get_onnx_dir_name(model_name, has_fbs):
    # Create a unique directory name for the ONNX model
    # Clearly indicate whether it's with or without FBS
    # Replace all hyphens with underscores
    model_name_safe = model_name.replace("/", "_").replace("-", "_")
    if has_fbs:
        return f"onnx_{model_name_safe}_with_fbs"
    else:
        return f"onnx_{model_name_safe}_without_fbs"


def onnx_exists(model_name: str, full_batch_size: int) -> Tuple[bool, str, str]:
    """
    Checks if qpc files already exists, removes the directory if files have
    been manipulated.
    ---------

    :model_name: `str`- HF Model card name.

    Return:
        onnx_exists and path to onnx file and directory
    """
    model_card_dir = os.path.join(QEFF_MODELS_DIR, str(model_name))
    os.makedirs(model_card_dir, exist_ok=True)

    # Determine if we're using full_batch_size
    has_fbs = full_batch_size is not None

    # ONNX handling
    onnx_dir_name = get_onnx_dir_name(model_name, has_fbs)
    onnx_dir_path = os.path.join(model_card_dir, onnx_dir_name)
    os.makedirs(onnx_dir_path, exist_ok=True)
    clipped_onnx_model_path = os.path.join(
        onnx_dir_path, model_name.replace("/", "_") + "_kv_clipped_fp16.onnx"
    )
    unclipped_onnx_model_path = clipped_onnx_model_path.replace(
        "_clipped_fp16.onnx", ".onnx"
    )

    # Compute the boolean indicating if the ONNX model exists
    onnx_exists_bool = False
    onnx_model_path = None
    if os.path.isfile(os.path.join(onnx_dir_path, "custom_io_fp16.yaml")):
        if os.path.isfile(clipped_onnx_model_path):
            onnx_exists_bool = True
            onnx_model_path = clipped_onnx_model_path
        elif os.path.isfile(unclipped_onnx_model_path):
            onnx_exists_bool = True
            onnx_model_path = unclipped_onnx_model_path

    # Return the boolean, onnx_dir_path, and onnx_model_path
    return onnx_exists_bool, onnx_dir_path, onnx_model_path


def load_hf_tokenizer(
    pretrained_model_name_or_path: str,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    padding_side: str = "right",
    **kwargs,
) -> Union[PreTrainedTokenizerFast, PreTrainedTokenizer]:
    # FIXME: Fix kwargs to take token, cache_dir and pass via kwargs only on
    # line 129
    logger.info("Loading Tokenizer")
    if hf_token is not None:
        login(hf_token)
    # Download tokenizer along with model if it doesn't exist
    model_hf_path = (
        pretrained_model_name_or_path
        if os.path.isdir(pretrained_model_name_or_path)
        else hf_download(
            repo_id=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            allow_patterns=["*.json", "*.py", "*token*", "*.txt"],
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_hf_path,
        padding_side=padding_side,
        trust_remote_code=True,
        **kwargs,
    )
    padding_check_and_fix(tokenizer)  # Check and fix tokenizer viability

    return tokenizer


def get_qpc_dir_name_infer(
    num_cores,
    mos,
    batch_size,
    prompt_len,
    ctx_len,
    mxfp6,
    mxint8,
    device_group,
    full_batch_size,
):
    qpc_base_dir_name = (
        f"qpc_{num_cores}cores_{batch_size}bs_{prompt_len}pl_{ctx_len}cl_{mos}mos"
        + f"{f'_{full_batch_size}fbs_' if full_batch_size is not None else '_'}"
        + f"{len(device_group) if device_group is not None else 1}"
        + "devices"
        + (
            "_mxfp6_mxint8"
            if (mxfp6 and mxint8)
            else "_mxfp6"
            if mxfp6
            else "_fp16_mxint8"
            if mxint8
            else "_fp16"
        )
    )

    return qpc_base_dir_name


def get_qpc_dir_path(
    model_card_name,
    num_cores,
    mos,
    batch_size,
    prompt_len,
    ctx_len,
    mxfp6,
    mxint8,
    device_group,
    full_batch_size,
):
    # Create a unique directory name for the QPC model based on all parameters
    qpc_base_dir_name = get_qpc_dir_name_infer(
        num_cores=num_cores,
        mos=mos,
        batch_size=batch_size,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        mxfp6=mxfp6,
        mxint8=mxint8,
        device_group=device_group,
        full_batch_size=full_batch_size,
    )
    model_card_dir = os.path.join(QEFF_MODELS_DIR, str(model_card_name))
    os.makedirs(model_card_dir, exist_ok=True)

    qpc_dir_path = os.path.join(model_card_dir, qpc_base_dir_name, "qpcs")
    return qpc_dir_path


def check_and_assign_cache_dir(local_model_dir, cache_dir):
    if local_model_dir is not None:
        if cache_dir is not None:
            logger.warning(
                f"Both local_model_dir ({local_model_dir}) and cache_dir "
                f"({cache_dir}) given. Using local_model_dir."
            )
        return None
    return cache_dir if cache_dir else None


def padding_check_and_fix(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> None:
    """
    Checks and fixes tokenizer padding side and pad_token_id viability.
    --------

    tokenizer: `Union[PreTrainedTokenizer, PreTrainedTokenizerFast]` -
    Pass model tokenizer to check and fix.
    """
    if tokenizer.padding_side != "right":
        logger.warning(
            "Setting tokenizer padding_side to 'right', got "
            f"{tokenizer.padding_side}"
        )
        tokenizer.padding_side = "right"

    if tokenizer.pad_token_id is None:
        assert (
            tokenizer.eos_token_id is not None
        ), "Found tokenizer.eos_token_id to be None, expected int"
        # If Pad token is out of range of vocab size
        if tokenizer.eos_token_id < tokenizer.vocab_size:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = tokenizer.vocab_size - 1


def get_padding_shape_from_config(config, batch_size, seq_len):
    """
    Gets padding dims from model config - number of kv heads and d_head
    and returns padding shape -
    (batch_size, number of kv heads, seq_len, hidden size)
    required for initialization of past_key_values
    --------

    :config: AutoConfig from pretrained model.
    :batch_size: int. number of input prompts used to create inputs
    :seq_len: int. sequence length to run the model for.

    Return:
        List[int, int, int, int]
    """

    if hasattr(
        config, "n_head"
    ):  # Assuming n_head is a key in the config (GPTs/CodeGen)
        n_heads = config.n_head
        d_head = config.n_embd // config.n_head
    elif hasattr(config, "num_key_value_heads") and hasattr(
        config, "num_attention_heads"
    ):  # Check for num_key_value_heads (Llama/Mistral)
        n_heads = config.num_key_value_heads
        d_head = config.hidden_size // config.num_attention_heads
    elif hasattr(
        config, "n_heads"
    ):  # Check for n_heads and d_model in the config (MPT Model)
        n_heads = config.n_heads
        d_head = config.d_model // config.n_heads
    elif hasattr(config, "new_decoder_architecture"):  # Check for Falcon
        new_decoder_architecture = config.new_decoder_architecture
        if (
            new_decoder_architecture
        ):  # multi_query is ignored when new_decoder_architecture is True
            n_heads = config.num_attention_heads
        else:
            if hasattr(config, "multi_query"):
                multi_query_value = config.multi_query
                n_heads = 1 if multi_query_value else config.num_attention_heads
        d_head = config.hidden_size // config.num_attention_heads
    else:
        raise ValueError(
            "Invalid model configuration: n_head/d_heads or num_key_value_heads"
            " not found."
        )
    padding_shape = [batch_size, n_heads, seq_len, d_head]
    if (
        hasattr(config, "architectures") and 
        config.architectures is not None and
        "GPTBigCodeForCausalLM" in config.architectures
    ):  # Check for Starcoder1 - 3D layout
        padding_shape = [batch_size, seq_len, d_head]
    return padding_shape


def get_num_layers_from_config(config):
    """
    Gets number of layers from model config
    --------

    :config: AutoConfig from pretrained model.

    Return:
        number of layers
    """

    if hasattr(
        config, "n_layer"
    ):  # Assuming n_layer is a key in the config (GPTs/CodeGen)
        n_layer = config.n_layer
    elif hasattr(config, "num_hidden_layers"):  # llama/Mistral/Falcon
        n_layer = config.num_hidden_layers
    elif hasattr(
        config, "n_layers"
    ):  # Check for n_layers in the config (MPT Model)
        n_layer = config.n_layers
    else:
        raise ValueError(
            "Invalid model configuration: n_layer/n_layers or num_hidden_layers"
            " not found."
        )

    return n_layer
