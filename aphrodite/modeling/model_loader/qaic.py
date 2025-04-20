import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoModelForCausalLM, PretrainedConfig

from aphrodite.common.config import (CacheConfig, DeviceConfig, ModelConfig,
                                     ParallelConfig, SchedulerConfig)
from aphrodite.common.sequence import SamplerOutput
from aphrodite.modeling.layers.logits_processor import LogitsProcessor
from aphrodite.modeling.layers.sampler import Sampler
from aphrodite.modeling.model_loader.qserve_model_runner import (
    QServeModelRunner)
from aphrodite.modeling.sampling_metadata import SamplingMetadata

QAIC_SUPPORTED_DTYPE = [
    "auto",
    "half",
    "float16",
    torch.float16,
]

QAIC_WARNING_DTYPE = [
    "bfloat16",
    torch.bfloat16
]

APHRODITE_CACHE_DTYPE_TO_QAIC_CACHE_DTYPE = {
    "auto": "fp16",
    "fp8": "mxint8",
    "mxint8": "mxint8"
}

class QaicCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.sampler = Sampler()
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                        logits_as_input=True)
        # Lazy initialized
        self.model: nn.Module

    def forward(
        self,
        input_ids: List[np.ndarray],
        positions: List[np.ndarray],
        batch_indices: List[int],
        is_prompt: bool,
    ) -> torch.Tensor:
        if is_prompt:
            logits_list = []
            for (iids,pids,bids) in zip(input_ids, positions, batch_indices):
                inputs = dict(
                    input_ids=iids,
                    position_ids=pids,
                    batch_index=np.array([[bids]])
                )
                qserve_inputs = {bids: inputs}
                logits = self.model.run(qserve_inputs, is_prompt)
                logits_list.append(logits)
            logits = np.concatenate(logits_list)
            return torch.from_numpy(logits)

        qserve_inputs = dict()
        for (iids,pids,bids) in zip(input_ids, positions, batch_indices):
            inputs = dict(
                input_ids=iids,
                position_ids=pids
            )
            qserve_inputs[bids] = inputs

        logits = self.model.run(qserve_inputs, is_prompt) # non-write
        # logits is a non-writable array. pytorch needs to have a
        # writable array to work properyly (else, behavior is undefined)
        # https://python-code.dev/articles/413443632
        return torch.from_numpy(np.copy(logits))

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        #breakpoint()
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_model(self, **kwargs):
        # Transform model to be QEfficient compliant
        # from QEfficient.cloud.export import main
        # main(model_name, cache_dir)
        model = QServeModelRunner(**kwargs)
        self.model = model

def _check_model_architecture(config: PretrainedConfig) -> None:
    from aphrodite.qaic.transformers.modeling_utils import (
        get_lists_of_cb_qeff_models)
    architectures = getattr(config, "architectures", [])

    for arch in architectures:
        if arch in get_lists_of_cb_qeff_models.architectures:
            return
    raise ValueError(
        f"Model architectures {architectures} are not supported in QEfficient "
        "transformer for qaic for now. Supported architectures: "
        f"{list(get_lists_of_cb_qeff_models.architectures)}")


def check_qpc_exists(qpc_path:str) -> bool:

    bin_path = os.path.join(qpc_path, "programqpc.bin")
    if not os.path.exists(qpc_path):
        return False
    if not os.path.isdir(qpc_path) and "programqpc.bin" in qpc_path:
        return True

    return os.path.exists(bin_path)


def get_qaic_model(model_config: ModelConfig,
                     parallel_config: ParallelConfig,
                     scheduler_config: SchedulerConfig,
                     cache_config: CacheConfig,
                     device_config: DeviceConfig) -> nn.Module:
    # Create a model instance.
    model = QaicCausalLM(model_config.hf_config)

    qpc_path = os.environ.get("APHRODITE_QAIC_QPC_PATH", None)

    num_cores = int(os.environ.get("APHRODITE_QAIC_NUM_CORES", 16))

    mos = int(os.environ.get("APHRODITE_QAIC_MOS", -1))

    dfs = os.environ.get("APHRODITE_QAIC_DFS_EN", True)
    if isinstance(dfs,str):
        dfs = dfs.lower() == "true"

    # Check if APHRODITE_QAIC_QPC_PATH path is valid
    if qpc_path and not check_qpc_exists(qpc_path):
        raise ValueError(
            f"Environment variable APHRODITE_QAIC_QPC_PATH is set!\n"
            f"QAIC qpc path {qpc_path} doesn't exist or didn't have "
            "compiled binary!\n"
            "Unset APHRODITE_QAIC_QPC_PATH, if you don't want to provide "
            "compiled qpc.\n")

    # Generate qpc using QEfficient transformer
    if not qpc_path:
        import hashlib

        import requests
        from QEfficient import QEFFAutoModelForCausalLM
        from QEfficient.utils import get_qpc_dir_name_infer, qpc_exists
        from requests.exceptions import HTTPError

        # Check if model architecture is supported by QEfficient transformer
        _check_model_architecture(model_config.hf_config)

        if model_config.dtype not in QAIC_SUPPORTED_DTYPE:
            if model_config.dtype in QAIC_WARNING_DTYPE:
                logger.warning(f"Dtype {model_config.dtype} not"  # noqa: G004
                        " supported by qaic switching to fp16")
            else:
                raise ValueError(
                    f"Currently qaic doesn't support dtype {model_config.dtype}"
                    " via aphrodite!"
                )

        mxfp6_en = False
        mxint8_en = False

        if (isinstance(model_config.quantization,str) and
             model_config.quantization == "mxfp6"):
            mxfp6_en = True

        if "mxint8" in APHRODITE_CACHE_DTYPE_TO_QAIC_CACHE_DTYPE[
            cache_config.cache_dtype]:
            mxint8_en = True

        cfg = {
            "num_cores": num_cores,
            "mos": mos,
            "batch_size": 1,
            "prompt_len": model_config.max_seq_len_to_capture,
            "ctx_len": scheduler_config.max_model_len,
            "mxfp6": mxfp6_en,
            "mxint8": mxint8_en,
            "device_group": device_config.device_group,
            "full_batch_size": scheduler_config.max_num_seqs,
            "aic_enable_depth_first": dfs,
            "qpc_dir_suffix": f"_{dfs}DFS_Aphrodite_" +
                str(hashlib.md5(str(model_config.hf_config).encode()).hexdigest())
        }

        qpc_base_dir_name = get_qpc_dir_name_infer(*(list(cfg.values())[:9]))
        qpc_base_dir_name = qpc_base_dir_name + "_" + cfg["qpc_dir_suffix"]
        qpc_path_exists, qpc_path = qpc_exists(
            model_name = model_config.model,
            qpc_base_dir_name=qpc_base_dir_name)

        if not qpc_path_exists:
            logger.info("Downloading model from Hugging face.")
            max_retries = 5
            retry_count = 0
            while retry_count < max_retries:
                try:
                    model_hf = AutoModelForCausalLM.from_pretrained(
                        model_config.model,
                        #cache_dir=hf_cache_dir,
                        trust_remote_code=model_config.trust_remote_code,
                        revision=model_config.revision,
                        code_revision=model_config.code_revision,
                        attn_implementation="eager"
                    )
                    break
                except requests.ReadTimeout as e:
                    logger.info(f"HF hub read timeout: {e}")
                    retry_count += 1

                except HTTPError as e:
                    retry_count = max_retries
                    if e.response.status_code == 401:
                        logger.error("You need to set HF_TOKEN environment"
                                    " variable to download private"
                                    " checkpoints.")
                    else:
                        raise e

            if retry_count >= max_retries:
                raise ValueError(
                    f"Unable to download model {model_config.model} from "
                    "Hugging face!")

            model_hf.eval()

            try:
                logger.info("Transforming and compiling model using "
                            "QEfficient library.")
                qeff_model = QEFFAutoModelForCausalLM(model_hf,
                    pretrained_model_name_or_path=model_config.model)

                qpc_path = qeff_model.export_and_compile(**cfg)
            except Exception as e:
                logger.error("Failed to transform and compile the model!")
                raise e

    logger.info(f"Using qpc:-{qpc_path}")
    # Load the weights from the cached or downloaded files.
    # model_config.qpc in None
    model.load_model(
        qpc_path=qpc_path,
        vocab_size=model_config.hf_config.vocab_size,
        device_id=device_config.device_group,
        seq_len=model_config.max_seq_len_to_capture,
        ctx_len=model_config.max_model_len,
        decode_bsz=scheduler_config.max_num_seqs)

    return model.eval()
