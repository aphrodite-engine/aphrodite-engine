import importlib
import os
import pickle
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import cloudpickle
import torch.nn as nn
from loguru import logger

from aphrodite.common.utils import is_hip

from .interfaces import (has_inner_state, has_noops, is_attention_free,
                         supports_multimodal, supports_pp)
from .interfaces_base import is_embedding_model, is_text_generation_model

_TEXT_GENERATION_MODELS = {
    # [Decoder-only]
    "AquilaForCausalLM": ('llama', 'LlamaForCausalLM'),
    "AquilaModel": ('llama', 'LlamaForCausalLM'),
    "ArcticForCausalLM": ('arctic', 'ArcticForCausalLM'),
    "BaiChuanForCausalLM": ('baichuan', 'BaiChuanForCausalLM'),
    "BaichuanForCausalLM": ('baichuan', 'BaichuanForCausalLM'),
    "BloomForCausalLM": ('bloom', 'BloomForCausalLM'),
    # ChatGLMModel supports multimodal
    "CohereForCausalLM": ('commandr', 'CohereForCausalLM'),
    "DbrxForCausalLM": ('dbrx', 'DbrxForCausalLM'),
    "DeciLMForCausalLM": ('nemotron_nas', 'DeciLMForCausalLM'),
    "DeepseekForCausalLM": ('deepseek', 'DeepseekForCausalLM'),
    "DeepseekV2ForCausalLM": ('deepseek_v2', 'DeepseekV2ForCausalLM'),
    "ExaoneForCausalLM": ('exaone', 'ExaoneForCausalLM'),
    "FalconForCausalLM": ('falcon', 'FalconForCausalLM'),
    "FalconMambaForCausalLM": ('mamba', 'MambaForCausalLM'),
    "GPT2LMHeadModel": ('gpt2', 'GPT2LMHeadModel'),
    "GPTBigCodeForCausalLM": ('gpt_bigcode', 'GPTBigCodeForCausalLM'),
    "GPTJForCausalLM": ('gpt_j', 'GPTJForCausalLM'),
    "GPTNeoXForCausalLM": ('gpt_neox', 'GPTNeoXForCausalLM'),
    "Gemma2ForCausalLM": ('gemma2', 'Gemma2ForCausalLM'),
    "GemmaForCausalLM": ('gemma', 'GemmaForCausalLM'),
    "GraniteForCausalLM": ('granite', 'GraniteForCausalLM'),
    "GraniteMoeForCausalLM": ('granitemoe', 'GraniteMoeForCausalLM'),
    "InternLM2ForCausalLM": ('internlm2', 'InternLM2ForCausalLM'),
    "InternLMForCausalLM": ('llama', 'LlamaForCausalLM'),
    "InternLM2VEForCausalLM": ("internlm2_ve", "InternLM2VEForCausalLM"),
    "JAISLMHeadModel": ('jais', 'JAISLMHeadModel'),
    "JambaForCausalLM": ('jamba', 'JambaForCausalLM'),
    "LLaMAForCausalLM": ('llama', 'LlamaForCausalLM'),
    "LlamaForCausalLM": ('llama', 'LlamaForCausalLM'),
    "MPTForCausalLM": ('mpt', 'MPTForCausalLM'),
    "MambaForCausalLM": ('mamba', 'MambaForCausalLM'),
    "MiniCPM3ForCausalLM": ('minicpm3', 'MiniCPM3ForCausalLM'),
    "MiniCPMForCausalLM": ('minicpm', 'MiniCPMForCausalLM'),
    "MistralForCausalLM": ('llama', 'LlamaForCausalLM'),
    "MixtralForCausalLM": ('mixtral', 'MixtralForCausalLM'),
    "MptForCausalLM": ('mpt', 'MPTForCausalLM'),
    "NemotronForCausalLM": ('nemotron', 'NemotronForCausalLM'),
    "NVLM_D": ("nvlm_d", "NVLM_D_Model"),
    "OPTForCausalLM": ('opt', 'OPTForCausalLM'),
    "OlmoForCausalLM": ('olmo', 'OlmoForCausalLM'),
    "OlmoeForCausalLM": ('olmoe', 'OlmoeForCausalLM'),
    "OrionForCausalLM": ('orion', 'OrionForCausalLM'),
    "PersimmonForCausalLM": ('persimmon', 'PersimmonForCausalLM'),
    "Phi3ForCausalLM": ('phi3', 'Phi3ForCausalLM'),
    "Phi3SmallForCausalLM": ('phi3_small', 'Phi3SmallForCausalLM'),
    "PhiForCausalLM": ('phi', 'PhiForCausalLM'),
    "PhiMoEForCausalLM": ('phimoe', 'PhiMoEForCausalLM'),
    "QuantMixtralForCausalLM": ('mixtral_quant', 'MixtralForCausalLM'),
    "Qwen2ForCausalLM": ('qwen2', 'Qwen2ForCausalLM'),
    "Qwen2MoeForCausalLM": ('qwen2_moe', 'Qwen2MoeForCausalLM'),
    "Qwen2VLForConditionalGeneration":
    ('qwen2_vl', 'Qwen2VLForConditionalGeneration'),
    "RWForCausalLM": ('falcon', 'FalconForCausalLM'),
    "SolarForCausalLM": ('solar', 'SolarForCausalLM'),
    "StableLMEpochForCausalLM": ('stablelm', 'StablelmForCausalLM'),
    "StableLmForCausalLM": ('stablelm', 'StablelmForCausalLM'),
    "Starcoder2ForCausalLM": ('starcoder2', 'Starcoder2ForCausalLM'),
    "XverseForCausalLM": ('xverse', 'XverseForCausalLM'),
    # [Encoder-decoder]
    "BartModel": ("bart", "BartForConditionalGeneration"),
    "BartForConditionalGeneration": ("bart", "BartForConditionalGeneration"),
    "Florence2ForConditionalGeneration": ("florence2", "Florence2ForConditionalGeneration"),  # noqa: E501
}


_EMBEDDING_MODELS = {
    # [Text-only]
    "BertModel": ("bert", "BertEmbeddingModel"),
    "Gemma2Model": ("gemma2", "Gemma2EmbeddingModel"),
    "MistralModel": ("llama", "LlamaEmbeddingModel"),
    "Qwen2ForRewardModel": ("qwen2_rm", "Qwen2ForRewardModel"),
    "Qwen2ForSequenceClassification": (
        "qwen2_cls", "Qwen2ForSequenceClassification"),
    # [Multimodal]
    "LlavaNextForConditionalGeneration": ("llava_next", "LlavaNextForConditionalGeneration"),  # noqa: E501
    "Phi3VForCausalLM": ("phi3v", "Phi3VForCausalLM"),
}

_MULTIMODAL_MODELS = {
    "Blip2ForConditionalGeneration": ('blip2', 'Blip2ForConditionalGeneration'),
    "ChameleonForConditionalGeneration":
    ('chameleon', 'ChameleonForConditionalGeneration'),
    "ChatGLMModel": ("chatglm", "ChatGLMForCausalLM"),
    "ChatGLMForConditionalGeneration": ("chatglm", "ChatGLMForCausalLM"),
    "FuyuForCausalLM": ('fuyu', 'FuyuForCausalLM'),
    "InternVLChatModel": ('internvl', 'InternVLChatModel'),
    "LlavaForConditionalGeneration": ('llava', 'LlavaForConditionalGeneration'),
    "LlavaNextForConditionalGeneration":
    ('llava_next', 'LlavaNextForConditionalGeneration'),
    "LlavaNextVideoForConditionalGeneration":
    ('llava_next_video', 'LlavaNextVideoForConditionalGeneration'),
    "LlavaOnevisionForConditionalGeneration":
    ('llava_onevision', 'LlavaOnevisionForConditionalGeneration'),
    "MiniCPMV": ('minicpmv', 'MiniCPMV'),
    "MllamaForConditionalGeneration": ('mllama',
                                       'MllamaForConditionalGeneration'),
    "MolmoForCausalLM": ('molmo', 'MolmoForCausalLM'),
    "PaliGemmaForConditionalGeneration":
    ('paligemma', 'PaliGemmaForConditionalGeneration'),
    "Phi3VForCausalLM": ('phi3v', 'Phi3VForCausalLM'),
    "PixtralForConditionalGeneration":
    ('pixtral', 'PixtralForConditionalGeneration'),
    "QWenLMHeadModel": ('qwen', 'QWenLMHeadModel'),
    "Qwen2VLForConditionalGeneration":
    ('qwen2_vl', 'Qwen2VLForConditionalGeneration'),
    "Qwen2AudioForConditionalGeneration": ("qwen2_audio", "Qwen2AudioForConditionalGeneration"),  # noqa: E501
    "UltravoxModel": ('ultravox', 'UltravoxModel'),
}


_SPECULATIVE_DECODING_MODELS = {
    "EAGLEModel": ("eagle", "EAGLE"),
    "MedusaModel": ("medusa", "Medusa"),
    "MLPSpeculatorPreTrainedModel": ("mlp_speculator", "MLPSpeculator"),
}

_APHRODITE_MODELS = {
    **_TEXT_GENERATION_MODELS,
    **_EMBEDDING_MODELS,
    **_MULTIMODAL_MODELS,
    **_SPECULATIVE_DECODING_MODELS,
}


# Models not supported by ROCm.
_ROCM_UNSUPPORTED_MODELS: List[str] = []

# Models partially supported by ROCm.
# Architecture -> Reason.
_ROCM_SWA_REASON = ("Sliding window attention (SWA) is not yet supported in "
                    "Triton flash attention. For half-precision SWA support, "
                    "please use CK flash attention by setting "
                    "`APHRODITE_USE_TRITON_FLASH_ATTN=0`")
_ROCM_PARTIALLY_SUPPORTED_MODELS: Dict[str, str] = {
    "Qwen2ForCausalLM":
    _ROCM_SWA_REASON,
    "MistralForCausalLM":
    _ROCM_SWA_REASON,
    "MixtralForCausalLM":
    _ROCM_SWA_REASON,
    "PaliGemmaForConditionalGeneration":
    ("ROCm flash attention does not yet "
     "fully support 32-bit precision on PaliGemma"),
    "Phi3VForCausalLM":
    ("ROCm Triton flash attention may run into compilation errors due to "
     "excessive use of shared memory. If this happens, disable Triton FA "
     "by setting `APHRODITE_USE_TRITON_FLASH_ATTN=0`")
}


@dataclass(frozen=True)
class _ModelInfo:
    is_text_generation_model: bool
    is_embedding_model: bool
    supports_multimodal: bool
    supports_pp: bool
    has_inner_state: bool
    is_attention_free: bool
    has_noops: bool

    @staticmethod
    def from_model_cls(model: Type[nn.Module]) -> "_ModelInfo":
        return _ModelInfo(
            is_text_generation_model=is_text_generation_model(model),
            is_embedding_model=is_embedding_model(model),
            supports_multimodal=supports_multimodal(model),
            supports_pp=supports_pp(model),
            has_inner_state=has_inner_state(model),
            is_attention_free=is_attention_free(model),
            has_noops=has_noops(model),
        )


class _BaseRegisteredModel(ABC):

    @abstractmethod
    def inspect_model_cls(self) -> _ModelInfo:
        raise NotImplementedError

    @abstractmethod
    def load_model_cls(self) -> Type[nn.Module]:
        raise NotImplementedError


@dataclass(frozen=True)
class _RegisteredModel(_BaseRegisteredModel):
    """
    Represents a model that has already been imported in the main process.
    """

    interfaces: _ModelInfo
    model_cls: Type[nn.Module]

    @staticmethod
    def from_model_cls(model_cls: Type[nn.Module]):
        return _RegisteredModel(
            interfaces=_ModelInfo.from_model_cls(model_cls),
            model_cls=model_cls,
        )

    def inspect_model_cls(self) -> _ModelInfo:
        return self.interfaces

    def load_model_cls(self) -> Type[nn.Module]:
        return self.model_cls


@dataclass(frozen=True)
class _LazyRegisteredModel(_BaseRegisteredModel):
    """
    Represents a model that has not been imported in the main process.
    """
    module_name: str
    class_name: str

    # Performed in another process to avoid initializing CUDA
    def inspect_model_cls(self) -> _ModelInfo:
        return _run_in_subprocess(
            lambda: _ModelInfo.from_model_cls(self.load_model_cls()))

    def load_model_cls(self) -> Type[nn.Module]:
        mod = importlib.import_module(self.module_name)
        return getattr(mod, self.class_name)


@lru_cache(maxsize=128)
def _try_load_model_cls(
    model_arch: str,
    model: _BaseRegisteredModel,
) -> Optional[Type[nn.Module]]:
    if is_hip():
        if model_arch in _ROCM_UNSUPPORTED_MODELS:
            raise ValueError(f"Model architecture '{model_arch}' is not "
                             "supported by ROCm for now.")

        if model_arch in _ROCM_PARTIALLY_SUPPORTED_MODELS:
            msg = _ROCM_PARTIALLY_SUPPORTED_MODELS[model_arch]
            logger.warning(
                f"Model architecture '{model_arch}' is partially "
                f"supported by ROCm: {msg}")

    try:
        return model.load_model_cls()
    except Exception:
        logger.exception(f"Error in loading model architecture '{model_arch}'")
        return None


@lru_cache(maxsize=128)
def _try_inspect_model_cls(
    model_arch: str,
    model: _BaseRegisteredModel,
) -> Optional[_ModelInfo]:
    try:
        return model.inspect_model_cls()
    except Exception:
        logger.exception(
            f"Error in inspecting model architecture '{model_arch}'")
        return None


@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    models: Dict[str, _BaseRegisteredModel] = field(default_factory=dict)

    def get_supported_archs(self) -> List[str]:
        return list(self.models.keys())

    def register_model(
        self,
        model_arch: str,
        model_cls: Union[Type[nn.Module], str],
    ) -> None:
        """
        Register an external model to be used in Aphrodite.

        :code:`model_cls` can be either:

        - A :class:`torch.nn.Module` class directly referencing the model.
        - A string in the format :code:`<module>:<class>` which can be used to
          lazily import the model. This is useful to avoid initializing CUDA
          when importing the model and thus the related error
          :code:`RuntimeError: Cannot re-initialize CUDA in forked subprocess`.
        """
        if model_arch in self.models:
            logger.warning(
                f"Model architecture {model_arch} is already registered, and "
                "will be overwritten by the new model class.")

        if isinstance(model_cls, str):
            split_str = model_cls.split(":")
            if len(split_str) != 2:
                msg = "Expected a string in the format `<module>:<class>`"
                raise ValueError(msg)

            model = _LazyRegisteredModel(*split_str)
        else:
            model = _RegisteredModel.from_model_cls(model_cls)

        self.models[model_arch] = model

    def _raise_for_unsupported(self, architectures: List[str]):
        all_supported_archs = self.get_supported_archs()

        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}")

    def _try_load_model_cls(self,
                            model_arch: str) -> Optional[Type[nn.Module]]:
        if model_arch not in self.models:
            return None

        return _try_load_model_cls(model_arch, self.models[model_arch])

    def _try_inspect_model_cls(self, model_arch: str) -> Optional[_ModelInfo]:
        if model_arch not in self.models:
            return None

        return _try_inspect_model_cls(model_arch, self.models[model_arch])

    def _normalize_archs(
        self,
        architectures: Union[str, List[str]],
    ) -> List[str]:
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            logger.warning("No model architectures are specified")

        return architectures

    def inspect_model_cls(
        self,
        architectures: Union[str, List[str]],
    ) -> _ModelInfo:
        architectures = self._normalize_archs(architectures)

        for arch in architectures:
            model_info = self._try_inspect_model_cls(arch)
            if model_info is not None:
                return model_info

        return self._raise_for_unsupported(architectures)

    def resolve_model_cls(
        self,
        architectures: Union[str, List[str]],
    ) -> Tuple[Type[nn.Module], str]:
        architectures = self._normalize_archs(architectures)

        for arch in architectures:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)

        return self._raise_for_unsupported(architectures)

    def is_text_generation_model(
        self,
        architectures: Union[str, List[str]],
    ) -> bool:
        return self.inspect_model_cls(architectures).is_text_generation_model

    def is_embedding_model(
        self,
        architectures: Union[str, List[str]],
    ) -> bool:
        return self.inspect_model_cls(architectures).is_embedding_model

    def is_multimodal_model(
        self,
        architectures: Union[str, List[str]],
    ) -> bool:
        return self.inspect_model_cls(architectures).supports_multimodal

    def is_pp_supported_model(
        self,
        architectures: Union[str, List[str]],
    ) -> bool:
        return self.inspect_model_cls(architectures).supports_pp

    def model_has_inner_state(self, architectures: Union[str,
                                                         List[str]]) -> bool:
        return self.inspect_model_cls(architectures).has_inner_state

    def is_attention_free_model(self, architectures: Union[str,
                                                           List[str]]) -> bool:
        return self.inspect_model_cls(architectures).is_attention_free

    def is_noops_model(
        self,
        architectures: Union[str, List[str]],
    ) -> bool:
        model_cls = self.inspect_model_cls(architectures)
        return model_cls.has_noops


ModelRegistry = _ModelRegistry({
    model_arch: _LazyRegisteredModel(
        module_name=f"aphrodite.modeling.models.{mod_relname}",
        class_name=cls_name,
    )
    for model_arch, (mod_relname, cls_name) in _APHRODITE_MODELS.items()
})

_T = TypeVar("_T")


def _run_in_subprocess(fn: Callable[[], _T]) -> _T:
    # NOTE: We use a temporary directory instead of a temporary file to avoid
    # issues like https://stackoverflow.com/questions/23212435/permission-denied-to-write-to-my-temporary-file
    with tempfile.TemporaryDirectory() as tempdir:
        output_filepath = os.path.join(tempdir, "registry_output.tmp")

        # `cloudpickle` allows pickling lambda functions directly
        input_bytes = cloudpickle.dumps((fn, output_filepath))

        # cannot use `sys.executable __file__` here because the script
        # contains relative imports
        returned = subprocess.run(
            [sys.executable, "-m", "aphrodite.modeling.models.registry"],
            input=input_bytes,
            capture_output=True)

        # check if the subprocess is successful
        try:
            returned.check_returncode()
        except Exception as e:
            # wrap raised exception to provide more information
            raise RuntimeError(f"Error raised in subprocess:\n"
                               f"{returned.stderr.decode()}") from e

        with open(output_filepath, "rb") as f:
            return pickle.load(f)


def _run() -> None:
    # Setup plugins
    from aphrodite.plugins import load_general_plugins
    load_general_plugins()

    fn, output_file = pickle.loads(sys.stdin.buffer.read())

    result = fn()

    with open(output_file, "wb") as f:
        f.write(pickle.dumps(result))


if __name__ == "__main__":
    _run()
