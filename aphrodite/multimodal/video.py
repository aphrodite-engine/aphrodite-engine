from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger

from aphrodite.common.config import ModelConfig
from aphrodite.common.utils import is_list_of
from aphrodite.inputs.registry import InputContext
from aphrodite.transformers_utils.processor import get_video_processor
from aphrodite.transformers_utils.tokenizer import get_tokenizer

from .base import MultiModalData, MultiModalInputs
from .image import ImagePlugin

cached_get_video_processor = lru_cache(get_video_processor)
cached_get_tokenizer = lru_cache(get_tokenizer)
VideoInput = Union[
    "np.ndarray",  # single video input
    List["np.ndarray"],
    # TODO: support more types
    # List[Image.Image], List[List[Image.Image]],
    # "torch.Tensor",
    # List["torch.Tensor"],
    # List[List["np.ndarrray"]],
    # List[List["torch.Tensor"]],
]


class VideoPlugin(ImagePlugin):
    """Plugin for video data."""

    def get_data_key(self) -> str:
        return "video"

    def _get_hf_video_processor(
        self,
        model_config: ModelConfig,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if mm_processor_kwargs is None:
            mm_processor_kwargs = {}
        return cached_get_video_processor(
            model_config.model,
            trust_remote_code=model_config.trust_remote_code,
            **mm_processor_kwargs)

    def _default_input_mapper(
        self,
        ctx: InputContext,
        data: MultiModalData[object],
        **mm_processor_kwargs,
    ) -> MultiModalInputs:
        model_config = ctx.model_config

        if isinstance(data, np.ndarray) or is_list_of(data, np.ndarray):
            video_processor = self._get_hf_video_processor(
                model_config,
                mm_processor_kwargs,
            )
            if video_processor is None:
                raise RuntimeError("No HuggingFace processor is available "
                                   "to process the video object")
            try:
                # NOTE: Similar to image; it may be a good idea to filter and
                # pass mm_processor_kwargs here too, but for now we don't to
                # avoid extra complexity if the initializer and preprocess
                # signatures of the processor don't align
                batch_data = video_processor(data, return_tensors="pt").data
            except Exception:
                logger.error(f"Failed to process video ({data})")
                raise
            return MultiModalInputs(batch_data)

        raise TypeError(f"Invalid video type: {type(data)}")

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        return 4096
