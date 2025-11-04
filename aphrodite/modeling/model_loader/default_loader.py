import dataclasses
import glob
import os
import time
from collections.abc import Generator, Iterable
from typing import cast

import torch
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from aphrodite.config import ModelConfig
from aphrodite.config.load import LoadConfig
from aphrodite.logger import init_logger
from aphrodite.modeling.model_loader.base_loader import BaseModelLoader
from aphrodite.modeling.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    fastsafetensors_weights_iterator,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    maybe_download_from_modelscope,
    multi_thread_pt_weights_iterator,
    multi_thread_safetensors_weights_iterator,
    np_cache_weights_iterator,
    pt_weights_iterator,
    safetensors_weights_iterator,
)
from aphrodite.platforms import current_platform
from aphrodite.quantization.torchao import torchao_version_at_least

logger = init_logger(__name__)


class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    # default number of thread when enable multithread weight loading
    DEFAULT_NUM_THREADS = 8

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        revision: str | None
        """The optional model revision."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        allow_patterns_overrides: list[str] | None = None
        """If defined, weights will load exclusively using these patterns."""

    counter_before_loading_weights: float = 0.0
    counter_after_loading_weights: float = 0.0

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        extra_config = load_config.model_loader_extra_config
        allowed_keys = {"enable_multithread_load", "num_threads"}
        unexpected_keys = set(extra_config.keys()) - allowed_keys

        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format {load_config.load_format}: {unexpected_keys}"
            )

    def _prepare_weights(
        self,
        model_name_or_path: str,
        revision: str | None,
        fall_back_to_pt: bool,
        allow_patterns_overrides: list[str] | None,
    ) -> tuple[str, list[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        model_name_or_path = maybe_download_from_modelscope(model_name_or_path, revision) or model_name_or_path

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME
        # Some quantized models use .pt files for storing the weights.
        if load_format == "auto":
            allow_patterns = ["*.safetensors", "*.bin"]
        elif load_format == "safetensors" or load_format == "fastsafetensors":
            use_safetensors = True
            allow_patterns = ["*.safetensors"]
        elif load_format == "mistral":
            use_safetensors = True
            allow_patterns = ["consolidated*.safetensors"]
            index_file = "consolidated.safetensors.index.json"
        elif load_format == "pt":
            allow_patterns = ["*.pt"]
        elif load_format == "npcache":
            allow_patterns = ["*.bin"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        else:
            hf_folder = model_name_or_path

        hf_weights_files: list[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    self.load_config.download_dir,
                    revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(hf_weights_files, hf_folder, index_file)
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(f"Cannot find any model weights with `{model_name_or_path}`")

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator_with_size(
        self, source: "Source"
    ) -> tuple[Generator[tuple[str, torch.Tensor], None, None], int]:
        """Get an iterator for the model weights and calculate total size."""
        extra_config = self.load_config.model_loader_extra_config
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )

        # Calculate total size by iterating through files
        total_bytes = 0
        for file_path in hf_weights_files:
            if os.path.exists(file_path):
                total_bytes += os.path.getsize(file_path)
        if self.load_config.load_format == "npcache":
            # Currently np_cache only support *.bin checkpoints
            assert use_safetensors is False
            weights_iterator = np_cache_weights_iterator(
                source.model_or_path,
                self.load_config.download_dir,
                hf_folder,
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
            )
        elif use_safetensors:
            if self.load_config.load_format == "fastsafetensors":
                weights_iterator = fastsafetensors_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                )
            else:
                if extra_config.get("enable_multithread_load"):
                    weights_iterator = multi_thread_safetensors_weights_iterator(
                        hf_weights_files,
                        self.load_config.use_tqdm_on_load,
                        max_workers=extra_config.get("num_threads", self.DEFAULT_NUM_THREADS),
                    )
                else:
                    weights_iterator = safetensors_weights_iterator(
                        hf_weights_files,
                        self.load_config.use_tqdm_on_load,
                        self.load_config.safetensors_load_strategy,
                    )
        else:
            if extra_config.get("enable_multithread_load"):
                weights_iterator = multi_thread_pt_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    self.load_config.pt_load_map_location,
                    max_workers=extra_config.get("num_threads", self.DEFAULT_NUM_THREADS),
                )
            else:
                weights_iterator = pt_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    self.load_config.pt_load_map_location,
                )

        if current_platform.is_tpu():
            from aphrodite.platforms.tpu import USE_TPU_INFERENCE

            if not USE_TPU_INFERENCE:
                # In PyTorch XLA, we should call `torch_xla.sync`
                # frequently so that not too many ops are accumulated
                # in the XLA program.
                import torch_xla

                def _xla_weights_iterator(iterator: Generator):
                    for weights in iterator:
                        yield weights
                        torch_xla.sync(wait=False)

                weights_iterator = _xla_weights_iterator(weights_iterator)

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return (
            ((source.prefix + name, tensor) for (name, tensor) in weights_iterator),
            total_bytes,
        )

    def _get_weights_iterator(self, source: "Source") -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        iterator, _ = self._get_weights_iterator_with_size(source)
        return iterator

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> tuple[Iterable[tuple[str, torch.Tensor]], int]:
        """
        Get all weights including primary and secondary sources with total size.
        """
        primary_weights = DefaultModelLoader.Source(
            model_config.model,
            model_config.revision,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )

        primary_iterator, primary_bytes = self._get_weights_iterator_with_size(primary_weights)
        total_bytes = primary_bytes

        # Collect all weight iterators and their sizes
        iterators = [primary_iterator]

        secondary_weights = cast(
            Iterable[DefaultModelLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        for source in secondary_weights:
            iterator, bytes_size = self._get_weights_iterator_with_size(source)
            iterators.append(iterator)
            total_bytes += bytes_size

        # Chain all iterators together
        from itertools import chain

        return chain.from_iterable(iterators), total_bytes

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(
            model_config.model,
            model_config.revision,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        if model_config.quantization == "torchao" and torchao_version_at_least("0.14.0"):
            self.load_config.safetensors_load_strategy = "torchao"
        weights_to_load = {name for name, _ in model.named_parameters()}

        # if we don't have `model.weight_metadata_and_attr_saved` defined and
        # set to True, it means that this is either offline quantization case
        # or the first run of online quantization
        # see online_quantization.py for detailed notes
        offline_quantization_or_first_run_of_online_quantization = not getattr(
            model, "weight_metadata_and_attr_saved", False
        )

        if model_config.quantization is None:
            # model is not quantized
            weights_iter, total_bytes = self.get_all_weights(model_config, model)
            from aphrodite.utils import tensor_progress_bar

            loaded_weights = model.load_weights(tensor_progress_bar(weights_iter, total_bytes, "Loading model weights"))
        elif offline_quantization_or_first_run_of_online_quantization:
            # case 1: offline quantized checkpoint
            # case 2: Step I1 first run of weight loading with
            # online quantization
            # see online_quantization.py for detailed notes
            weights_iter, total_bytes = self.get_all_weights(model_config, model)
            from aphrodite.utils import tensor_progress_bar

            loaded_weights = model.load_weights(tensor_progress_bar(weights_iter, total_bytes, "Loading model weights"))
        else:
            # to avoid circular dependency
            from aphrodite.modeling.model_loader.online_quantization import load_weights_and_online_quantize

            # subsequent runs of weight loading with online
            # quantization
            loaded_weights = load_weights_and_online_quantize(self, model, model_config)

        self.counter_after_loading_weights = time.perf_counter()
        logger.debug_once(
            "Loading weights took %.2f seconds",
            self.counter_after_loading_weights - self.counter_before_loading_weights,
            scope="local",
        )
        # We only enable strict check for non-quantized models
        # that have loaded weights tracking currently.
        if model_config.quantization is None and loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                raise ValueError(f"Following weights were not initialized from checkpoint: {weights_not_loaded}")
