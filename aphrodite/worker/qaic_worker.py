"""A QAIC worker class."""
from typing import List, Tuple

import torch
import torch.distributed
from loguru import logger

from aphrodite.common.config import (CacheConfig, DeviceConfig, ModelConfig,
                                     ParallelConfig, SchedulerConfig)
from aphrodite.common.sequence import SamplerOutput, SequenceGroupMetadata
from aphrodite.modeling import set_random_seed
from aphrodite.worker.qaic_model_runner import QaicModelRunner
from aphrodite.worker.worker_base import LoraNotSupportedWorkerBase


class QaicWorker(LoraNotSupportedWorkerBase):
    """A worker class that executes the model on a group of qaic devices.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from aphrodite.common.utils import init_cached_hf_modules
            init_cached_hf_modules()

        self.model_runner = QaicModelRunner(model_config, parallel_config,
                                              scheduler_config, cache_config,
                                              device_config)

    def init_device(self) -> None:
        """Initialize qaic device
        """
        # TBD
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Aphrodite core implementation of Chuncked prefill not supported
        if self.scheduler_config.chunked_prefill_enabled:
            logger.warning(
                "Aphrodite chunked prefill is not supported"
                " and thus will be disabled automatically."
                " qaic backend supports its own internal"
                " chunking that is enabled by default.")
            self.scheduler_config.chunked_prefill_enabled = False

    def load_model(self):
        """Load model from QEfficient Transformer library
        """
        self.model_runner.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of blocks available for the KV cache.

        This determines how many KV blocks can fit into the configured QAIC
        KV cache space.

        Note that since swapping is not yet supported, so this return
        num_cpu_blocks as 0.

        """
        # Set the number of QAIC KV blocks to be the same as the maximum number
        # of sequences that can be processed in a single batch. This is
        # equivalent to schedule without PagedAttention.
        num_gpu_blocks = self.scheduler_config.max_num_seqs

        # Swap not yet supported with Qaic backend.
        num_cpu_blocks = 0

        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache.
        """

        # Different values are not tested.
        assert (not self.scheduler_config.use_v2_block_manager
                ), ("v2 block manager is not"
                    " supported with qaic backend.")
        assert num_cpu_blocks == 0
        assert num_gpu_blocks == self.scheduler_config.max_num_seqs

        #self.cache_config.num_gpu_blocks = num_gpu_blocks+1
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        self.cache_config.block_size = self.scheduler_config.max_model_len

        # Set max num batched token
        self.scheduler_config.max_num_batched_tokens = \
                                 self.scheduler_config.max_model_len * \
                                 self.scheduler_config.max_num_seqs

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> List[SamplerOutput]:
        """Execute model on qaic devices.
        """

        # Avoid executing models, if input group is empty..
        if len(seq_group_metadata_list) == 0:
            return []

        # execute model on qaic
        output = self.model_runner.execute_model(seq_group_metadata_list)

        # Qaic worker only supports single-step output.
        return [output]

    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.

        speculative decoding is not yet supported in qaic devices.
        """
        raise NotImplementedError
