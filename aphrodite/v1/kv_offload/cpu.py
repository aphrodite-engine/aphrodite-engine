from collections.abc import Iterator

import torch

from aphrodite.config import AphroditeConfig, get_layers_from_aphrodite_config
from aphrodite.modeling.layers.attention_layer_base import AttentionLayerBase
from aphrodite.platforms import current_platform
from aphrodite.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from aphrodite.v1.kv_offload.backends.cpu import CPUBackend
from aphrodite.v1.kv_offload.lru_manager import LRUOffloadingManager
from aphrodite.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from aphrodite.v1.kv_offload.spec import OffloadingSpec
from aphrodite.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandler
from aphrodite.v1.kv_offload.worker.worker import OffloadingHandler


class CPUOffloadingSpec(OffloadingSpec):
    def __init__(self, aphrodite_config: AphroditeConfig):
        super().__init__(aphrodite_config)

        swap_space_bytes = self.extra_config.get("swap_space_bytes")
        if not swap_space_bytes:
            # Try to auto-calculate from kv_bytes_per_rank if available
            kv_bytes_per_rank = self.extra_config.get("kv_bytes_per_rank")
            if kv_bytes_per_rank is not None:
                swap_space_bytes = int(kv_bytes_per_rank)
            else:
                # Fallback: calculate from kv_offloading_size
                kv_offloading_size = aphrodite_config.cache_config.kv_offloading_size
                if kv_offloading_size is not None:
                    num_kv_ranks = (
                        aphrodite_config.parallel_config.tensor_parallel_size
                        * aphrodite_config.parallel_config.pipeline_parallel_size
                    )
                    swap_space_bytes = int(kv_offloading_size * (1 << 30) / num_kv_ranks)
                else:
                    raise Exception(
                        "swap_space_bytes must be specified in kv_connector_extra_config, "
                        "or kv_offloading_size must be set in CacheConfig"
                    )
        self.swap_space_bytes: int = swap_space_bytes

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._handler: OffloadingHandler | None = None

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_bytes_per_offloaded_block = self.aphrodite_config.cache_config.kv_bytes_per_block * (
                self.offloaded_block_size // self.gpu_block_size
            )
            num_blocks = self.swap_space_bytes // kv_bytes_per_offloaded_block
            kv_events_config = self.aphrodite_config.kv_events_config
            enable_events = kv_events_config is not None and kv_events_config.enable_kv_cache_events
            self._manager = LRUOffloadingManager(
                CPUBackend(block_size=self.offloaded_block_size, num_blocks=num_blocks), enable_events=enable_events
            )
        return self._manager

    def get_handlers(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handler:
            if not current_platform.is_cuda_alike():
                raise Exception("CPU Offloading is currently only supported on CUDA-alike GPUs")

            layer_names = list(kv_caches.keys())
            layers = get_layers_from_aphrodite_config(self.aphrodite_config, AttentionLayerBase, layer_names)
            attn_backends = {layer_name: layers[layer_name].get_attn_backend() for layer_name in layer_names}

            kv_bytes_per_offloaded_block = self.aphrodite_config.cache_config.kv_bytes_per_block * (
                self.offloaded_block_size // self.gpu_block_size
            )
            num_blocks = self.swap_space_bytes // kv_bytes_per_offloaded_block

            self._handler = CpuGpuOffloadingHandler(
                attn_backends=attn_backends,
                gpu_block_size=self.gpu_block_size,
                cpu_block_size=self.offloaded_block_size,
                num_cpu_blocks=num_blocks,
                gpu_caches=kv_caches,
            )

        assert self._handler is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handler
