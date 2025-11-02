from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

from aphrodite.logger import init_logger
from aphrodite.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from aphrodite.v1.kv_offload.worker.worker import OffloadingHandler

if TYPE_CHECKING:
    from aphrodite.config import AphroditeConfig

logger = init_logger(__name__)


class OffloadingSpec(ABC):
    """Spec for an offloading connector"""

    def __init__(self, aphrodite_config: "AphroditeConfig"):
        logger.warning(
            "Initializing OffloadingSpec. This API is experimental and "
            "subject to change in the future as we iterate the design."
        )
        self.aphrodite_config = aphrodite_config

        kv_transfer_config = aphrodite_config.kv_transfer_config
        assert kv_transfer_config is not None
        self.extra_config = kv_transfer_config.kv_connector_extra_config

        self.gpu_block_size = aphrodite_config.cache_config.block_size
        self.offloaded_block_size = int(
            self.extra_config.get("block_size", self.gpu_block_size)
        )

        assert self.offloaded_block_size % self.gpu_block_size == 0

    @abstractmethod
    def get_manager(self) -> OffloadingManager:
        """
        Get an OffloadingManager that will be used
        by the scheduler-side offloading connector to track
        offloaded blocks and manage evictions.
        """
        pass

    @abstractmethod
    def get_handlers(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        """
        Get offloading handlers along with their respective src and dst types.

        Args:
            kv_caches: A dictionary of layer_name -> gpu_kv_cache tensor.

        Yields:
            Tuples of (src_type, dst_type, offloading_handler).
        """
        pass
