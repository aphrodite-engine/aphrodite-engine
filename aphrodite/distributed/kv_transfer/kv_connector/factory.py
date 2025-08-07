import importlib
from typing import TYPE_CHECKING, Callable

from loguru import logger

import aphrodite.common.envs as envs
from aphrodite.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from aphrodite.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole

if TYPE_CHECKING:
    from aphrodite.common.config import AphroditeConfig


class KVConnectorFactory:
    _registry: dict[str, Callable[[], type[KVConnectorBase]]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str,
                           class_name: str) -> None:
        """Register a connector with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader() -> type[KVConnectorBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_connector(
        cls,
        config: "AphroditeConfig",
        role: KVConnectorRole,
    ) -> KVConnectorBase:
        if not envs.APHRODITE_USE_V1:
            raise ValueError("Attempting to initialize a V1 Connector, "
                             f"but found {envs.APHRODITE_USE_V1=}")

        kv_transfer_config = config.kv_transfer_config
        connector_name = kv_transfer_config.kv_connector
        if connector_name in cls._registry:
            connector_cls = cls._registry[connector_name]()
        else:
            connector_module_path = kv_transfer_config.kv_connector_module_path
            if connector_module_path is None:
                raise ValueError(
                    f"Unsupported connector type: {connector_name}")
            connector_module = importlib.import_module(connector_module_path)
            connector_cls = getattr(connector_module, connector_name)
        assert issubclass(connector_cls, KVConnectorBase)
        logger.info("Creating v1 connector with name: {} and engine_id: {}",
                    connector_cls.__name__, kv_transfer_config.engine_id)
        # NOTE(Kuntai): v1 connector is explicitly separated into two roles.
        # Scheduler connector:
        # - Co-locate with scheduler process
        # - Should only be used inside the Scheduler class
        # Worker connector:
        # - Co-locate with worker process
        # - Should only be used inside the forward context & attention layer
        # We build separately to enforce strict separation
        return connector_cls(config, role)


# Register various connectors here.
# The registration should not be done in each individual file, as we want to
# only load the files corresponding to the current connector.

KVConnectorFactory.register_connector(
    "SharedStorageConnector",
    "aphrodite.distributed.kv_transfer.kv_connector.v1.shared_storage_connector",
    "SharedStorageConnector")

KVConnectorFactory.register_connector(
    "P2pNcclConnector",
    "aphrodite.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector",
    "P2pNcclConnector")

KVConnectorFactory.register_connector(
    "LMCacheConnectorV1",
    "aphrodite.distributed.kv_transfer.kv_connector.v1.lmcache_connector",
    "LMCacheConnectorV1")

KVConnectorFactory.register_connector(
    "NixlConnector",
    "aphrodite.distributed.kv_transfer.kv_connector.v1.nixl_connector",
    "NixlConnector")

KVConnectorFactory.register_connector(
    "MultiConnector",
    "aphrodite.distributed.kv_transfer.kv_connector.v1.multi_connector",
    "MultiConnector")
