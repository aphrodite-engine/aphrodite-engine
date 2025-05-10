import importlib
from typing import TYPE_CHECKING, Callable, Dict, Type

from loguru import logger

import aphrodite.common.envs as envs
from aphrodite.distributed.kv_transfer.kv_connector.base import (
    KVConnectorBaseType)
from aphrodite.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1, KVConnectorRole)

from .base import KVConnectorBase

if TYPE_CHECKING:
    from aphrodite.common.config import AphroditeConfig



class KVConnectorFactory:
    _registry: Dict[str, Callable[[], Type[KVConnectorBaseType]]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str,
                           class_name: str) -> None:
        """Register a connector with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader() -> Type[KVConnectorBaseType]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_connector_v0(cls, rank: int, local_rank: int,
                            config: "AphroditeConfig") -> KVConnectorBase:
        if envs.APHRODITE_USE_V1:
            raise ValueError("Attempting to initialize a V0 Connector, "
                             f"but found {envs.APHRODITE_USE_V1=}")

        connector_name = config.kv_transfer_config.kv_connector
        if connector_name not in cls._registry:
            raise ValueError(f"Unsupported connector type: {connector_name}")

        connector_cls = cls._registry[connector_name]()
        assert issubclass(connector_cls, KVConnectorBase)
        return connector_cls(rank, local_rank, config)

    @classmethod
    def create_connector_v1(
        cls,
        config: "AphroditeConfig",
        role: KVConnectorRole,
    ) -> KVConnectorBase_V1:
        if not envs.APHRODITE_USE_V1:
            raise ValueError("Attempting to initialize a V1 Connector, "
                             f"but found {envs.APHRODITE_USE_V1=}")

        connector_name = config.kv_transfer_config.kv_connector
        connector_cls = cls._registry[connector_name]()
        assert issubclass(connector_cls, KVConnectorBase_V1)
        logger.info("Creating v1 connector with name: %s", connector_name)
        # NOTE: v1 connector is explicitly separated into two roles.
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
    "PyNcclConnector",
    "aphrodite.distributed.kv_transfer.kv_connector.simple_connector",
    "SimpleConnector")

KVConnectorFactory.register_connector(
    "MooncakeConnector",
    "aphrodite.distributed.kv_transfer.kv_connector.simple_connector",
    "SimpleConnector")

KVConnectorFactory.register_connector(
    "LMCacheConnector",
    "aphrodite.distributed.kv_transfer.kv_connector.lmcache_connector",
    "LMCacheConnector")

KVConnectorFactory.register_connector(
    "MooncakeStoreConnector",
    "aphrodite.distributed.kv_transfer.kv_connector.mooncake_store_connector",
    "MooncakeStoreConnector")

KVConnectorFactory.register_connector(
    "SharedStorageConnector",
    "aphrodite.distributed.kv_transfer.kv_connector.v1.shared_storage_connector",
    "SharedStorageConnector")

KVConnectorFactory.register_connector(
    "LMCacheConnectorV1",
    "aphrodite.distributed.kv_transfer.kv_connector.v1.lmcache_connector",
    "LMCacheConnectorV1")
