# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .connectors.base import OmniConnectorBase
from .connectors.mooncake_store_connector import MooncakeStoreConnector
from .connectors.shm_connector import SharedMemoryConnector
from .connectors.yuanrong_connector import YuanrongConnector

try:
    from aphrodite.omni.platforms.npu.omni_connectors.yuanrong_transfer_engine_connector import (
        YuanrongTransferEngineConnector as _YuanrongTransferEngineConnector,
    )

    YuanrongTransferEngineConnector: type[OmniConnectorBase] | None = _YuanrongTransferEngineConnector
except ImportError:
    YuanrongTransferEngineConnector = None

try:
    from .connectors.mooncake_transfer_engine_connector import (
        MooncakeTransferEngineConnector as _MooncakeTransferEngineConnector,
    )

    MooncakeTransferEngineConnector: type[OmniConnectorBase] | None = _MooncakeTransferEngineConnector
except ImportError:
    MooncakeTransferEngineConnector = None  # RDMA deps (msgspec/zmq/mooncake) not installed

try:
    from .connectors.mori_transfer_engine_connector import MoriTransferEngineConnector as _MoriTransferEngineConnector

    MoriTransferEngineConnector: type[OmniConnectorBase] | None = _MoriTransferEngineConnector
except ImportError:
    MoriTransferEngineConnector = None  # RDMA deps (msgspec/zmq/mori) not installed
from .factory import OmniConnectorFactory
from .utils.config import ConnectorSpec, OmniTransferConfig
from .utils.initialization import (
    build_stage_connectors,
    get_connectors_config_for_stage,
    get_stage_connector_config,
    initialize_connectors_from_config,
    initialize_orchestrator_connectors,
    load_omni_transfer_config,
)

# Backward-compatible alias: MooncakeConnector was renamed to MooncakeStoreConnector.
# Keep this alias for at least one release cycle.
MooncakeConnector = MooncakeStoreConnector

__all__ = [
    # Config
    "ConnectorSpec",
    "OmniTransferConfig",
    # Base classes and implementations
    "OmniConnectorBase",
    # Factory
    "OmniConnectorFactory",
    # Specific implementations
    "MooncakeConnector",  # compat alias → MooncakeStoreConnector
    "MooncakeStoreConnector",
    "MooncakeTransferEngineConnector",
    "MoriTransferEngineConnector",
    "SharedMemoryConnector",
    "YuanrongConnector",
    "YuanrongTransferEngineConnector",
    # Utilities
    "load_omni_transfer_config",
    "initialize_connectors_from_config",
    "get_connectors_config_for_stage",
    # Manager helpers
    "initialize_orchestrator_connectors",
    "get_stage_connector_config",
    "build_stage_connectors",
]
