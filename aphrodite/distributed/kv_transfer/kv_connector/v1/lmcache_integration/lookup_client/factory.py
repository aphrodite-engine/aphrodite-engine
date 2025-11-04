# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional, Union

from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig

from aphrodite.logger import init_logger

from .abstract_client import LookupClientInterface
from .hit_limit_lookup_client import HitLimitLookupClient
from .mooncake_lookup_client import MooncakeLookupClient

if TYPE_CHECKING:
    from aphrodite.config import AphroditeConfig

    from .lmcache_async_lookup_client import LMCacheAsyncLookupServer
    from .lmcache_lookup_client import LMCacheLookupServer

logger = init_logger(__name__)


class LookupClientFactory:
    """Factory for creating lookup clients and servers based on configuration."""

    @staticmethod
    def create_lookup_client(
        aphrodite_config: "AphroditeConfig",
        config: LMCacheEngineConfig,
    ) -> LookupClientInterface:
        """
        Create a lookup client based on the configuration.

        Args:
            aphrodite_config: The Aphrodite configuration
            config: The LMCache engine configuration

        Returns:
            A lookup client instance
        """

        # Check if external_lookup_client is configured
        if config.external_lookup_client is not None:
            if config.enable_async_loading:
                raise ValueError(
                    "Asynchronous loading is not supported for external lookup clients."
                )
            client = LookupClientFactory._create_external_lookup_client(
                config.external_lookup_client, aphrodite_config
            )
        else:
            from .lmcache_async_lookup_client import LMCacheAsyncLookupClient
            from .lmcache_lookup_client import LMCacheLookupClient

            if config.enable_async_loading:
                client = LMCacheAsyncLookupClient(aphrodite_config)
            else:
                client = LMCacheLookupClient(aphrodite_config)

        if config.hit_miss_ratio is not None and 0 <= config.hit_miss_ratio <= 1:
            return HitLimitLookupClient(client, config)
        return client

    @staticmethod
    def create_lookup_server(
        lmcache_engine: LMCacheEngine,
        aphrodite_config: "AphroditeConfig",
    ) -> Optional[Union["LMCacheLookupServer", "LMCacheAsyncLookupServer"]]:
        """
        Create a lookup server based on the configuration.

        Args:
            lmcache_engine: The LMCache engine instance
            aphrodite_config: The Aphrodite configuration

        Returns:
            A lookup server instance, or None if no server should be created
        """
        config = lmcache_engine.config
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 config is expected for lookup server and client"
        )

        # Only create the KV lookup API server on worker rank 0
        # when there are multiple workers and when not using external lookup client
        create_lookup_server_only_on_worker_0_for_mla = config.get_extra_config_value(
            "create_lookup_server_only_on_worker_0_for_mla",
            lmcache_engine.metadata.use_mla,
        )

        if config.external_lookup_client is None and (
            not create_lookup_server_only_on_worker_0_for_mla
            or lmcache_engine.metadata.worker_id == 0
        ):
            from .lmcache_async_lookup_client import LMCacheAsyncLookupServer
            from .lmcache_lookup_client import LMCacheLookupServer

            if config.enable_async_loading:
                return LMCacheAsyncLookupServer(lmcache_engine, aphrodite_config)
            else:
                return LMCacheLookupServer(lmcache_engine, aphrodite_config)

        return None

    @staticmethod
    def _create_external_lookup_client(
        external_lookup_uri: str,
        aphrodite_config: "AphroditeConfig",
    ) -> LookupClientInterface:
        """
        Create an external lookup client based on the URI format.

        Args:
            external_lookup_uri: URI in format <scheme>://<address>
            aphrodite_config: The Aphrodite configuration

        Returns:
            A lookup client instance

        Raises:
            ValueError: If the URI format is unsupported
        """
        # Parse URI scheme and address
        if "://" not in external_lookup_uri:
            raise ValueError(
                f"Invalid external lookup client URI format: {external_lookup_uri}. "
                "Expected format: <scheme>://<address>"
            )

        scheme, address = external_lookup_uri.split("://", 1)

        # Route to appropriate client based on scheme
        if scheme == "mooncakestore":
            return LookupClientFactory._create_mooncake_lookup_client(
                address, aphrodite_config
            )
        else:
            raise ValueError(
                f"Unsupported external lookup client scheme: {scheme}. "
                "Supported schemes: mooncakestore"
            )

    @staticmethod
    def _create_mooncake_lookup_client(
        master_address: str,
        aphrodite_config: "AphroditeConfig",
    ) -> "MooncakeLookupClient":
        """Create a MooncakeLookupClient instance."""
        from .mooncake_lookup_client import MooncakeLookupClient

        return MooncakeLookupClient(aphrodite_config, master_address)
