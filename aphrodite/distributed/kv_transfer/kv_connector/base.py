"""Defines the base type for KV cache connectors."""

from aphrodite.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1)

KVConnectorBase = KVConnectorBase_V1
KVConnectorBaseType = KVConnectorBase_V1

__all__ = ["KVConnectorBase", "KVConnectorBaseType"]
