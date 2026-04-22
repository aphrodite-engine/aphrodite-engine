# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

from aphrodite.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from aphrodite.v1.kv_offload.worker.worker import TransferSpec

ReqId = str


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: dict[ReqId, TransferSpec]
    reqs_to_store: dict[ReqId, TransferSpec]
    reqs_to_flush: set[str] | None = None
