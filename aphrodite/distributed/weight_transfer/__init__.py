# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
"""
Weight transfer engines for syncing model weights from trainers
to inference workers.
"""

from aphrodite.distributed.weight_transfer.factory import WeightTransferEngineFactory

__all__ = [
    "WeightTransferEngineFactory",
]
