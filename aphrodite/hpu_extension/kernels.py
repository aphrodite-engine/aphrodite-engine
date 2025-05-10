###############################################################################
# Copyright (C) 2025 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from .utils import logger
from functools import cache


@cache
def fsdpa():
    try:
        from habana_frameworks.torch.hpex.kernels import FusedSDPA
        return FusedSDPA
    except ImportError:
        logger().warning("Could not import HPU FusedSDPA kernel. "
                         "Aphrodite will use native implementation.")

@cache
def rms_norm():
    try:
        from habana_frameworks.torch.hpex.normalization import FusedRMSNorm
        return FusedRMSNorm
    except ImportError:
        logger().warning("Could not import HPU FusedRMSNorm kernel. "
                         "Aphrodite will use forward_native implementation "
                         "of RMSNorm.")
