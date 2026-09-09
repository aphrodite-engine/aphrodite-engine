# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Platform-dispatched layer implementations for HunyuanImage3.

The two blocks are selected by platform:

Layout::
    native/autoencoder_blocks.py          default ResnetBlock
    native/transformer_blocks.py          default ResBlock
    nvidia/autoencoder_blocks.py          CUDA ResnetBlock (fused)
    nvidia/transformer_blocks.py          CUDA ResBlock (fused)
"""

from aphrodite.omni.platforms import current_omni_platform

if current_omni_platform.is_cuda():
    from aphrodite.omni.diffusion.models.hunyuan_image3.layers.nvidia import ResBlock, ResnetBlock
else:
    from aphrodite.omni.diffusion.models.hunyuan_image3.layers.native import ResBlock, ResnetBlock

__all__ = ["ResBlock", "ResnetBlock"]
