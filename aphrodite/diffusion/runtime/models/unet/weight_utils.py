# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# Adapted from ComfyUI: comfy/supported_models.py, comfy/utils.py

# SPDX-License-Identifier: Apache-2.0
"""
Weight loading utilities for SDXL UNet.

Converts diffusers format state dicts to Aphrodite format.
"""

from collections.abc import Iterable

import torch


def state_dict_prefix_replace(
    state_dict: dict[str, torch.Tensor],
    replace_prefix: dict[str, str],
    filter_keys: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Replace prefixes in state dict keys.

    Args:
        state_dict: Original state dict
        replace_prefix: Dict mapping old prefix -> new prefix
        filter_keys: If True, only keep keys that match replace_prefix

    Returns:
        New state dict with replaced prefixes
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key
        matched = False

        for old_prefix, new_prefix in replace_prefix.items():
            if key.startswith(old_prefix):
                new_key = key.replace(old_prefix, new_prefix, 1)
                matched = True
                break

        if not filter_keys or matched:
            new_state_dict[new_key] = value

    return new_state_dict


def state_dict_key_replace(
    state_dict: dict[str, torch.Tensor],
    keys_to_replace: dict[str, str],
) -> dict[str, torch.Tensor]:
    """
    Replace specific keys in state dict.

    Args:
        state_dict: Original state dict
        keys_to_replace: Dict mapping old key -> new key

    Returns:
        New state dict with replaced keys
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = keys_to_replace.get(key, key)
        new_state_dict[new_key] = value

    return new_state_dict


def convert_sdxl_unet_state_dict(
    state_dict: dict[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
    prefix: str = "",
) -> dict[str, torch.Tensor]:
    """
    Convert SDXL UNet state dict from diffusers format to Aphrodite format.

    Diffusers format uses "unet." prefix, Aphrodite expects no prefix or custom prefix.

    Args:
        state_dict: State dict from diffusers or iterable of (name, tensor) tuples
        prefix: Optional prefix to add to all keys

    Returns:
        Converted state dict
    """
    # Convert iterable to dict if needed
    if isinstance(state_dict, Iterable) and not isinstance(state_dict, dict):
        state_dict = dict(state_dict)

    # Remove "unet." prefix if present
    replace_prefix = {}
    if prefix:
        # If we want to add a prefix, first remove any existing "unet." prefix
        replace_prefix["unet."] = prefix
    else:
        # Just remove "unet." prefix
        replace_prefix["unet."] = ""

    converted = state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False)

    return converted


def process_sdxl_unet_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    prefix: str = "",
) -> Iterable[tuple[str, torch.Tensor]]:
    """
    Process SDXL UNet weights for loading.

    Converts diffusers format weights to Aphrodite format.
    Based on ComfyUI's conversion logic in comfy/utils.py

    Args:
        weights: Iterable of (name, tensor) tuples
        prefix: Optional prefix to add to all keys

    Yields:
        (name, tensor) tuples with converted names
    """
    import re

    for name, tensor in weights:
        # Remove "unet." prefix if present
        if name.startswith("unet."):
            name = name[len("unet.") :]

        # Note: Block index conversion (down_blocks -> input_blocks, up_blocks -> output_blocks)
        # is handled in load_weights() with proper index calculation.
        # Here we only handle layer name conversions (ff.net, mid_block, etc.)

        # Initial conv and output conv (Diffusers -> ComfyUI/Aphrodite format)
        name = name.replace("conv_in.", "input_blocks.0.0.")
        name = name.replace("conv_out.", "out.2.")

        # Time embedding conversion (Diffusers -> ComfyUI/Aphrodite format)
        name = name.replace("time_embedding.linear_1.", "time_embed_0.")
        name = name.replace("time_embedding.linear_2.", "time_embed_2.")

        # ADM/label embedding conversion (Diffusers -> ComfyUI/Aphrodite format)
        name = name.replace("add_embedding.linear_1.", "label_emb_0.")
        name = name.replace("add_embedding.linear_2.", "label_emb_2.")
        name = name.replace("class_embedding.linear_1.", "label_emb_0.")
        name = name.replace("class_embedding.linear_2.", "label_emb_2.")

        # Convert mid_block
        name = name.replace("mid_block.resnets.0.", "middle_block.0.")
        # Handle all transformer blocks in mid_block (not just block 0)
        name = re.sub(
            r"^mid_block\.attentions\.0\.transformer_blocks\.(\d+)\.(.*)$",
            r"middle_block.1.transformer_blocks.\1.\2",
            name,
        )
        name = name.replace("mid_block.resnets.1.", "middle_block.2.")

        # Convert feed-forward layer names: ff.net.0 -> ff_0, ff.net.2 -> ff_2
        name = re.sub(r"\.ff\.net\.0\.", r".ff_0.", name)
        name = re.sub(r"\.ff\.net\.2\.", r".ff_2.", name)

        # Convert ResBlock layer names (Diffusers -> ComfyUI/Aphrodite format)
        # This handles resnets in down_blocks, up_blocks, and mid_block
        name = name.replace(".conv1.", ".in_layers.2.")
        name = name.replace(".norm1.", ".in_layers.0.")
        name = name.replace(".conv2.", ".out_layers.3.")
        name = name.replace(".norm2.", ".out_layers.0.")
        name = name.replace(".time_emb_proj.", ".emb_layers.1.")
        name = name.replace(".conv_shortcut.", ".skip_connection.")

        # Convert norm names: norm1 -> norm1, norm2 -> norm2, norm3 -> norm3 (should already match)
        # But check if there are any other patterns

        # Add custom prefix if provided
        if prefix:
            name = f"{prefix}{name}"

        yield (name, tensor)
