# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# Adapted from ComfyUI: comfy/supported_models.py, comfy/utils.py

# SPDX-License-Identifier: Apache-2.0
"""
Weight loading utilities for SDXL dual CLIP.

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


def clip_text_transformers_convert(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    text_model_prefix: str,
) -> dict[str, torch.Tensor]:
    """
    Convert CLIP text transformer state dict format.

    Converts from diffusers format to Aphrodite format.

    Args:
        state_dict: Original state dict
        prefix: Prefix for the CLIP model (e.g., "clip_g.")
        text_model_prefix: Prefix for text_model (e.g., "clip_g.transformer.")

    Returns:
        Converted state dict
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # Convert text_model.embeddings.* to transformer.embeddings.*
        if key.startswith(f"{prefix}text_model.embeddings"):
            new_key = key.replace(f"{prefix}text_model.embeddings", f"{text_model_prefix}embeddings", 1)

        # Convert text_model.encoder.* to transformer.encoder.*
        elif key.startswith(f"{prefix}text_model.encoder"):
            new_key = key.replace(f"{prefix}text_model.encoder", f"{text_model_prefix}encoder", 1)

        # Convert text_model.final_layer_norm to transformer.final_layer_norm
        elif key.startswith(f"{prefix}text_model.final_layer_norm"):
            new_key = key.replace(f"{prefix}text_model.final_layer_norm", f"{text_model_prefix}final_layer_norm", 1)

        # Keep other keys as-is
        new_state_dict[new_key] = value

    return new_state_dict


def convert_sdxl_clip_state_dict(
    state_dict: dict[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """
    Convert SDXL dual CLIP state dict from diffusers format to Aphrodite format.

    Diffusers format:
    - conditioner.embedders.0.transformer.text_model.* -> clip_l.transformer.text_model.*
    - conditioner.embedders.1.model.* -> clip_g.*

    Aphrodite format:
    - clip_l.transformer.*
    - clip_g.transformer.*

    Args:
        state_dict: State dict from diffusers or iterable of (name, tensor) tuples

    Returns:
        Converted state dict
    """
    # Convert iterable to dict if needed
    if isinstance(state_dict, Iterable) and not isinstance(state_dict, dict):
        state_dict = dict(state_dict)

    # Check if this is a single text_encoder component (text_model.*) or full pipeline (conditioner.embedders.*)
    has_conditioner = any(key.startswith("conditioner.embedders") for key in state_dict.keys())
    has_text_encoder_2 = any(key.startswith("text_encoder_2.") for key in state_dict.keys())

    if has_conditioner:
        # Full pipeline format: conditioner.embedders.*
        replace_prefix = {
            "conditioner.embedders.0.transformer.text_model": "clip_l.transformer.text_model",
            "conditioner.embedders.1.model.": "clip_g.",
        }
        converted = state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)
        # Step 2: Convert CLIP-G text transformer format
        converted = clip_text_transformers_convert(converted, "clip_g.", "clip_g.transformer.")
    elif has_text_encoder_2:
        # Loading from both text_encoder/ and text_encoder_2/ directories
        # text_model.* -> clip_l.text_model.* (from text_encoder/)
        # text_encoder_2.text_model.* -> clip_g.text_model.* (from text_encoder_2/)
        replace_prefix = {
            "text_model.": "clip_l.text_model.",
            "text_encoder_2.text_model.": "clip_g.text_model.",
        }
        converted = state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False)
    else:
        # Single text_encoder component format: text_model.*
        # For SDXL, the text_encoder directory contains CLIP-L weights
        # We need to map text_model.* to clip_l.text_model.*
        replace_prefix = {
            "text_model.": "clip_l.text_model.",
        }
        converted = state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False)

    return converted


def process_sdxl_clip_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """
    Process SDXL dual CLIP weights for loading.

    Converts diffusers format weights to Aphrodite format.
    If weights already have clip_l. or clip_g. prefixes, they are passed through.

    Args:
        weights: Iterable of (name, tensor) tuples

    Yields:
        (name, tensor) tuples with converted names
    """
    # Check if weights are already in Aphrodite format
    weights_list = list(weights)
    already_converted = any(name.startswith("clip_l.") or name.startswith("clip_g.") for name, _ in weights_list)

    if already_converted:
        # Already in correct format, just yield as-is
        for name, tensor in weights_list:
            yield (name, tensor)
    else:
        # Convert from diffusers format
        state_dict = dict(weights_list)
        converted = convert_sdxl_clip_state_dict(state_dict)

        for name, tensor in converted.items():
            yield (name, tensor)
