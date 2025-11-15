#!/usr/bin/env python3
"""
Test script to compare our UNet implementation with Diffusers' UNet.
This will help identify where our implementation differs.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".deps", "diffusers"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from diffusers import UNet2DConditionModel

from aphrodite.diffusion.runtime.models.unet.unet_model import SDXLUNetModel

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def test_unet_forward():
    """Test forward pass outputs match between Diffusers and our implementation."""

    # SDXL UNet config
    model_channels = 320
    num_res_blocks = [2, 2, 2, 2]
    channel_mult = (1, 2, 4, 4)
    transformer_depth = [0, 0, 2, 2, 10, 10]
    context_dim = 2048
    adm_in_channels = 2816

    print("Creating Diffusers UNet...")
    diffusers_unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="unet",
        torch_dtype=torch.float32,
    )
    diffusers_unet = diffusers_unet.to(device)
    diffusers_unet.eval()

    print("Creating our UNet...")
    our_unet = SDXLUNetModel(
        in_channels=4,
        out_channels=4,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks,
        channel_mult=channel_mult,
        transformer_depth=transformer_depth,
        context_dim=context_dim,
        adm_in_channels=adm_in_channels,
        num_heads=8,
    )
    our_unet = our_unet.to(device)
    our_unet.eval()

    # Load weights into our UNet
    print("Loading weights into our UNet...")
    checkpoint_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/"
        "snapshots/462165984030d82259a11f4367a4eed129e94a7b/unet/diffusion_pytorch_model.safetensors"
    )
    if os.path.exists(checkpoint_path):
        from safetensors.torch import load_file

        weights = load_file(checkpoint_path)
        # Convert to our format
        weights_list = [(k, v) for k, v in weights.items()]
        loaded = our_unet.load_weights(weights_list)
        print(f"Loaded {len(loaded)} weights")
    else:
        print("Checkpoint not found, using random weights")

    # Create test inputs
    batch_size = 1
    height, width = 64, 64  # Small size for testing
    latent_channels = 4

    print(f"\nTesting forward pass with inputs: batch={batch_size}, h={height}, w={width}")

    # Sample input
    sample = torch.randn(batch_size, latent_channels, height, width, dtype=torch.float32, device=device)

    # Timestep (SDXL uses timesteps in range [0, 1000])
    timestep = torch.tensor([500.0], dtype=torch.float32, device=device)

    # Text embeddings (SDXL uses two text encoders)
    encoder_hidden_states = torch.randn(batch_size, 77, context_dim, dtype=torch.float32, device=device)

    # Time embeddings (added_cond_kwargs for SDXL)
    # SDXL time_ids format: [original_size, crops_coords_top_left, target_size]
    # For 64x64 input: original_size=[64, 64], crops_coords_top_left=[0, 0], target_size=[64, 64]
    time_ids = torch.tensor([[64, 64, 0, 0, 64, 64]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    added_cond_kwargs = {
        "text_embeds": torch.randn(batch_size, 1280, dtype=torch.float32, device=device),
        "time_ids": time_ids,
    }

    print("\nRunning Diffusers UNet forward pass...")
    with torch.no_grad():
        diffusers_output = diffusers_unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

    print("Running our UNet forward pass...")
    with torch.no_grad():
        # Convert inputs to our format
        # Our UNet expects: (batch, channels, height, width), timestep, context, adm_cond
        adm_cond = torch.cat(
            [added_cond_kwargs["text_embeds"], added_cond_kwargs["time_ids"].reshape(batch_size, -1)], dim=-1
        )

        our_output = our_unet(
            sample,
            timestep,
            context=encoder_hidden_states,
            adm_cond=adm_cond,
        )

    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    print(f"\nDiffusers output shape: {diffusers_output.shape}")
    print(f"Our output shape: {our_output.shape}")

    if diffusers_output.shape != our_output.shape:
        print("❌ SHAPE MISMATCH!")
        return False

    print("\nDiffusers output stats:")
    print(f"  min: {diffusers_output.min().item():.6f}")
    print(f"  max: {diffusers_output.max().item():.6f}")
    print(f"  mean: {diffusers_output.mean().item():.6f}")
    print(f"  std: {diffusers_output.std().item():.6f}")
    print(f"  has_nan: {torch.isnan(diffusers_output).any().item()}")
    print(f"  has_inf: {torch.isinf(diffusers_output).any().item()}")

    print("\nOur output stats:")
    print(f"  min: {our_output.min().item():.6f}")
    print(f"  max: {our_output.max().item():.6f}")
    print(f"  mean: {our_output.mean().item():.6f}")
    print(f"  std: {our_output.std().item():.6f}")
    print(f"  has_nan: {torch.isnan(our_output).any().item()}")
    print(f"  has_inf: {torch.isinf(our_output).any().item()}")

    # Compute differences
    diff = (diffusers_output - our_output).abs()
    print("\nDifference stats:")
    print(f"  max_abs_diff: {diff.max().item():.6f}")
    print(f"  mean_abs_diff: {diff.mean().item():.6f}")
    print(f"  relative_error: {(diff / (diffusers_output.abs() + 1e-8)).max().item():.6f}")

    # Check if outputs are close
    atol = 1e-2
    rtol = 1e-2
    is_close = torch.allclose(diffusers_output, our_output, atol=atol, rtol=rtol)

    if is_close:
        print(f"\n✅ Outputs match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n❌ Outputs do NOT match within tolerance (atol={atol}, rtol={rtol})")

        # Find where differences are largest
        max_diff_idx = diff.argmax()
        max_diff_pos = np.unravel_index(max_diff_idx.item(), diff.shape)
        print(f"\nLargest difference at position {max_diff_pos}:")
        print(f"  Diffusers: {diffusers_output.flatten()[max_diff_idx].item():.6f}")
        print(f"  Ours: {our_output.flatten()[max_diff_idx].item():.6f}")
        print(f"  Diff: {diff.flatten()[max_diff_idx].item():.6f}")

    return is_close


if __name__ == "__main__":
    success = test_unet_forward()
    sys.exit(0 if success else 1)
