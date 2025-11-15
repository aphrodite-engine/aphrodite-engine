#!/usr/bin/env python3
"""Test if resblock weights are loading correctly."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".deps", "diffusers"))
sys.path.insert(0, os.path.dirname(__file__))

import logging

import torch
from diffusers import UNet2DConditionModel

from aphrodite.diffusion.runtime.models.unet.unet_model import SDXLUNetModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# SDXL UNet config
model_channels = 320
num_res_blocks = [2, 2, 2]
channel_mult = (1, 2, 4)
transformer_depth = [0, 0, 2, 2, 10, 10]
context_dim = 2048
adm_in_channels = 2816

diffusers_unet = UNet2DConditionModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="unet",
    torch_dtype=torch.float32,
)
diffusers_unet = diffusers_unet.to(device)
diffusers_unet.eval()

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

# Load weights
checkpoint_path = os.path.expanduser(
    "~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/"
    "snapshots/462165984030d82259a11f4367a4eed129e94a7b/unet/diffusion_pytorch_model.safetensors"
)
from safetensors.torch import load_file

weights = load_file(checkpoint_path)
weights_list = [(k, v) for k, v in weights.items()]
loaded = our_unet.load_weights(weights_list)
print(f"Loaded {len(loaded)} weights\n")

# Compare first resblock weights: down_blocks.0.resnets.0 -> input_blocks.1.0
print("=" * 80)
print("Comparing down_blocks.0.resnets.0 (Diffusers) vs input_blocks.1.0 (Ours)")
print("=" * 80)

# Get weight from Diffusers
diff_resblock = diffusers_unet.down_blocks[0].resnets[0]
diff_weight = diff_resblock.conv1.weight
print("\nDiffusers down_blocks.0.resnets.0.conv1.weight:")
print(f"  Shape: {diff_weight.shape}")
print(f"  min={diff_weight.min().item():.6f}, max={diff_weight.max().item():.6f}, mean={diff_weight.mean().item():.6f}")

# Get weight from our model
our_resblock = our_unet.input_blocks[1][0]  # input_blocks.1.0
our_weight = our_resblock.in_layers[2].weight  # The Conv2d is the 3rd layer (after GroupNorm and SiLU)
print("\nOur input_blocks.1.0.in_layers.2.weight:")
print(f"  Shape: {our_weight.shape}")
print(f"  min={our_weight.min().item():.6f}, max={our_weight.max().item():.6f}, mean={our_weight.mean().item():.6f}")

# Compare
if diff_weight.shape == our_weight.shape:
    weight_diff = (diff_weight - our_weight).abs()
    print("\nWeight difference:")
    print(f"  max_diff={weight_diff.max().item():.6e}, mean_diff={weight_diff.mean().item():.6e}")

    if weight_diff.max().item() < 1e-5:
        print("  ✅ Weights match!")
    else:
        print("  ⚠️  Weights do NOT match!")

        # Check what the weight should be named in our model
        print("\n  Checking weight names in our model (input_blocks.1):")
        for name, param in our_unet.named_parameters():
            if "input_blocks.1.0" in name and "weight" in name:
                print(f"    {name}: {param.shape}")

        # Check what we loaded for this block
        print("\n  Checking what weights were loaded for input_blocks.1:")
        for name in sorted(loaded):
            if "input_blocks.1" in name:
                print(f"    {name}")
else:
    print(f"  ⚠️  Shape mismatch: {diff_weight.shape} vs {our_weight.shape}")
