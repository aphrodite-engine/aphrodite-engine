#!/usr/bin/env python3
"""Test just the initial conv to see if weights are loading correctly."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.deps', 'diffusers'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from diffusers import UNet2DConditionModel
from aphrodite.diffusion.runtime.models.unet.unet_model import SDXLUNetModel
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# SDXL UNet config
model_channels = 320
num_res_blocks = [2, 2, 2]
channel_mult = (1, 2, 4)
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
    weights_list = [(k, v) for k, v in weights.items()]
    loaded = our_unet.load_weights(weights_list)
    print(f"Loaded {len(loaded)} weights")

# Test just the initial conv
print("\n" + "="*80)
print("Testing initial conv")
print("="*80)

sample = torch.randn(1, 4, 64, 64, dtype=torch.float32, device=device)

with torch.no_grad():
    # Diffusers
    diff_conv_out = diffusers_unet.conv_in(sample)
    
    # Ours
    our_conv_out = our_unet.input_blocks[0][0](sample)  # First layer of first block

print(f"\nDiffusers conv_in output:")
print(f"  Shape: {diff_conv_out.shape}")
print(f"  min={diff_conv_out.min().item():.4f}, max={diff_conv_out.max().item():.4f}, std={diff_conv_out.std().item():.4f}")

print(f"\nOur input_blocks.0 output:")
print(f"  Shape: {our_conv_out.shape}")
print(f"  min={our_conv_out.min().item():.4f}, max={our_conv_out.max().item():.4f}, std={our_conv_out.std().item():.4f}")

# Compare
diff = (diff_conv_out - our_conv_out).abs()
print(f"\nDifference:")
print(f"  max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}")
print(f"  cosine_sim={torch.nn.functional.cosine_similarity(diff_conv_out.flatten(), our_conv_out.flatten(), dim=0).item():.6f}")

if diff.max().item() < 0.01:
    print("\n✅ Initial conv weights match!")
else:
    print("\n⚠️  Initial conv weights do NOT match!")
    
    # Check if the weights themselves match
    diff_weight = diffusers_unet.conv_in.weight
    our_weight = our_unet.input_blocks[0][0].weight
    
    print(f"\nWeight comparison:")
    print(f"  Diffusers weight: {diff_weight.shape}")
    print(f"  Our weight: {our_weight.shape}")
    
    if diff_weight.shape == our_weight.shape:
        weight_diff = (diff_weight - our_weight).abs()
        print(f"  Weight max_diff={weight_diff.max().item():.6e}, mean_diff={weight_diff.mean().item():.6e}")
        
        if weight_diff.max().item() < 1e-5:
            print("  ✅ Weights match! The issue is in the forward pass.")
        else:
            print("  ⚠️  Weights do NOT match! The issue is in weight loading.")
    else:
        print("  ⚠️  Weight shapes don't match!")



