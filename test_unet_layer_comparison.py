#!/usr/bin/env python3
"""
Layer-by-layer comparison script to find where our UNet diverges from Diffusers.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.deps', 'diffusers'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from diffusers import UNet2DConditionModel
from aphrodite.diffusion.runtime.models.unet.unet_model import SDXLUNetModel
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def test_unet_layer_by_layer():
    """Compare UNet outputs layer by layer to find divergence points."""
    
    # SDXL UNet config (matching Diffusers' config.json)
    # Diffusers has 3 down_blocks with block_out_channels: [320, 640, 1280]
    # transformer_layers_per_block: [1, 2, 10] with down_block_types: ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"]
    # Level 0 (DownBlock2D): no attention, Level 1: 2 transformers/block, Level 2: 10 transformers/block
    # With 2 resnets per level, transformer_depth = [0, 0, 2, 2, 10, 10]
    model_channels = 320
    num_res_blocks = [2, 2, 2]  # 3 levels, 2 resnets per level
    channel_mult = (1, 2, 4)  # 3 levels: 320, 640, 1280
    transformer_depth = [0, 0, 2, 2, 10, 10]  # 6 blocks: 2 blocks per level, [0,0] for level 0, [2,2] for level 1, [10,10] for level 2
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
    else:
        print("Checkpoint not found, using random weights")
    
    # Create test inputs
    batch_size = 1
    height, width = 64, 64
    latent_channels = 4
    
    print(f"\nTesting forward pass with inputs: batch={batch_size}, h={height}, w={width}")
    
    # Sample input
    sample = torch.randn(batch_size, latent_channels, height, width, dtype=torch.float32, device=device)
    
    # Timestep
    timestep = torch.tensor([500.0], dtype=torch.float32, device=device)
    
    # Text embeddings
    encoder_hidden_states = torch.randn(batch_size, 77, context_dim, dtype=torch.float32, device=device)
    
    # ADM conditioning
    time_ids = torch.tensor([[64, 64, 0, 0, 64, 64]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    added_cond_kwargs = {
        "text_embeds": torch.randn(batch_size, 1280, dtype=torch.float32, device=device),
        "time_ids": time_ids,
    }
    
    # Hook to capture intermediate outputs from Diffusers UNet
    diffusers_activations = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            diffusers_activations[name] = output.detach().clone()
        return hook
    
    # Register hooks on Diffusers UNet
    diffusers_hooks = []
    
    # Hook the initial conv
    hook = diffusers_unet.conv_in.register_forward_hook(make_hook("conv_in"))
    diffusers_hooks.append(hook)
    
    # Hook input blocks (down_blocks)
    for i, block in enumerate(diffusers_unet.down_blocks):
        hook = block.register_forward_hook(make_hook(f"down_blocks.{i}"))
        diffusers_hooks.append(hook)
    
    # Hook middle block
    hook = diffusers_unet.mid_block.register_forward_hook(make_hook("mid_block"))
    diffusers_hooks.append(hook)
    
    # Hook output blocks (up_blocks)
    for i, block in enumerate(diffusers_unet.up_blocks):
        hook = block.register_forward_hook(make_hook(f"up_blocks.{i}"))
        diffusers_hooks.append(hook)
    
    # Hook the output projection
    hook = diffusers_unet.conv_out.register_forward_hook(make_hook("conv_out"))
    diffusers_hooks.append(hook)
    
    print("\nRunning Diffusers UNet forward pass...")
    with torch.no_grad():
        diffusers_output = diffusers_unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
    
    # Clean up hooks
    for hook in diffusers_hooks:
        hook.remove()
    
    # Now run our UNet and capture intermediate outputs
    print("Running our UNet forward pass with layer-by-layer comparison...")
    
    our_activations = {}
    
    # Hook our UNet
    our_hooks = []
    
    def make_our_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            our_activations[name] = output.detach().clone()
        return hook
    
    # Hook input blocks (all of them individually)
    for i, block in enumerate(our_unet.input_blocks):
        hook = block.register_forward_hook(make_our_hook(f"input_blocks.{i}"))
        our_hooks.append(hook)
    
    # Hook middle block
    hook = our_unet.middle_block.register_forward_hook(make_our_hook("middle_block"))
    our_hooks.append(hook)
    
    # Hook output blocks (all of them individually)
    for i, block in enumerate(our_unet.output_blocks):
        hook = block.register_forward_hook(make_our_hook(f"output_blocks.{i}"))
        our_hooks.append(hook)
    
    # Hook the output projection
    hook = our_unet.out.register_forward_hook(make_our_hook("out"))
    our_hooks.append(hook)
    
    with torch.no_grad():
        # Convert inputs to our format
        adm_cond = torch.cat([
            added_cond_kwargs["text_embeds"],
            added_cond_kwargs["time_ids"].reshape(batch_size, -1)
        ], dim=-1).to(device)
        
        our_output = our_unet(
            sample,
            timestep,
            context=encoder_hidden_states,
            adm_cond=adm_cond,
        )
    
    # Clean up hooks
    for hook in our_hooks:
        hook.remove()
    
    print("\n" + "="*80)
    print("LAYER-BY-LAYER COMPARISON")
    print("="*80)
    
    # Compare initial conv
    print("\n--- INITIAL CONV ---")
    if "conv_in" in diffusers_activations:
        diff_act = diffusers_activations["conv_in"]
        print(f"Diffusers conv_in: {diff_act.shape}")
        print(f"  Stats: min={diff_act.min().item():.4f}, max={diff_act.max().item():.4f}, std={diff_act.std().item():.4f}")
    
    if "input_blocks.0" in our_activations:
        our_act = our_activations["input_blocks.0"]
        print(f"Our input_blocks.0: {our_act.shape}")
        print(f"  Stats: min={our_act.min().item():.4f}, max={our_act.max().item():.4f}, std={our_act.std().item():.4f}")
    
    # Compare input blocks - need to map down_blocks to input_blocks correctly
    print("\n--- INPUT BLOCKS (down_blocks -> input_blocks) ---")
    # Diffusers has 4 down_blocks, but they map to multiple input_blocks each
    # down_blocks.0 maps to input_blocks 1-3 (after initial conv)
    # down_blocks.1 maps to input_blocks 4-6
    # down_blocks.2 maps to input_blocks 7-9
    # down_blocks.3 maps to input_blocks 10-11
    
    mapping = [
        (0, [1, 2, 3]),  # down_blocks.0 -> input_blocks 1-3
        (1, [4, 5, 6]),  # down_blocks.1 -> input_blocks 4-6
        (2, [7, 8, 9]),  # down_blocks.2 -> input_blocks 7-9
        (3, [10, 11]),   # down_blocks.3 -> input_blocks 10-11
    ]
    
    for diff_idx, our_indices in mapping:
        diff_name = f"down_blocks.{diff_idx}"
        if diff_name in diffusers_activations:
            diff_act = diffusers_activations[diff_name]
            print(f"\n{diff_name}: {diff_act.shape}")
            print(f"  Diffusers: min={diff_act.min().item():.4f}, max={diff_act.max().item():.4f}, std={diff_act.std().item():.4f}")
            print(f"  Maps to input_blocks {our_indices}:")
            
            # Compare with corresponding input blocks
            for our_idx in our_indices:
                our_name = f"input_blocks.{our_idx}"
                if our_name in our_activations:
                    our_act = our_activations[our_name]
                    print(f"    {our_name}: {our_act.shape}, min={our_act.min().item():.4f}, max={our_act.max().item():.4f}, std={our_act.std().item():.4f}")
                    
                    # If shapes match, compute difference
                    if diff_act.shape == our_act.shape:
                        diff = (diff_act - our_act).abs()
                        max_diff = diff.max().item()
                        mean_diff = diff.mean().item()
                        cosine_sim = torch.nn.functional.cosine_similarity(
                            diff_act.flatten(), our_act.flatten(), dim=0
                        ).item()
                        print(f"      Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}, Cosine sim: {cosine_sim:.6f}")
                        if max_diff > 0.1:
                            print(f"      ⚠️  LARGE DIFFERENCE!")
    
    # Compare middle block
    print("\n--- MIDDLE BLOCK ---")
    if "mid_block" in diffusers_activations and "middle_block" in our_activations:
        diff_act = diffusers_activations["mid_block"]
        our_act = our_activations["middle_block"]
        
        if diff_act.shape == our_act.shape:
            diff = (diff_act - our_act).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            cosine_sim = torch.nn.functional.cosine_similarity(
                diff_act.flatten(), our_act.flatten(), dim=0
            ).item()
            
            print(f"\nmid_block / middle_block:")
            print(f"  Shape: {diff_act.shape}")
            print(f"  Diffusers: min={diff_act.min().item():.4f}, max={diff_act.max().item():.4f}, std={diff_act.std().item():.4f}")
            print(f"  Ours:      min={our_act.min().item():.4f}, max={our_act.max().item():.4f}, std={our_act.std().item():.4f}")
            print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}, Cosine sim: {cosine_sim:.6f}")
            
            if max_diff > 0.1:
                print(f"  ⚠️  LARGE DIFFERENCE DETECTED!")
        else:
            print(f"\nmid_block / middle_block: SHAPE MISMATCH!")
            print(f"  Diffusers: {diff_act.shape}, Ours: {our_act.shape}")
    
    # Compare output blocks
    print("\n--- OUTPUT BLOCKS ---")
    for i in range(len(diffusers_unet.up_blocks)):
        diff_name = f"up_blocks.{i}"
        # Map up_blocks index to output_blocks index using ComfyUI's formula
        num_res_blocks_rev = list(reversed(num_res_blocks))
        start_idx = (num_res_blocks_rev[i] + 1) * i
        num_blocks = num_res_blocks_rev[i] + 1
        
        if diff_name in diffusers_activations:
            diff_act = diffusers_activations[diff_name]
            
            # Our output blocks are individual blocks, not grouped like Diffusers
            # So we need to compare with the corresponding range
            print(f"\n{diff_name}:")
            print(f"  Diffusers shape: {diff_act.shape}")
            print(f"  Maps to output_blocks {start_idx}-{start_idx+num_blocks-1}")
            
            # Compare with each corresponding output block
            for j in range(num_blocks):
                our_idx = start_idx + j
                our_name = f"output_blocks.{our_idx}"
                
                if our_name in our_activations:
                    our_act = our_activations[our_name]
                    
                    # Note: Diffusers up_blocks output might have different shape
                    # than individual output_blocks, so we compare stats
                    print(f"    {our_name}:")
                    print(f"      Shape: {our_act.shape}")
                    print(f"      Ours: min={our_act.min().item():.4f}, max={our_act.max().item():.4f}, std={our_act.std().item():.4f}")
    
    # Final output comparison
    print("\n--- FINAL OUTPUT ---")
    if diffusers_output.shape == our_output.shape:
        diff = (diffusers_output - our_output).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            diffusers_output.flatten(), our_output.flatten(), dim=0
        ).item()
        
        print(f"\nFinal output:")
        print(f"  Shape: {diffusers_output.shape}")
        print(f"  Diffusers: min={diffusers_output.min().item():.4f}, max={diffusers_output.max().item():.4f}, std={diffusers_output.std().item():.4f}")
        print(f"  Ours:      min={our_output.min().item():.4f}, max={our_output.max().item():.4f}, std={our_output.std().item():.4f}")
        print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}, Cosine sim: {cosine_sim:.6f}")
    else:
        print(f"\nFinal output: SHAPE MISMATCH!")
        print(f"  Diffusers: {diffusers_output.shape}, Ours: {our_output.shape}")

if __name__ == "__main__":
    test_unet_layer_by_layer()

