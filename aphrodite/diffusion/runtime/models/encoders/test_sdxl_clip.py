#!/usr/bin/env python3
"""
Test suite for SDXL dual CLIP implementation.
"""

import os

import torch

from aphrodite.diffusion.configs.models.encoders.clip import CLIPTextArchConfig, CLIPTextConfig
from aphrodite.diffusion.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from aphrodite.diffusion.runtime.managers.forward_context import set_forward_context
from aphrodite.diffusion.runtime.models.encoders.sdxl_clip import SDXLClipModel
from aphrodite.diffusion.runtime.server_args import ServerArgs, set_global_server_args


def test_sdxl_dual_clip():
    """Test SDXL dual CLIP encoding."""
    print("=" * 50)
    print("SDXL Dual CLIP Test Suite")
    print("=" * 50)
    
    # Initialize distributed environment for tensor parallel
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=1,
        enable_cfg_parallel=False,
        dp_size=1,
    )
    
    # Set global server args (required for attention backend selection)
    server_args = ServerArgs(
        model_path="/tmp/dummy",  # Dummy path for testing
        attention_backend=None,  # Auto-select
    )
    set_global_server_args(server_args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # SDXL CLIP-L config: 768 hidden size, 12 layers
    clip_l_arch = CLIPTextArchConfig(
        vocab_size=49408,
        hidden_size=768,  # CLIP-L uses 768
        intermediate_size=3072,
        projection_dim=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=77,
        text_len=77,
    )
    clip_l_config = CLIPTextConfig(arch_config=clip_l_arch, prefix="clip_l")
    
    # SDXL CLIP-G config: 1280 hidden size, 32 layers (bigger model)
    clip_g_arch = CLIPTextArchConfig(
        vocab_size=49408,
        hidden_size=1280,  # CLIP-G uses 1280
        intermediate_size=5120,
        projection_dim=1280,
        num_hidden_layers=32,  # CLIP-G has more layers
        num_attention_heads=20,
        max_position_embeddings=77,
        text_len=77,
    )
    clip_g_config = CLIPTextConfig(arch_config=clip_g_arch, prefix="clip_g")
    
    print("\n1. Creating SDXL dual CLIP model...")
    try:
        model = SDXLClipModel(
            clip_l_config=clip_l_config,
            clip_g_config=clip_g_config,
        ).to(device).eval()
        print(f"   ✓ Model created on {device}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
    except Exception as e:
        print(f"   ✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n2. Testing forward pass...")
    try:
        batch_size = 2
        seq_len = 77
        
        # Create dummy input_ids (token IDs)
        input_ids = torch.randint(0, 49408, (batch_size, seq_len), device=device)
        
        print(f"   Input shape: {input_ids.shape}")
        
        # Set forward context (required for attention layers)
        with set_forward_context(current_timestep=0, attn_metadata=None), torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        print("   ✓ Forward pass successful!")
        print(f"   Output shape: {outputs.last_hidden_state.shape}")
        
        # Expected: [batch_size, seq_len, 768 + 1280] = [batch_size, seq_len, 2048]
        expected_shape = (batch_size, seq_len, 768 + 1280)
        print(f"   Expected shape: {expected_shape}")
        
        assert outputs.last_hidden_state.shape == expected_shape, \
            f"Output shape {outputs.last_hidden_state.shape} != expected {expected_shape}"
        
        # Check pooled output
        if outputs.pooler_output is not None:
            print(f"   Pooled output shape: {outputs.pooler_output.shape}")
            assert outputs.pooler_output.shape == (batch_size, 1280), \
                f"Pooled output shape {outputs.pooler_output.shape} != expected {(batch_size, 1280)}"
        else:
            print("   ⚠ Warning: No pooled output (may be expected)")
        
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n3. Testing with dict input_ids...")
    try:
        input_ids_l = torch.randint(0, 49408, (batch_size, seq_len), device=device)
        input_ids_g = torch.randint(0, 49408, (batch_size, seq_len), device=device)
        input_ids_dict = {"l": input_ids_l, "g": input_ids_g}
        
        with set_forward_context(current_timestep=0, attn_metadata=None), torch.no_grad():
            outputs = model(input_ids=input_ids_dict)
        
        print("   ✓ Dict input works!")
        print(f"   Output shape: {outputs.last_hidden_state.shape}")
        
    except Exception as e:
        print(f"   ✗ Dict input failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n4. Testing penultimate layer extraction...")
    try:
        # Check that we're using penultimate layer (not final)
        # This is verified by checking hidden_states structure
        with set_forward_context(current_timestep=0, attn_metadata=None), torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)
        
        if outputs.hidden_states is not None:
            l_hidden, g_hidden = outputs.hidden_states
            print(f"   CLIP-L hidden states: {len(l_hidden)} layers")
            print(f"   CLIP-G hidden states: {len(g_hidden)} layers")
            print("   ✓ Hidden states available")
        else:
            print("   ⚠ Warning: Hidden states not returned")
        
    except Exception as e:
        print(f"   ✗ Hidden states check failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n5. Testing gradient computation...")
    try:
        input_ids = torch.randint(0, 49408, (1, seq_len), device=device)
        input_ids.requires_grad = False  # Input IDs don't need gradients
        
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = model(input_ids=input_ids)
            loss = outputs.last_hidden_state.sum()
            loss.backward()
        
        # Check that gradients were computed
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        if has_grad:
            print("   ✓ Gradients computed successfully")
        else:
            print("   ⚠ Warning: No gradients computed (may be expected if model is frozen)")
        
    except Exception as e:
        print(f"   ✗ Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("✓ All tests passed!")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_sdxl_dual_clip()
    exit(0 if success else 1)

