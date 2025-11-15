#!/usr/bin/env python3
"""
Test suite for SDXL dual CLIP config classes.
Test-driven development: tests first, then implementation.
"""

import os

import torch

from aphrodite.diffusion.configs.models.encoders.sdxl_clip import (
    SDXLClipConfig,
    SDXLClipGArchConfig,
    SDXLClipLArchConfig,
)
from aphrodite.diffusion.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from aphrodite.diffusion.runtime.managers.forward_context import set_forward_context
from aphrodite.diffusion.runtime.models.encoders.sdxl_clip import SDXLClipModel
from aphrodite.diffusion.runtime.server_args import ServerArgs, set_global_server_args


def test_sdxl_clip_config():
    """Test SDXL dual CLIP config creation and model instantiation."""
    print("=" * 50)
    print("SDXL Dual CLIP Config Test Suite")
    print("=" * 50)

    # Initialize distributed environment
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12357")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=1,
        enable_cfg_parallel=False,
        dp_size=1,
    )

    # Set global server args
    server_args = ServerArgs(
        model_path="/tmp/dummy",
        attention_backend=None,
    )
    set_global_server_args(server_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n1. Testing SDXL CLIP-L config creation...")
    try:
        clip_l_config = SDXLClipLArchConfig(
            vocab_size=49408,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=77,
        )

        print("   ✓ CLIP-L config creation successful!")
        print(f"   Hidden size: {clip_l_config.hidden_size}")
        print(f"   Num layers: {clip_l_config.num_hidden_layers}")
        assert clip_l_config.hidden_size == 768
        assert clip_l_config.num_hidden_layers == 12

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n2. Testing SDXL CLIP-G config creation...")
    try:
        clip_g_config = SDXLClipGArchConfig(
            vocab_size=49408,
            hidden_size=1280,
            intermediate_size=5120,
            num_hidden_layers=32,
            num_attention_heads=20,
            max_position_embeddings=77,
        )

        print("   ✓ CLIP-G config creation successful!")
        print(f"   Hidden size: {clip_g_config.hidden_size}")
        print(f"   Num layers: {clip_g_config.num_hidden_layers}")
        assert clip_g_config.hidden_size == 1280
        assert clip_g_config.num_hidden_layers == 32

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n3. Testing SDXL dual CLIP config creation...")
    try:
        dual_clip_config = SDXLClipConfig(
            clip_l_config=SDXLClipLArchConfig(),
            clip_g_config=SDXLClipGArchConfig(),
        )

        print("   ✓ Dual CLIP config creation successful!")
        print(f"   CLIP-L hidden size: {dual_clip_config.clip_l_config.hidden_size}")
        print(f"   CLIP-G hidden size: {dual_clip_config.clip_g_config.hidden_size}")
        assert dual_clip_config.clip_l_config.hidden_size == 768
        assert dual_clip_config.clip_g_config.hidden_size == 1280

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n4. Testing model instantiation from config...")
    try:
        # Create configs
        from aphrodite.diffusion.configs.models.encoders.clip import CLIPTextConfig

        clip_l_config = CLIPTextConfig()
        clip_l_config.hidden_size = 768
        clip_l_config.num_hidden_layers = 12
        clip_l_config.num_attention_heads = 12
        clip_l_config.intermediate_size = 3072

        clip_g_config = CLIPTextConfig()
        clip_g_config.hidden_size = 1280
        clip_g_config.num_hidden_layers = 32
        clip_g_config.num_attention_heads = 20
        clip_g_config.intermediate_size = 5120

        # Create model with configs
        model = (
            SDXLClipModel(
                clip_l_config=clip_l_config,
                clip_g_config=clip_g_config,
            )
            .to(device)
            .eval()
        )

        print("   ✓ Model instantiation successful!")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Verify model structure
        assert hasattr(model, "clip_l")
        assert hasattr(model, "clip_g")
        assert model.clip_l.config.hidden_size == 768
        assert model.clip_g.config.hidden_size == 1280

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n5. Testing forward pass with config values...")
    try:
        batch_size = 1
        seq_len = 77

        # Create input_ids for both CLIP-L and CLIP-G
        input_ids = torch.randint(0, 49408, (batch_size, seq_len), device=device)

        with set_forward_context(current_timestep=0, attn_metadata=None), torch.no_grad():
            outputs = model(input_ids=input_ids)

        print("   ✓ Forward pass successful!")
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Output shape: {outputs.last_hidden_state.shape}")
        print(f"   Pooled output shape: {outputs.pooler_output.shape}")

        # Verify output dimensions
        # CLIP-L (768) + CLIP-G (1280) = 2048
        assert outputs.last_hidden_state.shape == (batch_size, seq_len, 2048)
        assert outputs.pooler_output.shape == (batch_size, 1280)  # From CLIP-G

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n6. Testing config defaults...")
    try:
        # Test with minimal config (should use defaults)
        default_config = SDXLClipConfig()

        print("   ✓ Default config creation successful!")
        print(f"   Default CLIP-L hidden size: {default_config.clip_l_config.hidden_size}")
        print(f"   Default CLIP-G hidden size: {default_config.clip_g_config.hidden_size}")

        # Verify defaults match SDXL specifications
        assert default_config.clip_l_config.hidden_size == 768
        assert default_config.clip_g_config.hidden_size == 1280

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 50)
    print("✓ All tests passed!")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_sdxl_clip_config()
    exit(0 if success else 1)
