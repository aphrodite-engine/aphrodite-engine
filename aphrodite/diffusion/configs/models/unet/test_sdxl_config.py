#!/usr/bin/env python3
"""
Test suite for SDXL UNet config classes.
Test-driven development: tests first, then implementation.
"""

import os

import torch

from aphrodite.diffusion.configs.models.unet.sdxl import SDXLUNetArchConfig, SDXLUNetConfig
from aphrodite.diffusion.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from aphrodite.diffusion.runtime.managers.forward_context import set_forward_context
from aphrodite.diffusion.runtime.models.unet.unet_model import SDXLUNetModel
from aphrodite.diffusion.runtime.server_args import ServerArgs, set_global_server_args


def test_sdxl_unet_config():
    """Test SDXL UNet config creation and model instantiation."""
    print("=" * 50)
    print("SDXL UNet Config Test Suite")
    print("=" * 50)

    # Initialize distributed environment
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12356")
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

    print("\n1. Testing SDXL UNet config creation...")
    try:
        # Test arch config
        arch_config = SDXLUNetArchConfig(
            model_channels=320,
            transformer_depth=[0, 0, 2, 2, 10, 10],
            context_dim=2048,
            adm_in_channels=2816,
            use_linear_in_transformer=True,
        )

        # Test full config
        unet_config = SDXLUNetConfig(arch_config=arch_config)

        print("   ✓ Config creation successful!")
        print(f"   Model channels: {unet_config.model_channels}")
        print(f"   Context dim: {unet_config.context_dim}")
        print(f"   ADM in channels: {unet_config.adm_in_channels}")
        print(f"   Transformer depth: {unet_config.transformer_depth}")

        # Verify values match ComfyUI's SDXL config
        assert unet_config.model_channels == 320
        assert unet_config.context_dim == 2048
        assert unet_config.adm_in_channels == 2816
        assert unet_config.transformer_depth == [0, 0, 2, 2, 10, 10]
        assert unet_config.use_linear_in_transformer is True

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n2. Testing model instantiation from config...")
    try:
        # Create model from config
        model = (
            SDXLUNetModel(
                in_channels=4,
                out_channels=4,
                model_channels=unet_config.model_channels,
                num_res_blocks=2,
                channel_mult=(1, 2, 4, 4),
                context_dim=unet_config.context_dim,
                transformer_depth=unet_config.transformer_depth,
                num_heads=8,
                num_head_channels=64,
                adm_in_channels=unet_config.adm_in_channels,
                use_linear_in_transformer=unet_config.use_linear_in_transformer,
            )
            .to(device)
            .eval()
        )

        print("   ✓ Model instantiation successful!")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n3. Testing forward pass with config values...")
    try:
        batch_size = 1
        height, width = 64, 64  # Latent dimensions

        x = torch.randn(batch_size, 4, height, width, device=device)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)
        context = torch.randn(batch_size, 77, unet_config.context_dim, device=device)
        y = torch.randn(batch_size, unet_config.adm_in_channels, device=device)

        with set_forward_context(current_timestep=0, attn_metadata=None), torch.no_grad():
            output = model(x, timesteps, context=context, y=y)

        print("   ✓ Forward pass successful!")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        assert output.shape == x.shape

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n4. Testing config defaults...")
    try:
        # Test with minimal config (should use defaults)
        default_config = SDXLUNetConfig()

        print("   ✓ Default config creation successful!")
        print(f"   Default model channels: {default_config.model_channels}")
        print(f"   Default context dim: {default_config.context_dim}")

        # Verify defaults match SDXL base model
        assert default_config.model_channels == 320
        assert default_config.context_dim == 2048
        assert default_config.adm_in_channels == 2816

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
    success = test_sdxl_unet_config()
    exit(0 if success else 1)
