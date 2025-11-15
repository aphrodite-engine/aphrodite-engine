#!/usr/bin/env python3
"""
Test suite for SDXL pipeline config.
Test-driven development: tests first, then implementation.
"""

from aphrodite.diffusion.configs.models import VAEConfig
from aphrodite.diffusion.configs.models.encoders import SDXLClipConfig
from aphrodite.diffusion.configs.models.unet import SDXLUNetConfig
from aphrodite.diffusion.configs.pipelines.sdxl import SDXLPipelineConfig


def test_sdxl_pipeline_config():
    """Test SDXL pipeline config creation and validation."""
    print("=" * 50)
    print("SDXL Pipeline Config Test Suite")
    print("=" * 50)

    print("\n1. Testing SDXL pipeline config creation...")
    try:
        config = SDXLPipelineConfig()

        print("   ✓ Config creation successful!")
        print(f"   Is image gen: {config.is_image_gen}")
        print(f"   Embedded CFG scale: {config.embedded_cfg_scale}")
        print(f"   UNet config type: {type(config.unet_config).__name__}")
        print(f"   Text encoder configs: {len(config.text_encoder_configs)}")
        print(f"   VAE config type: {type(config.vae_config).__name__}")

        # Verify defaults
        assert config.is_image_gen is True
        assert config.embedded_cfg_scale == 6.0  # SDXL default
        assert isinstance(config.unet_config, SDXLUNetConfig)
        assert len(config.text_encoder_configs) == 1  # Single dual CLIP
        assert isinstance(config.text_encoder_configs[0], SDXLClipConfig)

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n2. Testing config with custom values...")
    try:
        # Create custom configs
        unet_config = SDXLUNetConfig()
        clip_config = SDXLClipConfig()
        vae_config = VAEConfig()

        config = SDXLPipelineConfig(
            is_image_gen=True,
            embedded_cfg_scale=7.5,
            unet_config=unet_config,
            text_encoder_configs=(clip_config,),
            vae_config=vae_config,
        )

        print("   ✓ Custom config creation successful!")
        assert config.embedded_cfg_scale == 7.5
        assert config.unet_config == unet_config
        assert config.text_encoder_configs[0] == clip_config

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n3. Testing config validation...")
    try:
        config = SDXLPipelineConfig()

        # Check that text encoder configs match expected structure
        assert len(config.text_encoder_configs) == 1
        assert isinstance(config.text_encoder_configs[0], SDXLClipConfig)

        # Check UNet config has correct ADM channels
        assert config.unet_config.adm_in_channels == 2816

        # Check text encoder precisions
        assert len(config.text_encoder_precisions) == 1

        print("   ✓ Config validation successful!")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n4. Testing preprocess/postprocess functions...")
    try:
        config = SDXLPipelineConfig()

        # Check that functions are defined
        assert len(config.preprocess_text_funcs) == 1
        assert len(config.postprocess_text_funcs) == 1
        assert callable(config.preprocess_text_funcs[0])
        assert callable(config.postprocess_text_funcs[0])

        print("   ✓ Preprocess/postprocess functions defined!")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n5. Testing latent shape preparation...")
    try:
        config = SDXLPipelineConfig()

        # Mock batch object
        class MockBatch:
            height = 1024
            width = 1024

        batch = MockBatch()
        batch_size = 1
        num_frames = 1  # Image generation

        # SDXL uses standard 4-channel latents with 8x VAE scale factor
        latent_shape = config.prepare_latent_shape(batch, batch_size, num_frames)

        print(f"   ✓ Latent shape: {latent_shape}")
        # Should be [batch_size, 4, height/8, width/8]
        expected_height = 1024 // 8
        expected_width = 1024 // 8
        assert latent_shape == (batch_size, 4, expected_height, expected_width)

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
    success = test_sdxl_pipeline_config()
    exit(0 if success else 1)
