#!/usr/bin/env python3
"""
Test suite for SDXL UNet weight loading utilities.
"""

import torch

from aphrodite.diffusion.runtime.models.unet.weight_utils import (
    convert_sdxl_unet_state_dict,
    process_sdxl_unet_weights,
    state_dict_key_replace,
    state_dict_prefix_replace,
)


def test_unet_weight_utils():
    """Test UNet weight conversion utilities."""
    print("=" * 50)
    print("SDXL UNet Weight Utils Test Suite")
    print("=" * 50)

    print("\n1. Testing state_dict_prefix_replace...")
    try:
        state_dict = {
            "unet.conv_in.weight": torch.randn(4, 4, 3, 3),
            "unet.time_embed.0.weight": torch.randn(1280, 320),
            "other.weight": torch.randn(10, 10),
        }

        replace_prefix = {"unet.": ""}
        converted = state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False)

        assert "conv_in.weight" in converted
        assert "time_embed.0.weight" in converted
        assert "other.weight" in converted
        assert "unet.conv_in.weight" not in converted

        print("   ✓ Prefix replacement successful!")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n2. Testing state_dict_key_replace...")
    try:
        state_dict = {
            "old_key_1": torch.randn(10),
            "old_key_2": torch.randn(20),
        }

        keys_to_replace = {"old_key_1": "new_key_1"}
        converted = state_dict_key_replace(state_dict, keys_to_replace)

        assert "new_key_1" in converted
        assert "old_key_2" in converted
        assert "old_key_1" not in converted

        print("   ✓ Key replacement successful!")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n3. Testing convert_sdxl_unet_state_dict...")
    try:
        state_dict = {
            "unet.conv_in.weight": torch.randn(4, 4, 3, 3),
            "unet.time_embed.0.weight": torch.randn(1280, 320),
        }

        converted = convert_sdxl_unet_state_dict(state_dict)

        assert "conv_in.weight" in converted
        assert "time_embed.0.weight" in converted
        assert "unet.conv_in.weight" not in converted

        print("   ✓ UNet state dict conversion successful!")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n4. Testing process_sdxl_unet_weights...")
    try:
        weights = [
            ("unet.conv_in.weight", torch.randn(4, 4, 3, 3)),
            ("unet.time_embed.0.weight", torch.randn(1280, 320)),
        ]

        processed = list(process_sdxl_unet_weights(weights))

        assert len(processed) == 2
        assert processed[0][0] == "conv_in.weight"
        assert processed[1][0] == "time_embed.0.weight"

        print("   ✓ UNet weight processing successful!")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n5. Testing with custom prefix...")
    try:
        weights = [
            ("unet.conv_in.weight", torch.randn(4, 4, 3, 3)),
        ]

        processed = list(process_sdxl_unet_weights(weights, prefix="model."))

        assert len(processed) == 1
        assert processed[0][0] == "model.conv_in.weight"

        print("   ✓ Custom prefix handling successful!")

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
    success = test_unet_weight_utils()
    exit(0 if success else 1)
