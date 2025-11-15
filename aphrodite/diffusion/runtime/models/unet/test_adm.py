#!/usr/bin/env python3
"""
Test suite for SDXL ADM conditioning.
"""

import torch

from aphrodite.diffusion.runtime.models.unet.adm_conditioning import (
    ADMConditioning,
    encode_sdxl_adm,
)


def test_adm_conditioning():
    """Test ADM conditioning encoding."""
    print("=" * 50)
    print("SDXL ADM Conditioning Test Suite")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dummy CLIP pooled output [B, 1280]
    batch_size = 2
    clip_pooled = torch.randn(batch_size, 1280, device=device)

    print("\n1. Testing SDXL base model ADM encoding...")
    try:
        adm = ADMConditioning(embed_dim=256)

        adm_cond = adm.encode_sdxl_base(
            clip_pooled=clip_pooled,
            height=1024,
            width=1024,
            crop_h=0,
            crop_w=0,
            target_height=1024,
            target_width=1024,
        )

        print("   ✓ Base ADM encoding successful!")
        print(f"   Input CLIP pooled shape: {clip_pooled.shape}")
        print(f"   Output ADM shape: {adm_cond.shape}")
        print(f"   Expected shape: ({batch_size}, 2816)")

        assert adm_cond.shape == (batch_size, 2816), f"ADM shape {adm_cond.shape} != expected ({batch_size}, 2816)"

        # Verify it contains CLIP pooled output
        assert torch.allclose(adm_cond[:, :1280], clip_pooled), "ADM should start with CLIP pooled output"

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n2. Testing SDXL refiner model ADM encoding...")
    try:
        adm_cond_refiner = adm.encode_sdxl_refiner(
            clip_pooled=clip_pooled,
            height=1024,
            width=1024,
            crop_h=0,
            crop_w=0,
            aesthetic_score=6.0,
        )

        print("   ✓ Refiner ADM encoding successful!")
        print(f"   Output ADM shape: {adm_cond_refiner.shape}")
        print(f"   Expected shape: ({batch_size}, 2560)")

        assert adm_cond_refiner.shape == (batch_size, 2560), (
            f"ADM shape {adm_cond_refiner.shape} != expected ({batch_size}, 2560)"
        )

        # Verify it contains CLIP pooled output
        assert torch.allclose(adm_cond_refiner[:, :1280], clip_pooled), "ADM should start with CLIP pooled output"

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n3. Testing convenience function...")
    try:
        # Base model
        adm_base = encode_sdxl_adm(
            clip_pooled=clip_pooled,
            height=768,
            width=768,
            is_refiner=False,
        )
        assert adm_base.shape == (batch_size, 2816), f"Base ADM shape {adm_base.shape} != expected ({batch_size}, 2816)"

        # Refiner model
        adm_refiner = encode_sdxl_adm(
            clip_pooled=clip_pooled,
            height=768,
            width=768,
            aesthetic_score=6.0,
            is_refiner=True,
        )
        assert adm_refiner.shape == (batch_size, 2560), (
            f"Refiner ADM shape {adm_refiner.shape} != expected ({batch_size}, 2560)"
        )

        print("   ✓ Convenience function works!")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n4. Testing different resolutions...")
    try:
        # Test various resolutions
        resolutions = [
            (512, 512),
            (768, 768),
            (1024, 1024),
            (1024, 768),
            (768, 1024),
        ]

        for h, w in resolutions:
            adm_cond = encode_sdxl_adm(
                clip_pooled=clip_pooled,
                height=h,
                width=w,
                is_refiner=False,
            )
            assert adm_cond.shape == (batch_size, 2816), f"ADM shape incorrect for resolution {h}x{w}"

        print("   ✓ All resolutions work!")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n5. Testing negative prompt aesthetic score...")
    try:
        # Positive prompt (default)
        adm_pos = encode_sdxl_adm(
            clip_pooled=clip_pooled,
            height=768,
            width=768,
            aesthetic_score=6.0,
            is_refiner=True,
            is_negative=False,
        )

        # Negative prompt (should use 2.5)
        adm_neg = encode_sdxl_adm(
            clip_pooled=clip_pooled,
            height=768,
            width=768,
            aesthetic_score=None,  # Should default to 2.5 for negative
            is_refiner=True,
            is_negative=True,
        )

        # They should be different (different aesthetic scores)
        assert not torch.allclose(adm_pos, adm_neg), "Positive and negative ADM should differ"

        print("   ✓ Negative prompt handling works!")

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
    success = test_adm_conditioning()
    exit(0 if success else 1)
