#!/usr/bin/env python3
"""
Test suite for SDXL dual CLIP weight loading utilities.
"""

import torch

from aphrodite.diffusion.runtime.models.encoders.sdxl_clip_weight_utils import (
    clip_text_transformers_convert,
    convert_sdxl_clip_state_dict,
    process_sdxl_clip_weights,
    state_dict_prefix_replace,
)


def test_sdxl_clip_weight_utils():
    """Test SDXL dual CLIP weight conversion utilities."""
    print("=" * 50)
    print("SDXL Dual CLIP Weight Utils Test Suite")
    print("=" * 50)
    
    print("\n1. Testing state_dict_prefix_replace...")
    try:
        state_dict = {
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight": torch.randn(49408, 768),
            "conditioner.embedders.1.model.text_model.embeddings.token_embedding.weight": torch.randn(49408, 1280),
            "other.weight": torch.randn(10, 10),
        }
        
        replace_prefix = {
            "conditioner.embedders.0.transformer.text_model": "clip_l.transformer.text_model",
            "conditioner.embedders.1.model.": "clip_g.",
        }
        converted = state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)
        
        assert "clip_l.transformer.text_model.embeddings.token_embedding.weight" in converted
        assert "clip_g.text_model.embeddings.token_embedding.weight" in converted
        assert "other.weight" not in converted  # Filtered out
        
        print("   ✓ Prefix replacement successful!")
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n2. Testing clip_text_transformers_convert...")
    try:
        state_dict = {
            "clip_g.text_model.embeddings.token_embedding.weight": torch.randn(49408, 1280),
            "clip_g.text_model.encoder.layers.0.self_attn.q_proj.weight": torch.randn(1280, 1280),
            "clip_g.text_model.final_layer_norm.weight": torch.randn(1280),
        }
        
        converted = clip_text_transformers_convert(state_dict, "clip_g.", "clip_g.transformer.")
        
        assert "clip_g.transformer.embeddings.token_embedding.weight" in converted
        assert "clip_g.transformer.encoder.layers.0.self_attn.q_proj.weight" in converted
        assert "clip_g.transformer.final_layer_norm.weight" in converted
        
        print("   ✓ CLIP text transformer conversion successful!")
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n3. Testing convert_sdxl_clip_state_dict...")
    try:
        state_dict = {
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight": torch.randn(49408, 768),
            "conditioner.embedders.1.model.text_model.embeddings.token_embedding.weight": torch.randn(49408, 1280),
            "conditioner.embedders.1.model.text_model.encoder.layers.0.self_attn.q_proj.weight": torch.randn(1280, 1280),
        }
        
        converted = convert_sdxl_clip_state_dict(state_dict)
        
        assert "clip_l.transformer.text_model.embeddings.token_embedding.weight" in converted
        assert "clip_g.transformer.embeddings.token_embedding.weight" in converted
        assert "clip_g.transformer.encoder.layers.0.self_attn.q_proj.weight" in converted
        
        print("   ✓ Full SDXL CLIP state dict conversion successful!")
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n4. Testing process_sdxl_clip_weights...")
    try:
        weights = [
            ("conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight", torch.randn(49408, 768)),
            ("conditioner.embedders.1.model.text_model.embeddings.token_embedding.weight", torch.randn(49408, 1280)),
        ]
        
        processed = list(process_sdxl_clip_weights(weights))
        
        assert len(processed) == 2
        assert "clip_l.transformer.text_model.embeddings.token_embedding.weight" in [p[0] for p in processed]
        assert "clip_g.transformer.embeddings.token_embedding.weight" in [p[0] for p in processed]
        
        print("   ✓ SDXL CLIP weight processing successful!")
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n5. Testing with already converted weights...")
    try:
        # Test with weights that already have clip_l. and clip_g. prefixes
        weights = [
            ("clip_l.transformer.embeddings.token_embedding.weight", torch.randn(49408, 768)),
            ("clip_g.transformer.embeddings.token_embedding.weight", torch.randn(49408, 1280)),
        ]
        
        processed = list(process_sdxl_clip_weights(weights))
        
        assert len(processed) == 2
        assert processed[0][0] == "clip_l.transformer.embeddings.token_embedding.weight"
        assert processed[1][0] == "clip_g.transformer.embeddings.token_embedding.weight"
        
        print("   ✓ Already converted weights handling successful!")
        
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
    success = test_sdxl_clip_weight_utils()
    exit(0 if success else 1)

