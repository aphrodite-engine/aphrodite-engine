# SDXL Implementation Review - Comparison with ComfyUI

## Current Status
- ✅ UNet architecture implemented
- ✅ Dual CLIP (CLIP-L + CLIP-G) implemented  
- ✅ ADM conditioning implemented
- ✅ Weight loading utilities implemented
- ✅ Pipeline integration complete
- ⚠️ Attention produces inf values (clamped to allow progress)
- ❌ VAE decoding produces NaN values

## Key Differences Found

### 1. Attention Implementation
**ComfyUI (`attention_basic`):**
- Conditionally casts to float32 based on `attn_precision`
- Does NOT clamp sim before softmax
- Does NOT subtract max before softmax
- Just does `sim.softmax(dim=-1)` directly

**Our Implementation:**
- Always casts to float32 for bfloat16/float16
- Clamps sim to [-500, 500] after einsum (to handle inf)
- Does softmax directly (removed clamping/subtracting max to match ComfyUI)

**Issue:** We're getting inf values in sim after einsum, which suggests q/k values are extremely large. This might indicate:
- Linear layer weights are too large
- Input to attention has inf values
- Scale factor computation is wrong

### 2. VAE Decoding
**ComfyUI:**
- Uses `process_out` which divides latents by `scale_factor` (0.13025 for SDXL)
- Latents after scaling are in range [-390, 384] (matches our output)
- VAE decode works correctly

**Our Implementation:**
- Divides latents by scaling_factor (0.13025) ✓
- Latents after scaling: [-390, 384] ✓ (matches ComfyUI)
- VAE decode produces NaN ❌

**Issue:** VAE is producing NaN even though latents look reasonable. Possible causes:
- VAE weights not loaded correctly
- Numerical issue in VAE forward pass
- VAE expects different input format/range

### 3. Latent Scaling
**ComfyUI (`SDXL.process_out`):**
```python
def process_out(self, latent):
    return latent / self.scale_factor  # 0.13025
```

**Our Implementation:**
```python
latents = latents / scaling_factor  # 0.13025
```

This matches ComfyUI ✓

## Next Steps to Debug

1. **Attention Inf Values:**
   - Check if q/k values from linear layers are reasonable
   - Verify scale factor computation (dim_head ** -0.5)
   - Check if input to attention has inf values

2. **VAE NaN:**
   - Verify VAE weights are loaded correctly
   - Check VAE forward pass for numerical issues
   - Test VAE with known good latents to isolate issue
   - Check if VAE expects different input dtype/format

3. **Compare with Working Implementation:**
   - Load same weights in ComfyUI and check intermediate values
   - Compare attention outputs at each layer
   - Compare VAE inputs/outputs

