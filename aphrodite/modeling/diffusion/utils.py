#!/usr/bin/env python3
"""
Diffusion utilities for Stable Diffusion inference.
Includes classifier-free guidance, latent initialization, and other helpers.
"""

from typing import Optional, Union, Tuple, List, Any
import warnings

import torch
import torch.nn.functional as F
from loguru import logger


def classifier_free_guidance(
    unet_model: torch.nn.Module,
    latents: torch.Tensor,
    timestep: Union[int, torch.Tensor],
    text_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    guidance_scale: float = 7.5,
    do_classifier_free_guidance: bool = True,
) -> torch.Tensor:
    """
    Perform classifier-free guidance for a single denoising step.
    
    Args:
        unet_model: The UNet denoising model
        latents: Current latent samples [batch_size, 4, H//8, W//8]
        timestep: Current timestep (scalar or tensor)
        text_embeddings: Conditional text embeddings [batch_size, seq_len, dim]
        negative_embeddings: Unconditional text embeddings [batch_size, seq_len, dim]
        guidance_scale: Strength of CFG (1.0 = no guidance, higher = stronger)
        do_classifier_free_guidance: Whether to apply CFG
    
    Returns:
        Predicted noise with CFG applied
    """
    if not do_classifier_free_guidance or guidance_scale <= 1.0:
        # No CFG - just run conditional prediction
        with torch.no_grad():
            unet_output = unet_model(
                sample=latents,
                timestep=timestep,
                encoder_hidden_states=text_embeddings,
            )
            
            # Handle both Diffusers output format and raw tensor format
            if hasattr(unet_output, 'sample'):
                noise_pred = unet_output.sample  # Diffusers format
            else:
                noise_pred = unet_output  # Raw tensor format (Aphrodite)
                
        return noise_pred
    
    # CFG with batching trick for efficiency
    batch_size = latents.shape[0]
    
    # Concatenate latents for both conditional and unconditional predictions
    latents_input = torch.cat([latents, latents], dim=0)
    
    # Concatenate text embeddings: [negative, positive]
    text_embeddings_input = torch.cat([negative_embeddings, text_embeddings], dim=0)
    
    # Expand timestep for batch
    if isinstance(timestep, torch.Tensor):
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep_input = timestep.repeat(2 * batch_size)
    else:
        timestep_input = torch.tensor([timestep] * (2 * batch_size), 
                                    device=latents.device, dtype=torch.long)
    
    # Single forward pass for both conditional and unconditional
    with torch.no_grad():
        unet_output = unet_model(
            sample=latents_input,
            timestep=timestep_input,
            encoder_hidden_states=text_embeddings_input,
        )
        
        # Handle both Diffusers output format and raw tensor format
        if hasattr(unet_output, 'sample'):
            noise_pred = unet_output.sample  # Diffusers format
        else:
            noise_pred = unet_output  # Raw tensor format (Aphrodite)
    
    # Split predictions
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
    
    # Apply classifier-free guidance
    noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    
    return noise_pred_cfg


def prepare_text_embeddings(
    text_embeddings: torch.Tensor,
    negative_embeddings: Optional[torch.Tensor] = None,
    guidance_scale: float = 7.5,
    batch_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Prepare text embeddings for CFG.
    
    Args:
        text_embeddings: Conditional text embeddings
        negative_embeddings: Unconditional text embeddings (if None, uses zeros)
        guidance_scale: CFG scale
        batch_size: Batch size to expand to
    
    Returns:
        Tuple of (conditional_embeds, unconditional_embeds, do_cfg)
    """
    do_cfg = guidance_scale > 1.0
    
    # Expand text embeddings to batch size if needed
    if text_embeddings.shape[0] != batch_size:
        text_embeddings = text_embeddings.repeat(batch_size, 1, 1)
    
    # Prepare negative embeddings
    if negative_embeddings is None:
        if do_cfg:
            # Create zero embeddings for unconditional
            negative_embeddings = torch.zeros_like(text_embeddings)
        else:
            negative_embeddings = text_embeddings  # Won't be used
    else:
        # Expand negative embeddings to batch size if needed
        if negative_embeddings.shape[0] != batch_size:
            negative_embeddings = negative_embeddings.repeat(batch_size, 1, 1)
    
    return text_embeddings, negative_embeddings, do_cfg


def init_latents(
    batch_size: int,
    num_channels: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    init_noise_sigma: float = 1.0,
) -> torch.Tensor:
    """
    Initialize random latents for diffusion.
    
    Args:
        batch_size: Number of samples to generate
        num_channels: Number of latent channels (typically 4 for SD)
        height: Latent height (image_height // 8 for SD)
        width: Latent width (image_width // 8 for SD)
        dtype: Tensor dtype
        device: Tensor device
        generator: Random generator for reproducibility
        init_noise_sigma: Initial noise scaling factor from scheduler
    
    Returns:
        Random latent tensor scaled by noise sigma
    """
    shape = (batch_size, num_channels, height, width)
    
    # Generate random noise
    latents = torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    
    # Scale by scheduler's initial noise sigma
    latents = latents * init_noise_sigma
    
    return latents


def encode_image_to_latents(
    vae_model: torch.nn.Module,
    image: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Encode an image to VAE latents.
    
    Args:
        vae_model: VAE encoder model
        image: Input image tensor [batch_size, 3, height, width] in [-1, 1]
        generator: Random generator for VAE stochasticity
    
    Returns:
        Encoded latents [batch_size, 4, height//8, width//8]
    """
    with torch.no_grad():
        latents = vae_model.encode(image)
    
    return latents


def decode_latents_to_image(
    vae_model: torch.nn.Module,
    latents: torch.Tensor,
) -> torch.Tensor:
    """
    Decode VAE latents to images.
    
    Args:
        vae_model: VAE decoder model
        latents: Input latents [batch_size, 4, height//8, width//8]
    
    Returns:
        Decoded images [batch_size, 3, height, width] in [-1, 1]
    """
    with torch.no_grad():
        images = vae_model.decode(latents)
    
    return images


def tensor_to_pil_images(
    tensor: torch.Tensor,
    do_denormalize: bool = True,
) -> List[Any]:
    """
    Convert tensor images to PIL Images.
    
    Args:
        tensor: Image tensor [batch_size, 3, height, width]
        do_denormalize: Whether to denormalize from [-1, 1] to [0, 1]
    
    Returns:
        List of PIL Images
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        raise ImportError("PIL is required for image conversion. Install with: pip install Pillow")
    
    # Denormalize if needed
    if do_denormalize:
        tensor = (tensor + 1.0) / 2.0
    
    # Clamp to valid range
    tensor = torch.clamp(tensor, 0.0, 1.0)
    
    # Convert to numpy
    tensor = tensor.cpu().permute(0, 2, 3, 1).numpy()
    
    # Convert to uint8
    tensor = (tensor * 255).astype(np.uint8)
    
    # Convert to PIL Images
    images = [Image.fromarray(img) for img in tensor]
    
    return images


def pil_images_to_tensor(
    images: List[Any],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    do_normalize: bool = True,
) -> torch.Tensor:
    """
    Convert PIL Images to tensor.
    
    Args:
        images: List of PIL Images
        device: Target device
        dtype: Target dtype
        do_normalize: Whether to normalize from [0, 1] to [-1, 1]
    
    Returns:
        Image tensor [batch_size, 3, height, width]
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError("NumPy is required for image conversion")
    
    # Convert to numpy arrays
    arrays = [np.array(img) for img in images]
    
    # Stack and convert to tensor
    tensor = torch.from_numpy(np.stack(arrays)).float()
    
    # Normalize to [0, 1]
    tensor = tensor / 255.0
    
    # Permute to [batch, channels, height, width]
    tensor = tensor.permute(0, 3, 1, 2)
    
    # Normalize to [-1, 1] if requested
    if do_normalize:
        tensor = tensor * 2.0 - 1.0
    
    # Move to device and convert dtype
    tensor = tensor.to(device=device, dtype=dtype)
    
    return tensor


def check_tensor_shapes(
    latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    timestep: Union[int, torch.Tensor],
    expected_latent_shape: Optional[Tuple[int, ...]] = None,
    expected_text_shape: Optional[Tuple[int, ...]] = None,
) -> bool:
    """
    Validate tensor shapes for diffusion inputs.
    
    Args:
        latents: Latent tensor
        text_embeddings: Text embedding tensor
        timestep: Timestep (scalar or tensor)
        expected_latent_shape: Expected latent shape (optional)
        expected_text_shape: Expected text embedding shape (optional)
    
    Returns:
        True if all shapes are valid
    """
    try:
        # Check basic requirements
        assert latents.dim() == 4, f"Latents must be 4D, got {latents.dim()}D"
        assert text_embeddings.dim() == 3, f"Text embeddings must be 3D, got {text_embeddings.dim()}D"
        
        # Check batch size consistency
        batch_size = latents.shape[0]
        text_batch_size = text_embeddings.shape[0]
        
        # Text embeddings can be 1 (broadcast) or match latents batch size
        assert text_batch_size == 1 or text_batch_size == batch_size, \
            f"Text batch size {text_batch_size} incompatible with latent batch size {batch_size}"
        
        # Check timestep
        if isinstance(timestep, torch.Tensor):
            assert timestep.dim() <= 1, f"Timestep tensor must be 0D or 1D, got {timestep.dim()}D"
            if timestep.dim() == 1:
                assert len(timestep) == batch_size or len(timestep) == 1, \
                    f"Timestep batch size {len(timestep)} incompatible with latent batch size {batch_size}"
        
        # Check expected shapes if provided
        if expected_latent_shape:
            assert latents.shape == expected_latent_shape, \
                f"Latent shape {latents.shape} doesn't match expected {expected_latent_shape}"
        
        if expected_text_shape:
            assert text_embeddings.shape == expected_text_shape, \
                f"Text shape {text_embeddings.shape} doesn't match expected {expected_text_shape}"
        
        return True
        
    except AssertionError as e:
        logger.error(f"Tensor shape validation failed: {e}")
        return False


def get_guidance_scale_schedule(
    guidance_scale: float,
    num_inference_steps: int,
    guidance_rescale: float = 0.0,
    schedule_type: str = "constant",
) -> List[float]:
    """
    Get guidance scale schedule for dynamic CFG.
    
    Args:
        guidance_scale: Base guidance scale
        num_inference_steps: Number of denoising steps
        guidance_rescale: Rescaling factor for guidance (0.0 = no rescaling)
        schedule_type: Type of schedule ("constant", "linear_decay", "cosine_decay")
    
    Returns:
        List of guidance scales for each step
    """
    if schedule_type == "constant":
        return [guidance_scale] * num_inference_steps
    
    elif schedule_type == "linear_decay":
        # Linear decay from guidance_scale to 1.0
        scales = torch.linspace(guidance_scale, 1.0, num_inference_steps).tolist()
        
    elif schedule_type == "cosine_decay":
        # Cosine decay from guidance_scale to 1.0
        steps = torch.arange(num_inference_steps, dtype=torch.float32)
        cosine_factor = 0.5 * (1 + torch.cos(torch.pi * steps / (num_inference_steps - 1)))
        scales = (1.0 + (guidance_scale - 1.0) * cosine_factor).tolist()
        
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    # Apply rescaling if requested
    if guidance_rescale > 0.0:
        scales = [scale * (1.0 + guidance_rescale * (scale - 1.0)) for scale in scales]
    
    return scales


def optimize_memory_usage(
    enable_attention_slicing: bool = True,
    enable_cpu_offload: bool = False,
    enable_sequential_cpu_offload: bool = False,
) -> dict:
    """
    Configure memory optimization settings.
    
    Args:
        enable_attention_slicing: Enable attention slicing to reduce memory
        enable_cpu_offload: Enable model CPU offloading
        enable_sequential_cpu_offload: Enable sequential CPU offloading
    
    Returns:
        Dictionary of optimization settings
    """
    settings = {
        "attention_slicing": enable_attention_slicing,
        "cpu_offload": enable_cpu_offload,
        "sequential_cpu_offload": enable_sequential_cpu_offload,
    }
    
    if enable_sequential_cpu_offload and enable_cpu_offload:
        logger.warning("Both cpu_offload and sequential_cpu_offload enabled. Using sequential_cpu_offload.")
        settings["cpu_offload"] = False
    
    return settings
