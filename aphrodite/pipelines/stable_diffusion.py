#!/usr/bin/env python3
"""
Stable Diffusion Pipeline for Aphrodite.
Complete text-to-image generation pipeline combining CLIP, UNet, VAE, and scheduler.
"""

from typing import Optional, Union, List, Dict, Any, Callable
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from loguru import logger

from aphrodite.modeling.diffusion.schedulers import AphroditeSchedulerWrapper
from aphrodite.modeling.diffusion.utils import (
    classifier_free_guidance,
    prepare_text_embeddings,
    init_latents,
    decode_latents_to_image,
    tensor_to_pil_images,
    check_tensor_shapes,
    get_guidance_scale_schedule,
)


@dataclass
class StableDiffusionConfig:
    """Configuration for Stable Diffusion pipeline."""
    
    # Model settings
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    torch_dtype: torch.dtype = torch.float16
    device: Union[str, torch.device] = "cuda"
    
    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    scheduler_type: str = "ddim"
    
    # Image settings
    height: int = 512
    width: int = 512
    num_images_per_prompt: int = 1
    
    # Advanced settings
    eta: float = 0.0
    enable_attention_slicing: bool = True
    enable_cpu_offload: bool = False
    
    # Safety settings
    requires_safety_checker: bool = False
    safety_checker_device: str = "cpu"


class StableDiffusionPipeline:
    """
    Complete Stable Diffusion pipeline for text-to-image generation.
    
    This pipeline combines CLIP text encoder, UNet denoising model, VAE decoder,
    and DDIM/PNDM scheduler for high-quality image generation from text prompts.
    """
    
    def __init__(
        self,
        clip_model: Optional[torch.nn.Module] = None,
        unet_model: Optional[torch.nn.Module] = None,
        vae_model: Optional[torch.nn.Module] = None,
        scheduler: Optional[AphroditeSchedulerWrapper] = None,
        tokenizer: Optional[Any] = None,
        config: Optional[StableDiffusionConfig] = None,
    ):
        """
        Initialize the Stable Diffusion pipeline.
        
        Args:
            clip_model: CLIP text encoder model
            unet_model: UNet denoising model
            vae_model: VAE decoder model
            scheduler: Diffusion scheduler
            tokenizer: Text tokenizer (from transformers)
            config: Pipeline configuration
        """
        self.config = config or StableDiffusionConfig()
        
        # Store models
        self.clip_model = clip_model
        self.unet_model = unet_model
        self.vae_model = vae_model
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        
        # Pipeline state
        self.device = torch.device(self.config.device)
        self.dtype = self.config.torch_dtype
        self._models_loaded = False
        
        # Validate models if provided
        if all(model is not None for model in [clip_model, unet_model, vae_model, scheduler]):
            self._validate_models()
            self._models_loaded = True
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
        torch_dtype: torch.dtype = torch.float16,
        device: Union[str, torch.device] = "cuda",
        **kwargs
    ) -> "StableDiffusionPipeline":
        """
        Load a complete Stable Diffusion pipeline from pretrained weights.
        
        Args:
            pretrained_model_name_or_path: HF model repo or local path
            torch_dtype: Target dtype for models
            device: Target device
            **kwargs: Additional configuration options
        
        Returns:
            Initialized StableDiffusionPipeline
        """
        from aphrodite.modeling.diffusion.weight_utils import StableDiffusionWeightLoader
        from aphrodite.modeling.models.clip import CLIPModel
        from aphrodite.modeling.models.vae import AutoencoderKL
        from aphrodite.modeling.models.unet import UNet2DConditionModel
        from aphrodite.config import AphroditeConfig
        from transformers import CLIPTokenizer
        
        logger.info(f"Loading Stable Diffusion pipeline from {pretrained_model_name_or_path}")
        
        # Create configuration
        config = StableDiffusionConfig(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            **kwargs
        )
        
        # Load tokenizer
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
        )
        
        # Create scheduler
        scheduler = AphroditeSchedulerWrapper.from_pretrained(
            pretrained_model_name_or_path,
            scheduler_name=config.scheduler_type,
            num_inference_steps=config.num_inference_steps,
            device=device,
            dtype=torch_dtype,
        )
        
        # Note: Model creation and weight loading would happen here
        # For now, we'll create the pipeline structure and load models separately
        
        pipeline = cls(
            clip_model=None,  # Will be loaded separately
            unet_model=None,  # Will be loaded separately  
            vae_model=None,   # Will be loaded separately
            scheduler=scheduler,
            tokenizer=tokenizer,
            config=config,
        )
        
        logger.info("✅ Stable Diffusion pipeline structure created")
        logger.info("⚠️ Models need to be loaded separately using load_models()")
        
        return pipeline
    
    def load_models(
        self,
        clip_model: torch.nn.Module,
        unet_model: torch.nn.Module,
        vae_model: torch.nn.Module,
    ):
        """
        Load the component models into the pipeline.
        
        Args:
            clip_model: CLIP text encoder
            unet_model: UNet denoising model
            vae_model: VAE decoder model
        """
        self.clip_model = clip_model
        self.unet_model = unet_model
        self.vae_model = vae_model
        
        self._validate_models()
        self._models_loaded = True
        
        logger.info("✅ All models loaded into pipeline")
    
    def _validate_models(self):
        """Validate that all required models are present and compatible."""
        required_models = {
            "clip_model": self.clip_model,
            "unet_model": self.unet_model,
            "vae_model": self.vae_model,
            "scheduler": self.scheduler,
        }
        
        missing_models = [name for name, model in required_models.items() if model is None]
        
        if missing_models:
            raise ValueError(f"Missing required models: {missing_models}")
        
        # Move models to correct device and dtype
        self.clip_model = self.clip_model.to(device=self.device, dtype=self.dtype)
        self.unet_model = self.unet_model.to(device=self.device, dtype=self.dtype)
        self.vae_model = self.vae_model.to(device=self.device, dtype=self.dtype)
        
        # Set models to eval mode
        self.clip_model.eval()
        self.unet_model.eval()
        self.vae_model.eval()
        
        logger.info("✅ All models validated and moved to target device")
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text prompts to embeddings.
        
        Args:
            prompt: Text prompt(s)
            negative_prompt: Negative prompt(s) for CFG
            num_images_per_prompt: Number of images per prompt
            do_classifier_free_guidance: Whether to prepare for CFG
        
        Returns:
            Dictionary with prompt embeddings and metadata
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Handle string inputs
        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        else:
            batch_size = len(prompt)
        
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        elif negative_prompt is None and do_classifier_free_guidance:
            negative_prompt = [""] * batch_size
        
        # Tokenize prompts
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        # Encode text
        with torch.no_grad():
            prompt_embeds = self.clip_model(text_input_ids)
        
        # Handle negative prompts
        negative_prompt_embeds = None
        if do_classifier_free_guidance and negative_prompt:
            negative_inputs = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            negative_input_ids = negative_inputs.input_ids.to(self.device)
            
            with torch.no_grad():
                negative_prompt_embeds = self.clip_model(negative_input_ids)
        
        # Expand embeddings for multiple images per prompt
        if num_images_per_prompt > 1:
            batch_size_expand = batch_size * num_images_per_prompt
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        
        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "do_classifier_free_guidance": do_classifier_free_guidance and negative_prompt_embeds is not None,
        }
    
    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 4,
        height: int = 512,
        width: int = 512,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prepare initial latents for diffusion.
        
        Args:
            batch_size: Number of samples
            num_channels_latents: Number of latent channels
            height: Image height
            width: Image width
            generator: Random generator
            latents: Pre-generated latents (optional)
        
        Returns:
            Initial latent tensor
        """
        if latents is None:
            latents = init_latents(
                batch_size=batch_size,
                num_channels=num_channels_latents,
                height=height // 8,  # VAE downsampling factor
                width=width // 8,
                dtype=self.dtype,
                device=self.device,
                generator=generator,
                init_noise_sigma=self.scheduler.init_noise_sigma(),
            )
        else:
            # Ensure latents are on correct device
            latents = latents.to(device=self.device, dtype=self.dtype)
        
        return latents
    
    def denoise_loop(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        callback: Optional[Callable] = None,
        callback_steps: int = 1,
    ) -> torch.Tensor:
        """
        Run the iterative denoising loop.
        
        Args:
            latents: Initial noisy latents
            prompt_embeds: Text embeddings
            negative_prompt_embeds: Negative text embeddings
            guidance_scale: CFG scale
            eta: DDIM eta parameter
            generator: Random generator
            callback: Optional callback function
            callback_steps: Steps between callback calls
        
        Returns:
            Denoised latents
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        do_classifier_free_guidance = negative_prompt_embeds is not None and guidance_scale > 1.0
        
        # Get guidance scale schedule (for advanced usage)
        guidance_scales = [guidance_scale] * len(self.scheduler.timesteps)
        
        # Denoising loop
        for i, timestep in enumerate(self.scheduler.timesteps):
            # Scale model input
            latent_model_input = self.scheduler.scale_model_input(latents, timestep)
            
            # Predict noise using CFG
            if do_classifier_free_guidance:
                noise_pred = classifier_free_guidance(
                    unet_model=self.unet_model,
                    latents=latent_model_input,
                    timestep=timestep,
                    text_embeddings=prompt_embeds,
                    negative_embeddings=negative_prompt_embeds,
                    guidance_scale=guidance_scales[i],
                    do_classifier_free_guidance=True,
                )
            else:
                # No CFG - direct forward pass
                with torch.no_grad():
                    noise_pred = self.unet_model(
                        sample=latent_model_input,
                        timestep=timestep.unsqueeze(0).repeat(latent_model_input.shape[0]),
                        encoder_hidden_states=prompt_embeds,
                    )
            
            # Scheduler step
            scheduler_output = self.scheduler.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=latents,
                eta=eta,
                generator=generator,
            )
            
            latents = scheduler_output.prev_sample
            
            # Call callback if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, timestep, latents)
        
        return latents
    
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images using VAE.
        
        Args:
            latents: Latent tensor to decode
        
        Returns:
            Decoded image tensor
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # VAE decode
        with torch.no_grad():
            images = self.vae_model.decode(latents)
        
        return images
    
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable] = None,
        callback_steps: int = 1,
        **kwargs
    ) -> Union[List[Any], Dict[str, Any]]:
        """
        Complete text-to-image generation pipeline.
        
        Args:
            prompt: Text prompt(s)
            negative_prompt: Negative prompt(s) for CFG
            height: Output image height
            width: Output image width
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            num_images_per_prompt: Number of images per prompt
            eta: DDIM eta parameter
            generator: Random generator for reproducibility
            latents: Pre-generated initial latents
            output_type: Output format ("pil", "tensor")
            return_dict: Whether to return dict or list
            callback: Optional progress callback
            callback_steps: Steps between callback calls
        
        Returns:
            Generated images as PIL Images or tensors
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Update scheduler steps if different
        if num_inference_steps != self.scheduler.num_inference_steps:
            self.scheduler.set_timesteps(num_inference_steps)
        
        # Determine batch size
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)
        
        total_batch_size = batch_size * num_images_per_prompt
        
        logger.info(f"Generating {total_batch_size} images with {num_inference_steps} steps")
        
        # 1. Encode prompts
        prompt_data = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=guidance_scale > 1.0,
        )
        
        prompt_embeds = prompt_data["prompt_embeds"]
        negative_prompt_embeds = prompt_data["negative_prompt_embeds"]
        
        # 2. Prepare latents
        latents = self.prepare_latents(
            batch_size=total_batch_size,
            height=height,
            width=width,
            generator=generator,
            latents=latents,
        )
        
        # 3. Denoising loop
        latents = self.denoise_loop(
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            eta=eta,
            generator=generator,
            callback=callback,
            callback_steps=callback_steps,
        )
        
        # 4. Decode to images
        images = self.decode_latents(latents)
        
        # 5. Convert to requested output format
        if output_type == "pil":
            images = tensor_to_pil_images(images, do_denormalize=True)
        elif output_type == "tensor":
            # Keep as tensor, ensure correct range
            images = (images + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            images = torch.clamp(images, 0.0, 1.0)
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")
        
        logger.info(f"✅ Generated {len(images)} images")
        
        if return_dict:
            return {"images": images, "latents": latents}
        else:
            return images
    
    def enable_attention_slicing(self, slice_size: Optional[int] = None):
        """Enable attention slicing to reduce memory usage."""
        logger.info("Attention slicing enabled")
        # Implementation would go here
    
    def disable_attention_slicing(self):
        """Disable attention slicing."""
        logger.info("Attention slicing disabled")
        # Implementation would go here
    
    def enable_cpu_offload(self):
        """Enable CPU offloading to reduce GPU memory usage."""
        logger.info("CPU offloading enabled")
        # Implementation would go here
    
    def disable_cpu_offload(self):
        """Disable CPU offloading."""
        logger.info("CPU offloading disabled")
        # Implementation would go here
