"""
Diffusion schedulers for Stable Diffusion inference.
"""

from typing import Optional, Union, Dict, Any
import warnings

import torch
from diffusers import DDIMScheduler, PNDMScheduler
from diffusers.schedulers.scheduling_utils import SchedulerOutput
from loguru import logger


class AphroditeSchedulerWrapper:
    """
    Base wrapper for Diffusers schedulers to integrate with Aphrodite.

    This wrapper provides a consistent interface for different scheduler types
    and handles the conversion between Aphrodite and Diffusers formats.
    """

    def __init__(
        self,
        scheduler_name: str = "ddim",
        num_inference_steps: int = 50,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.scheduler_name = scheduler_name.lower()
        self.num_inference_steps = num_inference_steps
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float16
        self.scheduler = None
        self._init_scheduler()

    def _init_scheduler(self):
        """Initialize the underlying Diffusers scheduler."""
        if self.scheduler_name == "ddim":
            self.scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
                prediction_type="epsilon",
            )
        elif self.scheduler_name == "pndm":
            self.scheduler = PNDMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                skip_prk_steps=True,
                set_alpha_to_one=False,
                prediction_type="epsilon",
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")

        # Set inference timesteps
        self.scheduler.set_timesteps(self.num_inference_steps)
        
        # Transfer timesteps to correct device
        self._transfer_to_device()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        scheduler_name: str = "ddim",
        num_inference_steps: int = 50,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs
    ) -> "AphroditeSchedulerWrapper":
        """
        Load a scheduler from a pretrained Stable Diffusion model.

        Args:
            model_path: Path to the SD model
            scheduler_name: Type of scheduler ('ddim' or 'pndm')
            num_inference_steps: Number of denoising steps
            device: Target device
            dtype: Target dtype
            **kwargs: Additional arguments passed to scheduler

        Returns:
            AphroditeSchedulerWrapper instance
        """
        wrapper = cls.__new__(cls)
        wrapper.scheduler_name = scheduler_name.lower()
        wrapper.num_inference_steps = num_inference_steps
        wrapper.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wrapper.dtype = dtype or torch.float16

        # Load from pretrained (don't pass device/dtype to Diffusers scheduler)
        scheduler_kwargs = {k: v for k, v in kwargs.items() if k not in ['device', 'dtype']}
        
        if wrapper.scheduler_name == "ddim":
            wrapper.scheduler = DDIMScheduler.from_pretrained(
                model_path, subfolder="scheduler", **scheduler_kwargs
            )
        elif wrapper.scheduler_name == "pndm":
            wrapper.scheduler = PNDMScheduler.from_pretrained(
                model_path, subfolder="scheduler", **scheduler_kwargs
            )
        else:
            raise ValueError(
                f"Unsupported scheduler: {wrapper.scheduler_name}")

        # Set inference timesteps
        wrapper.scheduler.set_timesteps(wrapper.num_inference_steps)
        
        # Transfer to device
        wrapper._transfer_to_device()

        return wrapper

    def _transfer_to_device(self):
        """Transfer scheduler tensors to the correct device."""
        if hasattr(self.scheduler, 'timesteps') and self.scheduler.timesteps is not None:
            # Timesteps must remain as long tensors for indexing
            self.scheduler.timesteps = self.scheduler.timesteps.to(device=self.device, dtype=torch.long)
        
        # Transfer other scheduler tensors if they exist (these can use the target dtype)
        for attr_name in ['alphas_cumprod', 'alpha_t', 'beta_t', 'sigma_t']:
            if hasattr(self.scheduler, attr_name):
                attr_value = getattr(self.scheduler, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    setattr(self.scheduler, attr_name, attr_value.to(device=self.device, dtype=self.dtype))

    def to(self, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None):
        """Transfer scheduler to device and/or dtype."""
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self._transfer_to_device()
        return self

    def set_timesteps(self, num_inference_steps: int):
        """Set new timesteps and transfer to device."""
        self.num_inference_steps = num_inference_steps
        self.scheduler.set_timesteps(num_inference_steps)
        self._transfer_to_device()

    @property
    def timesteps(self) -> torch.Tensor:
        """Get the timesteps for the diffusion process."""
        return self.scheduler.timesteps

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Scale the input sample for the UNet model.

        Args:
            sample: Input latent sample
            timestep: Current timestep

        Returns:
            Scaled sample
        """
        # Ensure timestep is on the correct device (must be long for indexing)
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(device=self.device, dtype=torch.long)
        else:
            # Convert scalar to long tensor
            timestep = torch.tensor(timestep, device=self.device, dtype=torch.long)
        
        return self.scheduler.scale_model_input(sample, timestep)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ) -> SchedulerOutput:
        """
        Perform one denoising step.

        Args:
            model_output: Predicted noise from the UNet
            timestep: Current timestep
            sample: Current noisy sample
            eta: Random noise factor (for DDIM)
            generator: Random number generator
            **kwargs: Additional scheduler arguments

        Returns:
            SchedulerOutput with prev_sample and pred_original_sample
        """
        # Ensure timestep is on the correct device and dtype (must be long for indexing)
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(device=self.device, dtype=torch.long)
        else:
            # Convert scalar to long tensor
            timestep = torch.tensor(timestep, device=self.device, dtype=torch.long)
        
        # Handle different scheduler signatures
        step_kwargs = {
            "model_output": model_output,
            "timestep": timestep,
            "sample": sample,
            **kwargs
        }

        # Add scheduler-specific parameters
        if self.scheduler_name == "ddim":
            step_kwargs["eta"] = eta
            if generator is not None:
                step_kwargs["generator"] = generator
        elif self.scheduler_name == "pndm":
            # PNDM doesn't use eta or generator
            pass

        try:
            return self.scheduler.step(**step_kwargs)
        except Exception as e:
            logger.error(f"Scheduler step failed: {e}")
            logger.error(f"Scheduler: {self.scheduler_name}, timestep: {timestep}, shapes: model_output={model_output.shape}, sample={sample.shape}")
            raise

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to clean samples for training or initialization.

        Args:
            original_samples: Clean samples
            noise: Random noise
            timesteps: Timesteps for noise addition

        Returns:
            Noisy samples
        """
        return self.scheduler.add_noise(original_samples, noise, timesteps)

    def init_noise_sigma(self) -> float:
        """Get the initial noise sigma value."""
        return self.scheduler.init_noise_sigma


class DiffusionPipeline:
    """
    Simple diffusion pipeline for text-to-image generation.

    This class orchestrates the diffusion process using CLIP, UNet, VAE,
    and scheduler components.
    """

    def __init__(
        self,
        scheduler: AphroditeSchedulerWrapper,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
    ):
        self.scheduler = scheduler
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

    def __call__(
        self,
        clip_model,  # CLIP text encoder
        unet_model,  # UNet denoising model
        vae_model,   # VAE decoder
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        height: int = 512,
        width: int = 512,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the diffusion pipeline.

        Args:
            clip_model: CLIP text encoder model
            unet_model: UNet denoising model
            vae_model: VAE decoder model
            prompt_embeds: Text embeddings from CLIP
            negative_prompt_embeds: Negative text embeddings for CFG
            height: Output image height
            width: Output image width
            num_images_per_prompt: Number of images to generate
            generator: Random number generator
            latents: Initial latents (if None, random noise is used)

        Returns:
            Generated images as tensors
        """
        batch_size = prompt_embeds.shape[0]
        device = prompt_embeds.device
        dtype = prompt_embeds.dtype

        # Prepare latents
        if latents is None:
            shape = (
                batch_size * num_images_per_prompt,
                4,  # VAE latent channels
                height // 8,  # VAE downsampling factor
                width // 8,
            )
            latents = torch.randn(
                shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            latents = latents * self.scheduler.init_noise_sigma()

        # Prepare text embeddings for CFG
        if self.guidance_scale > 1.0 and negative_prompt_embeds is not None:
            # Concatenate for classifier-free guidance
            text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])
            latents = torch.cat([latents] * 2)
        else:
            text_embeddings = prompt_embeds

        # Denoising loop
        for i, timestep in enumerate(self.scheduler.timesteps):
            # Scale the input for the UNet
            latent_model_input = self.scheduler.scale_model_input(
                latents, timestep)

            # Predict noise
            with torch.no_grad():
                noise_pred = unet_model(
                    sample=latent_model_input,
                    timestep=timestep.unsqueeze(0).repeat(
                        latent_model_input.shape[0]),
                    encoder_hidden_states=text_embeddings,
                )

            # Classifier-free guidance
            if (self.guidance_scale > 1.0 and
                    negative_prompt_embeds is not None):
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                # Remove duplicate latents
                latents = latents[:batch_size * num_images_per_prompt]

            # Compute previous sample
            scheduler_output = self.scheduler.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=latents,
                generator=generator,
            )
            latents = scheduler_output.prev_sample

        # Decode latents to images
        with torch.no_grad():
            images = vae_model.decode(latents)

        return images


def create_scheduler(
    scheduler_type: str = "ddim",
    num_inference_steps: int = 50,
    model_path: Optional[str] = None,
    **kwargs
) -> AphroditeSchedulerWrapper:
    """
    Factory function to create a scheduler.

    Args:
        scheduler_type: Type of scheduler ('ddim' or 'pndm')
        num_inference_steps: Number of denoising steps
        model_path: Optional path to load scheduler config from
        **kwargs: Additional scheduler arguments

    Returns:
        AphroditeSchedulerWrapper instance
    """
    if model_path:
        return AphroditeSchedulerWrapper.from_pretrained(
            model_path, scheduler_type, num_inference_steps, **kwargs
        )
    else:
        return AphroditeSchedulerWrapper(scheduler_type, num_inference_steps)
