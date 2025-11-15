# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Decoding stage for diffusion pipelines.
"""

import weakref

import torch

from aphrodite.diffusion.configs.models.vaes.base import VAEArchConfig
from aphrodite.diffusion.configs.pipelines.qwen_image import (
    QwenImageEditPipelineConfig,
    QwenImagePipelineConfig,
)
from aphrodite.diffusion.runtime.distributed import get_local_torch_device
from aphrodite.diffusion.runtime.loader.component_loader import VAELoader
from aphrodite.diffusion.runtime.models.vaes.common import ParallelTiledVAE
from aphrodite.diffusion.runtime.pipelines.pipeline_batch_info import OutputBatch, Req
from aphrodite.diffusion.runtime.pipelines.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from aphrodite.diffusion.runtime.pipelines.stages.validators import VerificationResult
from aphrodite.diffusion.runtime.server_args import ServerArgs, get_global_server_args
from aphrodite.diffusion.utils import PRECISION_TO_TYPE
from aphrodite.logger import init_logger

logger = init_logger(__name__)


class DecodingStage(PipelineStage):
    """
    Stage for decoding latent representations into pixel space.

    This stage handles the decoding of latent representations into the final
    output format (e.g., pixel values).
    """

    def __init__(self, vae, pipeline=None) -> None:
        self.vae: ParallelTiledVAE = vae
        self.pipeline = weakref.ref(pipeline) if pipeline else None

    @property
    def parallelism_type(self) -> StageParallelismType:
        if get_global_server_args().enable_cfg_parallel:
            return StageParallelismType.MAIN_RANK_ONLY
        return StageParallelismType.REPLICATED

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify decoding stage inputs."""
        result = VerificationResult()
        # Denoised latents for VAE decoding: [batch_size, channels, frames, height_latents, width_latents]
        # result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify decoding stage outputs."""
        result = VerificationResult()
        # Decoded video/images: [batch_size, channels, frames, height, width]
        # result.add_check("output", batch.output, [V.is_tensor, V.with_dims(5)])
        return result

    def scale_and_shift(self, vae_arch_config: VAEArchConfig, latents: torch.Tensor, server_args):
        # 1. scale
        is_qwen_image = isinstance(server_args.pipeline_config, (QwenImagePipelineConfig, QwenImageEditPipelineConfig))
        if is_qwen_image:
            scaling_factor = 1.0 / torch.tensor(vae_arch_config.latents_std, device=latents.device).view(
                1, vae_arch_config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
        else:
            scaling_factor = vae_arch_config.scaling_factor

        # Check if scaling_factor is valid (not 0 or None)
        if scaling_factor == 0 or scaling_factor is None:
            raise ValueError(f"Invalid scaling_factor: {scaling_factor}. Must be non-zero.")
        
        if isinstance(scaling_factor, torch.Tensor):
            latents = latents / scaling_factor.to(latents.device, latents.dtype)
        else:
            latents = latents / scaling_factor

        # 2. shift
        if is_qwen_image:
            shift_factor = (
                torch.tensor(vae_arch_config.latents_mean)
                .view(1, vae_arch_config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
        else:
            shift_factor = getattr(vae_arch_config, "shift_factor", None)

        # Apply shifting if needed
        if shift_factor is not None:
            if isinstance(shift_factor, torch.Tensor):
                latents += shift_factor.to(latents.device, latents.dtype)
            else:
                latents += shift_factor
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, server_args: ServerArgs) -> torch.Tensor:
        """
        Decode latent representations into pixel space using VAE.

        Args:
            latents: Input latent tensor with shape (batch, channels, frames, height_latents, width_latents)
                    or (batch, channels, height_latents, width_latents) for images
            server_args: Configuration containing:
                - disable_autocast: Whether to disable automatic mixed precision (default: False)
                - pipeline_config.vae_precision: VAE computation precision ("fp32", "fp16", "bf16")
                - pipeline_config.vae_tiling: Whether to enable VAE tiling for memory efficiency

        Returns:
            Decoded video tensor with shape (batch, channels, frames, height, width),
            normalized to [0, 1] range and moved to CPU as float32
        """
        self.vae = self.vae.to(get_local_torch_device())
        latents = latents.to(get_local_torch_device())
        
        # Handle 4D (images) vs 5D (videos) latents
        is_image = latents.dim() == 4
        if is_image:
            # Add frame dimension: [B, C, H, W] -> [B, C, 1, H, W]
            latents = latents.unsqueeze(2)
        
        # Setup VAE precision
        # Force fp32 for VAE decoding to prevent NaN (ComfyUI default behavior)
        # VAE decoding is very sensitive to precision issues, especially with large latent values
        vae_dtype = PRECISION_TO_TYPE.get(server_args.pipeline_config.vae_precision, torch.float32)
        # Always use fp32 for VAE to match ComfyUI's default behavior and prevent NaN
        if vae_dtype != torch.float32:
            logger.warning(f"VAE precision was set to {server_args.pipeline_config.vae_precision}, but forcing fp32 to prevent NaN issues")
            vae_dtype = torch.float32
        vae_autocast_enabled = False  # Disable autocast since we're using fp32
        vae_arch_config = server_args.pipeline_config.vae_config.arch_config
        
        # Ensure VAE model is in fp32
        if next(self.vae.parameters()).dtype != torch.float32:
            logger.info("Converting VAE model to float32 for decoding")
            self.vae = self.vae.to(dtype=torch.float32)

        # Check if latents are all zeros (this would indicate a problem)
        if torch.allclose(latents, torch.zeros_like(latents), atol=1e-6):
            logger.error(f"ERROR: Latents are all zeros before VAE decoding! Shape: {latents.shape}")
            raise ValueError("Latents are all zeros - denoising may have failed")
        
        # Debug: Check latents before scaling
        logger.info(f"Latents before scaling - shape: {latents.shape}, min: {latents.min().item():.6f}, max: {latents.max().item():.6f}, mean: {latents.mean().item():.6f}")
        
        # scale and shift
        latents = self.scale_and_shift(vae_arch_config, latents, server_args)
        
        # Debug: Check latents after scaling
        logger.info(f"Latents after scaling - shape: {latents.shape}, min: {latents.min().item():.6f}, max: {latents.max().item():.6f}, mean: {latents.mean().item():.6f}, scaling_factor: {vae_arch_config.scaling_factor}")
        
        # Check if latents became all zeros after scaling
        if torch.allclose(latents, torch.zeros_like(latents), atol=1e-6):
            logger.error(f"ERROR: Latents became all zeros after scaling! scaling_factor: {vae_arch_config.scaling_factor}")
            raise ValueError("Latents became all zeros after scaling - check scaling_factor")

        # Decode latents frame by frame (VAE expects 4D: [B, C, H, W])
        batch_size, channels, num_frames, height, width = latents.shape
        decoded_frames = []
        
        with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled):
            try:
                # TODO: make it more specific
                if server_args.pipeline_config.vae_tiling:
                    self.vae.enable_tiling()
            except Exception:
                pass
            
            for frame_idx in range(num_frames):
                # Extract frame: [B, C, H, W]
                frame_latents = latents[:, :, frame_idx, :, :]
                if not vae_autocast_enabled:
                    frame_latents = frame_latents.to(vae_dtype)
                
                # Check for NaN/inf in frame latents before VAE decode
                if torch.isnan(frame_latents).any() or torch.isinf(frame_latents).any():
                    logger.error(f"ERROR: NaN/inf in frame {frame_idx} latents before VAE decode!")
                    logger.error(f"  Latents stats: min={frame_latents.min().item():.6f}, max={frame_latents.max().item():.6f}, "
                               f"has_nan={torch.isnan(frame_latents).any().item()}, has_inf={torch.isinf(frame_latents).any().item()}")
                    # Replace NaN/inf with zeros
                    frame_latents = torch.nan_to_num(frame_latents, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Ensure frame latents are in fp32 before VAE decode
                frame_latents = frame_latents.to(dtype=torch.float32)
                
                # Decode frame
                try:
                    decoded_frame = self.vae.decode(frame_latents)
                except Exception as e:
                    logger.error(f"ERROR: VAE decode failed for frame {frame_idx}: {e}")
                    logger.error(f"  Frame latents stats: min={frame_latents.min().item():.6f}, max={frame_latents.max().item():.6f}, "
                               f"shape={frame_latents.shape}, dtype={frame_latents.dtype}, device={frame_latents.device}")
                    raise
                
                # Convert output to float32 (like ComfyUI does)
                decoded_frame = decoded_frame.float()
                
                # Debug: Check decoded output before denormalization
                logger.info(f"Decoded frame {frame_idx} before denorm - shape: {decoded_frame.shape}, min: {decoded_frame.min().item():.6f}, max: {decoded_frame.max().item():.6f}, mean: {decoded_frame.mean().item():.6f}")
                
                # De-normalize image to [0, 1] range
                decoded_frame = (decoded_frame / 2 + 0.5).clamp(0, 1)
                
                # Debug: Check after denormalization
                logger.info(f"Decoded frame {frame_idx} after denorm - min: {decoded_frame.min().item():.6f}, max: {decoded_frame.max().item():.6f}, mean: {decoded_frame.mean().item():.6f}")
                
                decoded_frames.append(decoded_frame)
        
        # Stack frames: [B, C, T, H, W]
        image = torch.stack(decoded_frames, dim=2)
        
        # Remove frame dimension for images
        if is_image:
            image = image.squeeze(2)
        
        return image

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        """
        Decode latent representations into pixel space.

        This method processes the batch through the VAE decoder, converting latent
        representations to pixel-space video/images. It also optionally decodes
        trajectory latents for visualization purposes.

        Args:
            batch: The current batch containing:
                - latents: Tensor to decode (batch, channels, frames, height_latents, width_latents)
                - return_trajectory_decoded (optional): Flag to decode trajectory latents
                - trajectory_latents (optional): Latents at different timesteps
                - trajectory_timesteps (optional): Corresponding timesteps
            server_args: Configuration containing:
                - output_type: "latent" to skip decoding, otherwise decode to pixels
                - vae_cpu_offload: Whether to offload VAE to CPU after decoding
                - model_loaded: Track VAE loading state
                - model_paths: Path to VAE model if loading needed

        Returns:
            Modified batch with:
                - output: Decoded frames (batch, channels, frames, height, width) as CPU float32
                - trajectory_decoded (if requested): List of decoded frames per timestep
        """
        # load vae if not already loaded (used for memory constrained devices)
        pipeline = self.pipeline() if self.pipeline else None
        if not server_args.model_loaded["vae"]:
            loader = VAELoader()
            self.vae = loader.load(server_args.model_paths["vae"], server_args)
            if pipeline:
                pipeline.add_module("vae", self.vae)
            server_args.model_loaded["vae"] = True

        if server_args.output_type == "latent":
            frames = batch.latents
        else:
            frames = self.decode(batch.latents, server_args)

        # decode trajectory latents if needed
        if batch.return_trajectory_decoded:
            trajectory_decoded = []
            assert batch.trajectory_latents is not None, "batch should have trajectory latents"
            for idx in range(batch.trajectory_latents.shape[1]):
                # batch.trajectory_latents is [batch_size, timesteps, channels, frames, height, width]
                cur_latent = batch.trajectory_latents[:, idx, :, :, :, :]
                cur_timestep = batch.trajectory_timesteps[idx]
                logger.info("decoding trajectory latent for timestep: %s", cur_timestep)
                decoded_frames = self.decode(cur_latent, server_args)
                trajectory_decoded.append(decoded_frames.cpu().float())
        else:
            trajectory_decoded = None

        # Convert to CPU float32 for compatibility
        frames = frames.cpu().float()

        # Update batch with decoded image
        output_batch = OutputBatch(
            output=frames,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            trajectory_decoded=trajectory_decoded,
        )

        # Offload models if needed
        if hasattr(self, "maybe_free_model_hooks"):
            self.maybe_free_model_hooks()

        if server_args.vae_cpu_offload:
            self.vae.to("cpu")

        if torch.backends.mps.is_available():
            del self.vae
            if pipeline is not None and "vae" in pipeline.modules:
                del pipeline.modules["vae"]
            server_args.model_loaded["vae"] = False

        return output_batch
