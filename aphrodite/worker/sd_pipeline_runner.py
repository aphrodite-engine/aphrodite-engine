#!/usr/bin/env python3
"""
Stable Diffusion Pipeline Model Runner for Aphrodite.
Handles text-to-image generation using the complete SD pipeline.
"""

from typing import List, Optional, Union, Dict, Any, Tuple
import torch
from loguru import logger

from aphrodite.worker.model_runner import GPUModelRunnerBase
from aphrodite.worker.model_runner_base import ModelRunnerInputBase
from aphrodite.common.sequence import IntermediateTensors
from aphrodite.pipelines.stable_diffusion import StableDiffusionPipeline, StableDiffusionConfig
from aphrodite.modeling.diffusion.weight_utils import load_sd15_weights
from aphrodite.modeling.models.clip import CLIPModel
from aphrodite.modeling.models.vae import AutoencoderKL
from aphrodite.modeling.models.unet import UNet2DConditionModel


class ModelInputForGPUWithSDMetadata(ModelRunnerInputBase):
    """Model input for SD pipeline with metadata."""
    
    prompts: List[str]
    negative_prompts: Optional[List[str]] = None
    height: int = 512
    width: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    eta: float = 0.0
    generator_seed: Optional[int] = None
    output_type: str = "pil"
    
    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        """Convert to broadcastable tensor dictionary."""
        return {
            "prompts": self.prompts,
            "negative_prompts": self.negative_prompts,
            "height": self.height,
            "width": self.width,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "num_images_per_prompt": self.num_images_per_prompt,
            "eta": self.eta,
            "generator_seed": self.generator_seed,
            "output_type": self.output_type,
        }
    
    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional[Any] = None,
    ) -> "ModelInputForGPUWithSDMetadata":
        """Create from broadcasted tensor dictionary."""
        return cls(
            prompts=tensor_dict.get("prompts", [""]),
            negative_prompts=tensor_dict.get("negative_prompts", None),
            height=tensor_dict.get("height", 512),
            width=tensor_dict.get("width", 512),
            num_inference_steps=tensor_dict.get("num_inference_steps", 50),
            guidance_scale=tensor_dict.get("guidance_scale", 7.5),
            num_images_per_prompt=tensor_dict.get("num_images_per_prompt", 1),
            eta=tensor_dict.get("eta", 0.0),
            generator_seed=tensor_dict.get("generator_seed", None),
            output_type=tensor_dict.get("output_type", "pil"),
        )


class SDPipelineRunner(GPUModelRunnerBase[ModelInputForGPUWithSDMetadata]):
    """
    Model runner for Stable Diffusion pipeline operations.
    
    This runner manages the complete SD pipeline including CLIP, UNet, VAE,
    and scheduler components for text-to-image generation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.pipeline: Optional[StableDiffusionPipeline] = None
        self.clip_model: Optional[CLIPModel] = None
        self.unet_model: Optional[UNet2DConditionModel] = None
        self.vae_model: Optional[AutoencoderKL] = None
        
        # Initialize pipeline components
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the SD pipeline components."""
        logger.info("Initializing Stable Diffusion pipeline components...")
        
        try:
            # Create pipeline configuration
            config = StableDiffusionConfig(
                torch_dtype=self.model_config.dtype,
                device=self.device,
                num_inference_steps=50,  # Default, can be overridden per request
                guidance_scale=7.5,
                scheduler_type="ddim",
                height=512,
                width=512,
            )
            
            # Create individual models (weights will be loaded separately)
            self._create_component_models()
            
            # Create the pipeline
            self.pipeline = StableDiffusionPipeline(
                clip_model=self.clip_model,
                unet_model=self.unet_model,
                vae_model=self.vae_model,
                scheduler=None,  # Will be created by pipeline
                tokenizer=None,  # Will be loaded by pipeline
                config=config,
            )
            
            # Load weights
            self._load_pipeline_weights()
            
            logger.info("✅ SD pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize SD pipeline: {e}")
            raise
    
    def _create_component_models(self):
        """Create the individual SD component models."""
        # Create mock configs for each component
        # Note: In a real implementation, these would be loaded from the actual model configs
        
        class MockCLIPConfig:
            def __init__(self):
                self.model_config = MockModelConfig()
                self.quant_config = self.quant_config
                self.cache_config = None
        
        class MockVAEConfig:
            def __init__(self):
                self.model_config = MockVAEModelConfig()
                self.quant_config = self.quant_config
        
        class MockUNetConfig:
            def __init__(self):
                self.model_config = MockUNetModelConfig()
                self.quant_config = self.quant_config
        
        class MockModelConfig:
            def __init__(self):
                from transformers import CLIPConfig
                self.hf_config = CLIPConfig.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
        
        class MockVAEModelConfig:
            def __init__(self):
                from aphrodite.transformers_utils.config import AutoencoderKLConfig
                self.hf_config = AutoencoderKLConfig(
                    in_channels=3,
                    out_channels=3,
                    latent_channels=4,
                    block_out_channels=[128, 256, 512, 512],
                    layers_per_block=2,
                    scaling_factor=0.18215,
                )
        
        class MockUNetModelConfig:
            def __init__(self):
                from aphrodite.transformers_utils.config import UNet2DConditionModelConfig
                self.hf_config = UNet2DConditionModelConfig(
                    in_channels=4,
                    out_channels=4,
                    block_out_channels=[320, 640, 1280, 1280],
                    layers_per_block=2,
                    attention_head_dim=8,
                    cross_attention_dim=768,
                    norm_num_groups=32,
                    down_block_types=['CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'],
                    up_block_types=['UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'],
                )
        
        # Create models
        logger.info("Creating CLIP model...")
        self.clip_model = CLIPModel(aphrodite_config=MockCLIPConfig())
        
        logger.info("Creating VAE model...")
        self.vae_model = AutoencoderKL(aphrodite_config=MockVAEConfig())
        
        logger.info("Creating UNet model...")
        self.unet_model = UNet2DConditionModel(aphrodite_config=MockUNetConfig())
        
        logger.info("✅ All component models created")
    
    def _load_pipeline_weights(self):
        """Load weights for all pipeline components."""
        logger.info("Loading SD 1.5 weights for all components...")
        
        try:
            # Load weights using our weight loading utility
            results = load_sd15_weights(
                clip_model=self.clip_model,
                vae_model=self.vae_model,
                unet_model=self.unet_model,
                torch_dtype=self.model_config.dtype,
            )
            
            logger.info("✅ All weights loaded successfully")
            logger.info(f"Weight loading coverage: {results['summary']['loading_coverage']:.1%}")
            
            # Load models into pipeline
            self.pipeline.load_models(
                clip_model=self.clip_model,
                unet_model=self.unet_model,
                vae_model=self.vae_model,
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to load pipeline weights: {e}")
            raise
    
    def load_model(self) -> None:
        """Load the SD pipeline. Called by base class."""
        # Pipeline is already initialized in __init__
        logger.info("SD pipeline already loaded and ready")
    
    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPUWithSDMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[Union[List[torch.Tensor], IntermediateTensors]]:
        """
        Execute the SD pipeline for text-to-image generation.
        
        Args:
            model_input: Input containing prompts and generation parameters
            kv_caches: Not used for SD pipeline
            intermediate_tensors: Not used for SD pipeline
            num_steps: Not used for SD pipeline
        
        Returns:
            Generated images
        """
        if self.pipeline is None:
            raise RuntimeError("SD pipeline not initialized")
        
        # Set up generator for reproducibility
        generator = None
        if model_input.generator_seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(model_input.generator_seed)
        
        # Generate images
        try:
            logger.info(f"Generating images for {len(model_input.prompts)} prompts")
            
            output = self.pipeline(
                prompt=model_input.prompts,
                negative_prompt=model_input.negative_prompts,
                height=model_input.height,
                width=model_input.width,
                num_inference_steps=model_input.num_inference_steps,
                guidance_scale=model_input.guidance_scale,
                num_images_per_prompt=model_input.num_images_per_prompt,
                eta=model_input.eta,
                generator=generator,
                output_type=model_input.output_type,
                return_dict=True,
            )
            
            logger.info(f"✅ Generated {len(output['images'])} images")
            
            # Return the generated images
            return output["images"]
            
        except Exception as e:
            logger.error(f"❌ SD pipeline execution failed: {e}")
            raise
    
    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> ModelInputForGPUWithSDMetadata:
        """Create model input from tensor dictionary."""
        # Extract generation parameters from tensor dict
        prompts = tensor_dict.get("prompts", [""])
        negative_prompts = tensor_dict.get("negative_prompts", None)
        height = tensor_dict.get("height", 512)
        width = tensor_dict.get("width", 512)
        num_inference_steps = tensor_dict.get("num_inference_steps", 50)
        guidance_scale = tensor_dict.get("guidance_scale", 7.5)
        num_images_per_prompt = tensor_dict.get("num_images_per_prompt", 1)
        eta = tensor_dict.get("eta", 0.0)
        generator_seed = tensor_dict.get("generator_seed", None)
        output_type = tensor_dict.get("output_type", "pil")
        
        return ModelInputForGPUWithSDMetadata(
            prompts=prompts,
            negative_prompts=negative_prompts,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator_seed=generator_seed,
            output_type=output_type,
        )
    
    def prepare_model_input(
        self,
        seq_group_metadata_list,
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> ModelInputForGPUWithSDMetadata:
        """Prepare model input for SD pipeline."""
        # Extract prompts from sequence group metadata
        prompts = []
        negative_prompts = []
        generation_params = {}
        
        for seq_group_metadata in seq_group_metadata_list:
            # Extract prompt from the sequence
            seq_data = seq_group_metadata.seq_data
            if seq_data:
                prompt = seq_data.get("prompt", "")
                prompts.append(prompt)
                
                # Extract negative prompt if provided
                negative_prompt = seq_data.get("negative_prompt", "")
                if negative_prompt:
                    negative_prompts.append(negative_prompt)
                
                # Extract generation parameters
                for param in ["height", "width", "num_inference_steps", "guidance_scale", 
                             "num_images_per_prompt", "eta", "generator_seed", "output_type"]:
                    if param in seq_data:
                        generation_params[param] = seq_data[param]
        
        # Use negative prompts only if provided for all prompts
        if len(negative_prompts) != len(prompts):
            negative_prompts = None
        
        return ModelInputForGPUWithSDMetadata(
            prompts=prompts,
            negative_prompts=negative_prompts,
            height=generation_params.get("height", 512),
            width=generation_params.get("width", 512),
            num_inference_steps=generation_params.get("num_inference_steps", 50),
            guidance_scale=generation_params.get("guidance_scale", 7.5),
            num_images_per_prompt=generation_params.get("num_images_per_prompt", 1),
            eta=generation_params.get("eta", 0.0),
            generator_seed=generation_params.get("generator_seed", None),
            output_type=generation_params.get("output_type", "pil"),
        )
    
    def get_supported_sd_tasks(self) -> List[str]:
        """Get supported SD pipeline tasks."""
        return ["text2img"]
