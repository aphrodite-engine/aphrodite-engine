#!/usr/bin/env python3
"""
Weight loading utilities for Stable Diffusion models.
Handles downloading, caching, and loading weights from Hugging Face repos.
"""

import os
import torch
from typing import Dict, Optional, Union, Tuple, Any
from pathlib import Path

from loguru import logger
from aphrodite.config import AphroditeConfig


class StableDiffusionWeightLoader:
    """Utility class for loading Stable Diffusion weights from Hugging Face."""
    
    # Default model repositories
    DEFAULT_REPOS = {
        "sd15": "runwayml/stable-diffusion-v1-5",
        "sd21": "stabilityai/stable-diffusion-2-1",
        "sd21_base": "stabilityai/stable-diffusion-2-1-base",
    }
    
    def __init__(
        self,
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
        cache_dir: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        device_map: Optional[str] = None,
        low_cpu_mem_usage: bool = True,
    ):
        """Initialize the weight loader.
        
        Args:
            pretrained_model_name_or_path: HF model repo or local path
            cache_dir: Directory to cache downloaded weights
            torch_dtype: Target dtype for loaded weights (fp16/bf16/fp32)
            device_map: Device mapping strategy
            low_cpu_mem_usage: Use low CPU memory loading when possible
        """
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.cache_dir = cache_dir
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.low_cpu_mem_usage = low_cpu_mem_usage
        
        # Resolve shorthand names
        if pretrained_model_name_or_path in self.DEFAULT_REPOS:
            self.pretrained_model_name_or_path = self.DEFAULT_REPOS[pretrained_model_name_or_path]
    
    def load_clip_weights(
        self, 
        clip_model: torch.nn.Module,
        subfolder: str = "text_encoder",
    ) -> Dict[str, Any]:
        """Load CLIP text encoder weights.
        
        Args:
            clip_model: The CLIP model to load weights into
            subfolder: Subfolder in the repo containing CLIP weights
            
        Returns:
            Dict with loading statistics
        """
        logger.info(f"Loading CLIP weights from {self.pretrained_model_name_or_path}/{subfolder}")
        
        try:
            from transformers import CLIPTextModel
            
            # Load the reference model
            reference_model = CLIPTextModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder=subfolder,
                cache_dir=self.cache_dir,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
            )
            
            # Extract weights as (name, tensor) pairs
            weights = [(name, param.data) for name, param in reference_model.named_parameters()]
            
            # Load into our model
            loaded_params = clip_model.load_weights(weights)
            
            logger.info(f"✅ CLIP weights loaded: {len(loaded_params)} parameters")
            
            return {
                "loaded_params": len(loaded_params),
                "total_params": len(list(clip_model.named_parameters())),
                "dtype": self.torch_dtype,
                "source": f"{self.pretrained_model_name_or_path}/{subfolder}",
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to load CLIP weights: {e}")
            raise
    
    def load_vae_weights(
        self,
        vae_model: torch.nn.Module,
        subfolder: str = "vae",
    ) -> Dict[str, Any]:
        """Load VAE (AutoencoderKL) weights.
        
        Args:
            vae_model: The VAE model to load weights into
            subfolder: Subfolder in the repo containing VAE weights
            
        Returns:
            Dict with loading statistics
        """
        logger.info(f"Loading VAE weights from {self.pretrained_model_name_or_path}/{subfolder}")
        
        try:
            from diffusers import AutoencoderKL
            
            # Load the reference model
            reference_model = AutoencoderKL.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder=subfolder,
                cache_dir=self.cache_dir,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
            )
            
            # Extract weights as (name, tensor) pairs
            weights = [(name, param.data) for name, param in reference_model.named_parameters()]
            
            # Load into our model
            loaded_params = vae_model.load_weights(weights)
            
            logger.info(f"✅ VAE weights loaded: {len(loaded_params)} parameters")
            
            return {
                "loaded_params": len(loaded_params),
                "total_params": len(list(vae_model.named_parameters())),
                "dtype": self.torch_dtype,
                "source": f"{self.pretrained_model_name_or_path}/{subfolder}",
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to load VAE weights: {e}")
            raise
    
    def load_unet_weights(
        self,
        unet_model: torch.nn.Module,
        subfolder: str = "unet",
    ) -> Dict[str, Any]:
        """Load UNet weights.
        
        Args:
            unet_model: The UNet model to load weights into
            subfolder: Subfolder in the repo containing UNet weights
            
        Returns:
            Dict with loading statistics
        """
        logger.info(f"Loading UNet weights from {self.pretrained_model_name_or_path}/{subfolder}")
        
        try:
            from diffusers import UNet2DConditionModel
            
            # Load the reference model
            reference_model = UNet2DConditionModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder=subfolder,
                cache_dir=self.cache_dir,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
            )
            
            # Extract weights as (name, tensor) pairs
            weights = [(name, param.data) for name, param in reference_model.named_parameters()]
            
            # Load into our model
            loaded_params = unet_model.load_weights(weights)
            
            logger.info(f"✅ UNet weights loaded: {len(loaded_params)} parameters")
            
            return {
                "loaded_params": len(loaded_params),
                "total_params": len(list(unet_model.named_parameters())),
                "dtype": self.torch_dtype,
                "source": f"{self.pretrained_model_name_or_path}/{subfolder}",
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to load UNet weights: {e}")
            raise
    
    def load_all_weights(
        self,
        clip_model: torch.nn.Module,
        vae_model: torch.nn.Module,
        unet_model: torch.nn.Module,
    ) -> Dict[str, Any]:
        """Load all Stable Diffusion component weights.
        
        Args:
            clip_model: CLIP text encoder model
            vae_model: VAE (AutoencoderKL) model
            unet_model: UNet denoising model
            
        Returns:
            Dict with comprehensive loading statistics
        """
        logger.info(f"🎯 Loading complete Stable Diffusion weights from {self.pretrained_model_name_or_path}")
        
        results = {}
        
        # Load each component
        try:
            results["clip"] = self.load_clip_weights(clip_model)
            results["vae"] = self.load_vae_weights(vae_model)
            results["unet"] = self.load_unet_weights(unet_model)
            
            # Summary statistics
            total_loaded = sum(r["loaded_params"] for r in results.values())
            total_params = sum(r["total_params"] for r in results.values())
            
            results["summary"] = {
                "total_loaded_params": total_loaded,
                "total_model_params": total_params,
                "loading_coverage": total_loaded / total_params if total_params > 0 else 0.0,
                "dtype": self.torch_dtype,
                "source_repo": self.pretrained_model_name_or_path,
            }
            
            coverage = results["summary"]["loading_coverage"]
            if coverage > 0.99:
                logger.info(f"🎉 All weights loaded successfully! Coverage: {coverage:.1%}")
            else:
                logger.warning(f"⚠️ Partial weight loading. Coverage: {coverage:.1%}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to load Stable Diffusion weights: {e}")
            raise


def load_sd15_weights(
    clip_model: torch.nn.Module,
    vae_model: torch.nn.Module,
    unet_model: torch.nn.Module,
    torch_dtype: torch.dtype = torch.float16,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function to load SD 1.5 weights.
    
    Args:
        clip_model: CLIP text encoder model
        vae_model: VAE model
        unet_model: UNet model
        torch_dtype: Target dtype (fp16/bf16/fp32)
        cache_dir: Cache directory for downloaded weights
        
    Returns:
        Loading statistics dictionary
    """
    loader = StableDiffusionWeightLoader(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
    )
    
    return loader.load_all_weights(clip_model, vae_model, unet_model)


def load_custom_sd_weights(
    clip_model: torch.nn.Module,
    vae_model: torch.nn.Module,
    unet_model: torch.nn.Module,
    pretrained_model_name_or_path: str,
    torch_dtype: torch.dtype = torch.float16,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Load weights from a custom Stable Diffusion checkpoint.
    
    Args:
        clip_model: CLIP text encoder model
        vae_model: VAE model  
        unet_model: UNet model
        pretrained_model_name_or_path: HF repo or local path
        torch_dtype: Target dtype
        cache_dir: Cache directory
        
    Returns:
        Loading statistics dictionary
    """
    loader = StableDiffusionWeightLoader(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
    )
    
    return loader.load_all_weights(clip_model, vae_model, unet_model)


def get_optimal_dtype(device: Union[str, torch.device]) -> torch.dtype:
    """Get the optimal dtype for the given device.
    
    Args:
        device: Target device
        
    Returns:
        Recommended dtype (fp16 for CUDA, fp32 for CPU)
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    if device.type == "cuda":
        # Check if bfloat16 is supported (Ampere and newer)
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    else:
        # CPU or other devices
        return torch.float32


def verify_weight_loading(
    model: torch.nn.Module,
    expected_param_count: Optional[int] = None,
    check_device: Optional[torch.device] = None,
    check_dtype: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    """Verify that model weights were loaded correctly.
    
    Args:
        model: The model to verify
        expected_param_count: Expected number of parameters
        check_device: Expected device for all parameters
        check_dtype: Expected dtype for all parameters
        
    Returns:
        Verification results
    """
    results = {
        "total_params": 0,
        "device_mismatches": 0,
        "dtype_mismatches": 0,
        "uninitialized_params": 0,
        "devices": set(),
        "dtypes": set(),
    }
    
    for name, param in model.named_parameters():
        results["total_params"] += 1
        results["devices"].add(str(param.device))
        results["dtypes"].add(str(param.dtype))
        
        # Check device
        if check_device and param.device != check_device:
            results["device_mismatches"] += 1
        
        # Check dtype
        if check_dtype and param.dtype != check_dtype:
            results["dtype_mismatches"] += 1
        
        # Check for uninitialized parameters (all zeros or very small values)
        if torch.allclose(param.data, torch.zeros_like(param.data), atol=1e-8):
            results["uninitialized_params"] += 1
    
    # Convert sets to lists for JSON serialization
    results["devices"] = list(results["devices"])
    results["dtypes"] = list(results["dtypes"])
    
    # Check parameter count
    if expected_param_count:
        results["param_count_match"] = results["total_params"] == expected_param_count
    
    # Overall health check
    results["healthy"] = (
        results["device_mismatches"] == 0 and
        results["dtype_mismatches"] == 0 and
        results["uninitialized_params"] == 0 and
        (not expected_param_count or results.get("param_count_match", True))
    )
    
    return results
