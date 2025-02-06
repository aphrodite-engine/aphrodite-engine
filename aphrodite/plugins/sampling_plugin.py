from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from aphrodite.common.sampling_params import SamplerID


@dataclass
class SamplingPluginMetadata:
    """Metadata for sampling plugin tensors"""
    tensor_names: List[str]  # Names of tensors to add to SamplingTensors
    param_names: List[str]   # Names of params to add to SamplingParams
    sampler_id: SamplerID    # Unique ID for this sampling method
    default_position: Optional[int] = None  # Default position in sampler order, None for last

class SamplingPlugin(ABC):
    """Base class for sampling method plugins"""
    
    @abstractmethod
    def get_metadata(self) -> SamplingPluginMetadata:
        """Return metadata about tensors and params needed by this plugin"""
        pass

    @abstractmethod
    def verify_params(self, params: Dict[str, Any]) -> None:
        """Verify sampling parameters are valid"""
        pass

    @abstractmethod
    def should_apply(self, params: Dict[str, Any]) -> bool:
        """Check if this sampling method should be applied based on params"""
        pass
        
    @abstractmethod
    def create_tensors(
        self,
        params: Dict[str, Any],
        n_seqs: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Dict[str, torch.Tensor]:
        """Create tensors needed for sampling"""
        pass

    @abstractmethod
    def apply_sampling(
        self,
        logits: torch.Tensor,
        tensors: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply the sampling method to modify logits"""
        pass
