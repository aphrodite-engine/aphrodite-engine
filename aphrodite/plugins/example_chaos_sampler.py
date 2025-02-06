import torch
from loguru import logger

from aphrodite.common.sampler_ids import SamplerID
from aphrodite.plugins.sampling_plugin import (SamplingPlugin,
                                               SamplingPluginMetadata)


class ChaosSamplerPlugin(SamplingPlugin):
    def get_metadata(self) -> SamplingPluginMetadata:
        try:
            sampler_id = getattr(SamplerID, "CHAOS")
        except AttributeError:
            sampler_id = SamplerID.register("CHAOS", 100)
            
        return SamplingPluginMetadata(
            tensor_names=["chaos_enabled"],
            param_names=["chaos_enabled"],
            sampler_id=sampler_id,
            default_position=None
        )

    def verify_params(self, params: dict) -> None:
        enabled = params.get("chaos_enabled", 0)
        if not isinstance(enabled, int) or enabled not in (0, 1):
            raise ValueError("chaos_enabled must be 0 or 1")

    def should_apply(self, params: dict) -> bool:
        return params.get("chaos_enabled", 0) == 1

    def create_tensors(
        self,
        params: dict,
        n_seqs: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> dict:
        return {
            "chaos_enabled": torch.full((n_seqs,), 
                                     params.get("chaos_enabled", 0),
                                     device=device,
                                     dtype=torch.int)
        }

    def apply_sampling(
        self,
        logits: torch.Tensor,
        tensors: dict
    ) -> torch.Tensor:
        logger.info("Applying chaos sampling!")
        # Completely scramble the logits
        return torch.randn_like(logits) * 100
