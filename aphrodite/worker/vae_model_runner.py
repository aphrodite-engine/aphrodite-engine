"""VAE model runner for image encoding/decoding tasks in Aphrodite."""

import dataclasses
from typing import Any, Optional

import torch

from aphrodite.common.sequence import IntermediateTensors
from aphrodite.config import AphroditeConfig
from aphrodite.worker.model_runner import (GPUModelRunnerBase,
                                           ModelInputForGPU,
                                           ModelInputForGPUBuilder)


@dataclasses.dataclass(frozen=True)
class ModelInputForGPUWithVAEMetadata(ModelInputForGPU):
    """Model input for GPU with VAE-specific metadata."""

    # VAE-specific fields
    vae_operation: Optional[str] = None  # "encode" or "decode"
    image_data: Optional[torch.Tensor] = None  # For encoding
    latent_data: Optional[torch.Tensor] = None  # For decoding


class VAEModelRunner(GPUModelRunnerBase[ModelInputForGPUWithVAEMetadata]):
    """Model runner for VAE (Variational Autoencoder) models.

    This runner handles image encoding and latent decoding operations
    for Stable Diffusion VAE models.
    """

    _model_input_cls: type[ModelInputForGPUWithVAEMetadata] = \
        ModelInputForGPUWithVAEMetadata
    _builder_cls: type[ModelInputForGPUBuilder] = ModelInputForGPUBuilder

    def __init__(
        self,
        aphrodite_config: AphroditeConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
    ):
        super().__init__(
            aphrodite_config=aphrodite_config,
            kv_cache_dtype=kv_cache_dtype,
            is_driver_worker=is_driver_worker
        )

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPUWithVAEMetadata,
        kv_caches: list[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[torch.Tensor]:
        """Execute VAE model for encoding or decoding."""

        if num_steps > 1:
            raise ValueError(
                "VAEModelRunner does not support multi-step execution.",
            )

        model_executable = self.model

        # Handle VAE operations
        if model_input.vae_operation == "encode":
            if model_input.image_data is None:
                raise ValueError("Image data required for VAE encoding")

            # Encode images to latents
            output = model_executable.encode(model_input.image_data)

        elif model_input.vae_operation == "decode":
            if model_input.latent_data is None:
                raise ValueError("Latent data required for VAE decoding")

            # Decode latents to images
            output = model_executable.decode(model_input.latent_data)

        elif model_input.vae_operation is None:
            # Handle profiling/dummy run case - just return dummy output
            dummy_latent = torch.zeros(
                1, 4, 8, 8, device=self.device, dtype=self.model_config.dtype,
            )
            output = model_executable.decode(dummy_latent)

        else:
            raise ValueError(
                f"Unsupported VAE operation: {model_input.vae_operation}",
            )

        return output

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: dict[str, Any],
    ) -> ModelInputForGPUWithVAEMetadata:
        """Create model input from broadcasted tensor dict."""
        return ModelInputForGPUWithVAEMetadata(
            input_tokens=tensor_dict.get("input_tokens", torch.tensor([])),
            input_positions=tensor_dict.get(
                "input_positions", torch.tensor([]),
            ),
            attn_metadata=tensor_dict.get("attn_metadata"),
            seq_lens=tensor_dict.get("seq_lens"),
            query_lens=tensor_dict.get("query_lens"),
            lora_mapping=tensor_dict.get("lora_mapping"),
            lora_requests=tensor_dict.get("lora_requests"),
            multi_modal_kwargs=tensor_dict.get("multi_modal_kwargs"),
            virtual_engine=tensor_dict.get("virtual_engine", 0),
            vae_operation=tensor_dict.get("vae_operation"),
            image_data=tensor_dict.get("image_data"),
            latent_data=tensor_dict.get("latent_data"),
        )

    def prepare_model_input(
        self,
        seq_group_metadata_list: list,
        virtual_engine: int = 0,
        finished_requests_ids: Optional[list[str]] = None,
    ) -> ModelInputForGPUWithVAEMetadata:
        """Prepare VAE model input."""

        # For VAE, we'll need to handle image/latent data differently
        # This is a simplified version - we'll need to integrate with the
        # request system

        return ModelInputForGPUWithVAEMetadata(
            input_tokens=torch.tensor([]),  # VAE doesn't use tokens
            input_positions=torch.tensor([]),
            attn_metadata=None,
            seq_lens=None,
            query_lens=None,
            lora_mapping=None,
            lora_requests=None,
            multi_modal_kwargs=None,
            virtual_engine=virtual_engine,
            vae_operation=None,  # Will be set by the calling code
            image_data=None,
            latent_data=None,
        )
