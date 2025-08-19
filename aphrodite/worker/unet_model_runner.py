"""UNet model runner for diffusion models."""

import dataclasses
from typing import List, Optional, Union

import torch

from aphrodite.common.sequence import IntermediateTensors
from aphrodite.config import AphroditeConfig
from aphrodite.worker.gpu_model_runner import (GPUModelRunnerBase,
                                               ModelInputForGPUBuilder)
from aphrodite.worker.model_runner_base import (ModelRunnerInputBase,
                                                ModelRunnerInputBuilderBase)


@dataclasses.dataclass(frozen=True)
class ModelInputForGPUWithUNetMetadata(ModelRunnerInputBase):
    """
    Model input for UNet models, containing sample, timestep, and
    encoder hidden states.
    Used by UNetModelRunner.
    """
    # Noisy sample to denoise
    sample: Optional[torch.Tensor] = None
    # Timestep for diffusion process
    timestep: Optional[torch.Tensor] = None
    # Text embeddings for conditioning
    encoder_hidden_states: Optional[torch.Tensor] = None
    # UNet operation type
    unet_operation: Optional[str] = None

    def as_broadcastable_tensor_dict(self) -> dict:
        tensor_dict = {}
        if self.sample is not None:
            tensor_dict["sample"] = self.sample
        if self.timestep is not None:
            tensor_dict["timestep"] = self.timestep
        if self.encoder_hidden_states is not None:
            tensor_dict["encoder_hidden_states"] = self.encoder_hidden_states
        if self.unet_operation is not None:
            tensor_dict["unet_operation"] = self.unet_operation
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: dict,
        attn_backend=None,
    ) -> "ModelInputForGPUWithUNetMetadata":
        return cls(
            sample=tensor_dict.pop("sample", None),
            timestep=tensor_dict.pop("timestep", None),
            encoder_hidden_states=tensor_dict.pop(
                "encoder_hidden_states", None),
            unet_operation=tensor_dict.pop("unet_operation", None),
        )


class UNetModelRunner(GPUModelRunnerBase[ModelInputForGPUWithUNetMetadata]):
    """
    Model runner for UNet diffusion models.

    This runner handles UNet models that take noisy samples, timesteps, and
    conditioning information to predict noise for denoising.
    """

    _model_input_cls: type[ModelInputForGPUWithUNetMetadata] = \
        ModelInputForGPUWithUNetMetadata
    _builder_cls: type[ModelRunnerInputBuilderBase] = ModelInputForGPUBuilder

    def __init__(
        self,
        aphrodite_config: AphroditeConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
    ):
        super().__init__(
            aphrodite_config=aphrodite_config,
            kv_cache_dtype=kv_cache_dtype,
            is_driver_worker=is_driver_worker,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPUWithUNetMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[Union[List[torch.Tensor], IntermediateTensors]]:
        if num_steps > 1:
            raise ValueError(
                "UNetModelRunner does not support multi-step execution.")

        model_executable = self.model

        if model_input.unet_operation == "denoise_step":
            if (model_input.sample is None
                or model_input.timestep is None
                or model_input.encoder_hidden_states is None):
                raise ValueError(
                    "Sample, timestep, and encoder_hidden_states required for"
                    " UNet denoising"
                )

            output = model_executable(
                sample=model_input.sample,
                timestep=model_input.timestep,
                encoder_hidden_states=model_input.encoder_hidden_states,
            )

        elif model_input.unet_operation is None:
            # Handle profiling/dummy run case - return dummy output
            batch_size = 1
            height = width = 8  # Minimal size for testing
            dummy_sample = torch.zeros(
                batch_size, 4, height, width,
                device=self.device,
                dtype=self.model_config.dtype
            )
            dummy_timestep = torch.zeros(
                batch_size,
                device=self.device,
                dtype=torch.long
            )
            dummy_encoder_hidden_states = torch.zeros(
                batch_size, 77, 768,  # Standard SD text encoder dimensions
                device=self.device,
                dtype=self.model_config.dtype
            )

            output = model_executable(
                sample=dummy_sample,
                timestep=dummy_timestep,
                encoder_hidden_states=dummy_encoder_hidden_states,
            )

        else:
            raise ValueError(
                f"Unsupported UNet operation: {model_input.unet_operation}")

        return output

    def make_model_input_from_broadcasted_tensor_dict(
        self, tensor_dict: dict
    ) -> ModelInputForGPUWithUNetMetadata:
        return ModelInputForGPUWithUNetMetadata.from_broadcasted_tensor_dict(
            tensor_dict
        )

    def prepare_model_input(
        self,
        seq_group_metadata_list,
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> ModelInputForGPUWithUNetMetadata:
        """
        Prepare model input for UNet inference.

        For UNet models, we expect the sequence group metadata to contain
        the necessary tensors (sample, timestep, encoder_hidden_states).
        """
        # For now, return a minimal input that will trigger the dummy path
        # In a full implementation, this would extract the actual diffusion
        # data from the sequence group metadata
        return ModelInputForGPUWithUNetMetadata(
            sample=None,
            timestep=None,
            encoder_hidden_states=None,
            unet_operation=None,  # This will trigger dummy path
        )
