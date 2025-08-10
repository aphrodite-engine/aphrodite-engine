from typing import List, Optional

import torch

from aphrodite.common.config import AphroditeConfig
from aphrodite.common.sequence import IntermediateTensors
from aphrodite.modeling.layers.sampler import SamplerOutput
from aphrodite.multimodal import MultiModalKwargs
from aphrodite.worker.neuronx_distributed_model_runner import (
    NeuronxDistributedModelRunner)


class MultiStepNeuronxDistributedModelRunner(NeuronxDistributedModelRunner):
    """A model runner for multi-step decoding using the
    neuronx-distributed-inference framework"""

    def __init__(
        self,
        aphrodite_config: AphroditeConfig,
    ):
        super().__init__(aphrodite_config)

    def load_model(self) -> None:
        from aphrodite.modeling.model_loader.neuronx_distributed import (
            get_neuron_speculation_model)
        self.model = get_neuron_speculation_model(
            self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            speculation_config=self.speculative_config)

    @torch.inference_mode()
    def execute_model(
        self,
        model_input,
        kv_caches: Optional[List[torch.Tensor]] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        sampling_params = torch.tensor([[
            seq_group.sampling_params.top_k,
            seq_group.sampling_params.top_p,
            seq_group.sampling_params.temperature,
        ] for seq_group in model_input.sampling_metadata.seq_groups])

        logits = self.model(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            input_block_ids=model_input.input_block_ids,
            sampling_params=sampling_params,
            **MultiModalKwargs.as_kwargs(
                model_input.multi_modal_kwargs or {},
                device=self.device,
            ),
        )

        output = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        return output
