from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Generic, TypeVar

from aphrodite.common.pooling_params import PoolingParams
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.config import AphroditeConfig
from aphrodite.endpoints.openai.protocol import IOProcessorResponse
from aphrodite.inputs.data import PromptType
from aphrodite.outputs import PoolingRequestOutput

IOProcessorInput = TypeVar("IOProcessorInput")
IOProcessorOutput = TypeVar("IOProcessorOutput")


class IOProcessor(ABC, Generic[IOProcessorInput, IOProcessorOutput]):
    def __init__(self, aphrodite_config: AphroditeConfig):
        self.aphrodite_config = aphrodite_config

    @abstractmethod
    def pre_process(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        raise NotImplementedError

    async def pre_process_async(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        return self.pre_process(prompt, request_id, **kwargs)

    @abstractmethod
    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        raise NotImplementedError

    async def post_process_async(
        self,
        model_output: AsyncGenerator[tuple[int, PoolingRequestOutput]],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        # We cannot guarantee outputs are returned in the same order they were
        # fed to Aphrodite.
        # Let's sort them by id before post_processing
        sorted_output = sorted(
            [(i, item) async for i, item in model_output], key=lambda output: output[0]
        )
        collected_output = [output[1] for output in sorted_output]
        return self.post_process(collected_output, request_id, **kwargs)

    @abstractmethod
    def parse_request(self, request: Any) -> IOProcessorInput:
        raise NotImplementedError

    def validate_or_generate_params(
        self, params: SamplingParams | PoolingParams | None = None
    ) -> SamplingParams | PoolingParams:
        return params or PoolingParams()

    @abstractmethod
    def output_to_response(
        self, plugin_output: IOProcessorOutput
    ) -> IOProcessorResponse:
        raise NotImplementedError
