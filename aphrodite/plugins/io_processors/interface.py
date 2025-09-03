from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Generic, Optional, TypeVar, Union

from aphrodite.config import AphroditeConfig
from aphrodite.endpoints.openai.protocol import IOProcessorResponse
from aphrodite.inputs.data import PromptType
from aphrodite.common.outputs import PoolingRequestOutput

IOProcessorInput = TypeVar('IOProcessorInput')
IOProcessorOutput = TypeVar('IOProcessorOutput')


class IOProcessor(ABC, Generic[IOProcessorInput, IOProcessorOutput]):

    def __init__(self, aphrodite_config: AphroditeConfig):
        self.aphrodite_config = aphrodite_config

    @abstractmethod
    def pre_process(
        self,
        prompt: IOProcessorInput,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Union[PromptType, Sequence[PromptType]]:
        raise NotImplementedError

    async def pre_process_async(
        self,
        prompt: IOProcessorInput,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Union[PromptType, Sequence[PromptType]]:
        return self.pre_process(prompt, request_id, **kwargs)

    @abstractmethod
    def post_process(self,
                     model_output: Sequence[PoolingRequestOutput],
                     request_id: Optional[str] = None,
                     **kwargs) -> IOProcessorOutput:
        raise NotImplementedError

    async def post_process_async(
        self,
        model_output: AsyncGenerator[tuple[int, PoolingRequestOutput]],
        request_id: Optional[str] = None,
        **kwargs,
    ) -> IOProcessorOutput:
        collected_output = [item async for i, item in model_output]
        return self.post_process(collected_output, request_id, **kwargs)

    @abstractmethod
    def parse_request(self, request: Any) -> IOProcessorInput:
        raise NotImplementedError

    @abstractmethod
    def output_to_response(
            self, plugin_output: IOProcessorOutput) -> IOProcessorResponse:
        raise NotImplementedError
