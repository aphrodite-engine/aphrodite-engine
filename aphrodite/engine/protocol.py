import enum
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterable, Mapping
from typing import Any

from aphrodite.common.pooling_params import PoolingParams
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.config import AphroditeConfig, ModelConfig
from aphrodite.inputs.data import PromptType
from aphrodite.logger import init_logger
from aphrodite.lora.request import LoRARequest
from aphrodite.outputs import PoolingRequestOutput, RequestOutput
from aphrodite.plugins.io_processors import IOProcessor
from aphrodite.tasks import SupportedTask
from aphrodite.transformers_utils.tokenizer import AnyTokenizer
from aphrodite.v1.engine import EngineCoreRequest
from aphrodite.v1.engine.processor import Processor

logger = init_logger(__name__)


class Device(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()


class EngineClient(ABC):
    """Protocol class for Clients to Engine"""

    aphrodite_config: AphroditeConfig
    model_config: ModelConfig
    processor: Processor
    io_processor: IOProcessor | None

    @property
    @abstractmethod
    def is_running(self) -> bool: ...

    @property
    @abstractmethod
    def is_stopped(self) -> bool: ...

    @property
    @abstractmethod
    def errored(self) -> bool: ...

    @property
    @abstractmethod
    def dead_error(self) -> BaseException: ...

    @abstractmethod
    def generate(
        self,
        prompt: EngineCoreRequest | PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request."""
        ...

    @abstractmethod
    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        truncate_prompt_tokens: int | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Generate outputs for a request from a pooling model."""
        ...

    @abstractmethod
    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort a request.

        Args:
            request_id: The unique id of the request,
                        or an iterable of such ids.
        """
        ...

    @abstractmethod
    async def get_tokenizer(self) -> AnyTokenizer:
        """Get the tokenizer"""
        ...

    @abstractmethod
    async def is_tracing_enabled(self) -> bool: ...

    @abstractmethod
    async def do_log_stats(self) -> None: ...

    @abstractmethod
    async def check_health(self) -> None:
        """Raise if unhealthy"""
        ...

    @abstractmethod
    async def start_profile(self) -> None:
        """Start profiling the engine"""
        ...

    @abstractmethod
    async def stop_profile(self) -> None:
        """Stop profiling the engine"""
        ...

    @abstractmethod
    async def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache"""
        ...

    @abstractmethod
    async def reset_prefix_cache(self, device: Device | None = None) -> None:
        """Reset the prefix cache"""
        ...

    @abstractmethod
    async def sleep(self, level: int = 1) -> None:
        """Sleep the engine"""
        ...

    @abstractmethod
    async def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up the engine"""
        ...

    @abstractmethod
    async def is_sleeping(self) -> bool:
        """Check whether the engine is sleeping"""
        ...

    @abstractmethod
    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        ...

    async def scale_elastic_ep(self, new_data_parallel_size: int, drain_timeout: int = 300) -> None:
        """Scale the engine"""
        raise NotImplementedError

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        """Perform a collective RPC call to the given path."""
        raise NotImplementedError

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """Get supported tasks"""
        raise NotImplementedError
