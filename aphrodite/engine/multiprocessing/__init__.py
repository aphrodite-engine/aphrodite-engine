from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from aphrodite import PoolingParams
from aphrodite.common.outputs import RequestOutput
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.inputs import PromptType
from aphrodite.lora.request import LoRARequest
from aphrodite.prompt_adapter.request import PromptAdapterRequest

APHRODITE_RPC_SUCCESS_STR = "SUCCESS"

IPC_INPUT_EXT = "_input_socket"
IPC_OUTPUT_EXT = "_output_socket"
IPC_HEALTH_EXT = "_health_socket"
IPC_DATA_EXT = "_data_socket"


class MQEngineDeadError(RuntimeError):
    pass


@dataclass
class RPCProcessRequest:
    prompt: PromptType
    params: Union[SamplingParams, PoolingParams]
    request_id: str
    lora_request: Optional[LoRARequest] = None
    prompt_adapter_request: Optional[PromptAdapterRequest] = None
    priority: int = 0


@dataclass
class RPCError:
    request_id: Optional[str]
    is_engine_errored: bool
    exception: BaseException


@dataclass
class RPCAbortRequest:
    request_id: str



class RPCStartupRequest(Enum):
    IS_SERVER_READY = 1


@dataclass
class RPCStartupResponse:
    tracing_enabled: bool


@dataclass
class RPCShutdownRequest:
    pass


RPC_REQUEST_T = Union[RPCProcessRequest, RPCAbortRequest, RPCStartupRequest]

REQUEST_OUTPUTS_T = Union[List[RequestOutput], RPCError]


def ENGINE_DEAD_ERROR(
        error: Optional[BaseException] = None) -> MQEngineDeadError:
    if error is None:
        return MQEngineDeadError(
            "Engine loop is not running. Inspect the stacktrace to "
            "find the original error")

    return MQEngineDeadError(
        "Engine loop is not running. Inspect the stacktrace to "
        f"find the original error: {repr(error)}.")
