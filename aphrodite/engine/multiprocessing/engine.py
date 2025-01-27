import os
import pickle
import signal
import sys
from contextlib import contextmanager
from typing import Iterator, List, Optional, Union

import cloudpickle
import zmq
from loguru import logger

from aphrodite import AphroditeEngine, AsyncEngineArgs, SamplingParams
from aphrodite.common.config import (DecodingConfig, LoRAConfig, ModelConfig,
                                     ParallelConfig, SchedulerConfig)
from aphrodite.common.outputs import RequestOutput
from aphrodite.engine.multiprocessing import (APHRODITE_RPC_SUCCESS_STR,
                                              ENGINE_DEAD_ERROR, IPC_DATA_EXT,
                                              IPC_HEALTH_EXT, IPC_INPUT_EXT,
                                              IPC_OUTPUT_EXT,
                                              REQUEST_OUTPUTS_T,
                                              RPCAbortRequest, RPCError,
                                              RPCHealthRequest,
                                              RPCProcessRequest,
                                              RPCShutdownRequest,
                                              RPCStartupRequest,
                                              RPCStartupResponse)

CONFIG_TYPE = Union[ModelConfig, DecodingConfig, ParallelConfig,
                    SchedulerConfig, LoRAConfig]

POLLING_TIMEOUT_MS = 10000
HEALTHY_RESPONSE = (pickle.dumps(APHRODITE_RPC_SUCCESS_STR), )


class MQAphroditeEngine:
    """A multiprocessing wrapper for :class:`AphroditeEngine`.

    This class is used to wrap the :class:`AphroditeEngine` class to enable use
    in concurrnet manner. It runs a background loop and uses zeromq to 
    receive new requests and stream outputs incrementally via ipc.
    
    The :class:`AphroditeEngine.generate` is kicked off when a new 
    RPCGenerateRequest is received by the input_socket.
    
    The self.engine_loop checks the input_socket for new requests,
    adds them to the AphroditeEngine if there are any, calls the internal
    :class:`AphroditeEngine.step()`, and sends the RequestOutputs back over
    the output_socket.

    If use_async_sockets is set, the logic associated with reading new
    requests from the socket and sending data to the socket is passed
    as a callback to the llm_engine, which calls the logic asynchronously
    such that the IPC can be overlapped with the GPU.

    Args:
        ipc_path: Base path for zeromq interprocess messaging
        use_async_sockets: Whether to make send/recv async with GPU
        log_requests: Whether to log the requests.
        *args: Arguments for :class:`AphroditeEngine`.
        **kwargs: Arguments for :class:`AphroditeEngine`.
    """

    def __init__(self,
                 ipc_path: str,
                 use_async_sockets: bool,
                 *args,
                 log_requests: bool = True,
                 **kwargs) -> None:
        # For MQAphroditeEngine, we can use cached outputs, since each new
        # request output is immediately pickled and send over the socket,
        # which frees the python object to be reused again.
        use_cached_outputs = True
        self.engine = AphroditeEngine(*args,
                                **kwargs,
                                use_cached_outputs=use_cached_outputs)
        self.log_requests = log_requests

        self.use_async_sockets = use_async_sockets
        if self.use_async_sockets:
            self.engine.process_request_outputs_callback = \
                self._async_socket_engine_callback

        self.ctx = zmq.Context()  # type: ignore[attr-defined]

        # Receive input from the client.
        self.input_socket = self.ctx.socket(zmq.constants.PULL)
        self.input_socket.bind(f"{ipc_path}{IPC_INPUT_EXT}")

        # Send output stream back to client.
        self.output_socket = self.ctx.socket(zmq.constants.PUSH)
        self.output_socket.bind(f"{ipc_path}{IPC_OUTPUT_EXT}")

        # Send health status back to client.
        self.health_socket = self.ctx.socket(zmq.constants.PUSH)
        self.health_socket.bind(f"{ipc_path}{IPC_HEALTH_EXT}")

        # IPC path for the data socket.
        self.data_ipc_path = f"{ipc_path}{IPC_DATA_EXT}"

        # Error state.
        self._errored_with: Optional[BaseException] = None

    @property
    def dead_error(self) -> BaseException:
        if self._errored_with is not None:
            return ENGINE_DEAD_ERROR(self._errored_with)
        else:
            return ENGINE_DEAD_ERROR()

    @classmethod
    def from_engine_args(cls, engine_args: AsyncEngineArgs, ipc_path: str):
        """Creates an MQAphroditeEngine from the engine arguments."""

        engine_config = engine_args.create_engine_config()

        executor_class = AphroditeEngine._get_executor_cls(engine_config)

        return cls(
            ipc_path=ipc_path,
            use_async_sockets=engine_config.model_config.use_async_output_proc,
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats)

    def start(self):
        try:
            try:
                logger.debug("Starting Startup Loop.")
                self.run_startup_loop()
                logger.debug("Starting Engine Loop.")
                self.run_engine_loop()
            except Exception as e:
                logger.exception(repr(e))
        except KeyboardInterrupt:
            logger.debug("Shutting down MQAphroditeEngine.")
        finally:
            logger.debug("MQAphroditeEngine is shut down.")
            self.cleanup()

    def cleanup(self):
        """Cleanup zeromq state on shutdown."""
        # Closes all sockets and destroys context.
        self.ctx.destroy(linger=0)
        del self.engine

    @contextmanager
    def make_data_socket(
            self) -> Iterator[zmq.Socket]:  # type: ignore[name-defined]
        socket = self.ctx.socket(zmq.constants.ROUTER)
        try:
            socket.bind(self.data_ipc_path)
            yield socket
        finally:
            socket.close(linger=0)

    def run_startup_loop(self) -> None:
        """Startup loop for sending data from Engine -> Client."""

        with self.make_data_socket() as socket:
            response: Union[RPCStartupResponse, BaseException]
            try:
                identity, message = socket.recv_multipart(copy=False)
                request: RPCStartupRequest = pickle.loads(message.buffer)

                # Handle the query from the Client.
                if request == RPCStartupRequest.IS_SERVER_READY:
                    response = RPCStartupResponse(
                        tracing_enabled=False)

            except Exception as e:
                response = e

            socket.send_multipart((identity, pickle.dumps(response)),
                                  copy=False)

    def run_engine_loop(self):
        """Core busy loop of the AphroditeEngine."""

        while True:
            if not self.engine.has_unfinished_requests():
                # Poll until there is work to do.
                while self.input_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                    self.engine.do_log_stats()
                    logger.debug("Waiting for new requests in engine loop.")

            # Handle any input from the client.
            self.handle_new_input()

            # Engine step.
            request_outputs = self.engine_step()

            # Send request outputs (if async, done in engine_step callback).
            if not self.use_async_sockets:
                self._send_outputs(request_outputs)

    def engine_step(self) -> List[RequestOutput]:
        """Engine step wrapper with error handling."""

        try:
            return self.engine.step()
        except SystemExit:
            raise
        except BaseException as e:
            self._set_errored(e)
            rpc_err = RPCError(request_id=None,
                               is_engine_errored=True,
                               exception=e)
            self._send_outputs(rpc_err)
            raise e

    def handle_new_input(self):
        """Handle new input from the socket"""
        try:
            while self.input_socket.poll(timeout=0) != 0:
                frames = self.input_socket.recv_multipart(copy=False)
                request = pickle.loads(frames[0].buffer)

                if isinstance(request, RPCProcessRequest):
                    if len(frames) > 1:
                        # Use cloudpickle for logits processors
                        assert isinstance(request.params, SamplingParams)
                        lprocs = cloudpickle.loads(frames[1].buffer)
                        request.params.logits_processors = lprocs
                    self._handle_process_request(request)
                elif isinstance(request, RPCAbortRequest):
                    self._handle_abort_request(request)
                elif isinstance(request, RPCHealthRequest):
                    self._handle_health_request()
                elif isinstance(request, RPCShutdownRequest):
                    self.engine.shutdown()
                    self._send_outputs(APHRODITE_RPC_SUCCESS_STR)
                    break
                else:
                    raise ValueError("Unknown RPCRequest Type: {request}")

        except Exception as e:
            self._set_errored(e)
            self._send_unhealthy(e)
            raise e

    def _handle_process_request(self, request: RPCProcessRequest):
        """Handle RPCProcessRequest by  adding it to the AphroditeEngine."""
        request_id = request.request_id

        if self._errored_with is not None:
            rpc_err = RPCError(request_id=request_id,
                               is_engine_errored=True,
                               exception=ENGINE_DEAD_ERROR(self._errored_with))
            self._send_outputs(rpc_err)

        try:
            self.engine.add_request(
                request_id=request_id,
                prompt=request.prompt,
                params=request.params,
                lora_request=request.lora_request,
                prompt_adapter_request=request.prompt_adapter_request)

            if self.log_requests:
                logger.info(f"Added request {request.request_id}.")

        except Exception as e:
            # We do not set self._errored = True here, since the error
            # is due to an issue adding this request to the engine,
            # rather than an issue with the engine itself.
            is_errored = self._errored_with is not None
            rpc_err = RPCError(request_id=request_id,
                               is_engine_errored=is_errored,
                               exception=e)
            self._send_outputs(rpc_err)

            # Remove request from the engine.
            self.engine.abort_request(request_id)

    def _handle_abort_request(self, request: RPCAbortRequest):
        self.engine.abort_request(request.request_id)
        if self.log_requests:
            logger.info(f"Aborted request {request.request_id}.")

    def _handle_health_request(self):
        if self._errored_with is not None:
            self._send_unhealthy(self._errored_with)

        # Raises error if unhealthy.
        self.engine.check_health()
        self._send_healthy()

    def _send_outputs(self, outputs: REQUEST_OUTPUTS_T):
        """Send List of RequestOutput to RPCClient."""
        if outputs:
            output_bytes = pickle.dumps(outputs)
            self.output_socket.send_multipart((output_bytes, ), copy=False)

    def _send_healthy(self):
        """Send HEALTHY message to RPCClient."""
        self.health_socket.send_multipart(HEALTHY_RESPONSE, copy=False)

    def _send_unhealthy(self, error: BaseException):
        """Send UNHEALTHY message to RPCClient."""
        error_bytes = pickle.dumps(error)
        self.health_socket.send_multipart((error_bytes, ), copy=False)

    def _async_socket_engine_callback(self,
                                      request_outputs: REQUEST_OUTPUTS_T):
        """Callback used by engine to make socket handling async with GPU."""
        self._send_outputs(request_outputs)
        self.handle_new_input()

    def _set_errored(self, e: BaseException):
        """Log and set errored status if this is the first issue."""
        if self._errored_with is None:
            self._errored_with = e


def run_mp_engine(engine_args: AsyncEngineArgs, ipc_path: str):
    def signal_handler(*_) -> None:
        with open(os.devnull, 'w') as devnull:
            sys.stderr = devnull
        raise KeyboardInterrupt("MQAphroditeEngine terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    try:
        engine = MQAphroditeEngine.from_engine_args(engine_args=engine_args,
                                              ipc_path=ipc_path)
        engine.start()
    except KeyboardInterrupt as e:
        if str(e) == "MQAphroditeEngine terminated":
            pass
        else:
            raise
