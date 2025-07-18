import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future
from inspect import isclass, signature
from logging import DEBUG
from typing import Any, Callable, Optional, TypeVar, Union

import msgspec
import zmq
from loguru import logger

from aphrodite.common.config import AphroditeConfig, ParallelConfig
from aphrodite.common.utils import resolve_obj_by_qualname, zmq_socket_ctx
from aphrodite.distributed import (
    stateless_destroy_torch_distributed_process_group)
from aphrodite.executor.multiproc_worker_utils import _add_prefix
from aphrodite.lora.request import LoRARequest
from aphrodite.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from aphrodite.v1.core.kv_cache_utils import (get_kv_cache_config,
                                              unify_kv_cache_configs)
from aphrodite.v1.core.sched.interface import SchedulerInterface
from aphrodite.v1.core.sched.output import SchedulerOutput
from aphrodite.v1.core.sched.scheduler import Scheduler as V1Scheduler
from aphrodite.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                                 EngineCoreRequestType, UtilityOutput)
from aphrodite.v1.engine.mm_input_cache import MirroredProcessingCache
from aphrodite.v1.executor.abstract import Executor
from aphrodite.v1.kv_cache_interface import KVCacheConfig
from aphrodite.v1.outputs import ModelRunnerOutput
from aphrodite.v1.request import Request, RequestStatus
from aphrodite.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from aphrodite.v1.structured_output import StructuredOutputManager
from aphrodite.version import __version__ as APHRODITE_VERSION

POLLING_TIMEOUT_S = 2.5

_R = TypeVar('_R')  # Return type for collective_rpc


class EngineCore:
    """Inner loop of Aphrodite's Engine."""

    def __init__(self,
                 aphrodite_config: AphroditeConfig,
                 executor_class: type[Executor],
                 log_stats: bool,
                 executor_fail_callback: Optional[Callable] = None):
        assert aphrodite_config.model_config.runner_type != "pooling"

        display_config = int(os.environ.get("APHRODITE_DISPLAY_CONFIG", "0"))
        log_level = os.environ.get("APHRODITE_LOG_LEVEL", "INFO")

        if display_config == 1 or log_level == "DEBUG":
            logger.info("Initializing a V1 LLM engine (v{}) with config: {}\n",
                        APHRODITE_VERSION, aphrodite_config)
        else:
            logger.info(
                "Initializing a V1 LLM engine (v{}).\n"
                "Pass APHRODITE_DISPLAY_CONFIG=1 to see the full config.",
                APHRODITE_VERSION)

        self.log_stats = log_stats

        # Setup Model.
        self.model_executor = executor_class(aphrodite_config)
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(
                executor_fail_callback)

        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = \
            self._initialize_kv_caches(aphrodite_config)

        aphrodite_config.cache_config.num_gpu_blocks = num_gpu_blocks
        aphrodite_config.cache_config.num_cpu_blocks = num_cpu_blocks

        self.structured_output_manager = StructuredOutputManager(
            aphrodite_config)

        # Setup scheduler.
        if isinstance(aphrodite_config.scheduler_config.scheduler_cls, str):
            Scheduler = resolve_obj_by_qualname(
                aphrodite_config.scheduler_config.scheduler_cls)
        else:
            Scheduler = aphrodite_config.scheduler_config.scheduler_cls

        # This warning can be removed once the V1 Scheduler interface is
        # finalized and we can maintain support for scheduler classes that
        # implement it
        if Scheduler is not V1Scheduler:
            logger.warning(
                "Using configured V1 scheduler class {}. "
                "This scheduler interface is not public and "
                "compatibility may not be maintained.",
                aphrodite_config.scheduler_config.scheduler_cls)

        self.scheduler: SchedulerInterface = Scheduler(
            aphrodite_config=aphrodite_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=aphrodite_config.parallel_config.
            data_parallel_size > 1,
            log_stats=self.log_stats,
        )

        # Setup MM Input Mapper.
        self.mm_input_cache_server = MirroredProcessingCache(
            aphrodite_config.model_config)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: Optional[queue.Queue[tuple[Future[ModelRunnerOutput],
                                                     SchedulerOutput]]] = None
        if self.batch_queue_size > 1:
            logger.info("Batch queue is enabled with size {}",
                        self.batch_queue_size)
            self.batch_queue = queue.Queue(self.batch_queue_size)
        self.aphrodite_config = aphrodite_config

    def _initialize_kv_caches(
            self, aphrodite_config: AphroditeConfig
    ) -> tuple[int, int, KVCacheConfig]:
        start = time.time()

        # Get all kv cache needed by the model
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        # Profiles the peak memory usage of the model to determine how much
        # memory can be allocated for kv cache.
        available_gpu_memory = self.model_executor.determine_available_memory()

        assert len(kv_cache_specs) == len(available_gpu_memory)
        # Get the kv cache tensor size
        kv_cache_configs = [
            get_kv_cache_config(aphrodite_config, kv_cache_spec_one_worker,
                                available_gpu_memory_one_worker)
            for kv_cache_spec_one_worker, available_gpu_memory_one_worker in
            zip(kv_cache_specs, available_gpu_memory)
        ]

        # Since we use a shared centralized controller, we need the
        # `kv_cache_config` to be consistent across all workers to make sure
        # all the memory operators can be applied to all workers.
        unify_kv_cache_configs(kv_cache_configs)

        # All workers have the same kv_cache_config except layer names, so use
        # an arbitrary one to initialize the scheduler.
        assert all([
            cfg.num_blocks == kv_cache_configs[0].num_blocks
            for cfg in kv_cache_configs
        ])
        num_gpu_blocks = kv_cache_configs[0].num_blocks
        num_cpu_blocks = 0
        scheduler_kv_cache_config = kv_cache_configs[0]

        # Initialize kv cache and warmup the execution
        self.model_executor.initialize_from_config(kv_cache_configs)

        elapsed = time.time() - start
        logger.info(("init engine (profile, create kv cache, "
                     "warmup model) took {:.2f} seconds"), elapsed)
        return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config

    def add_request(self, request: EngineCoreRequest):
        """Add request to the scheduler."""

        if request.mm_hashes is not None:
            # Here, if hash exists for a multimodal input, then it will be
            # fetched from the cache, else it will be added to the cache.
            # Note that the cache here is mirrored with the client cache, so
            # anything that has a hash must have a HIT cache entry here
            # as well.
            assert request.mm_inputs is not None
            request.mm_inputs = self.mm_input_cache_server.get_and_update_p1(
                request.mm_inputs, request.mm_hashes)

        req = Request.from_engine_core_request(request)
        if req.use_structured_output:
            # Start grammar compilation asynchronously
            self.structured_output_manager.grammar_init(req)

        self.scheduler.add_request(req)

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids,
                                       RequestStatus.FINISHED_ABORTED)

    def step(self) -> EngineCoreOutputs:
        """Schedule, execute, and make output."""

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return EngineCoreOutputs(
                outputs=[],
                scheduler_stats=self.scheduler.make_stats(),
            )
        scheduler_output = self.scheduler.schedule()
        output = self.model_executor.execute_model(scheduler_output)
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, output)  # type: ignore

        return engine_core_outputs

    def step_with_batch_queue(self) -> Optional[EngineCoreOutputs]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """
        assert self.batch_queue is not None

        engine_core_outputs = None
        scheduler_output = None
        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        if not self.batch_queue.full():
            scheduler_output = self.scheduler.schedule()
            if scheduler_output.total_num_scheduled_tokens > 0:
                future = self.model_executor.execute_model(scheduler_output)
                self.batch_queue.put_nowait(
                    (future, scheduler_output))  # type: ignore

        scheduled_batch = (scheduler_output is not None
                           and scheduler_output.total_num_scheduled_tokens > 0)

        # If no more requests can be scheduled and the job queue is not empty,
        # block until the first batch in the job queue is finished.
        # TODO(comaniac): Ideally we should peek the first batch in the
        # job queue to check if it's finished before scheduling a new batch,
        # but peeking the first element in a queue is not thread-safe,
        # so we need more work.
        if not scheduled_batch and not self.batch_queue.empty():
            future, scheduler_output = self.batch_queue.get_nowait()
            # Blocking until the first result is available.
            model_output = future.result()
            self.batch_queue.task_done()
            engine_core_outputs = self.scheduler.update_from_output(
                scheduler_output, model_output)

        return engine_core_outputs

    def shutdown(self):
        self.structured_output_manager.clear_backend()
        if self.model_executor:
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()

    def profile(self, is_start: bool = True):
        self.model_executor.profile(is_start)

    def reset_prefix_cache(self):
        self.scheduler.reset_prefix_cache()

    def sleep(self, level: int = 1):
        self.model_executor.sleep(level)

    def wake_up(self, tags: Optional[list[str]] = None):
        self.model_executor.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        self.model_executor.collective_rpc("execute_dummy_batch")

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.model_executor.save_sharded_state(path=path,
                                               pattern=pattern,
                                               max_size=max_size)

    def collective_rpc(self,
                       method: Union[str, Callable[..., _R]],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict[str, Any]] = None) -> list[_R]:
        return self.model_executor.collective_rpc(method, timeout, args,
                                                  kwargs)


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b'ENGINE_CORE_DEAD'

    def __init__(
        self,
        input_path: str,
        output_path: str,
        aphrodite_config: AphroditeConfig,
        executor_class: type[Executor],
        log_stats: bool,
        engine_index: int = 0,
    ):
        input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()

        executor_fail_callback = lambda: input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b''))

        super().__init__(aphrodite_config, executor_class, log_stats,
                         executor_fail_callback)

        self.step_fn = (self.step if self.batch_queue is None else
                        self.step_with_batch_queue)
        self.engines_running = False

        # Background Threads and Queues for IO. These enable us to
        # overlap ZMQ socket IO with GPU since they release the GIL,
        # and to overlap some serialization/deserialization with the
        # model forward pass.
        # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
        self.input_queue = input_queue
        self.output_queue = queue.Queue[Union[EngineCoreOutputs, bytes]]()
        threading.Thread(target=self.process_input_socket,
                         args=(input_path, engine_index),
                         daemon=True).start()
        self.output_thread = threading.Thread(
            target=self.process_output_socket,
            args=(output_path, engine_index),
            daemon=True)
        self.output_thread.start()

    @staticmethod
    def run_engine_core(*args,
                        dp_rank: int = 0,
                        local_dp_rank: int = 0,
                        **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: Optional[EngineCoreProc] = None
        try:
            parallel_config: ParallelConfig = kwargs[
                "aphrodite_config"].parallel_config
            if parallel_config.data_parallel_size > 1:
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                parallel_config.data_parallel_rank_local = local_dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                engine_core = EngineCoreProc(*args, **kwargs)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            if engine_core is not None:
                engine_core.shutdown()

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()

    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        waited = False
        while not self.engines_running and not (self.scheduler.has_requests()):
            if logging.getLogger().isEnabledFor(
                    logging.DEBUG) and self.input_queue.empty():
                logger.debug("EngineCore waiting for work.")
                waited = True
            req = self.input_queue.get()
            self._handle_client_request(*req)

        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self):
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        outputs = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        if outputs is not None:
            self.output_queue.put_nowait(outputs)

    def _handle_client_request(self, request_type: EngineCoreRequestType,
                               request: Any) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.ADD:
            self.add_request(request)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            call_id, method_name, args = request
            output = UtilityOutput(call_id)
            try:
                method = getattr(self, method_name)
                output.result = method(
                    *self._convert_msgspec_args(method, args))
            except BaseException as e:
                logger.exception("Invocation of {} method failed", method_name)
                output.failure_message = (f"Call to {method_name} method"
                                          f" failed: {str(e)}")
            self.output_queue.put_nowait(
                EngineCoreOutputs(utility_output=output))
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error("Unrecognized input request type encountered: {}",
                         request_type)

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
         arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation) if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation) else v
            for v, p in zip(args, arg_types))

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal("Aphrodite shutdown signal from EngineCore failed "
                         "to send. Please report this issue.")

    def process_input_socket(self, input_path: str, engine_index: int):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()
        identity = engine_index.to_bytes(length=2, byteorder="little")

        with zmq_socket_ctx(input_path,
                            zmq.DEALER,
                            identity=identity,
                            bind=False) as socket:

            # Send ready message to front-end once input socket is connected.
            message_dict = {
                'type':
                'READY',
                'num_gpu_blocks':
                self.aphrodite_config.cache_config.num_gpu_blocks,
            }
            message = json.dumps(message_dict).encode('utf-8')
            socket.send(message)

            while True:
                # (RequestType, RequestData)
                type_frame, *data_frames = socket.recv_multipart(copy=False)
                request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                # Deserialize the request data.
                decoder = add_request_decoder if (
                    request_type
                    == EngineCoreRequestType.ADD) else generic_decoder
                request = decoder.decode(data_frames)

                # Push to input queue for core busy loop.
                self.input_queue.put_nowait((request_type, request))

    def process_output_socket(self, output_path: str, engine_index: int):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Keep references to outputs and buffers until zmq is finished
        # with them (outputs may contain tensors/np arrays whose
        # backing buffers were extracted for zero-copy send).
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with zmq_socket_ctx(output_path, zmq.constants.PUSH,
                            linger=4000) as socket:
            while True:
                outputs = self.output_queue.get()
                if outputs == EngineCoreProc.ENGINE_CORE_DEAD:
                    socket.send(outputs, copy=False)
                    break
                assert not isinstance(outputs, bytes)
                outputs.engine_index = engine_index

                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = socket.send_multipart(buffers,
                                                copy=False,
                                                track=True)
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < 2:
                    # Keep at most 2 buffers to reuse.
                    reuse_buffers.append(buffer)


class DPEngineCoreProc(EngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        aphrodite_config: AphroditeConfig,
        executor_class: type[Executor],
        log_stats: bool,
    ):
        # Add process-specific prefix to stdout and stderr before
        # we initialize the engine.
        from multiprocessing import current_process
        process_name = current_process().name
        pid = os.getpid()
        _add_prefix(sys.stdout, process_name, pid)
        _add_prefix(sys.stderr, process_name, pid)

        dp_size = aphrodite_config.parallel_config.data_parallel_size
        dp_rank = aphrodite_config.parallel_config.data_parallel_rank
        local_dp_rank = aphrodite_config.parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        from aphrodite.platforms import current_platform
        if current_platform.is_cuda_alike():
            from aphrodite.platforms.cuda import (
                device_id_to_physical_device_id)
            tp_size = aphrodite_config.parallel_config.tensor_parallel_size
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(device_id_to_physical_device_id(i))
                for i in range(local_dp_rank * tp_size, (local_dp_rank + 1) *
                               tp_size))

        self.local_dp_rank = local_dp_rank
        self.dp_group = aphrodite_config.parallel_config.stateless_init_dp_group(
        )
        self.current_wave = 0

        # Initialize the engine after setting up environment.
        super().__init__(input_path, output_path, aphrodite_config,
                         executor_class, log_stats, dp_rank)

        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        self.counter = 0

    def shutdown(self):
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def add_request(self, request: EngineCoreRequest):
        if request.current_wave != self.current_wave:
            if request.current_wave > self.current_wave:
                self.current_wave = request.current_wave
            elif not self.engines_running:
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.output_queue.put_nowait(
                    EngineCoreOutputs(start_wave=self.current_wave))

        super().add_request(request)

    def _handle_client_request(self, request_type: EngineCoreRequestType,
                               request: Any) -> None:
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            new_wave: int = request
            if new_wave >= self.current_wave:
                self.current_wave = new_wave
                if not self.engines_running:
                    logger.debug("EngineCore starting idle loop for wave {}.",
                                 new_wave)
                    self.engines_running = True
        else:
            super()._handle_client_request(request_type, request)

    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()

            if local_unfinished_reqs:
                # 2) Step the engine core.
                self._process_engine_step()

                # Check if we have now finished all requests.
                local_unfinished_reqs = (
                    self.scheduler.has_unfinished_requests())
            else:
                if self.scheduler.has_finished_requests():
                    # There are no unfinished requests, but there are some
                    # finished requests remaining to be removed from the
                    # batch state. This engine step won't perform a forward
                    # pass but will flush the finished requests to ensure
                    # up-to-date state is returned in the engine outputs.
                    self._process_engine_step()

                if not self.engines_running:
                    # All engines are idle.
                    continue

                # There must be unfinished requests in DP peers, run a
                # dummy forward pass.
                self.execute_dummy_batch()

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs)

            if not self.engines_running:
                if self.local_dp_rank == 0:
                    # Notify client that we are pausing the loop.
                    logger.debug("Wave {} finished, pausing engine loop.",
                                 self.current_wave)
                    self.output_queue.put_nowait(
                        EngineCoreOutputs(wave_complete=self.current_wave))
                self.current_wave += 1

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:

        # Optimization - only perform finish-sync all-reduce every 24 steps.
        self.counter += 1
        if self.counter != 24:
            return True
        self.counter = 0

        return ParallelConfig.has_unfinished_dp(self.dp_group,
                                                local_unfinished)
