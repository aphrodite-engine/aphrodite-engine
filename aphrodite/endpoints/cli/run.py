import argparse
import signal
from typing import Optional

import uvloop
from loguru import logger

import aphrodite
import aphrodite.common.envs as envs
from aphrodite.utils import (FlexibleArgumentParser, decorate_logs,
                                    get_tcp_uri, set_process_title)
from aphrodite.endpoints.cli.types import CLISubcommand
from aphrodite.endpoints.openai.api_server import (run_server,
                                                   run_server_worker,
                                                   setup_server)
from aphrodite.endpoints.openai.args import (make_arg_parser,
                                             validate_parsed_serve_args)
from aphrodite.endpoints.utils import (
    APHRODITE_SUBCMD_PARSER_EPILOG, show_filtered_argument_or_group_from_help)
from aphrodite.usage.usage_lib import UsageContext
from aphrodite.v1.engine.core import EngineCoreProc
from aphrodite.v1.engine.utils import (CoreEngineProcManager,
                                       launch_core_engines)
from aphrodite.v1.executor.abstract import Executor
from aphrodite.v1.metrics.prometheus import setup_multiprocess_prometheus
from aphrodite.v1.utils import (APIServerProcessManager,
                                wait_for_completion_or_failure)


class ServeSubcommand(CLISubcommand):
    """The `run` subcommand for the Aphrodite CLI. """
    name = "run"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # If model is specified in CLI (as positional arg), it takes precedence
        if hasattr(args, 'model_tag') and args.model_tag is not None:
            args.model = args.model_tag

        if args.headless or args.api_server_count < 1:
            run_headless(args)
        else:
            if args.api_server_count > 1:
                run_multi_api_server(args)
            else:
                # Single API server (this process).
                uvloop.run(run_server(args))

    def validate(self, args: argparse.Namespace) -> None:
        validate_parsed_serve_args(args)

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            "run",
            help="Start the Aphrodite OpenAI Compatible API server.",
            description="Start the Aphrodite OpenAI Compatible API server.",
            usage="aphrodite run [model_tag] [options]")

        serve_parser = make_arg_parser(serve_parser)
        show_filtered_argument_or_group_from_help(serve_parser, ["run"])
        serve_parser.epilog = APHRODITE_SUBCMD_PARSER_EPILOG
        return serve_parser


def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]


def run_headless(args: argparse.Namespace):

    if args.api_server_count > 1:
        raise ValueError("api_server_count can't be set in headless mode")

    # Create the EngineConfig.
    engine_args = aphrodite.AsyncEngineArgs.from_cli_args(args)
    usage_context = UsageContext.OPENAI_API_SERVER
    aphrodite_config = engine_args.create_engine_config(usage_context=usage_context,
                                                   headless=True)

    if not envs.APHRODITE_USE_V1:
        raise ValueError("Headless mode is only supported for V1")

    if engine_args.data_parallel_hybrid_lb:
        raise ValueError("data_parallel_hybrid_lb is not applicable in "
                         "headless mode")

    parallel_config = aphrodite_config.parallel_config
    local_engine_count = parallel_config.data_parallel_size_local

    if local_engine_count <= 0:
        raise ValueError("data_parallel_size_local must be > 0 in "
                         "headless mode")

    host = parallel_config.data_parallel_master_ip
    port = engine_args.data_parallel_rpc_port  # add to config too
    handshake_address = get_tcp_uri(host, port)

    # Catch SIGTERM and SIGINT to allow graceful shutdown.
    def signal_handler(signum, frame):
        logger.debug("Received {} signal.", signum)
        raise SystemExit

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info(
        "Launching {} data parallel engine(s) in headless mode, "
        "with head node address {}.", local_engine_count, handshake_address)

    # Create the engines.
    engine_manager = CoreEngineProcManager(
        target_fn=EngineCoreProc.run_engine_core,
        local_engine_count=local_engine_count,
        start_index=aphrodite_config.parallel_config.data_parallel_rank,
        local_start_index=0,
        aphrodite_config=aphrodite_config,
        local_client=False,
        handshake_address=handshake_address,
        executor_class=Executor.get_class(aphrodite_config),
        log_stats=not engine_args.disable_log_stats,
    )

    try:
        engine_manager.join_first()
    finally:
        logger.info("Shutting down.")
        engine_manager.close()


def run_multi_api_server(args: argparse.Namespace):

    assert not args.headless
    num_api_servers = args.api_server_count
    assert num_api_servers > 0

    orig_mm_processor_cache_gb = args.mm_processor_cache_gb

    if num_api_servers > 1:
        setup_multiprocess_prometheus()

        # Not compatible with API server scale-out
        args.mm_processor_cache_gb = 0

    listen_address, sock = setup_server(args)

    engine_args = aphrodite.AsyncEngineArgs.from_cli_args(args)
    usage_context = UsageContext.OPENAI_API_SERVER
    aphrodite_config = engine_args.create_engine_config(usage_context=usage_context)
    model_config = aphrodite_config.model_config

    if num_api_servers > 1:
        if not envs.APHRODITE_USE_V1:
            raise ValueError("api_server_count > 1 is only supported for V1")

        if envs.APHRODITE_ALLOW_RUNTIME_LORA_UPDATING:
            raise ValueError("APHRODITE_ALLOW_RUNTIME_LORA_UPDATING cannot be used "
                             "with api_server_count > 1")

        if model_config.is_multimodal_model and orig_mm_processor_cache_gb > 0:
            logger.warning("Multi-modal processor cache is disabled because "
                           "it is not compatible with `api_server_count > 1`.")

    executor_class = Executor.get_class(aphrodite_config)
    log_stats = not engine_args.disable_log_stats

    parallel_config = aphrodite_config.parallel_config
    dp_rank = parallel_config.data_parallel_rank
    external_dp_lb = parallel_config.data_parallel_external_lb
    hybrid_dp_lb = parallel_config.data_parallel_hybrid_lb
    assert external_dp_lb or hybrid_dp_lb or dp_rank == 0

    api_server_manager: Optional[APIServerProcessManager] = None

    with launch_core_engines(aphrodite_config, executor_class, log_stats,
                             num_api_servers) as (local_engine_manager,
                                                  coordinator, addresses):

        # Construct common args for the APIServerProcessManager up-front.
        api_server_manager_kwargs = dict(
            target_server_fn=run_api_server_worker_proc,
            listen_address=listen_address,
            sock=sock,
            args=args,
            num_servers=num_api_servers,
            input_addresses=addresses.inputs,
            output_addresses=addresses.outputs,
            stats_update_address=coordinator.get_stats_publish_address()
            if coordinator else None)

        # For dp ranks > 0 in external/hybrid DP LB modes, we must delay the
        # start of the API servers until the local engine is started
        # (after the launcher context manager exits),
        # since we get the front-end stats update address from the coordinator
        # via the handshake with the local engine.
        if dp_rank == 0 or not (external_dp_lb or hybrid_dp_lb):
            # Start API servers using the manager.
            api_server_manager = APIServerProcessManager(
                **api_server_manager_kwargs)

    # Start API servers now if they weren't already started.
    if api_server_manager is None:
        api_server_manager_kwargs["stats_update_address"] = (
            addresses.frontend_stats_publish_address)
        api_server_manager = APIServerProcessManager(
            **api_server_manager_kwargs)

    # Wait for API servers
    wait_for_completion_or_failure(api_server_manager=api_server_manager,
                                   engine_manager=local_engine_manager,
                                   coordinator=coordinator)


def run_api_server_worker_proc(listen_address,
                               sock,
                               args,
                               client_config=None,
                               **uvicorn_kwargs) -> None:
    """Entrypoint for individual API server worker processes."""

    # Set process title and add process-specific prefix to stdout and stderr.
    server_index = client_config.get("client_index", 0) if client_config else 0
    set_process_title("APIServer", str(server_index))
    decorate_logs()

    uvloop.run(
        run_server_worker(listen_address, sock, args, client_config,
                          **uvicorn_kwargs))
