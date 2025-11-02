"""The CLI endpoint of Aphrodite

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage."""

import importlib.metadata
import sys

from aphrodite.logger import init_logger

logger = init_logger(__name__)


def main():
    import aphrodite.endpoints.cli.benchmark.main
    import aphrodite.endpoints.cli.collect_env
    import aphrodite.endpoints.cli.openai
    import aphrodite.endpoints.cli.run
    import aphrodite.endpoints.cli.run_batch
    import aphrodite.endpoints.cli.tokenizer
    from aphrodite.endpoints.utils import (APHRODITE_SUBCMD_PARSER_EPILOG,
                                           cli_env_setup)
    from aphrodite.utils.argparse_utils import FlexibleArgumentParser

    CMD_MODULES = [
        aphrodite.endpoints.cli.openai,
        aphrodite.endpoints.cli.run,
        aphrodite.endpoints.cli.benchmark.main,
        aphrodite.endpoints.cli.collect_env,
        aphrodite.endpoints.cli.run_batch,
        aphrodite.endpoints.cli.tokenizer,
    ]

    cli_env_setup()

    # For 'aphrodite bench *': use CPU instead of UnspecifiedPlatform by default
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        logger.debug(
            "Bench command detected, must ensure current platform is not "
            "UnspecifiedPlatform to avoid device type inference error"
        )
        from aphrodite import platforms

        if platforms.current_platform.is_unspecified():
            from aphrodite.platforms.cpu import CpuPlatform

            platforms.current_platform = CpuPlatform()
            logger.info(
                "Unspecified platform detected, switching to CPU Platform instead."
            )

    parser = FlexibleArgumentParser(
        description="Aphrodite CLI",
        epilog=APHRODITE_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=importlib.metadata.version('aphrodite-engine'),
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in CMD_MODULES:
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(
                dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
