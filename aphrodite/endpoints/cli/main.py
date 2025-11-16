"""The CLI endpoint of Aphrodite

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage."""

import importlib.metadata
import sys
from shutil import which

from aphrodite.logger import init_logger

logger = init_logger(__name__)


def main():
    import aphrodite.endpoints.cli.benchmark.main
    import aphrodite.endpoints.cli.collect_env
    import aphrodite.endpoints.cli.openai
    import aphrodite.endpoints.cli.run
    import aphrodite.endpoints.cli.run_batch
    import aphrodite.endpoints.cli.tokenizer
    from aphrodite.endpoints.utils import APHRODITE_SUBCMD_PARSER_EPILOG, cli_env_setup
    from aphrodite.utils.argparse_utils import FlexibleArgumentParser

    CMD_MODULES = [
        aphrodite.endpoints.cli.openai,
        aphrodite.endpoints.cli.run,
        aphrodite.endpoints.cli.benchmark.main,
        aphrodite.endpoints.cli.collect_env,
        aphrodite.endpoints.cli.run_batch,
        aphrodite.endpoints.cli.tokenizer,
    ]

    if len(sys.argv) > 1 and sys.argv[1] == "diffusion":
        try:
            import aphrodite.endpoints.cli.diffusion

            CMD_MODULES.append(aphrodite.endpoints.cli.diffusion)
        except ImportError:
            pip_cmd = "uv pip" if which("uv") else "pip"
            install_cmd = f"{pip_cmd} install aphrodite-engine[diffusion]"
            error_msg = f"Failed to import diffusion module. Please install it using:\n  {install_cmd}"

            original_excepthook = sys.excepthook

            def clean_excepthook(exc_type, exc_value, exc_traceback):
                if exc_type is ImportError and exc_value and "Failed to import diffusion module" in str(exc_value):
                    print(str(exc_value), file=sys.stderr)
                    sys.exit(1)
                else:
                    original_excepthook(exc_type, exc_value, exc_traceback)

            sys.excepthook = clean_excepthook
            raise ImportError(error_msg) from None

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
            logger.info("Unspecified platform detected, switching to CPU Platform instead.")

    parser = FlexibleArgumentParser(
        description="Aphrodite CLI",
        epilog=APHRODITE_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=importlib.metadata.version("aphrodite-engine"),
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in CMD_MODULES:
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
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
