'''The CLI endpoint of Aphrodite

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage.'''
from __future__ import annotations

import importlib.metadata


def main():
    import aphrodite.endpoints.cli.benchmark.main
    import aphrodite.endpoints.cli.collect_env
    import aphrodite.endpoints.cli.openai
    import aphrodite.endpoints.cli.run_batch
    import aphrodite.endpoints.cli.run
    from aphrodite.common.utils import FlexibleArgumentParser
    from aphrodite.endpoints.utils import (APHRODITE_SUBCMD_PARSER_EPILOG,
                                           cli_env_setup)

    CMD_MODULES = [
        aphrodite.endpoints.cli.openai,
        aphrodite.endpoints.cli.run,
        aphrodite.endpoints.cli.benchmark.main,
        aphrodite.endpoints.cli.collect_env,
        aphrodite.endpoints.cli.run_batch,
    ]

    cli_env_setup()

    parser = FlexibleArgumentParser(
        description="Aphrodite CLI",
        epilog=APHRODITE_SUBCMD_PARSER_EPILOG,
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=importlib.metadata.version('aphrodite'),
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
