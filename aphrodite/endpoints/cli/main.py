# The CLI entrypoint to Aphrodite.
import signal
import sys

import aphrodite.endpoints.cli.benchmark.main
import aphrodite.endpoints.cli.collect_env
import aphrodite.endpoints.cli.openai
import aphrodite.endpoints.cli.run
import aphrodite.version
from aphrodite.common.utils import FlexibleArgumentParser
from aphrodite.endpoints.utils import cli_env_setup

CMD_MODULES = [
    aphrodite.endpoints.cli.openai,
    aphrodite.endpoints.cli.run,
    aphrodite.endpoints.cli.benchmark.main,
    aphrodite.endpoints.cli.collect_env,
]


def register_signal_handlers():

    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)


def main():
    cli_env_setup()

    parser = FlexibleArgumentParser(description="Aphrodite CLI")
    parser.add_argument('-v',
                        '--version',
                        action='version',
                        version=aphrodite.version.__version__)
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
