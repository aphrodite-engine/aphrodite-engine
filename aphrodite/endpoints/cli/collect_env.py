import argparse

from aphrodite.collect_env import main as collect_env_main
from aphrodite.common.utils import FlexibleArgumentParser
from aphrodite.endpoints.cli.types import CLISubcommand
from aphrodite.endpoints.openai.args import make_arg_parser


class CollectEnvSubcommand(CLISubcommand):
    """The `run` subcommand for the Aphrodite CLI. """

    def __init__(self):
        self.name = "collect-env"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        """Collect information about the environment."""
        collect_env_main()

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            "collect-env",
            help="Start collecting environment information.",
            description="Start collecting environment information.",
            usage="aphrodite collect-env")
        return make_arg_parser(serve_parser)


def cmd_init() -> list[CLISubcommand]:
    return [CollectEnvSubcommand()]
