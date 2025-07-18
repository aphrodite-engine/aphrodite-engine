import argparse

import uvloop

from aphrodite.common.utils import FlexibleArgumentParser
from aphrodite.endpoints.cli.types import CLISubcommand
from aphrodite.endpoints.openai.api_server import run_server
from aphrodite.endpoints.openai.args import (make_arg_parser,
                                             validate_parsed_serve_args)


class ServeSubcommand(CLISubcommand):
    """The `run` subcommand for the Aphrodite CLI. """

    def __init__(self):
        self.name = "run"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # If model is specified in CLI (as positional arg), it takes precedence
        if hasattr(args, 'model_tag') and args.model_tag is not None:
            args.model = args.model_tag

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
        serve_parser.add_argument("model_tag",
                                  type=str,
                                  nargs='?',
                                  help="The model tag to run "
                                  "(optional if specified in config)")
        serve_parser.add_argument(
            "--config",
            type=str,
            default='',
            required=False,
            help="Read CLI options from a YAML config file."
        )

        return make_arg_parser(serve_parser)


def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]
