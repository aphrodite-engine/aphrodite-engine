"""CLI subcommand for running the tokenizer-only server."""
from __future__ import annotations

import argparse

import uvloop

from aphrodite.endpoints.cli.types import CLISubcommand

try:
    from typing import TYPE_CHECKING
except ImportError:
    TYPE_CHECKING = False

if TYPE_CHECKING:
    from aphrodite.utils import FlexibleArgumentParser


class TokenizerSubcommand(CLISubcommand):
    """The `tokenizer` subcommand for the Aphrodite CLI."""

    name = "tokenizer"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        """Run the tokenizer server."""
        from aphrodite.endpoints.tokenizer_server import run_server

        uvloop.run(run_server(args))

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Add CLI arguments for the tokenizer command."""
        from aphrodite.endpoints.tokenizer_server import make_arg_parser

        return make_arg_parser(parser)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        """Initialize the subparser for the tokenizer command."""
        parser = subparsers.add_parser(
            "tokenizer",
            help="Start a lightweight tokenizer-only server",
            description=(
                "Start a tokenizer-only HTTP server that provides tokenization "
                "endpoints without running the full Aphrodite engine."
            ),
            usage="aphrodite tokenizer MODEL [options]",
        )
        return TokenizerSubcommand.add_cli_args(parser)


def cmd_init() -> list[CLISubcommand]:
    """Initialize the tokenizer subcommand."""
    return [TokenizerSubcommand()]
