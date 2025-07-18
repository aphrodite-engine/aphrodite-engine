import argparse

from aphrodite.common.utils import FlexibleArgumentParser
from aphrodite.endpoints.cli.types import CLISubcommand


class BenchmarkSubcommandBase(CLISubcommand):
    """ The base class of subcommands for aphrodite bench. """

    @property
    def help(self) -> str:
        """The help message of the subcommand."""
        raise NotImplementedError

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        """Add the CLI arguments to the parser."""
        raise NotImplementedError

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        """Run the benchmark.

        Args:
            args: The arguments to the command.
        """
        raise NotImplementedError

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.help,
            usage=f"aphrodite bench {self.name} [options]")
        self.add_cli_args(parser)
        return parser
