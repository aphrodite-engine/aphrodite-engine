import argparse

from aphrodite.benchmarks.latency import add_cli_args, main
from aphrodite.endpoints.cli.benchmark.base import BenchmarkSubcommandBase
from aphrodite.endpoints.cli.types import CLISubcommand


class BenchmarkLatencySubcommand(BenchmarkSubcommandBase):
    """ The `latency` subcommand for aphrodite bench. """

    def __init__(self):
        self.name = "latency"
        super().__init__()

    @property
    def help(self) -> str:
        return "Benchmark the latency of a single batch of requests."

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)


def cmd_init() -> list[CLISubcommand]:
    return [BenchmarkLatencySubcommand()]
