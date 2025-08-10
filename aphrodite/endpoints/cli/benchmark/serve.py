import argparse

from aphrodite.benchmarks.serve import add_cli_args, main
from aphrodite.endpoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarkServingSubcommand(BenchmarkSubcommandBase):
    """ The `run` subcommand for aphrodite bench. """

    name = "run"
    help = "Benchmark the online serving throughput."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
