import argparse

from aphrodite.benchmarks.latency import add_cli_args, main
from aphrodite.endpoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarkLatencySubcommand(BenchmarkSubcommandBase):
    """ The `latency` subcommand for aphrodite bench. """

    name = "latency"
    help = "Benchmark the latency of a single batch of requests."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
