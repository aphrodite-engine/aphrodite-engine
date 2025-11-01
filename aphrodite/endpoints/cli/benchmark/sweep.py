import argparse

from aphrodite.benchmarks.sweep.cli import add_cli_args, main
from aphrodite.endpoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarkSweepSubcommand(BenchmarkSubcommandBase):
    """The `sweep` subcommand for `aphrodite bench`."""

    name = "sweep"
    help = "Benchmark for a parameter sweep."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
