# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
import argparse

from aphrodite.benchmarks.serve import add_cli_args, main
from aphrodite.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarkServingSubcommand(BenchmarkSubcommandBase):
    """The `serve` subcommand for `aphrodite bench`."""

    name = "serve"
    help = "Benchmark the online serving throughput."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
