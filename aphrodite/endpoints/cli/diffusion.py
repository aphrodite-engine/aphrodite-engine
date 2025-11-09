"""Integration of diffusion CLI commands into the main Aphrodite CLI."""

import argparse
from typing import TYPE_CHECKING

from aphrodite.endpoints.cli.types import CLISubcommand

if TYPE_CHECKING:
    from aphrodite.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser


class DiffusionGenerateSubcommand(CLISubcommand):
    """Adapter for the diffusion generate subcommand."""

    name = "diffusion-generate"

    def __init__(self):
        from aphrodite.diffusion.runtime.entrypoints.cli.generate import GenerateSubcommand

        self._diffusion_cmd = GenerateSubcommand()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from aphrodite.diffusion.runtime.entrypoints.cli.generate import generate_cmd

        generate_cmd(args)

    def validate(self, args: argparse.Namespace) -> None:
        self._diffusion_cmd.validate(args)

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        from typing import cast

        from aphrodite.diffusion.runtime.entrypoints.cli.generate import add_multimodal_gen_generate_args

        generate_parser = subparsers.add_parser(
            "diffusion-generate",
            help="Run inference on a diffusion model",
            usage=(
                "aphrodite diffusion-generate "
                "(--model-path MODEL_PATH_OR_ID --prompt PROMPT) | "
                "--config CONFIG_FILE [OPTIONS]"
            ),
        )

        generate_parser = add_multimodal_gen_generate_args(generate_parser)

        return cast(FlexibleArgumentParser, generate_parser)


class DiffusionServeSubcommand(CLISubcommand):
    """Adapter for the diffusion serve subcommand."""

    name = "diffusion-serve"

    def __init__(self):
        from aphrodite.diffusion.runtime.entrypoints.cli.serve import ServeSubcommand

        self._diffusion_cmd = ServeSubcommand()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from aphrodite.diffusion.runtime.entrypoints.cli.serve import execute_serve_cmd

        execute_serve_cmd(args, None)

    def validate(self, args: argparse.Namespace) -> None:
        self._diffusion_cmd.validate(args)

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        from typing import cast

        from aphrodite.diffusion.runtime.entrypoints.cli.serve import add_multimodal_gen_serve_args

        serve_parser = subparsers.add_parser(
            "diffusion-serve",
            help="Launch the diffusion model server and start FastAPI listener.",
            usage="aphrodite diffusion-serve --model-path MODEL_PATH_OR_ID [OPTIONS]",
        )

        serve_parser = add_multimodal_gen_serve_args(serve_parser)

        return cast(FlexibleArgumentParser, serve_parser)


def cmd_init() -> list[CLISubcommand]:
    return [DiffusionGenerateSubcommand(), DiffusionServeSubcommand()]
