"""Integration of diffusion CLI commands into the main Aphrodite CLI."""

import argparse
from typing import TYPE_CHECKING

from aphrodite.diffusion.runtime.entrypoints.cli.generate import generate_cmd
from aphrodite.endpoints.cli.types import CLISubcommand
from aphrodite.endpoints.utils import APHRODITE_SUBCMD_PARSER_EPILOG

if TYPE_CHECKING:
    from aphrodite.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser


class DiffusionGenerateSubcommand(CLISubcommand):
    """Adapter for the diffusion generate subcommand."""

    name = "generate"

    def __init__(self):
        from aphrodite.diffusion.runtime.entrypoints.cli.generate import GenerateSubcommand

        self._diffusion_cmd = GenerateSubcommand()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        generate_cmd(args)

    def validate(self, args: argparse.Namespace) -> None:
        self._diffusion_cmd.validate(args)

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        from typing import cast

        from aphrodite.diffusion.runtime.entrypoints.cli.generate import add_multimodal_gen_generate_args

        generate_parser = subparsers.add_parser(
            "generate",
            help="Run inference on a diffusion model",
            usage=("aphrodite diffusion generate [MODEL_TAG] [--prompt PROMPT] | --config CONFIG_FILE [OPTIONS]"),
        )

        generate_parser.add_argument(
            "model_tag",
            type=str,
            nargs="?",
            help="The model path or ID to use (optional if specified via --model or in config)",
        )

        generate_parser = add_multimodal_gen_generate_args(generate_parser)
        generate_parser.epilog = APHRODITE_SUBCMD_PARSER_EPILOG.format(subcmd="diffusion generate")

        return cast(FlexibleArgumentParser, generate_parser)


class DiffusionRunSubcommand(CLISubcommand):
    """Adapter for the diffusion run subcommand (formerly serve)."""

    name = "run"

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

        run_parser = subparsers.add_parser(
            "run",
            help="Launch the diffusion model server and start FastAPI listener.",
            usage="aphrodite diffusion run [MODEL_TAG] [OPTIONS]",
        )

        run_parser.add_argument(
            "model_tag",
            type=str,
            nargs="?",
            help="The model path or ID to use (optional if specified via --model or in config)",
        )

        run_parser = add_multimodal_gen_serve_args(run_parser)
        run_parser.epilog = APHRODITE_SUBCMD_PARSER_EPILOG.format(subcmd="diffusion run")

        return cast(FlexibleArgumentParser, run_parser)


class DiffusionSubcommand(CLISubcommand):
    """The `diffusion` subcommand for the Aphrodite CLI with nested subcommands."""

    name = "diffusion"
    help = "Diffusion model commands."

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        if hasattr(args, "dispatch_function"):
            args.dispatch_function(args)
        else:
            # This shouldn't happen if argparse is configured correctly
            raise ValueError("No subcommand specified for diffusion")

    def validate(self, args: argparse.Namespace) -> None:
        if hasattr(args, "diffusion_subcommand") and args.diffusion_subcommand:
            # Validate the nested subcommand
            if args.diffusion_subcommand == "generate":
                DiffusionGenerateSubcommand().validate(args)
            elif args.diffusion_subcommand == "run":
                DiffusionRunSubcommand().validate(args)

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        diffusion_parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.help,
            usage=f"aphrodite {self.name} {{generate,run}} [options]",
        )
        diffusion_subparsers = diffusion_parser.add_subparsers(required=True, dest="diffusion_subcommand")

        generate_cmd = DiffusionGenerateSubcommand()
        run_cmd = DiffusionRunSubcommand()

        generate_cmd.subparser_init(diffusion_subparsers).set_defaults(dispatch_function=generate_cmd.cmd)
        run_cmd.subparser_init(diffusion_subparsers).set_defaults(dispatch_function=run_cmd.cmd)

        return diffusion_parser


def cmd_init() -> list[CLISubcommand]:
    return [DiffusionSubcommand()]
