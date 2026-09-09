# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
CLI entry point for Sonar Omni that intercepts Sonar commands.
"""

import importlib.metadata
import sys


def main():
    """Main CLI entry point that intercepts Sonar commands."""
    # Check if --omni flag is present
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        sys.argv[1] = "serve"
    if "--omni" not in sys.argv:
        from aphrodite.entrypoints.cli.main import main as aphrodite_main

        aphrodite_main()
        return
    else:
        # Force colored logging even when piped (e.g. `| tee`).
        # Must be set before any Sonar import because the logger
        # formatter is configured at import time via _use_color().
        import os

        if "APHRODITE_LOGGING_COLOR" not in os.environ:
            os.environ["APHRODITE_LOGGING_COLOR"] = "1"

        import aphrodite.omni.entrypoints.cli.benchmark.main
        import aphrodite.omni.entrypoints.cli.serve
        from aphrodite.entrypoints.serve.utils.api_utils import APHRODITE_SUBCMD_PARSER_EPILOG, cli_env_setup
        from aphrodite.omni.utils.tracking_parser import TrackingArgumentParser

        CMD_MODULES = [
            aphrodite.omni.entrypoints.cli.serve,
            aphrodite.omni.entrypoints.cli.benchmark.main,
        ]

        cli_env_setup()

        from aphrodite.omni.entrypoints.cli.serve import _ensure_aphrodite_platform

        _ensure_aphrodite_platform()

        parser = TrackingArgumentParser(
            description="Sonar OMNI CLI",
            epilog=APHRODITE_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
        )
        try:
            _omni_version = importlib.metadata.version("aphrodite-engine")
        except importlib.metadata.PackageNotFoundError:
            try:
                from aphrodite.omni.version import __version__ as _omni_version  # type: ignore
            except Exception:
                _omni_version = "dev"
        parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=_omni_version,
        )
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        args = parser.parse_args()
        if args.subparser in cmds:
            cmds[args.subparser].validate(args)

        if hasattr(args, "dispatch_function"):
            args.dispatch_function(args)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
