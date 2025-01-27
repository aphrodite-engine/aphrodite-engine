import pytest

from aphrodite.common.utils import FlexibleArgumentParser
from aphrodite.engine.args_tools import EngineArgs


@pytest.mark.parametrize(("arg", "expected"), [
    (None, None),
    ("image=16", {
        "image": 16
    }),
    ("image=16,video=2", {
        "image": 16,
        "video": 2
    }),
    ("Image=16, Video=2", {
        "image": 16,
        "video": 2
    }),
])


def test_limit_mm_per_prompt_parser(arg, expected):
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    if arg is None:
        args = parser.parse_args([])
    else:
        args = parser.parse_args(["--limit-mm-per-prompt", arg])

    assert args.limit_mm_per_prompt == expected


def test_mm_processor_kwargs_prompt_parser(arg, expected):
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    if arg is None:
        args = parser.parse_args([])
    else:
        args = parser.parse_args(["--mm-processor-kwargs", arg])
    assert args.mm_processor_kwargs == expected