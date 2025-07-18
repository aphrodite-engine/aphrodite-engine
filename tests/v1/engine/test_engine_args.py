from argparse import ArgumentError

import pytest

from aphrodite.common import envs
from aphrodite.common.config import AphroditeConfig
from aphrodite.common.utils import FlexibleArgumentParser
from aphrodite.engine.args_tools import EngineArgs
from aphrodite.usage.usage_lib import UsageContext

if not envs.APHRODITE_USE_V1:
    pytest.skip(
        "Skipping V1 tests. Rerun with `APHRODITE_USE_V1=1` to test.",
        allow_module_level=True,
    )


def test_prefix_caching_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args([])
    aphrodite_config = EngineArgs.from_cli_args(
        args=args).create_engine_config()
    assert (aphrodite_config.cache_config.enable_prefix_caching
            ), "V1 turns on prefix caching by default."

    # Turn it off possible with flag.
    args = parser.parse_args(["--no-enable-prefix-caching"])
    aphrodite_config = EngineArgs.from_cli_args(
        args=args).create_engine_config()
    assert not aphrodite_config.cache_config.enable_prefix_caching

    # Turn it on with flag.
    args = parser.parse_args(["--enable-prefix-caching"])
    aphrodite_config = EngineArgs.from_cli_args(
        args=args).create_engine_config()
    assert aphrodite_config.cache_config.enable_prefix_caching

    # default hash algorithm is "builtin"
    assert aphrodite_config.cache_config.prefix_caching_hash_algo == "builtin"

    # set hash algorithm to sha256
    args = parser.parse_args(["--prefix-caching-hash-algo", "sha256"])
    aphrodite_config = EngineArgs.from_cli_args(
        args=args).create_engine_config()
    assert aphrodite_config.cache_config.prefix_caching_hash_algo == "sha256"

    # set hash algorithm to builtin
    args = parser.parse_args(["--prefix-caching-hash-algo", "builtin"])
    aphrodite_config = EngineArgs.from_cli_args(
        args=args).create_engine_config()
    assert aphrodite_config.cache_config.prefix_caching_hash_algo == "builtin"

    # an invalid hash algorithm raises an error
    parser.exit_on_error = False
    with pytest.raises(ArgumentError):
        args = parser.parse_args(["--prefix-caching-hash-algo", "invalid"])


def test_defaults_with_usage_context():
    engine_args = EngineArgs(model="facebook/opt-125m")
    aphrodite_config: AphroditeConfig = engine_args.create_engine_config(
        UsageContext.LLM_CLASS)

    from aphrodite.platforms import current_platform
    device_name = current_platform.get_device_name().lower()
    if "h100" in device_name or "h200" in device_name:
        # For H100 and H200, we use larger default values.
        default_llm_tokens = 16384
        default_server_tokens = 8192
        default_max_num_seqs = 1024
    else:
        default_llm_tokens = 8192
        default_server_tokens = 2048
        default_max_num_seqs = 256

    assert aphrodite_config.scheduler_config.max_num_seqs == default_max_num_seqs
    assert aphrodite_config.scheduler_config.max_num_batched_tokens == default_llm_tokens  # noqa: E501

    engine_args = EngineArgs(model="facebook/opt-125m")
    aphrodite_config = engine_args.create_engine_config(
        UsageContext.OPENAI_API_SERVER)
    assert aphrodite_config.scheduler_config.max_num_seqs == default_max_num_seqs
    assert aphrodite_config.scheduler_config.max_num_batched_tokens == default_server_tokens  # noqa: E501
