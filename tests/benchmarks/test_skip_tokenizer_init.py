# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for --skip-tokenizer-init with a custom dataset."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import aphrodite.benchmarks.serve as serve_module
from aphrodite.utils.argparse_utils import FlexibleArgumentParser


@pytest.mark.benchmark
def test_main_async_skip_tokenizer_init_with_custom_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "prompt": {
                    "data": {
                        "data": "https://example.com/image.tif",
                        "data_format": "url",
                        "out_data_format": "b64_json",
                    }
                }
            }
        )
        + "\n"
    )

    parser = FlexibleArgumentParser()
    serve_module.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--backend",
            "aphrodite-pooling",
            "--dataset-name",
            "custom",
            "--dataset-path",
            str(dataset_path),
            "--model",
            "pooling-model",
            "--endpoint",
            "/pooling",
            "--num-prompts",
            "1",
            "--skip-chat-template",
            "--skip-tokenizer-init",
        ]
    )
    mock_result = {
        "completed": 1,
        "failed": 0,
        "total_input_tokens": 1,
        "total_output_tokens": 1,
        "request_throughput": 1.0,
        "output_throughput": 1.0,
        "total_token_throughput": 1.0,
        "input_lens": [],
        "output_lens": [],
        "start_times": [],
        "ttfts": [],
        "itls": [],
        "generated_texts": [],
        "errors": [],
        "duration": 1.0,
    }

    with patch.object(
        serve_module,
        "benchmark",
        new=AsyncMock(return_value=mock_result),
    ):
        asyncio.run(serve_module.main_async(args))
