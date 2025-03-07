from http import HTTPStatus

import openai
import pytest
import requests
from prometheus_client.parser import text_string_to_metric_families
from transformers import AutoTokenizer

from ...utils import RemoteOpenAIServer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module")
def default_server_args():
    return [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "1024",
        "--enforce-eager",
        "--max-num-seqs",
        "128",
    ]


@pytest.fixture(scope="module",
                params=[
                    "",
                    "--enable-chunked-prefill",
                    "--disable-frontend-multiprocessing",
                ])
def client(default_server_args, request):
    if request.param:
        default_server_args.append(request.param)
    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server.get_async_client()


_PROMPT = "Hello my name is Robert and I love magic"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_TOKENIZED_PROMPT = tokenizer(_PROMPT)["input_ids"]

_NUM_REQUESTS = 10
_NUM_PROMPT_TOKENS_PER_REQUEST = len(_TOKENIZED_PROMPT)
_NUM_GENERATION_TOKENS_PER_REQUEST = 10

# {metric_family: [(suffix, expected_value)]}
EXPECTED_VALUES = {
    "aphrodite:time_to_first_token_seconds": [("_count", _NUM_REQUESTS)],
    "aphrodite:time_per_output_token_seconds":
    [("_count", _NUM_REQUESTS * (_NUM_GENERATION_TOKENS_PER_REQUEST - 1))],
    "aphrodite:e2e_request_latency_seconds": [("_count", _NUM_REQUESTS)],
    "aphrodite:request_prompt_tokens":
    [("_sum", _NUM_REQUESTS * _NUM_PROMPT_TOKENS_PER_REQUEST),
     ("_count", _NUM_REQUESTS)],
    "aphrodite:request_generation_tokens":
    [("_sum", _NUM_REQUESTS * _NUM_GENERATION_TOKENS_PER_REQUEST),
     ("_count", _NUM_REQUESTS)],
    "aphrodite:request_params_n": [("_count", _NUM_REQUESTS)],
    "aphrodite:prompt_tokens": [("_total",
                            _NUM_REQUESTS * _NUM_PROMPT_TOKENS_PER_REQUEST)],
    "aphrodite:generation_tokens":
    [("_total", _NUM_REQUESTS * _NUM_PROMPT_TOKENS_PER_REQUEST)],
    "aphrodite:request_success": [("_total", _NUM_REQUESTS)],
}


@pytest.mark.asyncio
async def test_metrics_counts(client: openai.AsyncOpenAI):
    base_url = str(client.base_url)[:-3].strip("/")

    for _ in range(_NUM_REQUESTS):
        # sending a request triggers the metrics to be logged.
        await client.completions.create(
            model=MODEL_NAME,
            prompt=_TOKENIZED_PROMPT,
            max_tokens=_NUM_GENERATION_TOKENS_PER_REQUEST)

    response = requests.get(base_url + "/metrics")
    print(response.text)
    assert response.status_code == HTTPStatus.OK

    # Loop over all expected metric_families
    for metric_family, suffix_values_list in EXPECTED_VALUES.items():
        found_metric = False

        # Check to see if the metric_family is found in the prom endpoint.
        for family in text_string_to_metric_families(response.text):
            if family.name == metric_family:
                found_metric = True

                # Check that each suffix is found in the prom endpoint.
                for suffix, expected_value in suffix_values_list:
                    metric_name_w_suffix = f"{metric_family}{suffix}"
                    found_suffix = False

                    for sample in family.samples:
                        if sample.name == metric_name_w_suffix:
                            found_suffix = True

                            # For each suffix, value sure the value matches
                            # what we expect.
                            assert sample.value == expected_value, (
                                f"{metric_name_w_suffix} expected value of "
                                f"{expected_value} did not match found value "
                                f"{sample.value}")
                            break
                    assert found_suffix, (
                        f"Did not find {metric_name_w_suffix} in prom endpoint"
                    )
                break

        assert found_metric, (f"Did not find {metric_family} in prom endpoint")


EXPECTED_METRICS = [
    "aphrodite:num_requests_running",
    "aphrodite:num_requests_swapped",
    "aphrodite:num_requests_waiting",
    "aphrodite:gpu_cache_usage_perc",
    "aphrodite:cpu_cache_usage_perc",
    "aphrodite:time_to_first_token_seconds_sum",
    "aphrodite:time_to_first_token_seconds_bucket",
    "aphrodite:time_to_first_token_seconds_count",
    "aphrodite:time_per_output_token_seconds_sum",
    "aphrodite:time_per_output_token_seconds_bucket",
    "aphrodite:time_per_output_token_seconds_count",
    "aphrodite:e2e_request_latency_seconds_sum",
    "aphrodite:e2e_request_latency_seconds_bucket",
    "aphrodite:e2e_request_latency_seconds_count",
    "aphrodite:request_prompt_tokens_sum",
    "aphrodite:request_prompt_tokens_bucket",
    "aphrodite:request_prompt_tokens_count",
    "aphrodite:request_generation_tokens_sum",
    "aphrodite:request_generation_tokens_bucket",
    "aphrodite:request_generation_tokens_count",
    "aphrodite:request_params_n_sum",
    "aphrodite:request_params_n_bucket",
    "aphrodite:request_params_n_count",
    "aphrodite:num_preemptions_total",
    "aphrodite:prompt_tokens_total",
    "aphrodite:generation_tokens_total",
    "aphrodite:request_success_total",
    "aphrodite:cache_config_info",
    # labels in cache_config_info
    "block_size",
    "cache_dtype",
    "cpu_offload_gb",
    "enable_prefix_caching",
    "gpu_memory_utilization",
    "num_cpu_blocks",
    "num_gpu_blocks",
    "num_gpu_blocks_override",
    "sliding_window",
    "swap_space_bytes",
]


@pytest.mark.asyncio
async def test_metrics_exist(client: openai.AsyncOpenAI):
    base_url = str(client.base_url)[:-3].strip("/")

    # sending a request triggers the metrics to be logged.
    await client.completions.create(model=MODEL_NAME,
                                    prompt="Hello, my name is",
                                    max_tokens=5,
                                    temperature=0.0)

    response = requests.get(base_url + "/metrics")
    assert response.status_code == HTTPStatus.OK

    for metric in EXPECTED_METRICS:
        assert metric in response.text
