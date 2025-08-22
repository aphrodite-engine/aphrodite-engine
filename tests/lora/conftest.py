import tempfile
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

import aphrodite
from aphrodite.config import LoRAConfig
from aphrodite.distributed import (cleanup_dist_env_and_memory,
                              init_distributed_environment,
                              initialize_model_parallel)
from aphrodite.modeling.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               RowParallelLinear)
from aphrodite.modeling.layers.logits_processor import LogitsProcessor
from aphrodite.modeling.layers.sampler import Sampler
from aphrodite.modeling.layers.vocab_parallel_embedding import ParallelLMHead
from aphrodite.modeling.model_loader import get_model
from aphrodite.modeling.models.interfaces import SupportsLoRA
from aphrodite.platforms import current_platform


@pytest.fixture()
def should_do_global_cleanup_after_test(request) -> bool:
    """Allow subdirectories to skip global cleanup by overriding this fixture.
    This can provide a ~10x speedup for non-GPU unit tests since they don't need
    to initialize torch.
    """

    return not request.node.get_closest_marker("skip_global_cleanup")


@pytest.fixture(autouse=True)
def cleanup_fixture(should_do_global_cleanup_after_test: bool):
    yield
    if should_do_global_cleanup_after_test:
        cleanup_dist_env_and_memory(shutdown_ray=True)


@pytest.fixture
def dist_init():
    temp_file = tempfile.mkstemp()[1]

    backend = "nccl"
    if current_platform.is_cpu():
        backend = "gloo"

    init_distributed_environment(world_size=1,
                                 rank=0,
                                 distributed_init_method=f"file://{temp_file}",
                                 local_rank=0,
                                 backend=backend)
    initialize_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory(shutdown_ray=True)


@pytest.fixture
def dist_init_torch_only():
    if torch.distributed.is_initialized():
        return
    backend = "nccl"
    if current_platform.is_cpu():
        backend = "gloo"

    temp_file = tempfile.mkstemp()[1]
    torch.distributed.init_process_group(world_size=1,
                                         rank=0,
                                         init_method=f"file://{temp_file}",
                                         backend=backend)


class DummyLoRAModel(nn.Sequential, SupportsLoRA):
    pass


@pytest.fixture
def dummy_model() -> nn.Module:
    model = DummyLoRAModel(
        OrderedDict([
            ("dense1", ColumnParallelLinear(764, 100)),
            ("dense2", RowParallelLinear(100, 50)),
            (
                "layer1",
                nn.Sequential(
                    OrderedDict([
                        ("dense1", ColumnParallelLinear(100, 10)),
                        ("dense2", RowParallelLinear(10, 50)),
                    ])),
            ),
            ("act2", nn.ReLU()),
            ("output", ColumnParallelLinear(50, 10)),
            ("outact", nn.Sigmoid()),
            # Special handling for lm_head & sampler
            ("lm_head", ParallelLMHead(512, 10)),
            ("logits_processor", LogitsProcessor(512)),
            ("sampler", Sampler())
        ]))
    model.config = MagicMock()
    model.embedding_modules = {"lm_head": "lm_head"}
    return model


@pytest.fixture
def dummy_model_gate_up() -> nn.Module:
    model = DummyLoRAModel(
        OrderedDict([
            ("dense1", ColumnParallelLinear(764, 100)),
            ("dense2", RowParallelLinear(100, 50)),
            (
                "layer1",
                nn.Sequential(
                    OrderedDict([
                        ("dense1", ColumnParallelLinear(100, 10)),
                        ("dense2", RowParallelLinear(10, 50)),
                    ])),
            ),
            ("act2", nn.ReLU()),
            ("gate_up_proj", MergedColumnParallelLinear(50, [5, 5])),
            ("outact", nn.Sigmoid()),
            # Special handling for lm_head & sampler
            ("lm_head", ParallelLMHead(512, 10)),
            ("logits_processor", LogitsProcessor(512)),
            ("sampler", Sampler())
        ]))
    model.config = MagicMock()
    model.packed_modules_mapping = {
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
    model.embedding_modules = {"lm_head": "lm_head"}
    return model


@pytest.fixture(scope="session")
def sql_lora_huggingface_id():
    # huggingface repo id is used to test lora runtime downloading.
    return "yard1/llama-2-7b-sql-lora-test"


@pytest.fixture(scope="session")
def sql_lora_files(sql_lora_huggingface_id):
    return snapshot_download(repo_id=sql_lora_huggingface_id)


@pytest.fixture(scope="session")
def mixtral_lora_files():
    # Note: this module has incorrect adapter_config.json to test
    return snapshot_download(repo_id="SangBinCho/mixtral-lora")


@pytest.fixture(scope="session")
def gemma_lora_files():
    return snapshot_download(repo_id="wskwon/gemma-7b-test-lora")


@pytest.fixture(scope="session")
def chatglm3_lora_files():
    return snapshot_download(repo_id="jeeejeee/chatglm3-text2sql-spider")


@pytest.fixture(scope="session")
def baichuan_lora_files():
    return snapshot_download(repo_id="jeeejeee/baichuan7b-text2sql-spider")


@pytest.fixture(scope="session")
def baichuan_zero_lora_files():
    # all the lora_B weights are initialized to zero.
    return snapshot_download(repo_id="jeeejeee/baichuan7b-zero-init")


@pytest.fixture(scope="session")
def baichuan_regex_lora_files():
    return snapshot_download(repo_id="jeeejeee/baichuan-7b-lora-zero-regex")


@pytest.fixture(scope="session")
def ilama_lora_files():
    return snapshot_download(repo_id="jeeejeee/ilama-text2sql-spider")


@pytest.fixture(scope="session")
def minicpmv_lora_files():
    return snapshot_download(repo_id="jeeejeee/minicpmv25-lora-pokemon")


@pytest.fixture(scope="session")
def qwen2vl_lora_files():
    return snapshot_download(repo_id="jeeejeee/qwen2-vl-lora-pokemon")


@pytest.fixture(scope="session")
def qwen25vl_lora_files():
    return snapshot_download(repo_id="jeeejeee/qwen25-vl-lora-pokemon")


@pytest.fixture(scope="session")
def tinyllama_lora_files():
    return snapshot_download(repo_id="jashing/tinyllama-colorist-lora")


@pytest.fixture(scope="session")
def phi2_lora_files():
    return snapshot_download(repo_id="isotr0py/phi-2-test-sql-lora")


@pytest.fixture(scope="session")
def long_context_lora_files_16k_1():
    return snapshot_download(repo_id="SangBinCho/long_context_16k_testing_1")


@pytest.fixture
def llama_2_7b_engine_extra_embeddings():
    cleanup_dist_env_and_memory(shutdown_ray=True)
    get_model_old = get_model

    def get_model_patched(**kwargs):
        kwargs["aphrodite_config"].lora_config = LoRAConfig(max_loras=4,
                                                       max_lora_rank=8)
        return get_model_old(**kwargs)

    with patch("aphrodite.worker.model_runner.get_model", get_model_patched):
        engine = aphrodite.LLM("meta-llama/Llama-2-7b-hf", enable_lora=False)
    yield engine.llm_engine
    del engine
    cleanup_dist_env_and_memory(shutdown_ray=True)


@pytest.fixture
def llama_2_7b_model_extra_embeddings(llama_2_7b_engine_extra_embeddings):
    yield (llama_2_7b_engine_extra_embeddings.modeling.driver_worker.
           model_runner.model)


@pytest.fixture(params=[True, False])
def run_with_both_engines_lora(request, monkeypatch):
    # Automatically runs tests twice, once with V1 and once without
    use_v1 = request.param
    # Tests decorated with `@skip_v1` are only run without v1
    skip_v1 = request.node.get_closest_marker("skip_v1")

    if use_v1:
        if skip_v1:
            pytest.skip("Skipping test on aphrodite V1")
        monkeypatch.setenv('APHRODITE_USE_V1', '1')
    else:
        monkeypatch.setenv('APHRODITE_USE_V1', '0')

    yield


@pytest.fixture
def reset_default_device():
    """
    Some tests, such as `test_punica_ops.py`, explicitly set the 
    default device, which can affect subsequent tests. Adding this fixture 
    helps avoid this problem.
    """
    original_device = torch.get_default_device()
    yield
    torch.set_default_device(original_device)
