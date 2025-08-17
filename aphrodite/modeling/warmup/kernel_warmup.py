"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""
from typing import TYPE_CHECKING

import torch

import aphrodite.common.envs as envs
from aphrodite.modeling.warmup.deep_gemm_warmup import deep_gemm_warmup
from aphrodite.platforms import current_platform
from aphrodite.utils.deep_gemm import is_deep_gemm_supported
from aphrodite.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from aphrodite.v1.worker.gpu_model_runner import GPUModelRunner
    from aphrodite.v1.worker.gpu_worker import Worker


def kernel_warmup(worker: "Worker"):
    # Deep GEMM warmup
    do_deep_gemm_warmup = (envs.APHRODITE_USE_DEEP_GEMM
                           and is_deep_gemm_supported()
                           and not envs.APHRODITE_SKIP_DEEP_GEMM_WARMUP)
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    # FlashInfer autotune for Blackwell (SM 10.0) GPUs
    if has_flashinfer() and current_platform.is_device_capability(100):
        flashinfer_autotune(worker.model_runner)


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    """
    Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.
    """
    from aphrodite.utils.flashinfer import autotune

    with torch.inference_mode(), autotune():
        # We skip EPLB here since we don't want to record dummy metrics
        # When autotuning with number of tokens m, flashinfer will autotune
        # operations for all number of tokens up to m.
        # So we only need to run with the max number of tokens.
        runner._dummy_run(runner.scheduler_config.max_num_batched_tokens,
                          skip_eplb=True,
                          is_profile=True)
