import torch

from aphrodite.engine.args_tools import EngineArgs
from aphrodite.common.sequence import ExecuteModelRequest
from aphrodite.common.utils import get_distributed_init_method, get_ip, get_open_port
from aphrodite.worker.worker import Worker


def test_swap() -> None:
    # Configure the engine.
    engine_args = EngineArgs(model="distilbert/distilgpt2",
                             dtype="half",
                             load_format="dummy")
    engine_config = engine_args.create_engine_config()
    engine_config.cache_config.num_gpu_blocks = 1000
    engine_config.cache_config.num_cpu_blocks = 1000

    # Create the worker.
    distributed_init_method = get_distributed_init_method(
        get_ip(), get_open_port())
    worker = Worker(
        aphrodite_config=engine_config,
        local_rank=0,
        rank=0,
        distributed_init_method=distributed_init_method,
        is_driver_worker=True,
    )

    # Initialize the worker.
    worker.init_device()
    worker.load_model()
    worker.initialize_cache(
        num_gpu_blocks=engine_config.cache_config.num_gpu_blocks,
        num_cpu_blocks=engine_config.cache_config.num_cpu_blocks)

    # Randomly initialize the cache.
    gpu_cache = worker.cache_engine[0].gpu_cache
    cpu_cache = worker.cache_engine[0].cpu_cache
    num_layers = len(gpu_cache)
    for i in range(num_layers):
        gpu_key_cache, gpu_value_cache = gpu_cache[i]
        gpu_key_cache.random_()
        gpu_value_cache.random_()
        cpu_key_cache, cpu_value_cache = cpu_cache[i]
        cpu_key_cache.random_()
        cpu_value_cache.random_()

    allclose = lambda a, b: torch.allclose(
        a.cuda(), b.cuda(), rtol=0.0, atol=0.0)

    # Test swap out.
    blocks_to_swap_out = [(3, 72), (56, 35), (84, 34)]
    execute_model_req = ExecuteModelRequest(
        seq_group_metadata_list=[],
        blocks_to_swap_in=[],
        blocks_to_swap_out=blocks_to_swap_out,
        blocks_to_copy=[],
    )
    worker.execute_model(execute_model_req=execute_model_req)

    for i in range(num_layers):
        gpu_key_cache, gpu_value_cache = gpu_cache[i]
        cpu_key_cache, cpu_value_cache = cpu_cache[i]
        for src, dst in blocks_to_swap_out:
            assert allclose(gpu_key_cache[src], cpu_key_cache[dst])
            assert allclose(gpu_value_cache[src], cpu_value_cache[dst])

    # Test swap in.
    execute_model_req.blocks_to_swap_out = []
    execute_model_req.blocks_to_swap_in = [
        (19, 45),
        (67, 23),
        (12, 78),
        (40, 99),
        (1, 71),
    ]
    worker.execute_model(execute_model_req=execute_model_req)

    for i in range(num_layers):
        gpu_key_cache, gpu_value_cache = gpu_cache[i]
        cpu_key_cache, cpu_value_cache = cpu_cache[i]
        for src, dst in execute_model_req.blocks_to_swap_in:
            assert allclose(gpu_key_cache[dst], cpu_key_cache[src])
            assert allclose(gpu_value_cache[dst], cpu_value_cache[src])
