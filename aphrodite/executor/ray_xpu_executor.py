import asyncio
from typing import List, Optional

import aphrodite.common.envs as envs
from aphrodite.common.utils import get_aphrodite_instance_id, make_async
from aphrodite.executor.ray_gpu_executor import (RayGPUExecutor,
                                                 RayGPUExecutorAsync)
from aphrodite.executor.xpu_executor import XPUExecutor


class RayXPUExecutor(RayGPUExecutor, XPUExecutor):

    def _get_env_vars_to_be_updated(self):
        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = self._run_workers("get_node_and_gpu_ids",
                                                    use_dummy_driver=True)

        APHRODITE_INSTANCE_ID = get_aphrodite_instance_id()

        # Set environment variables for the driver and workers.
        all_args_to_update_environment_variables = [({
            "APHRODITE_INSTANCE_ID":
            APHRODITE_INSTANCE_ID,
            "APHRODITE_TRACE_FUNCTION":
            str(envs.APHRODITE_TRACE_FUNCTION),
        }, ) for (_, _) in worker_node_and_gpu_ids]
        return all_args_to_update_environment_variables


class RayXPUExecutorAsync(RayXPUExecutor, RayGPUExecutorAsync):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver_exec_method = make_async(self.driver_worker.execute_method)
        self.pp_locks: Optional[List[asyncio.Lock]] = None
