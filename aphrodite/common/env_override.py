import os

import torch

from loguru import logger

# set some common config/environment variables that should be set
# for all processes created by aphrodite and all processes
# that interact with aphrodite workers.
# they are executed whenever `import aphrodite` is called.

if os.environ.get('NCCL_CUMEM_ENABLE', '0') != '0':
    logger.warning(
        "NCCL_CUMEM_ENABLE is set to {}, skipping override. "
        "This may increase memory overhead with cudagraph+allreduce: "
        "https://github.com/NVIDIA/nccl/issues/1234",
        os.environ['NCCL_CUMEM_ENABLE'])
elif not os.path.exists('/dev/nvidia-caps-imex-channels'):
    # NCCL requires NCCL_CUMEM_ENABLE to work with
    # multi-node NVLink, typically on GB200-NVL72 systems.
    # The ultimate way to detect multi-node NVLink is to use
    # NVML APIs, which are too expensive to call here.
    # As an approximation, we check the existence of
    # /dev/nvidia-caps-imex-channels, used by
    # multi-node NVLink to communicate across nodes.
    # This will still cost some GPU memory, but it is worthwhile
    # because we can get very fast cross-node bandwidth with NVLink.
    os.environ['NCCL_CUMEM_ENABLE'] = '0'

# it avoids unintentional cuda initialization from torch.cuda.is_available()
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'

os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
torch._inductor.config.compile_threads = 1
