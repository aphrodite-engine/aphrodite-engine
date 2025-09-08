import os

import torch

# set some common config/environment variables that should be set
# for all processes created by aphrodite and all processes
# that interact with aphrodite workers.
# they are executed whenever `import aphrodite` is called.

# it avoids unintentional cuda initialization from torch.cuda.is_available()
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'

os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
torch._inductor.config.compile_threads = 1
