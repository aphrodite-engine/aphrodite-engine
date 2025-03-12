import torch.nn as nn

import aphrodite.common.envs as envs
from aphrodite.common.utils import is_cpu, is_hip, is_triton, is_xpu
from aphrodite.compilation.levels import CompilationLevel
from aphrodite.platforms import current_platform


class CustomOp(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._forward_method = self.dispatch_forward()

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        """PyTorch-native implementation of the forward method.
        This method is optional. If implemented, it can be used with compilers
        such as torch.compile or PyTorch XLA. Also, it can be used for testing
        purposes.
        """
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        # By default, we assume that HIP ops are compatible with CUDA ops.
        return self.forward_cuda(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        raise NotImplementedError

    def forward_cpu(self, *args, **kwargs):
        # By default, we assume that CPU ops are compatible with CUDA ops.
        return self.forward_cuda(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs):
        # By default, we assume that TPU ops are compatible with the
        # PyTorch-native implementation.
        # NOTE: This is a placeholder for future extensions.
        return self.forward_native(*args, **kwargs)

    def forward_gaudi(self, *args, **kwargs):
        # By default, we assume that Gaudi ops are compatible with the
        # PyTorch-native implementation.
        # NOTE: This is a placeholder for future extensions.
        return self.forward_native(*args, **kwargs)

    def forward_triton(self, *args, **kwargs):
        raise NotImplementedError

    def dispatch_forward(self):
        # NOTE: Here we assume that Aphrodite was built for only one
        # specific backend. Currently, we do not support dynamic dispatching.
        if envs.APHRODITE_TORCH_COMPILE_LEVEL >= CompilationLevel.INDUCTOR:
            return self.forward_native
        if is_hip():
            return self.forward_hip
        elif is_cpu():
            return self.forward_cpu
        elif current_platform.is_tpu():
            return self.forward_tpu
        elif is_xpu():
            return self.forward_xpu
        elif is_triton():
            return self.forward_triton
        else:
            return self.forward_cuda
