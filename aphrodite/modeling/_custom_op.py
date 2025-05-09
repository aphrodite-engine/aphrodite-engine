from functools import lru_cache
from typing import Dict, Type

import torch.nn as nn
from loguru import logger

import aphrodite.common.envs as envs
from aphrodite.common.utils import (is_cpu, is_hip, is_triton, is_xpu,
                                    print_warning_once)
from aphrodite.compilation.levels import CompilationLevel
from aphrodite.platforms import current_platform


class CustomOp(nn.Module):
    """
    Base class for custom ops.
    Dispatches the forward method to the appropriate backend.
    """

    def __init__(self):
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
        enabled = self.enabled()
        logger.debug("custom op {} {}", self.__class__.name,
                     "enabled" if enabled else "disabled")

        if not enabled:
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


    @classmethod
    def enabled(cls) -> bool:
        # if no name, then it was not registered
        if not hasattr(cls, "name"):
            print_warning_once(
                f"Custom op {cls.__name__} was not registered, "
                f"which means it won't appear in the op registry. "
                f"It will be enabled/disabled based on the global settings.")
            return CustomOp.default_on()

        enabled = f"+{cls.name}" in envs.APHRODITE_CUSTOM_OPS
        disabled = f"-{cls.name}" in envs.APHRODITE_CUSTOM_OPS
        assert not (enabled
                    and disabled), f"Cannot enable and disable {cls.name}"

        return (CustomOp.default_on() or enabled) and not disabled

    # On by default if APHRODITE_TORCH_COMPILE_LEVEL < CompilationLevel.INDUCTOR
    # Specifying 'all' or 'none' in APHRODITE_CUSTOM_OPS takes precedence.
    @staticmethod
    @lru_cache()
    def default_on() -> bool:
        count_none = envs.APHRODITE_CUSTOM_OPS.count("none")
        count_all = envs.APHRODITE_CUSTOM_OPS.count("all")
        assert count_none + count_all <= 1, "Can only specify 'none' or 'all'"
        return envs.APHRODITE_TORCH_COMPILE_LEVEL < CompilationLevel.INDUCTOR \
            and not count_none > 0 or count_all > 0

    # Dictionary of all custom ops (classes, indexed by registered name).
    # To check if an op with a name is enabled, call .enabled() on the class.
    # Examples:
    # - MyOp.enabled()
    # - op_registry["my_op"].enabled()
    op_registry: Dict[str, Type['CustomOp']] = {}

    # Decorator to register custom ops.
    @classmethod
    def register(cls, name: str):

        def decorator(op_cls):
            assert name not in cls.op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            cls.op_registry[name] = op_cls
            return op_cls

        return decorator
