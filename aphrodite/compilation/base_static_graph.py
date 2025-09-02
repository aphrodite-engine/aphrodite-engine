from typing import Any, Callable, Protocol

from aphrodite.config import AphroditeConfig, CUDAGraphMode


class AbstractStaticGraphWrapper(Protocol):
    """
    StaticGraphWrapper interface that allows platforms to wrap a callable
    to be captured as a static graph.
    """

    def __init__(self, runnable: Callable, aphrodite_config: AphroditeConfig,
                 runtime_mode: CUDAGraphMode, **kwargs):
        """
        Initializes the StaticGraphWrapper class with graph capturing and
        execution-related configurations.

        Args:
            runnable (Callable): The callable to be wrapped and captured.
            aphrodite_config (AphroditeConfig): Global configuration for Aphrodite.
            runtime_mode (CUDAGraphMode): The style of the static
                graph runtime. See CUDAGraphMode in aphrodite/config.py.
                Note that only the subset enum `NONE`, `PIECEWISE` and `FULL`
                are used as concrete runtime mode for cudagraph dispatching.
        Keyword Args:
            kwargs: Additional keyword arguments for platform-specific
                configurations.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Any:
        """
        Executes the wrapped callable.

        If the current runtime mode in the ForwardContext matches the runtime
        mode of this instance, it replays the CUDAGraph or captures it using
        the callable if it hasn't been captured yet. Otherwise, it calls the
        original callable directly.

        Args:
            *args: Variable length input arguments to be passed into the
                callable.
            **kwargs: Keyword arguments to be passed into the callable.

        Returns:
            Any: Output of the executed callable.
        """
        raise NotImplementedError
