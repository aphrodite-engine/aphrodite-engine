import types
from importlib.util import find_spec

from loguru import logger

from aphrodite.common.logger import log_once

HAS_TRITON = (
    find_spec("triton") is not None
    or find_spec("pytorch-triton-xpu") is not None  # Not compatible
)

if not HAS_TRITON:
    logger.info("Triton not installed or not compatible; certain GPU-related"
                " functions will not be available.")

    class TritonPlaceholder(types.ModuleType):

        def __init__(self):
            super().__init__("triton")
            self.jit = self._dummy_decorator("jit")
            self.autotune = self._dummy_decorator("autotune")
            self.heuristics = self._dummy_decorator("heuristics")
            self.language = TritonLanguagePlaceholder()
            log_once(
                "WARNING",
                "Triton is not installed. Using dummy decorators. "
                "Install it via `pip install triton` to enable kernel"
                "compilation.")

        def _dummy_decorator(self, name):

            def decorator(func=None, **kwargs):
                if func is None:
                    return lambda f: f
                return func

            return decorator

    class TritonLanguagePlaceholder(types.ModuleType):

        def __init__(self):
            super().__init__("triton.language")
            self.constexpr = None
            self.dtype = None
            self.int64 = None
