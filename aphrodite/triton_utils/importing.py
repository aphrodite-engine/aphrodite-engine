from importlib.util import find_spec

from aphrodite.common.utils import print_warning_once
from aphrodite.platforms import current_platform

HAS_TRITON = find_spec("triton") is not None

if not HAS_TRITON and (current_platform.is_cuda() or
                       current_platform.is_rocm()):
    print_warning_once(
        "Triton not installed; certain GPU-related functions"
        " will be not be available."
    )
