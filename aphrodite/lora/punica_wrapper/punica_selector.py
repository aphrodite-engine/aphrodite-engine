from loguru import logger

from aphrodite.common.logger import log_once
from aphrodite.common.utils import resolve_obj_by_qualname
from aphrodite.platforms import current_platform

from .punica_base import PunicaWrapperBase


def get_punica_wrapper(*args, **kwargs) -> PunicaWrapperBase:
    punica_wrapper_qualname = current_platform.get_punica_wrapper()
    punica_wrapper_cls = resolve_obj_by_qualname(punica_wrapper_qualname)
    punica_wrapper = punica_wrapper_cls(*args, **kwargs)
    assert punica_wrapper is not None, \
        "the punica_wrapper_qualname(" + punica_wrapper_qualname + ") is wrong."
    log_once("INFO", "Using {}.", punica_wrapper_qualname.rsplit(".", 1)[1])
    return punica_wrapper
