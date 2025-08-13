import logging
import traceback
from itertools import chain
from typing import TYPE_CHECKING, Optional

from aphrodite.common import envs
from aphrodite.plugins import load_plugins_by_group
from aphrodite.utils import resolve_obj_by_qualname, supports_xccl

from .interface import _Backend  # noqa: F401
from .interface import CpuArchEnum, Platform, PlatformEnum

logger = logging.getLogger(__name__)


def aphrodite_version_matches_substr(substr: str) -> bool:
    """
    Check to see if the Aphrodite version matches a substring.
    """
    from importlib.metadata import PackageNotFoundError, version
    try:
        aphrodite_version = version("aphrodite-engine")
    except PackageNotFoundError as e:
        logger.warning(
            "The Aphrodite package was not found, so its version could not be "
            "inspected. This may cause platform detection to fail.")
        raise e
    return substr in aphrodite_version


def tpu_platform_plugin() -> Optional[str]:
    logger.debug("Checking if TPU platform is available.")

    # Check for Pathways TPU proxy
    if envs.APHRODITE_TPU_USING_PATHWAYS:
        logger.debug("Confirmed TPU platform is available via Pathways proxy.")
        return "tpu_commons.platforms.tpu_jax.TpuPlatform"

    # Check for libtpu installation
    try:
        # While it's technically possible to install libtpu on a
        # non-TPU machine, this is a very uncommon scenario. Therefore,
        # we assume that libtpu is installed only if the machine
        # has TPUs.

        import libtpu  # noqa: F401
        logger.debug("Confirmed TPU platform is available.")
        return "aphrodite.platforms.tpu.TpuPlatform"
    except Exception as e:
        logger.debug("TPU platform is not available because: {}", str(e))
        return None


def cuda_platform_plugin() -> Optional[str]:
    is_cuda = False
    logger.debug("Checking if CUDA platform is available.")
    try:
        from aphrodite.utils import import_pynvml
        pynvml = import_pynvml()
        pynvml.nvmlInit()
        try:
            # NOTE: Edge case: aphrodite cpu build on a GPU machine.
            # Third-party pynvml can be imported in cpu build,
            # we need to check if aphrodite is built with cpu too.
            # Otherwise, aphrodite will always activate cuda plugin
            # on a GPU machine, even if in a cpu build.
            is_cuda = (pynvml.nvmlDeviceGetCount() > 0
                       and not aphrodite_version_matches_substr("cpu"))
            if pynvml.nvmlDeviceGetCount() <= 0:
                logger.debug(
                    "CUDA platform is not available because no GPU is found.")
            if aphrodite_version_matches_substr("cpu"):
                logger.debug("CUDA platform is not available because"
                             " Aphrodite is built with CPU.")
            if is_cuda:
                logger.debug("Confirmed CUDA platform is available.")
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        logger.debug("Exception happens when checking CUDA platform: {}",
                     str(e))
        if "nvml" not in e.__class__.__name__.lower():
            # If the error is not related to NVML, re-raise it.
            raise e

        # CUDA is supported on Jetson, but NVML may not be.
        import os

        def cuda_is_jetson() -> bool:
            return os.path.isfile("/etc/nv_tegra_release") \
                or os.path.exists("/sys/class/tegra-firmware")

        if cuda_is_jetson():
            logger.debug("Confirmed CUDA platform is available on Jetson.")
            is_cuda = True
        else:
            logger.debug("CUDA platform is not available because: {}", str(e))

    return "aphrodite.platforms.cuda.CudaPlatform" if is_cuda else None


def rocm_platform_plugin() -> Optional[str]:
    is_rocm = False
    logger.debug("Checking if ROCm platform is available.")
    try:
        import amdsmi
        amdsmi.amdsmi_init()
        try:
            if len(amdsmi.amdsmi_get_processor_handles()) > 0:
                is_rocm = True
                logger.debug("Confirmed ROCm platform is available.")
            else:
                logger.debug("ROCm platform is not available because"
                             " no GPU is found.")
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception as e:
        logger.debug("ROCm platform is not available because: {}", str(e))

    return "aphrodite.platforms.rocm.RocmPlatform" if is_rocm else None


def xpu_platform_plugin() -> Optional[str]:
    is_xpu = False
    logger.debug("Checking if XPU platform is available.")
    try:
        # installed IPEX if the machine has XPUs.
        import intel_extension_for_pytorch  # noqa: F401
        import torch
        if supports_xccl():
            dist_backend = "xccl"
        else:
            dist_backend = "ccl"
            import oneccl_bindings_for_pytorch  # noqa: F401

        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            is_xpu = True
            from aphrodite.platforms.xpu import XPUPlatform
            XPUPlatform.dist_backend = dist_backend
            logger.debug("Confirmed {} backend is available.",
                         XPUPlatform.dist_backend)
            logger.debug("Confirmed XPU platform is available.")
    except Exception as e:
        logger.debug("XPU platform is not available because: {}", str(e))

    return "aphrodite.platforms.xpu.XPUPlatform" if is_xpu else None


def cpu_platform_plugin() -> Optional[str]:
    logger.debug("Checking if CPU platform is available.")
    try:
        import sys
        # On macOS, if MPS is built and available, do NOT activate CPU
        if sys.platform.startswith("darwin"):
            try:
                import torch  # type: ignore
                if torch.backends.mps.is_built() and torch.backends.mps.is_available():
                    logger.debug("MPS detected on macOS; CPU platform will not be activated.")
                    return None
            except Exception:
                # If torch import fails, fall through to CPU detection
                pass

        is_cpu = aphrodite_version_matches_substr("cpu")
        if is_cpu:
            logger.debug("Confirmed CPU platform is available because Aphrodite is built with CPU.")
        if not is_cpu:
            # As a fallback, allow CPU on macOS only if MPS is not available
            is_cpu = sys.platform.startswith("darwin")
            if is_cpu:
                logger.debug("Confirmed CPU platform is available on macOS (no MPS).")

    except Exception as e:
        logger.debug("CPU platform is not available because: {}", str(e))
        return None

    return "aphrodite.platforms.cpu.CpuPlatform" if is_cpu else None


def neuron_platform_plugin() -> Optional[str]:
    tnx_installed = False
    nxd_installed = False
    logger.debug("Checking if Neuron platform is available.")
    try:
        import transformers_neuronx  # noqa: F401
        tnx_installed = True
        logger.debug("Confirmed Neuron platform is available because"
                     " transformers_neuronx is found.")
    except ImportError:
        pass

    try:
        import neuronx_distributed_inference  # noqa: F401
        nxd_installed = True
        logger.debug("Confirmed Neuron platform is available because"
                     " neuronx_distributed_inference is found.")
    except ImportError:
        pass

    is_neuron = tnx_installed or nxd_installed
    return "aphrodite.platforms.neuron.NeuronPlatform" if is_neuron else None

def mps_platform_plugin() -> Optional[str]:  
    is_mps = False
    logger.debug("Checking if MPS platform is available.")  
    try:  
        import torch  
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():  
            is_mps = True
            logger.debug("Confirmed MPS platform is available.")  
            return "aphrodite.platforms.mps.MpsPlatform"  
        else:  
            logger.debug("MPS platform is not available.")  
            return None  
    except Exception as e:  
        logger.debug("MPS platform is not available because: {}", str(e))
        return None
    return "aphrodite.platforms.mps.MpsPlatform" if is_mps else None


builtin_platform_plugins = {
    'tpu': tpu_platform_plugin,
    'cuda': cuda_platform_plugin,
    'rocm': rocm_platform_plugin,
    'xpu': xpu_platform_plugin,
    'cpu': cpu_platform_plugin,
    'neuron': neuron_platform_plugin,
    'mps': mps_platform_plugin,
}


def resolve_current_platform_cls_qualname() -> str:
    platform_plugins = load_plugins_by_group('aphrodite.platform_plugins')

    activated_plugins = []

    for name, func in chain(builtin_platform_plugins.items(),
                            platform_plugins.items()):
        try:
            assert callable(func)
            platform_cls_qualname = func()
            if platform_cls_qualname is not None:
                activated_plugins.append(name)
        except Exception:
            pass

    activated_builtin_plugins = list(
        set(activated_plugins) & set(builtin_platform_plugins.keys()))
    activated_oot_plugins = list(
        set(activated_plugins) & set(platform_plugins.keys()))

    if len(activated_oot_plugins) >= 2:
        raise RuntimeError(
            "Only one platform plugin can be activated, but got: "
            f"{activated_oot_plugins}")
    elif len(activated_oot_plugins) == 1:
        platform_cls_qualname = platform_plugins[activated_oot_plugins[0]]()
        logger.info("Platform plugin {} is activated",
                    activated_oot_plugins[0])
    elif len(activated_builtin_plugins) >= 2:
        raise RuntimeError(
            "Only one platform plugin can be activated, but got: "
            f"{activated_builtin_plugins}")
    elif len(activated_builtin_plugins) == 1:
        platform_cls_qualname = builtin_platform_plugins[
            activated_builtin_plugins[0]]()
        logger.info("Automatically detected platform {}.",
                    activated_builtin_plugins[0])
    else:
        platform_cls_qualname = "aphrodite.platforms.interface.UnspecifiedPlatform"
        logger.info(
            "No platform detected, Aphrodite is running on UnspecifiedPlatform")
    return platform_cls_qualname


_current_platform = None
_init_trace: str = ''

if TYPE_CHECKING:
    current_platform: Platform


def __getattr__(name: str):
    if name == 'current_platform':
        # lazy init current_platform.
        # 1. out-of-tree platform plugins need `from aphrodite.platforms import
        #    Platform` so that they can inherit `Platform` class. Therefore,
        #    we cannot resolve `current_platform` during the import of
        #    `aphrodite.platforms`.
        # 2. when users use out-of-tree platform plugins, they might run
        #    `import aphrodite`, some aphrodite internal code might access
        #    `current_platform` during the import, and we need to make sure
        #    `current_platform` is only resolved after the plugins are loaded
        #    (we have tests for this, if any developer violate this, they will
        #    see the test failures).
        global _current_platform
        if _current_platform is None:
            platform_cls_qualname = resolve_current_platform_cls_qualname()
            _current_platform = resolve_obj_by_qualname(
                platform_cls_qualname)()
            global _init_trace
            _init_trace = "".join(traceback.format_stack())
        return _current_platform
    elif name in globals():
        return globals()[name]
    else:
        raise AttributeError(
            f"No attribute named '{name}' exists in {__name__}.")


__all__ = [
    'Platform', 'PlatformEnum', 'current_platform', 'CpuArchEnum',
    "_init_trace"
]
