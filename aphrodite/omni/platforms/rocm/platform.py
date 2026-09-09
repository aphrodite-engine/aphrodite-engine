# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from aphrodite import envs
from aphrodite.config import AphroditeConfig
from aphrodite.config.kernel import IrOpPriorityConfig
from aphrodite.logger import init_logger
from aphrodite.omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from aphrodite.omni.platforms.interface import OmniPlatform, OmniPlatformEnum
from aphrodite.omni.platforms.rocm.patch import apply_patches
from aphrodite.platforms.rocm import RocmPlatform

logger = init_logger(__name__)


class RocmOmniPlatform(OmniPlatform, RocmPlatform):
    """ROCm/AMD GPU implementation of OmniPlatform.

    Inherits all ROCm-specific implementations from Sonar's RocmPlatform,
    and adds Omni-specific interfaces from OmniPlatform.


    NOTE: AR Attention Backend Overriding Logic:
    ------------------------------------------
    Since Sonar v0.19.0, the default attention backend is ROCM_ATTN for ROCm.
    However, the compatibility of ROCM_ATTN with Omni is not guaranteed.
    Therefore, we still use TRITON_ATTN as the default attention backend,
    when the selected_backend is not specified.

    So the behaviour of the attention backend overriding logic currently lives in
    extract_legacy_stage_metadata in `aphrodite/omni/engine/stage_init_utils.py`

    ```
    if current_omni_platform.is_rocm():
        print(f"engine_args: {str(engine_args)}")
        if engine_args.get("attention_backend") is None:
            from aphrodite._aiter_ops import rocm_aiter_ops

            if rocm_aiter_ops.is_enabled():
                engine_args["attention_backend"] = "ROCM_AITER_FA"
            # Before Sonar v0.19.0, the default attention backend is TRITON_ATTN for ROCm.
            # Since Sonar v0.19.0, the default attention backend is ROCM_ATTN for ROCm.
            # However, the compatibility of ROCM_ATTN with Omni is not guaranteed.
            # Therefore, we still use TRITON_ATTN as the default attention backend,
            # when the selected_backend is not specified.
            engine_args["attention_backend"] = "TRITON_ATTN"
    ```

    """

    _omni_enum = OmniPlatformEnum.ROCM

    def __init__(self):
        super().__init__()
        apply_patches()

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "aphrodite.omni.worker.gpu_ar_worker.GPUARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "aphrodite.omni.worker.gpu_generation_worker.GPUGenerationWorker"

    @classmethod
    def has_flash_attn_package(cls) -> bool:
        from aphrodite.omni.diffusion.attention.backends.utils.fa import is_flash_attn_installed

        return is_flash_attn_installed()

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
        allow_trtllm_default: bool = False,
    ) -> str:
        """Get the diffusion attention backend class path for ROCm platform.

        ROCm supports FLASH_ATTN via the aiter library, and SDPA as fallback.

        Args:
            selected_backend: User-selected backend name (e.g., "FLASH_ATTN",
                "TORCH_SDPA"). If None, uses platform default.
            head_size: Attention head size.
            allow_trtllm_default: Does not support TRTLLM backend;
                arg accepted for signature parity but unused.
        Returns:
            Fully qualified class path of the selected backend.
        """
        from aphrodite._aiter_ops import is_aiter_found_and_supported

        # Check if aiter is available for Flash Attention support
        # aiter currently only is supported on gfx942 and gfx950
        # https://github.com/vllm-project/vllm/blob/main/vllm/_aiter_ops.py
        compute_capability = torch.cuda.get_device_capability()
        major, minor = compute_capability
        capability = major * 10 + minor
        aiter_supported = is_aiter_found_and_supported() and 90 < capability < 100

        if selected_backend is not None:
            backend_upper = selected_backend.upper()
            cls.validate_diffusion_attn_backend(backend_upper)
            if backend_upper in ("FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB"):
                logger.warning(
                    "HuggingFace kernels-backed FlashAttention is "
                    "not supported on ROCm. Falling back to local "
                    "FLASH_ATTN."
                )
                backend_upper = "FLASH_ATTN"

            if backend_upper == "FLASH_ATTN" and not aiter_supported:
                logger.warning(
                    "Flash Attention requires `aiter` library which is only supported "
                    "on gfx942 and gfx950. Falling back to TORCH_SDPA backend."
                )
                logger.debug("Defaulting to diffusion attention backend SDPA")
                return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
            backend = DiffusionAttentionBackendEnum[backend_upper]
            logger.debug("Using diffusion attention backend '%s'", backend_upper)
            return backend.get_path()

        # Choose to enable Flash Attention by default on ROCm
        # whenever possible as it is the fastest backend
        if aiter_supported:
            logger.debug("Defaulting to diffusion attention backend FLASH_ATTN")
            return DiffusionAttentionBackendEnum.FLASH_ATTN.get_path()

        logger.debug("Defaulting to diffusion attention backend SDPA")
        return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return True

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "aphrodite/omni/deploy"

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        if local_rank is None:
            return torch.device("cuda")
        return torch.device("cuda", local_rank)

    @classmethod
    def get_device_count(cls) -> int:
        return torch.accelerator.device_count()

    @classmethod
    def get_device_version(cls) -> str | None:
        if torch.version.hip is not None:
            hip_version = torch.version.hip
            return hip_version.split("-")[0]
        return None

    @classmethod
    def synchronize(cls) -> None:
        torch.accelerator.synchronize()

    @classmethod
    def record_device_event(cls) -> torch.Event | None:
        try:
            event = torch.Event()
            event.record()
            return event
        except Exception:
            logger.warning("Failed to record device event for cross-stream sync")
            return None

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        free, _ = torch.cuda.mem_get_info(device)
        return free

    @classmethod
    def get_device_memory(cls, device: torch.device | None = None) -> tuple[int, int]:
        free, total = torch.cuda.mem_get_info(device)
        return free, total

    @classmethod
    def set_device_control_env_var(cls, devices: str | int | None) -> None:
        import os

        if devices is None:
            cls.unset_device_control_env_var()
        else:
            os.environ["HIP_VISIBLE_DEVICES"] = str(devices)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)

    @classmethod
    def unset_device_control_env_var(cls) -> None:
        import os

        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    @classmethod
    def get_default_ir_op_priority(cls, aphrodite_config: AphroditeConfig) -> IrOpPriorityConfig:
        """Copied from upstream RocmPlatform with inductor-aware logic.

        When inductor is active (compiling) use native as the default;
        otherwise prefer aphrodite_c kernels where available.
        Preserves omni-specific is_custom_op_enabled('rms_norm') check.
        """
        from aphrodite.config.compilation import CompilationMode

        # TODO(luka/TJ) use aiter, aphrodite_c, native by default on ROCm
        cc = aphrodite_config.compilation_config
        using_inductor = cc.backend == "inductor" and cc.mode != CompilationMode.NONE
        default = ["native"] if using_inductor else ["aphrodite_c", "native"]

        # This (mostly) preserves previous CustomOp behavior
        # Necessary on ROCm because it's common that users
        # enable rms_norm to use the aiter kernel.
        # TODO(luka/TJ) remove env vars completely
        if (
            cc.is_custom_op_enabled("rms_norm")
            and envs.APHRODITE_ROCM_USE_AITER
            and envs.APHRODITE_ROCM_USE_AITER_RMSNORM
        ):
            rms_norm = ["aiter"] + default
        else:
            rms_norm = default

        return IrOpPriorityConfig.with_default(default, rms_norm=rms_norm, fused_add_rms_norm=rms_norm)
