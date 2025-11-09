# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/utils.py

import ctypes
import importlib
import importlib.util
import inspect
import math
import os
import signal
import socket
import sys
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, partial, wraps
from typing import Any, TypeVar, cast

import cloudpickle
import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from remote_pdb import RemotePdb
from torch.distributed.fsdp import MixedPrecisionPolicy

from aphrodite.logger import init_logger

T = TypeVar("T")
logger = init_logger(__name__)

# TODO(will): used to convert server_args.precision to torch.dtype. Find a
# cleaner way to do this.
PRECISION_TO_TYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

STR_BACKEND_ENV_VAR: str = "APHRODITE_ATTENTION_BACKEND"
STR_ATTN_CONFIG_ENV_VAR: str = "APHRODITE_ATTENTION_CONFIG"


prev_set_stream = torch.cuda.set_stream

_current_stream = None


def _patched_set_stream(stream: torch.cuda.Stream | None) -> None:
    global _current_stream
    _current_stream = stream
    if stream is not None:
        prev_set_stream(stream)


torch.cuda.set_stream = _patched_set_stream


def warn_for_unimplemented_methods(cls: type[T]) -> type[T]:
    """
    A replacement for `abc.ABC`.
    When we use `abc.ABC`, subclasses will fail to instantiate
    if they do not implement all abstract methods.
    Here, we only require `raise NotImplementedError` in the
    base class, and log a warning if the method is not implemented
    in the subclass.
    """

    original_init = cls.__init__

    def find_unimplemented_methods(self: object):
        unimplemented_methods = []
        for attr_name in dir(self):
            # bypass inner method
            if attr_name.startswith("_"):
                continue

            try:
                attr = getattr(self, attr_name)
                # get the func of callable method
                if callable(attr):
                    attr_func = attr.__func__
            except AttributeError:
                continue
            src = inspect.getsource(attr_func)
            if "NotImplementedError" in src:
                unimplemented_methods.append(attr_name)
        if unimplemented_methods:
            method_names = ",".join(unimplemented_methods)
            msg = f"Methods {method_names} not implemented in {self}"
            logger.warning(msg)

    @wraps(original_init)
    def wrapped_init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        find_unimplemented_methods(self)

    type.__setattr__(cls, "__init__", wrapped_init)
    return cls


def align_to(value: int, alignment: int) -> int:
    """align height, width according to alignment

    Args:
        value (int): height or width
        alignment (int): target alignment factor

    Returns:
        int: the aligned value
    """
    return int(math.ceil(value / alignment) * alignment)


def update_environment_variables(envs: dict[str, str]):
    for k, v in envs.items():
        if k in os.environ and os.environ[k] != v:
            logger.warning(
                "Overwriting environment variable %s from '%s' to '%s'",
                k,
                os.environ[k],
                v,
            )
        os.environ[k] = v


def run_method(obj: Any, method: str | bytes | Callable, args: tuple[Any], kwargs: dict[str, Any]) -> Any:
    """
    Run a method of an object with the given arguments and keyword arguments.
    If the method is string, it will be converted to a method using getattr.
    If the method is serialized bytes and will be deserialized using
    cloudpickle.
    If the method is a callable, it will be called directly.
    """
    if isinstance(method, bytes):
        func = partial(cloudpickle.loads(method), obj)
    elif isinstance(method, str):
        try:
            func = getattr(obj, method)
        except AttributeError:
            raise NotImplementedError(f"Method {method!r} is not implemented.") from None
    else:
        func = partial(method, obj)  # type: ignore
    return func(*args, **kwargs)


# TODO: validate that this is fine
def kill_itself_when_parent_died() -> None:
    # if sys.platform == "linux":
    # sigkill this process when parent worker manager dies
    PR_SET_PDEATHSIG = 1
    import platform

    if platform.system() == "Linux":
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    # elif platform.system() == "Darwin":
    #     libc = ctypes.CDLL("libc.dylib")
    #     logger.warning("kill_itself_when_parent_died is only supported in linux.")
    else:
        logger.warning("kill_itself_when_parent_died is only supported in linux.")


def get_exception_traceback() -> str:
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


class TypeBasedDispatcher:
    def __init__(self, mapping: list[tuple[type, Callable]]):
        self._mapping = mapping

    def __call__(self, obj: Any):
        for ty, fn in self._mapping:
            if isinstance(obj, ty):
                return fn(obj)
        raise ValueError(f"Invalid object: {obj}")


# For non-torch.distributed debugging
def remote_breakpoint() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("localhost", 0))  # Let the OS pick an ephemeral port.
        port = s.getsockname()[1]
        RemotePdb(host="localhost", port=port).set_trace()


@dataclass
class MixedPrecisionState:
    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None
    output_dtype: torch.dtype | None = None
    compute_dtype: torch.dtype | None = None
    mp_policy: MixedPrecisionPolicy | None = None


# Thread-local storage for mixed precision state
_mixed_precision_state = threading.local()


def get_mixed_precision_state() -> MixedPrecisionState:
    """Get the current mixed precision state."""
    if not hasattr(_mixed_precision_state, "state"):
        raise ValueError("Mixed precision state not set")
    return cast(MixedPrecisionState, _mixed_precision_state.state)


def set_mixed_precision_policy(
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    output_dtype: torch.dtype | None = None,
    mp_policy: MixedPrecisionPolicy | None = None,
):
    """Set mixed precision policy globally.

    Args:
        param_dtype: Parameter dtype used for training
        reduce_dtype: Reduction dtype used for gradients
        output_dtype: Optional output dtype
    """
    state = MixedPrecisionState(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
        mp_policy=mp_policy,
    )
    _mixed_precision_state.state = state


def get_compute_dtype() -> torch.dtype:
    """Get the current compute dtype from mixed precision policy.

    Returns:
        torch.dtype: The compute dtype to use, defaults to get_default_dtype() if no policy set
    """
    if not hasattr(_mixed_precision_state, "state"):
        return torch.get_default_dtype()
    else:
        state = get_mixed_precision_state()
        return state.param_dtype


def dict_to_3d_list(
    mask_strategy: dict[str, Any] | None = None,
    t_max: int | None = None,
    l_max: int | None = None,
    h_max: int | None = None,
) -> list[list[list[torch.Tensor | None]]]:
    """
    Convert a dictionary of mask indices to a 3D list of tensors.
    Args:
        mask_strategy: keys are "t_l_h", values are torch.Tensor masks.
        t_max, l_max, h_max: if provided (all three), force the output shape to (t_max, l_max, h_max).
                            If all three are None, infer shape from the data.
    """
    # Case 1: no data, but fixed shape requested
    if mask_strategy is None:
        assert t_max is not None and l_max is not None and h_max is not None, (
            "If mask_strategy is None, you must provide t_max, l_max, and h_max"
        )
        return [[[None for _ in range(h_max)] for _ in range(l_max)] for _ in range(t_max)]

    # Parse all keys into integer tuples
    indices = [tuple(map(int, key.split("_"))) for key in mask_strategy]

    # Decide on dimensions
    if t_max is None and l_max is None and h_max is None:
        # fully dynamic: infer from data
        max_timesteps_idx = max(t for t, _, _ in indices) + 1
        max_layer_idx = max(l for _, l, _ in indices) + 1  # noqa: E741
        max_head_idx = max(h for _, _, h in indices) + 1
    else:
        # require all three to be provided
        assert t_max is not None and l_max is not None and h_max is not None, (
            "Either supply none of (t_max, l_max, h_max) to infer dimensions, or supply all three to fix the shape."
        )
        max_timesteps_idx = t_max
        max_layer_idx = l_max
        max_head_idx = h_max

    # Preallocate
    result = [[[None for _ in range(max_head_idx)] for _ in range(max_layer_idx)] for _ in range(max_timesteps_idx)]

    # Fill in, skipping any out-of-bounds entries
    for key, value in mask_strategy.items():
        t, l, h = map(int, key.split("_"))  # noqa: E741
        if 0 <= t < max_timesteps_idx and 0 <= l < max_layer_idx and 0 <= h < max_head_idx:
            result[t][l][h] = value
        # else: silently ignore any key that doesn't fit

    return result


def set_random_seed(seed: int) -> None:
    from aphrodite.diffusion.runtime.platforms import current_platform

    current_platform.seed_everything(seed)


@lru_cache(maxsize=1)
def is_vsa_available() -> bool:
    return importlib.util.find_spec("vsa") is not None


@lru_cache(maxsize=1)
def is_vmoba_available() -> bool:
    if importlib.util.find_spec("kernel.csrc.attn.vmoba_attn.vmoba") is None:
        return False
    try:
        import flash_attn

        return flash_attn.__version__ >= "2.7.4"
    except Exception:
        return False


# adapted from: https://github.com/Wan-Video/Wan2.2/blob/main/wan/utils/utils.py
def masks_like(tensor, zero=False, generator=None, p=0.2) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    assert isinstance(tensor, list)
    out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    if zero:
        if generator is not None:
            for u, v in zip(out1, out2, strict=False):
                random_num = torch.rand(1, generator=generator, device=generator.device).item()
                if random_num < p:
                    u[:, 0] = (
                        torch.normal(
                            mean=-3.5,
                            std=0.5,
                            size=(1,),
                            device=u.device,
                            generator=generator,
                        )
                        .expand_as(u[:, 0])
                        .exp()
                    )
                    v[:, 0] = torch.zeros_like(v[:, 0])
                else:
                    u[:, 0] = u[:, 0]
                    v[:, 0] = v[:, 0]

        else:
            for u, v in zip(out1, out2, strict=False):
                u[:, 0] = torch.zeros_like(u[:, 0])
                v[:, 0] = torch.zeros_like(v[:, 0])

    return out1, out2


# adapted from: https://github.com/Wan-Video/Wan2.2/blob/main/wan/utils/utils.py
def best_output_size(w, h, dw, dh, expected_area):
    # float output size
    ratio = w / h
    ow = (expected_area * ratio) ** 0.5
    oh = expected_area / ow

    # process width first
    ow1 = int(ow // dw * dw)
    oh1 = int(expected_area / ow1 // dh * dh)
    assert ow1 % dw == 0 and oh1 % dh == 0 and ow1 * oh1 <= expected_area
    ratio1 = ow1 / oh1

    # process height first
    oh2 = int(oh // dh * dh)
    ow2 = int(expected_area / oh2 // dw * dw)
    assert oh2 % dh == 0 and ow2 % dw == 0 and ow2 * oh2 <= expected_area
    ratio2 = ow2 / oh2

    # compare ratios
    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio):
        return ow1, oh1
    else:
        return ow2, oh2


def save_decoded_latents_as_video(decoded_latents: list[torch.Tensor], output_path: str, fps: int):
    # Process outputs
    videos = rearrange(decoded_latents, "b c t h w -> t b c h w")
    frames = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=6)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        frames.append((x * 255).numpy().astype(np.uint8))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, format="mp4")
