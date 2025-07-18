###############################################################################
# Copyright (C) 2024-2025 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
import torch
import habana_frameworks.torch.utils.experimental as htexp

from aphrodite.platforms import current_platform


def is_hpu_gaudi2():
    return current_platform.is_hpu() and htexp._get_device_type(
    ) == htexp.synDeviceType.synDeviceGaudi2


def get_hpu_gaudi2_scale_factor():
    return (torch.finfo(torch.float8_e4m3fn).max /
            torch.finfo(torch.float8_e4m3fnuz).max)


EXP_WIDTH = {
    torch.float32: 8,
    torch.bfloat16: 8,
    torch.float8_e4m3fn: 4,
    torch.float8_e5m2: 5,
}


def get_default_exp_bias(dtype):
    exp_width = EXP_WIDTH[dtype]
    return 2 ** (exp_width - 1) - 1


EXP_BIAS_SETS = {
    ("GAUDI2", torch.float8_e4m3fn): [3, 7, 11, 15],
    ("GAUDI2", torch.float8_e5m2): [15],
    ("GAUDI3", torch.float8_e4m3fn): range(0, 63),
    ("GAUDI3", torch.float8_e5m2): range(0, 63),
}


MAX_RANGE = {
    torch.float32: torch.finfo(torch.float32).max,
    torch.bfloat16: torch.finfo(torch.bfloat16).max,
    torch.float8_e4m3fn: torch.finfo(torch.float8_e4m3fn).max,
    # float8_e4m3fn data type is 8-bit floating point consist of Exponent: 4, Mantissa: 3, bias: 7. It's supported by Gaudi3.
    torch.float8_e5m2: torch.finfo(torch.float8_e5m2).max
    # float8_e5m2 data type is 8-bit floating point consist of Exponent: 5, Mantissa: 2, bias: 15. IEEE 754, with NaN and inf.
}


try:
    MAX_RANGE[torch.float8_e4m3fnuz] = torch.finfo(torch.float8_e4m3fnuz).max
    # float8_e4m3fnuz data type is 8-bit floating point consist of Exponent: 4, Mantissa: 3, bias: 8 with 1 sign bit. It's supported by Gaudi2.
except AttributeError as e:
    pass


def get_fullscale(dtype, device, exp_bias=None):
    default_exp_bias = get_default_exp_bias(dtype)
    fullscale = 1
    if device == "GAUDI2" and dtype == torch.float8_e4m3fn:
        try:
            fullscale = MAX_RANGE[torch.float8_e4m3fnuz]
        except AttributeError as e:
            pass
    else:
        fullscale = MAX_RANGE[dtype]
    exp_bias = default_exp_bias if exp_bias is None else exp_bias
    fullscale = fullscale * (2 ** (default_exp_bias - exp_bias))
    return float(fullscale)


def get_fullscales_by_expbias_set(dtype, device, expbias_set):
    return [get_fullscale(dtype, device, exp_bias=eb) for eb in expbias_set]


def get_fp8_hw_alligned_scales(dtype, device):
    exp_bias_set = EXP_BIAS_SETS.get((device, dtype), None)
    return (
        None
        if exp_bias_set is None
        else [x / get_fullscale(dtype, device) for x in get_fullscales_by_expbias_set(dtype, device, exp_bias_set)]
    )


DEVICES_SCALE_FACTORS = {
    "GAUDI2": 4,
    "GAUDI3": 1,
}


FP8_143_SCALES = {
    device: get_fp8_hw_alligned_scales(torch.float8_e4m3fn, device) for device in DEVICES_SCALE_FACTORS.keys()
}


FP8_143_SCALES_TRAITS = {
    device: (
        min(FP8_143_SCALES[device]),
        max(FP8_143_SCALES[device]),
        DEVICES_SCALE_FACTORS[device],
    )
    for device in DEVICES_SCALE_FACTORS.keys()
}


class ScaleToPow2:
    def calc(self, scale):
        scale_pow2 = 2.0 ** torch.ceil(torch.log2(scale))
        return scale_pow2


class ConvertScaleToHwAligned:
    def __init__(self, device_type="GAUDI3"):
        self.device_type = "GAUDI2" if is_hpu_gaudi2() else "GAUDI3"
    def calc(self, scale):
        if self.device_type == "GAUDI2":
            scale = scale * get_hpu_gaudi2_scale_factor()
        scale_pow2 = ScaleToPow2().calc(scale)
        min_scale, max_scale, scale_factor = FP8_143_SCALES_TRAITS[self.device_type]
        scale_pow2_hw = torch.minimum(
            torch.maximum(
                2.0 ** (torch.ceil(torch.log2(scale_pow2) / scale_factor) * scale_factor),
                torch.tensor(min_scale, dtype=scale.dtype, device=scale.device),
                ),
            torch.tensor(max_scale, dtype=scale.dtype, device=scale.device),
        )
        return scale_pow2_hw