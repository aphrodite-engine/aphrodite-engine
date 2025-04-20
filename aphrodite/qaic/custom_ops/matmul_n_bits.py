# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math

import torch
from torch import nn


class QuantLinearTorchFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g,
        x,
        qself_qweight,
        qself_scales,
        qself_qzeros,
        g_idx,
        bits,
        group_size,
        in_features,
        out_features,
    ):
        input_tuple = (x, qself_qweight, qself_scales, qself_qzeros)
        input_tuple += (g_idx,) if g_idx is not None else ()
        return g.op(
            "com.microsoft::MatMulNBits",
            *input_tuple,
            outputs=1,
            K_i=in_features,
            N_i=out_features,
            bits_i=bits,
            block_size_i=group_size,
        )

    @staticmethod
    def forward(
        ctx,
        x,
        qself_qweight,
        qself_scales,
        qself_qzeros,
        g_idx,
        bits,
        group_size,
        in_features,
        out_features,
    ):
        if torch.onnx.is_in_onnx_export():
            return torch.zeros(
                x.shape[:-1] + (out_features,), dtype=x.dtype
            ).float()
        fp_weight = dequantize_blockwise_bits(
            qself_qweight,
            qself_scales,
            qself_qzeros,
            bits,
            group_size,
            g_idx,
            in_features,
            out_features,
        )[0].float()

        return torch.matmul(x.float(), fp_weight.T.float())


def dequantize_blockwise_bits(
    quant_values, scale, zero_point, bits, group_size, g_idx, rows, cols
):
    if bits != 4:
        raise ValueError(
            "Only bits=4 is supported for executing quantized model"
        )
    if group_size != 128:
        raise ValueError(
            "Only group_size=128 is supported for executing quantized model"
        )
    expand_quant_value = (
        quant_values.unsqueeze(-1)
        >> torch.tensor([[[[0, 4]]]], dtype=torch.int32)
    ) & 0x0F
    expand_quant_value = expand_quant_value.reshape(
        *quant_values.shape[:-1], -1
    )
    aligned_scale = scale.reshape(*quant_values.shape[:-1], 1)
    if zero_point.dtype == scale.dtype:
        expand_zero_point = zero_point.reshape(*quant_values.shape[:-1], -1)
    else:
        expand_zero_point = (
            zero_point.unsqueeze(-1)
            >> torch.tensor([[[[0, 4]]]], dtype=torch.int32)
        ) & 0x0F
        try:
            expand_zero_point = expand_zero_point.reshape(
                *quant_values.shape[:-1], -1
            )
        # FIXME: remove try-except
        except RuntimeError:
            expand_zero_point = expand_zero_point.reshape(
                quant_values.shape[0], -1, 1
            )
            expand_zero_point = expand_zero_point[:, : quant_values.shape[1]]
    if g_idx is not None and g_idx[:32].sum().item() != 0:
        float_values = (
            (
                expand_quant_value.reshape(expand_quant_value.shape[0], -1)
                - expand_zero_point[:, g_idx, 0]
            )
            * aligned_scale[:, g_idx, 0]
        ).to(scale.dtype)
    else:
        float_values = (
            (expand_quant_value - expand_zero_point) * aligned_scale
        ).to(scale.dtype)
    float_values = float_values.reshape(cols, -1)
    if rows != float_values.shape[-1]:
        float_values = float_values[:, :rows]
        expand_zero_point = expand_zero_point[:, :rows]
    if expand_zero_point.ndim == 3:
        expand_zero_point = expand_zero_point.squeeze(-1)
    if aligned_scale.ndim == 3:
        aligned_scale = aligned_scale.squeeze(-1)

    return float_values, expand_zero_point, aligned_scale


class QuantLinearORT(nn.Module):
    def __init__(self, bits, group_size, in_features, out_features, bias):
        super().__init__()
        if bits not in [2, 3, 4, 5, 6, 7, 8]:
            raise NotImplementedError("Only 2,4,5,6,7,8 bits are supported.")
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size if group_size != -1 else in_features
        self.act_order = None

        q_rows = in_features // self.group_size
        self.register_buffer(
            "qweight",
            torch.zeros(
                (out_features, q_rows, self.group_size // (8 // bits)),
                dtype=torch.uint8,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (q_rows + (q_rows & 1)) * (out_features // 8 * self.bits),
                dtype=torch.uint8,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (math.ceil(in_features / self.group_size) * out_features),
                dtype=torch.float16,
            ),
        )
        self.register_buffer(
            "g_idx",
            torch.tensor(
                [i // self.group_size for i in range(in_features)],
                dtype=torch.int32,
            ),
        )
        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=torch.float16)
            )
        else:
            self.bias = None

    def quant_weight(self, weight, scales, zeros, g_idx):
        scale_zeros = zeros * scales
        scale_mat = scales[g_idx]
        scale_zeros_mat = scale_zeros[g_idx]
        int_weight_T = torch.round(
            ((weight + scale_zeros_mat) / scale_mat).float()
        ).to(torch.int)
        return int_weight_T

    def pack_on_device(self, int_weight, int_zeros):
        if self.bits != 4:
            raise ValueError("only 4bit is supported by ONNXRUNTIME for now.")

        # Order of groups
        self.act_order = (
            self.g_idx[: self.group_size // self.bits].sum().item() != 0
        )

        intzeros_pt = (
            int_zeros.T
            if int_zeros.dtype == self.scales.dtype
            else int_zeros.T.byte()
        )
        scales_pt = self.scales.T.to(int_weight.device)
        intweight_pt = int_weight.byte()

        block_size = self.group_size
        rows, cols = intweight_pt.shape
        blob_size = block_size // 2
        k_blocks = (rows + block_size - 1) // block_size
        padded_rows = k_blocks * block_size
        pad_len = padded_rows - rows
        if pad_len > 0:
            intweight_pt = torch.nn.functional.pad(
                intweight_pt, (0, 0, 0, pad_len), "constant", 0
            )
        intzeros_pt = torch.nn.functional.pad(
            intzeros_pt, (0, intzeros_pt.shape[-1] & 1, 0, 0), "constant", 0
        )

        # Pack zeros if they are not float
        if int_zeros.dtype != self.scales.dtype:
            intzeros_pt = (intzeros_pt[:, 0::2]) | (intzeros_pt[:, 1::2] << 4)
            intzeros_pt = intzeros_pt.reshape(-1)

        # Pack weights
        intweight_pt_T = int_weight.T
        intweight_pt_T = (intweight_pt_T[:, 0::2]) | (
            intweight_pt_T[:, 1::2] << 4
        )
        intweight_pt_T = intweight_pt_T.reshape(cols, k_blocks, blob_size)

        scales_pt = scales_pt.reshape(-1)

        # Validation checks
        if (self.qweight.shape != intweight_pt_T.shape) and (
            self.qzeros.shape == intzeros_pt.shape
            or self.qzeros.dtype != intzeros_pt.dtype
        ):
            raise RuntimeError(
                "Something went wrong while packing the weights in "
                "QuantLinearORT module"
            )

        # Assign buffers
        self.scales = scales_pt.float()
        self.qweight = intweight_pt_T.byte()  # Convert to uint8
        if int_zeros.dtype != self.scales.dtype:
            self.qzeros = intzeros_pt.byte()  # Convert to uint8
        else:
            self.qzeros = intzeros_pt

    def pack(self, linear, scales, zeros, g_idx=None):
        layer_weight = linear.weight.data
        self.scales = scales.T
        self.g_idx = g_idx.clone()
        int_weight = self.quant_weight(layer_weight.T, scales.T, zeros.T, g_idx)
        return self.pack_on_device(int_weight, zeros.T)

    def forward(self, inputs):
        out = QuantLinearTorchFunction().apply(
            inputs,
            self.qweight,
            self.scales,
            self.qzeros,
            self.g_idx if self.act_order else None,
            self.bits,
            self.group_size,
            self.in_features,
            self.out_features,
        )
        out = out + self.bias if self.bias is not None else out
        return out
