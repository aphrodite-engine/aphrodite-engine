#pragma once

#include <ATen/Tensor.h>

void had_r_128
(
    const at::Tensor& input,
    const at::Tensor& output,
    const c10::optional<at::Tensor>& pre_scale,
    const c10::optional<at::Tensor>& post_scale,
    const float scale
);

void had_r_128_dual
(
    const at::Tensor& input1,
    const at::Tensor& output1,
    const c10::optional<at::Tensor>& pre_scale1,
    const c10::optional<at::Tensor>& post_scale1,
    const at::Tensor& input2,
    const at::Tensor& output2,
    const c10::optional<at::Tensor>& pre_scale2,
    const c10::optional<at::Tensor>& post_scale2,
    const float scale
);
