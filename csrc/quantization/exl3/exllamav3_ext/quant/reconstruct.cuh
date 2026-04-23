#pragma once

#include <ATen/Tensor.h>

void reconstruct
(
    at::Tensor unpacked,
    at::Tensor packed,
    int K,
    bool mcg,
    bool mul1
);
