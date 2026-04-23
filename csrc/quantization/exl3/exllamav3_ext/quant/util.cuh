#pragma once

#include <ATen/Tensor.h>
#include <tuple>

void count_inf_nan
(
    at::Tensor x,
    at::Tensor y
);