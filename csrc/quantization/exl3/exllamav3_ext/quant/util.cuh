#pragma once

#include "../util.h"
#include <tuple>

void count_inf_nan(at::Tensor x, at::Tensor y);

void exl3_make_gate_up_indices(at::Tensor out, at::Tensor indices,
                               int64_t offset);

void exl3_silu_mul(at::Tensor out, at::Tensor gate, at::Tensor up);
