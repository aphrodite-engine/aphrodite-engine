#pragma once

#include "../util.h"

void reconstruct(at::Tensor unpacked, at::Tensor packed, int K, bool mcg,
                 bool mul1);

void reconstruct_slice(at::Tensor unpacked, at::Tensor packed, int K, bool mcg,
                       bool mul1, int64_t n_offset);
