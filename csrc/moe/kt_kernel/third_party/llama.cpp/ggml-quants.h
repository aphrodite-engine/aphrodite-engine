#pragma once

#include "ggml.h"

#include <cstdint>

#define QK4_0 32
#define QK8_0 32
#define QK_K 256

struct block_q4_0 {
  ggml_fp16_t d;
  uint8_t qs[QK4_0 / 2];
};

struct block_q8_0 {
  ggml_fp16_t d;
  int8_t qs[QK8_0];
};

struct block_q4_K {
  ggml_fp16_t d;
  ggml_fp16_t dmin;
  uint8_t scales[12];
  uint8_t qs[QK_K / 2];
};

struct block_q8_K {
  float d;
  int8_t qs[QK_K];
  int16_t bsums[QK_K / 16];
};
