#pragma once

#include <ATen/Tensor.h>
#include "graph.cuh"

void hgemm_gr
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    Graph* graph
);

void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c
);