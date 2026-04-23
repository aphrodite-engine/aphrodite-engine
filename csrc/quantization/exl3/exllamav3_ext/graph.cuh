#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

using PPTR = std::tuple<int, void*>;

enum GraphedParams
{
    GP_end,

    GP_gemm_A,
    GP_gemm_B_trellis,
    GP_gemm_C,
    GP_gemm_B_suh,
    GP_gemm_A_had,
    GP_gemm_B_svh,

    GP_mgemm,
    GP_mgemm_A,
    GP_mgemm_C,
    GP_mgemm_indices,
    GP_mgemm_weights,

    GP_silu_mul_x,
    GP_silu_mul_y,
    GP_silu_mul_z,

    GP_gelu_mul_x,
    GP_gelu_mul_y,
    GP_gelu_mul_z,

    GP_relu2_mul_x,
    GP_relu2_mul_y,
    GP_relu2_mul_z,

    GP_xielu_x,
    GP_xielu_y,

    GP_add_sigmoid_gate,
    GP_add_sigmoid_gate_x,
    GP_add_sigmoid_gate_y,
    GP_add_sigmoid_gate_z,

    GP_add_sigmoid_gate_proj,
    GP_add_sigmoid_gate_proj_x,
    GP_add_sigmoid_gate_proj_y,
    GP_add_sigmoid_gate_proj_z,

    GP_add_x,
    GP_add_y,
    GP_add_z
};

class Graph
{
public:
    cudaStream_t capture_stream;
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    std::vector<std::tuple<void*, int, int>> graph_sites;
    std::vector<std::tuple<int, int, int>> graph_node_sites;

    std::vector<cudaGraphNode_t> nodes;
    std::vector<cudaKernelNodeParams> node_params;
    std::vector<void*> current_values;
    std::vector<bool> node_needs_update;

    bool need_cublas;
    bool ready;

    Graph();
    ~Graph();

    cudaStream_t capture_begin();
    void capture_end();

    void record_param(void* kernel, int param_id, int param_offset);
    void launch(std::vector<PPTR> params, cudaStream_t stream);

    void inspect_graph();
};
