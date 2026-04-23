#include <Python.h>
#include "graph.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
//#include <torch/extension.h>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_devctx.cuh"

Graph::Graph()
{
    ready = false;
    graph = NULL;
    graph_exec = NULL;
    need_cublas = false;
}

Graph::~Graph()
{
    if (graph) cudaGraphDestroy(graph);
    if (graph_exec) cudaGraphExecDestroy(graph_exec);
}

cudaStream_t Graph::capture_begin()
{
    // Make sure nothing is pending
    cudaDeviceSynchronize();

    // Create capture stream
    cuda_check(cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking));

    // Begin capture
    cuda_check(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal));
    return capture_stream;
}

void Graph::capture_end()
{
    // End capture
    cuda_check(cudaStreamEndCapture(capture_stream, &graph));
    cuda_check(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    //inspect_graph();

    // Get graph nodes
    size_t num_nodes;
    cudaGraphGetNodes(graph, nullptr, &num_nodes);
    nodes.resize(num_nodes);
    cudaGraphGetNodes(graph, nodes.data(), &num_nodes);

    // Store copies of all node param structures
    node_params.resize(num_nodes);
    node_needs_update.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
        node_needs_update[i] = false;

    int n = 0;
    int c = 0;
    while (true)
    {
        cudaGraphNodeType t{};
        cudaGraphNodeGetType(nodes[n], &t);

        // Node type: kernel
        if (t == cudaGraphNodeTypeKernel)
        {
            cudaGraphKernelNodeGetParams(nodes[n], &node_params[n]);

            for(; c < graph_sites.size(); c++)
            {
                void* func = std::get<0>(graph_sites[c]);
                if (func != node_params[n].func) break;

                int param_id     = std::get<1>(graph_sites[c]);
                int param_offset = std::get<2>(graph_sites[c]);

                graph_node_sites.push_back(std::tuple<int, int, int>(n, param_id, param_offset));
                if (param_id == GP_end) { c++; break; }
            }
        }

        n++;
        if (c == graph_sites.size()) break;
        if (n == num_nodes) TORCH_CHECK(false, "Graph recording failed");
    };

    // Destroy capture stream
    cuda_check(cudaStreamDestroy(capture_stream));

    // Graph is ready
    ready = true;
}

void Graph::record_param(void* kernel, int param_id, int param_offset)
{
    graph_sites.push_back(std::tuple<void*, int, int>(kernel, param_id, param_offset));
}

void Graph::launch(std::vector<PPTR> params, cudaStream_t stream)
{
    if (need_cublas)
    {
        cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(cublas_handle, stream);
        cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
        int device;
        cudaGetDevice(&device);
        void* ws = DevCtx::instance().get_ws(device);
        cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);
    }

    int p = 0;
    int n = 0;
    while (true)
    {
        if (std::get<1>(graph_node_sites[n]) == std::get<0>(params[p]))
        {
            if (std::get<0>(params[p]) != GP_end)
            {
                void* new_value  = std::get<1>(params[p]);
                int node_idx     = std::get<0>(graph_node_sites[n]);
                int param_offset = std::get<2>(graph_node_sites[n]);

                void** p_old_value = (void**) node_params[node_idx].kernelParams[param_offset];
                if (*p_old_value != new_value)
                {
                    *p_old_value = new_value;
                    node_needs_update[node_idx] = true;
                }
            }
            p++;
        }

        n++;
        if (p == params.size()) break;
        if (n == graph_node_sites.size()) TORCH_CHECK(false, "Graph update failed");
    }

    for (int n = 0; n < nodes.size(); ++n)
    {
        if (!node_needs_update[n]) continue;
        cudaGraphExecKernelNodeSetParams(graph_exec, nodes[n], &node_params[n]);
        node_needs_update[n] = false;
    }

    cudaGraphLaunch(graph_exec, stream);
}

void Graph::inspect_graph()
{
    // Get the number of nodes in the graph
    size_t numNodes;
    cudaGraphGetNodes(graph, nullptr, &numNodes);

    // Get the nodes in the graph
    std::vector<cudaGraphNode_t> nodes(numNodes);
    cudaGraphGetNodes(graph, nodes.data(), &numNodes);
    DBGI(nodes.size());

    // Inspect each node
    for (size_t i = 0; i < numNodes; ++i)
    {
        cudaGraphNodeType nodeType;
        cudaGraphNodeGetType(nodes[i], &nodeType);

        if (nodeType == cudaGraphNodeTypeKernel)
        {
            cudaKernelNodeParams nodeParams;
            cudaGraphKernelNodeGetParams(nodes[i], &nodeParams);
            std::cout << "Kernel node " << i << ":" << std::endl;
            std::cout << "  Function pointer: " << nodeParams.func << std::endl;
            std::cout << "  Grid dimensions: (" << nodeParams.gridDim.x << ", " << nodeParams.gridDim.y << ", " << nodeParams.gridDim.z << ")" << std::endl;
            std::cout << "  Block dimensions: (" << nodeParams.blockDim.x << ", " << nodeParams.blockDim.y << ", " << nodeParams.blockDim.z << ")" << std::endl;
            std::cout << "  Shared memory: " << nodeParams.sharedMemBytes << " bytes" << std::endl;

        } else {
            std::cout << "Node " << i << " is not a kernel node." << std::endl;
        }
    }
}

