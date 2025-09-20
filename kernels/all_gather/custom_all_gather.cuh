#pragma once

#include <cuda_runtime.h>
#include <torch/all.h>
#include <set>
#include <vector>
#include <memory>

#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif

namespace aphrodite {

// Forward declaration of the AllgatherOp class
class AllgatherOp;

// Define fptr_t for passing C++ object pointers to Python
using fptr_t = int64_t;

// C++ functions to be exposed to Python
fptr_t init_custom_ag(const std::vector<int64_t>& group_ranks, int64_t nccl_comm_ptr);
void all_gather(fptr_t _ag_op, torch::Tensor& input, torch::Tensor& output,
                const std::optional<std::vector<int64_t>>& sizes);
std::vector<torch::Tensor> all_gather_list(
    fptr_t _ag_op, const std::vector<torch::Tensor>& input_list,
    const std::optional<std::vector<int64_t>>& sizes);
void dispose_custom_ag(fptr_t _ag_op);

} // namespace aphrodite
