#include "custom_all_gather.cuh"

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <numeric>

#if ENABLE_MULTI_DEVICE
namespace aphrodite {
namespace runtime {
namespace TorchUtils {
enum class DataType : int32_t {
    kFP32,
    kFP16,
    kBF16,
    kINT32,
    kINT8,
    kUINT8,
    kINT64,
    kBOOL,
    kCOUNT,
};

inline DataType dataType(torch::ScalarType type) {
    switch (type) {
        case torch::ScalarType::Float: return DataType::kFP32;
        case torch::ScalarType::Half: return DataType::kFP16;
        case torch::ScalarType::BFloat16: return DataType::kBF16;
        case torch::ScalarType::Int: return DataType::kINT32;
        case torch::ScalarType::Char: return DataType::kINT8;
        case torch::ScalarType::Byte: return DataType::kUINT8;
        case torch::ScalarType::Long: return DataType::kINT64;
        case torch::ScalarType::Bool: return DataType::kBOOL;
        default: TORCH_CHECK(false, "Unsupported data type");
    }
}
} // namespace TorchUtils
} // namespace runtime
} // namespace aphrodite

#ifndef TLLM_LOG_TRACE
#define TLLM_LOG_TRACE(...) \
    do {                    \
    } while (0)
#endif

#ifndef COMM_SESSION
struct DummyCommSession {
    int getRank() const { return 0; }
} COMM_SESSION;
#endif

namespace aphrodite {

namespace {

const std::unordered_map<aphrodite::runtime::TorchUtils::DataType, ncclDataType_t>* getDtypeMap() {
    static const std::unordered_map<aphrodite::runtime::TorchUtils::DataType, ncclDataType_t> dtypeMap = {
        {aphrodite::runtime::TorchUtils::DataType::kFP32, ncclFloat},
        {aphrodite::runtime::TorchUtils::DataType::kFP16, ncclHalf},
        {aphrodite::runtime::TorchUtils::DataType::kBF16, ncclBfloat16},
        {aphrodite::runtime::TorchUtils::DataType::kINT32, ncclInt32},
        {aphrodite::runtime::TorchUtils::DataType::kINT8, ncclInt8},
        {aphrodite::runtime::TorchUtils::DataType::kUINT8, ncclUint8},
        {aphrodite::runtime::TorchUtils::DataType::kINT64, ncclInt64},
        {aphrodite::runtime::TorchUtils::DataType::kBOOL, ncclUint8}, // NCCL doesn't have bool, use uint8
    };
    return &dtypeMap;
}

class AllgatherOp {
public:
    AllgatherOp(std::set<int> group, ncclComm_t comm)
        : mGroup(std::move(group)), mNcclComm(comm)
    {
    }

    ~AllgatherOp() = default;

    void initialize() {
        TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        // NCCL communicator is now passed in the constructor
        TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
    }

    std::vector<torch::Tensor> run_list(torch::TensorList input_list, const std::optional<std::vector<int64_t>>& sizes) {
        TORCH_CHECK(mNcclComm != nullptr, "NCCL communicator not initialized.");

        bool use_nccl_allgather = !sizes.has_value() || 
            std::all_of(sizes.value().begin(), sizes.value().end(),
                [&sizes](int64_t size) { return size == sizes.value()[0]; });
        
        int64_t sum_sizes = sizes.has_value() ? std::accumulate(sizes.value().begin(), sizes.value().end(), 0, std::plus<>{}) : 0;
        
        std::vector<torch::Tensor> output_list;
        output_list.reserve(input_list.size());
        
        // NCCLCHECK_THROW(ncclGroupStart()); // Group operations might be managed by Aphrodite's distributed backend
        for (auto const& input : input_list) {
            TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device.");
            auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
            auto type = aphrodite::runtime::TorchUtils::dataType(input.scalar_type());

            std::vector<int64_t> outputShape = input.sizes().vec();
            if (sizes.has_value()) {
                outputShape[0] = sum_sizes;
            } else {
                outputShape[0] *= mGroup.size();
            }
            auto output = torch::empty(outputShape, input.options());

            if (use_nccl_allgather) {
                AT_CUDA_CHECK(ncclAllGather(input.data_ptr(), output.mutable_data_ptr(), input.numel(), (*getDtypeMap())[type],
                    mNcclComm, stream));
            } else {
                size_t numel_base = std::accumulate(outputShape.cbegin() + 1, outputShape.cend(), 1, std::multiplies<>{});
                int64_t split_offset = 0;

                for (int root_idx = 0; root_idx < static_cast<int>(mGroup.size()); ++root_idx) {
                    auto it = mGroup.begin();
                    std::advance(it, root_idx);
                    int root_rank = *it;

                    auto split_size = sizes.value()[root_idx];
                    AT_CUDA_CHECK(ncclBroadcast(input.data_ptr(),
                        output.index({torch::indexing::Slice(split_offset, torch::indexing::None)}).mutable_data_ptr(),
                        numel_base * split_size, (*getDtypeMap())[type], root_rank, mNcclComm, stream));
                    split_offset += split_size;
                }
            }
            output_list.push_back(output);
        }
        // NCCLCHECK_THROW(ncclGroupEnd()); // Group operations might be managed by Aphrodite's distributed backend
        return output_list;
    }

    torch::Tensor run(torch::Tensor input, const std::optional<std::vector<int64_t>>& sizes) {
        return run_list({input}, sizes)[0];
    }

private:
    std::set<int> mGroup;
    ncclComm_t mNcclComm; // Stored NCCL communicator
};

} // namespace

// C++ functions to be exposed to Python
fptr_t init_custom_ag(const std::vector<int64_t>& group_ranks, int64_t nccl_comm_ptr) {
    std::set<int> group_set(group_ranks.begin(), group_ranks.end());
    ncclComm_t comm = reinterpret_cast<ncclComm_t>(nccl_comm_ptr);
    AllgatherOp* op = new AllgatherOp(group_set, comm);
    op->initialize();
    return reinterpret_cast<fptr_t>(op);
}

void all_gather(fptr_t _ag_op, torch::Tensor& input, torch::Tensor& output,
                const std::optional<std::vector<int64_t>>& sizes) {
    AllgatherOp* op = reinterpret_cast<AllgatherOp*>(_ag_op);
    torch::Tensor result = op->run(input, sizes);
    output.copy_(result); // Copy result to the provided output tensor
}

std::vector<torch::Tensor> all_gather_list(
    fptr_t _ag_op, const std::vector<torch::Tensor>& input_list,
    const std::optional<std::vector<int64_t>>& sizes) {
    AllgatherOp* op = reinterpret_cast<AllgatherOp*>(_ag_op);
    return op->run_list(input_list, sizes);
}

void dispose_custom_ag(fptr_t _ag_op) {
    AllgatherOp* op = reinterpret_cast<AllgatherOp*>(_ag_op);
    delete op;
}

} // namespace aphrodite

#else // ENABLE_MULTI_DEVICE

// Dummy implementations for when multi-device is not enabled
namespace aphrodite {

fptr_t init_custom_ag(const std::vector<int64_t>& group_ranks, int64_t nccl_comm_ptr) {
    TORCH_CHECK(false, "Multi-device support not enabled.");
    return 0;
}

void all_gather(fptr_t _ag_op, torch::Tensor& input, torch::Tensor& output,
                const std::optional<std::vector<int64_t>>& sizes) {
    TORCH_CHECK(false, "Multi-device support not enabled.");
}

std::vector<torch::Tensor> all_gather_list(
    fptr_t _ag_op, const std::vector<torch::Tensor>& input_list,
    const std::optional<std::vector<int64_t>>& sizes) {
    TORCH_CHECK(false, "Multi-device support not enabled.");
    return {};
}

void dispose_custom_ag(fptr_t _ag_op) {
    TORCH_CHECK(false, "Multi-device support not enabled.");
}

} // namespace aphrodite

#endif // ENABLE_MULTI_DEVICE