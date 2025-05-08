// borrowed from https://github.com/pytorch-labs/applied-ai/tree/main/kernels/cuda/inference/hadamard_transform

#include <torch/all.h>
#include <pybind11/pybind11.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

using namespace torch::indexing;

template <torch::ScalarType dtype>
void run_fht(void* a, void* out, uint32_t numel, uint32_t had_size, cudaStream_t stream);

constexpr bool is_power_of_two(uint32_t x) {
    return x && !(x & (x - 1));
}

torch::Tensor hadamard_transform(at::Tensor& in, bool inplace) {
    auto dtype = in.scalar_type();
    TORCH_CHECK(dtype == torch::ScalarType::Half || dtype == torch::ScalarType::BFloat16, "Only fp16 and bf16 supported currently");
    TORCH_CHECK(in.is_cuda());
    
    const int had_size = in.size(-1);
    TORCH_CHECK(is_power_of_two(had_size) && (had_size <= (1U << 15)),
        "Only power of two Hadamard sizes up to 2^15 are supported, got ", had_size);
    
    const auto res_shape = in.sizes();
    torch::Tensor x = in.reshape({-1, had_size});
    
    auto numel = in.numel();
    if (numel % 256 != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, 0, 0, (256 - numel % 256) / had_size}));
    }
    
    if (x.stride(-1) != 1) {
        x = x.contiguous();
    }
    torch::Tensor out = inplace ? x : torch::empty_like(x);

    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype == torch::ScalarType::Half) {
        run_fht<torch::ScalarType::Half>(x.data_ptr(), out.data_ptr(), x.numel(), had_size, stream);
    } else {
        run_fht<torch::ScalarType::BFloat16>(x.data_ptr(), out.data_ptr(), x.numel(), had_size, stream);
    }

    if (numel % 256 != 0) {
        out = out.index({Slice(0, numel / had_size)});
    }

    if (inplace && out.data_ptr() != in.data_ptr()) {
        in.copy_(out.view(res_shape));
        return in;
    }
    return out.reshape(res_shape);
}