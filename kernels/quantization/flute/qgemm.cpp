#include <cuda_runtime.h>
#include <torch/library.h>
#include <torch/all.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/stl.h>
#include "cute/numeric/integral_constant.hpp"


torch::Tensor
hadamard_transform(at::Tensor& in,
                   bool inplace);


template <
  typename T,
  typename TQ,
  typename T2,
  typename NumBits,
  typename GroupSize
>
void
_qgemm_raw(int64_t M,
           int64_t N,
           int64_t K,
           int64_t P,
           const T * const __restrict__ A,
           const TQ* const __restrict__ Q,
                 T *       __restrict__ D,
           const T * const __restrict__ S,
           const T * const __restrict__ QM,
           const T2* const __restrict__ QM2,
               void*       __restrict__ workspace,
           const int64_t                    template_id,
           const int64_t                    num_sms,
           const cudaStream_t           stream);


template <
  typename T,
  typename NumBits,
  typename GroupSize
>
void
qgemm_raw(const at::Tensor&  input,
          const at::Tensor&  weight,
                at::Tensor&  output,
          const at::Tensor&  scales,
          const at::Tensor&  table,
          const at::Tensor&  table2,
                at::Tensor&  workspace,
          const int64_t          template_id,
          const int64_t          num_sms,
          const cudaStream_t stream)
{
    using namespace cute;
    using TQ = cute::uint16_t;
    using T2 = conditional_t<is_same_v<T, half_t>, __half2, __nv_bfloat162>;

    _qgemm_raw<
        T,
        TQ,
        T2,
        NumBits,
        GroupSize
    > (
        output.size(0),  // M
        output.size(1),  // N
        input .size(1),  // K
        weight.size(0),  // P
        reinterpret_cast<const T *>(input    .data_ptr()),
        reinterpret_cast<const TQ*>(weight   .data_ptr()),
        reinterpret_cast<      T *>(output   .data_ptr()),
        reinterpret_cast<const T *>(scales   .data_ptr()),
        reinterpret_cast<const T *>(table    .data_ptr()),
        reinterpret_cast<const T2*>(table2   .data_ptr()),
        reinterpret_cast<    void*>(workspace.data_ptr()),
        template_id,
        num_sms,
        stream);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


at::Tensor
qgemm_raw_simple(const at::Tensor&   input,
                 const at::Tensor&   weight,
                 const at::Tensor&   scales,
                 const at::Tensor&   table,
                 const at::Tensor&   table2,
                       at::Tensor&   workspace,
                 const cute::int64_t num_bits,
                 const cute::int64_t group_size,
                 const cute::int64_t template_id,
                 const cute::int64_t num_sms)
{

    // Set the device of this function, primarily used when
    // we have multiple devices in the same process.
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    // Get the current CUDA stream, primarily used
    // to make CUDA Graphs work.
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Squash the batch dimensions of the input tensor with its
    // next-to-last dimensions.
    const auto input_sizes = input.sizes().vec();
    const auto input_2d = input.reshape({-1, input_sizes.back()});
    auto output = at::empty(
        {
            input_2d.size(0),
            scales.size(0)
        },
        at::TensorOptions()
            .dtype(input_2d.dtype())
            .device(input_2d.device()));

#define RUN_QGEMM_RAW(T, NUM_BITS, GROUP_SIZE)  \
    do {                                        \
        qgemm_raw<                              \
            T,                                  \
            cute::Int<NUM_BITS>,                \
            cute::Int<GROUP_SIZE>               \
        > (                                     \
            input_2d,                           \
            weight,                             \
            output,                             \
            scales,                             \
            table,                              \
            table2,                             \
            workspace,                          \
            template_id,                        \
            num_sms,                            \
            stream);                            \
    } while (false)

#define RUN_QGEMM_RAW_SWITCH_GROUP_SIZE(T, NUM_BITS)  \
    do {                                              \
        switch (group_size)                           \
        {                                             \
        case 64:                                      \
            RUN_QGEMM_RAW(T, NUM_BITS, 64);           \
            break;                                    \
        case 128:                                     \
            RUN_QGEMM_RAW(T, NUM_BITS, 128);          \
            break;                                    \
        case 256:                                     \
            RUN_QGEMM_RAW(T, NUM_BITS, 256);          \
            break;                                    \
        default:                                      \
            AT_ERROR("Unsupported `group_size`");     \
        }                                             \
    } while (false)

#define RUN_QGEMM_RAW_SWITCH_NUM_BITS_AND_GROUP_SIZE(T)  \
    do {                                                 \
        switch (num_bits)                                \
        {                                                \
        case 2:                                          \
            RUN_QGEMM_RAW_SWITCH_GROUP_SIZE(T, 2);       \
            break;                                       \
        case 3:                                          \
            RUN_QGEMM_RAW_SWITCH_GROUP_SIZE(T, 3);       \
            break;                                       \
        case 4:                                          \
            RUN_QGEMM_RAW_SWITCH_GROUP_SIZE(T, 4);       \
            break;                                       \
        default:                                         \
            AT_ERROR("Unsupported `num_bits`");          \
        }                                                \
    } while (false)


    AT_DISPATCH_SWITCH(
        input.scalar_type(),
        "qgemm_raw_simple",
        AT_DISPATCH_CASE(
            at::ScalarType::Half,
            [&]() {
                RUN_QGEMM_RAW_SWITCH_NUM_BITS_AND_GROUP_SIZE(cute::half_t);
                return;
            }
        )
        AT_DISPATCH_CASE(
            at::ScalarType::BFloat16,
            [&]() {
                RUN_QGEMM_RAW_SWITCH_NUM_BITS_AND_GROUP_SIZE(cute::bfloat16_t);
                return;
            }
        )
    );

    auto output_sizes = input_sizes;
    output_sizes.back() = scales.size(0);
    return output.reshape(output_sizes);
}


at::Tensor
apply_hadamard(const at::Tensor&   input,
               const cute::int64_t hadamard_size)
{
    auto input_sizes = input.sizes();
    auto flat_input = input.reshape({-1, hadamard_size});
    auto had_input = hadamard_transform(
        flat_input, false
    );
    return had_input.reshape(input_sizes);
}


at::Tensor
qgemm_raw_simple_hadamard(const at::Tensor&   input,
                          const at::Tensor&   weight,
                          const at::Tensor&   scales,
                          const at::Tensor&   table,
                          const at::Tensor&   table2,
                                at::Tensor&   workspace,
                          const cute::int64_t num_bits,
                          const cute::int64_t group_size,
                          const cute::int64_t hadamard_size,
                          const cute::int64_t template_id,
                          const cute::int64_t num_sms)
{
    auto had_input = apply_hadamard(
        input,
        hadamard_size
    );

    return qgemm_raw_simple(
        had_input,
        weight,
        scales,
        table,
        table2,
        workspace,
        num_bits,
        group_size,
        template_id,
        num_sms
    );
}


// Registers _C as an extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(flute, m) {
    m.def("qgemm_raw_simple(Tensor input, Tensor weight, Tensor scales, Tensor table, Tensor table2, Tensor(a!) workspace, int num_bits, int group_size, int template_id, int num_sms) -> Tensor");
    m.def("qgemm_raw_simple_hadamard(Tensor input, Tensor weight, Tensor scales, Tensor table, Tensor table2, Tensor(a!) workspace, int num_bits, int group_size, int hadamard_size, int template_id, int num_sms) -> Tensor");
}


TORCH_LIBRARY_IMPL(flute, CUDA, m) {
    m.impl("qgemm_raw_simple", &qgemm_raw_simple);
    m.impl("qgemm_raw_simple_hadamard", &qgemm_raw_simple_hadamard);
}