#include "kernels.h"
#include <torch/all.h>

#if defined(__aarch64__) || defined(__ARM_NEON)

namespace {

template <typename T>
T* get_data_ptr(torch::Tensor& tensor) {
  return tensor.data_ptr<T>();
}

template <typename T>
const T* get_data_ptr(const torch::Tensor& tensor) {
  return tensor.data_ptr<T>();
}

}  // namespace

// Quantization wrappers
void mobile_int8_to_fp32(const torch::Tensor& src, torch::Tensor& dst,
                         double scale) {
  TORCH_CHECK(src.scalar_type() == torch::kInt8, "Input must be int8");
  TORCH_CHECK(dst.scalar_type() == torch::kFloat32, "Output must be float32");
  TORCH_CHECK(src.numel() == dst.numel(),
              "Input and output must have same size");

  aphrodite::mobile::int8_to_fp32(get_data_ptr<int8_t>(src),
                                  get_data_ptr<float>(dst), src.numel(),
                                  static_cast<float>(scale));
}

void mobile_fp32_to_int8(const torch::Tensor& src, torch::Tensor& dst,
                         double scale) {
  TORCH_CHECK(src.scalar_type() == torch::kFloat32, "Input must be float32");
  TORCH_CHECK(dst.scalar_type() == torch::kInt8, "Output must be int8");
  TORCH_CHECK(src.numel() == dst.numel(),
              "Input and output must have same size");

  aphrodite::mobile::fp32_to_int8(get_data_ptr<float>(src),
                                  get_data_ptr<int8_t>(dst), src.numel(),
                                  static_cast<float>(scale));
}

void mobile_dynamic_quantize_fp32_to_int8(const torch::Tensor& src,
                                          torch::Tensor& dst,
                                          torch::Tensor& computed_scale) {
  TORCH_CHECK(src.scalar_type() == torch::kFloat32, "Input must be float32");
  TORCH_CHECK(dst.scalar_type() == torch::kInt8, "Output must be int8");
  TORCH_CHECK(computed_scale.scalar_type() == torch::kFloat32,
              "Scale must be float32");
  TORCH_CHECK(src.numel() == dst.numel(),
              "Input and output must have same size");

  float scale;
  aphrodite::mobile::dynamic_quantize_fp32_to_int8(
      get_data_ptr<float>(src), get_data_ptr<int8_t>(dst), src.numel(), &scale);
  *get_data_ptr<float>(computed_scale) = scale;
}

// Matrix multiplication wrappers
void mobile_matmul_int8(const torch::Tensor& a,
                        const torch::Tensor& b_transposed, torch::Tensor& c,
                        double a_scale, double b_scale, double c_scale) {
  TORCH_CHECK(a.scalar_type() == torch::kInt8, "Matrix A must be int8");
  TORCH_CHECK(b_transposed.scalar_type() == torch::kInt8,
              "Matrix B must be int8");
  TORCH_CHECK(c.scalar_type() == torch::kInt8, "Matrix C must be int8");

  int64_t M = a.size(0);
  int64_t K = a.size(1);
  int64_t N = b_transposed.size(0);

  TORCH_CHECK(b_transposed.size(1) == K, "Matrix dimensions must match");
  TORCH_CHECK(c.size(0) == M && c.size(1) == N,
              "Output matrix dimensions must match");

  aphrodite::mobile::matmul_int8(
      get_data_ptr<int8_t>(a), get_data_ptr<int8_t>(b_transposed),
      get_data_ptr<int8_t>(c), M, K, N, static_cast<float>(a_scale),
      static_cast<float>(b_scale), static_cast<float>(c_scale));
}

void mobile_matmul_f16(const torch::Tensor& a,
                       const torch::Tensor& b_transposed, torch::Tensor& c) {
  TORCH_CHECK(a.scalar_type() == torch::kFloat16, "Matrix A must be float16");
  TORCH_CHECK(b_transposed.scalar_type() == torch::kFloat16,
              "Matrix B must be float16");
  TORCH_CHECK(c.scalar_type() == torch::kFloat16, "Matrix C must be float16");

  int64_t M = a.size(0);
  int64_t K = a.size(1);
  int64_t N = b_transposed.size(0);

  TORCH_CHECK(b_transposed.size(1) == K, "Matrix dimensions must match");
  TORCH_CHECK(c.size(0) == M && c.size(1) == N,
              "Output matrix dimensions must match");

  aphrodite::mobile::matmul_f16(get_data_ptr<__fp16>(a),
                                get_data_ptr<__fp16>(b_transposed),
                                get_data_ptr<__fp16>(c), M, K, N);
}

void mobile_matmul_f32(const torch::Tensor& a,
                       const torch::Tensor& b_transposed, torch::Tensor& c) {
  TORCH_CHECK(a.scalar_type() == torch::kFloat32, "Matrix A must be float32");
  TORCH_CHECK(b_transposed.scalar_type() == torch::kFloat32,
              "Matrix B must be float32");
  TORCH_CHECK(c.scalar_type() == torch::kFloat32, "Matrix C must be float32");

  int64_t M = a.size(0);
  int64_t K = a.size(1);
  int64_t N = b_transposed.size(0);

  TORCH_CHECK(b_transposed.size(1) == K, "Matrix dimensions must match");
  TORCH_CHECK(c.size(0) == M && c.size(1) == N,
              "Output matrix dimensions must match");

  aphrodite::mobile::matmul_f32(get_data_ptr<float>(a),
                                get_data_ptr<float>(b_transposed),
                                get_data_ptr<float>(c), M, K, N);
}

// Activation wrappers
void mobile_silu_f32(const torch::Tensor& input, torch::Tensor& output) {
  TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32,
              "Output must be float32");
  TORCH_CHECK(input.numel() == output.numel(),
              "Input and output must have same size");

  aphrodite::mobile::silu_f32(get_data_ptr<float>(input),
                              get_data_ptr<float>(output), input.numel());
}

void mobile_gelu_f32(const torch::Tensor& input, torch::Tensor& output) {
  TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32,
              "Output must be float32");
  TORCH_CHECK(input.numel() == output.numel(),
              "Input and output must have same size");

  aphrodite::mobile::gelu_f32(get_data_ptr<float>(input),
                              get_data_ptr<float>(output), input.numel());
}

// RMS Norm wrappers
void mobile_rms_norm_f32(const torch::Tensor& input,
                         const torch::Tensor& weight, torch::Tensor& output,
                         double eps) {
  TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32,
              "Weight must be float32");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32,
              "Output must be float32");

  int64_t batch_size = input.size(0);
  int64_t dims = input.size(1);

  TORCH_CHECK(weight.size(0) == dims, "Weight size must match input dimension");
  TORCH_CHECK(output.size(0) == batch_size && output.size(1) == dims,
              "Output shape must match input");

  aphrodite::mobile::rms_norm_f32(
      get_data_ptr<float>(input), get_data_ptr<float>(weight),
      get_data_ptr<float>(output), batch_size, dims, static_cast<float>(eps));
}

// Reduction wrappers
int64_t mobile_sum_all_int8(const torch::Tensor& data) {
  TORCH_CHECK(data.scalar_type() == torch::kInt8, "Input must be int8");
  return aphrodite::mobile::sum_all_int8(get_data_ptr<int8_t>(data),
                                         data.numel());
}

double mobile_mean_all_int8(const torch::Tensor& data) {
  TORCH_CHECK(data.scalar_type() == torch::kInt8, "Input must be int8");
  return aphrodite::mobile::mean_all_int8(get_data_ptr<int8_t>(data),
                                          data.numel());
}

// Scalar operation wrappers
void mobile_scalar_op_int8(const torch::Tensor& input, torch::Tensor& output,
                           double scalar_value, int64_t op_type) {
  TORCH_CHECK(input.scalar_type() == torch::kInt8, "Input must be int8");
  TORCH_CHECK(output.scalar_type() == torch::kInt8, "Output must be int8");
  TORCH_CHECK(input.numel() == output.numel(),
              "Input and output must have same size");
  TORCH_CHECK(op_type >= 0 && op_type <= 7, "Invalid operation type");

  aphrodite::mobile::ScalarOpType op =
      static_cast<aphrodite::mobile::ScalarOpType>(op_type);
  aphrodite::mobile::scalar_op_int8(get_data_ptr<int8_t>(input),
                                    get_data_ptr<int8_t>(output), input.numel(),
                                    static_cast<float>(scalar_value), op);
}

void mobile_scalar_op_f32(const torch::Tensor& input, torch::Tensor& output,
                          double scalar_value, int64_t op_type) {
  TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32,
              "Output must be float32");
  TORCH_CHECK(input.numel() == output.numel(),
              "Input and output must have same size");
  TORCH_CHECK(op_type >= 0 && op_type <= 7, "Invalid operation type");

  aphrodite::mobile::ScalarOpType op =
      static_cast<aphrodite::mobile::ScalarOpType>(op_type);
  aphrodite::mobile::scalar_op_f32(get_data_ptr<float>(input),
                                   get_data_ptr<float>(output), input.numel(),
                                   static_cast<float>(scalar_value), op);
}

#endif  // defined(__aarch64__) || defined(__ARM_NEON)
