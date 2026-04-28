#include <ATen/ATen.h>
#include <torch/library.h>

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define KTRANSFORMERS_CPU_ONLY 1
#if defined(__x86_64__)
#define USE_AMX_AVX_KERNEL 1
#endif

#include "cpu_backend/cpuinfer.h"
#include "operators/amx/bf16-moe.hpp"
#include "operators/amx/fp8-moe.hpp"
#include "operators/amx/fp8-perchannel-moe.hpp"
#include "operators/amx/k2-moe.hpp"
#include "operators/amx/moe.hpp"
#include "operators/avx2/bf16-moe.hpp"
#include "operators/avx2/fp8-moe.hpp"
#include "operators/avx2/gptq_int4-moe.hpp"
#include "operators/avx512/bf16-moe.hpp"

namespace {

template <typename T>
T* tensor_ptr(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.device().is_cpu(), "kt_kernel tensors must be CPU tensors");
  TORCH_CHECK(tensor.is_contiguous(), "kt_kernel tensors must be contiguous");
  return tensor.data_ptr<T>();
}

std::vector<std::vector<void*>> ptr_matrix_from_tensor(
    const at::Tensor& ptrs) {
  TORCH_CHECK(ptrs.device().is_cpu(), "pointer tensor must be on CPU");
  TORCH_CHECK(ptrs.scalar_type() == at::kLong,
              "pointer tensor must use int64 dtype");
  TORCH_CHECK(ptrs.dim() == 2, "pointer tensor must have shape [N, M]");
  auto contiguous = ptrs.contiguous();
  const int64_t rows = contiguous.size(0);
  const int64_t cols = contiguous.size(1);
  const int64_t* data = contiguous.data_ptr<int64_t>();

  std::vector<std::vector<void*>> out(rows);
  for (int64_t i = 0; i < rows; ++i) {
    out[i].reserve(cols);
    for (int64_t j = 0; j < cols; ++j) {
      out[i].push_back(reinterpret_cast<void*>(data[i * cols + j]));
    }
  }
  return out;
}

std::vector<int> int_vec_from_tensor(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.device().is_cpu(), "config tensors must be on CPU");
  TORCH_CHECK(tensor.scalar_type() == at::kLong,
              "config tensors must use int64 dtype");
  auto contiguous = tensor.contiguous();
  std::vector<int> out(contiguous.numel());
  const int64_t* data = contiguous.data_ptr<int64_t>();
  for (int64_t i = 0; i < contiguous.numel(); ++i) {
    out[i] = static_cast<int>(data[i]);
  }
  return out;
}

class KtCpuInferHandle {
 public:
  explicit KtCpuInferHandle(const at::Tensor& numa_nodes,
                            const at::Tensor& thread_counts) {
    WorkerPoolConfig config;
    config.subpool_numa_map = int_vec_from_tensor(numa_nodes);
    config.subpool_thread_count = int_vec_from_tensor(thread_counts);
    TORCH_CHECK(config.subpool_numa_map.size() ==
                    config.subpool_thread_count.size(),
                "numa_nodes and thread_counts must have the same length");
    TORCH_CHECK(!config.subpool_numa_map.empty(),
                "at least one CPU worker subpool is required");
    config.subpool_count = static_cast<int>(config.subpool_numa_map.size());
    cpuinfer = std::make_unique<CPUInfer>(config);
  }

  std::unique_ptr<CPUInfer> cpuinfer;
};

class KtMoeHandle {
 public:
  std::shared_ptr<MoE_Interface> moe;
  std::function<void(const at::Tensor&)> load_weights;
  std::future<void> forward_future;
  at::Tensor gpu_experts_mask;
  at::Tensor physical_map;
};

std::mutex registry_mutex;
std::atomic<int64_t> next_handle{1};
std::unordered_map<int64_t, std::shared_ptr<KtCpuInferHandle>> cpu_handles;
std::unordered_map<int64_t, std::shared_ptr<KtMoeHandle>> moe_handles;

template <typename MoeTP>
std::shared_ptr<KtMoeHandle> make_moe_handle(const GeneralMOEConfig& config) {
  using MoeClass = TP_MOE<MoeTP>;
  auto typed_moe = std::make_shared<MoeClass>(config);
  auto handle = std::make_shared<KtMoeHandle>();
  handle->moe = typed_moe;
  handle->load_weights = [typed_moe](const at::Tensor& physical_map) {
    typed_moe->config.physical_to_logical_map =
        const_cast<int64_t*>(tensor_ptr<int64_t>(physical_map));
    typed_moe->load_weights();
  };
  return handle;
}

int64_t insert_cpu_handle(std::shared_ptr<KtCpuInferHandle> handle) {
  const int64_t id = next_handle.fetch_add(1);
  std::lock_guard<std::mutex> lock(registry_mutex);
  cpu_handles[id] = std::move(handle);
  return id;
}

int64_t insert_moe_handle(std::shared_ptr<KtMoeHandle> handle) {
  const int64_t id = next_handle.fetch_add(1);
  std::lock_guard<std::mutex> lock(registry_mutex);
  moe_handles[id] = std::move(handle);
  return id;
}

std::shared_ptr<KtCpuInferHandle> get_cpu_handle(int64_t handle) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  auto it = cpu_handles.find(handle);
  TORCH_CHECK(it != cpu_handles.end(), "invalid kt_kernel CPU handle");
  return it->second;
}

std::shared_ptr<KtMoeHandle> get_moe_handle(int64_t handle) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  auto it = moe_handles.find(handle);
  TORCH_CHECK(it != moe_handles.end(), "invalid kt_kernel MoE handle");
  return it->second;
}

}  // namespace

int64_t kt_create_cpu_infer(const at::Tensor& numa_nodes,
                            const at::Tensor& thread_counts) {
  return insert_cpu_handle(
      std::make_shared<KtCpuInferHandle>(numa_nodes, thread_counts));
}

void kt_destroy_cpu_infer(int64_t handle, const at::Tensor& dispatch_key) {
  (void)dispatch_key;
  std::lock_guard<std::mutex> lock(registry_mutex);
  cpu_handles.erase(handle);
}

int64_t kt_create_moe(
    int64_t cpu_handle_id, const std::string& method, int64_t layer_idx,
    int64_t expert_num, int64_t num_experts_per_tok, int64_t hidden_size,
    int64_t intermediate_size, const at::Tensor& gpu_experts_mask,
    int64_t max_len, const std::string& path, bool load, bool save,
    const at::Tensor& gate_projs, const at::Tensor& up_projs,
    const at::Tensor& down_projs, const at::Tensor& gate_scales,
    const at::Tensor& up_scales, const at::Tensor& down_scales,
    int64_t quant_bits, int64_t group_size, bool zero_point,
    bool per_channel, int64_t activation_type) {
  auto cpu_handle = get_cpu_handle(cpu_handle_id);

  TORCH_CHECK(gpu_experts_mask.device().is_cpu(),
              "gpu_experts_mask must be on CPU");
  TORCH_CHECK(gpu_experts_mask.scalar_type() == at::kBool,
              "gpu_experts_mask must use bool dtype");
  auto gpu_mask = gpu_experts_mask.contiguous();

  GeneralMOEConfig config(static_cast<int>(expert_num),
                          static_cast<int>(num_experts_per_tok),
                          static_cast<int>(hidden_size),
                          static_cast<int>(intermediate_size));
  config.layer_idx = static_cast<int>(layer_idx);
  config.pool = cpu_handle->cpuinfer->backend_;
  config.gpu_experts_mask = reinterpret_cast<uint8_t*>(gpu_mask.data_ptr());
  config.compute_num_gpu_experts();
  config.max_len = static_cast<int>(max_len);
  config.path = path;
  config.load = load;
  config.save = save;
  config.gate_projs = ptr_matrix_from_tensor(gate_projs);
  config.up_projs = ptr_matrix_from_tensor(up_projs);
  config.down_projs = ptr_matrix_from_tensor(down_projs);
  config.gate_scales = ptr_matrix_from_tensor(gate_scales);
  config.up_scales = ptr_matrix_from_tensor(up_scales);
  config.down_scales = ptr_matrix_from_tensor(down_scales);
  config.quant_config.bits = static_cast<int>(quant_bits);
  config.quant_config.group_size = static_cast<int>(group_size);
  config.quant_config.zero_point = zero_point;
  config.quant_config.per_channel = per_channel;
  config.activation_type = static_cast<int>(activation_type);

  std::shared_ptr<KtMoeHandle> handle;
#if defined(__x86_64__)
  if (method == "AMXINT4") {
    handle = make_moe_handle<AMX_MOE_TP<amx::GemmKernel224Int4>>(config);
  } else if (method == "AMXINT8") {
    handle = make_moe_handle<AMX_MOE_TP<amx::GemmKernel224Int8>>(config);
  } else if (method == "RAWINT4") {
    handle =
        make_moe_handle<AMX_K2_MOE_TP<amx::GemmKernel224Int4SmallKGroup>>(
            config);
  } else if (method == "FP8") {
#if defined(__AVX512F__)
    handle = make_moe_handle<AMX_FP8_MOE_TP<amx::GemmKernel224FP8>>(config);
#else
    handle = make_moe_handle<AVX2_FP8_MOE_TP<avx2::GemmKernelAVX2FP8>>(
        config);
#endif
  } else if (method == "FP8_AVX2") {
    handle = make_moe_handle<AVX2_FP8_MOE_TP<avx2::GemmKernelAVX2FP8>>(
        config);
  } else if (method == "FP8_PERCHANNEL") {
#if defined(__AVX512F__)
    handle = make_moe_handle<
        AMX_FP8_PERCHANNEL_MOE_TP<amx::GemmKernel224FP8PerChannel>>(config);
#else
    TORCH_CHECK(false,
                "FP8_PERCHANNEL kt_kernel backend requires AVX512 support");
#endif
  } else if (method == "BF16") {
#if defined(__AVX512F__)
    handle = make_moe_handle<AMX_BF16_MOE_TP<amx::GemmKernel224BF16>>(
        config);
#else
    handle = make_moe_handle<AVX2_BF16_MOE_TP<avx2::GemmKernelAVX2BF16>>(
        config);
#endif
  } else if (method == "BF16_AVX2") {
    handle = make_moe_handle<AVX2_BF16_MOE_TP<avx2::GemmKernelAVX2BF16>>(
        config);
  } else if (method == "BF16_AVX512") {
#if defined(__AVX512BF16__) && defined(__AVX512F__)
    handle =
        make_moe_handle<AVX512_BF16_MOE_TP<avx512::GemmKernelAVX512BF16>>(
            config);
#else
    TORCH_CHECK(false,
                "BF16_AVX512 kt_kernel backend requires AVX512 BF16 support");
#endif
  } else if (method == "GPTQ_INT4") {
    handle =
        make_moe_handle<AVX2_GPTQ_INT4_MOE_TP<avx2::GemmKernelAVX2GPTQInt4>>(
            config);
  } else {
    TORCH_CHECK(false, "unsupported kt_kernel MoE method: ", method);
  }
#else
  TORCH_CHECK(false, "kt_kernel MoE is currently supported only on x86_64");
#endif

  // Keep the mask tensor alive for the lifetime of the native config pointer.
  handle->gpu_experts_mask = gpu_mask;
  return insert_moe_handle(std::move(handle));
}

void kt_destroy_moe(int64_t handle, const at::Tensor& dispatch_key) {
  (void)dispatch_key;
  std::lock_guard<std::mutex> lock(registry_mutex);
  moe_handles.erase(handle);
}

void kt_moe_load_weights(int64_t handle, const at::Tensor& physical_map) {
  auto moe_handle = get_moe_handle(handle);
  moe_handle->physical_map = physical_map.contiguous();
  moe_handle->load_weights(moe_handle->physical_map);
}

void kt_moe_submit_forward(int64_t handle, const at::Tensor& batch_size,
                           const at::Tensor& expert_ids,
                           const at::Tensor& weights,
                           const at::Tensor& input,
                           const at::Tensor& output, bool incremental) {
  auto moe_handle = get_moe_handle(handle);
  const int qlen = *tensor_ptr<int32_t>(batch_size);
  const int k = static_cast<int>(expert_ids.size(-1));
  const int64_t* expert_ids_ptr = tensor_ptr<int64_t>(expert_ids);
  const float* weights_ptr = tensor_ptr<float>(weights);
  const void* input_ptr = input.data_ptr();
  void* output_ptr = const_cast<void*>(output.data_ptr());
  moe_handle->forward_future = std::async(
      std::launch::async,
      [moe = moe_handle->moe, qlen, k, expert_ids_ptr, weights_ptr, input_ptr,
       output_ptr, incremental]() {
        moe->forward(qlen, k, expert_ids_ptr, weights_ptr, input_ptr,
                     output_ptr, incremental);
      });
}

void kt_moe_sync_forward(int64_t handle, const at::Tensor& dispatch_key) {
  (void)dispatch_key;
  auto moe_handle = get_moe_handle(handle);
  if (moe_handle->forward_future.valid()) {
    moe_handle->forward_future.get();
  }
}
