#include "dispatch_utils.h"

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>
#include <curand_kernel.h>

#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif

namespace aphrodite {

constexpr int TOP_K_MAX = 256;

template <typename T>
struct TopK_2 {
  int idx = -1;
  T val;

  __device__ __forceinline__ void insert(T elem, int elem_idx) {
    // Use float comparison to avoid operator ambiguity
    if (static_cast<float>(elem) > static_cast<float>(val)) {
      val = elem;
      idx = elem_idx;
    }
  }

  __device__ __forceinline__ void init() {
    // Initialize with appropriate minimum value based on type
    if constexpr (std::is_same_v<T, float>) {
      val = -FLT_MAX;
    } else {
      val = static_cast<T>(-65504.0f);  // Half precision min
    }
    idx = -1;
  }
};

template <typename T>
__device__ __forceinline__ TopK_2<T> reduce_topk_op(const TopK_2<T>& a, const TopK_2<T>& b) {
  // Use float comparison to avoid operator ambiguity
  return static_cast<float>(a.val) > static_cast<float>(b.val) ? a : b;
}

// Stage 1: Find top-k values per block
template <typename scalar_t, int BLOCK_SIZE, int BLOCKS_PER_SEQ>
__global__ void topk_stage1_kernel(
    const scalar_t* __restrict__ logits,          // [num_seqs, vocab_size]
    scalar_t* __restrict__ tmp_logits,            // [num_seqs, vocab_size]
    int* __restrict__ topk_tmp_id_buf,           // [num_seqs, BLOCKS_PER_SEQ * k]
    scalar_t* __restrict__ topk_tmp_val_buf,     // [num_seqs, BLOCKS_PER_SEQ * k]
    const int* __restrict__ topk_values,         // [num_seqs] - k values per sequence
    const int num_seqs,
    const int vocab_size) {

  typedef cub::BlockReduce<TopK_2<scalar_t>, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int tid = threadIdx.x;
  const int seq_idx = blockIdx.x / BLOCKS_PER_SEQ;
  const int block_lane = blockIdx.x % BLOCKS_PER_SEQ;

  if (seq_idx >= num_seqs) return;

  const int k = topk_values[seq_idx];
  if (k == 0) return;

  const int logits_offset = seq_idx * vocab_size;
  const int tmp_buf_offset = seq_idx * BLOCKS_PER_SEQ * TOP_K_MAX + block_lane * k;

  // Copy logits to temporary buffer
  for (int i = tid + block_lane * BLOCK_SIZE; i < vocab_size; i += BLOCK_SIZE * BLOCKS_PER_SEQ) {
    tmp_logits[logits_offset + i] = logits[logits_offset + i];
  }

  // Find top-k values iteratively
  for (int ite = 0; ite < k; ite++) {
    TopK_2<scalar_t> partial;
    partial.init();

    // Each thread finds its maximum
    for (int i = tid + block_lane * BLOCK_SIZE; i < vocab_size; i += BLOCK_SIZE * BLOCKS_PER_SEQ) {
      int idx = logits_offset + i;
      partial.insert(tmp_logits[idx], idx);
    }

    // Reduce across block
    TopK_2<scalar_t> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<scalar_t>);

    if (tid == 0) {
      topk_tmp_id_buf[tmp_buf_offset + ite] = total.idx;
      topk_tmp_val_buf[tmp_buf_offset + ite] = total.val;
      if (total.idx >= 0) {
        tmp_logits[total.idx] = static_cast<scalar_t>(-65504.0f);  // Safe min for half precision
      }
    }
    __syncthreads();
  }
}

// Stage 2: Merge results and sample
template <typename scalar_t, int BLOCK_SIZE, int BLOCKS_PER_SEQ>
__global__ void topk_stage2_sampling_kernel(
    const int* __restrict__ topk_tmp_id_buf,      // [num_seqs, BLOCKS_PER_SEQ * k]
    scalar_t* __restrict__ topk_tmp_val_buf,      // [num_seqs, BLOCKS_PER_SEQ * k]
    int64_t* __restrict__ output_ids,             // [num_seqs]
    float* __restrict__ output_logprobs,          // [num_seqs] optional
    const int* __restrict__ topk_values,          // [num_seqs]
    const float* __restrict__ top_p_values,       // [num_seqs] optional
    curandState_t* __restrict__ curand_states,    // [num_seqs]
    const int num_seqs,
    const int vocab_size,
    const bool normalize_logprobs) {

  typedef cub::BlockReduce<TopK_2<float>, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  extern __shared__ char shared_mem[];

  const int tid = threadIdx.x;
  const int seq_idx = blockIdx.x;

  if (seq_idx >= num_seqs) return;

  const int k = topk_values[seq_idx];
  if (k == 0) return;

  const float top_p = top_p_values ? top_p_values[seq_idx] : 1.0f;
  const int stride = TOP_K_MAX * BLOCKS_PER_SEQ;

  // Shared memory arrays
  int* s_id = reinterpret_cast<int*>(shared_mem);
  float* s_val = reinterpret_cast<float*>(s_id + k);

  __shared__ float s_sum;
  if (tid == 0) {
    s_sum = 0.0f;
  }
  __syncthreads();

  scalar_t* val_buf = topk_tmp_val_buf + seq_idx * stride;

  // Find top-k across all blocks
  float max_logit = -FLT_MAX;
  for (int ite = 0; ite < k; ite++) {
    TopK_2<float> partial;
    partial.init();

    // Each thread searches in the merged buffer
    for (int i = tid; i < k * BLOCKS_PER_SEQ; i += BLOCK_SIZE) {
      partial.insert(static_cast<float>(val_buf[i]), i);
    }

    TopK_2<float> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<float>);

    if (tid == 0) {
      if (ite == 0) {
        max_logit = total.val;
      }
      s_id[ite] = total.idx;
      val_buf[total.idx] = static_cast<scalar_t>(-65504.0f);  // Safe min for half precision

      // Convert to probability
      total.val = expf(total.val - max_logit);
      s_val[ite] = total.val;
      s_sum += total.val;
    }
    __syncthreads();
  }

  // Sample from top-k
  if (tid == 0) {
    float rand_num = curand_states ?
                     curand_uniform(&curand_states[seq_idx]) * top_p * s_sum :
                     top_p * s_sum;

    int selected_idx = k - 1;  // Default to last element
    for (int i = 0; i < k; i++) {
      rand_num -= s_val[i];
      if (rand_num <= 0.0f) {
        selected_idx = i;
        break;
      }
    }

    // Get actual token id
    int buffer_idx = s_id[selected_idx];
    int token_id = buffer_idx >= 0 ?
                   topk_tmp_id_buf[seq_idx * stride + buffer_idx] % vocab_size :
                   vocab_size - 1;

    output_ids[seq_idx] = token_id;

    // Optional: output log probability
    if (output_logprobs) {
      float log_prob = logf(s_val[selected_idx]);
      if (normalize_logprobs) {
        log_prob -= logf(s_sum);
      }
      output_logprobs[seq_idx] = log_prob;
    }
  }
}

}  // namespace aphrodite

void topk_topp_sampling(
    torch::Tensor& logits,              // [num_seqs, vocab_size]
    torch::Tensor& output_ids,          // [num_seqs]
    const torch::Tensor& top_k_values,  // [num_seqs]
    const std::optional<torch::Tensor>& top_p_values,  // [num_seqs] optional
    const std::optional<torch::Tensor>& curand_states, // [num_seqs] optional
    std::optional<torch::Tensor>& output_logprobs,     // [num_seqs] optional
    bool normalize_logprobs = false) {

  TORCH_CHECK(logits.is_contiguous());
  TORCH_CHECK(output_ids.is_contiguous());
  TORCH_CHECK(top_k_values.is_contiguous());

  int num_seqs = logits.size(0);
  int vocab_size = logits.size(1);

  if (num_seqs == 0) return;

  // Allocate temporary buffers
  constexpr int BLOCKS_PER_SEQ = 8;
  auto options = torch::TensorOptions()
                     .dtype(logits.dtype())
                     .device(logits.device());

  torch::Tensor tmp_logits = torch::empty_like(logits);
  torch::Tensor topk_tmp_id_buf = torch::empty({num_seqs, BLOCKS_PER_SEQ * aphrodite::TOP_K_MAX},
                                                torch::kInt32).to(logits.device());
  torch::Tensor topk_tmp_val_buf = torch::empty({num_seqs, BLOCKS_PER_SEQ * aphrodite::TOP_K_MAX},
                                                 options);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(logits));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Determine grid and block sizes
  constexpr int BLOCK_SIZE_STAGE1 = 256;
  constexpr int BLOCK_SIZE_STAGE2 = 128;

  dim3 grid1(num_seqs * BLOCKS_PER_SEQ);
  dim3 block1(BLOCK_SIZE_STAGE1);

  dim3 grid2(num_seqs);
  dim3 block2(BLOCK_SIZE_STAGE2);

  // Calculate shared memory size for stage 2
  // Assuming max k = TOP_K_MAX
  size_t shared_mem_size = aphrodite::TOP_K_MAX * sizeof(int) +
                          aphrodite::TOP_K_MAX * sizeof(float);

  APHRODITE_DISPATCH_FLOATING_TYPES(
      logits.scalar_type(), "topk_sampling", [&] {
        // Stage 1: Find top-k per block
        aphrodite::topk_stage1_kernel<scalar_t, BLOCK_SIZE_STAGE1, BLOCKS_PER_SEQ>
            <<<grid1, block1, 0, stream>>>(
                logits.data_ptr<scalar_t>(),
                tmp_logits.data_ptr<scalar_t>(),
                topk_tmp_id_buf.data_ptr<int>(),
                topk_tmp_val_buf.data_ptr<scalar_t>(),
                top_k_values.data_ptr<int>(),
                num_seqs,
                vocab_size);

        // Stage 2: Merge and sample
        aphrodite::topk_stage2_sampling_kernel<scalar_t, BLOCK_SIZE_STAGE2, BLOCKS_PER_SEQ>
            <<<grid2, block2, shared_mem_size, stream>>>(
                topk_tmp_id_buf.data_ptr<int>(),
                topk_tmp_val_buf.data_ptr<scalar_t>(),
                output_ids.data_ptr<int64_t>(),
                output_logprobs.has_value() ?
                    output_logprobs.value().data_ptr<float>() : nullptr,
                top_k_values.data_ptr<int>(),
                top_p_values.has_value() ?
                    top_p_values.value().data_ptr<float>() : nullptr,
                curand_states.has_value() ?
                    static_cast<curandState_t*>(curand_states.value().data_ptr()) : nullptr,
                num_seqs,
                vocab_size,
                normalize_logprobs);
      });
}
