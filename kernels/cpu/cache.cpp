#include <map>
#include <vector>

#include "cpu_types.hpp"

#if defined(__x86_64__)
  #define DISPATCH_MACRO APHRODITE_DISPATCH_FLOATING_TYPES_WITH_E5M2
#else
  #define DISPATCH_MACRO APHRODITE_DISPATCH_FLOATING_TYPES
#endif

namespace {
template <typename scalar_t>
void copy_blocks_cpu_impl(std::vector<torch::Tensor> const& key_caches,
                          std::vector<torch::Tensor> const& value_caches,
                          const torch::Tensor& mapping_pairs,
                          const int element_num_per_block,
                          const int layer_num) {
  const size_t pair_num = mapping_pairs.size(0);
  const size_t block_bytes = sizeof(scalar_t) * element_num_per_block;
#pragma omp parallel for collapse(2)
  for (int layer = 0; layer < layer_num; ++layer) {
    for (size_t pair = 0; pair < pair_num; ++pair) {
      int64_t source_offset =
          element_num_per_block * mapping_pairs[pair][0].item<int64_t>();
      int64_t target_offset =
          element_num_per_block * mapping_pairs[pair][1].item<int64_t>();
      scalar_t* key_cache_ptr = key_caches[layer].data_ptr<scalar_t>();
      scalar_t* source_ptr = key_cache_ptr + source_offset;
      scalar_t* target_ptr = key_cache_ptr + target_offset;
      std::memcpy(target_ptr, source_ptr, block_bytes);

      scalar_t* value_cache_ptr = value_caches[layer].data_ptr<scalar_t>();
      source_ptr = value_cache_ptr + source_offset;
      target_ptr = value_cache_ptr + target_offset;
      std::memcpy(target_ptr, source_ptr, block_bytes);
    }
  }
}

template <typename scalar_t>
void reshape_and_cache_cpu_impl(
    const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
    scalar_t* __restrict__ key_cache, scalar_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping, const int num_tokens,
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x) {
  const int block_elem_num = num_heads * head_size * block_size;

#pragma omp parallel for collapse(2)
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
      const int64_t slot_idx = slot_mapping[token_idx];
      if (slot_idx >= 0) {
        int src_key_head_idx = token_idx * key_stride + head_idx * head_size;
        int src_value_head_idx =
            token_idx * value_stride + head_idx * head_size;
        const scalar_t* src_key_head_ptr = key + src_key_head_idx;
        const scalar_t* src_value_head_ptr = value + src_value_head_idx;
        const int64_t block_index = slot_idx / block_size;
        const int64_t block_offset = slot_idx % block_size;
        scalar_t* target_key_head_ptr = key_cache +
                                        block_elem_num * block_index +
                                        head_idx * block_size * head_size;
        scalar_t* target_value_head_ptr = value_cache +
                                          block_elem_num * block_index +
                                          head_idx * block_size * head_size;

        for (int src_key_idx = 0; src_key_idx < head_size; src_key_idx += x) {
          const int64_t target_offset =
              src_key_idx * block_size + block_offset * x;
          for (int i = 0; i < x; ++i) {
            target_key_head_ptr[target_offset + i] =
                src_key_head_ptr[src_key_idx + i];
          }
        }

        for (int src_value_idx = 0; src_value_idx < head_size;
             ++src_value_idx) {
          const int64_t target_offset =
              src_value_idx * block_size + block_offset;
          target_value_head_ptr[target_offset] =
              src_value_head_ptr[src_value_idx];
        }
      }
    }
  }
}
};  // namespace

template <typename scalar_t>
void concat_and_cache_mla_cpu_impl(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    scalar_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank
                                      // + pe_dim)]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int num_tokens,                      //
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size                       //
) {
#pragma omp parallel for
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    const int64_t slot_idx = slot_mapping[token_idx];
    // NOTE: slot_idx can be -1 if the token is padded
    if (slot_idx < 0) {
      continue;
    }
    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;

    auto copy = [&](const scalar_t* __restrict__ src,
                    scalar_t* __restrict__ dst, int src_stride, int dst_stride,
                    int size, int offset) {
      for (int i = 0; i < size; i++) {
        const int64_t src_idx = token_idx * src_stride + i;
        const int64_t dst_idx =
            block_idx * block_stride + block_offset * entry_stride + i + offset;
        dst[dst_idx] = src[src_idx];
      }
    };

    copy(kv_c, kv_cache, kv_c_stride, block_stride, kv_lora_rank, 0);
    copy(k_pe, kv_cache, k_pe_stride, block_stride, pe_dim, kv_lora_rank);
  }
}

// Note: the key_caches and value_caches vectors are constant but
// not the Tensors they contain. The vectors need to be const refs
// in order to satisfy pytorch's C++ operator registration code.
void copy_blocks(std::vector<torch::Tensor> const& key_caches,
                 std::vector<torch::Tensor> const& value_caches,
                 const torch::Tensor& block_mapping) {
  unsigned num_layers = key_caches.size();
  TORCH_CHECK(num_layers == value_caches.size());
  if (num_layers == 0) {
    return;
  }

  const int element_num_per_block = key_caches[0][0].numel();
  DISPATCH_MACRO(key_caches[0].scalar_type(), "copy_blocks_cpu_impl", [&] {
    CPU_KERNEL_GUARD_IN(copy_blocks_cpu_impl)
    copy_blocks_cpu_impl<scalar_t>(key_caches, value_caches, block_mapping,
                                   element_num_per_block, num_layers);
    CPU_KERNEL_GUARD_OUT(copy_blocks_cpu_impl)
  });
}

void reshape_and_cache(torch::Tensor& key, torch::Tensor& value,
                       torch::Tensor& key_cache, torch::Tensor& value_cache,
                       torch::Tensor& slot_mapping,
                       const std::string& kv_cache_dtype,
                       torch::Tensor& k_scale, torch::Tensor& v_scale) {
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  DISPATCH_MACRO(key.scalar_type(), "reshape_and_cache_cpu_impl", [&] {
    CPU_KERNEL_GUARD_IN(reshape_and_cache_cpu_impl)
    reshape_and_cache_cpu_impl<scalar_t>(
        key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
        key_cache.data_ptr<scalar_t>(), value_cache.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int64_t>(), num_tokens, key_stride, value_stride,
        num_heads, head_size, block_size, x);
    CPU_KERNEL_GUARD_OUT(reshape_and_cache_cpu_impl)
  });
}

void concat_and_cache_mla(
    torch::Tensor& kv_c,          // [num_tokens, kv_lora_rank]
    torch::Tensor& k_pe,          // [num_tokens, pe_dim]
    torch::Tensor& kv_cache,      // [num_blocks, block_size, (kv_lora_rank +
                                  // pe_dim)]
    torch::Tensor& slot_mapping,  // [num_tokens] or [num_actual_tokens]
    const std::string& kv_cache_dtype, torch::Tensor& scale) {
  int num_tokens = slot_mapping.size(0);
  int kv_lora_rank = kv_c.size(1);
  int pe_dim = k_pe.size(1);
  int block_size = kv_cache.size(1);

  TORCH_CHECK(kv_cache.size(2) == kv_lora_rank + pe_dim);
  TORCH_CHECK(kv_cache_dtype != "fp8");

  int kv_c_stride = kv_c.stride(0);
  int k_pe_stride = k_pe.stride(0);
  int block_stride = kv_cache.stride(0);
  int entry_stride = kv_cache.stride(1);

  APHRODITE_DISPATCH_FLOATING_TYPES(
      kv_c.scalar_type(), "concat_and_cache_mla_cpu_impl", [&] {
        CPU_KERNEL_GUARD_IN(concat_and_cache_mla_cpu_impl)
        concat_and_cache_mla_cpu_impl<scalar_t>(
            kv_c.data_ptr<scalar_t>(), k_pe.data_ptr<scalar_t>(),
            kv_cache.data_ptr<scalar_t>(), slot_mapping.data_ptr<int64_t>(),
            num_tokens, block_stride, entry_stride, kv_c_stride, k_pe_stride,
            kv_lora_rank, pe_dim, block_size);
        CPU_KERNEL_GUARD_OUT(concat_and_cache_mla_cpu_impl)
      });
}

void swap_blocks(torch::Tensor& src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping) {
  TORCH_CHECK(false, "swap_blocks is unsupported on CPU.")
}
