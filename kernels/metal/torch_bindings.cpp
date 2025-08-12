#include <ATen/ATen.h>
#include <torch/torch.h>
#include <torch/library.h>

#include "../ops.h"
#include "../core/registration.h"


void swap_blocks(torch::Tensor &src, torch::Tensor &dst,
                 const torch::Tensor &block_mapping);
void copy_blocks(const std::vector<torch::Tensor> &key_caches,
                 const std::vector<torch::Tensor> &value_caches,
                 const torch::Tensor &block_mapping);
void reshape_and_cache(torch::Tensor &key, torch::Tensor &value,
                       torch::Tensor &key_cache, torch::Tensor &value_cache,
                       torch::Tensor &slot_mapping, const std::string &kv_cache_dtype,
                       torch::Tensor &k_scale, torch::Tensor &v_scale);
void reshape_and_cache_flash(torch::Tensor &key, torch::Tensor &value,
                             torch::Tensor &key_cache, torch::Tensor &value_cache,
                             torch::Tensor &slot_mapping, const std::string &kv_cache_dtype,
                             torch::Tensor &k_scale, torch::Tensor &v_scale);
void convert_fp8(torch::Tensor &dst_cache, torch::Tensor &src_cache,
                 const double scale, const std::string &kv_cache_dtype);


int64_t get_device_attribute(int64_t attribute, int64_t device_id);
int64_t get_max_shared_memory_per_block_device_attribute(int64_t device_id);



TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "paged_attention_v1("
      "    Tensor! out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"
      "    int tp_rank, int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step) -> ()");
  m.def(
      "paged_attention_v2("
      "    Tensor! out, Tensor! exp_sums, Tensor! max_logits,"
      "    Tensor! tmp_out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"
      "    int tp_rank, int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step) -> ()");
}

TORCH_LIBRARY_FRAGMENT(CONCAT(_C, _cache_ops), m) {
  m.def("swap_blocks(Tensor src, Tensor! dst, Tensor block_mapping) -> ()");
  m.def(
      "copy_blocks(Tensor(a!)[] key_caches, Tensor[](b!) value_caches, "
      "Tensor block_mapping) -> ()");
  m.def(
      "reshape_and_cache(Tensor key, Tensor value,"
      "                  Tensor! key_cache, Tensor! value_cache,"
      "                  Tensor slot_mapping,"
      "                  str kv_cache_dtype,"
      "                  Tensor k_scale, Tensor v_scale) -> ()");
  m.def(
      "reshape_and_cache_flash(Tensor key, Tensor value,"
      "                        Tensor! key_cache,"
      "                        Tensor! value_cache,"
      "                        Tensor slot_mapping,"
      "                        str kv_cache_dtype,"
      "                        Tensor k_scale, Tensor v_scale) -> ()");
  m.def(
      "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, "
      "str kv_cache_dtype) -> ()");
}


TORCH_LIBRARY_IMPL_EXPAND(_C, MPS, ops) {
  ops.impl("paged_attention_v1", &paged_attention_v1);
  ops.impl("paged_attention_v2", &paged_attention_v2);
}

TORCH_LIBRARY_IMPL_EXPAND(CONCAT(_C, _cache_ops), MPS, cache_ops) {
  cache_ops.impl("swap_blocks", &swap_blocks);
  cache_ops.impl("copy_blocks", &copy_blocks);
  cache_ops.impl("reshape_and_cache", &reshape_and_cache);
  cache_ops.impl("reshape_and_cache_flash", &reshape_and_cache_flash);
  cache_ops.impl("convert_fp8", &convert_fp8);
}


TORCH_LIBRARY_FRAGMENT(CONCAT(_C, _cuda_utils), m) {
  m.def("get_device_attribute(int attribute, int device_id) -> int");
  m.def("get_max_shared_memory_per_block_device_attribute(int device_id) -> int");
}

TORCH_LIBRARY_IMPL_EXPAND(CONCAT(_C, _cuda_utils), MPS, cuda_utils) {
  cuda_utils.impl("get_device_attribute", &get_device_attribute);
  cuda_utils.impl("get_max_shared_memory_per_block_device_attribute",
                  &get_max_shared_memory_per_block_device_attribute);
}


TORCH_LIBRARY_IMPL_EXPAND(CONCAT(_C, _cuda_utils), CPU, cuda_utils_cpu) {
  cuda_utils_cpu.impl("get_device_attribute", &get_device_attribute);
  cuda_utils_cpu.impl("get_max_shared_memory_per_block_device_attribute",
                      &get_max_shared_memory_per_block_device_attribute);
}


TORCH_LIBRARY_IMPL_EXPAND(CONCAT(_C, _cuda_utils), CompositeImplicitAutograd, cuda_utils_fallback) {
  cuda_utils_fallback.impl("get_device_attribute", &get_device_attribute);
  cuda_utils_fallback.impl(
      "get_max_shared_memory_per_block_device_attribute",
      &get_max_shared_memory_per_block_device_attribute);
}

REGISTER_EXTENSION(_C)


