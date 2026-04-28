/**
 * @Description  : AVX512 BF16 MoE base class (ported from avx2/moe_base.hpp)
 * @Author       : Claude
 * @Date         : 2026-03-18
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * Uses AVX512 BF16 helpers for activation, expert merge, and weighted output
 * accumulation. AMX tile configuration calls (T::config()) are kept as no-ops.
 **/
#ifndef CPUINFER_OPERATOR_AVX512_MOE_BASE_H
#define CPUINFER_OPERATOR_AVX512_MOE_BASE_H

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../../cpu_backend/shared_mem_buffer.h"
#include "../../cpu_backend/worker_pool.h"
#include "../common.hpp"
#include "../moe-tp.hpp"
#include "avx512_bf16_gemm.hpp"
#include "avx512_bf16_utils.hpp"
#include "llama.cpp/ggml.h"

template <class T, class Derived>
class AVX512_MOE_BASE {
 public:
  int tp_part_idx = 0;

  ggml_bf16_t* m_local_input_ = nullptr;
  ggml_bf16_t* m_local_gate_output_ = nullptr;
  ggml_bf16_t* m_local_up_output_ = nullptr;
  ggml_bf16_t* m_local_down_output_ = nullptr;

  std::vector<std::vector<int>> m_local_pos_;
  std::vector<int> m_local_num_;
  std::vector<int> m_expert_id_map_;
  std::vector<ggml_bf16_t*> m_local_input_ptr_;
  std::vector<ggml_bf16_t*> m_local_gate_output_ptr_;
  std::vector<ggml_bf16_t*> m_local_up_output_ptr_;
  std::vector<ggml_bf16_t*> m_local_down_output_ptr_;

  std::vector<std::shared_ptr<typename T::BufferA>> gate_up_ba_;
  std::vector<std::shared_ptr<typename T::BufferB>> gate_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> gate_bc_;
  std::vector<std::shared_ptr<typename T::BufferB>> up_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> up_bc_;
  std::vector<std::shared_ptr<typename T::BufferA>> down_ba_;
  std::vector<std::shared_ptr<typename T::BufferB>> down_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> down_bc_;

  size_t pool_count_ = 0;
  size_t gate_up_ba_pool_bytes_ = 0;
  size_t gate_bc_pool_bytes_ = 0;
  size_t up_bc_pool_bytes_ = 0;
  size_t down_ba_pool_bytes_ = 0;
  size_t down_bc_pool_bytes_ = 0;
  void* gate_up_ba_pool_ = nullptr;
  void* gate_bc_pool_ = nullptr;
  void* up_bc_pool_ = nullptr;
  void* down_ba_pool_ = nullptr;
  void* down_bc_pool_ = nullptr;

  GeneralMOEConfig config_;
  using input_t = ggml_bf16_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = T::ELEMENT_SIZE;

  AVX512_MOE_BASE(GeneralMOEConfig config, int tp_part_idx_) : tp_part_idx(tp_part_idx_), config_(config) {
    init();
    derived()->derived_init();
  }

  void init() {
    if (config_.load && config_.path == "") {
      config_.load = false;
    }

    MemoryRequest mem_requests;
    mem_requests.append_pointer(
        &m_local_input_, sizeof(ggml_bf16_t) * config_.num_experts_per_tok * config_.max_len * config_.hidden_size);
    mem_requests.append_pointer(&m_local_gate_output_, sizeof(ggml_bf16_t) * config_.num_experts_per_tok *
                                                           config_.max_len * config_.intermediate_size);
    mem_requests.append_pointer(&m_local_up_output_, sizeof(ggml_bf16_t) * config_.num_experts_per_tok *
                                                         config_.max_len * config_.intermediate_size);
    mem_requests.append_pointer(&m_local_down_output_, sizeof(ggml_bf16_t) * config_.num_experts_per_tok *
                                                           config_.max_len * config_.hidden_size);

    m_local_pos_.resize(config_.max_len);
    for (int i = 0; i < config_.max_len; i++) {
      m_local_pos_[i].resize(config_.num_experts_per_tok);
    }
    m_expert_id_map_.resize(config_.expert_num);
    m_local_num_.resize(config_.expert_num);
    m_local_input_ptr_.resize(config_.expert_num);
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num);

    for (size_t i = 0; i < config_.expert_num; i++) {
      gate_up_ba_.push_back(make_buffer_a(config_.max_len, config_.hidden_size, nullptr));
      gate_bc_.push_back(make_buffer_c(config_.max_len, config_.intermediate_size, nullptr));
      up_bc_.push_back(make_buffer_c(config_.max_len, config_.intermediate_size, nullptr));
      down_ba_.push_back(make_buffer_a(config_.max_len, config_.intermediate_size, nullptr));
      down_bc_.push_back(make_buffer_c(config_.max_len, config_.hidden_size, nullptr));

      void* gate_bb_ptr =
          std::aligned_alloc(64, buffer_b_required_size(config_.intermediate_size, config_.hidden_size));
      gate_bb_.push_back(make_buffer_b(config_.intermediate_size, config_.hidden_size, gate_bb_ptr));

      void* up_bb_ptr = std::aligned_alloc(64, buffer_b_required_size(config_.intermediate_size, config_.hidden_size));
      up_bb_.push_back(make_buffer_b(config_.intermediate_size, config_.hidden_size, up_bb_ptr));

      void* down_bb_ptr =
          std::aligned_alloc(64, buffer_b_required_size(config_.hidden_size, config_.intermediate_size));
      down_bb_.push_back(make_buffer_b(config_.hidden_size, config_.intermediate_size, down_bb_ptr));
    }

    pool_count_ = config_.max_len * config_.num_experts_per_tok + config_.expert_num * T::M_STEP;

    gate_up_ba_pool_bytes_ = buffer_a_required_size(pool_count_, config_.hidden_size) + pool_count_ * 64;
    gate_bc_pool_bytes_ = buffer_c_required_size(pool_count_, config_.intermediate_size) + pool_count_ * 64;
    up_bc_pool_bytes_ = buffer_c_required_size(pool_count_, config_.intermediate_size) + pool_count_ * 64;
    down_ba_pool_bytes_ = buffer_a_required_size(pool_count_, config_.intermediate_size) + pool_count_ * 64;
    down_bc_pool_bytes_ = buffer_c_required_size(pool_count_, config_.hidden_size) + pool_count_ * 64;

    mem_requests.append_pointer(&gate_up_ba_pool_, gate_up_ba_pool_bytes_);
    mem_requests.append_pointer(&gate_bc_pool_, gate_bc_pool_bytes_);
    mem_requests.append_pointer(&up_bc_pool_, up_bc_pool_bytes_);
    mem_requests.append_pointer(&down_ba_pool_, down_ba_pool_bytes_);
    mem_requests.append_pointer(&down_bc_pool_, down_bc_pool_bytes_);

    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
  }

  ~AVX512_MOE_BASE() = default;

  void warm_up() {
    int qlen = config_.max_len;
    std::vector<uint8_t> input(sizeof(ggml_bf16_t) * qlen * config_.hidden_size);
    std::vector<uint8_t> output(sizeof(ggml_bf16_t) * qlen * config_.hidden_size);
    std::vector<int64_t> expert_ids(qlen * config_.num_experts_per_tok);
    std::vector<float> weights(qlen * config_.num_experts_per_tok);
    for (int i = 0; i < qlen * config_.num_experts_per_tok; i++) {
      expert_ids[i] = i % config_.expert_num;
      weights[i] = 0.01;
    }
    forward(qlen, config_.num_experts_per_tok, expert_ids.data(), weights.data(), input.data(), output.data());
  }

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    if (qlen > 1) {
      forward_prefill(qlen, k, expert_ids, weights, input, output);
    } else {
      forward_decode(k, expert_ids, weights, input, output);
    }
  }

  template <typename... Args>
  void load_weights(Args&&... args) {
    derived()->load_weights(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void write_weights_to_buffer(Args&&... args) const {
    derived_const()->write_weights_to_buffer(std::forward<Args>(args)...);
  }

  void forward_prefill(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                       void* output) {
    auto pool = config_.pool->get_subpool(tp_part_idx);

    int activated_expert = 0;
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[i * k + j])) {
          continue;
        }
        m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
      }
    }

    for (int i = 0; i < config_.expert_num; i++) {
      if (m_local_num_[i] > 0) {
        m_expert_id_map_[activated_expert] = i;
        activated_expert++;
      }
    }

    // Assign pool memory to buffers
    size_t offset = 0;
    void* gate_up_ba_pool_ptr = gate_up_ba_pool_;
    void* gate_bc_pool_ptr = gate_bc_pool_;
    void* up_bc_pool_ptr = up_bc_pool_;
    void* down_ba_pool_ptr = down_ba_pool_;
    void* down_bc_pool_ptr = down_bc_pool_;
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };

    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];

      if (m_local_num_[i] == 0) continue;

      size_t max_m = (m_local_num_[i] + M_STEP - 1) / M_STEP * M_STEP;
      gate_up_ba_[i]->max_m = max_m;
      gate_up_ba_[i]->set_data(gate_up_ba_pool_ptr);
      gate_up_ba_pool_ptr = (void*)((uintptr_t)gate_up_ba_pool_ptr + align64(buffer_a_required_size(max_m, config_.hidden_size)));

      gate_bc_[i]->max_m = max_m;
      gate_bc_[i]->set_data(gate_bc_pool_ptr);
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr + align64(buffer_c_required_size(max_m, config_.intermediate_size)));

      up_bc_[i]->max_m = max_m;
      up_bc_[i]->set_data(up_bc_pool_ptr);
      up_bc_pool_ptr = (void*)((uintptr_t)up_bc_pool_ptr + align64(buffer_c_required_size(max_m, config_.intermediate_size)));

      down_ba_[i]->max_m = max_m;
      down_ba_[i]->set_data(down_ba_pool_ptr);
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr + align64(buffer_a_required_size(max_m, config_.intermediate_size)));

      down_bc_[i]->max_m = max_m;
      down_bc_[i]->set_data(down_bc_pool_ptr);
      down_bc_pool_ptr = (void*)((uintptr_t)down_bc_pool_ptr + align64(buffer_c_required_size(max_m, config_.hidden_size)));
    }

    auto direct_or_pool = [&](int count, auto&& fn) {
      if (qlen < 10) {
        for (int i = 0; i < count; i++) fn(i);
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr);
      }
    };

    // Copy input to per-expert buffers
    direct_or_pool(qlen, [&](int i) {
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[i * k + j])) continue;
        memcpy(m_local_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size,
               (ggml_bf16_t*)input + i * config_.hidden_size, sizeof(ggml_bf16_t) * config_.hidden_size);
      }
    });

    // Pack input into BufferA (trivial memcpy for AVX512)
    direct_or_pool(activated_expert, [this](int task_id) {
      int expert_idx = m_expert_id_map_[task_id];
      gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
    });

    // Gate + Up GEMM
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int) { T::config(); },
        [this, nth, qlen](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          derived()->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
          if (do_up) {
            up_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_output_ptr_[expert_idx], ith, nth);
          } else {
            gate_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], ith, nth);
          }
        },
        nullptr);

    // Activation: SiLU(gate) * up.
    apply_activation(activated_expert, nth, qlen);

    // Pack activation output into BufferA for down projection
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], 0, 1);
        },
        nullptr);

    // Down GEMM
    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * activated_expert, [](int) { T::config(); },
        [this, nth, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          derived()->do_down_gemm(expert_idx, ith, nth, qlen);
          down_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_down_output_ptr_[expert_idx], ith, nth);
        },
        nullptr);

    // Weighted sum of expert outputs.
    pool->do_work_stealing_job(
        qlen, nullptr,
        [this, output, k, expert_ids, weights](int i) {
          for (int e = 0; e < config_.hidden_size; e += 16) {
            __m512 x0 = _mm512_setzero_ps();
            for (int j = 0; j < k; j++) {
              if (config_.should_skip_expert(expert_ids[i * k + j])) continue;
              __m512 weight = _mm512_set1_ps(weights[i * k + j]);
              __m512 d0 = avx512::load_16xbf16_to_fp32(
                  m_local_down_output_ptr_[expert_ids[i * k + j]] +
                  m_local_pos_[i][j] * config_.hidden_size + e);
              x0 = _mm512_fmadd_ps(d0, weight, x0);
            }
            _mm512_storeu_ps((float*)output + i * config_.hidden_size + e, x0);
          }
        },
        nullptr);
  }

  void forward_decode(int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    int qlen = 1;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    int activated_expert = 0;
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);
    for (int i = 0; i < k; i++) {
      if (config_.should_skip_expert(expert_ids[i])) continue;
      m_expert_id_map_[activated_expert] = expert_ids[i];
      m_local_pos_[0][i] = 0;
      m_local_num_[expert_ids[i]] = qlen;
      activated_expert++;
    }

    size_t offset = 0;
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      m_local_gate_output_ptr_[expert_idx] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[expert_idx] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[expert_idx] = m_local_down_output_ + offset * config_.hidden_size;
      offset += qlen;
    }

    // Assign pool memory for decode
    void* gate_bc_pool_ptr = gate_bc_pool_;
    void* up_bc_pool_ptr = up_bc_pool_;
    void* down_ba_pool_ptr = down_ba_pool_;
    void* down_bc_pool_ptr = down_bc_pool_;
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };

    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      size_t max_m = (qlen + M_STEP - 1) / M_STEP * M_STEP;

      gate_bc_[expert_idx]->max_m = max_m;
      gate_bc_[expert_idx]->set_data(gate_bc_pool_ptr);
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr + align64(buffer_c_required_size(max_m, config_.intermediate_size)));

      up_bc_[expert_idx]->max_m = max_m;
      up_bc_[expert_idx]->set_data(up_bc_pool_ptr);
      up_bc_pool_ptr = (void*)((uintptr_t)up_bc_pool_ptr + align64(buffer_c_required_size(max_m, config_.intermediate_size)));

      down_ba_[expert_idx]->max_m = max_m;
      down_ba_[expert_idx]->set_data(down_ba_pool_ptr);
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr + align64(buffer_a_required_size(max_m, config_.intermediate_size)));

      down_bc_[expert_idx]->max_m = max_m;
      down_bc_[expert_idx]->set_data(down_bc_pool_ptr);
      down_bc_pool_ptr = (void*)((uintptr_t)down_bc_pool_ptr + align64(buffer_c_required_size(max_m, config_.hidden_size)));
    }

    // Pack input into BufferA for each activated expert
    void* gate_up_ba_pool_ptr = gate_up_ba_pool_;
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      size_t max_m = (qlen + M_STEP - 1) / M_STEP * M_STEP;
      gate_up_ba_[expert_idx]->max_m = max_m;
      gate_up_ba_[expert_idx]->set_data(gate_up_ba_pool_ptr);
      gate_up_ba_pool_ptr = (void*)((uintptr_t)gate_up_ba_pool_ptr + align64(buffer_a_required_size(max_m, config_.hidden_size)));
      gate_up_ba_[expert_idx]->from_mat(qlen, (ggml_bf16_t*)input, 0, 1);
    }

    // Gate + Up GEMM
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int) { T::config(); },
        [this, nth, qlen](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          derived()->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
          if (do_up) {
            up_bc_[expert_idx]->to_mat(qlen, m_local_up_output_ptr_[expert_idx], ith, nth);
          } else {
            gate_bc_[expert_idx]->to_mat(qlen, m_local_gate_output_ptr_[expert_idx], ith, nth);
          }
        },
        nullptr);

    // Activation
    apply_activation(activated_expert, nth, qlen);

    // Pack for down projection
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_mat(qlen, m_local_gate_output_ptr_[expert_idx], 0, 1);
        },
        nullptr);

    // Down GEMM
    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * activated_expert, [](int) { T::config(); },
        [this, nth, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          derived()->do_down_gemm(expert_idx, ith, nth, qlen);
          down_bc_[expert_idx]->to_mat(qlen, m_local_down_output_ptr_[expert_idx], ith, nth);
        },
        nullptr);

    // Weighted sum.
    for (int e = 0; e < config_.hidden_size; e += 16) {
      __m512 x0 = _mm512_setzero_ps();
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[j])) continue;
        __m512 weight = _mm512_set1_ps(weights[j]);
        __m512 d0 = avx512::load_16xbf16_to_fp32(
            m_local_down_output_ptr_[expert_ids[j]] + m_local_pos_[0][j] * config_.hidden_size + e);
        x0 = _mm512_fmadd_ps(d0, weight, x0);
      }
      _mm512_storeu_ps((float*)output + e, x0);
    }
  }

 protected:
  Derived* derived() { return static_cast<Derived*>(this); }
  const Derived* derived_const() const { return static_cast<const Derived*>(this); }

  void derived_init() {}

  // Buffer creation/size delegation (CRTP)
  size_t buffer_a_required_size(size_t m, size_t k) const { return derived_const()->buffer_a_required_size_impl(m, k); }
  size_t buffer_b_required_size(size_t n, size_t k) const { return derived_const()->buffer_b_required_size_impl(n, k); }
  size_t buffer_c_required_size(size_t m, size_t n) const { return derived_const()->buffer_c_required_size_impl(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a(size_t m, size_t k, void* data) const {
    return derived_const()->make_buffer_a_impl(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b(size_t n, size_t k, void* data) const {
    return derived_const()->make_buffer_b_impl(n, k, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c(size_t m, size_t n, void* data) const {
    return derived_const()->make_buffer_c_impl(m, n, data);
  }

  // SiLU activation.
  void apply_activation(int activated_expert, int nth, int qlen) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    auto fn = [this, nth](int task_id) {
      int expert_idx = m_expert_id_map_[task_id / nth];
      int ith = task_id % nth;
      auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
      for (int i = 0; i < m_local_num_[expert_idx]; i++) {
        ggml_bf16_t* gate_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
        ggml_bf16_t* up_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
        int j = n_start;
        for (; j + 16 <= n_end; j += 16) {
          __m512 gate_val = avx512::load_16xbf16_to_fp32(gate_ptr + j);
          __m512 up_val = avx512::load_16xbf16_to_fp32(up_ptr + j);
          __m512 result = config_.activation_type == 1
                               ? avx512::gelu_fn(gate_val, up_val)
                               : avx512::act_fn(gate_val, up_val);
          avx512::store_fp32_to_bf16(gate_ptr + j, result);
        }
        // Scalar tail
        for (; j < n_end; j++) {
          float g = GGML_BF16_TO_FP32(gate_ptr[j]);
          float u = GGML_BF16_TO_FP32(up_ptr[j]);
          float act = config_.activation_type == 1
                          ? 0.5f * g * (1.0f + erff(g * 0.7071067811865476f))
                          : g / (1.0f + expf(-g));
          gate_ptr[j] = GGML_FP32_TO_BF16(act * u);
        }
      }
    };

    if (activated_expert == 0) return;

    if (qlen < 10) {
      for (int task_id = 0; task_id < nth * activated_expert; task_id++) fn(task_id);
    } else {
      pool->do_work_stealing_job(nth * activated_expert, nullptr, fn, nullptr);
    }
  }
};

// ============================================================================
// TP_MOE specialization for AVX512_MOE_BASE derived classes
// ============================================================================

template <class T, class Derived>
class TP_MOE<AVX512_MOE_BASE<T, Derived>> : public TP_MOE_Common<AVX512_MOE_BASE<T, Derived>> {
 public:
  using TP_MOE_Common<AVX512_MOE_BASE<T, Derived>>::TP_MOE_Common;

  void load_weights() override { throw std::runtime_error("Not Implemented"); }

  void write_weight_scale_to_buffer(int gpu_tp_count, int gpu_experts_num,
                                    const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    throw std::runtime_error("Not Implemented");
  }

  void merge_results(int qlen, void* output, bool incremental) override {
    auto& config = this->config;
    auto& tp_count = this->tp_count;
    auto& local_output_numa = this->local_output_numa;
    auto& tp_configs = this->tp_configs;

    auto merge_fn = [this, output, incremental, &config, &tp_count, &local_output_numa, &tp_configs](int token_nth) {
      float* merge_to = local_output_numa[0] + token_nth * tp_configs[0].hidden_size;
      if (incremental) {
        // Convert BF16 output to FP32 and add.
        for (int e = 0; e < config.hidden_size; e += 16) {
          __m512 out = avx512::load_16xbf16_to_fp32((ggml_bf16_t*)output + token_nth * config.hidden_size + e);
          __m512 merged = _mm512_add_ps(_mm512_loadu_ps(merge_to + e), out);
          _mm512_storeu_ps(merge_to + e, merged);
        }
      }
      // Sum across TP parts
      for (int i = 1; i < tp_count; i++) {
        float* merge_from = local_output_numa[i] + token_nth * tp_configs[i].hidden_size;
        for (int e = 0; e < tp_configs[i].hidden_size; e += 16) {
          __m512 merged = _mm512_add_ps(_mm512_loadu_ps(merge_to + e), _mm512_loadu_ps(merge_from + e));
          _mm512_storeu_ps(merge_to + e, merged);
        }
      }
      // Convert FP32 -> BF16 output
      for (int e = 0; e < config.hidden_size; e += 16) {
        avx512::store_fp32_to_bf16(
            (ggml_bf16_t*)output + token_nth * config.hidden_size + e,
            _mm512_loadu_ps(merge_to + e));
      }
    };

    auto pool = config.pool;
    if (qlen < 10) {
      for (int i = 0; i < qlen; i++) merge_fn(i);
    } else {
      pool->do_work_stealing_job(qlen, nullptr, merge_fn, nullptr);
    }
  }

  void merge_results(int qlen, void* output) override { merge_results(qlen, output, false); }
};

#endif  // CPUINFER_OPERATOR_AVX512_MOE_BASE_H
