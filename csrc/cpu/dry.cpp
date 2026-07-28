#include "libtorch_stable/torch_utils.h"

#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <vector>

namespace {

inline bool is_breaker(const std::vector<int64_t>& breaker_ids, int64_t token) {
  return std::find(breaker_ids.begin(), breaker_ids.end(), token) !=
         breaker_ids.end();
}

template <typename history_t>
std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor>
dry_scan_penalties_cpu_impl(const torch::stable::Tensor& token_history_ids,
                            const torch::stable::Tensor& token_history_lens,
                            const torch::stable::Tensor& dry_multiplier,
                            const torch::stable::Tensor& allowed_lengths,
                            const torch::stable::Tensor& sequence_breakers_ids,
                            const torch::stable::Tensor& ranges,
                            const torch::stable::Tensor& max_ngram,
                            const torch::stable::Tensor& max_occurrences,
                            const torch::stable::Tensor& early_exit_match_len,
                            int64_t vocab_size) {
  STD_TORCH_CHECK(token_history_ids.device().is_cpu(),
                  "token_history_ids must be on CPU");
  STD_TORCH_CHECK(token_history_lens.device().is_cpu(),
                  "token_history_lens must be on CPU");
  STD_TORCH_CHECK(dry_multiplier.device().is_cpu(),
                  "dry_multiplier must be on CPU");
  STD_TORCH_CHECK(allowed_lengths.device().is_cpu(),
                  "allowed_lengths must be on CPU");
  STD_TORCH_CHECK(sequence_breakers_ids.device().is_cpu(),
                  "sequence_breakers_ids must be on CPU");
  STD_TORCH_CHECK(ranges.device().is_cpu(), "ranges must be on CPU");
  STD_TORCH_CHECK(max_ngram.device().is_cpu(), "max_ngram must be on CPU");
  STD_TORCH_CHECK(max_occurrences.device().is_cpu(),
                  "max_occurrences must be on CPU");
  STD_TORCH_CHECK(early_exit_match_len.device().is_cpu(),
                  "early_exit_match_len must be on CPU");

  auto history = torch::stable::contiguous(token_history_ids);
  auto history_lens_c = torch::stable::contiguous(token_history_lens);
  auto dry_multiplier_c = torch::stable::contiguous(dry_multiplier);
  auto allowed_lengths_c = torch::stable::contiguous(allowed_lengths);
  auto breakers_c = torch::stable::contiguous(sequence_breakers_ids);
  auto ranges_c = torch::stable::contiguous(ranges);
  auto max_ngram_c = torch::stable::contiguous(max_ngram);
  auto max_occurrences_c = torch::stable::contiguous(max_occurrences);
  auto early_exit_c = torch::stable::contiguous(early_exit_match_len);

  const auto batch_size = history.size(0);
  const auto max_history_len = history.size(1);
  const auto max_breakers = breakers_c.dim() > 1 ? breakers_c.size(1) : 0;

  const auto* history_ptr = history.const_data_ptr<history_t>();
  const auto* history_lens_ptr = history_lens_c.const_data_ptr<int32_t>();
  const auto* dry_multiplier_ptr = dry_multiplier_c.const_data_ptr<float>();
  const auto* allowed_lengths_ptr = allowed_lengths_c.const_data_ptr<int32_t>();
  const auto* breakers_ptr = breakers_c.const_data_ptr<int64_t>();
  const auto* ranges_ptr = ranges_c.const_data_ptr<int32_t>();
  const auto* max_ngram_ptr = max_ngram_c.const_data_ptr<int32_t>();
  const auto* max_occurrences_ptr = max_occurrences_c.const_data_ptr<int32_t>();
  const auto* early_exit_ptr = early_exit_c.const_data_ptr<int32_t>();

  const auto history_at = [&](int64_t row, int64_t col) -> int64_t {
    return static_cast<int64_t>(history_ptr[row * max_history_len + col]);
  };

  std::vector<int64_t> row_indices;
  std::vector<int64_t> token_indices;
  std::vector<int64_t> match_lens;
  row_indices.reserve(batch_size * 4);
  token_indices.reserve(batch_size * 4);
  match_lens.reserve(batch_size * 4);

  for (int64_t row = 0; row < batch_size; ++row) {
    if (dry_multiplier_ptr[row] == 0.0f) {
      continue;
    }

    const int64_t history_len =
        std::min<int64_t>(history_lens_ptr[row], max_history_len);
    if (history_len < 2) {
      continue;
    }

    std::vector<int64_t> breaker_ids;
    breaker_ids.reserve(max_breakers);
    for (int64_t j = 0; j < max_breakers; ++j) {
      const int64_t token = breakers_ptr[row * max_breakers + j];
      if (token != vocab_size) {
        breaker_ids.push_back(token);
      }
    }

    const int64_t last_token = history_at(row, history_len - 1);
    if (is_breaker(breaker_ids, last_token)) {
      continue;
    }

    const int64_t range_limit = ranges_ptr[row];
    const int64_t start_idx =
        range_limit > 0 ? std::max<int64_t>(0, history_len - range_limit) : 0;

    int64_t curr_max_ngram = -1;
    const int64_t ngram_cap =
        std::min<int64_t>(history_len - start_idx, max_ngram_ptr[row] + 1);
    for (int64_t ngram_idx = 0; ngram_idx < ngram_cap; ++ngram_idx) {
      if (is_breaker(breaker_ids,
                     history_at(row, history_len - ngram_idx - 1))) {
        break;
      }
      curr_max_ngram = ngram_idx;
    }

    if (curr_max_ngram <= allowed_lengths_ptr[row]) {
      continue;
    }

    std::vector<int64_t> endpoint_indexes;
    endpoint_indexes.reserve(max_occurrences_ptr[row]);
    for (int64_t idx = start_idx; idx < history_len - 1; ++idx) {
      if (history_at(row, idx) == last_token) {
        endpoint_indexes.push_back(idx);
      }
    }
    if (endpoint_indexes.empty()) {
      continue;
    }
    if (static_cast<int64_t>(endpoint_indexes.size()) >
        max_occurrences_ptr[row]) {
      endpoint_indexes.erase(endpoint_indexes.begin(),
                             endpoint_indexes.end() - max_occurrences_ptr[row]);
    }

    std::vector<std::pair<int64_t, int64_t>> penalties;
    penalties.reserve(endpoint_indexes.size());
    for (auto it = endpoint_indexes.rbegin(); it != endpoint_indexes.rend();
         ++it) {
      const int64_t idx = *it;
      int64_t match_len = 0;
      const int64_t max_unwind =
          std::min<int64_t>(idx - start_idx, curr_max_ngram);
      for (int64_t unwind = 1; unwind <= max_unwind; ++unwind) {
        const int64_t candidate_tok = history_at(row, idx - unwind);
        if (is_breaker(breaker_ids, candidate_tok) ||
            candidate_tok != history_at(row, history_len - unwind - 1)) {
          break;
        }
        match_len = unwind;
      }

      if (match_len <= 0) {
        continue;
      }

      const int64_t next_token = history_at(row, idx + 1);
      const int64_t new_len = match_len + 1;
      auto found = std::find_if(
          penalties.begin(), penalties.end(),
          [&](const auto& entry) { return entry.first == next_token; });
      if (found == penalties.end()) {
        penalties.emplace_back(next_token, new_len);
      } else {
        found->second = std::max<int64_t>(found->second, new_len);
      }

      if (new_len >= early_exit_ptr[row]) {
        break;
      }
    }

    for (const auto& [token, len] : penalties) {
      row_indices.push_back(row);
      token_indices.push_back(token);
      match_lens.push_back(len);
    }
  }

  const torch::stable::Device cpu_device(torch::headeronly::DeviceType::CPU);
  const auto make_output = [&](const std::vector<int64_t>& values) {
    auto output = torch::stable::empty({static_cast<int64_t>(values.size())},
                                       torch::headeronly::ScalarType::Long,
                                       std::nullopt, cpu_device);
    if (!values.empty()) {
      std::memcpy(output.mutable_data_ptr<int64_t>(), values.data(),
                  values.size() * sizeof(int64_t));
    }
    return output;
  };

  return {make_output(row_indices), make_output(token_indices),
          make_output(match_lens)};
}

}  // namespace

std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor>
dry_scan_penalties_cpu(const torch::stable::Tensor& token_history_ids,
                       const torch::stable::Tensor& token_history_lens,
                       const torch::stable::Tensor& dry_multiplier,
                       const torch::stable::Tensor& allowed_lengths,
                       const torch::stable::Tensor& sequence_breakers_ids,
                       const torch::stable::Tensor& ranges,
                       const torch::stable::Tensor& max_ngram,
                       const torch::stable::Tensor& max_occurrences,
                       const torch::stable::Tensor& early_exit_match_len,
                       int64_t vocab_size) {
  if (token_history_ids.scalar_type() == torch::headeronly::ScalarType::Int) {
    return dry_scan_penalties_cpu_impl<int32_t>(
        token_history_ids, token_history_lens, dry_multiplier, allowed_lengths,
        sequence_breakers_ids, ranges, max_ngram, max_occurrences,
        early_exit_match_len, vocab_size);
  }
  if (token_history_ids.scalar_type() == torch::headeronly::ScalarType::Long) {
    return dry_scan_penalties_cpu_impl<int64_t>(
        token_history_ids, token_history_lens, dry_multiplier, allowed_lengths,
        sequence_breakers_ids, ranges, max_ngram, max_occurrences,
        early_exit_match_len, vocab_size);
  }
  STD_TORCH_CHECK(false, "token_history_ids must be int32 or int64");
}
