#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

namespace {

inline bool is_breaker(const std::vector<int64_t>& breaker_ids, int64_t token) {
  return std::find(breaker_ids.begin(), breaker_ids.end(), token) !=
         breaker_ids.end();
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> dry_scan_penalties_cpu(
    const torch::Tensor& token_history_ids,
    const torch::Tensor& token_history_lens,
    const torch::Tensor& dry_multiplier, const torch::Tensor& allowed_lengths,
    const torch::Tensor& sequence_breakers_ids, const torch::Tensor& ranges,
    const torch::Tensor& max_ngram, const torch::Tensor& max_occurrences,
    const torch::Tensor& early_exit_match_len, int64_t vocab_size) {
  TORCH_CHECK(token_history_ids.device().is_cpu(),
              "token_history_ids must be on CPU");
  TORCH_CHECK(token_history_lens.device().is_cpu(),
              "token_history_lens must be on CPU");
  TORCH_CHECK(dry_multiplier.device().is_cpu(),
              "dry_multiplier must be on CPU");
  TORCH_CHECK(allowed_lengths.device().is_cpu(),
              "allowed_lengths must be on CPU");
  TORCH_CHECK(sequence_breakers_ids.device().is_cpu(),
              "sequence_breakers_ids must be on CPU");
  TORCH_CHECK(ranges.device().is_cpu(), "ranges must be on CPU");
  TORCH_CHECK(max_ngram.device().is_cpu(), "max_ngram must be on CPU");
  TORCH_CHECK(max_occurrences.device().is_cpu(),
              "max_occurrences must be on CPU");
  TORCH_CHECK(early_exit_match_len.device().is_cpu(),
              "early_exit_match_len must be on CPU");

  auto history = token_history_ids.contiguous();
  auto history_lens_c = token_history_lens.contiguous();
  auto dry_multiplier_c = dry_multiplier.contiguous();
  auto allowed_lengths_c = allowed_lengths.contiguous();
  auto breakers_c = sequence_breakers_ids.contiguous();
  auto ranges_c = ranges.contiguous();
  auto max_ngram_c = max_ngram.contiguous();
  auto max_occurrences_c = max_occurrences.contiguous();
  auto early_exit_c = early_exit_match_len.contiguous();

  const auto batch_size = history.size(0);
  const auto max_history_len = history.size(1);
  const auto max_breakers = breakers_c.dim() > 1 ? breakers_c.size(1) : 0;

  auto history_acc = history.accessor<int64_t, 2>();
  auto history_lens_acc = history_lens_c.accessor<int32_t, 1>();
  auto dry_multiplier_acc = dry_multiplier_c.accessor<float, 1>();
  auto allowed_lengths_acc = allowed_lengths_c.accessor<int32_t, 1>();
  auto breakers_acc = breakers_c.accessor<int64_t, 2>();
  auto ranges_acc = ranges_c.accessor<int32_t, 1>();
  auto max_ngram_acc = max_ngram_c.accessor<int32_t, 1>();
  auto max_occurrences_acc = max_occurrences_c.accessor<int32_t, 1>();
  auto early_exit_acc = early_exit_c.accessor<int32_t, 1>();

  std::vector<int64_t> row_indices;
  std::vector<int64_t> token_indices;
  std::vector<int64_t> match_lens;
  row_indices.reserve(batch_size * 4);
  token_indices.reserve(batch_size * 4);
  match_lens.reserve(batch_size * 4);

  for (int64_t row = 0; row < batch_size; ++row) {
    if (dry_multiplier_acc[row] == 0.0f) {
      continue;
    }

    const int64_t history_len =
        std::min<int64_t>(history_lens_acc[row], max_history_len);
    if (history_len < 2) {
      continue;
    }

    std::vector<int64_t> breaker_ids;
    breaker_ids.reserve(max_breakers);
    for (int64_t j = 0; j < max_breakers; ++j) {
      const int64_t token = breakers_acc[row][j];
      if (token != vocab_size) {
        breaker_ids.push_back(token);
      }
    }

    const int64_t last_token = history_acc[row][history_len - 1];
    if (is_breaker(breaker_ids, last_token)) {
      continue;
    }

    const int64_t range_limit = ranges_acc[row];
    const int64_t start_idx =
        range_limit > 0 ? std::max<int64_t>(0, history_len - range_limit) : 0;

    int64_t curr_max_ngram = 0;
    const int64_t max_ngram_val = max_ngram_acc[row];
    const int64_t ngram_cap =
        std::min<int64_t>(history_len - start_idx, max_ngram_val + 1);
    for (curr_max_ngram = 0; curr_max_ngram < ngram_cap; ++curr_max_ngram) {
      if (is_breaker(breaker_ids,
                     history_acc[row][history_len - curr_max_ngram - 1])) {
        break;
      }
    }

    const int64_t min_ngram = allowed_lengths_acc[row];
    if (curr_max_ngram <= min_ngram) {
      continue;
    }

    std::vector<int64_t> endpoint_indexes;
    endpoint_indexes.reserve(max_occurrences_acc[row]);
    for (int64_t idx = start_idx; idx < history_len - 1; ++idx) {
      if (history_acc[row][idx] == last_token) {
        endpoint_indexes.push_back(idx);
      }
    }
    if (endpoint_indexes.empty()) {
      continue;
    }
    const int64_t max_occurrences_val = max_occurrences_acc[row];
    if (static_cast<int64_t>(endpoint_indexes.size()) > max_occurrences_val) {
      endpoint_indexes.erase(endpoint_indexes.begin(),
                             endpoint_indexes.end() - max_occurrences_val);
    }

    std::vector<std::pair<int64_t, int64_t>> penalties;
    penalties.reserve(endpoint_indexes.size());
    const int64_t early_exit_val = early_exit_acc[row];
    for (auto it = endpoint_indexes.rbegin(); it != endpoint_indexes.rend();
         ++it) {
      const int64_t idx = *it;
      int64_t match_len = 0;
      const int64_t max_unwind =
          std::min<int64_t>(idx - start_idx, curr_max_ngram);
      for (int64_t unwind = 1; unwind <= max_unwind; ++unwind) {
        const int64_t candidate_tok = history_acc[row][idx - unwind];
        if (is_breaker(breaker_ids, candidate_tok)) {
          break;
        }
        if (candidate_tok != history_acc[row][history_len - unwind - 1]) {
          break;
        }
        match_len = unwind;
      }

      if (match_len <= 0) {
        continue;
      }

      const int64_t next_token = history_acc[row][idx + 1];
      const int64_t new_len = match_len + 1;
      auto found = std::find_if(
          penalties.begin(), penalties.end(),
          [&](const auto& entry) { return entry.first == next_token; });
      if (found == penalties.end()) {
        penalties.emplace_back(next_token, new_len);
      } else {
        found->second = std::max<int64_t>(found->second, new_len);
      }

      if (new_len >= early_exit_val) {
        break;
      }
    }

    for (const auto& [token, len] : penalties) {
      row_indices.push_back(row);
      token_indices.push_back(token);
      match_lens.push_back(len);
    }
  }

  auto long_opts =
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
  if (row_indices.empty()) {
    auto empty = torch::empty({0}, long_opts);
    return {empty, empty.clone(), empty.clone()};
  }

  return {
      torch::tensor(row_indices, long_opts),
      torch::tensor(token_indices, long_opts),
      torch::tensor(match_lens, long_opts),
  };
}
