#include <cuda_fp16.h>
#include "coop_autotune.cuh"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <vector>

#include "../util.cuh"
#include "../util.h"

// #define CACHEDEBUG 1

namespace {

constexpr uint64_t COOP_AUTOTUNE_VERSION = 3;
constexpr char DISK_CACHE_MAGIC[8] = {'E', 'X', '3', 'A', 'T', 'U', 'N', 'E'};
constexpr uint32_t DISK_CACHE_FORMAT = 1;

struct ExpandedCandidate {
  void* kernel;
  int block_dim;
  int num_sms;
  int concurrency;
  int tag;
  float latency;
  std::vector<float> samples;
};

struct DiskCacheHeader {
  char magic[8];
  uint32_t format;
  uint32_t record_size;
};

struct DiskCacheRecordV1 {
  uint64_t hash;
  int32_t tag;
  int32_t block_dim;
  int32_t num_sms;
  int32_t concurrency;
  uint32_t reserved0;
  uint32_t reserved1;
};

std::map<uint64_t, CoopAutotuneLaunch> launch_cache;

std::set<std::tuple<int, void*, size_t>> attr_set;

std::mutex disk_mutex;
bool disk_cache_loaded = false;
std::map<uint64_t, DiskCacheRecordV1> disk_cache;

uint64_t salt_hash(uint64_t hash) {
  hash ^= COOP_AUTOTUNE_VERSION;
  hash *= 1099511628211ull;
  return hash;
}

std::filesystem::path disk_cache_path() {
  const char* override_path = std::getenv("EXLLAMAV3_TUNE_CACHE");
  if (override_path && override_path[0]) {
    std::filesystem::path path = std::filesystem::path(override_path);
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec))
      return path / "coop_autotune_v1.bin";
    return path;
  }

  std::filesystem::path base;
#ifdef _WIN32
  const char* local_app_data = std::getenv("LOCALAPPDATA");
  if (local_app_data && local_app_data[0])
    base = std::filesystem::path(local_app_data);
  else {
    const char* user_profile = std::getenv("USERPROFILE");
    if (!user_profile || !user_profile[0]) return {};
    std::filesystem::path user_path = std::filesystem::path(user_profile);
    std::filesystem::path local_path = user_path / "AppData" / "Local";
    std::error_code ec;
    if (std::filesystem::exists(local_path, ec))
      base = local_path;
    else
      base = user_path;
  }

  return base / "exllamav3" / "autotune" / "coop_autotune_v1.bin";
#else
  const char* xdg = std::getenv("XDG_CACHE_HOME");
  if (xdg && xdg[0])
    base = std::filesystem::path(xdg);
  else {
    const char* home = std::getenv("HOME");
    if (!home || !home[0]) return {};
    base = std::filesystem::path(home) / ".cache";
  }

  return base / "exllamav3" / "autotune" / "coop_autotune_v1.bin";
#endif
}

bool read_disk_header(std::ifstream& in) {
  DiskCacheHeader header;
  in.read((char*)&header, sizeof(header));
  if (!in) return false;
  if (std::memcmp(header.magic, DISK_CACHE_MAGIC, sizeof(header.magic)) != 0)
    return false;
  if (header.format != DISK_CACHE_FORMAT) return false;
  if (header.record_size < sizeof(DiskCacheRecordV1)) return false;
  return true;
}

void load_disk_cache_locked() {
  if (disk_cache_loaded) return;
  disk_cache_loaded = true;

  std::filesystem::path path = disk_cache_path();
  if (path.empty() || !std::filesystem::exists(path)) return;

  std::ifstream in(path, std::ios::binary);
  if (!in || !read_disk_header(in)) return;

  DiskCacheHeader header;
  in.seekg(0, std::ios::beg);
  in.read((char*)&header, sizeof(header));
  if (!in) return;

  std::vector<char> record_bytes(header.record_size);
  while (in.read(record_bytes.data(), header.record_size)) {
    DiskCacheRecordV1 record;
    std::memcpy(&record, record_bytes.data(), sizeof(record));
    disk_cache[record.hash] = record;
  }

#ifdef CACHEDEBUG
  printf("coop_autotune cache loaded: path=%s records=%zu\n",
         path.string().c_str(), disk_cache.size());
#endif
}

void append_disk_cache_locked(const DiskCacheRecordV1& record) {
  std::filesystem::path path = disk_cache_path();
  if (path.empty()) return;

  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  if (ec) return;

  bool write_header = !std::filesystem::exists(path) ||
                      std::filesystem::file_size(path, ec) == 0;
  std::ofstream out(path, std::ios::binary | std::ios::app);
  if (!out) return;

  if (write_header) {
    DiskCacheHeader header;
    std::memcpy(header.magic, DISK_CACHE_MAGIC, sizeof(header.magic));
    header.format = DISK_CACHE_FORMAT;
    header.record_size = sizeof(DiskCacheRecordV1);
    out.write((const char*)&header, sizeof(header));
  }

  out.write((const char*)&record, sizeof(record));
}

bool launch_from_disk_cache(
    uint64_t hash, const std::vector<CoopAutotuneCandidate>& candidates,
    CoopAutotuneLaunch* launch_config) {
  std::lock_guard<std::mutex> lock(disk_mutex);
  load_disk_cache_locked();

  auto lookup = disk_cache.find(hash);
  if (lookup == disk_cache.end()) return false;

  const DiskCacheRecordV1& record = lookup->second;
#ifdef CACHEDEBUG
  printf(
      "coop_autotune cache lookup: hash=%016llx tag=%d block_dim=%d num_sms=%d "
      "concurrency=%d\n",
      (unsigned long long)record.hash, record.tag, record.block_dim,
      record.num_sms, record.concurrency);
#endif

  for (const CoopAutotuneCandidate& candidate : candidates) {
    if (candidate.tag != record.tag) continue;
    if (candidate.block_dim != record.block_dim) continue;
    if (record.num_sms < 1 || record.num_sms > candidate.max_num_sms) continue;

    int max_concurrency = MAX(candidate.max_concurrency, 1);
    int total_sms =
        candidate.total_sms > 0 ? candidate.total_sms : candidate.max_num_sms;
    int expected_concurrency =
        MAX(MIN(total_sms / record.num_sms, max_concurrency), 1);
    if (record.concurrency != expected_concurrency) continue;

    *launch_config = {candidate.kernel, record.block_dim, record.num_sms,
                      record.concurrency, record.tag};
#ifdef CACHEDEBUG
    printf(
        "coop_autotune cache hit: hash=%016llx tag=%d block_dim=%d num_sms=%d "
        "concurrency=%d\n",
        (unsigned long long)record.hash, launch_config->tag,
        launch_config->block_dim, launch_config->num_sms,
        launch_config->concurrency);
#endif
    return true;
  }

#ifdef CACHEDEBUG
  printf(
      "coop_autotune cache rejected: hash=%016llx tag=%d block_dim=%d "
      "num_sms=%d concurrency=%d\n",
      (unsigned long long)record.hash, record.tag, record.block_dim,
      record.num_sms, record.concurrency);
#endif
  return false;
}

void store_disk_cache(uint64_t hash, const CoopAutotuneLaunch& launch_config) {
  DiskCacheRecordV1 record = {hash,
                              launch_config.tag,
                              launch_config.block_dim,
                              launch_config.num_sms,
                              launch_config.concurrency,
                              0,
                              0};

  std::lock_guard<std::mutex> lock(disk_mutex);
  load_disk_cache_locked();
  disk_cache[hash] = record;
  append_disk_cache_locked(record);

#ifdef CACHEDEBUG
  printf(
      "coop_autotune cache store: hash=%016llx tag=%d block_dim=%d num_sms=%d "
      "concurrency=%d\n",
      (unsigned long long)record.hash, record.tag, record.block_dim,
      record.num_sms, record.concurrency);
#endif
}

void set_kernel_attr_once(void* kernel, size_t smem) {
  int device;
  cuda_check(cudaGetDevice(&device));

  auto key = std::make_tuple(device, kernel, smem);
  if (attr_set.find(key) != attr_set.end()) return;

  cuda_check(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  attr_set.insert(key);
}

std::mutex thrash_mutex;
std::map<int, std::pair<void*, size_t>> thrash_buffers;

// Candidates in one autotune session share the same input tensors, so
// back-to-back timed launches keep each other's operands mutually warm in L2
// regardless of production's actual scattered, per-token-random expert access
// pattern.
void thrash_l2(cudaStream_t stream) {
  int device;
  cuda_check(cudaGetDevice(&device));

  void* buf;
  size_t size;
  {
    std::lock_guard<std::mutex> lock(thrash_mutex);
    auto it = thrash_buffers.find(device);
    if (it == thrash_buffers.end()) {
      int l2_bytes = 0;
      cuda_check(
          cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device));
      // 2x the reported L2 capacity to reliably evict prior residents
      // regardless of associativity/replacement-policy quirks; generous
      // fallback if the query fails
      size_t alloc_size =
          (size_t)(l2_bytes > 0 ? l2_bytes : 64 * 1024 * 1024) * 2;
      void* ptr;
      cuda_check(cudaMalloc(&ptr, alloc_size));
      thrash_buffers[device] = {ptr, alloc_size};
      buf = ptr;
      size = alloc_size;
    } else {
      buf = it->second.first;
      size = it->second.second;
    }
  }
  cuda_check(cudaMemsetAsync(buf, 0, size, stream));
}

float trimmed_mean(std::vector<float>& samples) {
  TORCH_CHECK(!samples.empty(), "CoopKernelAutotuner: no timing samples");
  std::sort(samples.begin(), samples.end());

  size_t trim = samples.size() / 4;
  size_t begin = trim;
  size_t end = samples.size() - trim;
  if (begin >= end) {
    begin = 0;
    end = samples.size();
  }

  double sum = 0.0;
  for (size_t i = begin; i < end; ++i) sum += samples[i];
  return (float)(sum / (double)(end - begin));
}

void measure_candidate_sample(ExpandedCandidate& candidate, void** kernel_args,
                              size_t smem, cudaStream_t stream, int repeats,
                              cudaEvent_t start, cudaEvent_t end) {
  thrash_l2(stream);
  cuda_check(cudaEventRecord(start, stream));
  for (int i = 0; i < repeats; ++i) {
    cuda_check(cudaLaunchCooperativeKernel(
        candidate.kernel, dim3(candidate.num_sms, 1, candidate.concurrency),
        candidate.block_dim, kernel_args, smem, stream));
  }
  cuda_check(cudaEventRecord(end, stream));
  cuda_check(cudaEventSynchronize(end));

  float ms = 0.0f;
  cuda_check(cudaEventElapsedTime(&ms, start, end));
  candidate.samples.push_back(ms / (float)repeats);
}

void keep_best(std::vector<ExpandedCandidate>& candidates, size_t limit) {
  for (ExpandedCandidate& candidate : candidates)
    candidate.latency = trimmed_mean(candidate.samples);

  std::sort(candidates.begin(), candidates.end(),
            [](const ExpandedCandidate& a, const ExpandedCandidate& b) {
              return a.latency < b.latency;
            });
  if (candidates.size() > limit) candidates.resize(limit);
}

void measure_stage(std::vector<ExpandedCandidate>& candidates,
                   void** kernel_args, size_t smem, cudaStream_t stream,
                   int rounds, int repeats, size_t keep, cudaEvent_t start,
                   cudaEvent_t end) {
  TORCH_CHECK(!candidates.empty(),
              "CoopKernelAutotuner: no candidates in stage");

  for (ExpandedCandidate& candidate : candidates) {
    candidate.samples.clear();
    candidate.samples.reserve(rounds);
    set_kernel_attr_once(candidate.kernel, smem);

    // One untimed launch avoids first-use effects from contaminating the first
    // measured round.
    cuda_check(cudaLaunchCooperativeKernel(
        candidate.kernel, dim3(candidate.num_sms, 1, candidate.concurrency),
        candidate.block_dim, kernel_args, smem, stream));
  }
  cuda_check(cudaStreamSynchronize(stream));

  std::vector<int> order(candidates.size());
  std::iota(order.begin(), order.end(), 0);

  for (int round = 0; round < rounds; ++round) {
    size_t n = order.size();
    size_t step = 2 * (size_t)round + 1;
    while (std::gcd(step, n) != 1) step += 2;
    size_t pos = ((uint64_t)round * 1103515245ull + 12345ull) % n;

    for (size_t i = 0; i < n; ++i) {
      measure_candidate_sample(candidates[order[pos]], kernel_args, smem,
                               stream, repeats, start, end);
      pos = (pos + step) % n;
    }
  }

  keep_best(candidates, keep);
}

CoopAutotuneLaunch tune(
    const std::vector<CoopAutotuneCandidate>& base_candidates,
    void** kernel_args, size_t smem, cudaStream_t stream, size_t numel_B) {
  std::vector<ExpandedCandidate> candidates;
  for (const CoopAutotuneCandidate& base : base_candidates) {
    int total_tiles = numel_B / base.block_dim / 16;
    int min_num_sms = MAX(2, MIN(total_tiles / 32, base.max_num_sms) /
                                 base.max_concurrency / 2 * 2);

    TORCH_CHECK(base.kernel, "CoopKernelAutotuner: null kernel candidate");
    TORCH_CHECK(base.block_dim > 0, "CoopKernelAutotuner: invalid block_dim");
    TORCH_CHECK(base.max_num_sms > 0,
                "CoopKernelAutotuner: invalid max_num_sms");
    int max_concurrency = MAX(base.max_concurrency, 1);
    int total_sms = base.total_sms > 0 ? base.total_sms : base.max_num_sms;

    if (max_concurrency > 1 || base.max_num_sms == 1) {
      int concurrency = MAX(MIN(total_sms, max_concurrency), 1);
      candidates.push_back(
          {base.kernel, base.block_dim, 1, concurrency, base.tag, 0.0f, {}});
    }

    for (int num_sms = min_num_sms; num_sms <= base.max_num_sms * 85 / 100;
         num_sms += 2) {
      int concurrency = MAX(MIN(total_sms / num_sms, max_concurrency), 1);
      candidates.push_back({base.kernel,
                            base.block_dim,
                            num_sms,
                            concurrency,
                            base.tag,
                            0.0f,
                            {}});
    }

    if (base.max_num_sms > 1) {
      int concurrency =
          MAX(MIN(total_sms / base.max_num_sms, max_concurrency), 1);
      candidates.push_back({base.kernel,
                            base.block_dim,
                            base.max_num_sms,
                            concurrency,
                            base.tag,
                            0.0f,
                            {}});
    }
  }
  TORCH_CHECK(!candidates.empty(), "CoopKernelAutotuner: no candidates");

  cudaEvent_t start;
  cudaEvent_t end;
  cuda_check(cudaEventCreate(&start));
  cuda_check(cudaEventCreate(&end));

  int repeats = 8;
  if (numel_B > 1e6) repeats = 5;
  if (numel_B > 1e7) repeats = 4;
  if (numel_B > 1e8) repeats = 3;
  if (numel_B > 2e8) repeats = 2;
  int max_rounds = 32;
  if (numel_B > 1e6) max_rounds = 12;
  if (numel_B > 1e7) max_rounds = 8;
  if (numel_B > 1e8) max_rounds = 5;
  if (numel_B > 2e8) max_rounds = 2;
  int max_cands = 7;
  if (numel_B > 1e6) max_cands = 5;
  if (numel_B > 1e7) max_cands = 4;
  if (numel_B > 1e8) max_cands = 3;
  if (numel_B > 2e8) max_cands = 2;

  if (candidates.size() > 1 && max_cands > 4)
    measure_stage(candidates, kernel_args, smem, stream, MIN(2, max_rounds),
                  repeats, MIN(6, max_cands), start, end);
  if (candidates.size() > 1 && max_cands > 1)
    measure_stage(candidates, kernel_args, smem, stream, MIN(8, max_rounds),
                  repeats, MIN(3, max_cands), start, end);
  if (candidates.size() > 1)
    measure_stage(candidates, kernel_args, smem, stream, MIN(14, max_rounds),
                  repeats, 1, start, end);

  cuda_check(cudaEventDestroy(start));
  cuda_check(cudaEventDestroy(end));

  const ExpandedCandidate& best = candidates[0];
  return {best.kernel, best.block_dim, best.num_sms, best.concurrency,
          best.tag};
}

}  // namespace

bool CoopKernelAutotuner::launch_locked(uint64_t hash, void** kernel_args,
                                        size_t smem, cudaStream_t stream,
                                        CoopAutotuneLaunch* out_launch_config) {
  CoopAutotuneLaunch launch_config;
  hash = salt_hash(hash);

  {
    auto lookup = launch_cache.find(hash);
    if (lookup == launch_cache.end()) return false;
    launch_config = lookup->second;
  }

  set_kernel_attr_once(launch_config.kernel, smem);
  cuda_check(cudaLaunchCooperativeKernel(
      launch_config.kernel,
      dim3(launch_config.num_sms, 1, launch_config.concurrency),
      launch_config.block_dim, kernel_args, smem, stream));

  if (out_launch_config) *out_launch_config = launch_config;
  return true;
}

CoopAutotuneLaunch CoopKernelAutotuner::launch(
    uint64_t hash, const std::vector<CoopAutotuneCandidate>& candidates,
    void** kernel_args, size_t smem, cudaStream_t stream, size_t numel_B) {
  CoopAutotuneLaunch launch_config;
  uint64_t salted_hash = salt_hash(hash);

  if (launch_locked(hash, kernel_args, smem, stream, &launch_config))
    return launch_config;

  if (launch_from_disk_cache(salted_hash, candidates, &launch_config)) {
    launch_cache[salted_hash] = launch_config;
    set_kernel_attr_once(launch_config.kernel, smem);
    cuda_check(cudaLaunchCooperativeKernel(
        launch_config.kernel,
        dim3(launch_config.num_sms, 1, launch_config.concurrency),
        launch_config.block_dim, kernel_args, smem, stream));
    return launch_config;
  }

  {
    launch_config = tune(candidates, kernel_args, smem, stream, numel_B);
    auto inserted = launch_cache.emplace(salted_hash, launch_config);
    launch_config = inserted.first->second;
  }
  store_disk_cache(salted_hash, launch_config);

  set_kernel_attr_once(launch_config.kernel, smem);
  cuda_check(cudaLaunchCooperativeKernel(
      launch_config.kernel,
      dim3(launch_config.num_sms, 1, launch_config.concurrency),
      launch_config.block_dim, kernel_args, smem, stream));

  return launch_config;
}
