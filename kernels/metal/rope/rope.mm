#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <torch/torch.h>
#include <dlfcn.h>
#include <string>

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

static std::string getModuleDirectory() {
  Dl_info dl_info;
  if (dladdr((void *)getModuleDirectory, &dl_info)) {
    std::string path(dl_info.dli_fname);
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
      return path.substr(0, pos);
    }
  }
  return ".";
}

struct RotaryParams {
  uint32_t head_size;
  uint32_t rot_dim;
  uint32_t embed_dim;
  uint32_t total_heads;
  uint32_t num_tokens;
};

static id<MTLLibrary> loadMetalLibrary(id<MTLDevice> device) {
  std::string moduleDir = getModuleDirectory();
  std::string metallibPath = moduleDir + "/" + METALLIB_PATH;
  NSString *metallibPathStr = [NSString stringWithUTF8String:metallibPath.c_str()];
  NSURL *metallibURL = [NSURL fileURLWithPath:metallibPathStr];
  NSError *error = nil;
  id<MTLLibrary> lib = [device newLibraryWithURL:metallibURL error:&error];
  if (!lib) {
    NSLog(@"[rope.mm] Failed to load Metal library at %@: %@", metallibPathStr, error.localizedDescription);
  }
  return lib;
}

static void dispatch_rotary_kernel(torch::Tensor &tensor,
                                   const torch::Tensor &positions,
                                   const torch::Tensor &cos_sin_cache,
                                   bool is_neox) {
  using at::mps::MPSStream;
  MPSStream *stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream, "Failed to get MPS stream");
  id<MTLDevice> device = stream->device();
  id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
  TORCH_CHECK(cmdBuf, "Failed to get command buffer");
  id<MTLLibrary> lib = loadMetalLibrary(device);
  TORCH_CHECK(lib, "Missing metallib for rope");

  const auto dtype = tensor.scalar_type();
  NSString *fnName = nil;
  if (is_neox) {
    if (dtype == at::kFloat) fnName = @"rotary_neox_float";
    else if (dtype == at::kHalf) fnName = @"rotary_neox_half";
  } else {
    if (dtype == at::kFloat) fnName = @"rotary_gptj_float";
    else if (dtype == at::kHalf) fnName = @"rotary_gptj_half";
  }
  TORCH_CHECK(fnName, "Unsupported dtype for rope kernel or invalid mode");

  NSError *error = nil;
  id<MTLFunction> fn = [lib newFunctionWithName:fnName];
  TORCH_CHECK(fn, "Missing Metal function");
  id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
  TORCH_CHECK(pso, error.localizedDescription.UTF8String);

  const int64_t head_size = tensor.size(-1);
  const int64_t rot_dim = cos_sin_cache.size(-1);
  const int64_t embed_dim = rot_dim / 2;
  const int64_t total_heads = tensor.size(-2);
  const int64_t num_tokens = tensor.size(0);

  RotaryParams params{(uint32_t)head_size, (uint32_t)rot_dim, (uint32_t)embed_dim,
                      (uint32_t)total_heads, (uint32_t)num_tokens};

  dispatch_queue_t q = stream->queue();
  dispatch_sync(q, ^{
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    TORCH_CHECK(enc, "Failed to create compute encoder");
    [enc setComputePipelineState:pso];
    [enc setBuffer:getMTLBufferStorage(tensor)
            offset:tensor.storage_offset() * tensor.element_size()
           atIndex:0];
    [enc setBuffer:getMTLBufferStorage(positions)
            offset:positions.storage_offset() * positions.element_size()
           atIndex:1];
    [enc setBuffer:getMTLBufferStorage(cos_sin_cache)
            offset:cos_sin_cache.storage_offset() * cos_sin_cache.element_size()
           atIndex:2];
    id<MTLBuffer> paramBuf = [device newBufferWithBytes:&params length:sizeof(RotaryParams) options:MTLResourceStorageModeShared];
    [enc setBuffer:paramBuf offset:0 atIndex:3];

    uint32_t work = (uint32_t)(num_tokens * total_heads * embed_dim);
    uint32_t tpt = std::min<uint32_t>(256, work);
    MTLSize tg = MTLSizeMake(tpt, 1, 1);
    uint32_t groups = (work + tpt - 1) / tpt;
    MTLSize grid = MTLSizeMake(groups * tpt, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    stream->synchronize(at::mps::SyncType::COMMIT);
  });
}

void rotary_embedding(torch::Tensor& positions,
                      torch::Tensor& query,
                      c10::optional<torch::Tensor> key,
                      int64_t head_size,
                      torch::Tensor& cos_sin_cache,
                      bool is_neox) {
  TORCH_CHECK(query.device().is_mps(), "rotary_embedding (MPS): query must be on MPS");

  // BF16/Half fallback using ATen tensor ops for exact parity
  if (query.scalar_type() == at::kBFloat16 || query.scalar_type() == at::kHalf) {
    auto pos_flat = positions.flatten();
    const int64_t num_tokens = pos_flat.numel();
    auto cos_sin = cos_sin_cache.index_select(0, pos_flat.toType(at::kLong));
    auto chunks = cos_sin.chunk(2, /*dim=*/-1);
    auto cos = chunks[0];
    auto sin = chunks[1];
    const int64_t rot_dim = cos.size(-1) * 2;
    const int64_t embed_dim = cos.size(-1);

    auto q_shape = query.sizes().vec();
    auto q_view = query.view({num_tokens, -1, head_size});
    auto q_rot = q_view.narrow(-1, 0, rot_dim);
    auto q_pass = q_view.narrow(-1, rot_dim, head_size - rot_dim);
    auto cos_b = cos.unsqueeze(1).to(q_view.dtype());
    auto sin_b = sin.unsqueeze(1).to(q_view.dtype());
    at::Tensor q_rot_new;
    if (is_neox) {
      auto parts = q_rot.chunk(2, -1);
      auto x1 = parts[0];
      auto x2 = parts[1];
      auto o1 = x1 * cos_b - x2 * sin_b;
      auto o2 = x2 * cos_b + x1 * sin_b;
      q_rot_new = at::cat({o1, o2}, -1);
    } else {
      auto qr2 = q_rot.view({num_tokens, -1, embed_dim, 2});
      auto even = qr2.select(-1, 0);
      auto odd = qr2.select(-1, 1);
      auto new_even = even * cos_b - odd * sin_b;
      auto new_odd = odd * cos_b + even * sin_b;
      q_rot_new = at::stack({new_even, new_odd}, -1).reshape({num_tokens, -1, rot_dim});
    }
    auto q_new = at::cat({q_rot_new, q_pass}, -1).view(q_shape);
    query.copy_(q_new);

    if (key.has_value()) {
      auto k_shape = key->sizes().vec();
      auto k_view = key->view({num_tokens, -1, head_size});
      auto k_rot = k_view.narrow(-1, 0, rot_dim);
      auto k_pass = k_view.narrow(-1, rot_dim, head_size - rot_dim);
      at::Tensor k_rot_new;
      if (is_neox) {
        auto parts = k_rot.chunk(2, -1);
        auto x1 = parts[0];
        auto x2 = parts[1];
        auto o1 = x1 * cos_b - x2 * sin_b;
        auto o2 = x2 * cos_b + x1 * sin_b;
        k_rot_new = at::cat({o1, o2}, -1);
      } else {
        auto kr2 = k_rot.view({num_tokens, -1, embed_dim, 2});
        auto even = kr2.select(-1, 0);
        auto odd = kr2.select(-1, 1);
        auto new_even = even * cos_b - odd * sin_b;
        auto new_odd = odd * cos_b + even * sin_b;
        k_rot_new = at::stack({new_even, new_odd}, -1).reshape({num_tokens, -1, rot_dim});
      }
      auto k_new = at::cat({k_rot_new, k_pass}, -1).view(k_shape);
      key->copy_(k_new);
    }
    return;
  }

  // Reshape to [num_tokens, total_heads, head_size]
  const int64_t num_tokens = positions.numel();
  auto q_view = query.view({num_tokens, -1, head_size});

  // cos_sin_cache must be [max_pos, rot_dim]; ensure dtype float for kernel
  auto cache_f32 = cos_sin_cache.to(at::kFloat);

  // Dispatch on query using Metal
  dispatch_rotary_kernel(q_view, positions, cache_f32, is_neox);

  if (key.has_value()) {
    auto k_view = key->view({num_tokens, -1, head_size});
    if (k_view.scalar_type() == at::kBFloat16) {
      // Handled by early BF16 branch, nothing to do
    } else {
      dispatch_rotary_kernel(k_view, positions, cache_f32, is_neox);
    }
  }
}
