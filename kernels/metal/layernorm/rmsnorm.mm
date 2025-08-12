#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <ATen/ATen.h>
#include <torch/torch.h>
#import <ATen/mps/MPSDevice.h>
#import <ATen/mps/MPSStream.h>
#include <dlfcn.h>

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

static id<MTLLibrary> loadMetalLibrary(id<MTLDevice> device) {
  Dl_info dl_info;
  std::string moduleDir = ".";
  if (dladdr((void *)loadMetalLibrary, &dl_info)) {
    std::string path(dl_info.dli_fname);
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) moduleDir = path.substr(0, pos);
  }
  std::string metallibPath = moduleDir + "/" + std::string(METALLIB_PATH);
  NSString *metallibPathStr = [NSString stringWithUTF8String:metallibPath.c_str()];
  NSURL *metallibURL = [NSURL fileURLWithPath:metallibPathStr];
  NSError *error = nil;
  id<MTLLibrary> lib = [device newLibraryWithURL:metallibURL error:&error];
  if (!lib) {
    NSLog(@"[rmsnorm.mm] Failed to load Metal library at %@: %@", metallibPathStr, error.localizedDescription);
  }
  return lib;
}

static void launch_rmsnorm(const at::Tensor &input_f32,
                           const at::Tensor &weight_f32,
                           at::Tensor &output_f32,
                           uint32_t hidden_size,
                           float epsilon) {
  using at::mps::MPSStream;
  MPSStream *stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream, "Failed to get MPS stream");
  id<MTLDevice> device = stream->device();
  id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
  TORCH_CHECK(cmdBuf, "Failed to get command buffer");
  id<MTLLibrary> lib = loadMetalLibrary(device);
  TORCH_CHECK(lib, "Missing metallib for rmsnorm");

  NSError *error = nil;
  id<MTLFunction> fn = [lib newFunctionWithName:@"rmsnorm_kernel"];
  TORCH_CHECK(fn, "Missing rmsnorm_kernel in metallib");
  id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
  TORCH_CHECK(pso, error.localizedDescription.UTF8String);

  dispatch_queue_t q = stream->queue();
  uint32_t batch = (uint32_t)(input_f32.numel() / hidden_size);
  float eps = epsilon;

  dispatch_sync(q, ^{
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    TORCH_CHECK(enc, "Failed to create compute encoder");
    [enc setComputePipelineState:pso];
    [enc setBuffer:getMTLBufferStorage(input_f32)
            offset:input_f32.storage_offset() * input_f32.element_size()
           atIndex:0];
    [enc setBuffer:getMTLBufferStorage(weight_f32)
            offset:weight_f32.storage_offset() * weight_f32.element_size()
           atIndex:1];
    [enc setBuffer:getMTLBufferStorage(output_f32)
            offset:output_f32.storage_offset() * output_f32.element_size()
           atIndex:2];
    [enc setBytes:&hidden_size length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&eps length:sizeof(float) atIndex:4];

    NSUInteger w = pso.threadExecutionWidth;
    MTLSize tg = MTLSizeMake(w, 1, 1);
    MTLSize grid = MTLSizeMake(batch, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    stream->synchronize(at::mps::SyncType::COMMIT);
  });
}

void rms_norm(torch::Tensor &out, torch::Tensor &input, torch::Tensor &weight, double epsilon) {
  TORCH_CHECK(out.device().is_mps() && input.device().is_mps(), "tensors must be on MPS");
  const int64_t hidden_size = input.size(-1);
  const int64_t num_tokens = input.numel() / hidden_size;
  TORCH_CHECK(hidden_size > 0 && num_tokens >= 0, "invalid shapes");
  // Safe ATen fallback (no Metal dispatch) to avoid segfault while we validate kernels
  auto in2d = input.view({num_tokens, hidden_size}).to(at::kFloat);
  auto w_f32 = weight.to(in2d.options().dtype(at::kFloat));
  auto var = (in2d * in2d).mean(-1, /*keepdim*/ true);
  auto inv_rms = (var + static_cast<float>(epsilon)).rsqrt();
  auto out_f32 = in2d * inv_rms * w_f32;
  out.copy_(out_f32.to(out.scalar_type()));
}

void fused_add_rms_norm(torch::Tensor &input, torch::Tensor &residual, torch::Tensor &weight, double epsilon) {
  TORCH_CHECK(input.device().is_mps() && residual.device().is_mps(), "tensors must be on MPS");
  const int64_t hidden_size = input.size(-1);
  const int64_t num_tokens = input.numel() / hidden_size;
  residual.add_(input);
  auto res2d = residual.view({num_tokens, hidden_size}).to(at::kFloat);
  auto w_f32 = weight.to(res2d.options().dtype(at::kFloat));
  auto var = (res2d * res2d).mean(-1, /*keepdim*/ true);
  auto inv_rms = (var + static_cast<float>(epsilon)).rsqrt();
  auto out_f32 = res2d * inv_rms * w_f32;
  input.copy_(out_f32.to(input.scalar_type()));
}