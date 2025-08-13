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

namespace {
struct GatedParams {
  uint32_t d;
  uint32_t num_tokens;
  uint32_t op_id;      // 0..4
  float threshold;     // fatrelu
};

struct ElemParams {
  uint32_t d;
  uint32_t num_tokens;
  uint32_t op_id;      // 0..2
};

static id<MTLLibrary> loadActivationLibrary(id<MTLDevice> device) {
  std::string moduleDir = getModuleDirectory();
  std::string metallibPath = moduleDir + "/" + METALLIB_PATH;
  NSString *metallibPathStr = [NSString stringWithUTF8String:metallibPath.c_str()];
  NSURL *metallibURL = [NSURL fileURLWithPath:metallibPathStr];
  NSError *error = nil;
  id<MTLLibrary> lib = [device newLibraryWithURL:metallibURL error:&error];
  if (!lib) {
    NSLog(@"[activation.mm] Failed to load Metal library at %@: %@", metallibPathStr, error.localizedDescription);
  }
  return lib;
}

static void dispatch_gated(torch::Tensor &out, const torch::Tensor &input, uint32_t op_id, float threshold) {
  using at::mps::MPSStream;
  TORCH_CHECK(out.device().is_mps() && input.device().is_mps(), "tensors must be on MPS");
  TORCH_CHECK(input.dim() >= 1 && out.dim() == input.dim() && input.size(-1) % 2 == 0, "invalid shapes");
  const int64_t d = input.size(-1) / 2;
  const int64_t num_tokens = input.numel() / input.size(-1);
  // For exact numerical match with PyTorch native, use ATen computations for now (all dtypes)
  auto x = input.narrow(input.dim() - 1, 0, d);
  auto y = input.narrow(input.dim() - 1, d, d);
  at::Tensor z;
  switch (op_id) {
    case 0: z = at::silu(x) * y; break;                      // silu_and_mul
    case 1: z = x * at::silu(y); break;                      // mul_and_silu
    case 2: {                                                // gelu (none) via ATen for exact parity
      auto gx = at::gelu(x, "none");
      z = gx * y;
      break; }
    case 3: {                                                // gelu (tanh) via ATen for exact parity
      auto gx = at::gelu(x, "tanh");
      z = gx * y;
      break; }
    case 4: z = at::where(x > threshold, x, at::zeros_like(x)) * y; break;   // fatrelu
    default: TORCH_CHECK(false, "invalid op_id");
  }
  out.copy_(z);
}

static void dispatch_elem(torch::Tensor &out, const torch::Tensor &input, uint32_t op_id) {
  using at::mps::MPSStream;
  TORCH_CHECK(out.device().is_mps() && input.device().is_mps(), "tensors must be on MPS");
  const int64_t d = input.size(-1);
  const int64_t num_tokens = input.numel() / d;

  // For exact parity with forward_native: compute NewGELU using the same formula and dtype
  if (op_id == 0) {
    // Match forward_native exactly: 0.5 * x * (1 + tanh(c * (x + 0.044715 * pow(x,3))))
    const double c = std::sqrt(2.0 / M_PI);
    auto x = input;
    auto inner = c * (x + 0.044715 * at::pow(x, 3.0));
    auto z = 0.5 * x * (1 + at::tanh(inner));
    out.copy_(z);
    return;
  }

  if (input.scalar_type() == at::kBFloat16 || out.scalar_type() == at::kBFloat16) {
    at::Tensor z;
    switch (op_id) {
      case 0: z = 0.5 * input * (1 + at::tanh(0.79788456 * (input + 0.044715 * input * input * input))); break;
      case 1: z = 0.5 * input * (1 + at::tanh((input * 0.79788456) * (1 + input * 0.044715 * input))); break;
      case 2: z = input * at::sigmoid(1.702 * input); break;
      default: TORCH_CHECK(false, "invalid op_id");
    }
    out.copy_(z);
    return;
  }

  @autoreleasepool {
    MPSStream *stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(stream, "Failed to get MPS stream");
    id<MTLDevice> device = stream->device();
    id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
    TORCH_CHECK(cmdBuf, "Failed to get command buffer");
    id<MTLLibrary> lib = loadActivationLibrary(device);
    TORCH_CHECK(lib, "Missing activation metallib");

    NSString *fnName = nil;
    if (input.scalar_type() == at::kFloat) fnName = @"activation_elem_float";
    else if (input.scalar_type() == at::kHalf) fnName = @"activation_elem_half";
    else TORCH_CHECK(false, "Unsupported dtype for elementwise activation");

    NSError *error = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:fnName];
    TORCH_CHECK(fn, "Missing Metal function");
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
    TORCH_CHECK(pso, error.localizedDescription.UTF8String);

    ElemParams params{(uint32_t)d, (uint32_t)num_tokens, op_id};

    dispatch_queue_t q = stream->queue();
    dispatch_sync(q, ^{
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      TORCH_CHECK(enc, "Failed to create compute encoder");
      [enc setComputePipelineState:pso];
      [enc setBuffer:getMTLBufferStorage(input)
              offset:input.storage_offset() * input.element_size()
             atIndex:0];
      [enc setBuffer:getMTLBufferStorage(out)
              offset:out.storage_offset() * out.element_size()
             atIndex:1];
      id<MTLBuffer> paramBuf = [device newBufferWithBytes:&params length:sizeof(ElemParams) options:MTLResourceStorageModeShared];
      [enc setBuffer:paramBuf offset:0 atIndex:2];

      uint32_t total = (uint32_t)(num_tokens * d);
      uint32_t tpt = std::min<uint32_t>(256, total);
      MTLSize tg = MTLSizeMake(tpt, 1, 1);
      uint32_t groups = (total + tpt - 1) / tpt;
      MTLSize grid = MTLSizeMake(groups * tpt, 1, 1);
      [enc dispatchThreads:grid threadsPerThreadgroup:tg];
      [enc endEncoding];
      stream->synchronize(at::mps::SyncType::COMMIT);
    });
  }
}
} // namespace

void silu_and_mul(torch::Tensor &out, torch::Tensor &input) {
  dispatch_gated(out, input, /*op_id=*/0, /*threshold=*/0.0f);
}
void mul_and_silu(torch::Tensor &out, torch::Tensor &input) {
  dispatch_gated(out, input, /*op_id=*/1, /*threshold=*/0.0f);
}
void gelu_and_mul(torch::Tensor &out, torch::Tensor &input) {
  dispatch_gated(out, input, /*op_id=*/2, /*threshold=*/0.0f);
}
void gelu_tanh_and_mul(torch::Tensor &out, torch::Tensor &input) {
  dispatch_gated(out, input, /*op_id=*/3, /*threshold=*/0.0f);
}
void fatrelu_and_mul(torch::Tensor &out, torch::Tensor &input, double threshold) {
  dispatch_gated(out, input, /*op_id=*/4, /*threshold=*/(float)threshold);
}
void gelu_new(torch::Tensor &out, torch::Tensor &input) {
  dispatch_elem(out, input, /*op_id=*/0);
}
void gelu_fast(torch::Tensor &out, torch::Tensor &input) {
  dispatch_elem(out, input, /*op_id=*/1);
}
void gelu_quick(torch::Tensor &out, torch::Tensor &input) {
  dispatch_elem(out, input, /*op_id=*/2);
}


