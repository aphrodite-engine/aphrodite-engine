#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <torch/torch.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <string>
#include <vector>

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

// Helper function to get conversion kernel name
static std::string getConvertKernelName(torch::ScalarType src_dtype, torch::ScalarType dst_dtype) {
  std::string src_str, dst_str;
  
  auto dtype_to_string = [](torch::ScalarType dtype) -> std::string {
    switch (dtype) {
    case torch::kFloat: return "float";
    case torch::kHalf: return "half";
    case torch::kBFloat16: return "bfloat16_t";
    case torch::kUInt8: return "uchar";
    default: 
      TORCH_CHECK(false, "Unsupported dtype for convert_fp8: ", dtype);
    }
  };
  
  src_str = dtype_to_string(src_dtype);
  dst_str = dtype_to_string(dst_dtype);
  
  return "convert_fp8_" + src_str + "_to_" + dst_str;
}

void convert_fp8(torch::Tensor &dst_cache, torch::Tensor &src_cache,
                 const double scale, const std::string &kv_cache_dtype) {
  // Validate input tensors
  TORCH_CHECK(src_cache.device().is_mps() && dst_cache.device().is_mps(),
              "Both tensors must be on MPS device");
  TORCH_CHECK(src_cache.device() == dst_cache.device(),
              "Source and destination tensors must be on the same device");
  TORCH_CHECK(src_cache.numel() == dst_cache.numel(),
              "Source and destination tensors must have the same number of elements");
  TORCH_CHECK(src_cache.is_contiguous() && dst_cache.is_contiguous(),
              "Both tensors must be contiguous");

  const uint32_t num_elements = static_cast<uint32_t>(src_cache.numel());
  if (num_elements == 0) {
    return; // Nothing to convert
  }

  // Determine conversion kernel name
  std::string kernel_name = getConvertKernelName(src_cache.scalar_type(), dst_cache.scalar_type());

  @autoreleasepool {
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(stream, "Failed to get current MPS stream");

    id<MTLDevice> device = stream->device();
    id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
    TORCH_CHECK(cmdBuf, "Failed to get command buffer");

    // Load Metal library
    std::string moduleDir = getModuleDirectory();
    std::string metallibPath = moduleDir + "/" + METALLIB_PATH;
    NSString *metallibPathStr = [NSString stringWithUTF8String:metallibPath.c_str()];
    NSURL *metallibURL = [NSURL fileURLWithPath:metallibPathStr];
    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithURL:metallibURL error:&error];
    TORCH_CHECK(lib, "Failed to load Metal library at ", metallibPath, ": ",
                error ? error.localizedDescription.UTF8String : "unknown error");

    // Create kernel function
    NSString *kernelNameStr = [NSString stringWithUTF8String:kernel_name.c_str()];
    id<MTLFunction> fn = [lib newFunctionWithName:kernelNameStr];
    TORCH_CHECK(fn, "Failed to find Metal kernel function: ", kernel_name);

    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
    TORCH_CHECK(pso, "Failed to create compute pipeline state: ",
                error ? error.localizedDescription.UTF8String : "unknown error");

    dispatch_queue_t q = stream->queue();
    dispatch_sync(q, ^{
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      TORCH_CHECK(enc, "Failed to create compute encoder");

      [enc setComputePipelineState:pso];

      // Set buffers
      [enc setBuffer:getMTLBufferStorage(src_cache)
              offset:src_cache.storage_offset() * src_cache.element_size()
             atIndex:0];
      [enc setBuffer:getMTLBufferStorage(dst_cache)
              offset:dst_cache.storage_offset() * dst_cache.element_size()
             atIndex:1];

      // Set scale parameter
      float scale_f32 = static_cast<float>(scale);
      id<MTLBuffer> scaleBuf = [device newBufferWithBytes:&scale_f32
                                                   length:sizeof(float)
                                                  options:MTLResourceStorageModeShared];
      [enc setBuffer:scaleBuf offset:0 atIndex:2];

      // Set num_elements parameter
      id<MTLBuffer> numElementsBuf = [device newBufferWithBytes:&num_elements
                                                         length:sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];
      [enc setBuffer:numElementsBuf offset:0 atIndex:3];

      // Dispatch threads
      const uint32_t threads_per_threadgroup = std::min<uint32_t>(1024, num_elements);
      const uint32_t threadgroups = (num_elements + threads_per_threadgroup - 1) / threads_per_threadgroup;
      
      MTLSize threadsPerThreadgroup = MTLSizeMake(threads_per_threadgroup, 1, 1);
      MTLSize threadgroupsPerGrid = MTLSizeMake(threadgroups, 1, 1);

      [enc dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
      [enc endEncoding];
    });

    stream->synchronize(at::mps::SyncType::COMMIT);
  }
}