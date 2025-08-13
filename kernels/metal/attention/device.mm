#import <Metal/Metal.h>
#include <torch/torch.h>
#include <ATen/mps/MPSDevice.h>

int64_t get_device_attribute(int64_t attribute, int64_t device_id) {
  TORCH_CHECK(false, "get_device_attribute is not supported on Metal");
}

int64_t get_max_shared_memory_per_block_device_attribute(int64_t device_id) {
  // On macOS you can have multiple GPUs; fetch the N-th one.
  NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
  TORCH_CHECK(device_id >= 0 && device_id < (int64_t)all.count,
              "Invalid Metal device index");

  id<MTLDevice> dev = all[device_id];
  return static_cast<int64_t>(dev.maxThreadgroupMemoryLength);
}