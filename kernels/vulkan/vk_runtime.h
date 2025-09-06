#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "include/vulkan/vulkan.h"

namespace aphrodite {
namespace vkrt {

struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
};

struct Pipeline {
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout dsetLayout = VK_NULL_HANDLE;
  VkDescriptorPool dpool = VK_NULL_HANDLE;
  VkDescriptorSet dset = VK_NULL_HANDLE;
  VkShaderModule shader = VK_NULL_HANDLE;
};

struct Context {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  uint32_t queueFamily = 0;
  VkQueue queue = VK_NULL_HANDLE;
  VkCommandPool cmdPool = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  std::unordered_map<std::string, Pipeline> pipelines;
};

Context* get();
void init_once();
void shutdown();

// Compile-time embedded SPIR-V blobs loader
struct SpirvBlob { const uint32_t* data; size_t size; };
const SpirvBlob* get_spirv(const std::string& name);
const char* get_spv_dir();

// Helpers
Buffer create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                     VkMemoryPropertyFlags props);
void destroy_buffer(Buffer& buf);
void bind_device_ptr(Buffer& buf, const void* device_ptr);

Pipeline& get_or_create_pipeline(const std::string& name,
                                 const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                                 size_t pushConstantSize);

void begin();
void submit_and_wait();

}  // namespace vkrt
}  // namespace aphrodite


