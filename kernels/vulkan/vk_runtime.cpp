#include "vk_runtime.h"

#include <torch/all.h>
#include <mutex>

namespace aphrodite {
namespace vkrt {

static Context g_ctx;
static std::once_flag g_once;

Context* get() { return &g_ctx; }

void init_once() {
  std::call_once(g_once, [] {
    // Minimal instance/device setup (no validation)
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    vkCreateInstance(&ici, nullptr, &g_ctx.instance);

    uint32_t physCount = 0;
    vkEnumeratePhysicalDevices(g_ctx.instance, &physCount, nullptr);
    std::vector<VkPhysicalDevice> phys(physCount);
    vkEnumeratePhysicalDevices(g_ctx.instance, &physCount, phys.data());
    g_ctx.phys = phys[0];

    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_ctx.phys, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(g_ctx.phys, &qfCount, qf.data());
    g_ctx.queueFamily = 0;
    for (uint32_t i = 0; i < qfCount; ++i) {
      if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_ctx.queueFamily = i; break; }
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo dq{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    dq.queueFamilyIndex = g_ctx.queueFamily;
    dq.queueCount = 1;
    dq.pQueuePriorities = &prio;

    // Enable 16-bit storage for fp16 buffers if supported (needed by shaders using float16_t in SSBOs)
    VkPhysicalDeviceFeatures2 feats2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDevice16BitStorageFeatures feats16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    feats2.pNext = &feats16;
    vkGetPhysicalDeviceFeatures2(g_ctx.phys, &feats2);

    VkPhysicalDevice16BitStorageFeatures enable16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    if (feats16.storageBuffer16BitAccess) {
      enable16.storageBuffer16BitAccess = VK_TRUE;
      // Enable other 16-bit accesses if available (harmless if unused)
      if (feats16.uniformAndStorageBuffer16BitAccess)
        enable16.uniformAndStorageBuffer16BitAccess = VK_TRUE;
      if (feats16.storagePushConstant16)
        enable16.storagePushConstant16 = VK_TRUE;
      if (feats16.storageInputOutput16)
        enable16.storageInputOutput16 = VK_TRUE;
    }

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &dq;
    dci.pNext = &enable16; // safe if all fields are VK_FALSE
    vkCreateDevice(g_ctx.phys, &dci, nullptr, &g_ctx.device);
    vkGetDeviceQueue(g_ctx.device, g_ctx.queueFamily, 0, &g_ctx.queue);

    VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cp.queueFamilyIndex = g_ctx.queueFamily;
    cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(g_ctx.device, &cp, nullptr, &g_ctx.cmdPool);

    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = g_ctx.cmdPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    vkAllocateCommandBuffers(g_ctx.device, &ai, &g_ctx.cmd);

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(g_ctx.device, &fci, nullptr, &g_ctx.fence);
  });
}

void shutdown() {
  auto& c = g_ctx;
  for (auto& kv : c.pipelines) {
    auto& p = kv.second;
    if (p.pipeline) vkDestroyPipeline(c.device, p.pipeline, nullptr);
    if (p.layout) vkDestroyPipelineLayout(c.device, p.layout, nullptr);
    if (p.dset) vkFreeDescriptorSets(c.device, p.dpool, 1, &p.dset);
    if (p.dpool) vkDestroyDescriptorPool(c.device, p.dpool, nullptr);
    if (p.dsetLayout) vkDestroyDescriptorSetLayout(c.device, p.dsetLayout, nullptr);
    if (p.shader) vkDestroyShaderModule(c.device, p.shader, nullptr);
  }
  if (c.fence) vkDestroyFence(c.device, c.fence, nullptr);
  if (c.cmdPool) vkDestroyCommandPool(c.device, c.cmdPool, nullptr);
  if (c.device) vkDestroyDevice(c.device, nullptr);
  if (c.instance) vkDestroyInstance(c.instance, nullptr);
  c = {};
}

static uint32_t find_mem_type(uint32_t typeBits, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp{};
  vkGetPhysicalDeviceMemoryProperties(g_ctx.phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
      return i;
  }
  return UINT32_MAX;
}

Buffer create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                     VkMemoryPropertyFlags props) {
  Buffer b{};
  b.size = size;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = usage;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  vkCreateBuffer(g_ctx.device, &bi, nullptr, &b.buffer);
  VkMemoryRequirements mr{};
  vkGetBufferMemoryRequirements(g_ctx.device, b.buffer, &mr);
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = mr.size;
  mai.memoryTypeIndex = find_mem_type(mr.memoryTypeBits, props);
  vkAllocateMemory(g_ctx.device, &mai, nullptr, &b.memory);
  vkBindBufferMemory(g_ctx.device, b.buffer, b.memory, 0);
  return b;
}

void destroy_buffer(Buffer& buf) {
  if (buf.buffer) vkDestroyBuffer(g_ctx.device, buf.buffer, nullptr);
  if (buf.memory) vkFreeMemory(g_ctx.device, buf.memory, nullptr);
  buf = {};
}

void bind_device_ptr(Buffer& /*buf*/, const void* /*device_ptr*/) {
  // TODO: Use VK_KHR_buffer_device_address and import CUDA memory. Placeholder.
}

// Load SPIR-V from build dir at runtime (simpler than embedding for now)
static std::unordered_map<std::string, std::vector<uint32_t>> g_spv_cache;
const SpirvBlob* get_spirv(const std::string& name) {
  auto it = g_spv_cache.find(name);
  if (it == g_spv_cache.end()) {
    const char* dir = getenv("APHRODITE_VK_SPV_DIR");
#ifdef APHRODITE_VK_SPV_DIR_DEFAULT
    if (!dir || !*dir) dir = APHRODITE_VK_SPV_DIR_DEFAULT;
#endif
    if (!dir || !*dir) return nullptr;
    std::string path = std::string(dir) + "/" + name + ".spv";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return nullptr;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> data((sz + 3) / 4);
    fread(data.data(), 1, sz, f);
    fclose(f);
    it = g_spv_cache.emplace(name, std::move(data)).first;
  }
  static SpirvBlob blob;
  blob.data = it->second.data();
  blob.size = it->second.size() * sizeof(uint32_t);
  return &blob;
}

const char* get_spv_dir() {
  return getenv("APHRODITE_VK_SPV_DIR");
}

Pipeline& get_or_create_pipeline(const std::string& name,
                                 const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                                 size_t pushConstantSize) {
  auto& c = g_ctx;
  auto it = c.pipelines.find(name);
  if (it != c.pipelines.end()) return it->second;
  auto& P = c.pipelines[name];
  // Create descriptor set layout
  VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dlci.bindingCount = static_cast<uint32_t>(bindings.size());
  dlci.pBindings = bindings.data();
  vkCreateDescriptorSetLayout(c.device, &dlci, nullptr, &P.dsetLayout);

  // Pipeline layout
  VkPushConstantRange pcr{};
  pcr.offset = 0;
  pcr.size = static_cast<uint32_t>(pushConstantSize);
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &P.dsetLayout;
  if (pushConstantSize) {
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
  }
  vkCreatePipelineLayout(c.device, &plci, nullptr, &P.layout);

  // Shader module
  const SpirvBlob* blob = get_spirv(name);
  TORCH_CHECK(blob != nullptr, "Missing SPIR-V for ", name);
  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = blob->size;
  smci.pCode = blob->data;
  vkCreateShaderModule(c.device, &smci, nullptr, &P.shader);

  VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pci.stage.module = P.shader;
  pci.stage.pName = "main";
  pci.layout = P.layout;
  vkCreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &pci, nullptr, &P.pipeline);

  // Descriptor pool and set
  VkDescriptorPoolSize sizes[2] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
  };
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 2;
  dpci.pPoolSizes = sizes;
  vkCreateDescriptorPool(c.device, &dpci, nullptr, &P.dpool);
  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = P.dpool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &P.dsetLayout;
  vkAllocateDescriptorSets(c.device, &dsai, &P.dset);
  return P;
}

void begin() {
  auto& c = g_ctx;
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(c.cmd, &bi);
}

void submit_and_wait() {
  auto& c = g_ctx;
  vkEndCommandBuffer(c.cmd);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &c.cmd;
  vkQueueSubmit(c.queue, 1, &si, c.fence);
  vkWaitForFences(c.device, 1, &c.fence, VK_TRUE, UINT64_MAX);
  vkResetFences(c.device, 1, &c.fence);
  vkResetCommandPool(c.device, c.cmdPool, 0);
}

}  // namespace vkrt
}  // namespace aphrodite


