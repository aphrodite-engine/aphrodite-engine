
/*
 * Copyright (C) 2021-2025 Valve Corporation
 * Copyright (C) 2021-2025 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This file is ***GENERATED***.  Do Not Edit.
 * See scripts/gen_profiles_solution.py for modifications.
 */

#ifndef VULKAN_PROFILES_H_
#define VULKAN_PROFILES_H_ 1

#define VPAPI_ATTR

#ifdef __cplusplus
    extern "C" {
#endif

#include <vulkan/vulkan.h>

#if defined(VK_VERSION_1_1) && \
    defined(VK_ANDROID_external_memory_android_hardware_buffer) && \
    defined(VK_EXT_queue_family_foreign) && \
    defined(VK_EXT_swapchain_colorspace) && \
    defined(VK_GOOGLE_display_timing) && \
    defined(VK_KHR_android_surface) && \
    defined(VK_KHR_create_renderpass2) && \
    defined(VK_KHR_dedicated_allocation) && \
    defined(VK_KHR_descriptor_update_template) && \
    defined(VK_KHR_driver_properties) && \
    defined(VK_KHR_external_fence) && \
    defined(VK_KHR_external_fence_capabilities) && \
    defined(VK_KHR_external_fence_fd) && \
    defined(VK_KHR_external_memory) && \
    defined(VK_KHR_external_memory_capabilities) && \
    defined(VK_KHR_external_semaphore) && \
    defined(VK_KHR_external_semaphore_capabilities) && \
    defined(VK_KHR_external_semaphore_fd) && \
    defined(VK_KHR_get_memory_requirements2) && \
    defined(VK_KHR_get_physical_device_properties2) && \
    defined(VK_KHR_get_surface_capabilities2) && \
    defined(VK_KHR_incremental_present) && \
    defined(VK_KHR_maintenance1) && \
    defined(VK_KHR_sampler_mirror_clamp_to_edge) && \
    defined(VK_KHR_storage_buffer_storage_class) && \
    defined(VK_KHR_surface) && \
    defined(VK_KHR_swapchain) && \
    defined(VK_KHR_variable_pointers)
#define VP_ANDROID_baseline_2022 1
#define VP_ANDROID_BASELINE_2022_NAME "VP_ANDROID_baseline_2022"
#define VP_ANDROID_BASELINE_2022_SPEC_VERSION 2
#define VP_ANDROID_BASELINE_2022_MIN_API_VERSION VK_MAKE_VERSION(1, 1, 106)
#endif

#if defined(VK_VERSION_1_3) && \
    defined(VP_ANDROID_baseline_2022) && \
    defined(VK_ANDROID_external_format_resolve) && \
    defined(VK_EXT_4444_formats) && \
    defined(VK_EXT_custom_border_color) && \
    defined(VK_EXT_device_memory_report) && \
    defined(VK_EXT_external_memory_acquire_unmodified) && \
    defined(VK_EXT_index_type_uint8) && \
    defined(VK_EXT_line_rasterization) && \
    defined(VK_EXT_load_store_op_none) && \
    defined(VK_EXT_primitive_topology_list_restart) && \
    defined(VK_EXT_primitives_generated_query) && \
    defined(VK_EXT_provoking_vertex) && \
    defined(VK_EXT_scalar_block_layout) && \
    defined(VK_EXT_surface_maintenance1) && \
    defined(VK_EXT_swapchain_maintenance1) && \
    defined(VK_GOOGLE_surfaceless_query) && \
    defined(VK_IMG_relaxed_line_rasterization) && \
    defined(VK_KHR_16bit_storage) && \
    defined(VK_KHR_maintenance5) && \
    defined(VK_KHR_shader_float16_int8) && \
    defined(VK_KHR_vertex_attribute_divisor)
#define VP_ANDROID_15_minimums 1
#define VP_ANDROID_15_MINIMUMS_NAME "VP_ANDROID_15_minimums"
#define VP_ANDROID_15_MINIMUMS_SPEC_VERSION 1
#define VP_ANDROID_15_MINIMUMS_MIN_API_VERSION VK_MAKE_VERSION(1, 3, 273)
#endif

#if defined(VK_VERSION_1_3) && \
    defined(VP_ANDROID_15_minimums) && \
    defined(VP_ANDROID_baseline_2022) && \
    defined(VK_EXT_host_image_copy) && \
    defined(VK_EXT_image_2d_view_of_3d) && \
    defined(VK_EXT_multisampled_render_to_single_sampled) && \
    defined(VK_EXT_pipeline_protected_access) && \
    defined(VK_EXT_pipeline_robustness) && \
    defined(VK_EXT_shader_stencil_export) && \
    defined(VK_EXT_transform_feedback) && \
    defined(VK_KHR_8bit_storage) && \
    defined(VK_KHR_load_store_op_none) && \
    defined(VK_KHR_maintenance6) && \
    defined(VK_KHR_map_memory2) && \
    defined(VK_KHR_shader_expect_assume) && \
    defined(VK_KHR_shader_float_controls2) && \
    defined(VK_KHR_shader_maximal_reconvergence) && \
    defined(VK_KHR_shader_subgroup_rotate) && \
    defined(VK_KHR_shader_subgroup_uniform_control_flow) && \
    defined(VK_KHR_swapchain_mutable_format)
#define VP_ANDROID_16_minimums 1
#define VP_ANDROID_16_MINIMUMS_NAME "VP_ANDROID_16_minimums"
#define VP_ANDROID_16_MINIMUMS_SPEC_VERSION 1
#define VP_ANDROID_16_MINIMUMS_MIN_API_VERSION VK_MAKE_VERSION(1, 3, 276)
#endif

#if defined(VK_VERSION_1_0) && \
    defined(VK_EXT_swapchain_colorspace) && \
    defined(VK_GOOGLE_display_timing) && \
    defined(VK_KHR_android_surface) && \
    defined(VK_KHR_dedicated_allocation) && \
    defined(VK_KHR_descriptor_update_template) && \
    defined(VK_KHR_external_fence) && \
    defined(VK_KHR_external_fence_capabilities) && \
    defined(VK_KHR_external_fence_fd) && \
    defined(VK_KHR_external_memory) && \
    defined(VK_KHR_external_memory_capabilities) && \
    defined(VK_KHR_external_semaphore) && \
    defined(VK_KHR_external_semaphore_capabilities) && \
    defined(VK_KHR_external_semaphore_fd) && \
    defined(VK_KHR_get_memory_requirements2) && \
    defined(VK_KHR_get_physical_device_properties2) && \
    defined(VK_KHR_get_surface_capabilities2) && \
    defined(VK_KHR_incremental_present) && \
    defined(VK_KHR_maintenance1) && \
    defined(VK_KHR_storage_buffer_storage_class) && \
    defined(VK_KHR_surface) && \
    defined(VK_KHR_swapchain) && \
    defined(VK_KHR_variable_pointers)
#define VP_ANDROID_baseline_2021 1
#define VP_ANDROID_BASELINE_2021_NAME "VP_ANDROID_baseline_2021"
#define VP_ANDROID_BASELINE_2021_SPEC_VERSION 3
#define VP_ANDROID_BASELINE_2021_MIN_API_VERSION VK_MAKE_VERSION(1, 0, 68)
#endif

#if defined(VK_VERSION_1_3) && \
    defined(VK_KHR_global_priority)
#define VP_KHR_roadmap_2022 1
#define VP_KHR_ROADMAP_2022_NAME "VP_KHR_roadmap_2022"
#define VP_KHR_ROADMAP_2022_SPEC_VERSION 1
#define VP_KHR_ROADMAP_2022_MIN_API_VERSION VK_MAKE_VERSION(1, 3, 204)
#endif

#if defined(VK_VERSION_1_3) && \
    defined(VP_KHR_roadmap_2022) && \
    defined(VK_KHR_dynamic_rendering_local_read) && \
    defined(VK_KHR_index_type_uint8) && \
    defined(VK_KHR_line_rasterization) && \
    defined(VK_KHR_load_store_op_none) && \
    defined(VK_KHR_maintenance5) && \
    defined(VK_KHR_map_memory2) && \
    defined(VK_KHR_push_descriptor) && \
    defined(VK_KHR_shader_expect_assume) && \
    defined(VK_KHR_shader_float_controls2) && \
    defined(VK_KHR_shader_maximal_reconvergence) && \
    defined(VK_KHR_shader_quad_control) && \
    defined(VK_KHR_shader_subgroup_rotate) && \
    defined(VK_KHR_shader_subgroup_uniform_control_flow) && \
    defined(VK_KHR_vertex_attribute_divisor)
#define VP_KHR_roadmap_2024 1
#define VP_KHR_ROADMAP_2024_NAME "VP_KHR_roadmap_2024"
#define VP_KHR_ROADMAP_2024_SPEC_VERSION 1
#define VP_KHR_ROADMAP_2024_MIN_API_VERSION VK_MAKE_VERSION(1, 3, 276)
#endif

#if defined(VK_VERSION_1_1) && \
    defined(VK_EXT_descriptor_indexing) && \
    defined(VK_EXT_host_query_reset) && \
    defined(VK_EXT_inline_uniform_block) && \
    defined(VK_EXT_memory_budget) && \
    defined(VK_EXT_robustness2) && \
    defined(VK_EXT_scalar_block_layout) && \
    defined(VK_EXT_subgroup_size_control) && \
    defined(VK_EXT_texel_buffer_alignment) && \
    defined(VK_EXT_vertex_attribute_divisor) && \
    defined(VK_KHR_16bit_storage) && \
    defined(VK_KHR_8bit_storage) && \
    defined(VK_KHR_bind_memory2) && \
    defined(VK_KHR_create_renderpass2) && \
    defined(VK_KHR_dedicated_allocation) && \
    defined(VK_KHR_depth_stencil_resolve) && \
    defined(VK_KHR_descriptor_update_template) && \
    defined(VK_KHR_device_group) && \
    defined(VK_KHR_driver_properties) && \
    defined(VK_KHR_external_fence) && \
    defined(VK_KHR_external_memory) && \
    defined(VK_KHR_external_semaphore) && \
    defined(VK_KHR_get_memory_requirements2) && \
    defined(VK_KHR_image_format_list) && \
    defined(VK_KHR_maintenance1) && \
    defined(VK_KHR_maintenance2) && \
    defined(VK_KHR_maintenance3) && \
    defined(VK_KHR_multiview) && \
    defined(VK_KHR_relaxed_block_layout) && \
    defined(VK_KHR_sampler_mirror_clamp_to_edge) && \
    defined(VK_KHR_shader_draw_parameters) && \
    defined(VK_KHR_shader_float16_int8) && \
    defined(VK_KHR_storage_buffer_storage_class) && \
    defined(VK_KHR_swapchain) && \
    defined(VK_KHR_swapchain_mutable_format) && \
    defined(VK_KHR_timeline_semaphore) && \
    defined(VK_KHR_uniform_buffer_standard_layout) && \
    defined(VK_KHR_variable_pointers)
#define VP_LUNARG_desktop_baseline_2022 1
#define VP_LUNARG_DESKTOP_BASELINE_2022_NAME "VP_LUNARG_desktop_baseline_2022"
#define VP_LUNARG_DESKTOP_BASELINE_2022_SPEC_VERSION 2
#define VP_LUNARG_DESKTOP_BASELINE_2022_MIN_API_VERSION VK_MAKE_VERSION(1, 1, 139)
#endif

#if defined(VK_VERSION_1_2) && \
    defined(VK_EXT_descriptor_indexing) && \
    defined(VK_EXT_host_query_reset) && \
    defined(VK_EXT_image_robustness) && \
    defined(VK_EXT_inline_uniform_block) && \
    defined(VK_EXT_memory_budget) && \
    defined(VK_EXT_robustness2) && \
    defined(VK_EXT_subgroup_size_control) && \
    defined(VK_EXT_texel_buffer_alignment) && \
    defined(VK_EXT_vertex_attribute_divisor) && \
    defined(VK_KHR_16bit_storage) && \
    defined(VK_KHR_8bit_storage) && \
    defined(VK_KHR_bind_memory2) && \
    defined(VK_KHR_create_renderpass2) && \
    defined(VK_KHR_dedicated_allocation) && \
    defined(VK_KHR_depth_stencil_resolve) && \
    defined(VK_KHR_descriptor_update_template) && \
    defined(VK_KHR_device_group) && \
    defined(VK_KHR_driver_properties) && \
    defined(VK_KHR_external_fence) && \
    defined(VK_KHR_external_memory) && \
    defined(VK_KHR_external_semaphore) && \
    defined(VK_KHR_get_memory_requirements2) && \
    defined(VK_KHR_image_format_list) && \
    defined(VK_KHR_imageless_framebuffer) && \
    defined(VK_KHR_maintenance1) && \
    defined(VK_KHR_maintenance2) && \
    defined(VK_KHR_maintenance3) && \
    defined(VK_KHR_multiview) && \
    defined(VK_KHR_relaxed_block_layout) && \
    defined(VK_KHR_sampler_mirror_clamp_to_edge) && \
    defined(VK_KHR_separate_depth_stencil_layouts) && \
    defined(VK_KHR_shader_draw_parameters) && \
    defined(VK_KHR_shader_float16_int8) && \
    defined(VK_KHR_shader_float_controls) && \
    defined(VK_KHR_spirv_1_4) && \
    defined(VK_KHR_storage_buffer_storage_class) && \
    defined(VK_KHR_swapchain) && \
    defined(VK_KHR_swapchain_mutable_format) && \
    defined(VK_KHR_timeline_semaphore) && \
    defined(VK_KHR_uniform_buffer_standard_layout) && \
    defined(VK_KHR_variable_pointers)
#define VP_LUNARG_desktop_baseline_2023 1
#define VP_LUNARG_DESKTOP_BASELINE_2023_NAME "VP_LUNARG_desktop_baseline_2023"
#define VP_LUNARG_DESKTOP_BASELINE_2023_SPEC_VERSION 2
#define VP_LUNARG_DESKTOP_BASELINE_2023_MIN_API_VERSION VK_MAKE_VERSION(1, 2, 148)
#endif

#if defined(VK_VERSION_1_2) && \
    defined(VK_EXT_4444_formats) && \
    defined(VK_EXT_calibrated_timestamps) && \
    defined(VK_EXT_descriptor_indexing) && \
    defined(VK_EXT_extended_dynamic_state) && \
    defined(VK_EXT_extended_dynamic_state2) && \
    defined(VK_EXT_external_memory_host) && \
    defined(VK_EXT_host_query_reset) && \
    defined(VK_EXT_image_robustness) && \
    defined(VK_EXT_inline_uniform_block) && \
    defined(VK_EXT_memory_budget) && \
    defined(VK_EXT_pipeline_creation_cache_control) && \
    defined(VK_EXT_pipeline_creation_feedback) && \
    defined(VK_EXT_private_data) && \
    defined(VK_EXT_robustness2) && \
    defined(VK_EXT_scalar_block_layout) && \
    defined(VK_EXT_separate_stencil_usage) && \
    defined(VK_EXT_shader_atomic_float) && \
    defined(VK_EXT_shader_demote_to_helper_invocation) && \
    defined(VK_EXT_shader_subgroup_ballot) && \
    defined(VK_EXT_shader_subgroup_vote) && \
    defined(VK_EXT_shader_viewport_index_layer) && \
    defined(VK_EXT_subgroup_size_control) && \
    defined(VK_EXT_texel_buffer_alignment) && \
    defined(VK_EXT_vertex_attribute_divisor) && \
    defined(VK_KHR_16bit_storage) && \
    defined(VK_KHR_8bit_storage) && \
    defined(VK_KHR_bind_memory2) && \
    defined(VK_KHR_buffer_device_address) && \
    defined(VK_KHR_copy_commands2) && \
    defined(VK_KHR_create_renderpass2) && \
    defined(VK_KHR_dedicated_allocation) && \
    defined(VK_KHR_depth_stencil_resolve) && \
    defined(VK_KHR_descriptor_update_template) && \
    defined(VK_KHR_device_group) && \
    defined(VK_KHR_driver_properties) && \
    defined(VK_KHR_dynamic_rendering) && \
    defined(VK_KHR_external_fence) && \
    defined(VK_KHR_external_memory) && \
    defined(VK_KHR_external_semaphore) && \
    defined(VK_KHR_format_feature_flags2) && \
    defined(VK_KHR_get_memory_requirements2) && \
    defined(VK_KHR_image_format_list) && \
    defined(VK_KHR_imageless_framebuffer) && \
    defined(VK_KHR_maintenance1) && \
    defined(VK_KHR_maintenance2) && \
    defined(VK_KHR_maintenance3) && \
    defined(VK_KHR_multiview) && \
    defined(VK_KHR_push_descriptor) && \
    defined(VK_KHR_relaxed_block_layout) && \
    defined(VK_KHR_sampler_mirror_clamp_to_edge) && \
    defined(VK_KHR_separate_depth_stencil_layouts) && \
    defined(VK_KHR_shader_draw_parameters) && \
    defined(VK_KHR_shader_float16_int8) && \
    defined(VK_KHR_shader_float_controls) && \
    defined(VK_KHR_shader_non_semantic_info) && \
    defined(VK_KHR_shader_subgroup_extended_types) && \
    defined(VK_KHR_spirv_1_4) && \
    defined(VK_KHR_storage_buffer_storage_class) && \
    defined(VK_KHR_swapchain) && \
    defined(VK_KHR_swapchain_mutable_format) && \
    defined(VK_KHR_synchronization2) && \
    defined(VK_KHR_timeline_semaphore) && \
    defined(VK_KHR_uniform_buffer_standard_layout) && \
    defined(VK_KHR_variable_pointers)
#define VP_LUNARG_desktop_baseline_2024 1
#define VP_LUNARG_DESKTOP_BASELINE_2024_NAME "VP_LUNARG_desktop_baseline_2024"
#define VP_LUNARG_DESKTOP_BASELINE_2024_SPEC_VERSION 1
#define VP_LUNARG_DESKTOP_BASELINE_2024_MIN_API_VERSION VK_MAKE_VERSION(1, 2, 197)
#endif

#if defined(VK_VERSION_1_0)
#define VP_LUNARG_minimum_requirements_1_0 1
#define VP_LUNARG_MINIMUM_REQUIREMENTS_1_0_NAME "VP_LUNARG_minimum_requirements_1_0"
#define VP_LUNARG_MINIMUM_REQUIREMENTS_1_0_SPEC_VERSION 1
#define VP_LUNARG_MINIMUM_REQUIREMENTS_1_0_MIN_API_VERSION VK_MAKE_VERSION(1, 0, 68)
#endif

#if defined(VK_VERSION_1_1)
#define VP_LUNARG_minimum_requirements_1_1 1
#define VP_LUNARG_MINIMUM_REQUIREMENTS_1_1_NAME "VP_LUNARG_minimum_requirements_1_1"
#define VP_LUNARG_MINIMUM_REQUIREMENTS_1_1_SPEC_VERSION 1
#define VP_LUNARG_MINIMUM_REQUIREMENTS_1_1_MIN_API_VERSION VK_MAKE_VERSION(1, 1, 108)
#endif

#if defined(VK_VERSION_1_2)
#define VP_LUNARG_minimum_requirements_1_2 1
#define VP_LUNARG_MINIMUM_REQUIREMENTS_1_2_NAME "VP_LUNARG_minimum_requirements_1_2"
#define VP_LUNARG_MINIMUM_REQUIREMENTS_1_2_SPEC_VERSION 1
#define VP_LUNARG_MINIMUM_REQUIREMENTS_1_2_MIN_API_VERSION VK_MAKE_VERSION(1, 2, 131)
#endif

#if defined(VK_VERSION_1_3)
#define VP_LUNARG_minimum_requirements_1_3 1
#define VP_LUNARG_MINIMUM_REQUIREMENTS_1_3_NAME "VP_LUNARG_minimum_requirements_1_3"
#define VP_LUNARG_MINIMUM_REQUIREMENTS_1_3_SPEC_VERSION 1
#define VP_LUNARG_MINIMUM_REQUIREMENTS_1_3_MIN_API_VERSION VK_MAKE_VERSION(1, 3, 204)
#endif

#define VP_HEADER_VERSION_COMPLETE VK_MAKE_API_VERSION(0, 2, 0, VK_HEADER_VERSION)

#define VP_MAX_PROFILE_NAME_SIZE 256U

typedef struct VpProfileProperties {
    char        profileName[VP_MAX_PROFILE_NAME_SIZE];
    uint32_t    specVersion;
} VpProfileProperties;

typedef struct VpBlockProperties {
    VpProfileProperties profiles;
    uint32_t apiVersion;
    char blockName[VP_MAX_PROFILE_NAME_SIZE];
} VpBlockProperties;

typedef struct VpVideoProfileProperties {
    char name[VP_MAX_PROFILE_NAME_SIZE];
} VpVideoProfileProperties;

typedef enum VpInstanceCreateFlagBits {
    VP_INSTANCE_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VpInstanceCreateFlagBits;
typedef VkFlags VpInstanceCreateFlags;

typedef struct VpInstanceCreateInfo {
    const VkInstanceCreateInfo* pCreateInfo;
    VpInstanceCreateFlags       flags;
    uint32_t                    enabledFullProfileCount;
    const VpProfileProperties*  pEnabledFullProfiles;
    uint32_t                    enabledProfileBlockCount;
    const VpBlockProperties*    pEnabledProfileBlocks;
} VpInstanceCreateInfo;

typedef enum VpDeviceCreateFlagBits {
    VP_DEVICE_CREATE_DISABLE_ROBUST_BUFFER_ACCESS_BIT = 0x0000001,
    VP_DEVICE_CREATE_DISABLE_ROBUST_IMAGE_ACCESS_BIT = 0x0000002,
    VP_DEVICE_CREATE_DISABLE_ROBUST_ACCESS =
        VP_DEVICE_CREATE_DISABLE_ROBUST_BUFFER_ACCESS_BIT | VP_DEVICE_CREATE_DISABLE_ROBUST_IMAGE_ACCESS_BIT,

    VP_DEVICE_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VpDeviceCreateFlagBits;
typedef VkFlags VpDeviceCreateFlags;

typedef struct VpDeviceCreateInfo {
    const VkDeviceCreateInfo*   pCreateInfo;
    VpDeviceCreateFlags         flags;
    uint32_t                    enabledFullProfileCount;
    const VpProfileProperties*  pEnabledFullProfiles;
    uint32_t                    enabledProfileBlockCount;
    const VpBlockProperties*    pEnabledProfileBlocks;
} VpDeviceCreateInfo;

VK_DEFINE_HANDLE(VpCapabilities)

typedef enum VpCapabilitiesCreateFlagBits {
    VP_PROFILE_CREATE_STATIC_BIT = (1 << 0),
    //VP_PROFILE_CREATE_DYNAMIC_BIT = (1 << 1),
    VP_PROFILE_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VpCapabilitiesCreateFlagBits;

typedef VkFlags VpCapabilitiesCreateFlags;

// Pointers to some Vulkan functions - a subset used by the library.
// Used in VpCapabilitiesCreateInfo::pVulkanFunctions.

typedef struct VpVulkanFunctions {
    /// Required when using VP_DYNAMIC_VULKAN_FUNCTIONS.
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
    /// Required when using VP_DYNAMIC_VULKAN_FUNCTIONS.
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr;
    PFN_vkEnumerateInstanceVersion EnumerateInstanceVersion;
    PFN_vkEnumerateInstanceExtensionProperties EnumerateInstanceExtensionProperties;
    PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties;
    PFN_vkGetPhysicalDeviceFeatures2 GetPhysicalDeviceFeatures2;
    PFN_vkGetPhysicalDeviceProperties2 GetPhysicalDeviceProperties2;
    PFN_vkGetPhysicalDeviceFormatProperties2 GetPhysicalDeviceFormatProperties2;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties2 GetPhysicalDeviceQueueFamilyProperties2;
    PFN_vkCreateInstance CreateInstance;
    PFN_vkCreateDevice CreateDevice;
} VpVulkanFunctions;

/// Description of a Allocator to be created.
typedef struct VpCapabilitiesCreateInfo
{
    /// Flags for created allocator. Use #VpInstanceCreateFlagBits enum.
    VpCapabilitiesCreateFlags       flags;
    uint32_t                        apiVersion;
    const VpVulkanFunctions*        pVulkanFunctions;
} VpCapabilitiesCreateInfo;

VPAPI_ATTR VkResult vpCreateCapabilities(
    const VpCapabilitiesCreateInfo*             pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VpCapabilities*                             pCapabilities);

/// Destroys allocator object.
VPAPI_ATTR void vpDestroyCapabilities(
    VpCapabilities                              capabilities,
    const VkAllocationCallbacks*                pAllocator);

// Query the list of available profiles in the library
VPAPI_ATTR VkResult vpGetProfiles(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    uint32_t*                                   pPropertyCount,
    VpProfileProperties*                        pProperties);

// List the required profiles of a profile
VPAPI_ATTR VkResult vpGetProfileRequiredProfiles(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    uint32_t*                                   pPropertyCount,
    VpProfileProperties*                        pProperties);

// Query the profile required Vulkan API version
VPAPI_ATTR uint32_t vpGetProfileAPIVersion(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile);

// List the recommended fallback profiles of a profile
VPAPI_ATTR VkResult vpGetProfileFallbacks(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    uint32_t*                                   pPropertyCount,
    VpProfileProperties*                        pProperties);

// Query whether the profile has multiple variants. Profiles with multiple variants can only use vpGetInstanceProfileSupport and vpGetPhysicalDeviceProfileSupport capabilities of the library. Other function will return a VK_ERROR_UNKNOWN error
VPAPI_ATTR VkResult vpHasMultipleVariantsProfile(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    VkBool32*                                   pHasMultipleVariants);

// Check whether a profile is supported at the instance level
VPAPI_ATTR VkResult vpGetInstanceProfileSupport(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const char*                                 pLayerName,
    const VpProfileProperties*                  pProfile,
    VkBool32*                                   pSupported);

// Check whether a variant of a profile is supported at the instance level and report this list of blocks used to validate the profiles
VPAPI_ATTR VkResult vpGetInstanceProfileVariantsSupport(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const char*                                 pLayerName,
    const VpProfileProperties*                  pProfile,
    VkBool32*                                   pSupported,
    uint32_t*                                   pPropertyCount,
    VpBlockProperties*                          pProperties);

// Create a VkInstance with the profile instance extensions enabled
VPAPI_ATTR VkResult vpCreateInstance(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpInstanceCreateInfo*                 pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkInstance*                                 pInstance);

// Check whether a profile is supported by the physical device
VPAPI_ATTR VkResult vpGetPhysicalDeviceProfileSupport(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    VkInstance                                  instance,
    VkPhysicalDevice                            physicalDevice,
    const VpProfileProperties*                  pProfile,
    VkBool32*                                   pSupported);

// Check whether a variant of a profile is supported by the physical device and report this list of blocks used to validate the profiles
VPAPI_ATTR VkResult vpGetPhysicalDeviceProfileVariantsSupport(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    VkInstance                                  instance,
    VkPhysicalDevice                            physicalDevice,
    const VpProfileProperties*                  pProfile,
    VkBool32*                                   pSupported,
    uint32_t*                                   pPropertyCount,
    VpBlockProperties*                          pProperties);

// Create a VkDevice with the profile features and device extensions enabled
VPAPI_ATTR VkResult vpCreateDevice(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    VkPhysicalDevice                            physicalDevice,
    const VpDeviceCreateInfo*                   pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDevice*                                   pDevice);

// Query the list of instance extensions of a profile
VPAPI_ATTR VkResult vpGetProfileInstanceExtensionProperties(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    const char*                                 pBlockName,
    uint32_t*                                   pPropertyCount,
    VkExtensionProperties*                      pProperties);

// Query the list of device extensions of a profile
VPAPI_ATTR VkResult vpGetProfileDeviceExtensionProperties(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    const char*                                 pBlockName,
    uint32_t*                                   pPropertyCount,
    VkExtensionProperties*                      pProperties);

// Fill the feature structures with the requirements of a profile
VPAPI_ATTR VkResult vpGetProfileFeatures(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    const char*                                 pBlockName,
    void*                                       pNext);

// Query the list of feature structure types specified by the profile
VPAPI_ATTR VkResult vpGetProfileFeatureStructureTypes(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    const char*                                 pBlockName,
    uint32_t*                                   pStructureTypeCount,
    VkStructureType*                            pStructureTypes);

// Fill the property structures with the requirements of a profile
VPAPI_ATTR VkResult vpGetProfileProperties(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    const char*                                 pBlockName,
    void*                                       pNext);

// Query the list of property structure types specified by the profile
VPAPI_ATTR VkResult vpGetProfilePropertyStructureTypes(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    const char*                                 pBlockName,
    uint32_t*                                   pStructureTypeCount,
    VkStructureType*                            pStructureTypes);

// Fill the queue family property structures with the requirements of a profile
VPAPI_ATTR VkResult vpGetProfileQueueFamilyProperties(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    const char*                                 pBlockName,
    uint32_t*                                   pPropertyCount,
    VkQueueFamilyProperties2KHR*                pProperties);

// Query the list of queue family property structure types specified by the profile
VPAPI_ATTR VkResult vpGetProfileQueueFamilyStructureTypes(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    const char*                                 pBlockName,
    uint32_t*                                   pStructureTypeCount,
    VkStructureType*                            pStructureTypes);

// Query the list of formats with specified requirements by a profile
VPAPI_ATTR VkResult vpGetProfileFormats(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    const char*                                 pBlockName,
    uint32_t*                                   pFormatCount,
    VkFormat*                                   pFormats);

// Query the requirements of a format for a profile
VPAPI_ATTR VkResult vpGetProfileFormatProperties(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    const char*                                 pBlockName,
    VkFormat                                    format,
    void*                                       pNext);

// Query the list of format structure types specified by the profile
VPAPI_ATTR VkResult vpGetProfileFormatStructureTypes(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    const char*                                 pBlockName,
    uint32_t*                                   pStructureTypeCount,
    VkStructureType*                            pStructureTypes);

#ifdef VK_KHR_video_queue
// Query the list of video profiles specified by the profile
VPAPI_ATTR VkResult vpGetProfileVideoProfiles(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    const char*                                 pBlockName,
    uint32_t*                                   pVideoProfileCount,
    VpVideoProfileProperties*                   pVideoProfiles);

// Query the video profile info structures for a video profile defined by a profile
VPAPI_ATTR VkResult vpGetProfileVideoProfileInfo(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    const char*                                 pBlockName,
    uint32_t                                    videoProfileIndex,
    VkVideoProfileInfoKHR*                      pVideoProfileInfo);

// Query the list of video profile info structure types specified by the profile for a video profile
VPAPI_ATTR VkResult vpGetProfileVideoProfileInfoStructureTypes(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    const char*                                 pBlockName,
    uint32_t                                    videoProfileIndex,
    uint32_t*                                   pStructureTypeCount,
    VkStructureType*                            pStructureTypes);

// Query the video capabilities requirements for a video profile defined by a profile
VPAPI_ATTR VkResult vpGetProfileVideoCapabilities(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    const char*                                 pBlockName,
    uint32_t                                    videoProfileIndex,
    void*                                       pNext);

// Query the list of video capability structure types specified by the profile for a video profile
VPAPI_ATTR VkResult vpGetProfileVideoCapabilityStructureTypes(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    const char*                                 pBlockName,
    uint32_t                                    videoProfileIndex,
    uint32_t*                                   pStructureTypeCount,
    VkStructureType*                            pStructureTypes);

// Query the video format property requirements for a video profile defined by a profile
VPAPI_ATTR VkResult vpGetProfileVideoFormatProperties(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    const char*                                 pBlockName,
    uint32_t                                    videoProfileIndex,
    uint32_t*                                   pPropertyCount,
    VkVideoFormatPropertiesKHR*                 pProperties);

// Query the list of video format property structure types specified by the profile for a video profile
VPAPI_ATTR VkResult vpGetProfileVideoFormatStructureTypes(
#ifdef VP_USE_OBJECT
    VpCapabilities                              capabilities,
#endif//VP_USE_OBJECT
    const VpProfileProperties*                  pProfile,
    const char*                                 pBlockName,
    uint32_t                                    videoProfileIndex,
    uint32_t*                                   pStructureTypeCount,
    VkStructureType*                            pStructureTypes);
#endif  // VK_KHR_video_queue

#ifdef __cplusplus
}
#endif

#endif // VULKAN_PROFILES_H_
