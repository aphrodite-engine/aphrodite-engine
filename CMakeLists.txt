cmake_minimum_required(VERSION 3.26)

# When building directly using CMake, make sure you run the install step
# (it places the .so files in the correct location).
#
# Example:
# mkdir build && cd build
# cmake -G Ninja -DAPHRODITE_PYTHON_EXECUTABLE=`which python3` -DCMAKE_INSTALL_PREFIX=.. ..
# cmake --build . --target install
#
# If you want to only build one target, make sure to install it manually:
# cmake --build . --target _C
# cmake --install . --component _C
project(aphrodite_extensions LANGUAGES CXX)

# Optional: flag to enable Aphrodite's Vulkan backend wrappers (no third-party deps)
option(APHRODITE_ENABLE_VULKAN "Enable Aphrodite Vulkan backend wrappers" OFF)

# CUDA by default, can be overridden by using -DAPHRODITE_TARGET_DEVICE=... (used by setup.py)
set(APHRODITE_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for Aphrodite")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${APHRODITE_TARGET_DEVICE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${APHRODITE_PYTHON_PATH}")

# Prevent installation of dependencies (cutlass) by default.
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

#
# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
#
set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11" "3.12")

# Supported AMD GPU architectures.
set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1200;gfx1201")

#
# Supported/expected torch versions for CUDA/ROCm.
#
# Currently, having an incorrect pytorch version results in a warning
# rather than an error.
#
# Note: the CUDA torch version is derived from pyproject.toml and various
# requirements.txt files and should be kept consistent.  The ROCm torch
# versions are derived from docker/Dockerfile.rocm
#
set(TORCH_SUPPORTED_VERSION_CUDA "2.7.1")
set(TORCH_SUPPORTED_VERSION_ROCM "2.7.0")

#
# Try to find python package with an executable that exactly matches
# `APHRODITE_PYTHON_EXECUTABLE` and is one of the supported versions.
#
if (APHRODITE_PYTHON_EXECUTABLE)
  find_python_from_executable(${APHRODITE_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set APHRODITE_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

# Ensure the 'nvcc' command is in the PATH
find_program(NVCC_EXECUTABLE nvcc)
if (CUDA_FOUND AND NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found")
endif()

#
# Import torch cmake configuration.
# Torch also imports CUDA (and partially HIP) languages with some customizations,
# so there is no need to do this explicitly with check_language/enable_language,
# etc.
#
find_package(Torch REQUIRED)

# Supported NVIDIA architectures.
# This check must happen after find_package(Torch) because that's when CMAKE_CUDA_COMPILER_VERSION gets defined
if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  set(CUDA_SUPPORTED_ARCHS "6.0;6.1;7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0")
else()
  set(CUDA_SUPPORTED_ARCHS "6.0;6.1;7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0")
endif()

#
# Forward the non-CUDA device extensions to external CMake scripts.
#
if (NOT APHRODITE_TARGET_DEVICE STREQUAL "cuda" AND
    NOT APHRODITE_TARGET_DEVICE STREQUAL "rocm")
    if (APHRODITE_TARGET_DEVICE STREQUAL "cpu")
        include(${CMAKE_CURRENT_LIST_DIR}/cmake/cpu_extension.cmake)
    else()
        # Still allow optional Vulkan shader generation even if not building GPU extensions
        if(APHRODITE_ENABLE_VULKAN)
          message(STATUS "Vulkan shaders were generated (no GPU extension build)")
        endif()
        return()
    endif()
    return()
endif()

#
# Set up GPU language and check the torch version and warn if it isn't
# what is expected.
#
if (NOT HIP_FOUND AND CUDA_FOUND)
  set(APHRODITE_GPU_LANG "CUDA")

  if (NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} "
      "expected for CUDA build, saw ${Torch_VERSION} instead.")
  endif()
elseif(HIP_FOUND)
  set(APHRODITE_GPU_LANG "HIP")

  # Importing torch recognizes and sets up some HIP/ROCm configuration but does
  # not let cmake recognize .hip files. In order to get cmake to understand the
  # .hip extension automatically, HIP must be enabled explicitly.
  enable_language(HIP)

  # ROCm 5.X and 6.X
  if (ROCM_VERSION_DEV_MAJOR GREATER_EQUAL 5 AND
      NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_ROCM})
    message(WARNING "Pytorch version >= ${TORCH_SUPPORTED_VERSION_ROCM} "
      "expected for ROCm build, saw ${Torch_VERSION} instead.")
  endif()
else()
  message(FATAL_ERROR "Can't find CUDA or HIP installation.")
endif()


if(APHRODITE_GPU_LANG STREQUAL "CUDA")
  #
  # For cuda we want to be able to control which architectures we compile for on
  # a per-file basis in order to cut down on compile time. So here we extract
  # the set of architectures we want to compile for and remove the from the
  # CMAKE_CUDA_FLAGS so that they are not applied globally.
  #
  clear_cuda_arches(CUDA_ARCH_FLAGS)
  extract_unique_cuda_archs_ascending(CUDA_ARCHS "${CUDA_ARCH_FLAGS}")
  message(STATUS "CUDA target architectures: ${CUDA_ARCHS}")
  # Filter the target architectures by the supported supported archs
  # since for some files we will build for all CUDA_ARCHS.
  cuda_archs_loose_intersection(CUDA_ARCHS
    "${CUDA_SUPPORTED_ARCHS}" "${CUDA_ARCHS}")
  message(STATUS "CUDA supported target architectures: ${CUDA_ARCHS}")
else()
  #
  # For other GPU targets override the GPU architectures detected by cmake/torch
  # and filter them by the supported versions for the current language.
  # The final set of arches is stored in `APHRODITE_GPU_ARCHES`.
  #
  override_gpu_arches(APHRODITE_GPU_ARCHES
    ${APHRODITE_GPU_LANG}
    "${${APHRODITE_GPU_LANG}_SUPPORTED_ARCHS}")
endif()

#
# Query torch for additional GPU compilation flags for the given
# `APHRODITE_GPU_LANG`.
# The final set of arches is stored in `APHRODITE_GPU_FLAGS`.
#
get_torch_gpu_compiler_flags(APHRODITE_GPU_FLAGS ${APHRODITE_GPU_LANG})

#
# Set nvcc parallelism.
#
if(NVCC_THREADS AND APHRODITE_GPU_LANG STREQUAL "CUDA")
  list(APPEND APHRODITE_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()


#
# Use FetchContent for C++ dependencies that are compiled as part of Aphrodite's build process.
# setup.py will override FETCHCONTENT_BASE_DIR to play nicely with sccache.
# Each dependency that produces build artifacts should override its BINARY_DIR to avoid
# conflicts between build types. It should instead be set to ${CMAKE_BINARY_DIR}/<dependency>.
#
include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR}) # Ensure the directory exists
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

#
# Set rocm version dev int.
#
if(APHRODITE_GPU_LANG STREQUAL "HIP")
  #
  # Overriding the default -O set up by cmake, adding ggdb3 for the most verbose devug info
  #
  set(CMAKE_${APHRODITE_GPU_LANG}_FLAGS_DEBUG "${CMAKE_${APHRODITE_GPU_LANG}_FLAGS_DEBUG} -O0 -ggdb3")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb3")


  #
  # Certain HIP functions are marked as [[nodiscard]], yet aphrodite ignores the result which generates
  # a lot of warnings that always mask real issues. Suppressing until this is properly addressed.
  #
  set(CMAKE_${APHRODITE_GPU_LANG}_FLAGS "${CMAKE_${APHRODITE_GPU_LANG}_FLAGS} -Wno-unused-result")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")
endif()

#
# Define other extension targets
#

#
# cumem_allocator extension
#

set(APHRODITE_CUMEM_EXT_SRC
  "kernels/cumem_allocator.cpp")

set_gencode_flags_for_srcs(
  SRCS "${APHRODITE_CUMEM_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

if(APHRODITE_GPU_LANG STREQUAL "CUDA")
  message(STATUS "Enabling cumem allocator extension.")
  # link against cuda driver library
  list(APPEND CUMEM_LIBS CUDA::cuda_driver)
  define_gpu_extension_target(
    cumem_allocator
    DESTINATION aphrodite
    LANGUAGE CXX
    SOURCES ${APHRODITE_CUMEM_EXT_SRC}
    LIBRARIES ${CUMEM_LIBS}
    USE_SABI 3.8
    WITH_SOABI)
endif()

#
# _C extension
#

set(APHRODITE_EXT_SRC
  "kernels/mamba/mamba_ssm/selective_scan_fwd.cu"
  "kernels/cache_kernels.cu"
  "kernels/attention/paged_attention_v1.cu"
  "kernels/attention/paged_attention_v2.cu"
  "kernels/attention/merge_attn_states.cu"
  "kernels/attention/vertical_slash_index.cu"
  "kernels/pos_encoding_kernels.cu"
  "kernels/activation_kernels.cu"
  "kernels/layernorm_kernels.cu"
  "kernels/layernorm_quant_kernels.cu"
  "kernels/sampler.cu"
  "kernels/cuda_view.cu"
  "kernels/quantization/squeezellm/quant_cuda_kernel.cu"
  "kernels/quantization/gptq/q_gemm.cu"
  "kernels/quantization/compressed_tensors/int8_quant_kernels.cu"
  "kernels/quantization/fp8/common.cu"
  "kernels/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu"
  "kernels/quantization/gguf/gguf_kernel.cu"
  "kernels/quantization/activation_kernels.cu"
  "kernels/cuda_utils_kernels.cu"
  "kernels/prepare_inputs/advance_step.cu"
  "kernels/all_reduce/custom_all_reduce.cu"
  "kernels/torch_bindings.cpp")

  if(APHRODITE_GPU_LANG STREQUAL "CUDA")
  SET(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")

  # Set CUTLASS_REVISION. Used for FetchContent. Also fixes some bogus messages when building.
  set(CUTLASS_REVISION "v4.0.0" CACHE STRING "CUTLASS revision to use")

  # Use the specified CUTLASS source directory for compilation if APHRODITE_CUTLASS_SRC_DIR is provided
  if (DEFINED ENV{APHRODITE_CUTLASS_SRC_DIR})
    set(APHRODITE_CUTLASS_SRC_DIR $ENV{APHRODITE_CUTLASS_SRC_DIR})
  endif()

  if(APHRODITE_CUTLASS_SRC_DIR)
    if(NOT IS_ABSOLUTE APHRODITE_CUTLASS_SRC_DIR)
      get_filename_component(APHRODITE_CUTLASS_SRC_DIR "${APHRODITE_CUTLASS_SRC_DIR}" ABSOLUTE)
    endif()
    message(STATUS "The APHRODITE_CUTLASS_SRC_DIR is set, using ${APHRODITE_CUTLASS_SRC_DIR} for compilation")
    FetchContent_Declare(cutlass SOURCE_DIR ${APHRODITE_CUTLASS_SRC_DIR})
  else()
    FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/nvidia/cutlass.git
        # Please keep this in sync with CUTLASS_REVISION line above.
        GIT_TAG ${CUTLASS_REVISION}
        GIT_PROGRESS TRUE

        # Speed up CUTLASS download by retrieving only the specified GIT_TAG instead of the history.
        # Important: If GIT_SHALLOW is enabled then GIT_TAG works only with branch names and tags.
        # So if the GIT_TAG above is updated to a commit hash, GIT_SHALLOW must be set to FALSE
        GIT_SHALLOW TRUE
    )
  endif()
  FetchContent_MakeAvailable(cutlass)

  list(APPEND APHRODITE_EXT_SRC
    "kernels/quantization/fp6/fp6_linear.cu"
    "kernels/quantization/aqlm/gemm_kernels.cu"
    "kernels/quantization/vptq/gemm_kernels.cu"
    "kernels/quantization/awq/gemm_kernels.cu"
    "kernels/quantization/quip/origin_order.cu"
    "kernels/permute_cols.cu"
    "kernels/sampling/sampling.cu"
    "kernels/quantization/cutlass_w8a8/scaled_mm_entry.cu"
    "kernels/quantization/fp4/nvfp4_quant_entry.cu"
    "kernels/quantization/fp4/nvfp4_scaled_mm_entry.cu"
    "kernels/quantization/fp4/nvfp4_blockwise_moe_kernel.cu"
    "kernels/sparse/cutlass/sparse_scaled_mm_entry.cu"
    "kernels/cutlass_extensions/common.cpp"
    "kernels/attention/mla/cutlass_mla_entry.cu"
    "kernels/quantization/fp8/per_token_group_quant.cu")

  set_gencode_flags_for_srcs(
    SRCS "${APHRODITE_EXT_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")

  # If Vulkan is enabled, replace CUDA variants of activation/layernorm/rope with Vulkan wrappers
  if(APHRODITE_ENABLE_VULKAN)
    list(REMOVE_ITEM APHRODITE_EXT_SRC
      "kernels/pos_encoding_kernels.cu"
      "kernels/activation_kernels.cu"
      "kernels/layernorm_kernels.cu")
    list(APPEND APHRODITE_EXT_SRC
      "kernels/vulkan/activation_vk.cpp"
      "kernels/vulkan/layernorm_vk.cpp"
      "kernels/vulkan/pos_encoding_vk.cpp"
      "kernels/vulkan/vk_dispatch.cpp"
      "kernels/vulkan/vk_runtime.cpp")

    # Compile and embed a minimal set of Vulkan shaders
    find_program(GLSLC glslc REQUIRED)
    set(VK_SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels/vulkan/shaders)
    # Emit SPIR-V into source tree for a stable path during dev/editable installs
    set(VK_SPV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels/vulkan/spv)
    file(MAKE_DIRECTORY ${VK_SPV_DIR})

    # Manually compile a minimal set of shaders (extend as needed)
    add_custom_command(
      OUTPUT ${VK_SPV_DIR}/swiglu_f32.spv
      COMMAND ${GLSLC} -fshader-stage=compute --target-env=vulkan1.2 -O -I${VK_SHADERS_DIR}
              -DA_TYPE=float -DD_TYPE=float
              -o ${VK_SPV_DIR}/swiglu_f32.spv ${VK_SHADERS_DIR}/swiglu.comp
      DEPENDS ${VK_SHADERS_DIR}/swiglu.comp
      COMMENT "Compiling Vulkan shader swiglu_f32"
    )
    add_custom_command(
      OUTPUT ${VK_SPV_DIR}/swiglu_f16.spv
      COMMAND ${GLSLC} -fshader-stage=compute --target-env=vulkan1.2 -O -I${VK_SHADERS_DIR}
              -DA_TYPE=float16_t -DD_TYPE=float16_t -DRTE16=1
              -o ${VK_SPV_DIR}/swiglu_f16.spv ${VK_SHADERS_DIR}/swiglu.comp
      DEPENDS ${VK_SHADERS_DIR}/swiglu.comp
      COMMENT "Compiling Vulkan shader swiglu_f16"
    )
    add_custom_command(
      OUTPUT ${VK_SPV_DIR}/geglu_f32.spv
      COMMAND ${GLSLC} -fshader-stage=compute --target-env=vulkan1.2 -O -I${VK_SHADERS_DIR}
              -DA_TYPE=float -DD_TYPE=float
              -o ${VK_SPV_DIR}/geglu_f32.spv ${VK_SHADERS_DIR}/geglu.comp
      DEPENDS ${VK_SHADERS_DIR}/geglu.comp ${VK_SHADERS_DIR}/glu_head.comp ${VK_SHADERS_DIR}/glu_main.comp
      COMMENT "Compiling Vulkan shader geglu_f32"
    )
    add_custom_command(
      OUTPUT ${VK_SPV_DIR}/geglu_f16.spv
      COMMAND ${GLSLC} -fshader-stage=compute --target-env=vulkan1.2 -O -I${VK_SHADERS_DIR}
              -DA_TYPE=float16_t -DD_TYPE=float16_t -DRTE16=1
              -o ${VK_SPV_DIR}/geglu_f16.spv ${VK_SHADERS_DIR}/geglu.comp
      DEPENDS ${VK_SHADERS_DIR}/geglu.comp ${VK_SHADERS_DIR}/glu_head.comp ${VK_SHADERS_DIR}/glu_main.comp
      COMMENT "Compiling Vulkan shader geglu_f16"
    )
    set(VK_SPV_OUTPUTS
      ${VK_SPV_DIR}/swiglu_f32.spv
      ${VK_SPV_DIR}/swiglu_f16.spv
      ${VK_SPV_DIR}/geglu_f32.spv
      ${VK_SPV_DIR}/geglu_f16.spv)

    add_custom_target(aphrodite_vk_shaders ALL DEPENDS ${VK_SPV_OUTPUTS})
  endif()

  # Only build Marlin kernels if we are building for at least some compatible archs.
  # Keep building Marlin for 9.0 as there are some group sizes and shapes that
  # are not supported by Machete yet.
  # 9.0 for latest bf16 atomicAdd PTX
  cuda_archs_loose_intersection(MARLIN_ARCHS "8.0;8.7;9.0+PTX" "${CUDA_ARCHS}")
  if (MARLIN_ARCHS)

    #
    # For the Marlin kernels we automatically generate sources for various
    # preselected input type pairs and schedules.
    # Generate sources:
    set(MARLIN_GEN_SCRIPT
      ${CMAKE_CURRENT_SOURCE_DIR}/kernels/quantization/gptq_marlin/generate_kernels.py)
    file(MD5 ${MARLIN_GEN_SCRIPT} MARLIN_GEN_SCRIPT_HASH)

    message(STATUS "Marlin generation script hash: ${MARLIN_GEN_SCRIPT_HASH}")
    message(STATUS "Last run Marlin generate script hash: $CACHE{MARLIN_GEN_SCRIPT_HASH}")

    if (NOT DEFINED CACHE{MARLIN_GEN_SCRIPT_HASH}
        OR NOT $CACHE{MARLIN_GEN_SCRIPT_HASH} STREQUAL ${MARLIN_GEN_SCRIPT_HASH})
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=$PYTHONPATH
          ${Python_EXECUTABLE} ${MARLIN_GEN_SCRIPT}
        RESULT_VARIABLE marlin_generation_result
        OUTPUT_VARIABLE marlin_generation_result
        OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log
        ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log
      )

      if (NOT marlin_generation_result EQUAL 0)
        message(FATAL_ERROR "Marlin generation failed."
                            " Result: \"${marlin_generation_result}\""
                            "\nCheck the log for details: "
                            "${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log")
      else()
        set(MARLIN_GEN_SCRIPT_HASH ${MARLIN_GEN_SCRIPT_HASH}
            CACHE STRING "Last run Marlin generate script hash" FORCE)
        message(STATUS "Marlin generation completed successfully.")
      endif()
    else()
      message(STATUS "Marlin generation script has not changed, skipping generation.")
    endif()

    file(GLOB MARLIN_TEMPLATE_KERNEL_SRC "kernels/quantization/gptq_marlin/kernel_*.cu")
    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_TEMPLATE_KERNEL_SRC}"
      CUDA_ARCHS "${MARLIN_ARCHS}")

    list(APPEND APHRODITE_EXT_SRC ${MARLIN_TEMPLATE_KERNEL_SRC})
    set(MARLIN_SRCS
       "kernels/quantization/marlin/dense/marlin_cuda_kernel.cu"
       "kernels/quantization/marlin/sparse/marlin_24_cuda_kernel.cu"
       "kernels/quantization/marlin/qqq/marlin_qqq_gemm_kernel.cu"
       "kernels/quantization/gptq_marlin/gptq_marlin.cu"
       "kernels/quantization/gptq_marlin/gptq_marlin_repack.cu"
       "kernels/quantization/gptq_marlin/awq_marlin_repack.cu")
    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_SRCS}"
      CUDA_ARCHS "${MARLIN_ARCHS}")
    list(APPEND APHRODITE_EXT_SRC "${MARLIN_SRCS}")
    message(STATUS "Building Marlin kernels for archs: ${MARLIN_ARCHS}")
  else()
    message(STATUS "Not building Marlin kernels as no compatible archs found"
                   " in CUDA target architectures")
  endif()

  # Only build AllSpark kernels if we are building for at least some compatible archs.
  cuda_archs_loose_intersection(ALLSPARK_ARCHS "8.0;8.6;8.7;8.9" "${CUDA_ARCHS}")
  if (ALLSPARK_ARCHS)
    set(ALLSPARK_SRCS
       "kernels/quantization/gptq_allspark/allspark_repack.cu"
       "kernels/quantization/gptq_allspark/allspark_qgemm_w8a16.cu")
    set_gencode_flags_for_srcs(
      SRCS "${ALLSPARK_SRCS}"
      CUDA_ARCHS "${ALLSPARK_ARCHS}")
    list(APPEND APHRODITE_EXT_SRC "${ALLSPARK_SRCS}")
    message(STATUS "Building AllSpark kernels for archs: ${ALLSPARK_ARCHS}")
  else()
    message(STATUS "Not building AllSpark kernels as no compatible archs found"
                   " in CUDA target architectures")
  endif()


  set(SCALED_MM_3X_ARCHS)
  # The cutlass_scaled_mm kernels for Hopper (c3x, i.e. CUTLASS 3.x) require
  # CUDA 12.0 or later
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "9.0a;" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND SCALED_MM_ARCHS)
    set(SRCS
       "kernels/quantization/cutlass_w8a8/scaled_mm_c3x_sm90.cu"
       "kernels/quantization/cutlass_w8a8/c3x/scaled_mm_sm90_fp8.cu"
       "kernels/quantization/cutlass_w8a8/c3x/scaled_mm_sm90_int8.cu"
       "kernels/quantization/cutlass_w8a8/c3x/scaled_mm_azp_sm90_int8.cu"
       "kernels/quantization/cutlass_w8a8/c3x/scaled_mm_blockwise_sm90_fp8.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND APHRODITE_EXT_SRC "${SRCS}")
    list(APPEND APHRODITE_GPU_FLAGS "-DENABLE_SCALED_MM_SM90=1")
    # Let scaled_mm_c2x know it doesn't need to build these arches
    list(APPEND SCALED_MM_3X_ARCHS "${SCALED_MM_ARCHS}")
    message(STATUS "Building scaled_mm_c3x_sm90 for archs: ${SCALED_MM_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND SCALED_MM_ARCHS)
      message(STATUS "Not building scaled_mm_c3x_sm90 as CUDA Compiler version is "
                     "not >= 12.0, we recommend upgrading to CUDA 12.0 or "
                     "later if you intend on running FP8 quantized models on "
                     "Hopper.")
    else()
      message(STATUS "Not building scaled_mm_c3x_sm90 as no compatible archs found "
                     "in CUDA target architectures")
    endif()
  endif()

  # The cutlass_scaled_mm kernels for Geforce Blackwell SM120 (c3x, i.e. CUTLASS 3.x) require
  # CUDA 12.8 or later
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0;12.0a" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    set(SRCS
      "kernels/quantization/cutlass_w8a8/scaled_mm_c3x_sm120.cu"
      "kernels/quantization/cutlass_w8a8/c3x/scaled_mm_sm120_fp8.cu"
    )
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND APHRODITE_EXT_SRC "${SRCS}")
    list(APPEND APHRODITE_GPU_FLAGS "-DENABLE_SCALED_MM_SM120=1")
    # Let scaled_mm_c2x know it doesn't need to build these arches
    list(APPEND SCALED_MM_3X_ARCHS "${SCALED_MM_ARCHS}")
    message(STATUS "Building scaled_mm_c3x_sm120 for archs: ${SCALED_MM_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
      message(STATUS "Not building scaled_mm_c3x_sm120 as CUDA Compiler version is "
                     "not >= 12.8, we recommend upgrading to CUDA 12.8 or "
                     "later if you intend on running FP8 quantized models on "
                     "Blackwell.")
    else()
      message(STATUS "Not building scaled_mm_c3x_120 as no compatible archs found "
                     "in CUDA target architectures")
    endif()
  endif()


  # The cutlass_scaled_mm kernels for Blackwell SM100 (c3x, i.e. CUTLASS 3.x)
  # require CUDA 12.8 or later
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a;10.1a" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    set(SRCS
      "kernels/quantization/cutlass_w8a8/scaled_mm_c3x_sm100.cu"
      "kernels/quantization/cutlass_w8a8/c3x/scaled_mm_sm100_fp8.cu"
      "kernels/quantization/cutlass_w8a8/c3x/scaled_mm_blockwise_sm100_fp8.cu"
    )
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND APHRODITE_EXT_SRC "${SRCS}")
    list(APPEND APHRODITE_GPU_FLAGS "-DENABLE_SCALED_MM_SM100=1")
    # Let scaled_mm_c2x know it doesn't need to build these arches
    list(APPEND SCALED_MM_3X_ARCHS "${SCALED_MM_ARCHS}")
    message(STATUS "Building scaled_mm_c3x_sm100 for archs: ${SCALED_MM_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
      message(STATUS "Not building scaled_mm_c3x_sm100 as CUDA Compiler version is "
                     "not >= 12.8, we recommend upgrading to CUDA 12.8 or "
                     "later if you intend on running FP8 quantized models on "
                     "Blackwell.")
    else()
      message(STATUS "Not building scaled_mm_c3x_100 as no compatible archs found "
                     "in CUDA target architectures")
    endif()
  endif()

  #
  # For the cutlass_scaled_mm kernels we want to build the c2x (CUTLASS 2.x)
  # kernels for the remaining archs that are not already built for 3x.
  # (Build 8.9 for FP8)
  cuda_archs_loose_intersection(SCALED_MM_2X_ARCHS
    "7.5;8.0;8.7;8.9+PTX" "${CUDA_ARCHS}")
  # subtract out the archs that are already built for 3x
  list(REMOVE_ITEM SCALED_MM_2X_ARCHS ${SCALED_MM_3X_ARCHS})
  if (SCALED_MM_2X_ARCHS)
    set(SRCS "kernels/quantization/cutlass_w8a8/scaled_mm_c2x.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${SCALED_MM_2X_ARCHS}")
    list(APPEND APHRODITE_EXT_SRC "${SRCS}")
    list(APPEND APHRODITE_GPU_FLAGS "-DENABLE_SCALED_MM_C2X=1")
    message(STATUS "Building scaled_mm_c2x for archs: ${SCALED_MM_2X_ARCHS}")
  else()
    if (SCALED_MM_3X_ARCHS)
      message(STATUS "Not building scaled_mm_c2x as all archs are already built"
                     " for and covered by scaled_mm_c3x")
    else()
      message(STATUS "Not building scaled_mm_c2x as no compatible archs found "
                    "in CUDA target architectures")
    endif()
  endif()

  #
  # 2:4 Sparse Kernels

  # The 2:4 sparse kernels cutlass_scaled_sparse_mm and cutlass_compressor
  # require CUDA 12.2 or later (and only work on Hopper).
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "9.0a;" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.2 AND SCALED_MM_ARCHS)
    set(SRCS "kernels/sparse/cutlass/sparse_scaled_mm_c3x.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND APHRODITE_EXT_SRC "${SRCS}")
    list(APPEND APHRODITE_GPU_FLAGS "-DENABLE_SPARSE_SCALED_MM_C3X=1")
    message(STATUS "Building sparse_scaled_mm_c3x for archs: ${SCALED_MM_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.2 AND SCALED_MM_ARCHS)
      message(STATUS "Not building sparse_scaled_mm_c3x kernels as CUDA Compiler version is "
                     "not >= 12.2, we recommend upgrading to CUDA 12.2 or later "
                     "if you intend on running FP8 sparse quantized models on Hopper.")
    else()
      message(STATUS "Not building sparse_scaled_mm_c3x as no compatible archs found "
                     "in CUDA target architectures")
    endif()
  endif()

  # The nvfp4_scaled_mm_sm120 kernels for Geforce Blackwell SM120 require
  # CUDA 12.8 or later
  cuda_archs_loose_intersection(FP4_ARCHS "12.0;12.0a" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND FP4_ARCHS)
    set(SRCS
      "kernels/quantization/fp4/nvfp4_quant_kernels.cu"
      "kernels/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${FP4_ARCHS}")
    list(APPEND APHRODITE_EXT_SRC "${SRCS}")
    list(APPEND APHRODITE_GPU_FLAGS "-DENABLE_NVFP4_SM120=1")
    message(STATUS "Building NVFP4 for archs: ${FP4_ARCHS}")
  else()
    message(STATUS "Not building NVFP4 as no compatible archs were found.")
    # clear FP4_ARCHS
    set(FP4_ARCHS)
  endif()

  # FP4 Archs and flags
  cuda_archs_loose_intersection(FP4_ARCHS "10.0a" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND FP4_ARCHS)
    set(SRCS
      "kernels/quantization/fp4/nvfp4_quant_kernels.cu"
      "kernels/quantization/fp4/nvfp4_experts_quant.cu"
      "kernels/quantization/fp4/nvfp4_scaled_mm_kernels.cu"
      "kernels/quantization/fp4/nvfp4_blockwise_moe_kernel.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${FP4_ARCHS}")
    list(APPEND APHRODITE_EXT_SRC "${SRCS}")
    list(APPEND APHRODITE_GPU_FLAGS "-DENABLE_NVFP4_SM100=1")
    list(APPEND APHRODITE_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM100=1")
    message(STATUS "Building NVFP4 for archs: ${FP4_ARCHS}")
  else()
    message(STATUS "Not building NVFP4 as no compatible archs were found.")
    # clear FP4_ARCHS
    set(FP4_ARCHS)
  endif()

  # CUTLASS MLA Archs and flags
  cuda_archs_loose_intersection(MLA_ARCHS "10.0a" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND MLA_ARCHS)
    set(SRCS
      "kernels/attention/mla/cutlass_mla_kernels.cu"
      "kernels/attention/mla/sm100_cutlass_mla_kernel.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${MLA_ARCHS}")
    list(APPEND APHRODITE_EXT_SRC "${SRCS}")
    list(APPEND APHRODITE_GPU_FLAGS "-DENABLE_CUTLASS_MLA=1")
    # Add MLA-specific include directories only to MLA source files
    set_source_files_properties(${SRCS}
      PROPERTIES INCLUDE_DIRECTORIES "${CUTLASS_DIR}/examples/77_blackwell_fmha;${CUTLASS_DIR}/examples/common")
    message(STATUS "Building CUTLASS MLA for archs: ${MLA_ARCHS}")
  else()
    message(STATUS "Not building CUTLASS MLA as no compatible archs were found.")
    # clear MLA_ARCHS
    set(MLA_ARCHS)
  endif()

  # CUTLASS MoE kernels

  # The MoE kernel cutlass_moe_mm requires CUDA 12.3 or later (and ONLY works
  # on Hopper). get_cutlass_(pplx_)moe_mm_data should only be compiled
  # if it's possible to compile MoE kernels that use its output.
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "9.0a" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND SCALED_MM_ARCHS)
    set(SRCS "kernels/quantization/cutlass_w8a8/moe/grouped_mm_c3x_sm90.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND APHRODITE_EXT_SRC "${SRCS}")
    list(APPEND APHRODITE_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM90=1")
    message(STATUS "Building grouped_mm_c3x for archs: ${SCALED_MM_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND SCALED_MM_ARCHS)
      message(STATUS "Not building grouped_mm_c3x kernels as CUDA Compiler version is "
                     "not >= 12.3, we recommend upgrading to CUDA 12.3 or later "
                     "if you intend on running FP8 quantized MoE models on Hopper.")
    else()
      message(STATUS "Not building grouped_mm_c3x as no compatible archs found "
                     "in CUDA target architectures.")
    endif()
  endif()

  cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    set(SRCS "kernels/quantization/cutlass_w8a8/moe/grouped_mm_c3x_sm100.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND APHRODITE_EXT_SRC "${SRCS}")
    list(APPEND APHRODITE_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM100=1")
    message(STATUS "Building grouped_mm_c3x for archs: ${SCALED_MM_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
      message(STATUS "Not building grouped_mm_c3x kernels as CUDA Compiler version is "
                     "not >= 12.8, we recommend upgrading to CUDA 12.8 or later "
                     "if you intend on running FP8 quantized MoE models on Blackwell.")
    else()
      message(STATUS "Not building grouped_mm_c3x as no compatible archs found "
                     "in CUDA target architectures.")
    endif()
  endif()

  # moe_data.cu is used by all CUTLASS MoE kernels.
  cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0a" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND CUTLASS_MOE_DATA_ARCHS)
    set(SRCS "kernels/quantization/cutlass_w8a8/moe/moe_data.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${CUTLASS_MOE_DATA_ARCHS}")
    list(APPEND APHRODITE_EXT_SRC "${SRCS}")
    message(STATUS "Building moe_data for archs: ${CUTLASS_MOE_DATA_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND CUTLASS_MOE_DATA_ARCHS)
      message(STATUS "Not building moe_data as CUDA Compiler version is "
                     "not >= 12.3, we recommend upgrading to CUDA 12.3 or later "
                     "if you intend on running FP8 quantized MoE models on Hopper or Blackwell.")
    else()
      message(STATUS "Not building moe_data as no compatible archs found "
                     "in CUDA target architectures.")
    endif()
  endif()

  cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    set(SRCS "kernels/quantization/cutlass_w8a8/moe/blockwise_scaled_group_mm_sm100.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND APHRODITE_EXT_SRC "${SRCS}")
    list(APPEND APHRODITE_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM100=1")
    message(STATUS "Building blockwise_scaled_group_mm_sm100 for archs: ${SCALED_MM_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
      message(STATUS "Not building blockwise_scaled_group_mm_sm100 kernels as CUDA Compiler version is "
                     "not >= 12.8, we recommend upgrading to CUDA 12.8 or later "
                     "if you intend on running FP8 quantized MoE models on Blackwell.")
    else()
      message(STATUS "Not building blockwise_scaled_group_mm_sm100 as no compatible archs found "
                     "in CUDA target architectures")
    endif()
  endif()

  #
  # Machete kernels

  # The machete kernels only work on hopper and require CUDA 12.0 or later.
  # Only build Machete kernels if we are building for something compatible with sm90a
  cuda_archs_loose_intersection(MACHETE_ARCHS "9.0a" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0 AND MACHETE_ARCHS)
    #
    # For the Machete kernels we automatically generate sources for various
    # preselected input type pairs and schedules.
    # Generate sources:
    set(MACHETE_GEN_SCRIPT
      ${CMAKE_CURRENT_SOURCE_DIR}/kernels/quantization/machete/generate.py)
    file(MD5 ${MACHETE_GEN_SCRIPT} MACHETE_GEN_SCRIPT_HASH)

    message(STATUS "Machete generation script hash: ${MACHETE_GEN_SCRIPT_HASH}")
    message(STATUS "Last run machete generate script hash: $CACHE{MACHETE_GEN_SCRIPT_HASH}")

    if (NOT DEFINED CACHE{MACHETE_GEN_SCRIPT_HASH}
        OR NOT $CACHE{MACHETE_GEN_SCRIPT_HASH} STREQUAL ${MACHETE_GEN_SCRIPT_HASH})
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/kernels/cutlass_extensions/:${CUTLASS_DIR}/python/:${APHRODITE_PYTHON_PATH}:$PYTHONPATH
          ${Python_EXECUTABLE} ${MACHETE_GEN_SCRIPT}
        RESULT_VARIABLE machete_generation_result
        OUTPUT_VARIABLE machete_generation_output
        OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log
        ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log
      )

      if (NOT machete_generation_result EQUAL 0)
        message(FATAL_ERROR "Machete generation failed."
                            " Result: \"${machete_generation_result}\""
                            "\nCheck the log for details: "
                            "${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log")
      else()
        set(MACHETE_GEN_SCRIPT_HASH ${MACHETE_GEN_SCRIPT_HASH}
            CACHE STRING "Last run machete generate script hash" FORCE)
        message(STATUS "Machete generation completed successfully.")
      endif()
    else()
      message(STATUS "Machete generation script has not changed, skipping generation.")
    endif()

    # Add machete generated sources
    file(GLOB MACHETE_GEN_SOURCES "kernels/quantization/machete/generated/*.cu")
    list(APPEND APHRODITE_EXT_SRC ${MACHETE_GEN_SOURCES})

    # forward compatible
    set_gencode_flags_for_srcs(
      SRCS "${MACHETE_GEN_SOURCES}"
      CUDA_ARCHS "${MACHETE_ARCHS}")

    list(APPEND APHRODITE_EXT_SRC
      kernels/quantization/machete/machete_pytorch.cu)

    message(STATUS "Building Machete kernels for archs: ${MACHETE_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0
        AND MACHETE_ARCHS)
      message(STATUS "Not building Machete kernels as CUDA Compiler version is "
                     "not >= 12.0, we recommend upgrading to CUDA 12.0 or "
                     "later if you intend on running w4a16 quantized models on "
                     "Hopper.")
    else()
      message(STATUS "Not building Machete kernels as no compatible archs "
                     "found in CUDA target architectures")
    endif()
  endif()
# if CUDA endif
endif()

if (APHRODITE_GPU_LANG STREQUAL "HIP")
  # Add QuickReduce kernels
  list(APPEND APHRODITE_EXT_SRC
    "kernels/custom_quickreduce.cu"
  )
# if ROCM endif
endif()

message(STATUS "Enabling C extension.")

define_gpu_extension_target(
  _C
  DESTINATION aphrodite
  LANGUAGE ${APHRODITE_GPU_LANG}
  SOURCES ${APHRODITE_EXT_SRC}
  COMPILE_FLAGS ${APHRODITE_GPU_FLAGS}
  ARCHITECTURES ${APHRODITE_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

if(APHRODITE_ENABLE_VULKAN)
  # Hint CMake where to look for Vulkan
  if(DEFINED ENV{VULKAN_SDK})
    list(APPEND CMAKE_PREFIX_PATH $ENV{VULKAN_SDK})
    list(APPEND CMAKE_LIBRARY_PATH $ENV{VULKAN_SDK}/lib $ENV{VULKAN_SDK}/lib64)
    list(APPEND CMAKE_INCLUDE_PATH $ENV{VULKAN_SDK}/include)
  endif()
  find_package(Vulkan QUIET)
  if (Vulkan_FOUND)
    target_link_libraries(_C PRIVATE Vulkan::Vulkan)
    get_target_property(_vk_includes Vulkan::Vulkan INTERFACE_INCLUDE_DIRECTORIES)
    if(_vk_includes)
      target_include_directories(_C PRIVATE ${_vk_includes})
    endif()
  else()
    # Fallback: manually locate libvulkan and create an imported target
    set(_vk_candidates
      /usr/lib/x86_64-linux-gnu/libvulkan.so
      /usr/lib64/libvulkan.so
      /usr/lib/libvulkan.so)
    set(_vk_lib "")
    foreach(p ${_vk_candidates})
      if(EXISTS ${p})
        set(_vk_lib ${p})
        break()
      endif()
    endforeach()
    if(_vk_lib)
      set(_vk_inc /usr/include CACHE PATH "Vulkan headers path")
      add_library(Vulkan::Vulkan UNKNOWN IMPORTED)
      set_target_properties(Vulkan::Vulkan PROPERTIES
        IMPORTED_LOCATION ${_vk_lib}
        INTERFACE_INCLUDE_DIRECTORIES ${_vk_inc})
      target_link_libraries(_C PRIVATE Vulkan::Vulkan)
      target_include_directories(_C PRIVATE ${_vk_inc})
      message(STATUS "Linked Vulkan via fallback: ${_vk_lib}")
    else()
      message(WARNING "Vulkan SDK not fully found; building without linking Vulkan::Vulkan. Make sure runtime has libvulkan.")
    endif()
  endif()
  # As a last resort, force-add common system include roots so <vulkan/vulkan.h> resolves
  target_include_directories(_C PRIVATE /usr/include)
  if(DEFINED ENV{VULKAN_SDK})
    target_include_directories(_C PRIVATE $ENV{VULKAN_SDK}/include)
  endif()
  # Do not force /usr/include globally; restrict to Vulkan translation units via source-level includes
  target_compile_definitions(_C PRIVATE APHRODITE_ENABLE_VULKAN=1)
  # Ensure SPIR-V gets built before linking the extension
  add_dependencies(_C aphrodite_vk_shaders)
  # Provide a compile-time default for the runtime to find SPIR-Vs
  target_compile_definitions(_C PRIVATE APHRODITE_VK_SPV_DIR_DEFAULT="${VK_SPV_DIR}")
endif()

# If CUTLASS is compiled on NVCC >= 12.5, it by default uses
# cudaGetDriverEntryPointByVersion as a wrapper to avoid directly calling the
# driver API. This causes problems when linking with earlier versions of CUDA.
# Setting this variable sidesteps the issue by calling the driver directly.
target_compile_definitions(_C PRIVATE CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)

#
# _moe_C extension
#

set(APHRODITE_MOE_EXT_SRC
  "kernels/moe/torch_bindings.cpp"
  "kernels/moe/moe_align_sum_kernels.cu"
  "kernels/moe/topk_softmax_kernels.cu")

if(APHRODITE_GPU_LANG STREQUAL "CUDA")
  list(APPEND APHRODITE_MOE_EXT_SRC "kernels/moe/moe_wna16.cu")
endif()

if(APHRODITE_GPU_LANG STREQUAL "CUDA")
  set(MOE_PERMUTE_SRC
      "kernels/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu"
      "kernels/moe/moe_permute_unpermute_op.cu")

  list(APPEND APHRODITE_MOE_EXT_SRC "${MOE_PERMUTE_SRC}")
endif()

set_gencode_flags_for_srcs(
  SRCS "${APHRODITE_MOE_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

if(APHRODITE_GPU_LANG STREQUAL "CUDA")
  set(APHRODITE_MOE_WNA16_SRC
    "kernels/moe/moe_wna16.cu")

  set_gencode_flags_for_srcs(
    SRCS "${APHRODITE_MOE_WNA16_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")

  list(APPEND APHRODITE_MOE_EXT_SRC "${APHRODITE_MOE_WNA16_SRC}")
  # 9.0 for latest bf16 atomicAdd PTX
  cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "8.0;8.7;9.0+PTX" "${CUDA_ARCHS}")
  if (MARLIN_MOE_ARCHS)

    #
    # For the Marlin MOE kernels we automatically generate sources for various
    # preselected input type pairs and schedules.
    # Generate sources:
    set(MOE_MARLIN_GEN_SCRIPT
      ${CMAKE_CURRENT_SOURCE_DIR}/kernels/moe/marlin_moe_wna16/generate_kernels.py)
    file(MD5 ${MOE_MARLIN_GEN_SCRIPT} MOE_MARLIN_GEN_SCRIPT_HASH)

    message(STATUS "Marlin MOE generation script hash: ${MOE_MARLIN_GEN_SCRIPT_HASH}")
    message(STATUS "Last run Marlin MOE generate script hash: $CACHE{MOE_MARLIN_GEN_SCRIPT_HASH}")

    if (NOT DEFINED CACHE{MOE_MARLIN_GEN_SCRIPT_HASH}
        OR NOT $CACHE{MOE_MARLIN_GEN_SCRIPT_HASH} STREQUAL ${MOE_MARLIN_GEN_SCRIPT_HASH})
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=$PYTHONPATH
          ${Python_EXECUTABLE} ${MOE_MARLIN_GEN_SCRIPT}
        RESULT_VARIABLE moe_marlin_generation_result
        OUTPUT_VARIABLE moe_marlin_generation_output
        OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log
        ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log
      )

      if (NOT moe_marlin_generation_result EQUAL 0)
        message(FATAL_ERROR "Marlin MOE generation failed."
                            " Result: \"${moe_marlin_generation_result}\""
                            "\nCheck the log for details: "
                            "${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log")
      else()
        set(MOE_MARLIN_GEN_SCRIPT_HASH ${MOE_MARLIN_GEN_SCRIPT_HASH}
            CACHE STRING "Last run Marlin MOE generate script hash" FORCE)
        message(STATUS "Marlin MOE generation completed successfully.")
      endif()
    else()
      message(STATUS "Marlin MOE generation script has not changed, skipping generation.")
    endif()

    file(GLOB MOE_WNAA16_MARLIN_SRC "kernels/moe/marlin_moe_wna16/*.cu")
    set_gencode_flags_for_srcs(
      SRCS "${MOE_WNAA16_MARLIN_SRC}"
      CUDA_ARCHS "${MARLIN_MOE_ARCHS}")

    list(APPEND APHRODITE_MOE_EXT_SRC ${MOE_WNAA16_MARLIN_SRC})

    message(STATUS "Building Marlin MOE kernels for archs: ${MARLIN_MOE_ARCHS}")
  else()
    message(STATUS "Not building Marlin MOE kernels as no compatible archs found"
                   " in CUDA target architectures")
  endif()
endif()

message(STATUS "Enabling moe extension.")
define_gpu_extension_target(
  _moe_C
  DESTINATION aphrodite
  LANGUAGE ${APHRODITE_GPU_LANG}
  SOURCES ${APHRODITE_MOE_EXT_SRC}
  COMPILE_FLAGS ${APHRODITE_GPU_FLAGS}
  ARCHITECTURES ${APHRODITE_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

if(APHRODITE_GPU_LANG STREQUAL "HIP")
  #
  # _rocm_C extension
  #
  set(APHRODITE_ROCM_EXT_SRC
    "kernels/rocm/torch_bindings.cpp"
    "kernels/rocm/skinny_gemms.cu"
    "kernels/rocm/attention.cu")

  define_gpu_extension_target(
    _rocm_C
    DESTINATION aphrodite
    LANGUAGE ${APHRODITE_GPU_LANG}
    SOURCES ${APHRODITE_ROCM_EXT_SRC}
    COMPILE_FLAGS ${APHRODITE_GPU_FLAGS}
    ARCHITECTURES ${APHRODITE_GPU_ARCHES}
    USE_SABI 3
    WITH_SOABI)
endif()

# For CUDA we also build and ship some external projects.
if (APHRODITE_GPU_LANG STREQUAL "CUDA")

    # FlashMLA build - requires CUDA 12.3 or later and Hopper architecture
    cuda_archs_loose_intersection(FLASH_MLA_ARCHS "9.0;9.0a;10.0;10.1;12.0" "${CUDA_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.3 AND FLASH_MLA_ARCHS)
        set(FlashMLA_SOURCES
            kernels/flash_mla/flash_api.cpp
            kernels/flash_mla/flash_fwd_mla_bf16_sm90.cu
            kernels/flash_mla/flash_fwd_mla_fp16_sm90.cu
            kernels/flash_mla/flash_fwd_mla_metadata.cu)

        set_gencode_flags_for_srcs(
            SRCS "${FlashMLA_SOURCES}"
            CUDA_ARCHS "${FLASH_MLA_ARCHS}")

        define_gpu_extension_target(
            _flashmla_C
            DESTINATION aphrodite
            LANGUAGE ${APHRODITE_GPU_LANG}
            SOURCES ${FlashMLA_SOURCES}
            COMPILE_FLAGS ${APHRODITE_GPU_FLAGS}
            ARCHITECTURES ${APHRODITE_GPU_ARCHES}
            INCLUDE_DIRECTORIES 
                kernels/flash_mla/include
                ${CUTLASS_INCLUDE_DIR}
            USE_SABI 3
            WITH_SOABI)

        message(STATUS "Building FlashMLA for archs: ${FLASH_MLA_ARCHS}")
    else()
        if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.3 AND FLASH_MLA_ARCHS)
            message(STATUS "Not building FlashMLA kernels as CUDA Compiler version is "
                         "not > 12.3, we recommend upgrading to CUDA 12.3 or later "
                         "if you intend on using FlashMLA on Hopper.")
        else()
            message(STATUS "Not building FlashMLA as no compatible archs found "
                         "in CUDA target architectures")
        endif()
        # Create an empty target for setup.py when not targeting sm90a systems
        add_custom_target(_flashmla_C)
    endif()

    # vllm-flash-attn should be last as it overwrites some CMake functions
    include(cmake/external_project/vllm_flash_attn.cmake)
endif()
