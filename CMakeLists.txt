cmake_minimum_required(VERSION 3.26)

project(aphrodite_extensions LANGUAGES CXX)

# CUDA by default, can be overridden by using -DAPHRODITE_TARGET_DEVICE=... (used by setup.py)
set(APHRODITE_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for Aphrodite")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${APHRODITE_TARGET_DEVICE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${APHRODITE_PYTHON_PATH}")

#
# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
#
set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11" "3.12")

# Supported NVIDIA architectures.
set(CUDA_SUPPORTED_ARCHS "6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0")

# Supported AMD GPU architectures.
set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100;gfx1101")

#
# Supported/expected torch versions for CUDA/ROCm.
#
# Currently, having an incorrect pytorch version results in a warning
# rather than an error.
#
# Note: the CUDA torch version is derived from pyproject.toml and various
# requirements.txt files and should be kept consistent.  The ROCm torch
# versions are derived from Dockerfile.rocm
#
set(TORCH_SUPPORTED_VERSION_CUDA "2.4.0")
set(TORCH_SUPPORTED_VERSION_ROCM "2.5.0")

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
if(MSVC)
  find_package(CUDA REQUIRED)
  find_package(CUDAToolkit REQUIRED)
  # Add cuBLAS to the list of libraries to link against
  list(APPEND LIBS CUDA::cublas)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # Replace -std=c++20 with -std=c++17 in APHRODITE_GPU_FLAGS
  if(APHRODITE_GPU_LANG STREQUAL "CUDA")
    list(APPEND APHRODITE_GPU_FLAGS "--std=c++17" "-Xcompiler -Wno-return-type")
  endif()
endif()

#
# Add the `default` target which detects which extensions should be
# built based on platform/architecture.  This is the same logic that
# setup.py uses to select which extensions should be built and should
# be kept in sync.
#
# The `default` target makes direct use of cmake easier since knowledge
# of which extensions are supported has been factored in, e.g.
#
# mkdir build && cd build
# cmake -G Ninja -DAPHRODITE_PYTHON_EXECUTABLE=`which python3` -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=../aphrodite ..
# cmake --build . --target default
#
add_custom_target(default)
message(STATUS "Enabling core extension.")

# Define _core_C extension
#  built for (almost) every target platform, (excludes TPU and Neuron)

set(APHRODITE_EXT_SRC
  "kernels/core/torch_bindings.cpp")

define_gpu_extension_target(
  _core_C
  DESTINATION aphrodite
  LANGUAGE CXX
  SOURCES ${APHRODITE_EXT_SRC}
  COMPILE_FLAGS ${CXX_COMPILE_FLAGS}
  USE_SABI 3
  WITH_SOABI)

add_dependencies(default _core_C)

#
# Forward the non-CUDA device extensions to external CMake scripts.
#
if (NOT APHRODITE_TARGET_DEVICE STREQUAL "cuda" AND
    NOT APHRODITE_TARGET_DEVICE STREQUAL "rocm")
    if (APHRODITE_TARGET_DEVICE STREQUAL "cpu")
        include(${CMAKE_CURRENT_LIST_DIR}/cmake/cpu_extension.cmake)
    else()
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

#
# Override the GPU architectures detected by cmake/torch and filter them by
# the supported versions for the current language.
# The final set of arches is stored in `APHRODITE_GPU_ARCHES`.
#
override_gpu_arches(APHRODITE_GPU_ARCHES
  ${APHRODITE_GPU_LANG}
  "${${APHRODITE_GPU_LANG}_SUPPORTED_ARCHS}")

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
# Define other extension targets
#

#
# _C extension
#

set(APHRODITE_EXT_SRC
  "kernels/cache_kernels.cu"
  "kernels/attention/attention_kernels.cu"
  "kernels/pos_encoding_kernels.cu"
  "kernels/activation_kernels.cu"
  "kernels/layernorm_kernels.cu"
  "kernels/quantization/squeezellm/quant_cuda_kernel.cu"
  "kernels/quantization/gptq/q_gemm.cu"
  "kernels/quantization/compressed_tensors/int8_quant_kernels.cu"
  "kernels/quantization/fp8/common.cu"
  "kernels/cuda_utils_kernels.cu"
  "kernels/moe/align_block_size_kernel.cu"
  "kernels/prepare_inputs/advance_step.cu"
  "kernels/torch_bindings.cpp")

if(APHRODITE_GPU_LANG STREQUAL "CUDA")
  list(APPEND APHRODITE_EXT_SRC
    "kernels/quantization/fp6/fp6_linear.cu"
    "kernels/mamba/mamba_ssm/selective_scan_fwd.cu"
    "kernels/mamba/causal_conv1d/causal_conv1d.cu"
    "kernels/quantization/aqlm/gemm_kernels.cu"
    "kernels/quantization/awq/gemm_kernels.cu"
    "kernels/quantization/quip/origin_order.cu"
    "kernels/quantization/gptq_marlin/gptq_marlin.cu"
    "kernels/quantization/gptq_marlin/gptq_marlin_repack.cu"
    "kernels/quantization/marlin/dense/marlin_cuda_kernel.cu"
    "kernels/quantization/marlin/sparse/marlin_24_cuda_kernel.cu"
    "kernels/quantization/marlin/qqq/marlin_qqq_gemm_kernel.cu"
    "kernels/quantization/gguf/gguf_kernel.cu"
    "kernels/quantization/gptq_marlin/awq_marlin_repack.cu"
    "kernels/quantization/fp8/fp8_marlin.cu"
    "kernels/all_reduce/custom_all_reduce.cu"
    "kernels/permute_cols.cu"
    "kernels/sampling/sampling.cu"
    "kernels/flash_attn/flash_fwd_hdim32_bf16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim32_bf16_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim32_fp16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim32_fp16_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim64_bf16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim64_bf16_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim64_fp16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim64_fp16_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim96_bf16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim96_bf16_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim96_fp16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim96_fp16_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim128_bf16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim128_bf16_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim128_fp16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim128_fp16_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim160_bf16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim160_bf16_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim160_fp16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim160_fp16_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim192_bf16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim192_bf16_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim192_fp16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim192_fp16_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim224_bf16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim224_bf16_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim224_fp16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim224_fp16_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim256_bf16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim256_bf16_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim256_fp16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_hdim256_fp16_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim32_bf16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim32_bf16_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim32_fp16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim32_fp16_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim64_bf16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim64_bf16_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim64_fp16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim64_fp16_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim96_bf16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim96_bf16_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim96_fp16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim96_fp16_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim128_bf16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim128_bf16_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim128_fp16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim128_fp16_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim160_bf16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim160_bf16_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim160_fp16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim160_fp16_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim192_bf16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim192_bf16_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim192_fp16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim192_fp16_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim224_bf16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim224_bf16_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim224_fp16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim224_fp16_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim256_bf16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim256_bf16_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim256_fp16_causal_sm80.cu"
    "kernels/flash_attn/flash_fwd_split_hdim256_fp16_sm80.cu"
    )

  if(NOT MSVC)
    # Include CUTLASS only when needed
    include(FetchContent)
    SET(CUTLASS_ENABLE_HEADERS_ONLY=ON)
    FetchContent_Declare(
          cutlass
          GIT_REPOSITORY https://github.com/nvidia/cutlass.git
          GIT_TAG v3.5.1
          GIT_PROGRESS TRUE
          # Speed up CUTLASS download by retrieving only the specified GIT_TAG instead of the history.
          # Important: If GIT_SHALLOW is enabled then GIT_TAG works only with branch names and tags.
          # So if the GIT_TAG above is updated to a commit hash, GIT_SHALLOW must be set to FALSE
          GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(cutlass)

    list(APPEND APHRODITE_EXT_SRC
      "kernels/quantization/cutlass_w8a8/scaled_mm_entry.cu"
      "kernels/quantization/cutlass_w8a8/scaled_mm_c2x.cu"
      "kernels/quantization/cutlass_w8a8/scaled_mm_c3x.cu")

    # Enable sm90a for Hopper CUTLASS kernels when using newer CUDA
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)
      set_source_files_properties(
        "kernels/quantization/cutlass_w8a8/scaled_mm_c2x.cu"
        "kernels/quantization/cutlass_w8a8/scaled_mm_c3x.cu"
        PROPERTIES
        COMPILE_FLAGS
        "-gencode arch=compute_90a,code=sm_90a -Wno-psabi")
    endif()
  endif()

  #
  # Machete kernels

  # The machete kernels only work on hopper and require CUDA 12.0 or later.
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)
    #
    # For the Machete kernels we automatically generate sources for various 
    # preselected input type pairs and schedules.
    # Generate sources:
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E env 
      PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/kernels/cutlass_extensions/:${CUTLASS_DIR}/python/:${APHRODITE_PYTHON_PATH}:$PYTHONPATH 
        ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/kernels/quantization/machete/generate.py
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
      message(STATUS "Machete generation completed successfully.")
    endif()

    # Add machete generated sources
    file(GLOB MACHETE_GEN_SOURCES "kernels/quantization/machete/generated/*.cu")
    list(APPEND APHRODITE_EXT_SRC ${MACHETE_GEN_SOURCES})
    message(STATUS "Machete generated sources: ${MACHETE_GEN_SOURCES}")

    set_source_files_properties(
          ${MACHETE_GEN_SOURCES}
          PROPERTIES
          COMPILE_FLAGS
          "-gencode arch=compute_90a,code=sm_90a")
  endif()

  # Add pytorch binding for machete (add on even CUDA < 12.0 so that we can
  #  raise an error if the user that this was built with an incompatible 
  #  CUDA version)
  list(APPEND APHRODITE_EXT_SRC
    kernels/quantization/machete/machete_pytorch.cu)
endif()

define_gpu_extension_target(
  _C
  DESTINATION aphrodite
  LANGUAGE ${APHRODITE_GPU_LANG}
  SOURCES ${APHRODITE_EXT_SRC}
  COMPILE_FLAGS ${APHRODITE_GPU_FLAGS}
  ARCHITECTURES ${APHRODITE_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  LIBRARIES ${LIBS}
  USE_SABI 3
  WITH_SOABI)

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
  "kernels/moe/softmax.cu")

if(APHRODITE_GPU_LANG STREQUAL "CUDA")
  list(APPEND APHRODITE_MOE_EXT_SRC
      "kernels/moe/marlin_moe_ops.cu")
endif()

define_gpu_extension_target(
  _moe_C
  DESTINATION aphrodite
  LANGUAGE ${APHRODITE_GPU_LANG}
  SOURCES ${APHRODITE_MOE_EXT_SRC}
  COMPILE_FLAGS ${APHRODITE_GPU_FLAGS}
  ARCHITECTURES ${APHRODITE_GPU_ARCHES}
  USE_SABI 3
  WITH_SOABI)


if(APHRODITE_GPU_LANG STREQUAL "HIP")
  #
  # _rocm_C extension
  #
  set(APHRODITE_ROCM_EXT_SRC
    "kernels/rocm/torch_bindings.cpp"
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

if(APHRODITE_GPU_LANG STREQUAL "CUDA" OR APHRODITE_GPU_LANG STREQUAL "HIP")
  message(STATUS "Enabling C extension.")
  add_dependencies(default _C)

  message(STATUS "Enabling moe extension.")
  add_dependencies(default _moe_C)

endif()

if(APHRODITE_GPU_LANG STREQUAL "HIP")
  message(STATUS "Enabling rocm extension.")
  add_dependencies(default _rocm_C)
endif()
