cmake_minimum_required(VERSION 3.26)

project(aphrodite_flash_attn LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FA2_ENABLED ON)
set(FA3_ENABLED ON)

# CUDA by default, can be overridden by using -DAPHRODITE_TARGET_DEVICE=... (used by setup.py)
set(APHRODITE_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for Aphrodite")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${APHRODITE_TARGET_DEVICE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${APHRODITE_PYTHON_PATH}")

# Supported python versions. These should be kept in sync with setup.py.
set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11" "3.12" "3.13")

# Supported NVIDIA architectures.
set(CUDA_SUPPORTED_ARCHS "8.0;8.6;8.9;9.0")
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
    list(APPEND CUDA_SUPPORTED_ARCHS "10.0" "10.1" "12.0")
endif()

# Supported AMD GPU architectures.
set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100")

#
# Supported/expected torch versions for CUDA/ROCm.
#
# Currently, having an incorrect pytorch version results in a warning
# rather than an error.
#
# Note: these should be kept in sync with the torch version in setup.py.
# Likely should also be in sync with the Aphrodite version.
#
set(TORCH_SUPPORTED_VERSION_CUDA "2.4.0")

find_python_constrained_versions(${PYTHON_SUPPORTED_VERSIONS})

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
message(DEBUG "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

#
# Import torch cmake configuration.
# Torch also imports CUDA (and partially HIP) languages with some customizations,
# so there is no need to do this explicitly with check_language/enable_language,
# etc.
#
if (NOT Torch_FOUND)
    find_package(Torch REQUIRED)
endif()

#
# Set up GPU language and check the torch version and warn if it isn't
# what is expected.
#
if (NOT HIP_FOUND AND CUDA_FOUND)
    set(APHRODITE_GPU_LANG "CUDA")

    # Check CUDA is at least 11.6
    if (CUDA_VERSION VERSION_LESS 11.6)
        message(FATAL_ERROR "CUDA version 11.6 or greater is required.")
    endif ()

    if (NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
        message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} "
                "expected for CUDA build, saw ${Torch_VERSION} instead.")
    endif ()

    #
    # For cuda we want to be able to control which architectures we compile for on
    # a per-file basis in order to cut down on compile time. So here we extract
    # the set of architectures we want to compile for and remove the from the
    # CMAKE_CUDA_FLAGS so that they are not applied globally.
    #
    clear_cuda_arches(CUDA_ARCH_FLAGS)

    if (NOT CUDA_ARCHS)
        extract_unique_cuda_archs_ascending(CUDA_ARCHS "${CUDA_ARCH_FLAGS}")
    endif()
    message(STATUS "CUDA target architectures: ${CUDA_ARCHS}")
    # Filter the target architectures by the supported supported archs
    # since for some files we will build for all CUDA_ARCHS.
    cuda_archs_loose_intersection(CUDA_ARCHS
        "${CUDA_SUPPORTED_ARCHS}" "${CUDA_ARCHS}")
    message(STATUS "CUDA supported target architectures: ${CUDA_ARCHS}")
elseif (HIP_FOUND)
    message(FATAL_ERROR "ROCm build is not currently supported for aphrodite-flash-attn.")
else ()
    message(FATAL_ERROR "Can't find CUDA or HIP installation.")
endif ()

#
# Query torch for additional GPU compilation flags for the given
# `APHRODITE_GPU_LANG`.
# The final set of arches is stored in `APHRODITE_FA_GPU_FLAGS`.
#
get_torch_gpu_compiler_flags(APHRODITE_FA_GPU_FLAGS ${APHRODITE_GPU_LANG})

#
# Set nvcc parallelism.
#
if (NVCC_THREADS AND APHRODITE_GPU_LANG STREQUAL "CUDA")
    list(APPEND APHRODITE_FA_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif ()


# Other flags
list(APPEND APHRODITE_FA_GPU_FLAGS --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math)

# If CUTLASS is compiled on NVCC >= 12.5, it by default uses
# cudaGetDriverEntryPointByVersion as a wrapper to avoid directly calling the
# driver API. This causes problems when linking with earlier versions of CUDA.
# Setting this variable sidesteps the issue by calling the driver directly.
list(APPEND APHRODITE_FA_GPU_FLAGS -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)

# Replace instead of appending, nvcc doesn't like duplicate -O flags.
string(REPLACE "-O2" "-O3" CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")

#
# _C extension
#

if (FA2_ENABLED)
    file(GLOB FA2_GEN_SRCS "flash_attn/src/flash_fwd_*.cu")

    # For CUDA we set the architectures on a per file basis
    if (APHRODITE_GPU_LANG STREQUAL "CUDA")
        cuda_archs_loose_intersection(FA2_ARCHS "8.0;9.0;10.0;10.1;12.0" "${CUDA_ARCHS}")
        message(STATUS "FA2_ARCHS: ${FA2_ARCHS}")

        set_gencode_flags_for_srcs(
            SRCS "${FA2_GEN_SRCS}"
            CUDA_ARCHS "${FA2_ARCHS}")
    endif()

    define_gpu_extension_target(
        _aphrodite_fa2_C
        DESTINATION aphrodite_flash_attn
        LANGUAGE ${APHRODITE_GPU_LANG}
        SOURCES
            flash_attn/flash_api.cpp
            flash_attn/flash_api_sparse.cpp
            flash_attn/flash_api_torch_lib.cpp
            ${FA2_GEN_SRCS}
        COMPILE_FLAGS ${APHRODITE_FA_GPU_FLAGS}
        USE_SABI 3
        WITH_SOABI)

    target_include_directories(_aphrodite_fa2_C PRIVATE
        flash_attn
        flash_attn/src
        common
        cutlass/include)

    # custom definitions
    target_compile_definitions(_aphrodite_fa2_C PRIVATE
        FLASHATTENTION_DISABLE_BACKWARD
        FLASHATTENTION_DISABLE_DROPOUT
        # FLASHATTENTION_DISABLE_ALIBI
        # FLASHATTENTION_DISABLE_SOFTCAP
        FLASHATTENTION_DISABLE_UNEVEN_K
        # FLASHATTENTION_DISABLE_LOCAL
        FLASHATTENTION_DISABLE_PYBIND
    )
endif ()

# FA3 requires CUDA 12.0 or later
if (FA3_ENABLED AND ${CMAKE_CUDA_COMPILER_VERSION} GREATER_EQUAL 12.0)
    # BF16 source files
    file(GLOB FA3_BF16_GEN_SRCS
        "hopper/instantiations/flash_fwd_hdimall_bf16*_sm90.cu")
    file(GLOB FA3_BF16_GEN_SRCS_
        "hopper/instantiations/flash_fwd_hdimdiff_bf16*_sm90.cu")
    list(APPEND FA3_BF16_GEN_SRCS ${FA3_BF16_GEN_SRCS_})
    file(GLOB FA3_BF16_GEN_SRCS_
        "hopper/instantiations/flash_fwd_*_bf16_*_sm80.cu")
    list(APPEND FA3_BF16_GEN_SRCS ${FA3_BF16_GEN_SRCS_})
    # FP16 source files
    file(GLOB FA3_FP16_GEN_SRCS
        "hopper/instantiations/flash_fwd_hdimall_fp16*_sm90.cu")
    file(GLOB FA3_FP16_GEN_SRCS_
        "hopper/instantiations/flash_fwd_hdimdiff_fp16*_sm90.cu")
    list(APPEND FA3_FP16_GEN_SRCS ${FA3_FP16_GEN_SRCS_})
    file(GLOB FA3_FP16_GEN_SRCS_
        "hopper/instantiations/flash_fwd_*_fp16_*_sm80.cu")
    list(APPEND FA3_FP16_GEN_SRCS ${FA3_FP16_GEN_SRCS_})
    # FP8 source files
    file(GLOB FA3_FP8_GEN_SRCS
        "hopper/instantiations/flash_fwd_hdimall_e4m3*_sm90.cu")
    file(GLOB FA3_FP8_GEN_SRCS_
        "hopper/instantiations/flash_fwd_hdimdiff_e4m3*_sm90.cu")
    list(APPEND FA3_FP8_GEN_SRCS ${FA3_FP8_GEN_SRCS_})

    set(FA3_GEN_SRCS ${FA3_BF16_GEN_SRCS} ${FA3_FP16_GEN_SRCS} ${FA3_FP8_GEN_SRCS})

    # For CUDA we set the architectures on a per file basis
    # FaV3 is not yet supported in Blackwell
    if (APHRODITE_GPU_LANG STREQUAL "CUDA")
        cuda_archs_loose_intersection(FA3_ARCHS "8.0;9.0a;" "${CUDA_ARCHS}")
        message(STATUS "FA3_ARCHS: ${FA3_ARCHS}")

        set_gencode_flags_for_srcs(
            SRCS "${FA3_GEN_SRCS}"
            CUDA_ARCHS "${FA3_ARCHS}")
        set_gencode_flags_for_srcs(
            SRCS 
                hopper/flash_fwd_combine.cu
                hopper/flash_prepare_scheduler.cu
            CUDA_ARCHS "${FA3_ARCHS}")
    endif()

    define_gpu_extension_target(
        _aphrodite_fa3_C
        DESTINATION aphrodite_flash_attn
        LANGUAGE ${APHRODITE_GPU_LANG}
        SOURCES
            hopper/flash_fwd_combine.cu
            hopper/flash_prepare_scheduler.cu
            hopper/flash_api.cpp
            hopper/flash_api_torch_lib.cpp
            ${FA3_GEN_SRCS}
        COMPILE_FLAGS ${APHRODITE_FA_GPU_FLAGS}
        ARCHITECTURES ${APHRODITE_FA_GPU_ARCHES}
        USE_SABI 3
        WITH_SOABI)

    target_include_directories(_aphrodite_fa3_C PRIVATE
        hopper
        common
        cutlass/include)

    # custom definitions
    target_compile_definitions(_aphrodite_fa3_C PRIVATE
        FLASHATTENTION_DISABLE_BACKWARD
        FLASHATTENTION_DISABLE_DROPOUT
        # FLASHATTENTION_DISABLE_ALIBI
        # FLASHATTENTION_DISABLE_SOFTCAP
        FLASHATTENTION_DISABLE_UNEVEN_K
        # FLASHATTENTION_DISABLE_LOCAL
        FLASHATTENTION_DISABLE_PYBIND
        FLASHATTENTION_VARLEN_ONLY # Custom flag to save on binary size
    )
elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS 12.0)
    message(STATUS "FA3 is disabled because CUDA version is not 12.0 or later.")
endif ()