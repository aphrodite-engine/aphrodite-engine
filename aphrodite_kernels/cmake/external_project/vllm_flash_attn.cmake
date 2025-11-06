# vLLM flash attention requires APHRODITE_GPU_ARCHES to contain the set of target
# arches in the CMake syntax (75-real, 89-virtual, etc), since we clear the
# arches in the CUDA case (and instead set the gencodes on a per file basis)
# we need to manually set APHRODITE_GPU_ARCHES here.
if(APHRODITE_GPU_LANG STREQUAL "CUDA")
  foreach(_ARCH ${CUDA_ARCHS})
    string(REPLACE "." "" _ARCH "${_ARCH}")
    list(APPEND APHRODITE_GPU_ARCHES "${_ARCH}-real")
  endforeach()
  
  # Filter CUDA_ARCHS for FA3: only include 9.0 if present and no higher architectures exist
  # FA3 should only be built for sm90, not for higher architectures like sm100/sm120
  # Store original CUDA_ARCHS and create filtered version for vllm_flash_attn
  set(_ORIGINAL_CUDA_ARCHS ${CUDA_ARCHS})
  set(_HAS_HIGHER_ARCH FALSE)
  foreach(_ARCH ${CUDA_ARCHS})
    # Remove any suffix (like 'a' or 'f') to get the base version
    string(REGEX REPLACE "^([0-9]+\\.[0-9]+).*" "\\1" _BASE_ARCH "${_ARCH}")
    if(_BASE_ARCH VERSION_GREATER_EQUAL "10.0")
      set(_HAS_HIGHER_ARCH TRUE)
      break()
    endif()
  endforeach()

  # If there are higher architectures, disable FA3 entirely
  # FA3 should only be built for sm90, not for higher architectures like sm100/sm120
  if(_HAS_HIGHER_ARCH)
    set(_FILTERED_CUDA_ARCHS ${CUDA_ARCHS})
    list(REMOVE_ITEM _FILTERED_CUDA_ARCHS "9.0")
    list(REMOVE_ITEM _FILTERED_CUDA_ARCHS "9.0a")
    # Temporarily override CUDA_ARCHS for the vllm_flash_attn subproject
    set(CUDA_ARCHS ${_FILTERED_CUDA_ARCHS})
    # Disable FA3_ENABLED to prevent the target from being created
    set(FA3_ENABLED OFF CACHE BOOL "Disable FA3 when higher architectures are present" FORCE)
    set(_FA3_WAS_DISABLED TRUE)
  else()
    set(_FA3_WAS_DISABLED FALSE)
  endif()
endif()

#
# Build vLLM flash attention from source
#
# IMPORTANT: This has to be the last thing we do, because aphrodite-flash-attn uses the same macros/functions as vLLM.
# Because functions all belong to the global scope, aphrodite-flash-attn's functions overwrite vLLMs.
# They should be identical but if they aren't this is a massive footgun.
#
# The aphrodite-flash-attn install rules are nested under aphrodite to make sure the library gets installed in the correct place.
# To only install aphrodite-flash-attn, use --component _aphrodite_fa2_C (for FA2) or --component _aphrodite_fa3_C (for FA3).
# If no component is specified, aphrodite-flash-attn is still installed.

# If APHRODITE_FLASH_ATTN_SRC_DIR is set, aphrodite-flash-attn is installed from that directory instead of downloading.
# This is to enable local development of aphrodite-flash-attn within vLLM.
# It can be set as an environment variable or passed as a cmake argument.
# The environment variable takes precedence.
if (DEFINED ENV{APHRODITE_FLASH_ATTN_SRC_DIR})
  set(APHRODITE_FLASH_ATTN_SRC_DIR $ENV{APHRODITE_FLASH_ATTN_SRC_DIR})
endif()

if(APHRODITE_FLASH_ATTN_SRC_DIR)
  FetchContent_Declare(
          aphrodite-flash-attn SOURCE_DIR 
          ${APHRODITE_FLASH_ATTN_SRC_DIR}
          BINARY_DIR ${CMAKE_BINARY_DIR}/aphrodite_flash_attn
  )
else()
  FetchContent_Declare(
          aphrodite-flash-attn
          GIT_REPOSITORY https://github.com/AlpinDale/flash-attention.git
          GIT_TAG 3d9959a77d8f5d9dcbc67c7346cbe52643264592
          GIT_PROGRESS TRUE
          # Don't share the aphrodite-flash-attn build between build types
          BINARY_DIR ${CMAKE_BINARY_DIR}/aphrodite_flash_attn
  )
endif()

# Ensure the aphrodite_kernels/aphrodite_flash_attn directory exists before installation
install(CODE "file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/aphrodite_kernels/aphrodite_flash_attn\")" ALL_COMPONENTS)

# Pass CUTLASS variables to the flash-attention subproject
# CUTLASS should already be available from the main CMakeLists.txt
if(APHRODITE_GPU_LANG STREQUAL "CUDA")
  # Try to find CUTLASS include directory from various possible variables
  set(_CUTLASS_INCLUDE_DIR "")
  if(DEFINED CUTLASS_INCLUDE_DIR)
    set(_CUTLASS_INCLUDE_DIR ${CUTLASS_INCLUDE_DIR})
  elseif(DEFINED CUTLASS_DIR)
    set(_CUTLASS_INCLUDE_DIR ${CUTLASS_DIR}/include)
  elseif(DEFINED cutlass_SOURCE_DIR)
    set(_CUTLASS_INCLUDE_DIR ${cutlass_SOURCE_DIR}/include)
  endif()

  if(_CUTLASS_INCLUDE_DIR)
    set(CUTLASS_INCLUDE_DIR ${_CUTLASS_INCLUDE_DIR} CACHE PATH "Path to CUTLASS include directory" FORCE)
    message(STATUS "Setting CUTLASS_INCLUDE_DIR for flash-attention: ${CUTLASS_INCLUDE_DIR}")
  endif()
  unset(_CUTLASS_INCLUDE_DIR)
endif()

# Fetch the aphrodite-flash-attn library
FetchContent_MakeAvailable(aphrodite-flash-attn)
message(STATUS "aphrodite-flash-attn is available at ${aphrodite-flash-attn_SOURCE_DIR}")

# Restore original CUDA_ARCHS and FA3_ENABLED if they were modified
if(APHRODITE_GPU_LANG STREQUAL "CUDA" AND DEFINED _ORIGINAL_CUDA_ARCHS)
  set(CUDA_ARCHS ${_ORIGINAL_CUDA_ARCHS})
  # Restore FA3_ENABLED if we disabled it
  if(_FA3_WAS_DISABLED)
    set(FA3_ENABLED ON CACHE BOOL "Enable FA3" FORCE)
  endif()
  unset(_ORIGINAL_CUDA_ARCHS)
  unset(_FILTERED_CUDA_ARCHS)
  unset(_HAS_HIGHER_ARCH)
  unset(_FA3_WAS_DISABLED)
endif()

# Override the installation destination for the aphrodite-flash-attn targets
# so they install to the correct location that setuptools expects
# Also add CUTLASS include directories from the parent project
if(TARGET _vllm_fa2_C)
    set_target_properties(_vllm_fa2_C PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/aphrodite_kernels/aphrodite_flash_attn)
    # Add parent project's CUTLASS include directories
    if(APHRODITE_GPU_LANG STREQUAL "CUDA")
        if(DEFINED CUTLASS_INCLUDE_DIR)
            target_include_directories(_vllm_fa2_C PRIVATE ${CUTLASS_INCLUDE_DIR})
        elseif(DEFINED CUTLASS_DIR)
            target_include_directories(_vllm_fa2_C PRIVATE ${CUTLASS_DIR}/include)
        elseif(DEFINED cutlass_SOURCE_DIR)
            target_include_directories(_vllm_fa2_C PRIVATE ${cutlass_SOURCE_DIR}/include)
        endif()
    endif()
    install(TARGETS _vllm_fa2_C 
            LIBRARY DESTINATION aphrodite_kernels/aphrodite_flash_attn 
            COMPONENT _vllm_fa2_C)
endif()

if(TARGET _vllm_fa3_C)
    set_target_properties(_vllm_fa3_C PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/aphrodite_kernels/aphrodite_flash_attn)
    # Add parent project's CUTLASS include directories
    if(APHRODITE_GPU_LANG STREQUAL "CUDA")
        if(DEFINED CUTLASS_INCLUDE_DIR)
            target_include_directories(_vllm_fa3_C PRIVATE ${CUTLASS_INCLUDE_DIR})
        elseif(DEFINED CUTLASS_DIR)
            target_include_directories(_vllm_fa3_C PRIVATE ${CUTLASS_DIR}/include)
        elseif(DEFINED cutlass_SOURCE_DIR)
            target_include_directories(_vllm_fa3_C PRIVATE ${cutlass_SOURCE_DIR}/include)
        endif()
    endif()
    install(TARGETS _vllm_fa3_C 
            LIBRARY DESTINATION aphrodite_kernels/aphrodite_flash_attn 
            COMPONENT _vllm_fa3_C)
endif()

# Copy over the aphrodite-flash-attn python files (duplicated for fa2 and fa3, in
# case only one is built, in the case both are built redundant work is done)
install(
  DIRECTORY ${aphrodite-flash-attn_SOURCE_DIR}/vllm_flash_attn/
  DESTINATION aphrodite_kernels/aphrodite_flash_attn
  COMPONENT _vllm_fa2_C
)

install(
  DIRECTORY ${aphrodite-flash-attn_SOURCE_DIR}/vllm_flash_attn/
  DESTINATION aphrodite_kernels/aphrodite_flash_attn
  COMPONENT _vllm_fa3_C
)
