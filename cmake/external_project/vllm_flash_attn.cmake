# vLLM flash attention requires APHRODITE_GPU_ARCHES to contain the set of target
# arches in the CMake syntax (75-real, 89-virtual, etc), since we clear the
# arches in the CUDA case (and instead set the gencodes on a per file basis)
# we need to manually set APHRODITE_GPU_ARCHES here.
if(APHRODITE_GPU_LANG STREQUAL "CUDA")
  foreach(_ARCH ${CUDA_ARCHS})
    string(REPLACE "." "" _ARCH "${_ARCH}")
    list(APPEND APHRODITE_GPU_ARCHES "${_ARCH}-real")
  endforeach()
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
          GIT_REPOSITORY https://github.com/vllm-project/flash-attention.git
          GIT_TAG 8f468e7da54a8e2f98abfa7c38636aac91c0cba1
          GIT_PROGRESS TRUE
          # Don't share the aphrodite-flash-attn build between build types
          BINARY_DIR ${CMAKE_BINARY_DIR}/aphrodite_flash_attn
  )
endif()

# Ensure the aphrodite/aphrodite_flash_attn directory exists before installation
install(CODE "file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/aphrodite/aphrodite_flash_attn\")" ALL_COMPONENTS)

# Fetch the aphrodite-flash-attn library
FetchContent_MakeAvailable(aphrodite-flash-attn)
message(STATUS "aphrodite-flash-attn is available at ${aphrodite-flash-attn_SOURCE_DIR}")

# Override the installation destination for the aphrodite-flash-attn targets
# so they install to the correct location that setuptools expects
if(TARGET _vllm_fa2_C)
    set_target_properties(_vllm_fa2_C PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/aphrodite/aphrodite_flash_attn)
    install(TARGETS _vllm_fa2_C 
            LIBRARY DESTINATION aphrodite/aphrodite_flash_attn 
            COMPONENT _vllm_fa2_C)
endif()

if(TARGET _vllm_fa3_C)
    set_target_properties(_vllm_fa3_C PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/aphrodite/aphrodite_flash_attn)
    install(TARGETS _vllm_fa3_C 
            LIBRARY DESTINATION aphrodite/aphrodite_flash_attn 
            COMPONENT _vllm_fa3_C)
endif()

# Copy over the aphrodite-flash-attn python files (duplicated for fa2 and fa3, in
# case only one is built, in the case both are built redundant work is done)
install(
  DIRECTORY ${aphrodite-flash-attn_SOURCE_DIR}/vllm_flash_attn/
  DESTINATION aphrodite/aphrodite_flash_attn
  COMPONENT _vllm_fa2_C
)

install(
  DIRECTORY ${aphrodite-flash-attn_SOURCE_DIR}/vllm_flash_attn/
  DESTINATION aphrodite/aphrodite_flash_attn
  COMPONENT _vllm_fa3_C
)
