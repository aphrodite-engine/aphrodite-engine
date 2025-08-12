cmake_minimum_required(VERSION 3.26)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  message(FATAL_ERROR "MPS extension can only be built on macOS.")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/utils.cmake)

find_package(Torch REQUIRED)
find_package(Python REQUIRED COMPONENTS Development)

enable_language(OBJCXX)

find_library(FOUNDATION_FRAMEWORK Foundation)
find_library(METAL_FRAMEWORK Metal)
find_library(METALKIT_FRAMEWORK MetalKit)

include_directories("${CMAKE_SOURCE_DIR}/kernels")


set(METAL_SRC
  ${CMAKE_SOURCE_DIR}/kernels/metal/attention/attention/paged_attention.metal
  ${CMAKE_SOURCE_DIR}/kernels/metal/attention/float8.metal
  ${CMAKE_SOURCE_DIR}/kernels/metal/attention/utils.metal
  ${CMAKE_SOURCE_DIR}/kernels/metal/attention/cache/reshape_and_cache.metal
  ${CMAKE_SOURCE_DIR}/kernels/metal/attention/cache/copy_blocks.metal
  ${CMAKE_SOURCE_DIR}/kernels/metal/attention/convert_fp8.metal
)

set(METALLIB ${CMAKE_CURRENT_BINARY_DIR}/aphrodite_paged_attention.metallib)


set(METAL_AIR_FILES)
foreach(SRC ${METAL_SRC})
  get_filename_component(SRC_NAME ${SRC} NAME_WE)
  get_filename_component(SRC_DIR ${SRC} DIRECTORY)
  set(AIR_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SRC_NAME}.air)
  add_custom_command(
    OUTPUT ${AIR_FILE}
    COMMAND xcrun -sdk macosx metal -std=macos-metal2.4 -O3 -gline-tables-only -c ${SRC} -o ${AIR_FILE}
    DEPENDS ${SRC}
    VERBATIM
    COMMENT "Compiling Metal source ${SRC_NAME}.metal -> ${SRC_NAME}.air"
  )
  list(APPEND METAL_AIR_FILES ${AIR_FILE})
endforeach()


add_custom_command(
  OUTPUT ${METALLIB}
  COMMAND xcrun -sdk macosx metallib ${METAL_AIR_FILES} -o ${METALLIB}
  DEPENDS ${METAL_AIR_FILES}
  VERBATIM
  COMMENT "Linking Metal objects -> ${METALLIB}"
)

add_custom_target(aphrodite_metal_lib ALL DEPENDS ${METALLIB})


add_compile_definitions(METALLIB_PATH="aphrodite_paged_attention.metallib")

set(MPS_EXT_SRCS
  ${CMAKE_SOURCE_DIR}/kernels/metal/attention/paged_attention.mm
  ${CMAKE_SOURCE_DIR}/kernels/metal/attention/cache.mm
  ${CMAKE_SOURCE_DIR}/kernels/metal/attention/convert_fp8.mm
  ${CMAKE_SOURCE_DIR}/kernels/metal/attention/device.mm
  ${CMAKE_SOURCE_DIR}/kernels/metal/torch_bindings.cpp
)


add_library(_C MODULE ${MPS_EXT_SRCS})
target_compile_features(_C PRIVATE cxx_std_17)
add_dependencies(_C aphrodite_metal_lib)

target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} Python::Python ${FOUNDATION_FRAMEWORK} ${METAL_FRAMEWORK} ${METALKIT_FRAMEWORK})
target_include_directories(_C PRIVATE ${Python_INCLUDE_DIRS})
set_target_properties(_C PROPERTIES PREFIX "" OUTPUT_NAME "_C")

target_compile_definitions(_C PRIVATE -DAPHRODITE_CPU_EXTENSION=0)

install(TARGETS _C DESTINATION aphrodite)


install(FILES ${METALLIB} DESTINATION aphrodite)
