cmake_minimum_required(VERSION 3.26)

project(aphrodite_kernels_redirect NONE)

message(FATAL_ERROR
  "The native build has moved to the repository root. "
  "Run `python tools/generate_cmake_presets.py && cmake --preset release && cmake --build --preset release --target install` "
  "from the aphrodite-engine root directory.")
