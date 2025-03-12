cmake_minimum_required(VERSION 3.15)
project(c_utils)

find_package(pybind11 REQUIRED)
pybind11_add_module(c_utils utils_binding.cpp utils.c)