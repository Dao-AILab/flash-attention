cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_CXX_COMPILER "/usr/bin/hipcc")

project(fmha_api)

include_directories

find_package(Torch REQUIRED)

add_executable(fmha_api fmha_api.cpp)
target_link_libraries(fmha_api "${TORCH_LIBRARIES}")
set_property(TARGET fmha_api PROPERTY CXX_STANDARD 14)


