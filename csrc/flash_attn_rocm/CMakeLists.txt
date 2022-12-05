cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(fmha_api)

set(CMAKE_CXX_COMPILER /opt/rocm/hip/bin/hipcc)
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_PREFIX_PATH "/opt/conda/lib/python3.7/site-packages/torch/share/cmake")
find_package(Torch REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel/library/include)
include_directories(/opt/conda/include/python3.7m)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src FLA_SRCS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel/library/src/utility CK_SRCS)

add_executable(fmha_api fmha_api.cpp ${FLA_SRCS} ${CK_SRCS})
target_link_libraries(fmha_api "${TORCH_LIBRARIES}")

message("${TORCH_LIBRARIES}")
