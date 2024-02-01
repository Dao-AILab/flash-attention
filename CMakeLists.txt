cmake_minimum_required(VERSION 3.18)

# Set CMAKE_C_FLAGS_RELEASE without /DNDEBUG
set(CMAKE_C_FLAGS_RELEASE "/MD /O2 /Ob2 /DDLL_EXPORT /DNOT_PY_BUILD" CACHE STRING "Release flags" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Ob2 /DDLL_EXPORT /DNOT_PY_BUILD" CACHE STRING "Release flags" FORCE)
set(CMAKE_CUDA_FLAGS_RELEASE "/MD /O2 /Ob2 /DDLL_EXPORT /DNOT_PY_BUILD" CACHE STRING "Release flags" FORCE)

project(FlashAttention)

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED PATHS ${LIBTORCH_PATH})

if(NOT CUDA_VERSION VERSION_GREATER_EQUAL "11.6")
  message(FATAL_ERROR "CUDA version must be at least 11.6")
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++17;-O3;-U__CUDA_NO_HALF_OPERATORS__;-U__CUDA_NO_HALF_CONVERSIONS__;-U__CUDA_NO_HALF2_OPERATORS__;-U__CUDA_NO_BFLOAT16_CONVERSIONS__;--expt-relaxed-constexpr;--expt-extended-lambda;--use_fast_math;--threads;24;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_90,code=sm_90;)

if(CUDA_VERSION VERSION_GREATER "11.8")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_90,code=sm_90)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/src
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass/include
    ${CUDA_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

cuda_add_library(flash_attn SHARED
    csrc/flash_attn/flash_api.cpp
    csrc/flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim96_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim96_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim160_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim160_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim224_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim224_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim256_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim32_fp16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim96_fp16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim96_bf16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim160_fp16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim160_bf16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim192_fp16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim192_bf16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim224_fp16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim224_bf16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim256_fp16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim256_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim160_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim160_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim224_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim224_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_sm80.cu
)

target_link_libraries(flash_attn "${TORCH_LIBRARIES}")
set_property(TARGET flash_attn PROPERTY CXX_STANDARD 17)
