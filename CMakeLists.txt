cmake_minimum_required(VERSION 3.26)

project(vllm_flash_attn LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FA2_ENABLED ON)
set(FA3_ENABLED ON)

# CUDA by default, can be overridden by using -DVLLM_TARGET_DEVICE=... (used by setup.py)
set(VLLM_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for vLLM")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${VLLM_TARGET_DEVICE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${VLLM_PYTHON_PATH}")

# Supported python versions. These should be kept in sync with setup.py.
set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11" "3.12")

# Supported NVIDIA architectures.
set(CUDA_SUPPORTED_ARCHS "8.0;8.6;8.9;9.0")

# Supported AMD GPU architectures.
set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100")

#
# Supported/expected torch versions for CUDA/ROCm.
#
# Currently, having an incorrect pytorch version results in a warning
# rather than an error.
#
# Note: these should be kept in sync with the torch version in setup.py.
# Likely should also be in sync with the vLLM version.
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
    set(VLLM_GPU_LANG "CUDA")

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
    message(FATAL_ERROR "ROCm build is not currently supported for vllm-flash-attn.")
else ()
    message(FATAL_ERROR "Can't find CUDA or HIP installation.")
endif ()

#
# Query torch for additional GPU compilation flags for the given
# `VLLM_GPU_LANG`.
# The final set of arches is stored in `VLLM_FA_GPU_FLAGS`.
#
get_torch_gpu_compiler_flags(VLLM_FA_GPU_FLAGS ${VLLM_GPU_LANG})

#
# Set nvcc parallelism.
#
if (NVCC_THREADS AND VLLM_GPU_LANG STREQUAL "CUDA")
    list(APPEND VLLM_FA_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif ()


# Other flags
list(APPEND VLLM_FA_GPU_FLAGS --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math)

# If CUTLASS is compiled on NVCC >= 12.5, it by default uses
# cudaGetDriverEntryPointByVersion as a wrapper to avoid directly calling the
# driver API. This causes problems when linking with earlier versions of CUDA.
# Setting this variable sidesteps the issue by calling the driver directly.
list(APPEND VLLM_FA_GPU_FLAGS -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)

# Replace instead of appending, nvcc doesn't like duplicate -O flags.
string(REPLACE "-O2" "-O3" CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")

#
# _C extension
#

if (FA2_ENABLED)
    file(GLOB FA2_GEN_SRCS "csrc/flash_attn/src/flash_fwd_*.cu")

    # For CUDA we set the architectures on a per file basis
    if (VLLM_GPU_LANG STREQUAL "CUDA")
        cuda_archs_loose_intersection(FA2_ARCHS "8.0;9.0" "${CUDA_ARCHS}")
        message(STATUS "FA2_ARCHS: ${FA2_ARCHS}")

        set_gencode_flags_for_srcs(
            SRCS "${FA2_GEN_SRCS}"
            CUDA_ARCHS "${FA2_ARCHS}")
    endif()

    define_gpu_extension_target(
        _vllm_fa2_C
        DESTINATION vllm_flash_attn
        LANGUAGE ${VLLM_GPU_LANG}
        SOURCES
            csrc/flash_attn/flash_api.cpp
            csrc/flash_attn/flash_api_sparse.cpp
            csrc/flash_attn/flash_api_torch_lib.cpp
            ${FA2_GEN_SRCS}
        COMPILE_FLAGS ${VLLM_FA_GPU_FLAGS}
        USE_SABI 3
        WITH_SOABI)

    target_include_directories(_vllm_fa2_C PRIVATE
        csrc/flash_attn
        csrc/flash_attn/src
        csrc/common
        csrc/cutlass/include)

    # custom definitions
    target_compile_definitions(_vllm_fa2_C PRIVATE
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
    # BF16 source files (we only use paged, split, packgqa and sm80)
    #   we only use paged, split and packgqa since for paged_kv or varlen_q 
    #   PackedGQA is the only one that is used
    file(GLOB FA3_BF16_GEN_SRCS 
        "hopper/instantiations/flash_fwd_hdimall_bf16_*paged*_sm90.cu")
    file(GLOB FA3_BF16_GEN_SRCS_ 
        "hopper/instantiations/flash_fwd_hdimall_bf16_*packgqa*_sm90.cu")
    list(APPEND FA3_BF16_GEN_SRCS ${FA3_BF16_GEN_SRCS_})
    file(GLOB FA3_BF16_GEN_SRCS_ 
        "hopper/instantiations/flash_fwd_hdimall_bf16_*split*_sm90.cu")
    list(APPEND FA3_BF16_GEN_SRCS ${FA3_BF16_GEN_SRCS_})
    file(GLOB FA3_BF16_GEN_SRCS_ 
        "hopper/instantiations/flash_fwd_*_bf16_*_sm80.cu")
    list(APPEND FA3_BF16_GEN_SRCS ${FA3_BF16_GEN_SRCS_})
    # FP16 source files (we only use paged, split and packgqa and sm80)
    #   we only use paged, split and packgqa since for paged_kv or varlen_q 
    #   PackedGQA is the only one that is used
    file(GLOB FA3_FP16_GEN_SRCS 
        "hopper/instantiations/flash_fwd_hdimall_fp16_*paged*_sm90.cu")
    file(GLOB FA3_FP16_GEN_SRCS_ 
        "hopper/instantiations/flash_fwd_hdimall_fp16_*packgqa*_sm90.cu")
    list(APPEND FA3_FP16_GEN_SRCS ${FA3_FP16_GEN_SRCS_})
    file(GLOB FA3_FP16_GEN_SRCS_ 
        "hopper/instantiations/flash_fwd_hdimall_fp16_*split*_sm90.cu")
    list(APPEND FA3_FP16_GEN_SRCS ${FA3_FP16_GEN_SRCS_})
    file(GLOB FA3_FP16_GEN_SRCS_ 
        "hopper/instantiations/flash_fwd_*_fp16_*_sm80.cu")
    list(APPEND FA3_FP16_GEN_SRCS ${FA3_FP16_GEN_SRCS_})

    # TODO add fp8 source files when FP8 is enabled
    set(FA3_GEN_SRCS ${FA3_BF16_GEN_SRCS} ${FA3_FP16_GEN_SRCS})

    # For CUDA we set the architectures on a per file basis
    if (VLLM_GPU_LANG STREQUAL "CUDA")
        cuda_archs_loose_intersection(FA3_ARCHS "8.0;9.0a" "${CUDA_ARCHS}")
        message(STATUS "FA3_ARCHS: ${FA3_ARCHS}")

        set_gencode_flags_for_srcs(
            SRCS "${FA3_GEN_SRCS}"
            CUDA_ARCHS "${FA3_ARCHS}")
        set_gencode_flags_for_srcs(
            SRCS "hopper/flash_fwd_combine.cu"
            CUDA_ARCHS "${FA3_ARCHS}")
    endif()

    define_gpu_extension_target(
        _vllm_fa3_C
        DESTINATION vllm_flash_attn
        LANGUAGE ${VLLM_GPU_LANG}
        SOURCES
            hopper/flash_fwd_combine.cu
            hopper/flash_api.cpp
            hopper/flash_api_torch_lib.cpp
            ${FA3_GEN_SRCS}
        COMPILE_FLAGS ${VLLM_FA_GPU_FLAGS}
        ARCHITECTURES ${VLLM_FA_GPU_ARCHES}
        USE_SABI 3
        WITH_SOABI)

    target_include_directories(_vllm_fa3_C PRIVATE
        hopper
        csrc/common
        csrc/cutlass/include)

    # custom definitions
    target_compile_definitions(_vllm_fa3_C PRIVATE
        FLASHATTENTION_DISABLE_BACKWARD
        FLASHATTENTION_DISABLE_DROPOUT
        # FLASHATTENTION_DISABLE_ALIBI
        # FLASHATTENTION_DISABLE_SOFTCAP
        FLASHATTENTION_DISABLE_UNEVEN_K
        # FLASHATTENTION_DISABLE_LOCAL
        FLASHATTENTION_DISABLE_PYBIND
        FLASHATTENTION_DISABLE_FP8 # TODO Enable FP8
        FLASHATTENTION_PACKGQA_ONLY # Custom flag to save on binary size
        FLASHATTENTION_VARLEN_ONLY  # Custom flag to save on binary size
    )
elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS 12.0)
    message(STATUS "FA3 is disabled because CUDA version is not 12.0 or later.")
endif ()