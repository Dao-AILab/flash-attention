/******************************************************************************
 * Copyright (c) 2026, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda_runtime.h>

#ifdef TORCH_TARGET_VERSION
#include <torch/csrc/stable/macros.h>
#define FLASHATTENTION_CUDA_CHECK(EXPR) STD_CUDA_CHECK(EXPR)
#define FLASHATTENTION_CUDA_KERNEL_LAUNCH_CHECK() STD_CUDA_KERNEL_LAUNCH_CHECK()
#else
#include <c10/cuda/CUDAException.h>
#define FLASHATTENTION_CUDA_CHECK(EXPR) C10_CUDA_CHECK(EXPR)
#define FLASHATTENTION_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif
