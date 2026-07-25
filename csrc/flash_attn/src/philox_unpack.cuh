// This is purely so that it works with torch 2.1. For torch 2.2+ we can include ATen/cuda/PhiloxUtils.cuh

#pragma once
#include <ATen/cuda/PhiloxCudaState.h>     // For at::PhiloxCudaState
#include <ATen/cuda/detail/UnpackRaw.cuh>  // For at::cuda::philox::unpack
