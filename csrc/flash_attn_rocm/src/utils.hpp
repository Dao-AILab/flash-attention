// BSD 3 Clause
// Copyright 2023 Advanced Micro Devices, Inc.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
// HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
// FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>

// torch headers
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/HIPGeneratorImpl.h>
#include <c10/hip/HIPGuard.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "ck/ck.hpp"
#include "ck/library/utility/device_memory.hpp"

#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "hip/hip_runtime.h"

#define CHECK_SHAPE(x, ...)                                                    \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}),                  \
              #x " must have shape (" #__VA_ARGS__ ")")

#define NEW_UNPACK                                                             \
  (TORCH_VERSION_MAJOR * 10000 + TORCH_VERSION_MINOR * 100 +                   \
   TORCH_VERSION_PATCH) > 11300

#define FMHA_CHECK_HIP(call)                                                   \
  do {                                                                         \
    hipError_t status_ = call;                                                 \
    if (status_ != hipSuccess) {                                               \
      fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__,           \
              hipGetErrorString(status_));                                     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

using Index = ck::index_t;

template <typename T>
static inline size_t get_size_in_bytes(size_t n, T dtype) {
  if (dtype == torch::kFloat32) {
    return n * 4;
  } else if (dtype == torch::kBFloat16) {
    return n * 2;
  } else if (dtype == torch::kFloat16) {
    return n * 2;
  } else if (dtype == torch::kInt32) {
    return n * 4;
  } else if (dtype == torch::kInt8) {
    return n;
  }
  return 0;
}

static std::tuple<uint64_t, uint64_t> unpack(at::PhiloxCudaState arg) {
  if (arg.captured_) {
#if NEW_UNPACK
    return std::make_tuple(
        static_cast<uint64_t>(*arg.seed_.ptr),
        static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_));
#else
    return std::make_tuple(
        arg.seed_,
        static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_));
#endif
  } else {
#if NEW_UNPACK
    return std::make_tuple(arg.seed_.val, arg.offset_.val);
#else
    return std::make_tuple(arg.seed_, arg.offset_.val);
#endif
  }
}

// get environment variables for internal usage
static inline bool get_env_(const char *env_var) {
  if (char *value = std::getenv(env_var)) {
    if (strcmp(value, "0") == 0) {
      return false;
    }
    return true;
  }
  return false;
}

// compute differences
static __global__ void compute_differences(const int *in, int *out, int len) {
  int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (i < len) {
    out[i] = in[i + 1] - in[i];
  }
}

// compute seqlens and move to host
static inline std::vector<int> get_host_seqlens(const int *d_seqlens_acc,
                                                int b) {
  int *d_seqlens;
  FMHA_CHECK_HIP(hipMalloc((void **)&d_seqlens, b * sizeof(int)));

  int threadsPerBlock = 256;
  int blocks = (b + threadsPerBlock - 1) / threadsPerBlock;

  compute_differences<<<dim3(blocks), dim3(threadsPerBlock), 0, 0>>>(
      d_seqlens_acc, d_seqlens, b);
  FMHA_CHECK_HIP(hipDeviceSynchronize());

  std::vector<int> h_seqlens(b);

  FMHA_CHECK_HIP(hipMemcpy(h_seqlens.data(), d_seqlens, b * sizeof(int),
                           hipMemcpyDeviceToHost));

  FMHA_CHECK_HIP(hipFree(d_seqlens));

  return h_seqlens;
}