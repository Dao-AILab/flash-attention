// BSD 3 Clause
// Copyright 2023 Advanced Micro Devices, Inc.
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <vector>

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/HIPGeneratorImpl.h>
// #include <c10/hip/HIPGuard.h>
// #include <c10/core/DeviceType.h>

#include "utils.h"

constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

struct QkvParams {
  using index_t = uint32_t;
  
  // The QKV matrices.
  std::vector<const void*> q_ptr; //changed to ck input type
  std::vector<const void*> k_ptr;
  std::vector<const void*> v_ptr;

  // The stride between rows of the Q, K and V matrices.
  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t v_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;

  // The number of heads.
  int h, h_k;
  // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
  // different from nheads (query).
  int h_h_k_ratio; // precompute h / h_k,
};

namespace fwd_device_gemm {
struct FlashFwdParams : public QkvParams {
  // The O matrix (output).
  std::vector<void*> o_ptr;
  
  // The stride between rows of O.
  index_t o_batch_stride;
  index_t o_row_stride;
  index_t o_head_stride;

  // The pointer to the P matrix.
  std::vector<void*> p_ptr;

  // The pointer to the softmax sum.
  std::vector<void*> softmax_lse_ptr;

  // The dimensions.
  int b, seqlen_q, seqlen_k, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded;

  // The scaling factors for the kernel.
  float scale_softmax;
  float scale_softmax_log2;

  // array of length b+1 holding starting offset of each sequence.
  int * __restrict__ cu_seqlens_q;
  int * __restrict__ cu_seqlens_k;

  int *__restrict__ blockmask;

  // The dropout probability (probability of keeping an activation).
  float p_dropout;
  uint8_t p_dropout_in_uint8_t;

  // Scale factor of 1 / (1 - p_dropout).
  float rp_dropout;
  float scale_softmax_rp_dropout;

  // Random state.
  at::PhiloxCudaState philox_args;

  bool is_bf16;
  bool is_causal;
  // bool is_performance_mode;
  // bool is_deterministic;
  // bool is_using_qloop;
};
} // namespace fwd_device_gemm

namespace bwd_device_gemm {
struct FlashBwdParams : public fwd_device_gemm::FlashFwdParams {

  // The dO and dQKV matrices.
  std::vector<const void*> do_ptr;
  std::vector<void*> dq_ptr;
  std::vector<void*> dk_ptr;
  std::vector<void*> dv_ptr;

  // To accumulate dQ
  std::vector<void*> dq_accum_ptr;
  std::vector<void*> dk_accum_ptr;
  std::vector<void*> dv_accum_ptr;

  // // To accumulate dK and dV in case we're splitting the bwd along seqlen_q
  // dimension void *__restrict__ dk_accum_ptr; void *__restrict__
  // dv_accum_ptr;

  // The stride between rows of the dO, dQ, dK and dV matrices.
  // TD [2022-04-16]: We're using 32-bit indexing to save registers.
  // The code probably won't work for arrays larger than 2GB.
  index_t do_batch_stride;
  index_t do_row_stride;
  index_t do_head_stride;
  index_t dq_batch_stride;
  index_t dk_batch_stride;
  index_t dv_batch_stride;
  index_t dq_row_stride;
  index_t dk_row_stride;
  index_t dv_row_stride;
  index_t dq_head_stride;
  index_t dk_head_stride;
  index_t dv_head_stride;

  // The pointer to the softmax d sum.
  std::vector<void*> dsoftmax_sum;
};
} // namespace bwd_device_gemm

// template<typename KernelParams>
// struct LaunchParams{
//   LaunchParams(hipDeviceProp_t *props,
//                hipStream_t stream,
//                bool is_dropout,
//                bool return_softmax)
//       : elts_per_thread_(0),
//         props_(props), 
//         stream_(stream), 
//         is_dropout_(is_dropout), 
//         return_softmax_(return_softmax) {}

//   size_t elts_per_thread_;
//   hipDeviceProp_t * props_;
//   hipStream_t stream_;
  
//   bool is_dropout_;
//   bool return_softmax_;

//   KernelParams params;
//   int num_full_heads;
//   int num_main_groups;
//   int heads_last_wave;
//   int main_steps;
//   int rest_steps;
// };