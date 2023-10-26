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

#include "bwd_device_gemm_template.hpp"
#include "params.hpp"

namespace bwd_device_gemm {
template <template <typename> typename DeviceGemmTemplate, typename DeviceGemmTraits>
class DeviceGemmInvoker {
  using Gemm = DeviceGemmTemplate<DeviceGemmTraits>;

 public:
  // constructor for batched gemm
  explicit DeviceGemmInvoker(FlashBwdBatchedParams &params, hipStream_t &stream) {
    auto gemm_ptr = std::make_unique<Gemm>();
    auto invoker_ptr = gemm_ptr->MakeInvokerPointer();
    auto argument_ptr = gemm_ptr->MakeArgumentPointer(
      params.q_ptr,
      params.k_ptr,
      params.z_ptr,
      params.v_ptr,
      params.out_ptr,
      params.softmax_lse_ptr,
      params.dsoftmax_ptr,
      params.dout_ptr, 
      params.dq_ptr,
      params.dk_ptr,
      params.dv_ptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      params.q_lengths,
      params.q_strides,
      params.k_lengths,
      params.k_strides,
      params.z_lengths,
      params.z_strides,
      params.v_lengths,
      params.v_strides,
      params.out_lengths,
      params.out_strides,
      params.lse_lengths,
      params.dk_lengths,
      params.dk_strides,
      params.dv_lengths,
      params.dv_strides,
      {},
      {},
      {},
      {},
      typename DeviceGemmTraits::QElementOp{},
      typename DeviceGemmTraits::KElementOp{},
      typename DeviceGemmTraits::Acc0ElementOp{params.softmax_scale},
      typename DeviceGemmTraits::VElementOp{},
      typename DeviceGemmTraits::OutElementOp{},
      params.p_dropout, 
      params.seeds);

    if (!gemm_ptr->IsSupportedArgument(argument_ptr.get())) {
      throw std::runtime_error(gemm_ptr->GetTypeString() + " does not support this problem");
    }
    auto time_kernel = get_env_("FLASH_ATTENTION_INTERNAL_ENABLE_TIME_KERNEL");
    auto avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{stream, time_kernel});

    if (time_kernel) {
      std::cout << "time elpase is " << avg_time << " ms" << std::endl;
    }
  }

  // constructor for grouped gemm
  explicit DeviceGemmInvoker(FlashBwdGroupedParams &params, hipStream_t &stream) {
    auto gemm_ptr = std::make_unique<Gemm>();
    auto invoker_ptr = gemm_ptr->MakeInvokerPointer();

    std::vector<typename Gemm::ProblemDesc> problem_descs;
    problem_descs.reserve(params.b);

    for (int i = 0; i < params.b; ++i) {
      problem_descs.push_back({
          params.q_lengths_vec[i],
          params.q_strides_vec[i],
          params.k_lengths_vec[i],
          params.k_strides_vec[i],
          params.z_lengths_vec[i],
          params.z_strides_vec[i],
          params.v_lengths_vec[i],
          params.v_strides_vec[i],
          params.out_lengths_vec[i],
          params.out_strides_vec[i],
          params.lse_lengths_vec[i],
          params.lse_strides_vec[i],
          params.dk_lengths_vec[i],
          params.dk_strides_vec[i],
          params.dv_lengths_vec[i],
          params.dv_strides_vec[i],
          {}, // acc0_biases_gs_ms_ns_lengths
          {}, // acc0_biases_gs_ms_ns_strides
          {}, // acc1_biases_gs_ms_os_lengths
          {}  // acc1_biases_gs_ms_os_strides
      });
    }

    TORCH_CHECK(problem_descs.size() == params.q_ptrs.size(), "Wrong q_ptrs size", params.q_ptrs.size());
    TORCH_CHECK(problem_descs.size() == params.k_ptrs.size(), "Wrong k_ptrs size", params.k_ptrs.size());
    TORCH_CHECK(problem_descs.size() == params.z_ptrs.size(), "Wrong z_ptrs size", params.z_ptrs.size());
    TORCH_CHECK(problem_descs.size() == params.v_ptrs.size(), "Wrong v_ptrs size", params.v_ptrs.size());
    TORCH_CHECK(problem_descs.size() == params.bwd_out_ptrs.size(), "Wrong out_ptrs size", params.bwd_out_ptrs.size());
    TORCH_CHECK(problem_descs.size() == params.bwd_softmax_lse_ptrs.size(), "Wrong softmax_lse_ptrs size", params.bwd_softmax_lse_ptrs.size());
    TORCH_CHECK(problem_descs.size() == params.dsoftmax_ptrs.size(), "Wrong dsoftmax_ptrs size", params.dsoftmax_ptrs.size());
    TORCH_CHECK(problem_descs.size() == params.dout_ptrs.size(), "Wrong dout_ptrs size", params.dout_ptrs.size());
    TORCH_CHECK(problem_descs.size() == params.dq_ptrs.size(), "Wrong dq_ptrs size", params.dq_ptrs.size());
    TORCH_CHECK(problem_descs.size() == params.dk_ptrs.size(), "Wrong dk_ptrs size", params.dk_ptrs.size());
    TORCH_CHECK(problem_descs.size() == params.dv_ptrs.size(), "Wrong dv_ptrs size", params.dv_ptrs.size());

    auto argument_ptr = gemm_ptr->MakeArgumentPointer(
        params.q_ptrs,
        params.k_ptrs,
        params.z_ptrs,
        params.v_ptrs,
        params.bwd_out_ptrs,
        params.bwd_softmax_lse_ptrs,
        params.dsoftmax_ptrs,
        params.dout_ptrs, 
        params.dq_ptrs,
        params.dk_ptrs,
        params.dv_ptrs,
        {},
        {},
        {},
        {},
        problem_descs,
        typename DeviceGemmTraits::QElementOp{},
        typename DeviceGemmTraits::KElementOp{},
        typename DeviceGemmTraits::Acc0ElementOp{params.softmax_scale},
        typename DeviceGemmTraits::VElementOp{},
        typename DeviceGemmTraits::OutElementOp{},
        params.p_dropout, 
        params.seeds);

    // specify workspace for problem_desc
    DeviceMem problem_desc_workspace{ gemm_ptr->GetWorkSpaceSize(argument_ptr.get()) };

    gemm_ptr->SetWorkSpacePointer(argument_ptr.get(),
                            problem_desc_workspace.GetDeviceBuffer());

    if (!gemm_ptr->IsSupportedArgument(argument_ptr.get())) {
      throw std::runtime_error(gemm_ptr->GetTypeString() + " does not support this problem");
    }
    auto time_kernel = get_env_("FLASH_ATTENTION_INTERNAL_ENABLE_TIME_KERNEL");
    auto avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{stream, time_kernel});

    if (time_kernel) {
      std::cout << "time elpase is " << avg_time << " ms" << std::endl;
    }
  }
};
} // namespace bwd_device_gemm