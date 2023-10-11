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
  using DeviceOp = DeviceGemmTemplate<DeviceGemmTraits>;

 public:
  // constructor for batched gemm
  explicit DeviceGemmInvoker(FlashBwdBatchedParams &params, hipStream_t &stream)
    : device_op_ptr(std::make_unique<DeviceOp>()),
      invoker_ptr(device_op_ptr->MakeInvokerPointer()),
      argument_ptr(device_op_ptr->MakeArgumentPointer(
        params.q_ptr,
        params.k_ptr,
        params.z_ptr,
        params.v_ptr,
        params.out_ptr,
        params.softmax_lse_ptr,
        params.d_ptr,
        params.dout_ptr, 
        params.dq_ptr,
        params.dk_ptr,
        params.dv_ptr,
        nullptr,
        nullptr,
        params.q_gs_ms_ks_lengths,
        params.q_gs_ms_ks_strides,
        params.k_gs_ns_ks_lengths,
        params.k_gs_ns_ks_strides,
        params.z_gs_ms_ns_lengths,
        params.z_gs_ms_ns_strides,
        params.v_gs_gemm1ns_gemm1ks_lengths,
        params.v_gs_gemm1ns_gemm1ks_strides,
        params.out_gs_ms_gemm1ns_lengths,
        params.out_gs_ms_gemm1ns_strides,
        params.lse_gs_ms_lengths,
        {},
        {},
        {},
        {},
        device_gemm_trait::AElementOp{},
        device_gemm_trait::B0ElementOp{},
        device_gemm_trait::Acc0ElementOp{params.softmax_scale},
        device_gemm_trait::B1ElementOp{},
        device_gemm_trait::CElementOp{},
        params.p_dropout, 
        params.seeds)) {}

  // constructor for grouped gemm
  explicit DeviceGemmInvoker(FlashBwdGroupedParams &params, hipStream_t &stream)
    : device_op_ptr(std::make_unique<DeviceOp>()),
      invoker_ptr(device_op_ptr->MakeInvokerPointer()), {

    std::vector<DeviceOp::ProblemDesc> problem_descs;
    problem_descs.reserve(params.b);

    for (int i = 0; i < params.b; ++i) {
      problem_descs.push_back({
          params.problem_descs[i].q_gs_ms_ks_lengths,
          params.problem_descs[i].q_gs_ms_ks_strides,
          params.problem_descs[i].k_gs_ns_ks_lengths,
          params.problem_descs[i].k_gs_ns_ks_strides,
          params.problem_descs[i].z_gs_ms_ns_lengths,
          params.problem_descs[i].z_gs_ms_ns_strides,
          params.problem_descs[i].v_gs_os_ns_lengths,
          params.problem_descs[i].v_gs_os_ns_strides,
          params.problem_descs[i].out_gs_ms_os_lengths,
          params.problem_descs[i].out_gs_ms_os_strides,
          params.problem_descs[i].lse_gs_ms_lengths,
          params.problem_descs[i].lse_gs_ms_strides,
          {}, // acc0_biases_gs_ms_ns_lengths
          {}, // acc0_biases_gs_ms_ns_strides
          {}, // acc1_biases_gs_ms_os_lengths
          {}  // acc1_biases_gs_ms_os_strides
      });
    }

    argument_ptr = device_op_ptr->MakeArgumentPointer(
        params.q_ptrs,
        params.k_ptrs,
        params.z_ptrs,
        params.v_ptrs,
        params.bwd_out_ptrs,
        params.bwd_softmax_lse_ptrs,
        params.d_ptrs,
        params.dout_ptrs, 
        params.dq_ptrs,
        params.dk_ptrs,
        params.dv_ptrs,
        {},
        {},
        problem_descs,
        device_gemm_trait::AElementOp{},
        device_gemm_trait::B0ElementOp{},
        device_gemm_trait::Acc0ElementOp{params.softmax_scale},
        device_gemm_trait::B1ElementOp{},
        device_gemm_trait::CElementOp{},
        params.p_dropout, 
        params.seeds);

    // specify workspace for problem_desc
    SimpleDeviceMem problem_desc_workspace{ device_op_ptr->GetWorkSpaceSize(argument_ptr.get()) };

    device_op_ptr->SetWorkSpacePointer(argument_ptr.get(),
                            problem_desc_workspace.GetDeviceBuffer());
  }

  float Run() {
    if (!device_op_ptr->IsSupportedArgument(argument_ptr.get())) {
      std::cout << device_op_ptr->GetTypeString() 
                << " does not support this problem"
                << std::endl;
      return 0;
    }

    return invoker_ptr->Run(argument_ptr.get(), StreamConfig{stream, time_kernel});

    // if (time_kernel) {
    //   std::cout << "time elpase is " << avg_time << " ms" << std::endl;
    // }
  }

 private:
  std::unique_ptr<DeviceOp> device_op_ptr;
  std::unique_ptr<DeviceOp::Invoker> invoker_ptr;
  std::unique_ptr<DeviceOp::Argument> argument_ptr;

  // static const bool time_kernel = false;
};
} // namespace bwd_device_gemm