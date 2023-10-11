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

#include "bwd_device_gemm_launcher.h"

#include "static_switch.h"

namespace bwd_device_gemm {
class FlashBwdRunner {
 public:
  template <bool kIsGrouped, int kHeadDim, typename T, bool kIsPadding, bool kIsCausal>
  void Run(FlashBwdParams &params, hipStream_t &stream);

 private:
  template <template <typename> typename DeviceGemmTemplate,
            typename T,
            device_gemm_trait::GemmSpec kGemmSpec,
            device_gemm_trait::MaskingSpec kMaskingSpec, 
            bool kIsDeterministic>
  void run_(FlashBwdParams &params, hipStream_t &stream) {
    if (!params.kIsUnitTestMode) {
      // benchmark mode
      // input, output, gemm, dropout, cshuffle, masking specialization, deterministic
      using DeviceGemmTraits = device_gemm_trait::Backward<T, 
                                                           T, 
                                                           device_gemm_trait::BFloat16, 
                                                           device_gemm_trait::Int16, 
                                                           8, 
                                                           kGemmSpec,
                                                           kMaskingSpec, 
                                                           kIsDeterministic>;
      using DeviceGemmInstance = DeviceGemmInstanceLauncher<DeviceGemmTemplate, DeviceGemmTraits>;
      auto device_gemm_instance_ptr = std::make_unique<DeviceGemmInstance>();
      device_gemm_instance_ptr->Launch(params, stream);
    } else { 
      // unit test mode
      // input, output, gemm, dropout, cshuffle, masking specialization, deterministic
      using DeviceGemmTraits = device_gemm_trait::Backward<T, 
                                                           device_gemm_trait::Float32, 
                                                           T,
                                                           std::conditional_t<std::is_same_v<T, device_gemm_trait::Float16>, device_gemm_trait::Int16, device_gemm_trait::Int32>,
                                                           4, 
                                                           kGemmSpec,
                                                           kMaskingSpec, 
                                                           kIsDeterministic>;
      using DeviceGemmInstance = DeviceGemmInstanceLauncher<DeviceGemmTemplate, DeviceGemmTraits>;
      auto device_gemm_instance_ptr = std::make_unique<DeviceGemmInstance>();
      device_gemm_instance_ptr->Launch(params, stream);
    }
  }
}; // class FlashBwdRunner
} // namespace bwd_device_gemm