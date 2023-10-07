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

#include "fwd_device_gemm_launcher.h"

#include "static_switch.h"

namespace fwd_device_gemm {
class FlashFwdRunner {
 public:
  // constructor
  explicit FlashFwdRunner(bool is_unit_test_mode, bool is_deterministic)
    : is_unit_test_mode_(is_unit_test_mode),
      is_deterministic_(is_deterministic) {}
 
  template <bool kIsGrouped, int kHeadDim, typename T,  bool kIsPadding, bool kIsCausal>
  void Run(FlashFwdParams &params, hipStream_t &stream);
 
 private:
  template <template <typename> typename DeviceGemmTemplate,
            typename T, 
            device_gemm_trait::GemmSpec kGemmSpec,
            device_gemm_trait::MaskingSpec kMaskingSpec, 
            bool kIsDeterministic>
  void run_(FlashFwdParams &params, hipStream_t &stream) {
    // input, output, gemm, dropout, cshuffle, masking specialization, deterministic
    using DeviceGemmTraits = device_gemm_trait::Forward<T,
                                                        kGemmSpec,
                                                        kMaskingSpec, 
                                                        kIsDeterministic>;
    using DeviceGemmInstance = DeviceGemmInstanceLauncher<DeviceGemmTemplate, DeviceGemmTraits>;
    auto device_gemm_instance_ptr = std::make_unique<DeviceGemmInstance>();
    device_gemm_instance_ptr->Launch(params, stream);
  }

  const bool is_unit_test_mode_;
  const bool is_deterministic_;
}; // class FlashFwdRunner
} // namespace fwd_device_gemm