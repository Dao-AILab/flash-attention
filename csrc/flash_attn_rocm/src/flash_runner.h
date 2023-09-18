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

#include <memory>
#include <cstdlib>

namespace fwd_device_gemm { class FlashFwdRunner; }
namespace bwd_device_gemm { class FlashBwdRunner; }
class FlashFwdParams;
class FlashBwdParams; 

class FlashRunner {
 public:
  // constructor
  explicit FlashRunner::FlashRunner()
    : pimpl_fwd_runner_(std::make_unique<fwd_device_gemm::FlashFwdRunner>(get_env_("FLASH_ATTENTION_INTERNAL_UNIT_TEST_MODE"), 
                                                                          get_env_("FLASH_ATTENTION_INTERNAL_DETERMINISTIC"))),
      pimpl_bwd_runner_(std::make_unique<bwd_device_gemm::FlashBwdRunner>(get_env_("FLASH_ATTENTION_INTERNAL_UNIT_TEST_MODE"), 
                                                                          get_env_("FLASH_ATTENTION_INTERNAL_DETERMINISTIC"))) {}

  void RunFwd(fwd_device_gemm::FlashFwdParams &params, hipStream_t &stream);
  void RunBwd(bwd_device_gemm::FlashBwdParams &params, hipStream_t &stream);

 private:
  // get environment variables for internal usage
  int get_env_(std::string env_var) {
    char* res = std::getenv(env_var);
    if (res == '0' || res == NULL) { return false; }
    else { return true; }
  }
   
  std::unique_ptr<fwd_device_gemm::FlashFwdRunner> pimpl_fwd_runner_;
  std::unique_ptr<bwd_device_gemm::FlashBwdRunner> pimpl_bwd_runner_;
}; // class FlashRunner